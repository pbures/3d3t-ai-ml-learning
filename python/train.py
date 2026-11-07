#!/usr/bin/env python3
"""CLI entrypoint for training TTT3DNet via Go gRPC Self-Play & MCTS Policy-Value Iteration."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ttt3d.grpc_client import VectorizedEnvClient
from ttt3d.network import TTT3DNet
from ttt3d.trainer import (
    ReplayBuffer,
    evaluate_against_baseline,
    generate_self_play_episodes,
    generate_tactical_threat_curriculum,
    generate_vectorized_rollouts,
    train_epoch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3T3D 3D ResNet via Go gRPC")
    parser.add_argument("--grpc-target", default="localhost:50051", help="Go gRPC server target")
    parser.add_argument("--iterations", type=int, default=10, help="Number of RL iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=10, help="Self-play games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=64, help="Go MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--steps-per-iter", type=int, default=100, help="Gradient steps per iteration")
    parser.add_argument("--channels", type=int, default=32, help="3D ResNet trunk channels")
    parser.add_argument("--blocks", type=int, default=2, help="Number of 3D ResNet blocks")
    parser.add_argument("--lr", type=float, default=1.5e-3, help="Learning rate")
    parser.add_argument("--blunder-weight", type=float, default=4.0, help="1-ply threat blunder suppression weight")
    parser.add_argument("--checkpoint", default="checkpoints/ttt3d_latest.pt", help="Output checkpoint path")
    parser.add_argument(
        "--export-go-bin",
        default="internal/agents/weights/ttt3d_latest.bin",
        help="Output fused binary weights path for Go embedding",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        model = TTT3DNet.load_checkpoint(ckpt_path).to(device)
    else:
        model = TTT3DNet(channels=args.channels, num_blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=80000)

    with VectorizedEnvClient(args.grpc_target) as client:
        for it in range(1, args.iterations + 1):
            generate_tactical_threat_curriculum(buffer=buffer, num_games=60, augment=True)
            generate_self_play_episodes(
                client=client,
                model=model,
                buffer=buffer,
                num_episodes=args.episodes_per_iter,
                temperature=0.8 if it <= args.iterations // 2 else 0.5,
                mcts_simulations=args.mcts_sims,
                use_go_mcts_teacher=True,
                augment=True,
            )
            generate_vectorized_rollouts(
                client=client,
                model=model,
                buffer=buffer,
                num_envs=8,
                num_steps=20,
                temperature=0.6,
                augment=True,
            )
            train_stats = train_epoch(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                batch_size=args.batch_size,
                steps_per_epoch=args.steps_per_iter,
                blunder_weight=args.blunder_weight,
                device=device,
            )
            eval_pure_rnd = evaluate_against_baseline(
                client, model, opponent="random", num_games=20, use_tactical_lookahead=False
            )
            eval_hybrid_heur = evaluate_against_baseline(
                client, model, opponent="heuristic", num_games=10, use_tactical_lookahead=True
            )
            print(
                f"[Iter {it:02d}/{args.iterations:02d}] "
                f"buffer={len(buffer)} "
                f"loss={train_stats['total_loss']:.4f} "
                f"(pi={train_stats['policy_loss']:.4f}, v={train_stats['value_loss']:.4f}) "
                f"pure_vs_random={eval_pure_rnd['win_rate']*100:.1f}% "
                f"hybrid_vs_heuristic={eval_hybrid_heur['win_rate']*100:.1f}%"
            )
            model.save_checkpoint(ckpt_path)
            if args.export_go_bin:
                model.export_fused_binary(Path(args.export_go_bin))


if __name__ == "__main__":
    main()
