#!/usr/bin/env python3
"""CLI tool to battle a trained PyTorch TTT3DNet bot on an HTTPS Arena Server (e.g., https://3t3d.buresovi.net)."""

from __future__ import annotations

import argparse
from pathlib import Path

from ttt3d.arena_bot import ArenaBattleClient, NeuralArenaBot
from ttt3d.network import TTT3DNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Play 3T3D Neural Bot battles on an HTTPS Arena Server")
    parser.add_argument("--arena-url", default="https://3t3d.buresovi.net", help="Target HTTPS Arena Server URL")
    parser.add_argument("--checkpoint", default="checkpoints/ttt3d_latest.pt", help="Trained TTT3DNet checkpoint")
    parser.add_argument("--opponent", default="heuristic", choices=["random", "heuristic", "alphabeta", "mcts"])
    parser.add_argument("--games", type=int, default=4, help="Number of paired games to play")
    parser.add_argument("--prefix", default="pytorch-nn", help="Game ID prefix for 3D Web UI replay list")
    parser.add_argument("--insecure", action="store_true", help="Skip TLS certificate verification (for localhost)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        print(f"Loading trained TTT3DNet checkpoint from {ckpt}...")
        model = TTT3DNet.load_checkpoint(ckpt)
    else:
        print(f"Checkpoint {ckpt} not found; initializing fresh TTT3DNet(channels=32, num_blocks=2)...")
        model = TTT3DNet(channels=32, num_blocks=2)

    bot = NeuralArenaBot(model=model, temperature=0.0, use_tactical_lookahead=True)
    client = ArenaBattleClient(base_url=args.arena_url, verify_tls=not args.insecure)

    result = client.run_tournament(
        bot=bot,
        opponent_agent=args.opponent,
        num_games=args.games,
        mode="client",
        prefix=args.prefix,
    )
    print(
        f"Tournament complete on {args.arena_url} vs {args.opponent}: "
        f"W={result['wins']} D={result['draws']} L={result['losses']} "
        f"(Score: {result['win_rate']*100:.1f}%)"
    )
    print(f"Games available for 3D step-by-step replay at {args.arena_url}: {result['game_ids']}")


if __name__ == "__main__":
    main()
