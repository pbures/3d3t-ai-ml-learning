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
    parser.add_argument(
        "--opponent",
        default="heuristic",
        choices=["random", "heuristic", "alphabeta", "deep", "mcts", "ttt3d-deep-gray1", "ttt3d-deep-gray1-pure", "ttt3dnet", "ttt3dnet-pure"],
    )
    parser.add_argument("--games", type=int, default=4, help="Number of paired games to play")
    parser.add_argument("--prefix", default="", help="Optional game ID prefix for 3D Web UI replay list")
    parser.add_argument("--name", default="", help="Optional human-readable match name for the tournament")
    parser.add_argument("--battle-id", default="", help="Optional existing or custom battle ID to group games under")
    parser.add_argument("--battle-name", default="", help="Optional human-readable battle name")
    parser.add_argument("--participant", default="TTT3DNet-Client", help="Client player / participant name in the battle")
    parser.add_argument("--date", default="", help="Optional battle date (ISO-8601 or YYYY-MM-DD)")
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
        match_name=args.name,
        battle_id=args.battle_id,
        battle_name=args.battle_name,
        participant=args.participant,
        battle_date=args.date,
    )
    print(
        f"Battle '{result.get('battle_name')}' (id={result.get('battle_id')}, participant={result.get('participant')}) "
        f"complete on {args.arena_url} vs {args.opponent}:\n"
        f"  Client Wins={result['wins']} | Draws={result['draws']} | Server Wins={result['losses']} "
        f"(Client Score: {result['win_rate']*100:.1f}%) -> Battle Winner: {result.get('winner_label')}"
    )
    print(f"Games available for 3D step-by-step replay at {args.arena_url}: {result['game_ids']}")


if __name__ == "__main__":
    main()
