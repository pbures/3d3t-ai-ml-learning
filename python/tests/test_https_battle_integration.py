"""End-to-end HTTPS integration tests for NeuralArenaBot, BotWebhookServer, and Go Arena Server."""

from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Generator

import pytest

from ttt3d.arena_bot import (
    WIN_LINE_MASKS,
    ArenaBattleClient,
    ArenaHTTPError,
    BotWebhookServer,
    NeuralArenaBot,
    check_bitboard_win,
    state_dto_to_canonical_tensor,
)
from ttt3d.network import TTT3DNet

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def https_arena_client() -> Generator[ArenaBattleClient, None, None]:
    """Builds and starts the Go `cmd/arena-server` over HTTPS (self-signed TLS) on a free port."""
    port = _find_free_port()
    sock_path = f"/tmp/3t3d_arena_py_{os.getpid()}.sock"
    bin_path = f"/tmp/3t3d_arena_srv_{os.getpid()}"
    if os.path.exists(sock_path):
        os.remove(sock_path)

    env = os.environ.copy()
    env["CGO_ENABLED"] = "0"
    subprocess.run(
        ["go", "build", "-o", bin_path, "./cmd/arena-server"],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    proc = subprocess.Popen(
        [
            bin_path,
            "-https-addr",
            f"127.0.0.1:{port}",
            "-grpc-endpoint",
            f"unix://{sock_path}",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    client = ArenaBattleClient(f"https://127.0.0.1:{port}", verify_tls=False)
    try:
        deadline = time.time() + 10.0
        while True:
            try:
                h = client.health()
                if h.get("status") == "ok":
                    break
            except Exception:  # noqa: BLE001
                if time.time() > deadline:
                    raise RuntimeError("Timed out waiting for Go HTTPS Arena Server")
                time.sleep(0.05)

        yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        for p in (sock_path, bin_path):
            if os.path.exists(p):
                os.remove(p)


def test_neural_arena_bot_tactical_shield_and_fallback_tensor() -> None:
    assert len(WIN_LINE_MASKS) == 76
    assert check_bitboard_win(WIN_LINE_MASKS[0]) is True
    assert check_bitboard_win(0) is False

    net = TTT3DNet(channels=16, num_blocks=2)
    bot_tactical = NeuralArenaBot(net, temperature=0.0, use_tactical_lookahead=True)
    bot_pure = NeuralArenaBot(net, temperature=0.0, use_tactical_lookahead=False)

    # 1. Test fallback canonical_tensor reconstruction from bitboards when canonical_tensor is omitted
    # White has 3 in a row at (0,0,0), (1,0,0), (2,0,0) -> bits 0, 1, 2. Action 3 wins immediately!
    white_bb = (1 << 0) | (1 << 1) | (1 << 2)
    black_bb = (1 << 4) | (1 << 5)
    dto_win = {
        "white_bitboard": white_bb,
        "black_bitboard": black_bb,
        "current_player": 1,
    }
    obs, mask = state_dto_to_canonical_tensor(dto_win)
    assert obs.shape == (3, 4, 4, 4)
    assert mask == 0xFFFF

    win_decision = bot_tactical.choose_move(dto_win)
    assert win_decision["action"] == 3
    assert win_decision["reason"] == "immediate_win"
    assert win_decision["value"] == 1.0

    pure_decision = bot_pure.choose_move(dto_win)
    assert pure_decision["reason"] == "neural_policy"
    assert 0 <= pure_decision["action"] < 16

    # 2. Test forced 1-ply block when opponent (White) has 3-in-a-row at bits 0, 1, 2 and it is Black's turn (player 2)
    dto_block = {
        "white_bitboard": white_bb,
        "black_bitboard": black_bb,
        "current_player": 2,
    }
    block_decision = bot_tactical.choose_move(dto_block)
    assert block_decision["action"] == 3
    assert block_decision["reason"] == "forced_block"

    # 3. Error when no legal moves remain
    with pytest.raises(ValueError, match="No legal moves"):
        bot_tactical.choose_move({"valid_moves_mask": 0})


def test_webhook_server_routes_and_error_handling() -> None:
    net = TTT3DNet(channels=16, num_blocks=2)
    bot = NeuralArenaBot(net)

    with BotWebhookServer(bot) as srv:
        # Calling start() again while running is a no-op
        srv.start()

        # GET /health -> 200
        with urllib.request.urlopen(f"{srv.base_url}/health") as resp:
            assert resp.status == 200

        # GET /unknown -> 404
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{srv.base_url}/unknown")
        assert exc_info.value.code == 404

        # POST /unknown -> 404
        req_404 = urllib.request.Request(f"{srv.base_url}/unknown", data=b"{}", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_404)
        assert exc_info.value.code == 404

        # POST /move with invalid JSON -> 400
        req_bad = urllib.request.Request(f"{srv.move_url}", data=b"{not_json", method="POST")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req_bad)
        assert exc_info.value.code == 400


def test_https_arena_all_game_rules_verification(
    https_arena_client: ArenaBattleClient,
) -> None:
    """Verifies all rules of 3T3D (Section 4.3 of SPECIFICATION.md) over HTTPS from Python."""
    # Rule 2.1 (Grid) & Rule 2.2 (Turns)
    created = https_arena_client.create_game(
        game_id="https-rules-test", white_name="Alice", black_name="Bob"
    )
    assert created["game_id"] == "https-rules-test"
    assert created["current_player"] == 1
    assert created["valid_moves_mask"] == 0xFFFF

    # Out-of-bounds action & out-of-bounds (x, y) rejected with HTTP 400
    with pytest.raises(ArenaHTTPError) as exc_info:
        https_arena_client.submit_move("https-rules-test", action=16)
    assert exc_info.value.status_code == 400

    with pytest.raises(ArenaHTTPError) as exc_info:
        https_arena_client.submit_move("https-rules-test", use_coords=True, x=4, y=0)
    assert exc_info.value.status_code == 400

    # Out-of-turn move (Black=2 trying to move on White=1's turn) rejected with HTTP 409
    with pytest.raises(ArenaHTTPError) as exc_info:
        https_arena_client.submit_move("https-rules-test", action=0, player=2)
    assert exc_info.value.status_code == 409

    # Rule 2.3a (Gravity): 4 balls in column (x=1, y=2 -> action=9) land at z=0, 1, 2, 3
    for expected_z in range(4):
        expected_player = 1 if (expected_z % 2 == 0) else 2
        res = https_arena_client.submit_move(
            "https-rules-test", use_coords=True, x=1, y=2, player=expected_player
        )
        assert res["last_move"]["action"] == 9
        assert res["last_move"]["z"] == expected_z
        assert res["last_move"]["player"] == expected_player

    # Rule 2.3b (Full Column): 5th move in column (1, 2) rejected and bit 9 is 0 in valid_moves_mask
    state_full = https_arena_client.get_game("https-rules-test")
    assert (state_full["valid_moves_mask"] & (1 << 9)) == 0
    with pytest.raises(ArenaHTTPError) as exc_info:
        https_arena_client.submit_move("https-rules-test", action=9)
    assert exc_info.value.status_code == 400

    # Rule 2.4a (Wins across all 76 winning lines):
    lines = https_arena_client.get_win_lines()
    assert len(lines) == 76
    # Verify representative winning lines across Orthogonal, 2D Planar Diagonal, and 3D Space Diagonal
    for line_idx in (0, 20, 40, 50, 62, 70, 75):
        win_state = https_arena_client.apply_preset(
            "https-rules-test", preset_type="win", line_index=line_idx, winner=1
        )
        assert win_state["status"] == "WHITE_WON"
        assert win_state["winner"] == 1
        assert win_state["win_line_index"] == line_idx
        assert len(win_state["win_line_coords"]) == 4

    # Post-win move rejected with HTTP 409
    with pytest.raises(ArenaHTTPError) as exc_info:
        https_arena_client.submit_move("https-rules-test", action=0)
    assert exc_info.value.status_code == 409

    # Rule 2.4b (64-move Draw):
    draw_state = https_arena_client.apply_preset("https-rules-test", preset_type="draw")
    assert draw_state["status"] == "DRAW"
    assert draw_state["winner"] == 0
    assert draw_state["move_count"] == 64
    assert draw_state["valid_moves_mask"] == 0

    # Reset restores initial state
    reset_state = https_arena_client.reset_game("https-rules-test")
    assert reset_state["status"] == "ONGOING"
    assert reset_state["move_count"] == 0


def test_https_webhook_and_client_neural_tournaments(
    https_arena_client: ArenaBattleClient,
) -> None:
    """Verifies end-to-end NeuralArenaBot battles in both Webhook and Client-Driven modes."""
    net = TTT3DNet(channels=16, num_blocks=2)
    bot = NeuralArenaBot(net, temperature=0.0, use_tactical_lookahead=True)

    with pytest.raises(ValueError, match="num_games must be positive"):
        https_arena_client.run_tournament(bot, num_games=0)
    with pytest.raises(ValueError, match="webhook_move_url is required"):
        https_arena_client.run_tournament(bot, num_games=2, mode="webhook")

    # Mode A: Webhook-orchestrated tournament (`POST /api/v1/matches/run`)
    with BotWebhookServer(bot) as webhook_srv:
        webhook_res = https_arena_client.run_tournament(
            bot=bot,
            opponent_agent="random",
            num_games=2,
            mode="webhook",
            webhook_move_url=webhook_srv.move_url,
            prefix="webhook-nn",
        )
        assert webhook_res["num_games"] == 2
        assert webhook_res["wins"] + webhook_res["draws"] + webhook_res["losses"] == 2
        assert len(webhook_res["game_ids"]) == 2

    # Mode B: Client-driven tournament (`play_client_match`) against HeuristicAgent
    client_res = https_arena_client.run_tournament(
        bot=bot,
        opponent_agent="heuristic",
        num_games=2,
        mode="client",
        prefix="client-nn",
    )
    assert client_res["num_games"] == 2
    assert client_res["wins"] + client_res["draws"] + client_res["losses"] == 2

    # Verify all played games appear in `GET /api/v1/games` and have full move histories for 3D replay
    games_list = https_arena_client.list_games()
    recorded_ids = {g["game_id"] for g in games_list}
    for gid in webhook_res["game_ids"] + client_res["game_ids"]:
        assert gid in recorded_ids
        replay = https_arena_client.get_game(gid)
        assert replay["status"] in ("WHITE_WON", "BLACK_WON", "DRAW")
        assert len(replay["history"]) == replay["move_count"]
        assert replay["move_count"] >= 7
