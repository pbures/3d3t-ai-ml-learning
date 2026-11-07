"""End-to-end gRPC integration tests verifying all 3T3D game rules and PyTorch RL pipeline."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Generator

import grpc
import numpy as np
import pytest
import torch

from ttt3d.grpc_client import VectorizedEnvClient
from ttt3d.network import TTT3DNet
from ttt3d.symmetry import (
    NUM_SYMMETRIES,
    transform_action,
    transform_state,
    transform_valid_mask,
)
from ttt3d.trainer import (
    ReplayBuffer,
    evaluate_against_baseline,
    generate_adversarial_sparring,
    generate_self_play_episodes,
    generate_vectorized_rollouts,
    train_epoch,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def grpc_client() -> Generator[VectorizedEnvClient, None, None]:
    """Builds and starts the Go `cmd/grpc-env` server on a local Unix Domain Socket."""
    sock_path = f"/tmp/3t3d_py_{os.getpid()}.sock"
    bin_path = f"/tmp/3t3d_grpc_env_{os.getpid()}"
    if os.path.exists(sock_path):
        os.remove(sock_path)

    env = os.environ.copy()
    env["CGO_ENABLED"] = "0"
    subprocess.run(
        ["go", "build", "-o", bin_path, "./cmd/grpc-env"],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
    )
    proc = subprocess.Popen(
        [bin_path, "-endpoint", f"unix://{sock_path}"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        deadline = time.time() + 10.0
        while not os.path.exists(sock_path):
            if time.time() > deadline:
                raise RuntimeError("Timed out waiting for Go gRPC unix socket")
            time.sleep(0.05)

        with VectorizedEnvClient(f"unix://{sock_path}") as client:
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


def test_grpc_rules_grid_turns_gravity_and_full_column(
    grpc_client: VectorizedEnvClient,
) -> None:
    # Rule 2.1 (Grid bounds) & Rule 2.2 (Turns: White=1 starts)
    s0 = grpc_client.reset("rules_basic")
    assert s0["current_player"] == 1
    assert s0["valid_moves_mask"] == 0xFFFF
    assert s0["obs"].shape == (3, 4, 4, 4)
    # Channel 2 at start has 1.0 on z=0 layer and 0.0 on z=1,2,3
    assert np.all(s0["obs"][2, 0, :, :] == 1.0)
    assert np.all(s0["obs"][2, 1:, :, :] == 0.0)

    with pytest.raises(grpc.RpcError):
        grpc_client.step(action=16, game_id="rules_basic")

    # Rule 2.3a (Gravity): 4 balls dropped in column (2, 1) -> action = 4*1 + 2 = 6
    # Land at z = 0, 1, 2, 3 while alternating players 1, 2, 1, 2
    for expected_z in range(4):
        expected_player = 1 if (expected_z % 2 == 0) else 2
        res = grpc_client.step(action=6, game_id="rules_basic")
        assert res["last_move"]["x"] == 2
        assert res["last_move"]["y"] == 1
        assert res["last_move"]["z"] == expected_z
        assert res["last_move"]["player"] == expected_player

    # Rule 2.3b (Full Column): Column 6 now has z=3 occupied -> excluded from valid_moves_mask
    s_after = grpc_client.get_state("rules_basic")
    assert (s_after["valid_moves_mask"] & (1 << 6)) == 0
    with pytest.raises(grpc.RpcError):
        grpc_client.step(action=6, game_id="rules_basic")


def test_grpc_all_76_win_lines_and_categories(
    grpc_client: VectorizedEnvClient,
) -> None:
    # Rule 2.4a: Verify all 76 winning lines and all 7 geometric classes
    lines = grpc_client.get_win_lines()
    assert len(lines) == 76
    categories = {line["category"] for line in lines}
    assert categories == {
        "Orthogonal-X",
        "Orthogonal-Y",
        "Orthogonal-Z",
        "Diag-XY",
        "Diag-XZ",
        "Diag-YZ",
        "Diag-XYZ",
    }

    # Verify a 3D Space Diagonal win: (0,0,0), (1,1,1), (2,2,2), (3,3,3)
    # Actions along the main diagonal are a=0 (need z=0), a=5 (need z=1), a=10 (need z=2), a=15 (need z=3)
    gid = "win_3d_diag"
    grpc_client.reset(gid)
    # Build support columns under a=5 (1 support ball), a=10 (2 support balls), a=15 (3 support balls)
    # Play setup moves so White lands on (0,0,0), (1,1,1), (2,2,2), (3,3,3):
    setup_moves = [
        0,   # W -> (0,0,0) [target 1/4]
        5,   # B -> (1,1,0) [support for a=5]
        5,   # W -> (1,1,1) [target 2/4]
        10,  # B -> (2,2,0) [support 1 for a=10]
        15,  # W -> (3,3,0) [support 1 for a=15]
        10,  # B -> (2,2,1) [support 2 for a=10]
        10,  # W -> (2,2,2) [target 3/4]
        15,  # B -> (3,3,1) [support 2 for a=15]
        1,   # W -> filler at (1,0,0)
        15,  # B -> (3,3,2) [support 3 for a=15]
        15,  # W -> (3,3,3) [target 4/4 -> 3D Space Diagonal Win!]
    ]
    last_res = None
    for m in setup_moves:
        last_res = grpc_client.step(action=m, game_id=gid)
    assert last_res is not None
    assert last_res["done"] is True
    assert last_res["reward"] == pytest.approx(1.0)
    assert last_res["state"]["status"] == 1  # WHITE_WON
    assert last_res["state"]["winner"] == 1
    assert last_res["state"]["win_line_mask"] != 0

    # No further moves allowed after game is won
    with pytest.raises(grpc.RpcError):
        grpc_client.step(action=2, game_id=gid)


def test_grpc_64_move_draw_rule(grpc_client: VectorizedEnvClient) -> None:
    # Rule 2.4b (Draw): 64 balls placed without any 4-in-a-row terminates in DRAW (status=3)
    # Layer 0 = L0, Layer 1 = transpose(L0), Layer 2 = NOT(L0), Layer 3 = NOT(transpose(L0))
    gid = "draw_64"
    grpc_client.reset(gid)

    l0 = [
        [True, False, True, False],
        [True, False, True, False],
        [False, True, False, True],
        [False, True, False, True],
    ]

    draw_seq: list[int] = []
    for z in range(4):
        white_cols = []
        black_cols = []
        for y in range(4):
            for x in range(4):
                is_white = l0[y][x] if (z % 2 == 0) else l0[x][y]
                if z >= 2:
                    is_white = not is_white
                action = 4 * y + x
                if is_white:
                    white_cols.append(action)
                else:
                    black_cols.append(action)
        for w_act, b_act in zip(white_cols, black_cols):
            draw_seq.extend([w_act, b_act])

    assert len(draw_seq) == 64

    res = None
    for idx, act in enumerate(draw_seq):
        res = grpc_client.step(action=act, game_id=gid)
        if idx < 63:
            assert res["done"] is False

    assert res is not None
    assert res["done"] is True
    assert res["reward"] == 0.0
    assert res["state"]["status"] == 3  # DRAW
    assert res["state"]["winner"] == 0
    assert res["state"]["move_count"] == 64
    assert res["state"]["valid_moves_mask"] == 0


def test_symmetry_matches_go_bitboard_engine_exactly(
    grpc_client: VectorizedEnvClient,
) -> None:
    """Verifies that all 8 D4 symmetries in Python match playing the transformed moves in Go."""
    base_moves = [0, 3, 0, 7, 5, 0, 12, 0, 1]  # Fills column 0 (4 balls) + asymmetric layout
    grpc_client.reset("sym_base")
    for m in base_moves:
        grpc_client.step(action=m, game_id="sym_base")
    base_state = grpc_client.get_state("sym_base")

    for sym_idx in range(NUM_SYMMETRIES):
        gid = f"sym_check_{sym_idx}"
        grpc_client.reset(gid)
        for m in base_moves:
            m_sym = transform_action(m, sym_idx)
            grpc_client.step(action=m_sym, game_id=gid)
        go_sym_state = grpc_client.get_state(gid)

        py_sym_obs = transform_state(base_state["obs"], sym_idx)
        py_sym_mask = transform_valid_mask(base_state["valid_moves_mask"], sym_idx)

        assert np.array_equal(go_sym_state["obs"], py_sym_obs), f"Mismatch at sym_idx={sym_idx}"
        assert go_sym_state["valid_moves_mask"] == py_sym_mask, f"Mask mismatch at sym_idx={sym_idx}"


def test_end_to_end_self_play_vectorized_pool_and_training(
    grpc_client: VectorizedEnvClient,
) -> None:
    torch.manual_seed(2026)
    rng = np.random.default_rng(2026)
    net = TTT3DNet(channels=16, num_blocks=2)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=5000)

    # 1. Generate MCTS-guided episodes over gRPC
    stats_mcts = generate_self_play_episodes(
        client=grpc_client,
        model=net,
        buffer=buffer,
        num_episodes=2,
        temperature=0.5,
        mcts_simulations=24,
        use_go_mcts_teacher=True,
        augment=True,
        rng=rng,
    )
    assert stats_mcts["episodes"] == 2
    assert stats_mcts["samples_added"] > 0
    assert len(buffer) == stats_mcts["samples_added"]

    # 2. Generate pure neural self-play episode
    stats_nn = generate_self_play_episodes(
        client=grpc_client,
        model=net,
        buffer=buffer,
        num_episodes=1,
        temperature=1.0,
        use_go_mcts_teacher=False,
        augment=False,
        rng=rng,
    )
    assert stats_nn["episodes"] == 1
    assert stats_nn["samples_added"] > 0

    # 2b. Generate adversarial sparring games against heuristic and alphabeta
    spar_added = generate_adversarial_sparring(
        client=grpc_client,
        model=net,
        buffer=buffer,
        num_games_per_opponent=2,
        augment=True,
        rng=rng,
    )
    assert spar_added > 0

    # 3. Generate vectorized rollouts across parallel Go bitboard environments
    vec_stats = generate_vectorized_rollouts(
        client=grpc_client,
        model=net,
        buffer=buffer,
        num_envs=4,
        num_steps=45,
        temperature=1.0,
        augment=True,
        rng=rng,
    )
    assert vec_stats["episodes_completed"] >= 1
    assert vec_stats["samples_added"] > 0

    # Also test reset_pool on specific environment indices
    reset_res = grpc_client.reset_pool(env_ids=[0, 2])
    assert reset_res["num_envs"] == 4
    assert reset_res["obs"].shape == (4, 3, 4, 4, 4)

    # 4. Run gradient optimization steps and verify finite, decreasing loss on fixed batch sequence
    m1 = train_epoch(
        net, optimizer, buffer, batch_size=64, steps_per_epoch=4, rng=np.random.default_rng(77)
    )
    _ = train_epoch(net, optimizer, buffer, batch_size=64, steps_per_epoch=20, rng=rng)
    m2 = train_epoch(
        net, optimizer, buffer, batch_size=64, steps_per_epoch=4, rng=np.random.default_rng(77)
    )
    assert np.isfinite(m1["total_loss"]) and np.isfinite(m2["total_loss"])
    assert m2["total_loss"] < m1["total_loss"]

    # 5. Evaluate trained network against Go baseline agent
    with pytest.raises(ValueError):
        evaluate_against_baseline(grpc_client, net, opponent="random", num_games=0)

    eval_res = evaluate_against_baseline(
        grpc_client, net, opponent="random", num_games=4, simulations=16
    )
    assert eval_res["num_games"] == 4
    assert eval_res["wins"] + eval_res["draws"] + eval_res["losses"] == 4
    assert 0.0 <= eval_res["win_rate"] <= 1.0
