"""Unit tests for TTT3DNet, 8-fold D4 symmetry, ReplayBuffer, and Policy-Value Loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ttt3d.grpc_client import unpack_batch_observations, unpack_observation
from ttt3d.network import (
    ILLEGAL_LOGIT_VALUE,
    TTT3DNet,
    mask_illegal_logits,
    valid_mask_to_tensor,
)
from ttt3d.symmetry import (
    NUM_SYMMETRIES,
    augment_sample,
    transform_action,
    transform_policy,
    transform_state,
    transform_valid_mask,
)
from ttt3d.trainer import (
    ReplayBuffer,
    compute_policy_value_loss,
    compute_tactical_action_masks,
    generate_tactical_threat_curriculum,
    rectify_tactical_sample,
    train_epoch,
)


def test_symmetry_group_properties() -> None:
    rng = np.random.default_rng(42)
    state = rng.random((3, 4, 4, 4), dtype=np.float32)
    policy = rng.random((16,), dtype=np.float32)
    policy /= policy.sum()

    # Identity (sym_idx = 0) leaves state, policy, action, and mask unchanged
    assert np.allclose(transform_state(state, 0), state)
    assert np.allclose(transform_policy(policy, 0), policy)
    for a in range(16):
        assert transform_action(a, 0) == a
    assert transform_valid_mask(0b1010_0101_1100_0011, 0) == 0b1010_0101_1100_0011

    # 4 rotations of 90 degrees return to identity
    s_rot = state.copy()
    p_rot = policy.copy()
    for _ in range(4):
        s_rot = transform_state(s_rot, 1)
        p_rot = transform_policy(p_rot, 1)
    assert np.allclose(s_rot, state)
    assert np.allclose(p_rot, policy)

    # 2 reflections (sym_idx = 4) return to identity
    s_ref = transform_state(transform_state(state, 4), 4)
    p_ref = transform_policy(transform_policy(policy, 4), 4)
    assert np.allclose(s_ref, state)
    assert np.allclose(p_ref, policy)

    # All 8 symmetries produce distinct permutations of an asymmetric state/policy
    augmented = augment_sample(state, policy, value=0.75)
    assert len(augmented) == NUM_SYMMETRIES
    unique_policies = {tuple(np.round(p, 5)) for _, p, _ in augmented}
    assert len(unique_policies) == NUM_SYMMETRIES
    for s_sym, p_sym, val in augmented:
        assert s_sym.shape == (3, 4, 4, 4)
        assert p_sym.shape == (16,)
        assert np.isclose(p_sym.sum(), 1.0)
        assert val == 0.75

    # Batched state symmetry (B, 3, 4, 4, 4)
    batch_state = np.stack([state, state], axis=0)
    batch_sym = transform_state(batch_state, 3)
    assert batch_sym.shape == (2, 3, 4, 4, 4)
    assert np.allclose(batch_sym[0], transform_state(state, 3))


def test_symmetry_validation_errors() -> None:
    with pytest.raises(ValueError, match="sym_idx"):
        transform_state(np.zeros((3, 4, 4, 4), dtype=np.float32), -1)
    with pytest.raises(ValueError, match="sym_idx"):
        transform_state(np.zeros((3, 4, 4, 4), dtype=np.float32), 8)
    with pytest.raises(ValueError, match="Expected state shape"):
        transform_state(np.zeros((4, 4, 4), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="Expected policy shape"):
        transform_policy(np.zeros((4, 4), dtype=np.float32), 1)
    with pytest.raises(ValueError, match="action must be in"):
        transform_action(16, 1)


def test_valid_mask_and_logit_masking() -> None:
    # Single int mask: only actions 0 and 15 legal
    mask_int = (1 << 0) | (1 << 15)
    t_mask = valid_mask_to_tensor(mask_int)
    assert t_mask.shape == (1, 16)
    assert bool(t_mask[0, 0]) is True
    assert bool(t_mask[0, 15]) is True
    assert bool(t_mask[0, 5]) is False

    # List, numpy bool, torch bool, torch int inputs
    assert valid_mask_to_tensor([0xFFFF, 0x0001]).shape == (2, 16)
    bool_np = np.ones((16,), dtype=bool)
    assert valid_mask_to_tensor(bool_np).shape == (1, 16)
    assert valid_mask_to_tensor(torch.ones((2, 16), dtype=torch.bool)).shape == (2, 16)
    assert valid_mask_to_tensor(torch.tensor(0xFFFF)).shape == (1, 16)

    with pytest.raises(TypeError):
        valid_mask_to_tensor("invalid")  # type: ignore[arg-type]

    logits = torch.zeros((1, 16), dtype=torch.float32)
    assert torch.equal(mask_illegal_logits(logits, None), logits)

    masked = mask_illegal_logits(logits, mask_int)
    assert float(masked[0, 0]) == 0.0
    assert float(masked[0, 15]) == 0.0
    assert float(masked[0, 3]) == pytest.approx(ILLEGAL_LOGIT_VALUE)

    # All-zero mask (terminal state) does not produce NaN after softmax
    zero_masked = mask_illegal_logits(logits, 0)
    probs = torch.softmax(zero_masked, dim=-1)
    assert not torch.isnan(probs).any()

    with pytest.raises(ValueError, match="Mask shape"):
        mask_illegal_logits(logits, [0xFFFF, 0xFFFF])


def test_network_forward_predict_and_select_action(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        TTT3DNet(channels=0, num_blocks=2)

    net = TTT3DNet(channels=32, num_blocks=2)
    x = torch.randn((4, 3, 4, 4, 4), dtype=torch.float32)
    masks = [0xFFFF, 0x000F, 0x8000, 0x0001]
    logits, values = net(x, valid_masks=masks)

    assert logits.shape == (4, 16)
    assert values.shape == (4, 1)
    assert torch.all(values >= -1.0) and torch.all(values <= 1.0)

    with pytest.raises(ValueError, match="Expected input tensor shape"):
        net(torch.randn((4, 4, 4, 4)))

    # Predict & select_action with restricted legal mask (only column 7 allowed)
    single_state = np.zeros((3, 4, 4, 4), dtype=np.float32)
    only_col_7 = 1 << 7
    probs, val = net.predict(single_state, valid_mask=only_col_7)
    assert probs.shape == (16,)
    assert probs[7] == pytest.approx(1.0)
    assert -1.0 <= val <= 1.0

    # Greedy selection (temp=0) and stochastic selection (temp=1.0)
    act_greedy, _, _ = net.select_action(single_state, valid_mask=only_col_7, temperature=0.0)
    assert act_greedy == 7

    act_stoch, _, _ = net.select_action(
        torch.from_numpy(single_state),
        valid_mask=(1 << 2) | (1 << 9),
        temperature=1.2,
        rng=np.random.default_rng(7),
    )
    assert act_stoch in (2, 9)

    with pytest.raises(ValueError, match="valid_mask is 0"):
        net.select_action(single_state, valid_mask=0)

    # Save & load checkpoint roundtrip
    ckpt_file = tmp_path / "models" / "ttt3d_test.pt"
    net.save_checkpoint(ckpt_file)
    loaded = TTT3DNet.load_checkpoint(ckpt_file)
    assert loaded.channels == 32
    assert loaded.num_blocks == 2
    probs_loaded, val_loaded = loaded.predict(single_state, valid_mask=0xFFFF)
    probs_orig, val_orig = net.predict(single_state, valid_mask=0xFFFF)
    assert np.allclose(probs_loaded, probs_orig)
    assert val_loaded == pytest.approx(val_orig)

    # Export fused binary weights for Go inference
    bin_file = tmp_path / "models" / "ttt3d_fused.bin"
    net.export_fused_binary(bin_file)
    assert bin_file.exists()
    raw_bin = bin_file.read_bytes()
    assert raw_bin[:4] == b"3T3D"
    assert len(raw_bin) > 600_000


def test_replay_buffer_and_training_convergence() -> None:
    torch.manual_seed(123)
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)

    buf = ReplayBuffer(capacity=100)
    with pytest.raises(ValueError, match="empty"):
        buf.sample_batch(4)

    net = TTT3DNet(channels=16, num_blocks=2)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4)

    with pytest.raises(ValueError, match="must not be empty"):
        train_epoch(net, opt, buf)

    # Create a synthetic tactical state where action 5 is the winning move (+1.0 value)
    rng = np.random.default_rng(123)
    state = np.zeros((3, 4, 4, 4), dtype=np.float32)
    state[0, 0, 1, 1] = 1.0
    state[2, :, :, :] = 0.0
    state[2, 0, :, :] = 1.0  # gravity surface on z=0
    target_pi = np.zeros((16,), dtype=np.float32)
    target_pi[5] = 1.0

    # Test both non-augmented and 8x augmented insertion
    added_single = buf.add_sample(state, target_pi, value=1.0, valid_mask=0xFFFF, augment=False)
    assert added_single == 1
    added_aug = buf.add_sample(state, target_pi, value=1.0, valid_mask=0xFFFF, augment=True)
    assert added_aug == 8
    assert len(buf) == 9

    initial_metrics = train_epoch(
        net, opt, buf, batch_size=8, steps_per_epoch=1, rng=rng
    )
    final_metrics = train_epoch(
        net, opt, buf, batch_size=8, steps_per_epoch=25, rng=rng
    )
    assert final_metrics["total_loss"] < initial_metrics["total_loss"]
    assert final_metrics["policy_loss"] < initial_metrics["policy_loss"]


def test_observation_unpack_helpers() -> None:
    raw = bytes([1] * 64 + [0] * 128)
    obs = unpack_observation(raw)
    assert obs.shape == (3, 4, 4, 4)
    assert np.all(obs[0] == 1.0)
    assert np.all(obs[1] == 0.0)

    with pytest.raises(ValueError, match="Expected 192 bytes"):
        unpack_observation(b"short")

    batch_obs = unpack_batch_observations(raw * 3, num_envs=3)
    assert batch_obs.shape == (3, 3, 4, 4, 4)
    assert unpack_batch_observations(b"", num_envs=0).shape == (0, 3, 4, 4, 4)

    with pytest.raises(ValueError, match="Expected 384 bytes"):
        unpack_batch_observations(raw, num_envs=2)


def test_tactical_action_masks_and_rectification() -> None:
    # 1. Empty mask returns (0, 0, False)
    empty_state = np.zeros((3, 4, 4, 4), dtype=np.float32)
    assert compute_tactical_action_masks(empty_state, 0) == (0, 0, False)

    # 2. Immediate 1-ply win on action 3 (my stones at 0, 1, 2 on z=0)
    win_state = np.zeros((3, 4, 4, 4), dtype=np.float32)
    win_state[0, 0, 0, 0] = 1.0
    win_state[0, 0, 0, 1] = 1.0
    win_state[0, 0, 0, 2] = 1.0
    win_state[2, 0, :, :] = 1.0
    wm, bm, fl = compute_tactical_action_masks(win_state, 0xFFFF)
    assert wm == (1 << 3)
    assert bm == (0xFFFF & ~(1 << 3))
    assert fl is False

    zero_pi = np.zeros(16, dtype=np.float32)
    rect_pi, rect_val, rect_bm = rectify_tactical_sample(win_state, zero_pi, -0.5, 0xFFFF)
    assert rect_pi[3] == pytest.approx(1.0)
    assert rect_val == 1.0
    assert rect_bm == bm

    # 3. Immediate 1-ply forced block on action 3 (opp stones at 0, 1, 2 on z=0)
    block_state = np.zeros((3, 4, 4, 4), dtype=np.float32)
    block_state[1, 0, 0, 0] = 1.0
    block_state[1, 0, 0, 1] = 1.0
    block_state[1, 0, 0, 2] = 1.0
    block_state[2, 0, :, :] = 1.0
    wm2, bm2, fl2 = compute_tactical_action_masks(block_state, 0xFFFF)
    assert wm2 == 0
    assert bm2 == (0xFFFF & ~(1 << 3))
    assert fl2 is False

    rect_pi2, rect_val2, _ = rectify_tactical_sample(block_state, zero_pi, 0.2, 0xFFFF)
    assert rect_pi2[3] == pytest.approx(1.0)
    assert rect_val2 == pytest.approx(0.2)

    # 4. Unstoppable opponent fork (opp has two 3-in-a-row threats at 3 and 7)
    fork_state = block_state.copy()
    fork_state[1, 0, 1, 0] = 1.0
    fork_state[1, 0, 1, 1] = 1.0
    fork_state[1, 0, 1, 2] = 1.0
    wm3, bm3, fl3 = compute_tactical_action_masks(fork_state, 0xFFFF)
    assert wm3 == 0 and bm3 == 0 and fl3 is True
    _, rect_val3, _ = rectify_tactical_sample(fork_state, zero_pi, 0.5, 0xFFFF)
    assert rect_val3 == -1.0

    # 5. Tactical threat curriculum generator populates ReplayBuffer
    buf = ReplayBuffer(capacity=1000)
    added = generate_tactical_threat_curriculum(buf, num_games=6, augment=True, rng=np.random.default_rng(7))
    assert added > 0 and len(buf) > 0
