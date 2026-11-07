"""8-fold 3D Dihedral Symmetry (D4) transformations for 4x4x4 Gravity 3T3D.

Because gravity pulls balls along the vertical Z-axis toward z=0, vertical flips
or rotations mixing Z with X/Y would violate gravity physics. However, the 4x4
horizontal (X, Y) plane is invariant under the 8 symmetries of the square (D4):
  - sym_idx in 0..3: k = sym_idx rotations by 90 degrees in the (Y, X) plane
  - sym_idx in 4..7: horizontal reflection across X (x -> 3 - x) followed by
                     k = (sym_idx - 4) rotations by 90 degrees in the (Y, X) plane.

Coordinate convention matching Go `internal/engine/board.go`:
  - State tensor has shape (3, 4, 4, 4) indexed as [channel, z, y, x].
  - Policy vector has shape (16,) indexed as action = 4 * y + x, which reshapes
    to (4, 4) indexed as [y, x].
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

NUM_SYMMETRIES = 8
GRID_DIM = 4
NUM_ACTIONS = 16


def _validate_sym_idx(sym_idx: int) -> Tuple[int, bool]:
    if not (0 <= sym_idx < NUM_SYMMETRIES):
        raise ValueError(f"sym_idx must be in [0, {NUM_SYMMETRIES - 1}], got {sym_idx}")
    k = sym_idx % 4
    reflect = sym_idx >= 4
    return k, reflect


def transform_state(state: np.ndarray, sym_idx: int) -> np.ndarray:
    """Applies D4 symmetry `sym_idx` (0..7) to a (3, 4, 4, 4) or (B, 3, 4, 4, 4) tensor."""
    k, reflect = _validate_sym_idx(sym_idx)
    if state.shape not in ((3, 4, 4, 4),):
        if state.ndim != 5 or state.shape[1:] != (3, 4, 4, 4):
            raise ValueError(
                f"Expected state shape (3, 4, 4, 4) or (B, 3, 4, 4, 4), got {state.shape}"
            )
    out = state
    if reflect:
        out = np.flip(out, axis=-1)
    if k > 0:
        out = np.rot90(out, k=k, axes=(-2, -1))
    return np.ascontiguousarray(out)


def transform_policy(policy: np.ndarray, sym_idx: int) -> np.ndarray:
    """Applies D4 symmetry `sym_idx` (0..7) to a 16-element policy vector."""
    k, reflect = _validate_sym_idx(sym_idx)
    if policy.shape != (NUM_ACTIONS,):
        raise ValueError(f"Expected policy shape ({NUM_ACTIONS},), got {policy.shape}")
    grid = policy.reshape(GRID_DIM, GRID_DIM)
    if reflect:
        grid = np.flip(grid, axis=-1)
    if k > 0:
        grid = np.rot90(grid, k=k, axes=(0, 1))
    return np.ascontiguousarray(grid.reshape(NUM_ACTIONS))


def transform_action(action: int, sym_idx: int) -> int:
    """Maps a single column action index in {0..15} under D4 symmetry `sym_idx`."""
    if not (0 <= action < NUM_ACTIONS):
        raise ValueError(f"action must be in [0, 15], got {action}")
    one_hot = np.zeros(NUM_ACTIONS, dtype=np.float32)
    one_hot[action] = 1.0
    transformed = transform_policy(one_hot, sym_idx)
    return int(np.argmax(transformed))


def transform_valid_mask(valid_mask: int, sym_idx: int) -> int:
    """Maps a 16-bit legal move bitmask under D4 symmetry `sym_idx`."""
    _validate_sym_idx(sym_idx)
    new_mask = 0
    for a in range(NUM_ACTIONS):
        if (valid_mask & (1 << a)) != 0:
            mapped_a = transform_action(a, sym_idx)
            new_mask |= 1 << mapped_a
    return new_mask


def augment_sample(
    state: np.ndarray, policy: np.ndarray, value: float
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Expands a single (state, policy, value) sample into all 8 D4 symmetric equivalents."""
    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
    val = float(value)
    for sym_idx in range(NUM_SYMMETRIES):
        s_sym = transform_state(state, sym_idx)
        p_sym = transform_policy(policy, sym_idx)
        samples.append((s_sym, p_sym, val))
    return samples
