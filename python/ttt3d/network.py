"""3D Residual Convolutional Neural Network (Policy + Value Dual Head) for 3T3D."""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CHANNELS = 3
GEO_CHANNELS = 16
STEM_CHANNELS = IN_CHANNELS + GEO_CHANNELS
GRID_DIM = 4
GRID_CELLS = 64
NUM_ACTIONS = 16
NUM_WIN_LINES = 76
ILLEGAL_LOGIT_VALUE = -1e9


def _build_win_line_matrix() -> torch.Tensor:
    """Builds the (76, 64) float32 binary incidence matrix for all 76 winning 4-in-a-row lines."""
    lines = []

    def add_line(coords: list[Tuple[int, int, int]]) -> None:
        row = [0.0] * GRID_CELLS
        for x, y, z in coords:
            row[16 * z + 4 * y + x] = 1.0
        lines.append(row)

    # 1. Orthogonal lines along X, Y, Z (48)
    for z in range(4):
        for y in range(4):
            add_line([(x, y, z) for x in range(4)])
    for z in range(4):
        for x in range(4):
            add_line([(x, y, z) for y in range(4)])
    for y in range(4):
        for x in range(4):
            add_line([(x, y, z) for z in range(4)])

    # 2. 2D Planar diagonals in XY, XZ, YZ planes (24)
    for z in range(4):
        add_line([(i, i, z) for i in range(4)])
        add_line([(i, 3 - i, z) for i in range(4)])
    for y in range(4):
        add_line([(i, y, i) for i in range(4)])
        add_line([(i, y, 3 - i) for i in range(4)])
    for x in range(4):
        add_line([(x, i, i) for i in range(4)])
        add_line([(x, i, 3 - i) for i in range(4)])

    # 3. 3D Space diagonals (4)
    add_line([(i, i, i) for i in range(4)])
    add_line([(3 - i, i, i) for i in range(4)])
    add_line([(i, 3 - i, i) for i in range(4)])
    add_line([(i, i, 3 - i) for i in range(4)])

    return torch.tensor(lines, dtype=torch.float32)


_WIN_LINE_MATRIX: torch.Tensor = _build_win_line_matrix()


def extract_geometric_features(x: torch.Tensor) -> torch.Tensor:
    """Augments canonical (B, 3, 4, 4, 4) input with 16 exact 76-line & multi-ply gravity feature planes.

    Returns a (B, 19, 4, 4, 4) volumetric tensor:
      - Channels 0..2: Raw canonical planes (MyPieces, OpponentPieces, GravitySurface)
      - Channel 3: Immediate 1-ply winning cells on the gravity surface
      - Channel 4: Immediate 1-ply opponent threat (forced 4th-ball block) cells on gravity surface
      - Channel 5: Poison gravity cells (playing at z immediately unlocks opponent 4th-ball win at z+1)
      - Channel 6: Self-trap gravity cells (playing at z unlocks own 4th-ball at z+1 for opponent to block)
      - Channel 7: Volumetric density of own unblocked 3-in-a-row empty completion cells
      - Channel 8: Volumetric density of opponent unblocked 3-in-a-row empty completion cells
      - Channel 9: Volumetric density of own open 2-in-a-row fork/threat lines per empty cell
      - Channel 10: Volumetric density of opponent open 2-in-a-row fork/threat lines per empty cell
      - Channel 11: 2nd-ply gravity landing surface (z = h + 1)
      - Channel 12: 3rd-ply gravity landing surface (z = h + 2)
      - Channel 13: Volumetric density of own stacked vertical fork builder & target cells
      - Channel 14: Volumetric density of opponent stacked vertical fork builder & target cells
      - Channel 15: Volumetric multi-line fork intersection cells for active player
      - Channel 16: Volumetric multi-line fork intersection cells for opponent
      - Channel 17: Upper-column 2-ply/3-ply trap & Zugzwang danger on the gravity surface
      - Channel 18: Net open 1-in-a-row ray balance per empty cell
    """
    bsz = x.size(0)
    w_mat = _WIN_LINE_MATRIX.to(device=x.device, dtype=x.dtype)  # (76, 64)
    w_mat_t = w_mat.t()  # (64, 76)

    flat = x.reshape(bsz, IN_CHANNELS, GRID_CELLS)
    my = (flat[:, 0, :] > 0.5).to(x.dtype)
    opp = (flat[:, 1, :] > 0.5).to(x.dtype)
    surf = (flat[:, 2, :] > 0.5).to(x.dtype)
    empty = ((my < 0.5) & (opp < 0.5)).to(x.dtype)
    empty_3d = empty.reshape(bsz, GRID_DIM, GRID_DIM, GRID_DIM)

    my_cnt = torch.matmul(my, w_mat_t)    # (B, 76)
    opp_cnt = torch.matmul(opp, w_mat_t)  # (B, 76)

    my_open = opp_cnt < 0.5
    opp_open = my_cnt < 0.5

    l_my3 = ((my_cnt > 2.5) & (my_cnt < 3.5) & my_open).to(x.dtype)
    l_opp3 = ((opp_cnt > 2.5) & (opp_cnt < 3.5) & opp_open).to(x.dtype)
    l_my2 = ((my_cnt > 1.5) & (my_cnt < 2.5) & my_open).to(x.dtype)
    l_opp2 = ((opp_cnt > 1.5) & (opp_cnt < 2.5) & opp_open).to(x.dtype)
    l_my1 = ((my_cnt > 0.5) & (my_cnt < 1.5) & my_open).to(x.dtype)
    l_opp1 = ((opp_cnt > 0.5) & (opp_cnt < 1.5) & opp_open).to(x.dtype)

    cell_my3 = (empty * torch.matmul(l_my3, w_mat)).reshape(bsz, GRID_DIM, GRID_DIM, GRID_DIM)
    cell_opp3 = (empty * torch.matmul(l_opp3, w_mat)).reshape(bsz, GRID_DIM, GRID_DIM, GRID_DIM)
    cell_my2 = (empty * torch.matmul(l_my2, w_mat)).reshape(bsz, GRID_DIM, GRID_DIM, GRID_DIM)
    cell_opp2 = (empty * torch.matmul(l_opp2, w_mat)).reshape(bsz, GRID_DIM, GRID_DIM, GRID_DIM)
    cell_my1 = (empty * torch.matmul(l_my1, w_mat)).reshape(bsz, GRID_DIM, GRID_DIM, GRID_DIM)
    cell_opp1 = (empty * torch.matmul(l_opp1, w_mat)).reshape(bsz, GRID_DIM, GRID_DIM, GRID_DIM)
    surf_3d = surf.reshape(bsz, GRID_DIM, GRID_DIM, GRID_DIM)

    imm_win_surf = surf_3d * (cell_my3 > 0.5).to(x.dtype)
    imm_blk_surf = surf_3d * (cell_opp3 > 0.5).to(x.dtype)

    poison_surf = torch.zeros_like(surf_3d)
    poison_surf[:, :3, :, :] = surf_3d[:, :3, :, :] * (cell_opp3[:, 1:, :, :] > 0.5).to(x.dtype)

    self_trap_surf = torch.zeros_like(surf_3d)
    self_trap_surf[:, :3, :, :] = surf_3d[:, :3, :, :] * (cell_my3[:, 1:, :, :] > 0.5).to(x.dtype)

    my3_vol = torch.clamp(cell_my3, max=2.0) * 0.5
    opp3_vol = torch.clamp(cell_opp3, max=2.0) * 0.5
    my2_vol = torch.clamp(cell_my2, max=4.0) * 0.25
    opp2_vol = torch.clamp(cell_opp2, max=4.0) * 0.25

    # Channels 11..12: 2nd-ply (z = h+1) and 3rd-ply (z = h+2) gravity landing horizons
    next_surf_2ply = torch.zeros_like(surf_3d)
    next_surf_2ply[:, 1:, :, :] = surf_3d[:, :3, :, :]

    upper_surf_3ply = torch.zeros_like(surf_3d)
    upper_surf_3ply[:, 2:, :, :] = surf_3d[:, :2, :, :]

    # Channels 13..14: Stacked vertical fork targets & completion lines for both players
    my_t3_bool = cell_my3 > 0.5
    opp_t3_bool = cell_opp3 > 0.5

    my_stack_adj = torch.zeros_like( empty_3d, dtype=torch.bool)
    my_stack_adj[:, 1:, :, :] |= my_t3_bool[:, :3, :, :] & (~opp_t3_bool[:, :3, :, :])
    my_stack_adj[:, :3, :, :] |= my_t3_bool[:, 1:, :, :]
    my_stack_adj &= (empty_3d > 0.5) & (~opp_t3_bool)
    my_stack_adj_f = my_stack_adj.to(x.dtype).reshape(bsz, GRID_CELLS)
    l_my_stack = l_my2 * (torch.matmul(my_stack_adj_f, w_mat_t) > 0.5).to(x.dtype)
    cell_my_stack = empty * torch.matmul(l_my_stack, w_mat) + (
        my_t3_bool.to(x.dtype).reshape(bsz, GRID_CELLS) * my_stack_adj_f
    )
    my_stack_fork_vol = torch.clamp(
        cell_my_stack.reshape(bsz, GRID_DIM, GRID_DIM, GRID_DIM), max=2.0
    ) * 0.5

    opp_stack_adj = torch.zeros_like(empty_3d, dtype=torch.bool)
    opp_stack_adj[:, 1:, :, :] |= opp_t3_bool[:, :3, :, :] & (~my_t3_bool[:, :3, :, :])
    opp_stack_adj[:, :3, :, :] |= opp_t3_bool[:, 1:, :, :]
    opp_stack_adj &= (empty_3d > 0.5) & (~my_t3_bool)
    opp_stack_adj_f = opp_stack_adj.to(x.dtype).reshape(bsz, GRID_CELLS)
    l_opp_stack = l_opp2 * (torch.matmul(opp_stack_adj_f, w_mat_t) > 0.5).to(x.dtype)
    cell_opp_stack = empty * torch.matmul(l_opp_stack, w_mat) + (
        opp_t3_bool.to(x.dtype).reshape(bsz, GRID_CELLS) * opp_stack_adj_f
    )
    opp_stack_fork_vol = torch.clamp(
        cell_opp_stack.reshape(bsz, GRID_DIM, GRID_DIM, GRID_DIM), max=2.0
    ) * 0.5

    # Channels 15..16: Multi-line fork intersection cells
    my_fork_intersect_vol = ((cell_my2 > 1.5) | ((cell_my2 > 0.5) & (cell_my3 > 0.5))).to(x.dtype)
    opp_fork_intersect_vol = ((cell_opp2 > 1.5) | ((cell_opp2 > 0.5) & (cell_opp3 > 0.5))).to(x.dtype)

    # Channel 17: Upper-column 2-ply/3-ply trap & Zugzwang danger on the gravity surface
    upper_trap_surf = torch.zeros_like(surf_3d)
    upper_trap_surf[:, :2, :, :] = surf_3d[:, :2, :, :] * (cell_opp3[:, 2:, :, :] > 0.5).to(x.dtype)
    upper_trap_surf[:, :3, :, :] = torch.maximum(
        upper_trap_surf[:, :3, :, :],
        surf_3d[:, :3, :, :] * (cell_opp2[:, 1:, :, :] > 1.5).to(x.dtype) * 0.5,
    )

    # Channel 18: Net open 1-in-a-row ray balance per empty cell
    open1_diff_vol = torch.clamp((cell_my1 - cell_opp1) * 0.2, min=-1.0, max=1.0)

    geo = torch.stack(
        [
            imm_win_surf,
            imm_blk_surf,
            poison_surf,
            self_trap_surf,
            my3_vol,
            opp3_vol,
            my2_vol,
            opp2_vol,
            next_surf_2ply,
            upper_surf_3ply,
            my_stack_fork_vol,
            opp_stack_fork_vol,
            my_fork_intersect_vol,
            opp_fork_intersect_vol,
            upper_trap_surf,
            open1_diff_vol,
        ],
        dim=1,
    )
    return torch.cat([x, geo], dim=1)


def valid_mask_to_tensor(
    valid_masks: Union[int, np.ndarray, torch.Tensor, list],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Converts 16-bit integer bitmask(s) or boolean mask(s) into a (B, 16) bool Tensor.

    Bit `a` (0..15) is True if column `a` is a legal move, False if column `a` is full.
    """
    if isinstance(valid_masks, int):
        arr = np.array([valid_masks], dtype=np.int64)
    elif isinstance(valid_masks, list):
        arr = np.asarray(valid_masks, dtype=np.int64)
    elif isinstance(valid_masks, np.ndarray):
        if valid_masks.dtype == np.bool_:
            t = torch.from_numpy(valid_masks)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            return t.to(device=device) if device is not None else t
        arr = valid_masks.astype(np.int64)
    elif isinstance(valid_masks, torch.Tensor):
        if valid_masks.dtype == torch.bool:
            t = valid_masks.unsqueeze(0) if valid_masks.ndim == 1 else valid_masks
            return t.to(device=device) if device is not None else t
        bits = torch.arange(NUM_ACTIONS, device=valid_masks.device)
        vm = valid_masks.unsqueeze(0) if valid_masks.ndim == 0 else valid_masks
        mask_bool = ((vm.unsqueeze(-1).long() >> bits) & 1).bool()
        return mask_bool.to(device=device) if device is not None else mask_bool
    else:
        raise TypeError(f"Unsupported valid_masks type: {type(valid_masks)}")

    if arr.ndim == 0:
        arr = arr.reshape(1)
    bits = np.arange(NUM_ACTIONS, dtype=np.int64)
    mask_np = ((arr[:, None] >> bits[None, :]) & 1).astype(bool)
    tensor = torch.from_numpy(mask_np)
    return tensor.to(device=device) if device is not None else tensor


def mask_illegal_logits(
    logits: torch.Tensor,
    valid_masks: Optional[Union[int, np.ndarray, torch.Tensor, list]] = None,
) -> torch.Tensor:
    """Sets logits of illegal actions (where valid_mask bit is 0) to ILLEGAL_LOGIT_VALUE (-1e9)."""
    if valid_masks is None:
        return logits
    mask_bool = valid_mask_to_tensor(valid_masks, device=logits.device)
    if mask_bool.shape != logits.shape:
        raise ValueError(
            f"Mask shape {mask_bool.shape} does not match logits shape {logits.shape}"
        )
    # Guard against terminal states where all 16 bits are 0 so softmax never produces NaN
    has_any_valid = mask_bool.any(dim=-1, keepdim=True)
    effective_mask = torch.where(has_any_valid, mask_bool, torch.ones_like(mask_bool))
    return logits.masked_fill(~effective_mask, ILLEGAL_LOGIT_VALUE)


class ResBlock3D(nn.Module):
    """3D Residual Block combining local 3x3x3 conv and full-cube 4x4x4 global conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv_global = nn.Conv3d(channels, channels, kernel_size=GRID_DIM, padding=0, bias=False)
        self.bn_global = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        local_feat = self.bn1(self.conv1(x))
        global_feat = self.bn_global(self.conv_global(x))
        out = F.relu(local_feat + global_feat)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class TTT3DNet(nn.Module):
    """Dual-head 3D Residual CNN for 4x4x4 Gravity Tic-Tac-Toe (3T3D).

    Input:
      - x: float32 Tensor of shape (B, 3, 4, 4, 4)
      - valid_masks: optional 16-bit legal action masks of length B

    Outputs:
      - policy_logits: Tensor of shape (B, 16) (masked with -1e9 on illegal actions if valid_masks given)
      - value: Tensor of shape (B, 1) in [-1.0, +1.0]
    """

    def __init__(self, channels: int = 64, num_blocks: int = 3) -> None:
        super().__init__()
        if channels <= 0 or num_blocks <= 0:
            raise ValueError("channels and num_blocks must be positive integers")
        self.channels = channels
        self.num_blocks = num_blocks

        # Stem 3D convolution over 11 channels (3 raw + 8 76-line geometric planes)
        self.stem_conv = nn.Conv3d(STEM_CHANNELS, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm3d(channels)

        # Residual tower (3x3x3 local + 4x4x4 full-board global receptive field)
        self.res_blocks = nn.ModuleList([ResBlock3D(channels) for _ in range(num_blocks)])

        # Policy Head: 1x1x1 Conv3d -> BN -> ReLU -> Linear -> 16 column logits
        self.policy_conv = nn.Conv3d(channels, 16, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm3d(16)
        self.policy_fc = nn.Linear(16 * GRID_CELLS, NUM_ACTIONS)

        # Value Head: 1x1x1 Conv3d -> BN -> ReLU -> Linear(64) -> ReLU -> Linear(1) -> Tanh
        self.value_conv = nn.Conv3d(channels, 8, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm3d(8)
        self.value_fc1 = nn.Linear(8 * GRID_CELLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(
        self,
        x: torch.Tensor,
        valid_masks: Optional[Union[int, np.ndarray, torch.Tensor, list]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 5 or x.shape[1:] != (IN_CHANNELS, GRID_DIM, GRID_DIM, GRID_DIM):
            raise ValueError(
                f"Expected input tensor shape (B, 3, 4, 4, 4), got {tuple(x.shape)}"
            )
        stem_in = extract_geometric_features(x)
        h = F.relu(self.stem_bn(self.stem_conv(stem_in)))
        for block in self.res_blocks:
            h = block(h)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(h)))
        p = p.reshape(p.size(0), -1)
        logits = self.policy_fc(p)
        if valid_masks is not None:
            logits = mask_illegal_logits(logits, valid_masks)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(h)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return logits, value

    @torch.no_grad()
    def predict(
        self,
        state: Union[np.ndarray, torch.Tensor],
        valid_mask: int = 0xFFFF,
        device: Optional[torch.device] = None,
    ) -> Tuple[np.ndarray, float]:
        """Runs single-state inference and returns (policy_probs_16, scalar_value)."""
        was_training = self.training
        self.eval()
        try:
            dev = device or next(self.parameters()).device
            if isinstance(state, np.ndarray):
                t = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float32))
            else:
                t = state.float()
            if t.ndim == 4:
                t = t.unsqueeze(0)
            t = t.to(dev)
            logits, val = self.forward(t, valid_masks=int(valid_mask))
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            # Explicitly zero out any masked illegal slots for exactness
            for a in range(NUM_ACTIONS):
                if (int(valid_mask) & (1 << a)) == 0 and int(valid_mask) != 0:
                    probs[a] = 0.0
            prob_sum = float(probs.sum())
            if prob_sum > 0:
                probs = probs / prob_sum
            return probs.astype(np.float32), float(val.item())
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        valid_mask: int = 0xFFFF,
        temperature: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, np.ndarray, float]:
        """Selects a legal column action in {0..15} using greedy (temp=0) or stochastic sampling."""
        if valid_mask == 0:
            raise ValueError("Cannot select action when valid_mask is 0 (no legal moves)")
        probs, value = self.predict(state, valid_mask=valid_mask)
        legal_actions = [a for a in range(NUM_ACTIONS) if (valid_mask & (1 << a)) != 0]

        if temperature <= 1e-6:
            best_action = max(legal_actions, key=lambda a: float(probs[a]))
            return int(best_action), probs, value

        # Temperature-scaled distribution over legal moves
        legal_probs = np.array([max(float(probs[a]), 1e-12) for a in legal_actions], dtype=np.float64)
        scaled = np.power(legal_probs, 1.0 / float(temperature))
        scaled /= scaled.sum()
        generator = rng if rng is not None else np.random.default_rng()
        chosen = int(generator.choice(legal_actions, p=scaled))
        return chosen, probs, value

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Saves model weights and architecture config to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "channels": self.channels,
                "num_blocks": self.num_blocks,
                "state_dict": self.state_dict(),
            },
            target,
        )

    @classmethod
    def load_checkpoint(
        cls,
        path: Union[str, Path],
        map_location: Union[str, torch.device] = "cpu",
    ) -> "TTT3DNet":
        """Loads a TTT3DNet checkpoint from disk."""
        payload = torch.load(Path(path), map_location=map_location, weights_only=True)
        model = cls(
            channels=int(payload.get("channels", 64)),
            num_blocks=int(payload.get("num_blocks", 3)),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        return model

    @torch.no_grad()
    def export_fused_binary(self, path: Union[str, Path]) -> None:
        """Exports BatchNorm-fused float32 weights into a compact binary file for Go inference."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        def fuse_conv_bn(conv: nn.Conv3d, bn: nn.BatchNorm3d) -> Tuple[np.ndarray, np.ndarray]:
            w = conv.weight.detach().cpu().float()
            gamma = bn.weight.detach().cpu().float()
            beta = bn.bias.detach().cpu().float()
            mean = bn.running_mean.detach().cpu().float()
            var = bn.running_var.detach().cpu().float()
            scale = gamma / torch.sqrt(var + bn.eps)
            w_fused = (w * scale.view(-1, 1, 1, 1, 1)).numpy().astype("<f4")
            b_fused = (beta - mean * scale).numpy().astype("<f4")
            return w_fused, b_fused

        chunks = [struct.pack("<4sIII", b"3T3D", 1, int(self.channels), int(self.num_blocks))]

        def add_pair(w: np.ndarray, b: np.ndarray) -> None:
            chunks.append(np.ascontiguousarray(w, dtype="<f4").tobytes())
            chunks.append(np.ascontiguousarray(b, dtype="<f4").tobytes())

        add_pair(*fuse_conv_bn(self.stem_conv, self.stem_bn))
        for block in self.res_blocks:
            add_pair(*fuse_conv_bn(block.conv1, block.bn1))
            add_pair(*fuse_conv_bn(block.conv_global, block.bn_global))
            add_pair(*fuse_conv_bn(block.conv2, block.bn2))

        add_pair(*fuse_conv_bn(self.policy_conv, self.policy_bn))
        add_pair(
            self.policy_fc.weight.detach().cpu().numpy().astype("<f4"),
            self.policy_fc.bias.detach().cpu().numpy().astype("<f4"),
        )

        add_pair(*fuse_conv_bn(self.value_conv, self.value_bn))
        add_pair(
            self.value_fc1.weight.detach().cpu().numpy().astype("<f4"),
            self.value_fc1.bias.detach().cpu().numpy().astype("<f4"),
        )
        add_pair(
            self.value_fc2.weight.detach().cpu().numpy().astype("<f4"),
            self.value_fc2.bias.detach().cpu().numpy().astype("<f4"),
        )

        target.write_bytes(b"".join(chunks))
