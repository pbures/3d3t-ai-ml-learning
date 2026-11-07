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
GRID_DIM = 4
GRID_CELLS = 64
NUM_ACTIONS = 16
ILLEGAL_LOGIT_VALUE = -1e9


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
    """3D Residual Block with two 3x3x3 volumetric convolutions and skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
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

        # Stem 3D convolution
        self.stem_conv = nn.Conv3d(IN_CHANNELS, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm3d(channels)

        # Residual tower
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
        h = F.relu(self.stem_bn(self.stem_conv(x)))
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
