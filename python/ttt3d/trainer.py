"""Self-Play Reinforcement Learning, Symmetry-Augmented Replay Buffer & Policy-Value Trainer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .grpc_client import VectorizedEnvClient
from .network import NUM_ACTIONS, TTT3DNet
from .symmetry import NUM_SYMMETRIES, transform_policy, transform_state, transform_valid_mask

PLAYER_WHITE = 1
PLAYER_BLACK = 2


@dataclass
class TrainingSample:
    """Single canonical (state, policy, value, valid_mask) training example."""

    state: np.ndarray       # shape (3, 4, 4, 4) float32
    policy: np.ndarray      # shape (16,) float32 probability distribution
    value: float            # scalar in [-1.0, +1.0] from active player's perspective
    valid_mask: int = 0xFFFF


class ReplayBuffer:
    """Circular experience buffer with automatic 8-fold D4 symmetry augmentation."""

    def __init__(self, capacity: int = 50000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._buffer: Deque[TrainingSample] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add_sample(
        self,
        state: np.ndarray,
        policy: np.ndarray,
        value: float,
        valid_mask: int = 0xFFFF,
        augment: bool = True,
    ) -> int:
        """Adds a sample (or all 8 D4 symmetric transforms if augment=True) to the buffer."""
        s_arr = np.ascontiguousarray(state, dtype=np.float32)
        p_arr = np.ascontiguousarray(policy, dtype=np.float32)
        val = float(value)
        mask = int(valid_mask)

        if not augment:
            self._buffer.append(TrainingSample(s_arr, p_arr, val, mask))
            return 1

        for sym_idx in range(NUM_SYMMETRIES):
            self._buffer.append(
                TrainingSample(
                    state=transform_state(s_arr, sym_idx),
                    policy=transform_policy(p_arr, sym_idx),
                    value=val,
                    valid_mask=transform_valid_mask(mask, sym_idx),
                )
            )
        return NUM_SYMMETRIES

    def sample_batch(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples a mini-batch of (states, target_policies, target_values, valid_masks)."""
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")
        generator = rng if rng is not None else np.random.default_rng()
        actual_bs = min(int(batch_size), len(self._buffer))
        indices = generator.choice(len(self._buffer), size=actual_bs, replace=False)

        states = np.stack([self._buffer[i].state for i in indices], axis=0)
        policies = np.stack([self._buffer[i].policy for i in indices], axis=0)
        values = np.array([self._buffer[i].value for i in indices], dtype=np.float32)
        masks = np.array([self._buffer[i].valid_mask for i in indices], dtype=np.int64)

        s_t = torch.from_numpy(states)
        p_t = torch.from_numpy(policies)
        v_t = torch.from_numpy(values)
        m_t = torch.from_numpy(masks)

        if device is not None:
            s_t = s_t.to(device)
            p_t = p_t.to(device)
            v_t = v_t.to(device)
            m_t = m_t.to(device)
        return s_t, p_t, v_t, m_t


def compute_policy_value_loss(
    model: TTT3DNet,
    states: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    valid_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes combined AlphaZero-style Policy Cross-Entropy + Value MSE loss."""
    logits, pred_values = model(states, valid_masks=valid_masks)
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(target_policies * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(pred_values.squeeze(-1), target_values.reshape(-1))
    total_loss = policy_loss + value_loss
    return total_loss, policy_loss, value_loss


def train_epoch(
    model: TTT3DNet,
    optimizer: torch.optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int = 64,
    steps_per_epoch: int = 20,
    device: Optional[torch.device] = None,
    max_grad_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Runs `steps_per_epoch` mini-batch gradient updates on `model` from `buffer`."""
    if len(buffer) == 0:
        raise ValueError("ReplayBuffer must not be empty before train_epoch")
    dev = device or next(model.parameters()).device
    model.train()

    total_losses: List[float] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []

    for _ in range(steps_per_epoch):
        states, target_pi, target_v, masks = buffer.sample_batch(
            batch_size=batch_size, rng=rng, device=dev
        )
        optimizer.zero_grad()
        loss, p_loss, v_loss = compute_policy_value_loss(
            model, states, target_pi, target_v, valid_masks=masks
        )
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_losses.append(float(loss.item()))
        policy_losses.append(float(p_loss.item()))
        value_losses.append(float(v_loss.item()))

    return {
        "total_loss": float(np.mean(total_losses)),
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
    }


def generate_self_play_episodes(
    client: VectorizedEnvClient,
    model: TTT3DNet,
    buffer: ReplayBuffer,
    num_episodes: int = 4,
    temperature: float = 1.0,
    mcts_simulations: int = 64,
    use_go_mcts_teacher: bool = True,
    augment: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Generates full games via gRPC and populates `buffer` with (state, pi, z) samples.

    When `use_go_mcts_teacher=True`, queries the fast Go MCTS engine for the search policy
    target `pi_mcts` and blends action selection with `model`, rapidly bootstrapping strong
    tactical play. When `use_go_mcts_teacher=False`, uses pure neural network self-play.
    """
    generator = rng if rng is not None else np.random.default_rng()
    samples_added = 0
    white_wins = 0
    black_wins = 0
    draws = 0

    for ep in range(num_episodes):
        game_id = f"selfplay_{ep}"
        state = client.reset(game_id=game_id)
        trajectory: List[Tuple[np.ndarray, np.ndarray, int, int, float]] = []

        done = False
        while not done:
            obs = state["obs"]
            valid_mask = int(state["valid_moves_mask"])
            current_player = int(state["current_player"])

            if use_go_mcts_teacher:
                mcts_res = client.select_action(
                    agent="mcts", game_id=game_id, simulations=mcts_simulations
                )
                target_pi = mcts_res["policy"]
                mcts_val = float(mcts_res["value"])
                # Sample from MCTS visit distribution when temperature > 0
                legal = [a for a in range(NUM_ACTIONS) if (valid_mask & (1 << a)) != 0]
                if temperature <= 1e-6:
                    action = int(mcts_res["action"])
                else:
                    probs = np.array([max(float(target_pi[a]), 1e-8) for a in legal], dtype=np.float64)
                    probs /= probs.sum()
                    action = int(generator.choice(legal, p=probs))
            else:
                action, target_pi, mcts_val = model.select_action(
                    obs, valid_mask=valid_mask, temperature=temperature, rng=generator
                )

            trajectory.append((obs, target_pi, current_player, valid_mask, mcts_val))
            step_res = client.step(action=action, game_id=game_id)
            state = step_res["state"]
            done = step_res["done"]

        winner = int(state["winner"])
        if winner == PLAYER_WHITE:
            white_wins += 1
        elif winner == PLAYER_BLACK:
            black_wins += 1
        else:
            draws += 1

        for obs, target_pi, player_at_step, mask_at_step, search_val in trajectory:
            if winner == 0:
                outcome_z = 0.0
            elif winner == player_at_step:
                outcome_z = 1.0
            else:
                outcome_z = -1.0

            blended_value = 0.5 * outcome_z + 0.5 * search_val if use_go_mcts_teacher else outcome_z
            samples_added += buffer.add_sample(
                state=obs,
                policy=target_pi,
                value=blended_value,
                valid_mask=mask_at_step,
                augment=augment,
            )

    return {
        "episodes": num_episodes,
        "samples_added": samples_added,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
    }


def generate_vectorized_rollouts(
    client: VectorizedEnvClient,
    model: TTT3DNet,
    buffer: ReplayBuffer,
    num_envs: int = 8,
    num_steps: int = 32,
    temperature: float = 1.0,
    augment: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, int]:
    """Collects batched self-play trajectories across `num_envs` parallel Go environments."""
    generator = rng if rng is not None else np.random.default_rng()
    pool_state = client.init_pool(num_envs=num_envs)
    obs_batch = pool_state["obs"]
    valid_masks = pool_state["valid_masks"]
    current_players = pool_state["current_players"]

    # Per-environment active episode trajectory: list of (obs, probs, player, valid_mask)
    active_trajectories: List[List[Tuple[np.ndarray, np.ndarray, int, int]]] = [
        [] for _ in range(num_envs)
    ]
    episodes_completed = 0
    samples_added = 0

    for _ in range(num_steps):
        actions = np.zeros(num_envs, dtype=np.uint8)
        for i in range(num_envs):
            act, probs, _ = model.select_action(
                obs_batch[i],
                valid_mask=int(valid_masks[i]),
                temperature=temperature,
                rng=generator,
            )
            actions[i] = act
            active_trajectories[i].append(
                (obs_batch[i], probs, int(current_players[i]), int(valid_masks[i]))
            )

        step_out = client.batch_step(actions)
        dones = step_out["dones"]
        winners = step_out["winners"]

        for i in range(num_envs):
            if dones[i]:
                episodes_completed += 1
                winner = int(winners[i])
                for s_obs, s_pi, s_player, s_mask in active_trajectories[i]:
                    if winner == 0:
                        z = 0.0
                    elif winner == s_player:
                        z = 1.0
                    else:
                        z = -1.0
                    samples_added += buffer.add_sample(
                        s_obs, s_pi, z, valid_mask=s_mask, augment=augment
                    )
                active_trajectories[i].clear()

        obs_batch = step_out["obs"]
        valid_masks = step_out["valid_masks"]
        current_players = step_out["current_players"]

    return {
        "episodes_completed": episodes_completed,
        "samples_added": samples_added,
    }


def evaluate_against_baseline(
    client: VectorizedEnvClient,
    model: TTT3DNet,
    opponent: str = "random",
    num_games: int = 10,
    simulations: int = 32,
) -> Dict[str, Any]:
    """Benchmarks `model` (greedy temp=0) against a Go baseline agent over `num_games`."""
    if num_games <= 0:
        raise ValueError("num_games must be positive")
    wins = 0
    draws = 0
    losses = 0

    for g_idx in range(num_games):
        game_id = f"eval_{opponent}_{g_idx}"
        nn_player = PLAYER_WHITE if (g_idx % 2 == 0) else PLAYER_BLACK
        state = client.reset(game_id=game_id)
        done = False

        while not done:
            if int(state["current_player"]) == nn_player:
                action, _, _ = model.select_action(
                    state["obs"],
                    valid_mask=int(state["valid_moves_mask"]),
                    temperature=0.0,
                )
            else:
                bot_res = client.select_action(
                    agent=opponent, game_id=game_id, simulations=simulations
                )
                action = int(bot_res["action"])

            step_res = client.step(action=action, game_id=game_id)
            state = step_res["state"]
            done = step_res["done"]

        winner = int(state["winner"])
        if winner == nn_player:
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1

    score = (wins + 0.5 * draws) / float(num_games)
    return {
        "opponent": opponent,
        "num_games": num_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": score,
    }
