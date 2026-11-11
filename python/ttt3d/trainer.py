"""Self-Play Reinforcement Learning, Symmetry-Augmented Replay Buffer & Policy-Value Trainer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .arena_bot import (
    WIN_LINE_MASKS,
    NeuralArenaBot,
    _build_canonical_from_bitboards,
    check_bitboard_win,
    evaluate_bitboard_position,
    find_immediate_wins,
    has_unstoppable_threat,
    landing_bit_for_action,
)
from .grpc_client import VectorizedEnvClient
from .network import NUM_ACTIONS, TTT3DNet, valid_mask_to_tensor
from .symmetry import NUM_SYMMETRIES, transform_policy, transform_state, transform_valid_mask

PLAYER_WHITE = 1
PLAYER_BLACK = 2

_POW2_64: Tuple[int, ...] = tuple(1 << i for i in range(64))


def _canonical_tensor_to_bitboards(state: np.ndarray) -> Tuple[int, int]:
    """Extracts 64-bit (my_bb, opp_bb) integer bitboards from a (3, 4, 4, 4) canonical tensor."""
    flat = state.reshape(3, 64)
    my_plane = flat[0]
    opp_plane = flat[1]
    my_bb = 0
    opp_bb = 0
    for bit in range(64):
        if my_plane[bit] > 0.5:
            my_bb |= _POW2_64[bit]
        if opp_plane[bit] > 0.5:
            opp_bb |= _POW2_64[bit]
    return my_bb, opp_bb


def compute_tactical_action_masks(
    state: np.ndarray, valid_mask: int
) -> Tuple[int, int, bool]:
    """Computes 1-ply tactical action bitmasks for canonical `state` (3, 4, 4, 4).

    Returns:
      - win_mask: 16-bit mask of legal actions that immediately win on this ply
      - blunder_mask: 16-bit mask of legal actions that either:
          (a) fail to play an immediate winning move when `win_mask != 0`, OR
          (b) hand the opponent an immediate 1-ply winning move on the next ply
              (missing a forced block or playing a poison column) when at least
              one safe continuation exists
      - forced_loss: True if `win_mask == 0` and EVERY legal action allows an
        immediate 1-ply opponent win (unstoppable opponent fork)
    """
    valid_mask = int(valid_mask) & 0xFFFF
    if valid_mask == 0:
        return 0, 0, False

    my_bb, opp_bb = _canonical_tensor_to_bitboards(state)
    occupied = my_bb | opp_bb

    # 1. Check immediate 1-ply winning moves for active player
    win_mask = 0
    for a in range(NUM_ACTIONS):
        if (valid_mask & (1 << a)) == 0:
            continue
        lbit = landing_bit_for_action(occupied, a)
        if lbit is not None and check_bitboard_win(my_bb | (1 << lbit)):
            win_mask |= 1 << a

    if win_mask != 0:
        blunder_mask = valid_mask & (~win_mask)
        return win_mask, blunder_mask, False

    # 2. Check which legal actions leave the opponent with an immediate 1-ply win
    lose_mask = 0
    for a in range(NUM_ACTIONS):
        if (valid_mask & (1 << a)) == 0:
            continue
        lbit = landing_bit_for_action(occupied, a)
        if lbit is None:
            continue
        next_occ = occupied | (1 << lbit)
        if len(find_immediate_wins(opp_bb, next_occ)) > 0:
            lose_mask |= 1 << a

    safe_mask = valid_mask & (~lose_mask)
    if lose_mask != 0 and safe_mask != 0:
        return 0, lose_mask, False
    if lose_mask != 0 and safe_mask == 0:
        return 0, 0, True
    return 0, 0, False


def rectify_tactical_sample(
    state: np.ndarray,
    policy: np.ndarray,
    value: float,
    valid_mask: int,
) -> Tuple[np.ndarray, float, int]:
    """Suppresses 1-ply losing moves in `policy` and clamps 1-ply terminal `value` targets."""
    p_rect = np.array(policy, dtype=np.float32, copy=True)
    v_rect = float(value)
    valid_mask = int(valid_mask) & 0xFFFF

    win_mask, blunder_mask, forced_loss = compute_tactical_action_masks(state, valid_mask)
    if win_mask != 0:
        # Active player can win in 1 move: concentrate 100% policy on winning moves
        for a in range(NUM_ACTIONS):
            if (win_mask & (1 << a)) == 0:
                p_rect[a] = 0.0
        total = float(p_rect.sum())
        if total > 1e-8:
            p_rect /= total
        else:
            win_actions = [a for a in range(NUM_ACTIONS) if (win_mask & (1 << a)) != 0]
            for a in win_actions:
                p_rect[a] = 1.0 / len(win_actions)
        v_rect = 1.0
        return p_rect, v_rect, blunder_mask

    if forced_loss:
        # Opponent has an unstoppable 1-ply double threat
        v_rect = -1.0
        return p_rect, v_rect, 0

    if blunder_mask != 0:
        # Suppress all 1-ply losing moves (missed blocks & poison columns) to 0 probability
        safe_mask = valid_mask & (~blunder_mask)
        for a in range(NUM_ACTIONS):
            if (safe_mask & (1 << a)) == 0:
                p_rect[a] = 0.0
        total = float(p_rect.sum())
        if total > 1e-8:
            p_rect /= total
        else:
            safe_actions = [a for a in range(NUM_ACTIONS) if (safe_mask & (1 << a)) != 0]
            for a in safe_actions:
                p_rect[a] = 1.0 / len(safe_actions)

    return p_rect, v_rect, blunder_mask


@dataclass
class TrainingSample:
    """Single canonical (state, policy, value, valid_mask, blunder_mask) training example."""

    state: np.ndarray       # shape (3, 4, 4, 4) float32
    policy: np.ndarray      # shape (16,) float32 probability distribution
    value: float            # scalar in [-1.0, +1.0] from active player's perspective
    valid_mask: int = 0xFFFF
    blunder_mask: int = 0


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
        rectify_tactical: bool = True,
        extra_blunder_mask: int = 0,
    ) -> int:
        """Adds a sample (or all 8 D4 symmetric transforms if augment=True) to the buffer."""
        s_arr = np.ascontiguousarray(state, dtype=np.float32)
        p_arr = np.ascontiguousarray(policy, dtype=np.float32)
        val = float(value)
        mask = int(valid_mask) & 0xFFFF
        blunder_mask = int(extra_blunder_mask) & mask

        if rectify_tactical:
            p_arr, val, tac_blunder = rectify_tactical_sample(s_arr, p_arr, val, mask)
            blunder_mask |= tac_blunder

        if not augment:
            self._buffer.append(TrainingSample(s_arr, p_arr, val, mask, blunder_mask))
            return 1

        for sym_idx in range(NUM_SYMMETRIES):
            self._buffer.append(
                TrainingSample(
                    state=transform_state(s_arr, sym_idx),
                    policy=transform_policy(p_arr, sym_idx),
                    value=val,
                    valid_mask=transform_valid_mask(mask, sym_idx),
                    blunder_mask=transform_valid_mask(blunder_mask, sym_idx),
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
        s_t, p_t, v_t, m_t, _ = self.sample_batch_with_blunders(
            batch_size=batch_size, rng=rng, device=device
        )
        return s_t, p_t, v_t, m_t

    def sample_batch_with_blunders(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Samples (states, target_policies, target_values, valid_masks, blunder_masks)."""
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from an empty ReplayBuffer")
        generator = rng if rng is not None else np.random.default_rng()
        actual_bs = min(int(batch_size), len(self._buffer))
        indices = generator.choice(len(self._buffer), size=actual_bs, replace=False)

        states = np.stack([self._buffer[i].state for i in indices], axis=0)
        policies = np.stack([self._buffer[i].policy for i in indices], axis=0)
        values = np.array([self._buffer[i].value for i in indices], dtype=np.float32)
        masks = np.array([self._buffer[i].valid_mask for i in indices], dtype=np.int64)
        blunders = np.array([self._buffer[i].blunder_mask for i in indices], dtype=np.int64)

        s_t = torch.from_numpy(states)
        p_t = torch.from_numpy(policies)
        v_t = torch.from_numpy(values)
        m_t = torch.from_numpy(masks)
        b_t = torch.from_numpy(blunders)

        if device is not None:
            s_t = s_t.to(device)
            p_t = p_t.to(device)
            v_t = v_t.to(device)
            m_t = m_t.to(device)
            b_t = b_t.to(device)
        return s_t, p_t, v_t, m_t, b_t


def compute_policy_value_loss(
    model: TTT3DNet,
    states: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    valid_masks: Optional[torch.Tensor] = None,
    blunder_masks: Optional[torch.Tensor] = None,
    blunder_weight: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes AlphaZero Policy Cross-Entropy + 1-Ply Blunder Suppression + Value MSE loss."""
    logits, pred_values = model(states, valid_masks=valid_masks)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)

    policy_ce_loss = -(target_policies * log_probs).sum(dim=-1).mean()
    blunder_loss = torch.zeros((), dtype=logits.dtype, device=logits.device)

    if blunder_masks is not None and blunder_weight > 0.0:
        blunder_bool = valid_mask_to_tensor(blunder_masks).to(logits.device)
        if valid_masks is not None:
            legal_bool = valid_mask_to_tensor(valid_masks).to(logits.device)
        else:
            legal_bool = torch.ones_like(blunder_bool)
        safe_bool = legal_bool & (~blunder_bool)
        has_tactical = blunder_bool.any(dim=-1) & safe_bool.any(dim=-1)

        if bool(has_tactical.any()):
            # 1. Unlikelihood penalty on probability mass assigned to 1-ply losing moves
            blunder_prob_mass = (probs * blunder_bool.float()).sum(dim=-1)
            unlikelihood = -torch.log((1.0 - blunder_prob_mass).clamp(min=1e-6))

            # 2. Symmetric logit margin penalty: pull target safe logit UP and push 1-ply blunder logits DOWN
            # Combines mean blunder violation with worst-case (max) blunder logit violation so argmax is safe
            target_safe_logit = (logits * target_policies).sum(dim=-1, keepdim=True)
            margin_violation = F.relu(logits - target_safe_logit + 2.5) * blunder_bool.float()
            blunder_count = blunder_bool.float().sum(dim=-1).clamp(min=1.0)
            mean_margin_loss = (margin_violation.pow(2).sum(dim=-1)) / blunder_count

            max_blunder_logit = logits.masked_fill(~blunder_bool, -1e9).max(dim=-1).values
            max_margin_loss = F.relu(max_blunder_logit - target_safe_logit.squeeze(-1) + 3.0).pow(2)
            margin_loss = 0.5 * mean_margin_loss + 0.5 * max_margin_loss

            tactical_mask_f = has_tactical.float()
            blunder_loss = (
                (unlikelihood + 0.35 * margin_loss) * tactical_mask_f
            ).sum() / tactical_mask_f.sum().clamp(min=1.0)

    policy_loss = policy_ce_loss + blunder_weight * blunder_loss
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
    blunder_weight: float = 4.0,
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
        states, target_pi, target_v, masks, blunders = buffer.sample_batch_with_blunders(
            batch_size=batch_size, rng=rng, device=dev
        )
        optimizer.zero_grad()
        loss, p_loss, v_loss = compute_policy_value_loss(
            model,
            states,
            target_pi,
            target_v,
            valid_masks=masks,
            blunder_masks=blunders,
            blunder_weight=blunder_weight,
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
        trajectory: List[Tuple[np.ndarray, np.ndarray, int, int, float, int]] = []

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
                # Mix MCTS teacher moves with model's own on-policy moves (DAgger exploration)
                legal = [a for a in range(NUM_ACTIONS) if (valid_mask & (1 << a)) != 0]
                if temperature > 1e-6 and float(generator.random()) < 0.35:
                    action, _, _ = model.select_action(
                        obs, valid_mask=valid_mask, temperature=temperature, rng=generator
                    )
                elif temperature <= 1e-6:
                    action = int(mcts_res["action"])
                else:
                    probs = np.array([max(float(target_pi[a]), 1e-8) for a in legal], dtype=np.float64)
                    probs /= probs.sum()
                    action = int(generator.choice(legal, p=probs))
            else:
                action, target_pi, mcts_val = model.select_action(
                    obs, valid_mask=valid_mask, temperature=temperature, rng=generator
                )

            trajectory.append((obs, target_pi, current_player, valid_mask, mcts_val, int(action)))
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

        for obs, target_pi, player_at_step, mask_at_step, search_val, act_taken in trajectory:
            if winner == 0:
                outcome_z = 0.0
            elif winner == player_at_step:
                outcome_z = 1.0
            else:
                outcome_z = -1.0

            my_bb, opp_bb = _canonical_tensor_to_bitboards(obs)
            teacher_pi, teacher_val = _compute_tactical_teacher_policy(my_bb, opp_bb, mask_at_step)
            blended_pi = 0.5 * target_pi + 0.5 * teacher_pi if use_go_mcts_teacher else teacher_pi
            pi_sum = float(blended_pi.sum())
            if pi_sum > 1e-8:
                blended_pi = blended_pi / pi_sum

            _, step_blunder_mask, _ = compute_tactical_action_masks(obs, mask_at_step)
            if (step_blunder_mask & (1 << act_taken)) != 0:
                # Direct 1-ply blunder penalty if the executed move missed an immediate threat
                blended_value = -1.0
            else:
                blended_value = (
                    0.4 * outcome_z + 0.3 * search_val + 0.3 * teacher_val
                    if use_go_mcts_teacher
                    else 0.5 * outcome_z + 0.5 * teacher_val
                )

            samples_added += buffer.add_sample(
                state=obs,
                policy=blended_pi,
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


_CELL_LINE_COUNTS: Tuple[int, ...] = tuple(
    sum(1 for m in WIN_LINE_MASKS if (m & (1 << b)) != 0) for b in range(64)
)


def _evaluate_bitboard_heuristic(my_bb: int, opp_bb: int) -> int:
    """Computes the 76-line + suspended-threat positional score from `my_bb`'s perspective."""
    return evaluate_bitboard_position(my_bb, opp_bb)


def _compute_tactical_teacher_policy(
    my_bb: int, opp_bb: int, valid_mask: int
) -> Tuple[np.ndarray, float]:
    """Computes a sharp tactical + 2-ply fork + 76-line positional policy and value target."""
    pi = np.zeros(NUM_ACTIONS, dtype=np.float32)
    valid_mask = int(valid_mask) & 0xFFFF
    if valid_mask == 0:
        return pi, 0.0

    occ = my_bb | opp_bb
    legal = [a for a in range(NUM_ACTIONS) if (valid_mask & (1 << a)) != 0]

    # 1. Immediate 1-ply win
    my_wins = find_immediate_wins(my_bb, occ)
    if my_wins:
        for a in my_wins:
            pi[a] = 1.0 / len(my_wins)
        return pi, 1.0

    # 2. Forced 1-ply block of opponent's 4th-ball threat
    opp_wins = find_immediate_wins(opp_bb, occ)
    if opp_wins:
        safe_blocks = [
            a
            for a in opp_wins
            if not find_immediate_wins(opp_bb, occ | (1 << (landing_bit_for_action(occ, a) or 0)))
        ]
        blocks = safe_blocks if safe_blocks else opp_wins
        for a in blocks:
            pi[a] = 1.0 / len(blocks)
        val = 0.0 if (len(opp_wins) == 1 and len(safe_blocks) > 0) else -1.0
        return pi, val

    # 3. Filter out 1-ply poison moves
    safe_moves: List[int] = []
    for a in legal:
        lbit = landing_bit_for_action(occ, a)
        if lbit is not None and not find_immediate_wins(opp_bb, occ | (1 << lbit)):
            safe_moves.append(a)
    if not safe_moves:
        for a in legal:
            pi[a] = 1.0 / len(legal)
        return pi, -1.0

    # 4. Offensive 2-way or stacked fork
    fork_moves: List[int] = []
    for a in safe_moves:
        lbit = landing_bit_for_action(occ, a)
        if lbit is not None and has_unstoppable_threat(my_bb | (1 << lbit), opp_bb, occ | (1 << lbit)):
            fork_moves.append(a)
    if fork_moves:
        for a in fork_moves:
            pi[a] = 1.0 / len(fork_moves)
        return pi, 1.0

    # 5. Defensive fork prevention (avoid letting opponent reply with a win or 2-way/stacked fork)
    non_fork_vulnerable: List[int] = []
    for a in safe_moves:
        lbit = landing_bit_for_action(occ, a)
        if lbit is None:
            continue
        next_my = my_bb | (1 << lbit)
        next_occ = occ | (1 << lbit)
        my_threats = find_immediate_wins(next_my, next_occ)
        opp_replies = my_threats if my_threats else [
            r for r in range(NUM_ACTIONS) if landing_bit_for_action(next_occ, r) is not None
        ]
        opp_can_fork = False
        for r in opp_replies:
            r_bit = landing_bit_for_action(next_occ, r)
            if r_bit is None:
                continue
            opp_after = opp_bb | (1 << r_bit)
            occ_after = next_occ | (1 << r_bit)
            if check_bitboard_win(opp_after):
                opp_can_fork = True
                break
            if has_unstoppable_threat(opp_after, next_my, occ_after) and not find_immediate_wins(next_my, occ_after):
                opp_can_fork = True
                break
        if not opp_can_fork:
            non_fork_vulnerable.append(a)

    candidates = non_fork_vulnerable if non_fork_vulnerable else safe_moves

    # 6. Rank candidates by 2-ply minimax over 76-line heuristic + cell connectivity
    scores = np.empty(len(candidates), dtype=np.float64)
    for idx, a in enumerate(candidates):
        lbit = landing_bit_for_action(occ, a)
        assert lbit is not None
        next_my = my_bb | (1 << lbit)
        next_occ = occ | (1 << lbit)
        my_threats = find_immediate_wins(next_my, next_occ)
        opp_replies = my_threats if my_threats else [
            r for r in range(NUM_ACTIONS) if landing_bit_for_action(next_occ, r) is not None
        ]
        worst_reply_score = 1e9
        for r in opp_replies:
            r_bit = landing_bit_for_action(next_occ, r)
            if r_bit is None:
                continue
            opp_after = opp_bb | (1 << r_bit)
            if check_bitboard_win(opp_after):
                reply_s = -10000.0
            else:
                reply_s = float(
                    _evaluate_bitboard_heuristic(next_my, opp_after)
                    + _CELL_LINE_COUNTS[lbit] * 2
                    - _CELL_LINE_COUNTS[r_bit] * 2
                )
            if reply_s < worst_reply_score:
                worst_reply_score = reply_s
        if worst_reply_score > 1e8:
            worst_reply_score = float(_evaluate_bitboard_heuristic(next_my, opp_bb) + _CELL_LINE_COUNTS[lbit] * 2)
        scores[idx] = worst_reply_score

    max_s = float(scores.max())
    weights = np.exp((scores - max_s) / 4.0)
    weights /= weights.sum()
    for idx, a in enumerate(candidates):
        pi[a] = float(weights[idx])

    val_est = float(np.tanh(max_s / 80.0))
    return pi, val_est


def _zero_prob_blunder_mask(policy: np.ndarray, valid_mask: int) -> int:
    """Returns a 16-bit mask of legal actions that have zero probability in `policy`."""
    valid_mask = int(valid_mask) & 0xFFFF
    zero_mask = 0
    for a in range(NUM_ACTIONS):
        if (valid_mask & (1 << a)) != 0 and float(policy[a]) <= 1e-6:
            zero_mask |= 1 << a
    if zero_mask != 0 and (valid_mask & (~zero_mask)) != 0:
        return zero_mask
    return 0


def generate_adversarial_sparring(
    client: VectorizedEnvClient,
    model: TTT3DNet,
    buffer: ReplayBuffer,
    num_games_per_opponent: int = 4,
    augment: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Runs DAgger sparring games of Pure `model` against `heuristic` and `alphabeta`."""
    generator = rng if rng is not None else np.random.default_rng()
    samples_added = 0

    for opp_name in ("heuristic", "alphabeta"):
        for g_idx in range(num_games_per_opponent):
            game_id = f"spar_{opp_name}_{g_idx}"
            nn_player = PLAYER_WHITE if (g_idx % 2 == 0) else PLAYER_BLACK
            # First two games per opponent use greedy temp=0.0 to hit exact deterministic eval lines
            temp = 0.0 if g_idx < 2 else 0.25
            state = client.reset(game_id=game_id)
            done = False
            traj: List[Tuple[np.ndarray, int, int, int, int]] = []

            while not done:
                obs = state["obs"]
                valid_mask = int(state["valid_moves_mask"])
                cur_player = int(state["current_player"])
                ab_res = client.select_action(agent="alphabeta", game_id=game_id)
                ab_act = int(ab_res["action"])

                if cur_player == nn_player:
                    act, _, _ = model.select_action(
                        obs, valid_mask=valid_mask, temperature=temp, rng=generator
                    )
                else:
                    opp_res = client.select_action(agent=opp_name, game_id=game_id)
                    act = int(opp_res["action"])

                traj.append((obs, cur_player, valid_mask, int(act), ab_act))
                step_res = client.step(action=int(act), game_id=game_id)
                state = step_res["state"]
                done = step_res["done"]

            winner = int(state["winner"])
            for obs, player_at_step, mask_at_step, act_taken, ab_act in traj:
                my_bb, opp_bb = _canonical_tensor_to_bitboards(obs)
                teacher_pi, teacher_val = _compute_tactical_teacher_policy(
                    my_bb, opp_bb, mask_at_step
                )
                extra_bm = _zero_prob_blunder_mask(teacher_pi, mask_at_step)

                if player_at_step == nn_player:
                    # Boost the depth-3 Alpha-Beta move if it is fork-safe
                    if teacher_pi[ab_act] > 0:
                        target_pi = 0.25 * teacher_pi
                        target_pi[ab_act] += 0.75
                        target_pi /= target_pi.sum()
                    else:
                        target_pi = teacher_pi
                    # If the pure model lost this game and deviated from the safe teacher move,
                    # explicitly mark the losing move in extra_bm so margin loss suppresses it
                    if winner != 0 and winner != nn_player and act_taken != ab_act and target_pi[ab_act] > 0.2:
                        target_pi[act_taken] = 0.0
                        s_tot = float(target_pi.sum())
                        if s_tot > 1e-8:
                            target_pi /= s_tot
                        extra_bm |= 1 << act_taken
                else:
                    # If the opponent won, also learn from the opponent's winning move if fork-safe
                    if winner == player_at_step and teacher_pi[act_taken] > 0:
                        target_pi = 0.25 * teacher_pi
                        target_pi[act_taken] += 0.75
                        target_pi /= target_pi.sum()
                    else:
                        target_pi = teacher_pi

                if winner == 0:
                    z = 0.5 * teacher_val
                elif winner == player_at_step:
                    z = 0.7 * 1.0 + 0.3 * teacher_val
                else:
                    z = 0.7 * (-1.0) + 0.3 * teacher_val

                samples_added += buffer.add_sample(
                    state=obs,
                    policy=target_pi,
                    value=z,
                    valid_mask=mask_at_step,
                    augment=augment,
                    rectify_tactical=True,
                    extra_blunder_mask=extra_bm,
                )

    return samples_added


def generate_tactical_threat_curriculum(
    buffer: ReplayBuffer,
    num_games: int = 64,
    augment: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Mines 1-ply win, 1-ply forced-block, and tactical states across all 76 winning lines."""
    generator = rng if rng is not None else np.random.default_rng()
    samples_added = 0

    # Phase A: Direct 76-line x 4-cell targeted win & forced-block drills across all heights/diagonals
    num_line_drills = max(1, num_games)
    for drill_idx in range(num_line_drills):
        line_idx = int((drill_idx * 7 + int(generator.integers(0, 76))) % len(WIN_LINE_MASKS))
        mask = WIN_LINE_MASKS[line_idx]
        bits = [b for b in range(64) if (mask & (1 << b)) != 0]
        missing_b = int(bits[int(generator.integers(0, 4))])
        col_missing = missing_b % 16
        z_missing = missing_b // 16
        three_bits = [b for b in bits if b != missing_b]

        # Skip vertical line case where missing_b is below another bit on the same line
        if any((b % 16) == col_missing and (b // 16) > z_missing for b in three_bits):
            continue

        my_bb = 0
        opp_bb = 0
        for b in three_bits:
            opp_bb |= 1 << b

        # Fill gravity support columns underneath all 4 cells of the line
        for b in bits:
            col = b % 16
            zb = b // 16
            for z_u in range(zb):
                bit_u = col + 16 * z_u
                if ((my_bb | opp_bb) & (1 << bit_u)) == 0:
                    if float(generator.random()) < 0.5:
                        my_bb |= 1 << bit_u
                    else:
                        opp_bb |= 1 << bit_u

        # Add 2..8 random distractor balls in other columns to diversify board contexts
        num_distractors = int(generator.integers(2, 9))
        for _ in range(num_distractors):
            d_col = int(generator.integers(0, 16))
            if d_col == col_missing:
                continue
            lbit = landing_bit_for_action(my_bb | opp_bb, d_col)
            if lbit is not None and (mask & (1 << lbit)) == 0:
                if float(generator.random()) < 0.5:
                    if not check_bitboard_win(my_bb | (1 << lbit)):
                        my_bb |= 1 << lbit
                else:
                    if not check_bitboard_win(opp_bb | (1 << lbit)):
                        opp_bb |= 1 << lbit

        occ = my_bb | opp_bb
        if check_bitboard_win(my_bb) or check_bitboard_win(opp_bb):
            continue

        # 1. Forced 4th-ball block drill (opponent has 3-in-a-row on `mask`)
        obs_blk, vmask_blk = _build_canonical_from_bitboards(my_bb, opp_bb)
        if vmask_blk != 0:
            pi_blk, val_blk = _compute_tactical_teacher_policy(my_bb, opp_bb, vmask_blk)
            samples_added += buffer.add_sample(
                state=obs_blk,
                policy=pi_blk,
                value=val_blk,
                valid_mask=vmask_blk,
                augment=augment,
                rectify_tactical=True,
            )

        # 2. Immediate 4th-ball win drill (active player has 3-in-a-row on `mask`)
        obs_win, vmask_win = _build_canonical_from_bitboards(opp_bb, my_bb)
        if vmask_win != 0:
            pi_win, val_win = _compute_tactical_teacher_policy(opp_bb, my_bb, vmask_win)
            samples_added += buffer.add_sample(
                state=obs_win,
                policy=pi_win,
                value=val_win,
                valid_mask=vmask_win,
                augment=augment,
                rectify_tactical=True,
            )

    # Phase B: Mixed tactical + heuristic game rollouts
    for _ in range(num_games):
        white_bb = 0
        black_bb = 0
        current_player = PLAYER_WHITE

        for _ in range(64):
            occupied = white_bb | black_bb
            my_bb = white_bb if current_player == PLAYER_WHITE else black_bb
            opp_bb = black_bb if current_player == PLAYER_WHITE else white_bb
            obs, valid_mask = _build_canonical_from_bitboards(my_bb, opp_bb)
            if valid_mask == 0:
                break

            teacher_pi, teacher_val = _compute_tactical_teacher_policy(my_bb, opp_bb, valid_mask)
            win_mask, blunder_mask, forced_loss = compute_tactical_action_masks(obs, valid_mask)
            if win_mask != 0 or blunder_mask != 0 or forced_loss or float(generator.random()) < 0.35:
                samples_added += buffer.add_sample(
                    state=obs,
                    policy=teacher_pi,
                    value=teacher_val,
                    valid_mask=valid_mask,
                    augment=augment,
                    rectify_tactical=True,
                )

            legal = [a for a in range(NUM_ACTIONS) if (valid_mask & (1 << a)) != 0]
            safe_actions = [a for a in legal if (blunder_mask & (1 << a)) == 0]
            if win_mask != 0 and float(generator.random()) < 0.5:
                win_actions = [a for a in legal if (win_mask & (1 << a)) != 0]
                act = int(generator.choice(win_actions))
            elif len(safe_actions) > 0 and float(generator.random()) < 0.75:
                probs = np.array([max(float(teacher_pi[a]), 1e-4) for a in safe_actions], dtype=np.float64)
                probs /= probs.sum()
                act = int(generator.choice(safe_actions, p=probs))
            else:
                act = int(generator.choice(legal))

            lbit = landing_bit_for_action(occupied, act)
            if lbit is None:
                break
            if current_player == PLAYER_WHITE:
                white_bb |= 1 << lbit
                if check_bitboard_win(white_bb):
                    break
                current_player = PLAYER_BLACK
            else:
                black_bb |= 1 << lbit
                if check_bitboard_win(black_bb):
                    break
                current_player = PLAYER_WHITE

    return samples_added


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

    # Per-environment active episode trajectory: list of (obs, probs, player, valid_mask, action)
    active_trajectories: List[List[Tuple[np.ndarray, np.ndarray, int, int, int]]] = [
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
                (obs_batch[i], probs, int(current_players[i]), int(valid_masks[i]), int(act))
            )

        step_out = client.batch_step(actions)
        dones = step_out["dones"]
        winners = step_out["winners"]

        for i in range(num_envs):
            if dones[i]:
                episodes_completed += 1
                winner = int(winners[i])
                for s_obs, _, s_player, s_mask, s_act in active_trajectories[i]:
                    my_bb, opp_bb = _canonical_tensor_to_bitboards(s_obs)
                    teacher_pi, teacher_val = _compute_tactical_teacher_policy(my_bb, opp_bb, s_mask)
                    _, step_blunder_mask, _ = compute_tactical_action_masks(s_obs, s_mask)
                    if (step_blunder_mask & (1 << s_act)) != 0:
                        # Heavily penalize any step that chose a 1-ply losing move
                        z = -1.0
                    elif winner == 0:
                        z = 0.5 * teacher_val
                    elif winner == s_player:
                        z = 0.5 * 1.0 + 0.5 * teacher_val
                    else:
                        z = 0.5 * (-1.0) + 0.5 * teacher_val
                    samples_added += buffer.add_sample(
                        s_obs, teacher_pi, z, valid_mask=s_mask, augment=augment
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
    use_tactical_lookahead: bool = True,
) -> Dict[str, Any]:
    """Benchmarks `model` (wrapped in `NeuralArenaBot`) against a Go baseline agent over `num_games`."""
    if num_games <= 0:
        raise ValueError("num_games must be positive")
    wins = 0
    draws = 0
    losses = 0
    bot = NeuralArenaBot(
        model=model,
        temperature=0.0,
        use_tactical_lookahead=use_tactical_lookahead,
    )

    for g_idx in range(num_games):
        game_id = f"eval_{opponent}_{g_idx}"
        nn_player = PLAYER_WHITE if (g_idx % 2 == 0) else PLAYER_BLACK
        state = client.reset(game_id=game_id)
        done = False

        while not done:
            if int(state["current_player"]) == nn_player:
                decision = bot.choose_move(state)
                action = int(decision["action"])
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
