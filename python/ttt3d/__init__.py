"""3T3D Neural Network, RL Pipeline & Remote HTTPS Arena Bot Package (Phases 4 & 5)."""

from .arena_bot import (
    WIN_LINE_MASKS,
    ArenaBattleClient,
    ArenaHTTPError,
    BotWebhookServer,
    NeuralArenaBot,
    check_bitboard_win,
    landing_bit_for_action,
    state_dto_to_canonical_tensor,
)
from .grpc_client import VectorizedEnvClient, unpack_batch_observations, unpack_observation
from .network import ResBlock3D, TTT3DNet, mask_illegal_logits, valid_mask_to_tensor
from .symmetry import (
    NUM_SYMMETRIES,
    augment_sample,
    transform_action,
    transform_policy,
    transform_state,
    transform_valid_mask,
)
from .trainer import (
    ReplayBuffer,
    TrainingSample,
    compute_policy_value_loss,
    evaluate_against_baseline,
    generate_self_play_episodes,
    generate_vectorized_rollouts,
    train_epoch,
)

__all__ = [
    "WIN_LINE_MASKS",
    "ArenaBattleClient",
    "ArenaHTTPError",
    "BotWebhookServer",
    "NeuralArenaBot",
    "check_bitboard_win",
    "landing_bit_for_action",
    "state_dto_to_canonical_tensor",
    "VectorizedEnvClient",
    "unpack_batch_observations",
    "unpack_observation",
    "ResBlock3D",
    "TTT3DNet",
    "mask_illegal_logits",
    "valid_mask_to_tensor",
    "NUM_SYMMETRIES",
    "augment_sample",
    "transform_action",
    "transform_policy",
    "transform_state",
    "transform_valid_mask",
    "ReplayBuffer",
    "TrainingSample",
    "compute_policy_value_loss",
    "evaluate_against_baseline",
    "generate_self_play_episodes",
    "generate_vectorized_rollouts",
    "train_epoch",
]
