"""Vectorized and Single-Game gRPC Client for the Go 3T3D Environment Server."""

from __future__ import annotations

import base64
import json
from typing import Any, Dict, List, Optional, Sequence, Union

import grpc
import numpy as np

CANONICAL_TENSOR_BYTES = 192
SERVICE_PREFIX = "/env3t3d.v1.EnvService"


def _serialize_json(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _deserialize_json(raw: bytes) -> Dict[str, Any]:
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _decode_bytes_field(val: Union[str, bytes, None]) -> bytes:
    if val is None:
        return b""
    if isinstance(val, bytes):
        return val
    if isinstance(val, str):
        return base64.b64decode(val) if val else b""
    raise TypeError(f"Unsupported byte field type: {type(val)}")


def unpack_observation(raw_bytes: Union[bytes, str]) -> np.ndarray:
    """Unpacks a 192-byte canonical tensor into a (3, 4, 4, 4) float32 NumPy array."""
    decoded = _decode_bytes_field(raw_bytes)
    if len(decoded) != CANONICAL_TENSOR_BYTES:
        raise ValueError(
            f"Expected {CANONICAL_TENSOR_BYTES} bytes for single observation, got {len(decoded)}"
        )
    arr = np.frombuffer(decoded, dtype=np.uint8).astype(np.float32)
    return arr.reshape(3, 4, 4, 4).copy()


def unpack_batch_observations(raw_bytes: Union[bytes, str], num_envs: int) -> np.ndarray:
    """Unpacks N * 192 bytes into an (N, 3, 4, 4, 4) float32 NumPy array."""
    decoded = _decode_bytes_field(raw_bytes)
    expected = int(num_envs) * CANONICAL_TENSOR_BYTES
    if len(decoded) != expected:
        raise ValueError(
            f"Expected {expected} bytes for {num_envs} environments, got {len(decoded)}"
        )
    if num_envs == 0:
        return np.zeros((0, 3, 4, 4, 4), dtype=np.float32)
    arr = np.frombuffer(decoded, dtype=np.uint8).astype(np.float32)
    return arr.reshape(int(num_envs), 3, 4, 4, 4).copy()


def _format_game_state(msg: Dict[str, Any]) -> Dict[str, Any]:
    history_raw = msg.get("history") or []
    return {
        "game_id": str(msg.get("game_id", "")),
        "white_bitboard": int(msg.get("white_bitboard", 0)),
        "black_bitboard": int(msg.get("black_bitboard", 0)),
        "current_player": int(msg.get("current_player", 0)),
        "status": int(msg.get("status", 0)),
        "winner": int(msg.get("winner", 0)),
        "win_line_mask": int(msg.get("win_line_mask", 0)),
        "win_line_index": int(msg.get("win_line_index", -1)),
        "valid_moves_mask": int(msg.get("valid_moves_mask", 0)),
        "move_count": int(msg.get("move_count", 0)),
        "obs": unpack_observation(msg.get("canonical_tensor", b"")),
        "history": [
            {
                "action": int(m.get("action", 0)),
                "x": int(m.get("x", 0)),
                "y": int(m.get("y", 0)),
                "z": int(m.get("z", 0)),
                "player": int(m.get("player", 0)),
                "move_number": int(m.get("move_number", 0)),
            }
            for m in history_raw
        ],
    }


class VectorizedEnvClient:
    """High-throughput gRPC client connecting PyTorch to the Go 3T3D Bitboard Engine."""

    def __init__(self, target: str = "localhost:50051") -> None:
        self.target = target
        self.channel = grpc.insecure_channel(target)
        self._reset_rpc = self.channel.unary_unary(
            f"{SERVICE_PREFIX}/Reset",
            request_serializer=_serialize_json,
            response_deserializer=_deserialize_json,
        )
        self._step_rpc = self.channel.unary_unary(
            f"{SERVICE_PREFIX}/Step",
            request_serializer=_serialize_json,
            response_deserializer=_deserialize_json,
        )
        self._get_state_rpc = self.channel.unary_unary(
            f"{SERVICE_PREFIX}/GetState",
            request_serializer=_serialize_json,
            response_deserializer=_deserialize_json,
        )
        self._get_win_lines_rpc = self.channel.unary_unary(
            f"{SERVICE_PREFIX}/GetWinLines",
            request_serializer=_serialize_json,
            response_deserializer=_deserialize_json,
        )
        self._init_pool_rpc = self.channel.unary_unary(
            f"{SERVICE_PREFIX}/InitPool",
            request_serializer=_serialize_json,
            response_deserializer=_deserialize_json,
        )
        self._reset_pool_rpc = self.channel.unary_unary(
            f"{SERVICE_PREFIX}/ResetPool",
            request_serializer=_serialize_json,
            response_deserializer=_deserialize_json,
        )
        self._batch_step_rpc = self.channel.unary_unary(
            f"{SERVICE_PREFIX}/BatchStep",
            request_serializer=_serialize_json,
            response_deserializer=_deserialize_json,
        )
        self._select_action_rpc = self.channel.unary_unary(
            f"{SERVICE_PREFIX}/SelectAction",
            request_serializer=_serialize_json,
            response_deserializer=_deserialize_json,
        )

    def close(self) -> None:
        self.channel.close()

    def __enter__(self) -> "VectorizedEnvClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def reset(self, game_id: str = "default") -> Dict[str, Any]:
        resp = self._reset_rpc({"game_id": game_id})
        return _format_game_state(resp)

    def get_state(self, game_id: str = "default") -> Dict[str, Any]:
        resp = self._get_state_rpc({"game_id": game_id})
        return _format_game_state(resp)

    def step(self, action: int, game_id: str = "default") -> Dict[str, Any]:
        resp = self._step_rpc({"game_id": game_id, "action": int(action)})
        last_move = resp.get("last_move") or {}
        return {
            "state": _format_game_state(resp.get("state") or {}),
            "last_move": {
                "action": int(last_move.get("action", 0)),
                "x": int(last_move.get("x", 0)),
                "y": int(last_move.get("y", 0)),
                "z": int(last_move.get("z", 0)),
                "player": int(last_move.get("player", 0)),
                "move_number": int(last_move.get("move_number", 0)),
            },
            "reward": float(resp.get("reward", 0.0)),
            "done": bool(resp.get("done", False)),
        }

    def get_win_lines(self) -> List[Dict[str, Any]]:
        resp = self._get_win_lines_rpc({})
        lines: List[Dict[str, Any]] = []
        for line in resp.get("lines") or []:
            lines.append(
                {
                    "index": int(line.get("index", 0)),
                    "category": str(line.get("category", "")),
                    "mask": int(line.get("mask", 0)),
                    "coords": [
                        {
                            "action": int(c.get("action", 0)),
                            "x": int(c.get("x", 0)),
                            "y": int(c.get("y", 0)),
                            "z": int(c.get("z", 0)),
                        }
                        for c in (line.get("coords") or [])
                    ],
                }
            )
        return lines

    def select_action(
        self,
        agent: str = "mcts",
        game_id: str = "default",
        simulations: int = 0,
    ) -> Dict[str, Any]:
        resp = self._select_action_rpc(
            {
                "game_id": game_id,
                "agent": agent,
                "simulations": int(simulations),
            }
        )
        return {
            "action": int(resp.get("action", 0)),
            "agent": str(resp.get("agent", "")),
            "policy": np.asarray(resp.get("policy") or [], dtype=np.float32),
            "value": float(resp.get("value", 0.0)),
        }

    def init_pool(self, num_envs: int) -> Dict[str, Any]:
        resp = self._init_pool_rpc({"num_envs": int(num_envs)})
        n = int(resp.get("num_envs", 0))
        return {
            "num_envs": n,
            "obs": unpack_batch_observations(resp.get("observations", b""), n),
            "valid_masks": np.asarray(resp.get("valid_masks") or [], dtype=np.int64),
            "current_players": np.asarray(resp.get("current_players") or [], dtype=np.int64),
        }

    def reset_pool(self, env_ids: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        ids = [int(i) for i in env_ids] if env_ids is not None else []
        resp = self._reset_pool_rpc({"env_ids": ids})
        n = int(resp.get("num_envs", 0))
        return {
            "num_envs": n,
            "obs": unpack_batch_observations(resp.get("observations", b""), n),
            "valid_masks": np.asarray(resp.get("valid_masks") or [], dtype=np.int64),
            "current_players": np.asarray(resp.get("current_players") or [], dtype=np.int64),
        }

    def batch_step(self, actions: Union[Sequence[int], np.ndarray, bytes]) -> Dict[str, Any]:
        if isinstance(actions, (bytes, bytearray)):
            action_bytes = bytes(actions)
        else:
            arr = np.asarray(actions, dtype=np.uint8)
            action_bytes = arr.tobytes()

        b64_actions = base64.b64encode(action_bytes).decode("ascii")
        resp = self._batch_step_rpc({"actions": b64_actions})
        n = int(resp.get("num_envs", 0))
        return {
            "num_envs": n,
            "obs": unpack_batch_observations(resp.get("observations", b""), n),
            "terminal_obs": unpack_batch_observations(resp.get("terminal_observations", b""), n),
            "rewards": np.asarray(resp.get("rewards") or [], dtype=np.float32),
            "dones": np.asarray(resp.get("dones") or [], dtype=bool),
            "winners": np.asarray(resp.get("winners") or [], dtype=np.int64),
            "valid_masks": np.asarray(resp.get("valid_masks") or [], dtype=np.int64),
            "current_players": np.asarray(resp.get("current_players") or [], dtype=np.int64),
            "win_line_masks": np.asarray(resp.get("win_line_masks") or [], dtype=np.uint64),
        }
