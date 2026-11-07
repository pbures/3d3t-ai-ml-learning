"""Remote HTTPS Arena Bot Server, Tactical+Neural Decision Engine, and Battle Client (Phase 5)."""

from __future__ import annotations

import json
import ssl
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from .network import NUM_ACTIONS, TTT3DNet

PLAYER_WHITE = 1
PLAYER_BLACK = 2


def _bit_index(x: int, y: int, z: int) -> int:
    return 16 * z + 4 * y + x


def _precompute_76_win_masks() -> Tuple[int, ...]:
    """Precomputes all 76 64-bit winning line bitmasks on the 4x4x4 cube."""
    masks: List[int] = []

    def add_line(coords: List[Tuple[int, int, int]]) -> None:
        m = 0
        for x, y, z in coords:
            m |= 1 << _bit_index(x, y, z)
        masks.append(m)

    # 1. Orthogonal lines along X, Y, Z (16 each = 48)
    for z in range(4):
        for y in range(4):
            add_line([(x, y, z) for x in range(4)])
    for z in range(4):
        for x in range(4):
            add_line([(x, y, z) for y in range(4)])
    for y in range(4):
        for x in range(4):
            add_line([(x, y, z) for z in range(4)])

    # 2. 2D Planar diagonals in XY, XZ, YZ planes (8 each = 24)
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

    return tuple(masks)


WIN_LINE_MASKS: Tuple[int, ...] = _precompute_76_win_masks()


def check_bitboard_win(bitboard: int) -> bool:
    """Returns True if `bitboard` contains any of the 76 4-in-a-row masks."""
    for mask in WIN_LINE_MASKS:
        if (bitboard & mask) == mask:
            return True
    return False


def landing_bit_for_action(occupied_mask: int, action: int) -> Optional[int]:
    """Returns the 0..63 bit index where a ball dropped in `action` (0..15) lands, or None if full."""
    for z in range(4):
        bit = action + 16 * z
        if (occupied_mask & (1 << bit)) == 0:
            return bit
    return None


def find_immediate_wins(player_bb: int, occupied_mask: int) -> List[int]:
    """Returns all legal column actions (0..15) that immediately form a 4-in-a-row for `player_bb`."""
    wins: List[int] = []
    for a in range(NUM_ACTIONS):
        lbit = landing_bit_for_action(occupied_mask, a)
        if lbit is not None and check_bitboard_win(player_bb | (1 << lbit)):
            wins.append(a)
    return wins


def _build_canonical_from_bitboards(my_bb: int, opp_bb: int) -> Tuple[np.ndarray, int]:
    occupied = my_bb | opp_bb
    valid_mask = 0
    flat = np.zeros(192, dtype=np.float32)
    for bit in range(64):
        if (my_bb & (1 << bit)) != 0:
            flat[bit] = 1.0
        if (opp_bb & (1 << bit)) != 0:
            flat[64 + bit] = 1.0
    for a in range(NUM_ACTIONS):
        lbit = landing_bit_for_action(occupied, a)
        if lbit is not None:
            valid_mask |= 1 << a
            flat[128 + lbit] = 1.0
    return flat.reshape(3, 4, 4, 4), valid_mask


def state_dto_to_canonical_tensor(state_dto: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """Extracts the (3, 4, 4, 4) float32 canonical tensor and 16-bit valid_moves_mask from GameStateDTO."""
    white_bb = int(state_dto.get("white_bitboard", 0))
    black_bb = int(state_dto.get("black_bitboard", 0))
    current_player = int(state_dto.get("current_player", PLAYER_WHITE))
    occupied = white_bb | black_bb

    valid_mask = state_dto.get("valid_moves_mask")
    if valid_mask is None:
        computed_mask = 0
        for a in range(NUM_ACTIONS):
            if landing_bit_for_action(occupied, a) is not None:
                computed_mask |= 1 << a
        valid_mask = computed_mask
    else:
        valid_mask = int(valid_mask)

    raw_tensor = state_dto.get("canonical_tensor")
    if isinstance(raw_tensor, list) and len(raw_tensor) == 192:
        arr = np.asarray(raw_tensor, dtype=np.float32).reshape(3, 4, 4, 4)
        return arr, valid_mask

    # Fallback reconstruction from 64-bit bitboards
    my_bb = white_bb if current_player == PLAYER_WHITE else black_bb
    opp_bb = black_bb if current_player == PLAYER_WHITE else white_bb
    flat = np.zeros(192, dtype=np.float32)
    for bit in range(64):
        if (my_bb & (1 << bit)) != 0:
            flat[bit] = 1.0
        if (opp_bb & (1 << bit)) != 0:
            flat[64 + bit] = 1.0
    for a in range(NUM_ACTIONS):
        lbit = landing_bit_for_action(occupied, a)
        if lbit is not None:
            flat[128 + lbit] = 1.0
    return flat.reshape(3, 4, 4, 4), valid_mask


class NeuralArenaBot:
    """Wraps a trained PyTorch `TTT3DNet` with an optional 1-ply tactical win/block shield."""

    def __init__(
        self,
        model: TTT3DNet,
        temperature: float = 0.0,
        use_tactical_lookahead: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.model = model
        self.temperature = float(temperature)
        self.use_tactical_lookahead = bool(use_tactical_lookahead)
        self.rng = rng if rng is not None else np.random.default_rng()

    def choose_move(self, state_dto: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates `GameStateDTO` and returns {"action": int, "value": float, "policy": list, "reason": str}."""
        obs, valid_mask = state_dto_to_canonical_tensor(state_dto)
        if valid_mask == 0:
            raise ValueError("No legal moves available in state_dto")

        action, probs, value = self.model.select_action(
            obs, valid_mask=valid_mask, temperature=self.temperature, rng=self.rng
        )
        legal_actions = [a for a in range(NUM_ACTIONS) if (valid_mask & (1 << a)) != 0]

        if not self.use_tactical_lookahead:
            return {
                "action": int(action),
                "value": float(value),
                "policy": [float(p) for p in probs],
                "reason": "neural_policy",
            }

        white_bb = int(state_dto.get("white_bitboard", 0))
        black_bb = int(state_dto.get("black_bitboard", 0))
        current_player = int(state_dto.get("current_player", PLAYER_WHITE))
        my_bb = white_bb if current_player == PLAYER_WHITE else black_bb
        opp_bb = black_bb if current_player == PLAYER_WHITE else white_bb
        occupied = white_bb | black_bb

        # 1. Check for immediate 1-ply winning moves for active player
        winning_moves = find_immediate_wins(my_bb, occupied)
        if winning_moves:
            best_win = max(winning_moves, key=lambda a: float(probs[a]))
            return {
                "action": int(best_win),
                "value": 1.0,
                "policy": [float(p) for p in probs],
                "reason": "immediate_win",
            }

        # 2. Check for forced 1-ply blocks if opponent threatens an immediate win on their next turn
        blocking_moves = find_immediate_wins(opp_bb, occupied)
        if blocking_moves:
            # Prefer a block that does not immediately give the opponent a win right above it
            safe_blocks = [
                a
                for a in blocking_moves
                if not find_immediate_wins(opp_bb, occupied | (1 << (landing_bit_for_action(occupied, a) or 0)))
            ]
            pool = safe_blocks if safe_blocks else blocking_moves
            best_block = max(pool, key=lambda a: float(probs[a]))
            return {
                "action": int(best_block),
                "value": float(value),
                "policy": [float(p) for p in probs],
                "reason": "forced_block",
            }

        # 3. Filter out blunders that hand the opponent an immediate 1-ply win on their reply
        safe_moves: List[int] = []
        for a in legal_actions:
            lbit = landing_bit_for_action(occupied, a)
            if lbit is None:
                continue
            next_occupied = occupied | (1 << lbit)
            if find_immediate_wins(opp_bb, next_occupied):
                continue
            safe_moves.append(a)

        candidates = safe_moves if safe_moves else legal_actions

        # 4. Check if any safe move creates an unstoppable double-threat fork (2+ winning moves next turn)
        fork_moves: List[int] = []
        for a in candidates:
            lbit = landing_bit_for_action(occupied, a)
            if lbit is None:
                continue
            next_my_bb = my_bb | (1 << lbit)
            next_occ = occupied | (1 << lbit)
            if len(find_immediate_wins(next_my_bb, next_occ)) >= 2:
                fork_moves.append(a)
        if fork_moves:
            best_fork = max(fork_moves, key=lambda a: float(probs[a]))
            return {
                "action": int(best_fork),
                "value": float(value),
                "policy": [float(p) for p in probs],
                "reason": "offensive_fork",
            }

        # 5. Filter out moves that allow the opponent to play a reply creating an unstoppable double-threat fork
        non_fork_vulnerable: List[int] = []
        for a in candidates:
            lbit = landing_bit_for_action(occupied, a)
            if lbit is None:
                continue
            next_my_bb = my_bb | (1 << lbit)
            next_occ = occupied | (1 << lbit)
            my_threats = find_immediate_wins(next_my_bb, next_occ)

            opp_can_fork = False
            # If we threaten a 1-ply win, opponent is forced to block it
            opp_replies = my_threats if my_threats else [
                r for r in range(NUM_ACTIONS) if landing_bit_for_action(next_occ, r) is not None
            ]
            for r in opp_replies:
                r_bit = landing_bit_for_action(next_occ, r)
                if r_bit is None:
                    continue
                opp_after = opp_bb | (1 << r_bit)
                occ_after = next_occ | (1 << r_bit)
                if check_bitboard_win(opp_after):
                    opp_can_fork = True
                    break
                if len(find_immediate_wins(opp_after, occ_after)) >= 2 and not find_immediate_wins(next_my_bb, occ_after):
                    opp_can_fork = True
                    break
            if not opp_can_fork:
                non_fork_vulnerable.append(a)

        if non_fork_vulnerable:
            candidates = non_fork_vulnerable

        # 6. Batched 1-ply Neural Value + Policy evaluation over candidate moves
        # Combines TTT3DNet policy prior pi(a|s) with 1-ply value lookahead -v(s') in a single batched forward pass
        next_obs_list: List[np.ndarray] = []
        threat_bonuses: List[float] = []
        for a in candidates:
            lbit = landing_bit_for_action(occupied, a)
            assert lbit is not None
            next_my_bb = my_bb | (1 << lbit)
            next_obs, _ = _build_canonical_from_bitboards(opp_bb, next_my_bb)
            next_obs_list.append(next_obs)
            threat_bonuses.append(0.15 * len(find_immediate_wins(next_my_bb, occupied | (1 << lbit))))

        was_training = self.model.training
        self.model.eval()
        try:
            dev = next(self.model.parameters()).device
            batch_t = torch.from_numpy(np.stack(next_obs_list, axis=0)).to(dev)
            with torch.no_grad():
                _, opp_vals_t = self.model(batch_t)
            opp_vals = opp_vals_t.squeeze(-1).cpu().numpy()
        finally:
            if was_training:
                self.model.train()

        best_candidate = candidates[0]
        best_combined_score = -1e9
        for idx, a in enumerate(candidates):
            my_next_val = -float(opp_vals[idx])
            combined = float(probs[a]) + 0.5 * my_next_val + threat_bonuses[idx]
            if combined > best_combined_score:
                best_combined_score = combined
                best_candidate = a

        reason = "neural_safe_policy" if (safe_moves and action not in candidates) else "neural_policy"
        return {
            "action": int(best_candidate),
            "value": float(value),
            "policy": [float(p) for p in probs],
            "reason": reason,
        }


class BotWebhookServer:
    """Multithreaded HTTP webhook microservice exposing POST /move for Go Arena matchmaking."""

    def __init__(self, bot: NeuralArenaBot, host: str = "127.0.0.1", port: int = 0) -> None:
        self.bot = bot
        self.host = host
        self._server = ThreadingHTTPServer((host, port), self._make_handler())
        self.port = int(self._server.server_address[1])
        self._thread: Optional[threading.Thread] = None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        bot_ref = self.bot

        class WebhookHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass  # Keep test output clean

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self._send_json(200, {"status": "ok"})
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/move":
                    self._send_json(404, {"error": "expected POST /move"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b""
                try:
                    state_dto = json.loads(raw.decode("utf-8"))
                    decision = bot_ref.choose_move(state_dto)
                    self._send_json(200, {"action": decision["action"]})
                except Exception as exc:  # noqa: BLE001
                    self._send_json(400, {"error": str(exc)})

            def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return WebhookHandler

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def move_url(self) -> str:
        return f"{self.base_url}/move"

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def __enter__(self) -> "BotWebhookServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class ArenaHTTPError(RuntimeError):
    """Raised when the Go HTTPS Arena Server returns a non-2xx HTTP status code."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = int(status_code)
        self.message = message


class ArenaBattleClient:
    """HTTPS client for interacting with the Go 3T3D Arena Server & 3D Replay Studio."""

    def __init__(self, base_url: str, verify_tls: bool = False, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        if verify_tls:
            self._ssl_ctx = ssl.create_default_context()
        else:
            self._ssl_ctx = ssl._create_unverified_context()  # noqa: SLF001

    def _request(
        self,
        method: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        data: Optional[bytes] = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(
                req,
                timeout=self.timeout,
                context=self._ssl_ctx if url.startswith("https://") else None,
            ) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            raise ArenaHTTPError(exc.code, err_body) from exc

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/api/v1/health")

    def get_win_lines(self) -> List[Dict[str, Any]]:
        resp = self._request("GET", "/api/v1/winlines")
        return list(resp.get("lines") or [])

    def list_games(self) -> List[Dict[str, Any]]:
        resp = self._request("GET", "/api/v1/games")
        return list(resp.get("games") or [])

    def create_game(
        self,
        game_id: str = "",
        white_name: str = "Player 1 (White)",
        black_name: str = "Player 2 (Black)",
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/games",
            {"game_id": game_id, "white_name": white_name, "black_name": black_name},
        )

    def get_game(self, game_id: str = "default") -> Dict[str, Any]:
        return self._request("GET", f"/api/v1/games/{game_id}")

    def reset_game(self, game_id: str = "default") -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/games/{game_id}/reset", {})

    def submit_move(
        self,
        game_id: str,
        action: Optional[int] = None,
        player: int = 0,
        use_coords: bool = False,
        x: int = 0,
        y: int = 0,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"use_coords": bool(use_coords)}
        if player != 0:
            body["player"] = int(player)
        if use_coords:
            body["x"] = int(x)
            body["y"] = int(y)
        elif action is not None:
            body["action"] = int(action)
        return self._request("POST", f"/api/v1/games/{game_id}/move", body)

    def trigger_ai_move(self, game_id: str, agent: str = "heuristic") -> Dict[str, Any]:
        return self._request("POST", f"/api/v1/games/{game_id}/ai-move", {"agent": agent})

    def apply_preset(
        self,
        game_id: str,
        preset_type: str,
        line_index: int = 0,
        winner: int = 1,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            f"/api/v1/games/{game_id}/preset",
            {"type": preset_type, "line_index": int(line_index), "winner": int(winner)},
        )

    def run_webhook_match(
        self,
        white_bot_url: str,
        black_bot_url: str,
        game_id: str = "",
        white_name: str = "White Bot",
        black_name: str = "Black Bot",
        timeout_ms: int = 5000,
    ) -> Dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/matches/run",
            {
                "game_id": game_id,
                "white_name": white_name,
                "black_name": black_name,
                "white_bot_url": white_bot_url,
                "black_bot_url": black_bot_url,
                "timeout_ms": int(timeout_ms),
            },
        )

    def play_client_match(
        self,
        bot: NeuralArenaBot,
        opponent_agent: str = "heuristic",
        game_id: str = "",
        bot_plays_white: bool = True,
        bot_name: str = "TTT3DNet",
    ) -> Dict[str, Any]:
        """Plays a full turn-by-turn match against a server-side agent over HTTPS."""
        opp_label = f"Go {opponent_agent.capitalize()}"
        w_name = f"{bot_name} (White)" if bot_plays_white else f"{opp_label} (White)"
        b_name = f"{opp_label} (Black)" if bot_plays_white else f"{bot_name} (Black)"
        nn_player = PLAYER_WHITE if bot_plays_white else PLAYER_BLACK

        state = self.create_game(game_id=game_id, white_name=w_name, black_name=b_name)
        actual_gid = str(state["game_id"])

        while str(state.get("status", "ONGOING")) == "ONGOING":
            cur_player = int(state["current_player"])
            if cur_player == nn_player:
                decision = bot.choose_move(state)
                move_resp = self.submit_move(
                    game_id=actual_gid,
                    action=decision["action"],
                    player=nn_player,
                )
            else:
                move_resp = self.trigger_ai_move(game_id=actual_gid, agent=opponent_agent)
            state = move_resp["state"]

        return state

    def run_tournament(
        self,
        bot: NeuralArenaBot,
        opponent_agent: str = "heuristic",
        num_games: int = 4,
        mode: str = "client",
        webhook_move_url: Optional[str] = None,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Runs a paired multi-game match alternating White and Black against `opponent_agent`."""
        if num_games <= 0:
            raise ValueError("num_games must be positive")
        wins = 0
        draws = 0
        losses = 0
        game_ids: List[str] = []

        server_bot_url = f"{self.base_url}/api/v1/bots/{opponent_agent}/move"

        for idx in range(num_games):
            bot_white = idx % 2 == 0
            nn_player = PLAYER_WHITE if bot_white else PLAYER_BLACK
            if bot_white:
                matchup = f"TTT3DNet-white-vs-{opponent_agent}-black-{idx + 1}"
            else:
                matchup = f"{opponent_agent}-white-vs-TTT3DNet-black-{idx + 1}"
            gid = f"{prefix}-{matchup}" if prefix else matchup

            if mode == "webhook":
                if not webhook_move_url:
                    raise ValueError("webhook_move_url is required when mode='webhook'")
                w_url = webhook_move_url if bot_white else server_bot_url
                b_url = server_bot_url if bot_white else webhook_move_url
                w_name = "TTT3DNet (White)" if bot_white else f"Go {opponent_agent.capitalize()} (White)"
                b_name = f"Go {opponent_agent.capitalize()} (Black)" if bot_white else "TTT3DNet (Black)"
                final_state = self.run_webhook_match(
                    white_bot_url=w_url,
                    black_bot_url=b_url,
                    game_id=gid,
                    white_name=w_name,
                    black_name=b_name,
                )
            else:
                final_state = self.play_client_match(
                    bot=bot,
                    opponent_agent=opponent_agent,
                    game_id=gid,
                    bot_plays_white=bot_white,
                )

            game_ids.append(str(final_state["game_id"]))
            winner = int(final_state.get("winner", 0))
            if winner == nn_player:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1

        return {
            "opponent": opponent_agent,
            "mode": mode,
            "num_games": num_games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": (wins + 0.5 * draws) / float(num_games),
            "game_ids": game_ids,
        }
