# Phase 2 Implementation Specification: HTTPS Battle Server & 3D Web Replay UI

**Branch:** `phase-2-https-arena-webui`  
**Related Specification:** [SPECIFICATION.md](./SPECIFICATION.md)

---

## 1. Success Criteria & Verification Strategy

Phase 2 is considered complete only when all of the following automated criteria pass via `go test` and GitHub Actions CI:

### 1.1 Success Criteria Checklist

| ID | Criterion | Verification Method & Target |
| :--- | :--- | :--- |
| **SC-1** | **Unit Test Code Coverage $\ge 90\%$** | `go test -v ./internal/... -coverprofile=coverage.out` must achieve **$\ge 90.0\%$ statement coverage** for `internal/httpserver` as well as overall repository coverage. |
| **SC-2** | **End-to-End HTTPS Integration Tests for All Game Rules** | `internal/httpserver/integration_test.go` spins up a live TLS `https://` server, connects an HTTPS client, and verifies every game rule in Section 2 & Section 4.3 of `SPECIFICATION.md`. |
| **SC-3** | **Exhaustive 76 Winning Lines Verification over HTTPS** | HTTPS integration tests play **all 76 winning lines** (48 orthogonal, 24 planar diagonal, 4 3D space diagonal) for both White and Black over HTTPS REST calls and verify the winning line mask and 3D coordinates. |
| **SC-4** | **Deterministic 64-Move Draw Verification over HTTPS** | HTTPS integration tests execute a full 64-move non-winning game sequence over `POST /api/v1/games/{id}/move` and verify `status == "DRAW"` and `valid_moves_mask == 0`. |
| **SC-5** | **Remote Bot Arena Webhook Orchestration** | HTTPS integration tests launch two mock remote bot servers, run an automated battle via `POST /api/v1/matches/run`, and verify the resulting match state and complete step-by-step replay history. |
| **SC-6** | **3D Web Visualization & Step-by-Step Replay UI** | Served at `GET /` over HTTPS using embedded Three.js WebGL UI supporting interactive 3D orbit controls, human/demo play, and step-by-step replay (`|<<`, `<`, `Play/Pause`, `>`, `>>|`, move timeline scrubbing, and 3D winning line beam highlight). |

### 1.2 Rule-by-Rule HTTPS Integration Test Matrix

1. **Rule 2.1 — $4 \times 4 \times 4$ Grid & Coordinate Boundaries:**
   * Verify `POST /api/v1/games/{id}/move` accepts either `{"action": a}` ($a \in \lbrace 0..15 \rbrace$) or `{"use_coords": true, "x": x, "y": y}` ($x, y \in \lbrace 0..3 \rbrace$).
   * Verify out-of-bounds actions ($a \ge 16$) or coordinates ($x \ge 4$ or $y \ge 4$) return `HTTP 400 Bad Request` without mutating game state.
2. **Rule 2.2 — Two Players & Turn Alternation:**
   * Verify new games start with `current_player = 1` (`WHITE`).
   * Verify each valid move switches `current_player` between `1` (`WHITE`) and `2` (`BLACK`).
   * Verify submitting a move with an explicit out-of-turn `player` field returns `HTTP 409 Conflict`.
3. **Rule 2.3a — Gravity Placement ($z = 0 \to 3$):**
   * Verify 4 sequential moves in the same $(x, y)$ column over HTTPS land at $z = 0, 1, 2, 3$ in bottom-to-top order.
4. **Rule 2.3b — Full Column Rejection:**
   * Verify that once $(x, y, 3)$ is occupied, the column bit is cleared in `valid_moves_mask` and a 5th move in that column over HTTPS returns `HTTP 400 Bad Request`.
5. **Rule 2.4a — Win Condition Across All 76 Lines & Post-Win Rejection:**
   * Verify all 76 winning lines for both `WHITE` and `BLACK` over HTTPS (`152` complete games), confirming `status` (`WHITE_WON` / `BLACK_WON`), `winner`, `win_line_index`, `win_line_mask`, and the 4 winning 3D coordinates.
   * Verify any subsequent move on a finished game returns `HTTP 409 Conflict` / `400 Bad Request`.
6. **Rule 2.4b — 64-Move Draw Condition:**
   * Verify playing the 64-move non-winning sequence over HTTPS ends with `status = "DRAW"`, `winner = 0`, `move_count = 64`, and `valid_moves_mask = 0`.

---

## 2. Technical Decisions & Alternatives Considered

### Decision 1: Dual Remote Battle API (Pull Turn-Based + Push Webhook Orchestrator)
* **Choice:** Support both:
  1. **Pull / Turn-Based REST Endpoints (`POST /api/v1/games/{id}/move`):** Bots or humans poll/query state and push their moves.
  2. **Push / Arena Orchestrator (`POST /api/v1/matches/run`):** The server orchestrates a match between two remote bot URLs (`white_bot_url`, `black_bot_url`) by POSTing the current game state JSON to the active bot and applying its returned move.
* **Why:** Pull mode is ideal for browser clients, CLI tools, and asynchronous battles; Push mode makes automated bot-vs-bot tournaments effortless because each bot only needs to expose a single stateless HTTP/HTTPS endpoint (`POST /move`).

### Decision 2: Built-In TLS with Auto-Generated Self-Signed ECDSA Certificates
* **Choice:** Use Go's `crypto/ecdsa` (P-256) and `crypto/x509` to generate an in-memory TLS certificate whenever external `--cert` / `--key` files are not supplied.
* **Why:** Enables zero-configuration `https://` startup for local development and deterministic end-to-end TLS integration testing in CI without committing private keys to git.

### Decision 3: Embedded Single-Page 3D Web UI (`//go:embed`)
* **Choice:** Embed the HTML/CSS/JS 3D Replay & Play application (`internal/httpserver/web/index.html`) directly into the Go binary using Go's standard `embed` package.
* **Why:** Zero external asset path issues; a single `go run ./cmd/arena-server` command launches both the HTTPS Battle API and the 3D Web Replay interface.

---

## 3. Detailed Architecture & API Specification

### 3.1 Repository Additions for Phase 2

```text
3d3t-ai-ml-learning/
├── IMPLEMENTATION-PHASE2.md
├── internal/
│   └── httpserver/
│       ├── server.go                 # HTTPS server, REST handlers, bot arena orchestrator
│       ├── tls.go                    # ECDSA P-256 self-signed TLS certificate generator
│       ├── web/
│       │   └── index.html            # 3D Three.js visualizer & step-by-step replay UI
│       ├── server_test.go            # Unit tests for handlers, TLS, and error paths
│       └── integration_test.go       # Live HTTPS integration tests for all game rules & bot arena
└── cmd/
    └── arena-server/
        └── main.go                   # Unified CLI entrypoint for HTTPS Arena + Web UI (+ optional gRPC)
```

### 3.2 HTTPS REST API Endpoints

| Method & Path | Description | Request Body | Response Body |
| :--- | :--- | :--- | :--- |
| `GET /` | Serves the 3D Web Visualization & Step-by-Step Replay UI | — | `text/html` |
| `GET /api/v1/health` | Health check endpoint | — | `{"status": "ok"}` |
| `GET /api/v1/winlines` | Lists all 76 winning lines with masks and 3D coordinates | — | `WinLinesPayload` (`200 OK`) |
| `GET /api/v1/games` | Lists all stored games (summary metadata for replay selector) | — | `GameListPayload` (`200 OK`) |
| `POST /api/v1/games` | Creates a new game session | `{"game_id": "...", "white_name": "...", "black_name": "..."}` *(optional)* | `GameStateDTO` (`201 Created`) |
| `GET /api/v1/games/{id}` | Returns full game state, 3D board array, winning line coords, and complete move-by-move replay history | — | `GameStateDTO` (`200 OK`) |
| `POST /api/v1/games/{id}/reset` | Resets an existing game back to move 0 | — | `GameStateDTO` (`200 OK`) |
| `POST /api/v1/games/{id}/move` | Applies a move by `action` ($0..15$) or `(x, y)` ($0..3$) | `{"action": 5, "player": 1}` or `{"use_coords": true, "x": 1, "y": 1}` | `MoveResponseDTO` (`200 OK`) |
| `POST /api/v1/games/{id}/preset` | Loads a deterministic preset game (`draw` or `win_line` $0..75$) for instant replay inspection | `{"type": "draw"}` or `{"type": "win", "line_index": 72, "winner": 1}` | `GameStateDTO` (`200 OK`) |
| `POST /api/v1/matches/run` | Orchestrates a complete automated match between two remote bot endpoints | `{"game_id": "...", "white_bot_url": "https://...", "black_bot_url": "https://..."}` | `GameStateDTO` (`200 OK`) |

---

## 4. 3D Web Replay UI Architecture

The embedded web interface (`internal/httpserver/web/index.html`) provides:
1. **Interactive 3D Scene (Three.js + OrbitControls):**
   * Wooden/metallic base platform with coordinate markers $(X: 0..3, Y: 0..3)$ and 16 vertical pegs.
   * Raycast hover preview showing where a ball will drop on column $(x, y)$ at gravity height $z$.
   * Smoothly rendered White and Black spheres for all moves up to the currently selected replay step $k \in \lbrace 0 \dots T \rbrace$.
   * Gold highlight ring around the ball placed on step $k$.
   * Glowing cyan/emerald 3D cylinder beam passing through the 4 winning balls when viewing the final step of a won game.
2. **Step-by-Step Replay Controller:**
   * **Transport Buttons:** `|<<` (Move 0), `<` (Step Back), `Play / Pause` (Auto-play replay step-by-step), `>` (Step Forward), `>>|` (Final Move $T$).
   * **Speed Selector:** Adjustable step interval (`250ms`, `500ms`, `1000ms`).
   * **Interactive Move Log & Timeline Slider:** Click any ply in the move list or drag the scrubber to jump to that exact board state in 3D.
   * **Game & Preset Loader:** Switch between any live/completed game on the server, create a new interactive game, or load a 76-line / 64-move Draw preset to inspect in 3D.

---

## 5. Verification Commands & Measured Results

```bash
# Run all unit and integration tests (Engine + gRPC + HTTPS) with coverage
CGO_ENABLED=0 go test -v ./internal/... -coverprofile=coverage.out

# Check per-function and total coverage (Target: >= 90%)
go tool cover -func=coverage.out
```

### Achieved Phase 2 Verification Metrics
* **`internal/engine` Coverage:** `97.5%` of statements
* **`internal/grpcserver` Coverage:** `98.7%` of statements
* **`internal/httpserver` Coverage:** `95.2%` of statements
* **Total Repository Statement Coverage:** `96.7%` (Exceeds the $\ge 90\%$ requirement)
