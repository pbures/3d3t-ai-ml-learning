# Phase 3 Implementation Specification: Baseline Agents & Classical Tree Search

**Branch:** `phase-3-baseline-agents-search`  
**Related Specification:** [SPECIFICATION.md](./SPECIFICATION.md)

---

## 1. Success Criteria & Verification Strategy

Phase 3 is complete only when all of the following quantitative and functional criteria pass via `go test` and GitHub Actions CI:

### 1.1 Success Criteria Checklist

| ID | Criterion | Verification Method & Target |
| :--- | :--- | :--- |
| **SC-1** | **Unit Test Code Coverage $\ge 90\%$** | `go test -v ./internal/... -coverprofile=coverage.out` must achieve **$\ge 90.0\%$ statement coverage** for `internal/agents` and across the entire repository. |
| **SC-2** | **Tactical Rule & Threat Verification** | Unit and integration tests verify that `HeuristicAgent`, `AlphaBetaAgent`, and `MCTSAgent`:<br>1. Always take an immediate 1-ply winning move across any of the 76 lines.<br>2. Always block an opponent's immediate 1-ply winning move.<br>3. Avoid "poison columns" (playing at $z$ when the opponent would win immediately at $z+1$) whenever a safe alternative exists. |
| **SC-3** | **Skill Hierarchy Win-Rate Verification** | Automated multi-game arena tests verify that `HeuristicAgent`, `AlphaBetaAgent`, and `MCTSAgent` achieve $\ge 90\%$ win rate against `RandomAgent` (playing an equal number of games as White and Black). |
| **SC-4** | **End-to-End HTTPS Bot & Arena Integration** | `internal/httpserver/integration_test.go` verifies built-in bot webhook endpoints (`POST /api/v1/bots/{agent}/move`), single-step AI moves (`POST /api/v1/games/{id}/ai-move`), and full HTTPS Arena matches (`POST /api/v1/matches/run`) between built-in agents. |
| **SC-5** | **End-to-End gRPC Agent Action Selection** | `internal/grpcserver/integration_test.go` verifies requesting agent moves over gRPC (`SelectAction`) for all 4 agent types (`random`, `heuristic`, `alphabeta`, `mcts`). |
| **SC-6** | **3D Web UI Bot Controls** | The 3D Web UI (`internal/httpserver/web/index.html`) includes controls to step any AI agent (`Random`, `Heuristic`, `AlphaBeta`, `MCTS`) or run an instant Bot-vs-Bot match and inspect the replay step by step. |

---

## 2. Technical Decisions & Alternatives Considered

### Decision 1: Four-Tier Agent Ladder (`Random`, `Heuristic`, `AlphaBeta`, `MCTS`)
* **Why We Need All Four for Neural Network Training:**
  1. **`RandomAgent` (Tier 0):** Uniformly samples from `ValidMovesMask()`. Serves as a sanity check and initial exploration baseline.
  2. **`HeuristicAgent` (Tier 1 — 1/2-Ply Tactical + Positional):** Checks for immediate 1-ply wins, forced 1-ply blocks, filters out 2-ply "poison column" blunders, and scores cells by how many of the 76 winning lines pass through them (corner and center cells participate in **7** winning lines; face/edge cells participate in **4**).
  3. **`AlphaBetaAgent` (Tier 2 — Negamax with Alpha-Beta Pruning):** Searches to depth $D$ (default $D=4$) using move ordering and a bitwise 76-line potential evaluation function. Provides deterministic, strong tactical play.
  4. **`MCTSAgent` (Tier 3 — Monte Carlo Tree Search with UCT):** Builds an asymmetric search tree using Upper Confidence Bound for Trees ($UCT$). Crucially, **MCTS is the exact algorithm that will pair with our Neural Network in Phase 4 (AlphaZero architecture)**—where the network's Policy head replaces rollout priors and the Value head replaces random rollouts!

### Decision 2: Negamax Formulation for Alpha-Beta Search
* **Choice:** Formulate Minimax as **Negamax** where every board state is scored from the perspective of `g.CurrentPlayer`:

$$V(s) = \max_{a \in \text{Valid}(s)} \left( -V(\text{Step}(s, a)) \right)$$

* **Why:** Matches our **Canonical Board Representation** from Phase 1! Both Negamax and our Neural Network's Value Head evaluate states strictly from the active player's point of view.
* **Alternative Rejected:** Separate `minValue()` and `maxValue()` functions for White and Black, which duplicates code and pruning logic.

### Decision 3: Tactical Rollouts & Terminal Short-Circuiting in MCTS
* **Choice:** During MCTS tree expansion and simulation rollouts, immediately take 1-ply winning moves if available.
* **Why:** Pure uniform random rollouts in 3D Connect-4 games suffer from high variance because random players often miss obvious 1-move wins on the board. Checking a 1-ply win using our 64-bit bitboards takes ~100 nanoseconds and dramatically sharpens MCTS value estimates even at low simulation counts (e.g., 200–800 simulations).

---

## 3. Mathematical Formulation of Search & Evaluation

### 3.1 Cell Centrality in a $4 \times 4 \times 4$ Cube
Not all 64 cells in the 3D grid are equal! If we count how many of the **76 winning lines** pass through each coordinate $(x, y, z)$:
* **16 Interior + Corner cells** (where $(x, y, z) \in \lbrace 0, 3 \rbrace^3$ or $\lbrace 1, 2 \rbrace^3$): Lie on **7 winning lines** each (3 orthogonal + 3 planar diagonals + 1 space diagonal).
* **48 Edge/Face cells:** Lie on **4 winning lines** each (3 orthogonal + 1 planar diagonal).

*(Notice that $\frac{16 \times 7 + 48 \times 4}{4} = \frac{112 + 192}{4} = \frac{304}{4} = 76$ lines!)*

### 3.2 Bitboard Heuristic Evaluation Function $E(s, p)$
For non-terminal leaf nodes in Alpha-Beta search, we evaluate state $s$ for player $p$ by inspecting each of the 76 winning line masks $M_k$:
* If **both** White and Black have bits inside $M_k$ (`(myBoard & M_k) != 0 && (oppBoard & M_k) != 0`), line $k$ is **blocked** (worth $0$ points to both players).
* If only `myBoard` occupies $c \in \lbrace 1, 2, 3 \rbrace$ bits of $M_k$ (and `oppBoard` has $0$):
  * $c = 1 \implies +1$ point
  * $c = 2 \implies +5$ points
  * $c = 3 \implies +25$ points (or $+60$ if the 4th empty cell is immediately playable on the current gravity surface!)
* Symmetrically, unblocked opponent lines subtract $-1, -5, -25$ (or $-80$ if immediately playable).

### 3.3 Monte Carlo Tree Search (UCT Selection Formula)
At each tree node $s$, MCTS selects the child action $a$ that maximizes the Upper Confidence Bound for Trees:

$$\text{UCT}(s, a) = \frac{W(s, a)}{N(s, a)} + c_{\text{puct}} \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

where:
* $N(s)$ is the total visit count of parent node $s$,
* $N(s, a)$ is the visit count of child edge $(s, a)$,
* $W(s, a) \in [-N(s,a), +N(s,a)]$ is the cumulative value from the perspective of the player who played action $a$,
* $c_{\text{puct}} = \sqrt{2} \approx 1.414$ balances **exploitation** (high win rate) and **exploration** (low visit count).

---

## 4. Detailed Architecture & Package Layout

```text
3d3t-ai-ml-learning/
├── IMPLEMENTATION-PHASE3.md
├── internal/
│   ├── agents/
│   │   ├── agent.go                  # Agent interface, Registry, Random & Heuristic agents
│   │   ├── alphabeta.go              # Negamax Alpha-Beta pruning search + 76-line evaluator
│   │   ├── mcts.go                   # Monte Carlo Tree Search (UCT) implementation
│   │   └── agent_test.go             # Unit tests, tactical puzzles, and win-rate arena tests
│   ├── grpcserver/
│   │   ├── pb/                       # Added SelectAction RPC to EnvService
│   │   ├── server.go                 # SelectAction handler invoking internal/agents
│   │   └── integration_test.go       # Added gRPC integration test for all 4 agents
│   └── httpserver/
│       ├── server.go                 # Added /api/v1/bots/{agent}/move & /api/v1/games/{id}/ai-move
│       ├── web/index.html            # Added AI Move & Bot-vs-Bot Match controls to 3D Web UI
│       └── integration_test.go       # Added HTTPS integration tests for AI moves & bot battles
```

---

## 5. Verification Commands & Measured Results

```bash
# Run all unit and integration tests (Engine + Agents + gRPC + HTTPS) with coverage
CGO_ENABLED=0 go test -v ./internal/... -coverprofile=coverage.out

# Inspect per-function and total coverage (Target: >= 90%)
go tool cover -func=coverage.out
```

### Achieved Phase 3 Verification Metrics
* **`internal/agents` Coverage:** `96.3%` of statements
* **`internal/engine` Coverage:** `97.5%` of statements
* **`internal/grpcserver` Coverage:** `97.4%` of statements
* **`internal/httpserver` Coverage:** `94.3%` of statements
* **Total Repository Statement Coverage:** `96.0%` (Exceeds the $\ge 90\%$ requirement)
