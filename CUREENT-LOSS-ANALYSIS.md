# Current Loss Analysis: 3D Tic-Tac-Toe Neural Network Models (160 Completed Matches)

## 1. Server vs. Client Performance Summary

Across **160 completed games** on `https://3t3d.buresovi.net`, the server-hosted neural network agents (`ttt3dnet` and `ttt3dnet-pure`) played against two remote client bots (`Stupido 5 (Tic3Tac)` and `Stupido 6 (Tic3Tac)`):

| Matchup (Client vs. Server Model) | Games | Server Wins | Client Wins | Draws | Server Win % | Server Non-Loss % | Avg Moves / Game |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stupido 5 (Tic3Tac)** vs. `ttt3dnet` | 60 | **16** *(9 SW, 7 CW)* | **42** *(21 SW, 21 CW)* | **2** | **26.7%** | **30.0%** | 50.0 |
| **Stupido 5 (Tic3Tac)** vs. `ttt3dnet-pure` | 60 | **5** *(4 SW, 1 CW)* | **55** *(26 SW, 29 CW)* | **0** | **8.3%** | **8.3%** | 42.0 |
| **Stupido 6 (Tic3Tac)** vs. `ttt3dnet` | 20 | **0** *(0 SW, 0 CW)* | **20** *(10 SW, 10 CW)* | **0** | **0.0%** | **0.0%** | 39.4 |
| **Stupido 6 (Tic3Tac)** vs. `ttt3dnet-pure` | 20 | **0** *(0 SW, 0 CW)* | **20** *(10 SW, 10 CW)* | **0** | **0.0%** | **0.0%** | 33.6 |
| **Total Across All Matches** | **160** | **21** *(13 SW, 8 CW)* | **137** *(67 SW, 70 CW)* | **2** | **13.1%** | **14.4%** | **43.6** |

> **Note:** `SW` = Server started as White (first move); `CW` = Client started as White (first move).

### Winning Line Geometry Distribution (Across All 158 Decisive Games)

| Matchup | `Diag-XZ` | `Diag-YZ` | `Orthogonal-Y` | `Orthogonal-X` | `Diag-XY` | `Diag-XYZ` | `Orthogonal-Z` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stupido 5** vs. `ttt3dnet` (58 wins) | 10 | 15 | 13 | 10 | 3 | 7 | 0 |
| **Stupido 5** vs. `ttt3dnet-pure` (60 wins) | 12 | 11 | 10 | 8 | 10 | 5 | 4 |
| **Stupido 6** vs. `ttt3dnet` (20 wins) | 10 | 4 | 0 | 2 | 2 | 2 | 0 |
| **Stupido 6** vs. `ttt3dnet-pure` (20 wins) | 5 | 5 | 1 | 2 | 2 | 4 | 1 |
| **Total (158 decisive games)** | **37 (23.4%)** | **35 (22.2%)** | **24 (15.2%)** | **22 (13.9%)** | **17 (10.8%)** | **18 (11.4%)** | **5 (3.2%)** |

Vertical-plane diagonals (`Diag-XZ` and `Diag-YZ`) and 3D space diagonals (`Diag-XYZ`) account for **57.0% of all decisive games** (and **80.0%** of `Stupido 6` wins against `ttt3dnet`).

---

## 2. Failure-Mode Classification of All 137 Losses

Every single one of the 137 server losses was replayed move-by-move to inspect the exact board state on the server's final two turns before losing:

| Failure Mode at the Decisive Turn | `ttt3dnet` (62 Losses) | `ttt3dnet-pure` (75 Losses) | Total (137 Losses) |
| :--- | :---: | :---: | :---: |
| **1. Missed an immediate 1-ply block** (opponent already had 3-in-a-row playable on the current surface) | **0** *(0.0%)* | **0** *(0.0%)* | **0** *(0.0%)* |
| **2. Stepped into an unforced 1-ply "poison column"** (played at $z$ giving opponent immediate win at $z+1$ when safe alternatives existed) | **0** *(0.0%)* | **1** *(1.3%)* | **1** *(0.7%)* |
| **3. Allowed a Stacked Vertical Fork** (opponent created threats at $(x, y, z)$ and $(x, y, z+1)$ in the **same column**; when server blocked $z$, gravity unlocked $z+1$) | **30** *(48.4%)* | **46** *(61.3%)* | **76** *(55.5%)* |
| **4. Allowed a Double Immediate Surface Fork** (opponent created 2+ simultaneous playable threats in different columns on the current surface) | **19** *(30.6%)* | **20** *(26.7%)* | **39** *(28.5%)* |
| **5. Multi-column Zugzwang / Poison Exhaustion** (opponent accumulated suspended upper-layer threats across all remaining non-full columns until every legal move unlocked a winning cell above) | **13** *(21.0%)* | **8** *(10.7%)* | **21** *(15.3%)* |

---

## 3. Concrete Game Examples from `Stupido 6` Matches

1. **`SW - tic3tac-lmr-s6-ttt3dnet-b1`** (`ttt3dnet` as White, 54 moves):
   - On move 50, `Stupido 6` makes a forcing threat that compels `ttt3dnet` to block in column $(0,0)$ on move 51 (`z=1`), which raises the column height.
   - On move 52, `Stupido 6` plays $(0,1,2)$, completing a **stacked vertical fork** in column $(0,0)$ at `z=2` and `z=3`.
   - On move 53, `ttt3dnet` is forced to block $(0,0,2)$, immediately unlocking $(0,0,3)$ (`Diag-XZ` win on move 54).
2. **`CW - tic3tac-lmr-s6-ttt3dnet-w1`** (`ttt3dnet` as Black, 31 moves):
   - On move 29, `Stupido 6` plays $(1,1,3)$, creating a **stacked vertical fork** in column $(0,1)$ at `z=1` and `z=2`.
   - On move 30, `ttt3dnet` blocks $(0,1,1)$, unlocking $(0,1,2)$ (`Diag-XZ` win on move 31).
3. **`SW - tic3tac-lmr-s6-ttt3dnet-pure-b2`** (`ttt3dnet-pure` as White, 24 moves):
   - On move 22, `Stupido 6` plays $(0,1,1)$, simultaneously creating two immediate surface threats at $(1,0,0)$ and $(1,1,2)$ (**double immediate surface fork**). `ttt3dnet-pure` blocks $(1,1,2)$ on move 23, and `Stupido 6` wins at $(1,0,0)$ on move 24.
4. **`CW - tic3tac-lmr-s6-ttt3dnet-pure-w1`** (`ttt3dnet-pure` as Black, 31 moves):
   - On move 29, `Stupido 6` plays $(1,2,2)$, creating a **stacked vertical fork** in column $(1,0)$ at `z=1` and `z=2`. `ttt3dnet-pure` blocks $(1,0,1)$ on move 30, unlocking $(1,0,2)$ (`Diag-XZ` win on move 31).

---

## 4. Root Causes of Why the Server Models Were Losing

### Root Cause 1: Shallow Tactical Horizon (3 Plies vs. 4-to-15+ Ply Forcing Sequences)
In `internal/agents/neural.go` (`SelectActionWithPolicy`):
- **Step 5 (`nonForkVulnerable`):** Only checks whether the opponent has a **1-move** reply (`oppAct`) that creates a double immediate threat or a stacked vertical fork (`(oppThreats & (oppThreats >> 16)) != 0`). That is a **2-ply** check.
- **Step 6 (Hybrid Alpha-Beta Rollout):** Calls `ab := NewAlphaBetaAgent(3, n.seed)` and runs `ab.negamax(clone, 2, -999999, 999999, opp)`. Because `clone` already has 1 ply applied, `depth=2` only looks **3 plies ahead** (`my move -> opp move -> my move`).
- **Why this fails against `Stupido 5` and `Stupido 6`:** `Stupido 5` searches deeper, and `Stupido 6` (`tic3tac-lmr-s6`) adds **Late Move Reductions (LMR)** to search 6–10+ plies ahead. It executes **4-to-8-ply forcing sequences**:
  1. Play a forcing threat in Column A (forcing the server to block in Column A and raise its height).
  2. Play a second forcing threat in Column B (forcing another block).
  3. Play the setup move that completes a stacked vertical fork at `(x, y, z)` and `(x, y, z+1)`.
  Because the fork only materializes at ply 4 or 6, the 3-ply `AlphaBetaAgent` in `ttt3dnet` returns a routine heuristic score at ply 3 and is completely blind to the trap.

### Root Cause 2: No Check Extensions on Forced 1-Ply Blocks
In 3D Tic-Tac-Toe, whenever a move creates an immediate playable 3-in-a-row threat (`len(oppWins) == 1`), the opponent's reply is **strictly forced** (branching factor = 1).
- Standard `negamax` decrements `depth` by 1 on every ply, even when the reply is forced.
- A 6-ply forcing chain of 3 checks and 3 forced responses has an effective branching factor of 1 on the defensive plies, yet consumes 6 full depth levels in unextended search.

### Root Cause 3: Neural Prior Dilution in `ttt3dnet` (`0.008` Alpha-Beta Weight)
In `internal/agents/neural.go`:
```go
combined := probs[act] + 0.5*myNextVal + threatBonus + 0.008*float32(abVal) - ceilingPenalty
```
Unless `abVal` hits a terminal win/loss (`±90,000` within 3 plies), intermediate heuristic scores (`±50` to `±300` for building 2-in-a-row planes or allowing opponent 2-in-a-row setups) are multiplied by `0.008` (`±0.4` to `±2.4`), which is easily overridden by `probs[act]` + `0.5*myNextVal` when the neural network is overconfident.

### Root Cause 4: Pure Neural Network (`ttt3dnet-pure`) Input & Teacher Limitations
1. **Missing Multi-Ply / Stacked-Fork Input Features:** `extractGeometricFeatures` (11 channels in `neural.go` and `network.py`) encodes 1-ply wins, 1-ply blocks, and 1-ply poison cells (`z` where `z+1` is an opponent win), but does **not** explicitly encode:
   - **Stacked vertical fork completion targets** (playing at cell `c` completes a pair of threats at `(x,y,z)` and `(x,y,z+1)` in another column).
   - **2-ply forcing fork threats** (columns where playing creates a forced block that immediately leads to a double threat or stacked fork).
2. **Shallow Training Teachers:** During training (`python/train.py`), `TTT3DNet` was only trained against `alphabeta` at depth 3 and `mcts` with 64 simulations—it never faced deep LMR / iterative-deepening searches (`15+` plies in forcing/endgame lines) that punish subtle mid-game diagonal intersections.
