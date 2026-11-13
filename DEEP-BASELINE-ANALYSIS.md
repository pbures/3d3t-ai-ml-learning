# Deep Search Baseline: Loss Analysis & Fix (Nov 2026)

## 1. Battles analysed

| Battle (arena) | Games | Stupido 6 wins | deep wins |
|---|---|---|---|
| `tic3tac-arenafix-test-1113`: *Stupido vs deep (fixtest)* | 2 | 2 | 0 |
| `tic3tac-s6-vs-deep-1113`: *Stupido vs deep (s6deep1113)* | 20 | 18 | 2 |

All 22 games were downloaded through the MCP endpoint (`list_battles` → `get_battle` → `get_game`) and replayed locally against an
exact (node-limited) win/draw/loss solver, walking backwards from the final position. The local `deep` agent reproduced the
server's moves 16 of 17 times, so the local analysis applies to production.

## 2. Findings

1. **Every loss was decided long before the end.** Games lasted 44–63 plies, but every lost game was already proven lost
   20–33 plies before the end. Stupido 6 wins by **zugzwang endgames**: it accumulates suspended threats until every remaining
   column forces `deep` to play directly under one of them.
2. **Four outright blunders.** In 4 games, `deep` turned a proven win or draw into a proven loss, in positions with 30–36 empty
   cells:

   | Game | Ply | Before → after | deep played | Correct moves | deep's eval |
   |---|---|---|---|---|---|
   | s6deep-b2 | 30 | Win → Loss | 2 | 0,1,3,4,5,7,8,11,12,13,15 | +0.14 |
   | s6deep-b6 | 28 | Win → Loss | 11 | 12, 15 (15 was pruned by the root pre-filter) | −0.45 |
   | s6deep-w6 | 31 | Draw → Loss | 8 | 14 | −0.90 |
   | s6deep-w8 | 33 | Draw → Loss | 11 | 6, 12, 14 | −0.88 |
3. **The search was much shallower than it reported.**
   - Depths 1..15 ran with `lmrScale=3`: the PV was searched at `depth-2`, and late moves were reduced by `2 + depth/2 + i/2`
     with no full-depth re-search in null-window nodes. "Depth 22" in the logs meant about 5–7 real plies for most moves.
   - A "Phase-2 verification" pass XOR-ed junk into Zobrist keys (polluting the TT), then replaced the best move by comparing
     scores from searches of different depth and reduction settings.
   - An unsound 2-ply root "fork-vulnerability" filter deleted moves before any search ran. In b6 it removed one of the only two
     winning moves.
   - The evaluation ignored **zugzwang parity**, which is exactly what decides these endgames.

## 3. Fix (`internal/agents/deep*.go`)

| Component | File | Description |
|---|---|---|
| Exact WDL solver | `deep_solver.go` | Null-window alpha-beta over {−1,0,+1}. Incremental threat sets, sound pruning only (immediate win, double threat, stacked vertical fork, forced block, poison columns), and a packed 2-way 8-byte TT (2²³ entries, 64 MB, pooled). Uses history ordering and iterative widening per root move. |
| Honest PVS | `deep_pvs.go` | Logarithmic LMR (never on forcing moves) with full-depth re-search, check extensions for forced blocks, TT, killers, history. Incremental threat maintenance: placing a stone can only kill the opponent's threat at that very cell. |
| Parity evaluation | `deep_pvs.go` (`parityThreatScore`) | Connect-4 odd/even theory for even-height columns: suspended threats on z=2 favour White, on z=1/z=3 favour Black; only the lowest threat of each column counts. |
| Root driver | `deep.go` | Sound shortcuts (win, forced block, poison). Then PVS (45% of the budget, 25% from ply 26). From ply 20 the solver plays proven wins and vetoes proven losses; a proven draw is ranked as score 0 against unproven moves. Removed the Phase-2 verification pass and the unsound root filter. |
| Budget | `deep.go` | `DefaultDeepTimeBudget` raised from 1.5 s to **2.8 s** per move, below the 3 s cap. |
| Hybrid lookahead | `deep_hybrid.go` | The fixed-depth PVS used by the neural agents (`ttt3dnet`, `ttt3d-deep-gray1`) is unchanged in behaviour (its `lmrScale=2` semantics are kept); its dead branches were removed. |

## 4. Validation

- **Solver correctness:** matches a naive brute-force reference on 198 endgame positions from the arena games and on 60 random
  late positions (unit test). Null-window answers agree with the exact values.
- **Arena blunder positions:** the new agent plays a value-preserving move in b2, w6 and w8 (b2 and w8 proven by the solver
  within budget). b6 needs about 2× the solver throughput available in 2.8 s.
- **Head-to-head, new vs old deep** (10 random 4-ply openings × both colours, default budgets 2.8 s vs 1.5 s): **15–5**.
  An A/B run without the parity term scored 16–4. Against the old agent, which does not play for zugzwang, the difference is
  within noise. The term is kept because it targets the endgame structure Stupido exploits; re-check it in the next Stupido
  battle.
- **CI:** `go test -race ./internal/...` passes; total coverage is about 94%.
