# 3T3D Neural Network Model Card & Benchmark Report (`MODEL.md`)

## 1. Model Overview & Architecture

**`TTT3DNet`** ([`python/ttt3d/network.py`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/network.py)) is an AlphaZero-style **3D Residual Convolutional Neural Network** with an **11-channel 76-Line Geometric Ray Stem**, **Full-Cube $4 \times 4 \times 4$ Global + $3 \times 3 \times 3$ Local Residual Blocks**, and dual **Policy ($\pi$)** and **Value ($v$)** heads designed for $4 \times 4 \times 4$ 3D Gravity Tic-Tac-Toe (**3T3D**).

### 1.1 Input Representation & 76-Line Geometric Ray Stem
The network consumes the canonical 3D volumetric tensor of shape **`(B, 3, 4, 4, 4)`** (`float32`), indexed spatially as `[Channel, Z, Y, X]`:
- **Channel 0 (`MyPieces`):** `1.0` at coordinates $(x, y, z)$ occupied by the player to move, `0.0` elsewhere.
- **Channel 1 (`OpponentPieces`):** `1.0` at coordinates $(x, y, z)$ occupied by the opponent, `0.0` elsewhere.
- **Channel 2 (`GravitySurface`):** `1.0` at the exact $(x, y, z_{\text{land}})$ gravity landing cell for each legal column move $a = 4y + x \in \lbrace 0..15 \rbrace$, `0.0` elsewhere.

Inside [`extract_geometric_features`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/network.py#L64-L131) (and [`extractGeometricFeatures`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/internal/agents/neural.go#L240-L329) in Go), the `(B, 3, 4, 4, 4)` input is projected through the $76 \times 64$ winning-line incidence matrix $M \in \lbrace 0, 1 \rbrace^{76 \times 64}$ to form an **11-channel `(B, 11, 4, 4, 4)` volumetric stem tensor**:
- **Channels 3–6 (Tactical Gravity Surface Planes):**
  - `ImmediateWinSurface`: `1.0` at any gravity landing cell completing a 4-in-a-row for the active player.
  - `ImmediateBlockSurface`: `1.0` at any gravity landing cell completing a 4-in-a-row for the opponent (forced 4th-ball block).
  - `PoisonSurface`: `1.0` at any landing cell $(x, y, z)$ that immediately unlocks an opponent 4th-ball win at $(x, y, z+1)$.
  - `SelfTrapSurface`: `1.0` at any landing cell $(x, y, z)$ that unlocks the active player's own 4th-ball at $(x, y, z+1)$.
- **Channels 7–10 (Volumetric 3-in-a-Row & 2-in-a-Row Fork Planes):**
  - Per-cell density of unblocked 3-in-a-row completion cells and open 2-in-a-row fork lines for both players across all 64 cells.

### 1.2 Layer-by-Layer Topology & Parameter Count

To eliminate the receptive-field blindness of $3 \times 3 \times 3$ kernels on a length-4 cube, each [`ResBlock3D`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/network.py#L195-L213) combines a local `Conv3d(C → C, k=3×3×3, pad=1)` branch with a **full-cube global `Conv3d(C → C, k=4×4×4, pad=0)`** branch in parallel before activation:

| Component | Layers | Output Shape | Active Checkpoint (`C=32, B=2`) | Full Config (`C=64, B=3`) |
| :--- | :--- | :--- | :--- | :--- |
| **0. 76-Line Ray Feature Extractor** | Exact $76 \times 64$ line incidence projection (`3 → 11` channels) | `(B, 11, 4, 4, 4)` | `0` (exact) | `0` (exact) |
| **1. Volumetric Stem** | `Conv3d(11 → C, k=3×3×3, pad=1)` + `BatchNorm3d(C)` + `ReLU` | `(B, C, 4, 4, 4)` | `9,568` params | `19,136` params |
| **2. Global+Local 3D Residual Tower** | $B \times$ [`ResBlock3D`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/network.py#L195-L213) (`Conv3d(3×3×3) + Conv3d(4×4×4)` + `Conv3d(3×3×3)` + skip) | `(B, C, 4, 4, 4)` | `242,048` params | `1,451,136` params |
| **3a. Policy Head ($\pi$)** | `Conv3d(C → 16, k=1×1×1)` + `BN3d(16)` + `ReLU` + `Linear(1024 → 16)` + Illegal Mask (`-1e9`) | `(B, 16)` | `16,944` params | `17,456` params |
| **3b. Value Head ($v$)** | `Conv3d(C → 8, k=1×1×1)` + `BN3d(8)` + `ReLU` + `Linear(512 → 64)` + `ReLU` + `Linear(64 → 1)` + `Tanh` | `(B, 1)` | `33,169` params | `33,425` params |
| **Total Parameters** | — | — | **`301,729`** (`1.18 MB` on disk) | **`1,521,153`** (`5.85 MB` on disk) |

### 1.3 Inference Modes (`NeuralArenaBot` & Go `TTT3DNetAgent`)
At match time, [`NeuralArenaBot`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/arena_bot.py#L168-L343) and Go's [`TTT3DNetAgent`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/internal/agents/neural.go#L410-L574) support two inference modes:
1. **Pure Neural Mode (`use_tactical_lookahead=False` / `ttt3dnet-pure`):** Performs a single forward pass `(1, 3, 4, 4, 4)` through `TTT3DNet` (`1.46 ms`) and greedily selects $\arg\max_{a} \pi_\theta(a|s)$ over legal columns with zero tree search.
2. **Hybrid Neural + Lookahead Mode (`use_tactical_lookahead=True` / `ttt3dnet`, Default):**
   - Evaluates the current board $s$ with `TTT3DNet` to obtain prior action probabilities $\pi_\theta(a|s)$ and state value $v_\theta(s)$.
   - Applies a 64-bit bitboard **2-ply tactical, multi-column fork & vertical-stack fork shield** across all 76 winning lines (`has_unstoppable_threat`).
   - Evaluates all remaining candidate next-states $s'_1, \dots, s'_K$ in a **single batched forward pass** `(K, 3, 4, 4, 4)` through `TTT3DNet`'s Value Head:
     $$\text{Score}(a) = \pi_\theta(a|s) - 0.5 \, v_\theta(s'_a) + 0.15 \cdot |\text{Threats}(s'_a)|$$

---

## 2. Reinforcement Learning & Tactical Curriculum Training

The active checkpoint (`checkpoints/ttt3d_latest.pt` and embedded Go binary `internal/agents/weights/ttt3d_latest.bin`, `channels=32, num_blocks=2`) was trained using [`python/train.py`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/train.py) and [`python/ttt3d/trainer.py`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/trainer.py) with:
- **76-Line Tactical Drill Curriculum (`generate_tactical_threat_curriculum`):** Explicitly constructs gravity-supported 3-in-a-row win and forced 4th-ball block boards across all 76 winning lines and heights $z \in \lbrace 0, 1, 2, 3 \rbrace$.
- **DAgger Adversarial Sparring (`generate_adversarial_sparring`):** Spars the pure model directly against Go `heuristic` and `alphabeta` over gRPC, labeling visited trajectories with a 2-ply fork + depth-3 Alpha-Beta teacher.
- **Max-Blunder Logit Margin + Unlikelihood Loss (`compute_policy_value_loss`):**
  $$\mathcal{L}_{\text{blunder}}(\theta) = -\log\!\left(1 - \sum_{a \in \mathcal{A}_{\text{blunder}}} p_\theta(a \mid s)\right) + 0.35 \left[\frac{1}{|\mathcal{A}_{\text{blunder}}|}\sum_{a \in \mathcal{A}_{\text{blunder}}} \operatorname{ReLU}(\ell_\theta(a) - \ell_{\text{safe}} + 2.5)^2 + \operatorname{ReLU}\!\left(\max_{a \in \mathcal{A}_{\text{blunder}}} \ell_\theta(a) - \ell_{\text{safe}} + 3.0\right)^2\right]$$

### Pure Neural Network 4th-Ball Tactical Verification Across All 76 Winning Lines

Evaluated in **Pure Unshielded Mode (`use_tactical_lookahead=False`)** across all 76 winning lines (verified both in Python and in Go's `TestAll76LinesImmediateWinAndBlock`):

| Line Geometry Category | Number of Lines | Pure NN Forced 4th-Ball Block Accuracy | Pure NN Immediate 4th-Ball Win Accuracy |
| :--- | :---: | :---: | :---: |
| **Horizontal-$X$ Orthogonal** | `16` | **`100.0%`** (`183 / 183`) | **`100.0%`** (`183 / 183`) |
| **Horizontal-$Y$ Orthogonal** | `16` | **`100.0%`** (`199 / 199`) | **`100.0%`** (`199 / 199`) |
| **Vertical-$Z$ Orthogonal** | `16` | **`100.0%`** (`80 / 80`) | **`100.0%`** (`80 / 80`) |
| **2D Planar $XY$ Diagonals** | `8` | **`100.0%`** (`95 / 95`) | **`100.0%`** (`95 / 95`) |
| **2D Planar $XZ$ Diagonals** | `8` | **`100.0%`** (`82 / 82`) | **`100.0%`** (`82 / 82`) |
| **2D Planar $YZ$ Diagonals** | `8` | **`100.0%`** (`89 / 89`) | **`100.0%`** (`89 / 89`) |
| **3D Space Diagonals ($XYZ$)** | `4` | **`100.0%`** (`45 / 45`) | **`100.0%`** (`45 / 45`) |
| **All 76 Winning Lines Total** | **`76`** | **`100.0%` (`773 / 773`)** | **`100.0%` (`773 / 773`)** |

---

## 3. Success Rate Against Baseline Models (20-Game Paired Evaluation)

Each baseline opponent was evaluated over **20 paired games** (10 games with `TTT3DNet` playing first as `WHITE`, 10 games with `TTT3DNet` playing second as `BLACK`). Scores are computed as $\frac{\text{Wins} + 0.5 \times \text{Draws}}{20}$:

| Baseline Opponent | Opponent Description | Hybrid `NeuralArenaBot` (`W - D - L`) | Hybrid Score | Pure Unshielded `TTT3DNet` (`W - D - L`) | Pure NN Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`random`** | Uniform random over legal columns | **`20 - 0 - 0`** | **`100.0%`** | **`20 - 0 - 0`** | **`100.0%`** |
| **`heuristic`** | 1-ply win/block + poison filter + 76-line heuristic | **`20 - 0 - 0`** | **`100.0%`** | **`10 - 10 - 0`** | **`75.0%` (`0` losses)** |
| **`alphabeta`** | Depth-3 Negamax with $\alpha$-$\beta$ pruning & move ordering | **`10 - 0 - 10`** | **`50.0%`** | **`10 - 0 - 10`** | **`50.0%`** |
| **`mcts`** | 200 UCT simulations/move with tactical rollout policy | **`15 - 1 - 4`** | **`77.5%`** | **`17 - 0 - 3`** | **`85.0%`** |

### Key Benchmark Highlights:
- **Zero Losses Against `random` and `heuristic` in Pure Mode:** The single-pass unshielded `TTT3DNet` (`ttt3dnet-pure`) goes undefeated (**`20 - 0 - 0` vs `random`** and **`10 wins, 10 draws, 0 losses` vs `heuristic`**) without any external search or tactical wrapper.
- **Parity with Depth-3 Alpha-Beta (`50.0%`) & `85.0%` Win Rate vs 200-Sim MCTS:** In pure forward-pass mode (`1.46 ms`/move), `TTT3DNet` wins `10 / 20` games against depth-3 `alphabeta` and `17 / 20` games (`85.0%`) against 200-simulation `mcts`.

---

## 4. Per-Move Compute Latency & Throughput Benchmarks

Measured on Apple Silicon CPU (single thread, `batch_size = 1` per move decision over 200 timed moves after 20 warmup iterations):

| Inference Engine / Mode | Mean Latency per Move | Median (`p50`) | `p95` Latency | `p99` Latency | Throughput (Moves / Sec) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure `TTT3DNet` Forward Pass** (`use_tactical_lookahead=False`) | **`1.46 ms`** | `1.43 ms` | `1.68 ms` | `1.79 ms` | **`~686 moves/sec`** |
| **Hybrid `NeuralArenaBot`** (Batched 1-Ply Value + 2-Ply Fork Shield) | **`29.72 ms`** | `29.39 ms` | `31.46 ms` | `34.42 ms` | **`~33.6 moves/sec`** |
| **Go `RandomAgent`** (64-bit bitboard) | `< 0.01 ms` | `< 0.01 ms` | `< 0.01 ms` | `< 0.01 ms` | `> 100,000 moves/sec` |
| **Go `HeuristicAgent`** (1-ply + 76-line eval) | `0.04 ms` | `0.04 ms` | `0.06 ms` | `0.09 ms` | `~25,000 moves/sec` |
| **Go `AlphaBetaAgent`** (Depth 3 Negamax) | `1.85 ms` | `1.60 ms` | `3.10 ms` | `4.20 ms` | `~540 moves/sec` |
| **Go `MCTSAgent`** (200 simulations/move) | `14.50 ms` | `14.10 ms` | `18.20 ms` | `21.00 ms` | `~69 moves/sec` |

> [!NOTE]
> Both **Pure `TTT3DNet` (`1.46 ms`)** and **Hybrid `NeuralArenaBot` (`29.72 ms`)** operate well below human perception thresholds (`< 50 ms`) and easily satisfy real-time Arena webhook timeouts (`timeout_ms = 5000 ms`).
