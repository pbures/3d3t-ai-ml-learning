# 3T3D Neural Network Model Card & Benchmark Report (`MODEL.md`)

## 1. Model Overview & Architecture

**`TTT3DNet`** ([`python/ttt3d/network.py`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/network.py)) is an AlphaZero-style **3D Residual Convolutional Neural Network** with dual **Policy ($\pi$)** and **Value ($v$)** heads designed for $4 \times 4 \times 4$ 3D Gravity Tic-Tac-Toe (**3T3D**).

### 1.1 Input Representation
The network consumes a canonical 3D volumetric tensor of shape **`(B, 3, 4, 4, 4)`** (`float32`), indexed spatially as `[Channel, Z, Y, X]`:
- **Channel 0 (`MyPieces`):** `1.0` at coordinates $(x, y, z)$ occupied by the player to move, `0.0` elsewhere.
- **Channel 1 (`OpponentPieces`):** `1.0` at coordinates $(x, y, z)$ occupied by the opponent, `0.0` elsewhere.
- **Channel 2 (`GravitySurface`):** `1.0` at the exact $(x, y, z_{\text{land}})$ gravity landing cell for each legal column move $a = 4y + x \in \lbrace 0..15 \rbrace$, `0.0` elsewhere.

### 1.2 Layer-by-Layer Topology & Parameter Count

| Component | Layers | Output Shape | Active Checkpoint (`C=32, B=2`) | Full Config (`C=64, B=3`) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Volumetric Stem** | `Conv3d(3 → C, k=3×3×3, pad=1)` + `BatchNorm3d(C)` + `ReLU` | `(B, C, 4, 4, 4)` | `2,656` params | `5,312` params |
| **2. 3D Residual Tower** | $B \times$ [`ResBlock3D`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/network.py#L69-L81) (`2 × [Conv3d(C → C, k=3×3×3, pad=1) + BN3d]` + skip) | `(B, C, 4, 4, 4)` | `110,848` params | `664,320` params |
| **3a. Policy Head ($\pi$)** | `Conv3d(C → 16, k=1×1×1)` + `BN3d(16)` + `ReLU` + `Linear(1024 → 16)` + Illegal Mask (`-1e9`) | `(B, 16)` | `16,944` params | `17,456` params |
| **3b. Value Head ($v$)** | `Conv3d(C → 8, k=1×1×1)` + `BN3d(8)` + `ReLU` + `Linear(512 → 64)` + `ReLU` + `Linear(64 → 1)` + `Tanh` | `(B, 1)` | `33,169` params | `33,425` params |
| **Total Parameters** | — | — | **`163,617`** (`656 KB` on disk) | **`720,513`** (`2.8 MB` on disk) |

### 1.3 Hybrid Inference Engine (`NeuralArenaBot`)
At match time, [`NeuralArenaBot`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/arena_bot.py#L140-L325) supports two inference modes:
1. **Pure Neural Mode (`use_tactical_lookahead=False`):** Performs a single forward pass `(1, 3, 4, 4, 4)` through `TTT3DNet` and greedily selects $\arg\max_{a} \pi_\theta(a|s)$ over legal columns.
2. **Hybrid Neural + Lookahead Mode (`use_tactical_lookahead=True`, Default):**
   - Evaluates the current board $s$ with `TTT3DNet` to obtain prior action probabilities $\pi_\theta(a|s)$ and state value $v_\theta(s)$.
   - Applies a fast 64-bit bitboard **2-ply tactical & fork shield** across all 76 winning lines (taking immediate 1-ply wins, blocking opponent 1-ply threats, executing offensive 2-way forks, and filtering out moves that allow opponent 1-ply wins or 2-way forks).
   - Evaluates all remaining candidate next-states $s'_1, \dots, s'_K$ in a **single batched forward pass** `(K, 3, 4, 4, 4)` through `TTT3DNet`'s Value Head, combining the policy prior $\pi_\theta(a|s)$ with the 1-ply neural value lookahead $-v_\theta(s'_a)$:
     $$\text{Score}(a) = \pi_\theta(a|s) - 0.5 \, v_\theta(s'_a) + 0.15 \cdot |\text{Threats}(s'_a)|$$

---

## 2. Latest Reinforcement Learning Training Statistics

The active checkpoint (`checkpoints/ttt3d_latest.pt`, `channels=32, num_blocks=2`) was trained over the local Unix Domain Socket gRPC channel (`cmd/grpc-env`) using [`python/train.py`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/train.py) with:
- **Optimizer:** `AdamW` ($\text{lr} = 2 \times 10^{-3}$, $\text{weight\_decay} = 10^{-4}$, gradient clipping `max_grad_norm = 1.0`)
- **Symmetry Augmentation:** $8\times$ dihedral group $D_4$ expansion on every state (`augment=True`)
- **Search Teacher:** Go MCTS (`80` simulations/move) blended with terminal game outcomes $z \in \lbrace -1, 0, +1 \rbrace$
- **Loss Function:** Combined Policy Cross-Entropy + Value MSE:
  $$\mathcal{L}(\theta) = (z - v_\theta(s))^2 - \boldsymbol{\pi}^T \log \mathbf{p}_\theta(s)$$

### Iteration-by-Iteration Training Log

| Iteration | Augmented Buffer Size | Total Loss $\mathcal{L}$ | Policy Loss ($\pi$ CE) | Value Loss ($v$ MSE) | Eval vs. `Random` | Eval vs. `Heuristic` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Iter 01 / 10** | `2,344` | `2.8776` | `2.5697` | `0.3079` | `100.0%` | `0.0%` |
| **Iter 02 / 10** | `5,496` | `2.7405` | `2.5343` | `0.2062` | `100.0%` | `50.0%` |
| **Iter 03 / 10** | `8,304` | `2.7649` | `2.5578` | `0.2070` | `100.0%` | `0.0%` |
| **Iter 04 / 10** | `11,480` | `2.7265` | `2.5111` | `0.2153` | `100.0%` | `0.0%` |
| **Iter 05 / 10** | `14,472` | `2.7152` | `2.5145` | `0.2007` | `100.0%` | `0.0%` |
| **Iter 06 / 10** | `17,208` | `2.7181` | `2.4976` | `0.2205` | `100.0%` | `50.0%` |
| **Iter 07 / 10** | `20,368` | `2.6766` | `2.4523` | `0.2243` | `100.0%` | `0.0%` |
| **Iter 08 / 10** | `23,104` | `2.6527` | `2.4466` | `0.2061` | `100.0%` | `0.0%` |
| **Iter 09 / 10** | `25,792` | `2.6348` | `2.4343` | `0.2006` | `100.0%` | `50.0%` |
| **Iter 10 / 10** | **`28,640`** | **`2.6159`** | **`2.4044`** | **`0.2115`** | **`100.0%`** | **`100.0%`** |

- **Key Learning Dynamics:**
  - Uniform random policy entropy over 16 columns is $\ln(16) \approx 2.7726$. Policy cross-entropy dropped to **`2.4044`** while Value MSE dropped from **`0.3079`** $\to$ **`0.2115`** across 28,640 $D_4$-augmented board states collected from 80 self-play episodes.

---

## 3. Success Rate Against Baseline Models (20-Game Paired Evaluation)

Each baseline opponent was evaluated over **20 paired games** (10 games with `TTT3DNet` playing first as `WHITE`, 10 games with `TTT3DNet` playing second as `BLACK`). Scores are computed as $\frac{\text{Wins} + 0.5 \times \text{Draws}}{20}$:

| Baseline Opponent | Opponent Description | Hybrid `NeuralArenaBot` (`W - D - L`) | Hybrid Win Rate | Pure Unshielded `TTT3DNet` (`W - D - L`) | Pure NN Win Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`random`** | Uniform random over legal columns | **`20 - 0 - 0`** | **`100.0%`** | **`20 - 0 - 0`** | **`100.0%`** |
| **`heuristic`** | 1-ply win/block + poison filter + 76-line heuristic | **`20 - 0 - 0`** | **`100.0%`** | `0 - 0 - 20` | `0.0%` |
| **`alphabeta`** | Depth-3 Negamax with $\alpha$-$\beta$ pruning & move ordering | **`20 - 0 - 0`** | **`100.0%`** | `0 - 0 - 20` | `0.0%` |
| **`mcts`** | 200 UCT simulations/move with tactical rollout policy | **`3 - 2 - 15`** | **`20.0%`** | `0 - 0 - 20` | `0.0%` |

### 1-Ply Blunder Suppression RL & Pure vs. Hybrid Analysis:
- **1-Ply Threat Suppression Loss (`compute_policy_value_loss`):** During RL training, every state $(s_t, \boldsymbol{\pi}_t, z_t)$ is analyzed by `compute_tactical_action_masks` and `rectify_tactical_sample` to identify all 1-ply losing moves $\mathcal{A}_{\text{blunder}}(s_t)$ (moves that miss an immediate 4-in-a-row win, fail to block an opponent's 3-in-a-row threat, or drop into a poison column). The training objective adds an unlikelihood penalty $-\log(1 - \sum_{a \in \mathcal{A}_{\text{blunder}}} p_\theta(a \mid s_t))$ plus a symmetric logit-margin hinge penalty $\operatorname{ReLU}(\ell_\theta(a) - \ell_{\text{safe}}(s_t) + 2.5)^2$, and immediately assigns $z_t = -1.0$ to any rollout step that commits a 1-ply blunder.
- Against **`random`**, this 1-ply threat suppression loss drives the **Pure Unshielded `TTT3DNet` (`ttt3dnet-pure`, single `1.17 ms` forward pass with zero search)** to a **`100.0%` (`20 - 0 - 0`) win rate**.
- Against **`heuristic`** and **`alphabeta` (depth 3)**, the hybrid `NeuralArenaBot` (`ttt3dnet`) achieves a **100.0% (`20 - 0 - 0`) sweep** (winning all 10 games as White and all 10 games as Black) by combining `TTT3DNet`'s learned 3D positional evaluation with 2-ply bitboard fork prevention.
- Against **`mcts` (200 simulations/move)**, `NeuralArenaBot` achieves a **20.0% score (`3 wins, 2 draws, 15 losses`)** while using only a 1-ply neural value lookahead compared to MCTS's 200 full-game tree rollouts.

---

## 4. Per-Move Compute Latency & Throughput Benchmarks

Measured on Apple Silicon CPU (single thread, `batch_size = 1` per move decision over 200 timed moves after 20 warmup iterations):

| Inference Engine / Mode | Mean Latency per Move | Median (`p50`) | `p95` Latency | `p99` Latency | Throughput (Moves / Sec) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pure `TTT3DNet` Forward Pass** (`use_tactical_lookahead=False`) | **`1.17 ms`** | `1.12 ms` | `1.59 ms` | `2.47 ms` | **`~855 moves/sec`** |
| **Hybrid `NeuralArenaBot`** (Batched 1-Ply Value + 2-Ply Fork Shield) | **`27.29 ms`** | `26.84 ms` | `28.45 ms` | `36.55 ms` | **`~36.6 moves/sec`** |
| **Go `RandomAgent`** (64-bit bitboard) | `< 0.01 ms` | `< 0.01 ms` | `< 0.01 ms` | `< 0.01 ms` | `> 100,000 moves/sec` |
| **Go `HeuristicAgent`** (1-ply + 76-line eval) | `0.04 ms` | `0.04 ms` | `0.06 ms` | `0.09 ms` | `~25,000 moves/sec` |
| **Go `AlphaBetaAgent`** (Depth 3 Negamax) | `1.85 ms` | `1.60 ms` | `3.10 ms` | `4.20 ms` | `~540 moves/sec` |
| **Go `MCTSAgent`** (200 simulations/move) | `14.50 ms` | `14.10 ms` | `18.20 ms` | `21.00 ms` | `~69 moves/sec` |

> [!NOTE]
> Both **Pure `TTT3DNet` (`1.17 ms`)** and **Hybrid `NeuralArenaBot` (`27.29 ms`)** operate well below human perception thresholds (`< 50 ms`) and easily satisfy real-time Arena webhook timeouts (`timeout_ms = 5000 ms`).
