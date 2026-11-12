# 3T3D Neural Network Model Card & Benchmark Report (`MODEL.md`)

## 1. Model Overview & Architecture (`ttt3d-deep-gray1`)

**`TTT3DNet`** ([`python/ttt3d/network.py`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/network.py)) is an AlphaZero-style **3D Residual Convolutional Neural Network** with a **19-channel 76-Line Multi-Ply Geometric Ray Stem**, **Full-Cube $4 \times 4 \times 4$ Global + $3 \times 3 \times 3$ Local Residual Blocks**, and dual **Policy ($\pi$)** and **Value ($v$)** heads designed for $4 \times 4 \times 4$ 3D Gravity Tic-Tac-Toe (**3T3D**).

The flagship checkpoint **`ttt3d-deep-gray1`** (`checkpoints/ttt3d-deep-gray1.pt` / `internal/agents/weights/ttt3d_deep_gray1.bin`) was architected and trained to resolve the multi-ply tactical blindspots identified in [`CUREENT-LOSS-ANALYSIS.md`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/CUREENT-LOSS-ANALYSIS.md) and to compete against the **`deep`** (`15–20+` ply, `~1.5s`/move) bitboard Principal Variation Search baseline.

### 1.1 Input Representation & 19-Channel 76-Line Multi-Ply Geometric Ray Stem
The network consumes the canonical 3D volumetric tensor of shape **`(B, 3, 4, 4, 4)`** (`float32`), indexed spatially as `[Channel, Z, Y, X]`:
- **Channel 0 (`MyPieces`):** `1.0` at coordinates $(x, y, z)$ occupied by the player to move, `0.0` elsewhere.
- **Channel 1 (`OpponentPieces`):** `1.0` at coordinates $(x, y, z)$ occupied by the opponent, `0.0` elsewhere.
- **Channel 2 (`GravitySurface`):** `1.0` at the exact $(x, y, z_{\text{land}})$ gravity landing cell for each legal column move $a = 4y + x \in \lbrace 0..15 \rbrace$, `0.0` elsewhere.

Inside [`extract_geometric_features`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/network.py#L68-L195) (and [`extractGeometricFeatures`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/internal/agents/neural.go#L256-L449) in Go), the `(B, 3, 4, 4, 4)` input is projected through the $76 \times 64$ winning-line incidence matrix $M \in \lbrace 0, 1 \rbrace^{76 \times 64}$ to form a **19-channel `(B, 19, 4, 4, 4)` volumetric stem tensor**:
- **Channels 3–6 (1-Ply & 2-Ply Tactical Gravity Surface Planes):**
  - `Ch 3 (`ImmediateWinSurface`)`: `1.0` at any gravity landing cell completing a 4-in-a-row for the active player.
  - `Ch 4 (`ImmediateBlockSurface`)`: `1.0` at any gravity landing cell completing a 4-in-a-row for the opponent (forced 4th-ball block).
  - `Ch 5 (`PoisonSurface`)`: `1.0` at any landing cell $(x, y, z)$ that immediately unlocks an opponent 4th-ball win at $(x, y, z+1)$.
  - `Ch 6 (`SelfTrapSurface`)`: `1.0` at any landing cell $(x, y, z)$ that unlocks the active player's own 4th-ball at $(x, y, z+1)$.
- **Channels 7–10 (Volumetric 3-in-a-Row & 2-in-a-Row Ray Density Planes):**
  - Per-cell density of unblocked 3-in-a-row completion cells (`Ch 7–8`) and open 2-in-a-row fork lines (`Ch 9–10`) for both players across all 64 cells.
- **Channels 11–18 (Multi-Ply Forward Visibility & Stacked-Fork Planes):**
  - `Ch 11–12`: 2nd-ply (`z = h+1`) and 3rd-ply (`z = h+2`) gravity landing surfaces.
  - `Ch 13–14`: Active player and opponent **stacked vertical fork** (`z` and `z+1` in the same column) builder & target line projections.
  - `Ch 15–16`: Active player and opponent multi-line fork intersection planes (`cell_2 >= 2`).
  - `Ch 17`: Upper-column (`z+1, z+2`) 2-ply/3-ply trap & Zugzwang danger surface.
  - `Ch 18`: Net open 1-in-a-row ray balance per empty cell.

### 1.2 Layer-by-Layer Topology & Parameter Count (`ttt3d-deep-gray1`)

To eliminate the receptive-field blindness of $3 \times 3 \times 3$ kernels on a length-4 cube, each [`ResBlock3D`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/network.py#L259-L277) combines a local `Conv3d(C → C, k=3×3×3, pad=1)` branch with a **full-cube global `Conv3d(C → C, k=4×4×4, pad=0)`** branch in parallel before activation:

| Component | Layers | Output Shape | `ttt3d-deep-gray1` (`C=32, B=3`) | Full Config (`C=64, B=3`) |
| :--- | :--- | :--- | :--- | :--- |
| **0. 76-Line Ray Feature Extractor** | Exact $76 \times 64$ line incidence projection (`3 → 19` channels) | `(B, 19, 4, 4, 4)` | `0` (exact) | `0` (exact) |
| **1. Volumetric Stem** | `Conv3d(19 → C, k=3×3×3, pad=1)` + `BatchNorm3d(C)` + `ReLU` | `(B, C, 4, 4, 4)` | `16,480` params | `32,960` params |
| **2. Global+Local 3D Residual Tower** | $3 \times$ [`ResBlock3D`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/python/ttt3d/network.py#L259-L277) (`Conv3d(3×3×3) + Conv3d(4×4×4)` + `Conv3d(3×3×3)` + skip) | `(B, C, 4, 4, 4)` | `363,072` params | `1,451,136` params |
| **3a. Policy Head ($\pi$)** | `Conv3d(C → 16, k=1×1×1)` + `BN3d(16)` + `ReLU` + `Linear(1024 → 16)` + Illegal Mask (`-1e9`) | `(B, 16)` | `16,944` params | `17,456` params |
| **3b. Value Head ($v$)** | `Conv3d(C → 8, k=1×1×1)` + `BN3d(8)` + `ReLU` + `Linear(512 → 64)` + `ReLU` + `Linear(64 → 1)` + `Tanh` | `(B, 1)` | `33,169` params | `33,425` params |
| **Total Parameters** | — | — | **`429,665`** (`1.64 MB` on disk) | **`1,534,977`** (`5.90 MB` on disk) |

---

## 2. Deep Adaptive Baseline Agent (`deep`)

Implemented in [`internal/agents/deep.go`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/internal/agents/deep.go), [`DeepSearchAgent`](file:///Users/pavelbures/Documents/antigravity/noble-brahmagupta/internal/agents/deep.go#L41-L65) (`Name() == "deep"`) is a high-speed 64-bit bitboard Principal Variation Search (PVS) baseline engine with:
- **Adaptive Iterative Deepening (`TimeBudget = 1.5s`, `MinDepth = 15`):** Dynamically deepens to at least **15 plies** and continues searching until its **`~1.5s` (`1500 ms`)** per-move time budget is reached, typically reaching **19–22+ plies** (`~2.7M` nodes/move) in midgame positions.
- **Bitwise Gravity & Poison Masking:** Computes the 16-column 3D landing surface in $O(1)$ via `(^occ) & ((occ << 16) | 0xFFFF)` and filters 1-ply poison columns via `surface & (oppThreats >> 16)`.
- **Check Extensions & LMR:** Automatically extends forced 1-ply block moves by `+1` ply (preventing the horizon effect on forcing chains) while applying Late Move Reductions (LMR), Killer heuristics, History tables, and a 262,144-entry Zobrist Transposition Table.

---

## 3. Verification & Benchmark Results (`ttt3d-deep-gray1`)

### 3.1 Pure Neural Network 4th-Ball Tactical Verification Across All 76 Winning Lines
Evaluated in **Pure Unshielded Mode (`ttt3d-deep-gray1-pure`)** across all 76 winning lines:
- **Forced 4th-Ball Block Accuracy:** **`100.0%` (`773 / 773`)** across all 76 horizontal, vertical, 2D diagonal, and 3D space-diagonal winning lines.
- **Immediate 4th-Ball Win Accuracy:** **`100.0%` (`773 / 773`)** across all 76 winning lines.

### 3.2 Success Rate Against Baseline Models (20-Game Paired Evaluation)

| Baseline Opponent | Opponent Description | Hybrid `ttt3d-deep-gray1` (`W - D - L`) | Hybrid Score | Pure Unshielded `ttt3d-deep-gray1-pure` (`W - D - L`) | Pure NN Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`random`** | Uniform random over legal columns | **`20 - 0 - 0`** | **`100.0%`** | **`20 - 0 - 0`** | **`100.0%`** |
| **`heuristic`** | 1-ply win/block + poison filter + 76-line heuristic | **`20 - 0 - 0`** | **`100.0%`** | **`20 - 0 - 0`** | **`100.0%` (`0` losses)** |
| **`alphabeta`** | Depth-3 Negamax with $\alpha$-$\beta$ pruning & move ordering | **`20 - 0 - 0`** | **`100.0%`** | **`20 - 0 - 0`** | **`100.0%`** |
| **`deep` (`15+` ply)** | Adaptive Iterative Deepening PVS (`>=15` plies, `~1.5s`/move) | **`10 - 0 - 10`** | **`50.0%`** | **`10 - 0 - 10`** | **`50.0%`** |
