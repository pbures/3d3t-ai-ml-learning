# Phase 1 Implementation Specification: Go Core Bitboard Engine & Vectorized gRPC Interface

**Branch:** `phase-1-core-engine-grpc`  
**Related Specification:** [SPECIFICATION.md](./SPECIFICATION.md)

---

## 1. Success Criteria & Verification Strategy

Phase 1 is **only** considered complete when all of the following quantitative and functional success criteria pass automatically via `go test`:

### 1.1 Success Criteria Checklist

| ID | Criterion | Verification Method & Target |
| :--- | :--- | :--- |
| **SC-1** | **Unit Test Code Coverage $\ge 90\%$** | `go test ./internal/... -coverprofile=coverage.out` must report **$\ge 90.0\%$ statement coverage** for both `internal/engine` and `internal/grpcserver`. |
| **SC-2** | **End-to-End gRPC Integration Tests for All Game Rules** | `internal/grpcserver/integration_test.go` spins up a real gRPC server over a Unix Domain Socket (`unix:///tmp/3t3d_test_*.sock`), connects a real gRPC client, and verifies every rule in Section 2 & Section 4.3 of `SPECIFICATION.md`. |
| **SC-3** | **Exhaustive 76 Winning Lines Verification** | Both unit tests and gRPC integration tests iterate through **all 76 winning lines** (48 orthogonal, 24 planar diagonal, 4 space diagonal) and verify that completing each line triggers a win and returns the exact winning 4-bit mask. |
| **SC-4** | **Deterministic 64-Move Draw Verification** | Both unit tests and gRPC integration tests execute a full 64-move non-winning game sequence and verify that move 64 transitions the game status to `DRAW` with `valid_moves_mask == 0`. |
| **SC-5** | **Vectorized Batch RL Training Verification** | gRPC integration tests initialize a pool of $N=64$ parallel games (`InitPool`) and execute multi-step vectorized `BatchStep` calls, verifying canonical 3-channel tensor layout ($N \times 3 \times 4 \times 4 \times 4$), reward calculation, and automatic reset on terminal states. |

### 1.2 Rule-by-Rule gRPC Integration Test Matrix

To answer *"How do we verify Phase 1 is well implemented?"*, the integration test suite (`internal/grpcserver/integration_test.go`) tests the core engine strictly through the gRPC wire protocol against every specification rule:

1. **Rule 2.1 — $4 \times 4 \times 4$ Grid & Coordinate Boundaries:**
   * Verify actions $a \in \lbrace 0..15 \rbrace$ map bijectively to $(x, y) \in \lbrace 0..3 \rbrace \times \lbrace 0..3 \rbrace$ where $x = a \bmod 4$ and $y = \lfloor a / 4 \rfloor$.
   * Verify out-of-range actions ($a \ge 16$) over gRPC return an `InvalidArgument` status error without mutating game state.
2. **Rule 2.2 — Two Players & Turn Alternation:**
   * Verify a newly reset game over gRPC starts with `current_player = WHITE (1)` and `move_count = 0`.
   * Verify each valid `Step` strictly alternates `current_player` between `WHITE (1)` and `BLACK (2)`.
3. **Rule 2.3a — Gravity Placement ($z = 0 \to 3$):**
   * Verify playing column $(x, y)$ four consecutive times over gRPC places balls at $(x, y, 0)$, $(x, y, 1)$, $(x, y, 2)$, and $(x, y, 3)$ in exact bottom-to-top order.
4. **Rule 2.3b — Full Column Rejection:**
   * Verify that once $(x, y, 3)$ is occupied, bit $a = 4y + x$ in `valid_moves_mask` becomes `0`.
   * Verify attempting a 5th move in that column over gRPC returns an error (`InvalidArgument` / illegal move) and leaves the board state unchanged.
5. **Rule 2.4a — Win Condition Across All 76 Lines & Post-Win Immutability:**
   * Verify all 16 $X$-lines, 16 $Y$-lines, 16 $Z$-lines, 8 $XY$-diagonals, 8 $XZ$-diagonals, 8 $YZ$-diagonals, and 4 $XYZ$ 3D space diagonals over gRPC.
   * Verify that once `status` is `WHITE_WON` or `BLACK_WON`, any subsequent `Step` call over gRPC is rejected.
6. **Rule 2.4b — 64-Move Draw Condition:**
   * Play a complete 64-ply game over gRPC where neither player forms a 4-in-a-row; verify `status == DRAW`, `winner == NONE`, and `valid_moves_mask == 0`.

---

## 2. Technical Decisions & Alternatives Considered

### Decision 1: 64-Bit Bitboard State (`uint64`) vs. 3D Array (`[4][4][4]uint8`)
* **Choice:** Represent each player's pieces as a single `uint64` integer (`White uint64`, `Black uint64`).
* **Why:**
  * $4 \times 4 \times 4 = 64$ cells fits inside one 64-bit CPU register.
  * Copying a board state during tree search (MCTS/Minimax) copies 24 bytes instead of heap-allocating slices.
  * Computing all 16 legal moves takes a single bitwise shift: `uint16(^(White | Black) >> 48) & 0xFFFF`.
  * Checking a winning line is a single bitwise AND: `(board & mask) == mask`.
* **Alternative Rejected:** `[4][4][4]uint8` array. While slightly more obvious at first glance, checking 76 lines via nested loops is ~25x slower and adds branching overhead during millions of self-play rollouts.

### Decision 2: Canonical 3-Channel Observation Tensor ($3 \times 4 \times 4 \times 4$)
* **Choice:** Export observations to the Neural Network from the **perspective of the active player** using 3 binary channels ($3 \times 64 = 192$ bytes/floats per board):
  * **Channel 0 (`MyPieces`):** `1` where the current player has balls.
  * **Channel 1 (`OpponentPieces`):** `1` where the opponent has balls.
  * **Channel 2 (`GravityLanding`):** `1` at the exact $(x, y, z)$ coordinates where each legal column move will land on the current turn.
* **Why:**
  * **Perspective Invariance:** The Neural Network does not need separate weights for White vs. Black; it always learns *"How do I win from my current position?"*
  * **Gravity Awareness:** Including Channel 2 gives the 3D Convolutional layers explicit spatial knowledge of the gravity surface without having to infer vertical support from Channels 0 and 1.
* **Alternative Rejected:** Passing raw `White` and `Black` channels plus a scalar player indicator. That doubles sample complexity because the network must learn duplicate features for White and Black.

### Decision 3: Vectorized gRPC over Unix Domain Sockets (`unix://`)
* **Choice:** Expose a `BatchStep` RPC over a local Unix Domain Socket (`unix:///tmp/3t3d.sock`) that steps $N$ parallel environments in one call and packs observation tensors as contiguous byte buffers (`bytes`).
* **Why:**
  * **Bypasses TCP/IP & TLS:** Unix Domain Sockets execute entirely in OS kernel memory.
  * **Zero Serialization Bloat:** Sending $N \times 192$ bytes in a single `bytes` field avoids allocating thousands of nested Protobuf `repeated float` objects and lets Python load the buffer directly via `np.frombuffer(resp.observations, dtype=np.uint8).reshape(N, 3, 4, 4, 4)`.
* **Alternative Rejected:** Calling `Step` 1 game at a time over HTTP/JSON, which bottlenecks GPU utilization on network round-trips.

### Decision 4: Auto-Reset on Terminal State in `BatchStep`
* **Choice:** In vectorized mode (`BatchStep`), when environment $i$ reaches a terminal state (`WHITE_WON`, `BLACK_WON`, or `DRAW`), the server returns:
  * `dones[i] = true`
  * `rewards[i]` ($+1.0$ if the moving player won, $0.0$ if draw)
  * And **immediately resets** environment $i$ to a fresh empty board so `observations[i]` contains the initial state of the next episode.
* **Why:** Standard convention in vectorized Reinforcement Learning environments (OpenAI Gym `SyncVectorEnv` / EnvPool). It eliminates an extra `Reset(i)` round-trip between Python and Go whenever an episode finishes.

---

## 3. Detailed Architecture & Package Structure

### 3.1 Repository Layout for Phase 1

```text
3d3t-ai-ml-learning/
├── go.mod
├── go.sum
├── SPECIFICATION.md
├── IMPLEMENTATION-PHASE1.md
├── proto/
│   └── env.proto                     # Protobuf schema for single & vectorized gRPC
├── internal/
│   ├── engine/
│   │   ├── board.go                  # Bitboard state, gravity move logic, canonical tensors
│   │   ├── winlines.go               # Precomputation & lookup of the 76 winning uint64 masks
│   │   └── board_test.go             # Comprehensive unit tests (>= 90% coverage) & benchmarks
│   └── grpcserver/
│       ├── pb/                       # Generated Go Protobuf & gRPC code
│       │   ├── env.pb.go
│       │   └── env_grpc.pb.go
│       ├── server.go                 # gRPC service implementation (single & vectorized pool)
│       ├── server_test.go            # Unit tests for service methods & error cases
│       └── integration_test.go       # Full end-to-end gRPC integration tests over Unix socket
└── cmd/
    └── grpc-env/
        └── main.go                   # CLI entrypoint to run the gRPC server (UDS or TCP)
```

### 3.2 Component & Data Flow Diagram

```mermaid
sequenceDiagram
    participant Client as gRPC Client (Test / Python RL Trainer)
    participant UDS as Unix Domain Socket (unix:///tmp/3t3d.sock)
    participant Server as grpcserver.EnvServer
    participant Pool as Vectorized Env Pool [N]engine.Game
    participant Engine as engine.Game (Bitboard)

    Client->>UDS: InitPool(num_envs = N)
    UDS->>Server: InitPoolRequest
    Server->>Pool: Allocate N engine.NewGame()
    Pool-->>Server: Initial Canonical Observations (N x 192 bytes)
    Server-->>Client: BatchStateResponse (observations, valid_masks)

    loop Training / Integration Verification Loop
        Client->>UDS: BatchStep(actions: [N]uint8)
        UDS->>Server: BatchStepRequest
        par Step each environment i in 0..N-1
            Server->>Engine: ApplyMove(actions[i])
            Engine->>Engine: Resolve Gravity z & Update Bitboard
            Engine->>Engine: CheckWin() against [76]uint64 masks
            alt Game i is Terminal (Win or Draw)
                Engine-->>Server: done=true, reward (+1 or 0), winner
                Server->>Engine: Reset() for next episode
            else Game i is Ongoing
                Engine-->>Server: done=false, reward=0
            end
            Engine-->>Server: Write 192-byte Canonical Tensor + ValidMovesMask
        end
        Server-->>Client: BatchStepResponse (observations, rewards, dones, valid_masks, winners)
    end
```

---

## 4. Implementation Details & Mathematical Formulation

### 4.1 Coordinate, Action, and Bitboard Mapping

1. **3D Coordinate to Bit Index ($0 \dots 63$):**

$$\text{bitIndex}(x, y, z) = 16z + 4y + x$$

2. **Bit Index to 3D Coordinate:**

$$x = \text{bitIndex} \bmod 4, \quad y = \lfloor \text{bitIndex} / 4 \rfloor \bmod 4, \quad z = \lfloor \text{bitIndex} / 16 \rfloor$$

3. **Action $a \in \lbrace 0 \dots 15 \rbrace$ to Column $(x, y)$:**

$$a = 4y + x \iff x = a \bmod 4, \quad y = \lfloor a / 4 \rfloor$$

4. **Gravity Resolution via Bit Shifts:**
   For column action $a \in \lbrace 0 \dots 15 \rbrace$, the 4 vertical cells from bottom to top are at bit positions $a, a+16, a+32, a+48$.
   Let $\text{occ} = \text{White} \cup \text{Black}$. We find the lowest $z \in \lbrace 0, 1, 2, 3 \rbrace$ where bit $a + 16z$ of $\text{occ}$ is $0$:
   ```go
   for z := uint8(0); z < 4; z++ {
       bit := a + 16*z
       if (occupied & (uint64(1) << bit)) == 0 {
           return z, bit, true
       }
   }
   return 0, 0, false // Column is full
   ```

### 4.2 Precomputing the 76 Winning Lines (`[76]uint64`)

Every winning line is defined by a starting coordinate $(x_0, y_0, z_0)$ and a direction step vector $(\Delta x, \Delta y, \Delta z) \in \lbrace -1, 0, +1 \rbrace^3 \setminus \lbrace (0,0,0) \rbrace$:

$$\text{Mask} = \sum_{k=0}^{3} 2^{\text{bitIndex}(x_0 + k\Delta x,\; y_0 + k\Delta y,\; z_0 + k\Delta z)}$$

 categorized into 7 geometric groups (stored with metadata for testing and 3D UI highlighting):
1. **X-Orthogonal (16 lines):** $\Delta = (1, 0, 0)$ for $y \in \lbrace 0..3 \rbrace, z \in \lbrace 0..3 \rbrace$.
2. **Y-Orthogonal (16 lines):** $\Delta = (0, 1, 0)$ for $x \in \lbrace 0..3 \rbrace, z \in \lbrace 0..3 \rbrace$.
3. **Z-Orthogonal / Vertical (16 lines):** $\Delta = (0, 0, 1)$ for $x \in \lbrace 0..3 \rbrace, y \in \lbrace 0..3 \rbrace$.
4. **XY-Planar Diagonals (8 lines):** $\Delta = (1, 1, 0)$ and $\Delta = (1, -1, 0)$ for each horizontal slice $z \in \lbrace 0..3 \rbrace$.
5. **XZ-Planar Diagonals (8 lines):** $\Delta = (1, 0, 1)$ and $\Delta = (1, 0, -1)$ for each slice $y \in \lbrace 0..3 \rbrace$.
6. **YZ-Planar Diagonals (8 lines):** $\Delta = (0, 1, 1)$ and $\Delta = (0, 1, -1)$ for each slice $x \in \lbrace 0..3 \rbrace$.
7. **XYZ-Space Diagonals (4 lines):**
   * $(0,0,0) \to (3,3,3)$, $\Delta = (1, 1, 1)$
   * $(3,0,0) \to (0,3,3)$, $\Delta = (-1, 1, 1)$
   * $(0,3,0) \to (3,0,3)$, $\Delta = (1, -1, 1)$
   * $(0,0,3) \to (3,3,0)$, $\Delta = (1, 1, -1)$

**Total:** $16 + 16 + 16 + 8 + 8 + 8 + 4 = 76$ lines.

### 4.3 Canonical Observation Tensor Layout ($3 \times 4 \times 4 \times 4 = 192$ bytes)

For any game state, `CanonicalTensor()` outputs a flat `[192]byte` array (values `0` or `1`) in row-major order `[channel][z][y][x]`:
* **Offset $0 \dots 63$ (Channel 0 — Current Player):** `1` at $16z + 4y + x$ if the player whose turn it is occupies $(x, y, z)$.
* **Offset $64 \dots 127$ (Channel 1 — Opponent Player):** `1` at $64 + 16z + 4y + x$ if the opponent occupies $(x, y, z)$.
* **Offset $128 \dots 191$ (Channel 2 — Gravity Landing Surface):** For each column $a \in \lbrace 0..15 \rbrace$, if column $a$ is not full and its next available height is $z_{\text{land}}$, byte $128 + 16z_{\text{land}} + a$ is set to `1`.

---

## 5. Verification Commands & Measured Results

Phase 1 can be verified at any time using:

```bash
# Run all unit and gRPC integration tests with coverage profile
CGO_ENABLED=0 go test -v ./internal/... -coverprofile=coverage.out

# Inspect per-function and total coverage (Target: >= 90%)
go tool cover -func=coverage.out

# Run engine microbenchmarks
CGO_ENABLED=0 go test -bench=. -benchmem ./internal/engine/...
```

### Achieved Phase 1 Verification Metrics
* **`internal/engine` Coverage:** `97.5%` of statements
* **`internal/grpcserver` Coverage:** `98.7%` of statements
* **Total Statement Coverage:** `98.0%` (Exceeds the $\ge 90\%$ requirement)
* **Benchmark (`BenchmarkApplyMoveAndCheckWin`):** `5,584 ns/op` for a full 64-ply game (~`87 ns` per move & 76-line win check), `0 B/op`, `0 allocs/op`.
