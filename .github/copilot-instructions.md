# GitHub Copilot Code Review & Repository Instructions

When reviewing Pull Requests in this repository (`3d3t-ai-ml-learning`), verify adherence to our project specification (`SPECIFICATION.md`) and phase implementation documents (`IMPLEMENTATION-PHASE*.md`):

1. **Game Rules Integrity (`SPECIFICATION.md` Section 2):**
   - 3D grid is $4 \times 4 \times 4$ (64 positions) mapped to 64-bit bitboards (`bitIndex(x, y, z) = 16z + 4y + x`).
   - Two players (`WHITE = 1`, `BLACK = 2`) strictly alternate turns.
   - Gravity applies along the $Z$ axis ($z = 0 \to 3$). Playing in a full column ($z = 3$ occupied) is illegal.
   - All **76 winning lines** (48 orthogonal, 24 planar diagonal, 4 3D space diagonal) must be checked accurately, and a 64-move full board with no 4-in-a-row results in a `DRAW`.

2. **Testing & Coverage Requirements (`SPECIFICATION.md` Section 4):**
   - Every package must have unit tests achieving **$\ge 90\%$ statement coverage**.
   - Integration tests must verify all game rules end-to-end over the relevant transport layers (gRPC in Phase 1, HTTPS in Phase 2).

3. **Code Quality & Performance:**
   - Core engine functions (`internal/engine`) should remain allocation-free on the hot path to support high-speed tree search and vectorized reinforcement learning self-play.
   - Ensure concurrency safety (proper mutex locking in server/pool handlers).
