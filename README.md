# 3T3D — 3D Gravity Tic-Tac-Toe & Neural Network AI

**3T3D** is an end-to-end engineering and machine learning project centered around 3D Gravity Tic-Tac-Toe (also known as Qubic / 3D Connect-4), played on a $4 \times 4 \times 4$ vertical-gravity grid of 64 positions. Two players (`WHITE` and `BLACK`) alternate dropping balls onto one of the 16 vertical pegs $(x, y) \in \{0..3\}^2$, where gravity pulls each ball down to the lowest unoccupied level $z \in \{0..3\}$. The objective is to be the first player to complete a line of four balls of the same color along any of the **76 winning lines** spanning the cube: 48 orthogonal lines ($X$, $Y$, and vertical $Z$ axes), 24 planar diagonals ($XY$, $XZ$, and $YZ$ planes), or 4 three-dimensional space diagonals running corner-to-opposite-corner through the interior of the cube.

At its foundation, the project implements a high-performance, zero-allocation **Go (Golang) Bitboard Game Engine** alongside a dual-transport server architecture. Because the $4 \times 4 \times 4$ board contains exactly 64 positions, each player's state maps bijectively onto a single 64-bit unsigned integer (`uint64`), enabling constant-time move validation, gravity resolution, and 76-line win detection via bitwise masks. The Go backend exposes this engine through two specialized interfaces: a low-latency **Vectorized gRPC Service** over Unix Domain Sockets capable of stepping hundreds of concurrent game environments in parallel for reinforcement learning self-play, and an **HTTPS Battle & Matchmaking API** that supports both turn-based play and automated webhook-orchestrated bot-vs-bot tournaments.

To make gameplay and AI decision-making visually transparent, the Go server embeds an interactive **3D Web Visualization and Step-by-Step Replay Studio** built with Three.js and WebGL. Through the browser interface, users can rotate and zoom around the 3D board, play interactively by clicking pegs or column controls, trigger AI moves, or launch automated matches between agents. Every game stores a deterministic move-by-move history, allowing spectators and developers to scrub through any match step-by-step (`|<<`, `<`, `Play/Pause`, `>`, `>>|`), inspect board states at any ply, and visualize the exact 3D winning line highlighted across the cube.

Before introducing deep neural networks, the repository establishes a rigorous ladder of **Classical Search and Baseline Agents** (`Random`, `Heuristic`, `Minimax` with Alpha-Beta pruning, and `Monte Carlo Tree Search` with UCT rollouts). These classical agents exploit the geometric structure of the $4 \times 4 \times 4$ cube—such as prioritizing the 16 "power cells" (the 8 interior core cells and 8 outer corners that sit at the intersections of the 3D space diagonals) and detecting immediate tactical threats—to provide calibrated, reproducible benchmarks against which learned policies are evaluated.

Building upon this engine and benchmark suite, the machine learning pipeline uses **Python and PyTorch** to train a **3D Policy-Value Convolutional Neural Network** via self-play reinforcement learning. The neural network consumes a 3-channel $4 \times 4 \times 4$ canonical spatial representation of the board (current player's pieces, opponent's pieces, and playable gravity landing spots) and outputs both a 16-action policy distribution ($\pi$) and a scalar value estimate ($v \in [-1, +1]$), leveraging the 8-fold dihedral symmetry of the 3D cube during training and competing directly in the arena over the HTTPS Battle API.

## Programmatic Arena API & OpenAPI 3.1 Specification (`https://3t3d.buresovi.net`)

External players and developers can connect custom algorithms running on their own devices directly to the Arena Server over HTTPS:

- **OpenAPI 3.1 Specification & Interactive Docs:**
  - `GET https://3t3d.buresovi.net/openapi.json` — OpenAPI 3.1 JSON specification
  - `GET https://3t3d.buresovi.net/openapi.yaml` — OpenAPI 3.1 YAML specification
  - `GET https://3t3d.buresovi.net/docs` — Interactive Swagger UI documentation
- **Discover Implemented AI Models (`GET /api/v1/models`):**
  - Lists all built-in classical and neural models (`ttt3dnet`, `ttt3dnet-pure`, `mcts`, `alphabeta`, `heuristic`, `random`).
- **Create Named Matches & Play Against Built-in Models (`POST /api/v1/games`, `PATCH /api/v1/games/{id}`, `POST /api/v1/games/{id}/move`):**
  - Give any match a custom human-readable `name` (e.g. `"Qualifying Match #1 — LaptopBot vs TTT3DNet"`), custom `game_id`, `white_name`, `black_name`, and optional `opponent_agent` (`"ttt3dnet"`, `"ttt3dnet-pure"`, `"mcts"`, `"alphabeta"`, `"heuristic"`, `"random"`).
  - Submit moves programmatically via `POST /api/v1/games/{id}/move` using either `action` (`0..15`) or `(x, y)` coordinates (`use_coords: true`). When `opponent_agent` is set on the game or move request, the server automatically computes and applies the AI model's reply in the same call (`ai_move`).
- **Model Context Protocol (MCP) Server Layer (`POST https://3t3d.buresovi.net/mcp` & `-mcp-stdio`):**
  - Exposes a JSON-RPC 2.0 MCP server at `/mcp` (and `/api/v1/mcp`) as well as over stdio (`./arena-server -mcp-stdio`).
  - Provides MCP tools (`list_models`, `list_games`, `create_game`, `get_game`, `rename_game`, `play_move`, `trigger_ai_move`, `reset_game`, `run_match`) and the `openapi://3t3d/openapi.json` MCP resource so LLM agents and MCP clients can discover models, create named games, choose whether they or the AI starts (`human_player: 1` or `2`), and play moves directly.


