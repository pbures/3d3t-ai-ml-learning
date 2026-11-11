package httpserver

import (
	"bytes"
	"context"
	"crypto/tls"
	"crypto/x509"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pbures/3d3t-ai-ml-learning/internal/agents"
	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

//go:embed web/index.html web/docs.html web/openapi.yaml web/openapi.json
var webAssets embed.FS

// ModelInfoDTO describes one of the built-in AI models returned by `GET /api/v1/models`.
type ModelInfoDTO struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
}

// BuiltinModels lists all implemented AI models available in the Arena Server.
var BuiltinModels = []ModelInfoDTO{
	{
		ID:          "ttt3dnet",
		Name:        "TTT3DNet Hybrid (3D ResNet + 2-Ply Tactical Shield)",
		Type:        "neural_hybrid",
		Description: "3D Residual CNN (Policy + Value dual head) combined with a 64-bit bitboard 2-ply fork shield and 1-ply value lookahead.",
	},
	{
		ID:          "ttt3dnet-pure",
		Name:        "TTT3DNet Pure (Raw 3D ResNet Forward Pass)",
		Type:        "neural_pure",
		Description: "Single-pass 3D Residual CNN selecting argmax over legal column policy probabilities with zero tree search.",
	},
	{
		ID:          "mcts",
		Name:        "Monte Carlo Tree Search (250 UCT Simulations)",
		Type:        "search",
		Description: "Upper Confidence Bound applied to Trees (UCT) with tactical rollout policy.",
	},
	{
		ID:          "alphabeta",
		Name:        "Alpha-Beta Negamax (Depth 3)",
		Type:        "search",
		Description: "Depth-3 Negamax search with alpha-beta pruning, move ordering, and 76-line heuristic evaluation.",
	},
	{
		ID:          "heuristic",
		Name:        "Tactical Heuristic (1-Ply + 76 Lines)",
		Type:        "heuristic",
		Description: "Fast 1-ply immediate win/block check, poison-column filter, and 76-line positional connectivity scorer.",
	},
	{
		ID:          "random",
		Name:        "Random Baseline",
		Type:        "baseline",
		Description: "Uniformly random legal column selector.",
	},
}

// CoordDTO represents a 3D coordinate or move record in JSON payloads.
type CoordDTO struct {
	Action     uint8 `json:"action"`
	X          uint8 `json:"x"`
	Y          uint8 `json:"y"`
	Z          uint8 `json:"z"`
	Player     uint8 `json:"player,omitempty"`
	MoveNumber uint8 `json:"move_number,omitempty"`
}

// GameStateDTO is the full JSON representation of a 3T3D game session for remote bots and the 3D Web UI.
type GameStateDTO struct {
	GameID          string         `json:"game_id"`
	Name            string         `json:"name"`
	WhiteName       string         `json:"white_name"`
	BlackName       string         `json:"black_name"`
	OpponentAgent   string         `json:"opponent_agent,omitempty"`
	WhiteBitboard   uint64         `json:"white_bitboard"`
	BlackBitboard   uint64         `json:"black_bitboard"`
	CurrentPlayer   uint8          `json:"current_player"`
	Status          string         `json:"status"`
	StatusCode      uint8          `json:"status_code"`
	Winner          uint8          `json:"winner"`
	WinLineMask     uint64         `json:"win_line_mask"`
	WinLineIndex    int            `json:"win_line_index"`
	WinLineCategory string         `json:"win_line_category,omitempty"`
	WinLineCoords   []CoordDTO     `json:"win_line_coords,omitempty"`
	ValidMovesMask  uint16         `json:"valid_moves_mask"`
	ValidActions    []int          `json:"valid_actions"`
	MoveCount       uint8          `json:"move_count"`
	Board           [4][4][4]uint8 `json:"board"` // [z][y][x]
	CanonicalTensor []int          `json:"canonical_tensor"`
	History         []CoordDTO     `json:"history"`
}

// GameSummaryDTO is a compact descriptor returned when listing games (`GET /api/v1/games`).
type GameSummaryDTO struct {
	GameID        string `json:"game_id"`
	Name          string `json:"name"`
	WhiteName     string `json:"white_name"`
	BlackName     string `json:"black_name"`
	OpponentAgent string `json:"opponent_agent,omitempty"`
	Status        string `json:"status"`
	Winner        uint8  `json:"winner"`
	MoveCount     uint8  `json:"move_count"`
	CurrentPlayer uint8  `json:"current_player"`
}

// CreateGameRequest is the optional request body for `POST /api/v1/games`.
type CreateGameRequest struct {
	GameID        string `json:"game_id"`
	Name          string `json:"name"`
	WhiteName     string `json:"white_name"`
	BlackName     string `json:"black_name"`
	OpponentAgent string `json:"opponent_agent,omitempty"`
	HumanPlayer   uint8  `json:"human_player,omitempty"`
}

// UpdateGameRequest is the request body for `PATCH /api/v1/games/{id}` to rename a match or configure an AI opponent.
type UpdateGameRequest struct {
	Name          *string `json:"name,omitempty"`
	WhiteName     *string `json:"white_name,omitempty"`
	BlackName     *string `json:"black_name,omitempty"`
	OpponentAgent *string `json:"opponent_agent,omitempty"`
}

// MoveRequest is the JSON payload for `POST /api/v1/games/{id}/move` (and the response from remote webhook bots).
type MoveRequest struct {
	Player        uint8  `json:"player,omitempty"`         // Optional: if non-zero (1 or 2), verifies turn order
	Action        *int   `json:"action,omitempty"`         // Action in 0..15
	UseCoords     bool   `json:"use_coords,omitempty"`     // If true, uses (X, Y) instead of Action
	X             int    `json:"x,omitempty"`
	Y             int    `json:"y,omitempty"`
	OpponentAgent string `json:"opponent_agent,omitempty"` // Optional: built-in AI model to immediately reply after this move
	AutoReply     *bool  `json:"auto_reply,omitempty"`     // Optional: override whether configured OpponentAgent replies automatically
}

// MoveResponseDTO is returned after a valid move is applied via `POST /api/v1/games/{id}/move`.
type MoveResponseDTO struct {
	State    GameStateDTO `json:"state"`
	LastMove CoordDTO     `json:"last_move"`
	AIMove   *CoordDTO    `json:"ai_move,omitempty"`
	Reward   float32      `json:"reward"`
	Done     bool         `json:"done"`
}

// PresetRequest is the JSON payload for `POST /api/v1/games/{id}/preset`.
type PresetRequest struct {
	Type      string `json:"type"`       // "draw" or "win"
	LineIndex int    `json:"line_index"` // 0..75 when type == "win"
	Winner    uint8  `json:"winner"`     // 1 (White) or 2 (Black) when type == "win"
}

// AIMoveRequest is the JSON payload for `POST /api/v1/games/{id}/ai-move`.
type AIMoveRequest struct {
	Agent string `json:"agent"` // "ttt3dnet", "ttt3dnet-pure", "mcts", "alphabeta", "heuristic", "random"
}

// RunMatchRequest is the JSON payload for `POST /api/v1/matches/run` to orchestrate a bot-vs-bot match.
type RunMatchRequest struct {
	GameID      string `json:"game_id"`
	Name        string `json:"name,omitempty"`
	WhiteName   string `json:"white_name"`
	BlackName   string `json:"black_name"`
	WhiteBotURL string `json:"white_bot_url,omitempty"`
	BlackBotURL string `json:"black_bot_url,omitempty"`
	WhiteAgent  string `json:"white_agent,omitempty"`
	BlackAgent  string `json:"black_agent,omitempty"`
	TimeoutMs   int    `json:"timeout_ms,omitempty"`
}

// WinLineDTO describes one of the 76 winning lines for `GET /api/v1/winlines`.
type WinLineDTO struct {
	Index    int        `json:"index"`
	Category string     `json:"category"`
	Mask     uint64     `json:"mask"`
	Coords   []CoordDTO `json:"coords"`
}

type gameSession struct {
	id            string
	name          string
	whiteName     string
	blackName     string
	opponentAgent string
	game          *engine.Game
}

// ArenaServer manages HTTP/HTTPS REST battle endpoints, remote bot matchmaking, and the 3D Web Replay UI.
type ArenaServer struct {
	mu         sync.RWMutex
	sessions   map[string]*gameSession
	seqCounter uint64
	httpClient *http.Client
}

// NewArenaServer creates and initializes an ArenaServer with a "default" game session.
func NewArenaServer(botHTTPClient *http.Client) *ArenaServer {
	if botHTTPClient == nil {
		botHTTPClient = &http.Client{Timeout: 5 * time.Second}
	}
	s := &ArenaServer{
		sessions:   make(map[string]*gameSession),
		httpClient: botHTTPClient,
	}
	s.sessions["default"] = &gameSession{
		id:        "default",
		name:      "CW - default",
		whiteName: "Player 1 (White)",
		blackName: "Player 2 (Black)",
		game:      engine.NewGame(),
	}
	return s
}

func isServerWhitePlayer(whiteName, oppAgentName string, humanPlayer uint8) bool {
	if oppAgentName != "" && humanPlayer == uint8(engine.PlayerBlack) {
		return true
	}
	wLower := strings.ToLower(strings.TrimSpace(whiteName))
	if strings.HasPrefix(wLower, "ai ") || strings.Contains(wLower, "server") {
		return true
	}
	for _, m := range BuiltinModels {
		if wLower == m.ID {
			return true
		}
	}
	return false
}

func formatMatchName(rawName string, serverStartsWhite bool) string {
	clean := strings.TrimSpace(rawName)
	for _, p := range []string{"SW - ", "CW - ", "SW ", "CW "} {
		if strings.HasPrefix(clean, p) {
			clean = strings.TrimSpace(clean[len(p):])
			break
		}
	}
	if serverStartsWhite {
		return "SW - " + clean
	}
	return "CW - " + clean
}

func buildStateDTO(sess *gameSession) GameStateDTO {
	g := sess.game
	var board3D [4][4][4]uint8
	for z := uint8(0); z < 4; z++ {
		for y := uint8(0); y < 4; y++ {
			for x := uint8(0); x < 4; x++ {
				p, _ := g.CellAt(x, y, z)
				board3D[z][y][x] = uint8(p)
			}
		}
	}

	validMask := g.ValidMovesMask()
	validActions := make([]int, 0, 16)
	for a := uint8(0); a < 16; a++ {
		if (validMask & (uint16(1) << a)) != 0 {
			validActions = append(validActions, int(a))
		}
	}

	history := make([]CoordDTO, g.MoveCount)
	for i := uint8(0); i < g.MoveCount; i++ {
		rec := g.History[i]
		history[i] = CoordDTO{
			Action:     rec.Action,
			X:          rec.X,
			Y:          rec.Y,
			Z:          rec.Z,
			Player:     uint8(rec.Player),
			MoveNumber: rec.MoveNumber,
		}
	}

	tensor := g.CanonicalTensor()
	tensorSlice := make([]int, len(tensor))
	for i, b := range tensor {
		tensorSlice[i] = int(b)
	}

	matchName := sess.name
	if matchName == "" {
		matchName = sess.id
	}

	dto := GameStateDTO{
		GameID:          sess.id,
		Name:            matchName,
		WhiteName:       sess.whiteName,
		BlackName:       sess.blackName,
		OpponentAgent:   sess.opponentAgent,
		WhiteBitboard:   g.White,
		BlackBitboard:   g.Black,
		CurrentPlayer:   uint8(g.CurrentPlayer),
		Status:          g.Status.String(),
		StatusCode:      uint8(g.Status),
		Winner:          uint8(g.Winner),
		WinLineMask:     g.WinLineMask,
		WinLineIndex:    g.WinLineIndex,
		ValidMovesMask:  validMask,
		ValidActions:    validActions,
		MoveCount:       g.MoveCount,
		Board:           board3D,
		CanonicalTensor: tensorSlice,
		History:         history,
	}

	if g.WinLineIndex >= 0 && g.WinLineIndex < engine.TotalWinLines {
		wl := engine.WinLines[g.WinLineIndex]
		dto.WinLineCategory = string(wl.Category)
		dto.WinLineCoords = make([]CoordDTO, 4)
		for k, c := range wl.Coords {
			dto.WinLineCoords[k] = CoordDTO{
				Action: 4*c.Y + c.X,
				X:      c.X,
				Y:      c.Y,
				Z:      c.Z,
			}
		}
	}

	return dto
}

func writeJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

// Handler returns the root http.Handler serving the 3D Web Replay UI, OpenAPI 3.1 docs, and `/api/v1/*` endpoints.
func (s *ArenaServer) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleRootUI)
	mux.HandleFunc("/openapi.json", s.handleOpenAPIJSON)
	mux.HandleFunc("/api/v1/openapi.json", s.handleOpenAPIJSON)
	mux.HandleFunc("/openapi.yaml", s.handleOpenAPIYAML)
	mux.HandleFunc("/api/v1/openapi.yaml", s.handleOpenAPIYAML)
	mux.HandleFunc("/docs", s.handleDocsUI)
	mux.HandleFunc("/api/v1/docs", s.handleDocsUI)
	mux.HandleFunc("/mcp", s.handleMCP)
	mux.HandleFunc("/api/v1/mcp", s.handleMCP)
	mux.HandleFunc("/api/v1/health", s.handleHealth)
	mux.HandleFunc("/api/v1/models", s.handleModels)
	mux.HandleFunc("/api/v1/winlines", s.handleWinLines)
	mux.HandleFunc("/api/v1/games", s.handleGamesRoot)
	mux.HandleFunc("/api/v1/games/", s.handleGameByID)
	mux.HandleFunc("/api/v1/bots/", s.handleBotWebhook)
	mux.HandleFunc("/api/v1/matches/run", s.handleRunMatch)
	return mux
}

func (s *ArenaServer) handleRootUI(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" && r.URL.Path != "/index.html" {
		writeError(w, http.StatusNotFound, "route not found")
		return
	}
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	data, _ := webAssets.ReadFile("web/index.html")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func (s *ArenaServer) handleOpenAPIJSON(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	data, _ := webAssets.ReadFile("web/openapi.json")
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func (s *ArenaServer) handleOpenAPIYAML(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	data, _ := webAssets.ReadFile("web/openapi.yaml")
	w.Header().Set("Content-Type", "application/yaml; charset=utf-8")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func (s *ArenaServer) handleDocsUI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	data, _ := webAssets.ReadFile("web/docs.html")
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}

func (s *ArenaServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *ArenaServer) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"models": BuiltinModels})
}

func (s *ArenaServer) handleWinLines(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	lines := make([]WinLineDTO, engine.TotalWinLines)
	for i, wl := range engine.WinLines {
		coords := make([]CoordDTO, 4)
		for k, c := range wl.Coords {
			coords[k] = CoordDTO{Action: 4*c.Y + c.X, X: c.X, Y: c.Y, Z: c.Z}
		}
		lines[i] = WinLineDTO{
			Index:    wl.Index,
			Category: string(wl.Category),
			Mask:     wl.Mask,
			Coords:   coords,
		}
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"lines": lines})
}

func (s *ArenaServer) handleGamesRoot(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.mu.RLock()
		summaries := make([]GameSummaryDTO, 0, len(s.sessions))
		for _, sess := range s.sessions {
			matchName := sess.name
			if matchName == "" {
				matchName = sess.id
			}
			summaries = append(summaries, GameSummaryDTO{
				GameID:        sess.id,
				Name:          matchName,
				WhiteName:     sess.whiteName,
				BlackName:     sess.blackName,
				OpponentAgent: sess.opponentAgent,
				Status:        sess.game.Status.String(),
				Winner:        uint8(sess.game.Winner),
				MoveCount:     sess.game.MoveCount,
				CurrentPlayer: uint8(sess.game.CurrentPlayer),
			})
		}
		s.mu.RUnlock()
		sort.Slice(summaries, func(i, j int) bool {
			return summaries[i].GameID < summaries[j].GameID
		})
		writeJSON(w, http.StatusOK, map[string]interface{}{"games": summaries})

	case http.MethodPost:
		var req CreateGameRequest
		if r.Body != nil {
			bodyBytes, _ := io.ReadAll(r.Body)
			if len(bytes.TrimSpace(bodyBytes)) > 0 {
				if err := json.Unmarshal(bodyBytes, &req); err != nil {
					writeError(w, http.StatusBadRequest, "invalid JSON body")
					return
				}
			}
		}
		oppAgentName := strings.TrimSpace(req.OpponentAgent)
		var oppAgent agents.Agent
		if oppAgentName != "" {
			var err error
			oppAgent, err = agents.NewAgentByName(oppAgentName, 1)
			if err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
			oppAgentName = oppAgent.Name()
		}

		gameID := strings.TrimSpace(req.GameID)
		if gameID == "" {
			seq := atomic.AddUint64(&s.seqCounter, 1)
			gameID = fmt.Sprintf("game-%d", seq)
		}
		matchName := strings.TrimSpace(req.Name)
		if matchName == "" {
			matchName = gameID
		}
		whiteName := strings.TrimSpace(req.WhiteName)
		if whiteName == "" {
			if oppAgent != nil && req.HumanPlayer == uint8(engine.PlayerBlack) {
				whiteName = fmt.Sprintf("AI %s (White)", oppAgentName)
			} else {
				whiteName = "Player 1 (White)"
			}
		}
		blackName := strings.TrimSpace(req.BlackName)
		if blackName == "" {
			if oppAgent != nil && req.HumanPlayer != uint8(engine.PlayerBlack) {
				blackName = fmt.Sprintf("AI %s (Black)", oppAgentName)
			} else {
				blackName = "Player 2 (Black)"
			}
		}
		serverStartsWhite := isServerWhitePlayer(whiteName, oppAgentName, req.HumanPlayer)
		matchName = formatMatchName(matchName, serverStartsWhite)

		g := engine.NewGame()
		// If external player chose Black (2) and an opponent model is configured, let the AI make White's opening move
		if oppAgent != nil && req.HumanPlayer == uint8(engine.PlayerBlack) {
			if act, err := oppAgent.SelectAction(g); err == nil {
				_, _ = g.ApplyMove(act)
			}
		}

		s.mu.Lock()
		sess := &gameSession{
			id:            gameID,
			name:          matchName,
			whiteName:     whiteName,
			blackName:     blackName,
			opponentAgent: oppAgentName,
			game:          g,
		}
		s.sessions[gameID] = sess
		dto := buildStateDTO(sess)
		s.mu.Unlock()

		writeJSON(w, http.StatusCreated, dto)

	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (s *ArenaServer) handleGameByID(w http.ResponseWriter, r *http.Request) {
	trimmed := strings.TrimPrefix(r.URL.Path, "/api/v1/games/")
	parts := strings.Split(strings.Trim(trimmed, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		writeError(w, http.StatusBadRequest, "missing game_id")
		return
	}
	gameID := parts[0]
	subAction := ""
	if len(parts) > 1 {
		subAction = parts[1]
	}

	switch subAction {
	case "":
		switch r.Method {
		case http.MethodGet:
			s.mu.RLock()
			sess, exists := s.sessions[gameID]
			if !exists {
				s.mu.RUnlock()
				writeError(w, http.StatusNotFound, fmt.Sprintf("game %q not found", gameID))
				return
			}
			dto := buildStateDTO(sess)
			s.mu.RUnlock()
			writeJSON(w, http.StatusOK, dto)

		case http.MethodPatch:
			s.handleUpdateGame(w, r, gameID)

		default:
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		}

	case "reset":
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		s.mu.Lock()
		sess, exists := s.sessions[gameID]
		if !exists {
			sess = &gameSession{
				id:        gameID,
				name:      formatMatchName(gameID, false),
				whiteName: "Player 1 (White)",
				blackName: "Player 2 (Black)",
				game:      engine.NewGame(),
			}
			s.sessions[gameID] = sess
		} else {
			sess.game.Reset()
		}
		dto := buildStateDTO(sess)
		s.mu.Unlock()
		writeJSON(w, http.StatusOK, dto)

	case "move":
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		s.handleMove(w, r, gameID)

	case "preset":
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		s.handlePreset(w, r, gameID)

	case "ai-move":
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
		s.handleAIMove(w, r, gameID)

	default:
		writeError(w, http.StatusNotFound, fmt.Sprintf("unknown game sub-route %q", subAction))
	}
}

func (s *ArenaServer) handleUpdateGame(w http.ResponseWriter, r *http.Request, gameID string) {
	var req UpdateGameRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON update request")
		return
	}

	var validatedOpp string
	hasOppUpdate := false
	if req.OpponentAgent != nil {
		hasOppUpdate = true
		trimmed := strings.TrimSpace(*req.OpponentAgent)
		if trimmed != "" {
			ag, err := agents.NewAgentByName(trimmed, 1)
			if err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
			validatedOpp = ag.Name()
		}
	}

	s.mu.Lock()
	sess, exists := s.sessions[gameID]
	if !exists {
		s.mu.Unlock()
		writeError(w, http.StatusNotFound, fmt.Sprintf("game %q not found", gameID))
		return
	}
	if req.WhiteName != nil {
		wn := strings.TrimSpace(*req.WhiteName)
		if wn != "" {
			sess.whiteName = wn
		}
	}
	if req.BlackName != nil {
		bn := strings.TrimSpace(*req.BlackName)
		if bn != "" {
			sess.blackName = bn
		}
	}
	if hasOppUpdate {
		sess.opponentAgent = validatedOpp
	}
	if req.Name != nil {
		n := strings.TrimSpace(*req.Name)
		if n != "" {
			if strings.HasPrefix(n, "SW - ") || strings.HasPrefix(n, "CW - ") {
				sess.name = n
			} else {
				serverStartsWhite := isServerWhitePlayer(sess.whiteName, sess.opponentAgent, 0)
				sess.name = formatMatchName(n, serverStartsWhite)
			}
		}
	} else if req.WhiteName != nil {
		serverStartsWhite := isServerWhitePlayer(sess.whiteName, sess.opponentAgent, 0)
		sess.name = formatMatchName(sess.name, serverStartsWhite)
	}
	dto := buildStateDTO(sess)
	s.mu.Unlock()

	writeJSON(w, http.StatusOK, dto)
}

func (s *ArenaServer) handleAIMove(w http.ResponseWriter, r *http.Request, gameID string) {
	var req AIMoveRequest
	if r.Body != nil {
		bodyBytes, _ := io.ReadAll(r.Body)
		if len(bytes.TrimSpace(bodyBytes)) > 0 {
			if err := json.Unmarshal(bodyBytes, &req); err != nil {
				writeError(w, http.StatusBadRequest, "invalid JSON ai-move request")
				return
			}
		}
	}
	agentName := strings.TrimSpace(req.Agent)

	s.mu.RLock()
	sess, exists := s.sessions[gameID]
	if !exists {
		s.mu.RUnlock()
		writeError(w, http.StatusNotFound, fmt.Sprintf("game %q not found", gameID))
		return
	}
	if agentName == "" {
		if sess.opponentAgent != "" {
			agentName = sess.opponentAgent
		} else {
			agentName = "ttt3dnet"
		}
	}
	snapshot := sess.game.Clone()
	s.mu.RUnlock()

	if snapshot.Status != engine.StatusOngoing {
		writeError(w, http.StatusConflict, engine.ErrGameAlreadyOver.Error())
		return
	}

	ag, err := agents.NewAgentByName(agentName, int64(snapshot.MoveCount)+1)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	action, err := ag.SelectAction(snapshot)
	if err != nil {
		writeError(w, http.StatusConflict, err.Error())
		return
	}

	s.mu.Lock()
	rec, err := sess.game.ApplyMove(action)
	if err != nil {
		s.mu.Unlock()
		writeError(w, http.StatusConflict, err.Error())
		return
	}
	var reward float32
	done := sess.game.Status != engine.StatusOngoing
	if sess.game.Winner != engine.PlayerNone {
		reward = 1.0
	}
	resp := MoveResponseDTO{
		State: buildStateDTO(sess),
		LastMove: CoordDTO{
			Action:     rec.Action,
			X:          rec.X,
			Y:          rec.Y,
			Z:          rec.Z,
			Player:     uint8(rec.Player),
			MoveNumber: rec.MoveNumber,
		},
		Reward: reward,
		Done:   done,
	}
	s.mu.Unlock()

	writeJSON(w, http.StatusOK, resp)
}

func (s *ArenaServer) handleBotWebhook(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	trimmed := strings.TrimPrefix(r.URL.Path, "/api/v1/bots/")
	parts := strings.Split(strings.Trim(trimmed, "/"), "/")
	if len(parts) != 2 || parts[1] != "move" {
		writeError(w, http.StatusNotFound, "expected /api/v1/bots/{agent}/move")
		return
	}
	agentName := parts[0]

	var stateDTO GameStateDTO
	if err := json.NewDecoder(r.Body).Decode(&stateDTO); err != nil {
		writeError(w, http.StatusBadRequest, "invalid GameStateDTO body")
		return
	}

	ag, err := agents.NewAgentByName(agentName, int64(stateDTO.MoveCount)+1)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	g := &engine.Game{
		White:         stateDTO.WhiteBitboard,
		Black:         stateDTO.BlackBitboard,
		CurrentPlayer: engine.Player(stateDTO.CurrentPlayer),
		Status:        engine.GameStatus(stateDTO.StatusCode),
		MoveCount:     stateDTO.MoveCount,
	}
	if g.CurrentPlayer != engine.PlayerWhite && g.CurrentPlayer != engine.PlayerBlack {
		g.CurrentPlayer = engine.PlayerWhite
	}

	act, err := ag.SelectAction(g)
	if err != nil {
		writeError(w, http.StatusConflict, err.Error())
		return
	}
	actInt := int(act)
	writeJSON(w, http.StatusOK, MoveRequest{Action: &actInt})
}

func resolveMoveAction(req MoveRequest) (uint8, error) {
	if req.UseCoords {
		if req.X < 0 || req.X >= 4 || req.Y < 0 || req.Y >= 4 {
			return 0, fmt.Errorf("%w: x=%d, y=%d", engine.ErrInvalidCoords, req.X, req.Y)
		}
		return engine.XYToAction(uint8(req.X), uint8(req.Y))
	}
	if req.Action == nil {
		return 0, fmt.Errorf("%w: action is required when use_coords is false", engine.ErrInvalidAction)
	}
	if *req.Action < 0 || *req.Action >= engine.MaxActions {
		return 0, fmt.Errorf("%w: got %d", engine.ErrInvalidAction, *req.Action)
	}
	return uint8(*req.Action), nil
}

func (s *ArenaServer) handleMove(w http.ResponseWriter, r *http.Request, gameID string) {
	var req MoveRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON move request")
		return
	}

	action, err := resolveMoveAction(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	reqOppName := strings.TrimSpace(req.OpponentAgent)
	var reqOppAgent agents.Agent
	if reqOppName != "" {
		reqOppAgent, err = agents.NewAgentByName(reqOppName, 1)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	}

	s.mu.Lock()
	sess, exists := s.sessions[gameID]
	if !exists {
		s.mu.Unlock()
		writeError(w, http.StatusNotFound, fmt.Sprintf("game %q not found", gameID))
		return
	}

	if sess.game.Status != engine.StatusOngoing {
		s.mu.Unlock()
		writeError(w, http.StatusConflict, engine.ErrGameAlreadyOver.Error())
		return
	}

	if req.Player != 0 {
		if req.Player != uint8(engine.PlayerWhite) && req.Player != uint8(engine.PlayerBlack) {
			s.mu.Unlock()
			writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid player ID %d (must be 1 or 2)", req.Player))
			return
		}
		if req.Player != uint8(sess.game.CurrentPlayer) {
			s.mu.Unlock()
			writeError(w, http.StatusConflict, fmt.Sprintf("out of turn: it is player %d's (%s) turn, not player %d",
				sess.game.CurrentPlayer, sess.game.CurrentPlayer.String(), req.Player))
			return
		}
	}

	rec, err := sess.game.ApplyMove(action)
	if err != nil {
		s.mu.Unlock()
		if errors.Is(err, engine.ErrColumnFull) || errors.Is(err, engine.ErrInvalidAction) || errors.Is(err, engine.ErrInvalidCoords) {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
		writeError(w, http.StatusConflict, err.Error())
		return
	}

	lastMoveDTO := CoordDTO{
		Action:     rec.Action,
		X:          rec.X,
		Y:          rec.Y,
		Z:          rec.Z,
		Player:     uint8(rec.Player),
		MoveNumber: rec.MoveNumber,
	}

	autoReply := true
	if req.AutoReply != nil {
		autoReply = *req.AutoReply
	}
	oppName := reqOppName
	if oppName == "" {
		oppName = sess.opponentAgent
	}

	var aiMoveDTO *CoordDTO
	if autoReply && oppName != "" && sess.game.Status == engine.StatusOngoing {
		ag := reqOppAgent
		if ag == nil {
			ag, _ = agents.NewAgentByName(oppName, int64(sess.game.MoveCount)+1)
		}
		if ag != nil {
			if aiAct, aiErr := ag.SelectAction(sess.game.Clone()); aiErr == nil {
				if aiRec, applyErr := sess.game.ApplyMove(aiAct); applyErr == nil {
					aiMoveDTO = &CoordDTO{
						Action:     aiRec.Action,
						X:          aiRec.X,
						Y:          aiRec.Y,
						Z:          aiRec.Z,
						Player:     uint8(aiRec.Player),
						MoveNumber: aiRec.MoveNumber,
					}
				}
			}
		}
	}

	var reward float32
	done := sess.game.Status != engine.StatusOngoing
	if sess.game.Winner != engine.PlayerNone {
		if sess.game.Winner == rec.Player {
			reward = 1.0
		} else {
			reward = -1.0
		}
	}

	resp := MoveResponseDTO{
		State:    buildStateDTO(sess),
		LastMove: lastMoveDTO,
		AIMove:   aiMoveDTO,
		Reward:   reward,
		Done:     done,
	}
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, resp)
}

func (s *ArenaServer) handlePreset(w http.ResponseWriter, r *http.Request, gameID string) {
	var req PresetRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON preset request")
		return
	}

	var moves []uint8
	switch req.Type {
	case "draw":
		seq := engine.GenerateDrawSequence()
		moves = seq[:]
	case "win":
		if req.LineIndex < 0 || req.LineIndex >= engine.TotalWinLines {
			writeError(w, http.StatusBadRequest, "line_index must be in 0..75")
			return
		}
		winner := engine.PlayerWhite
		if req.Winner == uint8(engine.PlayerBlack) {
			winner = engine.PlayerBlack
		}
		moves = engine.GenerateWinSequence(req.LineIndex, winner)
	default:
		writeError(w, http.StatusBadRequest, fmt.Sprintf("unknown preset type %q (use 'draw' or 'win')", req.Type))
		return
	}

	s.mu.Lock()
	sess, exists := s.sessions[gameID]
	if !exists {
		sess = &gameSession{
			id:        gameID,
			name:      formatMatchName(gameID, false),
			whiteName: "Player 1 (White)",
			blackName: "Player 2 (Black)",
			game:      engine.NewGame(),
		}
		s.sessions[gameID] = sess
	} else {
		sess.game.Reset()
	}
	for _, a := range moves {
		_, _ = sess.game.ApplyMove(a)
	}
	dto := buildStateDTO(sess)
	s.mu.Unlock()

	writeJSON(w, http.StatusOK, dto)
}

func (s *ArenaServer) handleRunMatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var req RunMatchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON match request")
		return
	}

	whiteURL := strings.TrimSpace(req.WhiteBotURL)
	blackURL := strings.TrimSpace(req.BlackBotURL)
	whiteAgentName := strings.TrimSpace(req.WhiteAgent)
	blackAgentName := strings.TrimSpace(req.BlackAgent)

	if (whiteURL == "" && whiteAgentName == "") || (blackURL == "" && blackAgentName == "") {
		writeError(w, http.StatusBadRequest, "each side requires either a bot_url or an agent name")
		return
	}

	if whiteAgentName != "" {
		if _, err := agents.NewAgentByName(whiteAgentName, 1); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	}
	if blackAgentName != "" {
		if _, err := agents.NewAgentByName(blackAgentName, 2); err != nil {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
	}

	gameID := strings.TrimSpace(req.GameID)
	if gameID == "" {
		seq := atomic.AddUint64(&s.seqCounter, 1)
		gameID = fmt.Sprintf("match-%d", seq)
	}
	matchName := strings.TrimSpace(req.Name)
	if matchName == "" {
		matchName = gameID
	}
	whiteName := strings.TrimSpace(req.WhiteName)
	if whiteName == "" {
		if whiteAgentName != "" {
			whiteName = fmt.Sprintf("AI %s (White)", whiteAgentName)
		} else {
			whiteName = "Remote Bot (White)"
		}
	}
	blackName := strings.TrimSpace(req.BlackName)
	if blackName == "" {
		if blackAgentName != "" {
			blackName = fmt.Sprintf("AI %s (Black)", blackAgentName)
		} else {
			blackName = "Remote Bot (Black)"
		}
	}
	serverStartsWhite := (whiteAgentName != "" && whiteURL == "") || isServerWhitePlayer(whiteName, whiteAgentName, 0)
	matchName = formatMatchName(matchName, serverStartsWhite)

	sess := &gameSession{
		id:        gameID,
		name:      matchName,
		whiteName: whiteName,
		blackName: blackName,
		game:      engine.NewGame(),
	}

	s.mu.Lock()
	s.sessions[gameID] = sess
	s.mu.Unlock()

	timeout := 5 * time.Second
	if req.TimeoutMs > 0 {
		timeout = time.Duration(req.TimeoutMs) * time.Millisecond
	}

	for sess.game.Status == engine.StatusOngoing {
		s.mu.RLock()
		statePayload := buildStateDTO(sess)
		snapshot := sess.game.Clone()
		activePlayer := sess.game.CurrentPlayer
		s.mu.RUnlock()

		botURL := whiteURL
		agentName := whiteAgentName
		if activePlayer == engine.PlayerBlack {
			botURL = blackURL
			agentName = blackAgentName
		}

		var action uint8
		if agentName != "" && botURL == "" {
			ag, _ := agents.NewAgentByName(agentName, int64(snapshot.MoveCount)+1)
			act, err := ag.SelectAction(snapshot)
			if err != nil {
				writeError(w, http.StatusConflict, err.Error())
				return
			}
			action = act
		} else {
			bodyBytes, _ := json.Marshal(statePayload)
			ctx, cancel := context.WithTimeout(r.Context(), timeout)
			botReq, err := http.NewRequestWithContext(ctx, http.MethodPost, botURL, bytes.NewReader(bodyBytes))
			if err != nil {
				cancel()
				writeError(w, http.StatusBadGateway, fmt.Sprintf("invalid bot URL %q: %v", botURL, err))
				return
			}
			botReq.Header.Set("Content-Type", "application/json")

			botResp, err := s.httpClient.Do(botReq)
			if err != nil {
				cancel()
				writeError(w, http.StatusBadGateway, fmt.Sprintf("bot %q request failed: %v", botURL, err))
				return
			}
			var moveReq MoveRequest
			decodeErr := json.NewDecoder(botResp.Body).Decode(&moveReq)
			_ = botResp.Body.Close()
			cancel()

			if botResp.StatusCode != http.StatusOK || decodeErr != nil {
				writeError(w, http.StatusBadGateway, fmt.Sprintf("bot %q returned status %d (err=%v)", botURL, botResp.StatusCode, decodeErr))
				return
			}

			resolvedAct, err := resolveMoveAction(moveReq)
			if err != nil {
				writeError(w, http.StatusBadGateway, fmt.Sprintf("bot %q returned invalid move: %v", botURL, err))
				return
			}
			action = resolvedAct
		}

		s.mu.Lock()
		_, moveErr := sess.game.ApplyMove(action)
		s.mu.Unlock()
		if moveErr != nil {
			writeError(w, http.StatusBadGateway, fmt.Sprintf("bot attempted illegal move %d: %v", action, moveErr))
			return
		}
	}

	s.mu.RLock()
	finalDTO := buildStateDTO(sess)
	s.mu.RUnlock()
	writeJSON(w, http.StatusOK, finalDTO)
}

// StartTLSServer starts an HTTPS server on `addr` (e.g. "127.0.0.1:0" or ":8443") using an auto-generated
// ECDSA P-256 self-signed certificate, returning the base `https://host:port` URL, a preconfigured TLS
// `*http.Client` trusting that certificate, and a shutdown function.
func StartTLSServer(addr string) (*ArenaServer, string, *http.Client, func(), error) {
	tlsCert, x509Cert, err := GenerateSelfSignedTLS()
	if err != nil {
		return nil, "", nil, nil, err
	}

	certPool := x509.NewCertPool()
	certPool.AddCert(x509Cert)
	trustedClient := &http.Client{
		Timeout: 10 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{
				RootCAs:    certPool,
				MinVersion: tls.VersionTLS12,
			},
		},
	}

	arena := NewArenaServer(trustedClient)
	tlsCfg := &tls.Config{
		Certificates: []tls.Certificate{tlsCert},
		MinVersion:   tls.VersionTLS12,
	}

	rawLis, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, "", nil, nil, err
	}
	tlsLis := tls.NewListener(rawLis, tlsCfg)

	httpSrv := &http.Server{
		Handler:           arena.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		_ = httpSrv.Serve(tlsLis)
	}()

	baseURL := "https://" + rawLis.Addr().String()
	cleanup := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(ctx)
		_ = rawLis.Close()
	}

	return arena, baseURL, trustedClient, cleanup, nil
}

// StartHTTPServer starts a plain HTTP server on `addr` (e.g. ":8080") for deployment behind a
// TLS-terminating Kubernetes Ingress controller (such as nginx-ingress with cert-manager).
func StartHTTPServer(addr string) (*ArenaServer, string, func(), error) {
	arena := NewArenaServer(nil)
	rawLis, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, "", nil, err
	}

	httpSrv := &http.Server{
		Handler:           arena.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		_ = httpSrv.Serve(rawLis)
	}()

	baseURL := "http://" + rawLis.Addr().String()
	cleanup := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(ctx)
		_ = rawLis.Close()
	}

	return arena, baseURL, cleanup, nil
}
