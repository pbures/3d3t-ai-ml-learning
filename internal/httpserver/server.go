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
		ID:          "ttt3d-deep-gray1",
		Name:        "TTT3D Deep Gray 1 (19-Ch 3D ResNet + Deep PVS Lookahead)",
		Type:        "neural_hybrid",
		Description: "Newly trained 19-channel 3D Residual CNN with multi-ply gravity & stacked-fork stem trained against 15+ ply search, paired with check-extended PVS tactical lookahead.",
	},
	{
		ID:          "ttt3d-deep-gray1-pure",
		Name:        "TTT3D Deep Gray 1 Pure (Raw 19-Ch 3D ResNet Forward Pass)",
		Type:        "neural_pure",
		Description: "Single-pass 19-channel 3D Residual CNN trained against 15+ ply search, selecting argmax over legal column policy probabilities with zero tree search.",
	},
	{
		ID:          "deep",
		Name:        "Deep PVS + Exact Endgame Solver Baseline (~2.8s / Move)",
		Type:        "search",
		Description: "Iterative-deepening Principal Variation Search (Zobrist TT, logarithmic LMR with full re-search, killer/history ordering, check extensions, zugzwang-parity evaluation) combined with an exact win/draw/loss endgame solver from ply 20 that plays proven wins and avoids proven losses (~2.8s/move).",
	},
	{
		ID:          "ttt3dnet",
		Name:        "TTT3DNet Hybrid (3D ResNet + Tactical Shield)",
		Type:        "neural_hybrid",
		Description: "3D Residual CNN (Policy + Value dual head) combined with a 64-bit bitboard 2-ply fork shield and multi-ply value lookahead.",
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
	BattleID        string         `json:"battle_id"`
	BattleName      string         `json:"battle_name,omitempty"`
	Name            string         `json:"name"`
	Participant     string         `json:"participant"`
	ClientName      string         `json:"client_name"`
	ClientPlayer    uint8          `json:"client_player"`
	ServerPlayer    uint8          `json:"server_player"`
	WinnerSide      string         `json:"winner_side"`
	WinnerLabel     string         `json:"winner_label"`
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
	BattleID      string `json:"battle_id"`
	BattleName    string `json:"battle_name,omitempty"`
	Name          string `json:"name"`
	Participant   string `json:"participant"`
	ClientName    string `json:"client_name"`
	ClientPlayer  uint8  `json:"client_player"`
	ServerPlayer  uint8  `json:"server_player"`
	WinnerSide    string `json:"winner_side"`
	WinnerLabel   string `json:"winner_label"`
	WhiteName     string `json:"white_name"`
	BlackName     string `json:"black_name"`
	OpponentAgent string `json:"opponent_agent,omitempty"`
	Status        string `json:"status"`
	Winner        uint8  `json:"winner"`
	MoveCount     uint8  `json:"move_count"`
	CurrentPlayer uint8  `json:"current_player"`
}

// BattleSummaryDTO describes a battle (series of games) with its name, date, participant (client), winner, and statistics.
type BattleSummaryDTO struct {
	BattleID        string   `json:"battle_id"`
	Name            string   `json:"name"`
	Date            string   `json:"date"`
	Participant     string   `json:"participant"`
	ClientName      string   `json:"client_name"`
	OpponentAgent   string   `json:"opponent_agent,omitempty"`
	TotalGames      int      `json:"total_games"`
	CompletedGames  int      `json:"completed_games"`
	OngoingGames    int      `json:"ongoing_games"`
	ClientWins      int      `json:"client_wins"`
	ServerWins      int      `json:"server_wins"`
	Draws           int      `json:"draws"`
	ClientWinRate   float64  `json:"client_win_rate"`
	ServerWinRate   float64  `json:"server_win_rate"`
	ClientScoreRate float64  `json:"client_score_rate"`
	WinnerSide      string   `json:"winner_side"`
	WinnerLabel     string   `json:"winner_label"`
	GameIDs         []string `json:"game_ids"`
}

// BattleDetailDTO returns the battle metadata, statistics, winner, and all games belonging to the battle.
type BattleDetailDTO struct {
	BattleSummaryDTO
	Games []GameSummaryDTO `json:"games"`
}

// CreateBattleRequest is the request body for `POST /api/v1/battles`.
type CreateBattleRequest struct {
	BattleID      string `json:"battle_id,omitempty"`
	Name          string `json:"name,omitempty"`
	Date          string `json:"date,omitempty"`
	Participant   string `json:"participant,omitempty"`
	ClientName    string `json:"client_name,omitempty"`
	PlayerName    string `json:"player_name,omitempty"`
	OpponentAgent string `json:"opponent_agent,omitempty"`
}

// UpdateBattleRequest is the request body for `PATCH /api/v1/battles/{id}`.
type UpdateBattleRequest struct {
	Name          *string `json:"name,omitempty"`
	Date          *string `json:"date,omitempty"`
	Participant   *string `json:"participant,omitempty"`
	ClientName    *string `json:"client_name,omitempty"`
	OpponentAgent *string `json:"opponent_agent,omitempty"`
}

// CreateGameRequest is the optional request body for `POST /api/v1/games` and `POST /api/v1/battles/{id}/games`.
type CreateGameRequest struct {
	GameID        string `json:"game_id"`
	BattleID      string `json:"battle_id,omitempty"`
	Name          string `json:"name"`
	Participant   string `json:"participant,omitempty"`
	ClientName    string `json:"client_name,omitempty"`
	WhiteName     string `json:"white_name"`
	BlackName     string `json:"black_name"`
	OpponentAgent string `json:"opponent_agent,omitempty"`
	HumanPlayer   uint8  `json:"human_player,omitempty"`
	ClientPlayer  uint8  `json:"client_player,omitempty"`
}

// UpdateGameRequest is the request body for `PATCH /api/v1/games/{id}` to rename a match or configure an AI opponent.
type UpdateGameRequest struct {
	BattleID      *string `json:"battle_id,omitempty"`
	Name          *string `json:"name,omitempty"`
	Participant   *string `json:"participant,omitempty"`
	ClientName    *string `json:"client_name,omitempty"`
	ClientPlayer  *uint8  `json:"client_player,omitempty"`
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
	GameID       string `json:"game_id"`
	BattleID     string `json:"battle_id,omitempty"`
	Name         string `json:"name,omitempty"`
	Participant  string `json:"participant,omitempty"`
	ClientName   string `json:"client_name,omitempty"`
	ClientPlayer uint8  `json:"client_player,omitempty"`
	WhiteName    string `json:"white_name"`
	BlackName    string `json:"black_name"`
	WhiteBotURL  string `json:"white_bot_url,omitempty"`
	BlackBotURL  string `json:"black_bot_url,omitempty"`
	WhiteAgent   string `json:"white_agent,omitempty"`
	BlackAgent   string `json:"black_agent,omitempty"`
	TimeoutMs    int    `json:"timeout_ms,omitempty"`
}

// WinLineDTO describes one of the 76 winning lines for `GET /api/v1/winlines`.
type WinLineDTO struct {
	Index    int        `json:"index"`
	Category string     `json:"category"`
	Mask     uint64     `json:"mask"`
	Coords   []CoordDTO `json:"coords"`
}

type battleSession struct {
	id            string
	name          string
	date          string
	participant   string
	opponentAgent string
	seqOrder      uint64
}

type gameSession struct {
	id            string
	battleID      string
	name          string
	participant   string
	clientPlayer  uint8
	whiteName     string
	blackName     string
	opponentAgent string
	game          *engine.Game
}

// ArenaServer manages HTTP/HTTPS REST battle endpoints, remote bot matchmaking, and the 3D Web Replay UI.
type ArenaServer struct {
	mu         sync.RWMutex
	battles    map[string]*battleSession
	sessions   map[string]*gameSession
	seqCounter uint64
	battleSeq  uint64
	httpClient *http.Client
}

// NewArenaServer creates and initializes an ArenaServer with a "default" battle and "default" game session.
func NewArenaServer(botHTTPClient *http.Client) *ArenaServer {
	if botHTTPClient == nil {
		botHTTPClient = &http.Client{Timeout: 5 * time.Second}
	}
	nowStr := time.Now().UTC().Format(time.RFC3339)
	s := &ArenaServer{
		battles:    make(map[string]*battleSession),
		sessions:   make(map[string]*gameSession),
		httpClient: botHTTPClient,
	}
	s.battles["default"] = &battleSession{
		id:          "default",
		name:        "Default Arena Battle",
		date:        nowStr,
		participant: "Client Player",
		seqOrder:    0,
	}
	s.sessions["default"] = &gameSession{
		id:           "default",
		battleID:     "default",
		name:         "CW - default",
		participant:  "Client Player",
		clientPlayer: 1,
		whiteName:    "Player 1 (White)",
		blackName:    "Player 2 (Black)",
		game:         engine.NewGame(),
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

func resolveGameWinnerInfo(sess *gameSession, b *battleSession) (battleID, battleName, participant string, clientPlayer, serverPlayer uint8, winnerSide, winnerLabel string) {
	battleID = sess.battleID
	if battleID == "" {
		battleID = "default"
	}
	participant = strings.TrimSpace(sess.participant)
	if b != nil {
		battleName = b.name
		if participant == "" {
			participant = b.participant
		}
	}
	if participant == "" {
		participant = "Client Player"
	}

	clientPlayer = sess.clientPlayer
	if clientPlayer != 1 && clientPlayer != 2 {
		if isServerWhitePlayer(sess.whiteName, sess.opponentAgent, 0) {
			clientPlayer = 2
		} else {
			clientPlayer = 1
		}
	}
	serverPlayer = 3 - clientPlayer

	switch sess.game.Status {
	case engine.StatusOngoing:
		winnerSide = "ONGOING"
		winnerLabel = "ONGOING"
	case engine.StatusDraw:
		winnerSide = "DRAW"
		winnerLabel = "DRAW"
	case engine.StatusWhiteWon, engine.StatusBlackWon:
		if uint8(sess.game.Winner) == clientPlayer {
			winnerSide = "CLIENT"
			winnerLabel = fmt.Sprintf("CLIENT (%s)", participant)
		} else {
			winnerSide = "SERVER"
			if sess.opponentAgent != "" {
				winnerLabel = fmt.Sprintf("SERVER (%s)", sess.opponentAgent)
			} else {
				winnerLabel = "SERVER"
			}
		}
	default:
		winnerSide = "ONGOING"
		winnerLabel = "ONGOING"
	}
	return
}

func (s *ArenaServer) buildGameSummaryLocked(sess *gameSession) GameSummaryDTO {
	matchName := sess.name
	if matchName == "" {
		matchName = sess.id
	}
	b := s.battles[sess.battleID]
	if b == nil {
		b = s.battles["default"]
	}
	battleID, battleName, participant, clientPlayer, serverPlayer, winnerSide, winnerLabel := resolveGameWinnerInfo(sess, b)
	return GameSummaryDTO{
		GameID:        sess.id,
		BattleID:      battleID,
		BattleName:    battleName,
		Name:          matchName,
		Participant:   participant,
		ClientName:    participant,
		ClientPlayer:  clientPlayer,
		ServerPlayer:  serverPlayer,
		WinnerSide:    winnerSide,
		WinnerLabel:   winnerLabel,
		WhiteName:     sess.whiteName,
		BlackName:     sess.blackName,
		OpponentAgent: sess.opponentAgent,
		Status:        sess.game.Status.String(),
		Winner:        uint8(sess.game.Winner),
		MoveCount:     sess.game.MoveCount,
		CurrentPlayer: uint8(sess.game.CurrentPlayer),
	}
}

func (s *ArenaServer) buildBattleDetailLocked(b *battleSession) BattleDetailDTO {
	games := make([]GameSummaryDTO, 0)
	for _, sess := range s.sessions {
		sessBattle := sess.battleID
		if sessBattle == "" {
			sessBattle = "default"
		}
		if sessBattle == b.id {
			games = append(games, s.buildGameSummaryLocked(sess))
		}
	}
	sort.Slice(games, func(i, j int) bool {
		return games[i].GameID < games[j].GameID
	})

	gameIDs := make([]string, len(games))
	var completed, ongoing, clientWins, serverWins, draws int
	for i, g := range games {
		gameIDs[i] = g.GameID
		switch g.WinnerSide {
		case "CLIENT":
			clientWins++
			completed++
		case "SERVER":
			serverWins++
			completed++
		case "DRAW":
			draws++
			completed++
		default:
			ongoing++
		}
	}

	var clientWinRate, serverWinRate, clientScoreRate float64
	if completed > 0 {
		clientWinRate = float64(clientWins) / float64(completed)
		serverWinRate = float64(serverWins) / float64(completed)
		clientScoreRate = (float64(clientWins) + 0.5*float64(draws)) / float64(completed)
	}

	participant := b.participant
	if participant == "" {
		participant = "Client Player"
	}

	winnerSide := "ONGOING"
	winnerLabel := "ONGOING"
	if completed > 0 {
		if clientWins > serverWins {
			winnerSide = "CLIENT"
			winnerLabel = fmt.Sprintf("CLIENT (%s)", participant)
		} else if serverWins > clientWins {
			winnerSide = "SERVER"
			if b.opponentAgent != "" {
				winnerLabel = fmt.Sprintf("SERVER (%s)", b.opponentAgent)
			} else {
				winnerLabel = "SERVER"
			}
		} else {
			winnerSide = "TIE"
			winnerLabel = "TIE"
		}
	}

	summary := BattleSummaryDTO{
		BattleID:        b.id,
		Name:            b.name,
		Date:            b.date,
		Participant:     participant,
		ClientName:      participant,
		OpponentAgent:   b.opponentAgent,
		TotalGames:      len(games),
		CompletedGames:  completed,
		OngoingGames:    ongoing,
		ClientWins:      clientWins,
		ServerWins:      serverWins,
		Draws:           draws,
		ClientWinRate:   clientWinRate,
		ServerWinRate:   serverWinRate,
		ClientScoreRate: clientScoreRate,
		WinnerSide:      winnerSide,
		WinnerLabel:     winnerLabel,
		GameIDs:         gameIDs,
	}

	return BattleDetailDTO{
		BattleSummaryDTO: summary,
		Games:            games,
	}
}

func (s *ArenaServer) ensureBattleLocked(battleID, participant, oppAgent string) *battleSession {
	if battleID == "" {
		battleID = "default"
	}
	if b, exists := s.battles[battleID]; exists {
		return b
	}
	part := strings.TrimSpace(participant)
	if part == "" {
		part = "Client Player"
	}
	seq := atomic.AddUint64(&s.battleSeq, 1)
	b := &battleSession{
		id:            battleID,
		name:          battleID,
		date:          time.Now().UTC().Format(time.RFC3339),
		participant:   part,
		opponentAgent: oppAgent,
		seqOrder:      seq,
	}
	s.battles[battleID] = b
	return b
}

func buildStateDTOWithBattle(sess *gameSession, b *battleSession) GameStateDTO {
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
	for i, bVal := range tensor {
		tensorSlice[i] = int(bVal)
	}

	matchName := sess.name
	if matchName == "" {
		matchName = sess.id
	}

	battleID, battleName, participant, clientPlayer, serverPlayer, winnerSide, winnerLabel := resolveGameWinnerInfo(sess, b)

	dto := GameStateDTO{
		GameID:          sess.id,
		BattleID:        battleID,
		BattleName:      battleName,
		Name:            matchName,
		Participant:     participant,
		ClientName:      participant,
		ClientPlayer:    clientPlayer,
		ServerPlayer:    serverPlayer,
		WinnerSide:      winnerSide,
		WinnerLabel:     winnerLabel,
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

func (s *ArenaServer) buildStateLocked(sess *gameSession) GameStateDTO {
	b := s.battles[sess.battleID]
	if b == nil {
		b = s.battles["default"]
	}
	return buildStateDTOWithBattle(sess, b)
}

func buildStateDTO(sess *gameSession) GameStateDTO {
	return buildStateDTOWithBattle(sess, nil)
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
	mux.HandleFunc("/api/v1/battles", s.handleBattlesRoot)
	mux.HandleFunc("/api/v1/battles/", s.handleBattleByID)
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

func (s *ArenaServer) handleBattlesRoot(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.mu.RLock()
		battleList := make([]*battleSession, 0, len(s.battles))
		for _, b := range s.battles {
			battleList = append(battleList, b)
		}
		sort.Slice(battleList, func(i, j int) bool {
			if battleList[i].seqOrder != battleList[j].seqOrder {
				return battleList[i].seqOrder < battleList[j].seqOrder
			}
			return battleList[i].id < battleList[j].id
		})
		summaries := make([]BattleSummaryDTO, len(battleList))
		for i, b := range battleList {
			summaries[i] = s.buildBattleDetailLocked(b).BattleSummaryDTO
		}
		s.mu.RUnlock()
		writeJSON(w, http.StatusOK, map[string]interface{}{"battles": summaries})

	case http.MethodPost:
		var req CreateBattleRequest
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
		if oppAgentName != "" {
			ag, err := agents.NewAgentByName(oppAgentName, 1)
			if err != nil {
				writeError(w, http.StatusBadRequest, err.Error())
				return
			}
			oppAgentName = ag.Name()
		}

		participant := strings.TrimSpace(req.Participant)
		if participant == "" {
			participant = strings.TrimSpace(req.ClientName)
		}
		if participant == "" {
			participant = strings.TrimSpace(req.PlayerName)
		}
		if participant == "" {
			participant = "Client Player"
		}

		battleID := strings.TrimSpace(req.BattleID)
		seq := atomic.AddUint64(&s.battleSeq, 1)
		if battleID == "" {
			battleID = fmt.Sprintf("battle-%d", seq)
		}

		battleName := strings.TrimSpace(req.Name)
		if battleName == "" {
			if participant != "Client Player" {
				battleName = fmt.Sprintf("%s Battle", participant)
			} else {
				battleName = battleID
			}
		}

		battleDate := strings.TrimSpace(req.Date)
		if battleDate == "" {
			battleDate = time.Now().UTC().Format(time.RFC3339)
		}

		s.mu.Lock()
		b := &battleSession{
			id:            battleID,
			name:          battleName,
			date:          battleDate,
			participant:   participant,
			opponentAgent: oppAgentName,
			seqOrder:      seq,
		}
		s.battles[battleID] = b
		dto := s.buildBattleDetailLocked(b)
		s.mu.Unlock()

		writeJSON(w, http.StatusCreated, dto)

	default:
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
	}
}

func (s *ArenaServer) handleBattleByID(w http.ResponseWriter, r *http.Request) {
	trimmed := strings.TrimPrefix(r.URL.Path, "/api/v1/battles/")
	parts := strings.Split(strings.Trim(trimmed, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		writeError(w, http.StatusBadRequest, "missing battle_id")
		return
	}
	battleID := parts[0]
	subAction := ""
	if len(parts) > 1 {
		subAction = parts[1]
	}

	switch subAction {
	case "":
		switch r.Method {
		case http.MethodGet:
			s.mu.RLock()
			b, exists := s.battles[battleID]
			if !exists {
				s.mu.RUnlock()
				writeError(w, http.StatusNotFound, fmt.Sprintf("battle %q not found", battleID))
				return
			}
			dto := s.buildBattleDetailLocked(b)
			s.mu.RUnlock()
			writeJSON(w, http.StatusOK, dto)

		case http.MethodPatch:
			s.handleUpdateBattle(w, r, battleID)

		default:
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		}

	case "games":
		switch r.Method {
		case http.MethodGet:
			s.mu.RLock()
			b, exists := s.battles[battleID]
			if !exists {
				s.mu.RUnlock()
				writeError(w, http.StatusNotFound, fmt.Sprintf("battle %q not found", battleID))
				return
			}
			dto := s.buildBattleDetailLocked(b)
			s.mu.RUnlock()
			writeJSON(w, http.StatusOK, dto)

		case http.MethodPost:
			s.mu.RLock()
			_, exists := s.battles[battleID]
			s.mu.RUnlock()
			if !exists {
				writeError(w, http.StatusNotFound, fmt.Sprintf("battle %q not found", battleID))
				return
			}
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
			req.BattleID = battleID
			s.createGameFromRequest(w, req)

		default:
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		}

	default:
		writeError(w, http.StatusNotFound, fmt.Sprintf("unknown battle sub-route %q", subAction))
	}
}

func (s *ArenaServer) handleUpdateBattle(w http.ResponseWriter, r *http.Request, battleID string) {
	var req UpdateBattleRequest
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
	b, exists := s.battles[battleID]
	if !exists {
		s.mu.Unlock()
		writeError(w, http.StatusNotFound, fmt.Sprintf("battle %q not found", battleID))
		return
	}
	if req.Name != nil {
		n := strings.TrimSpace(*req.Name)
		if n != "" {
			b.name = n
		}
	}
	if req.Date != nil {
		d := strings.TrimSpace(*req.Date)
		if d != "" {
			b.date = d
		}
	}
	newParticipant := ""
	if req.Participant != nil {
		newParticipant = strings.TrimSpace(*req.Participant)
	} else if req.ClientName != nil {
		newParticipant = strings.TrimSpace(*req.ClientName)
	}
	if newParticipant != "" {
		b.participant = newParticipant
		for _, sess := range s.sessions {
			if sess.battleID == battleID {
				sess.participant = newParticipant
			}
		}
	}
	if hasOppUpdate {
		b.opponentAgent = validatedOpp
	}
	dto := s.buildBattleDetailLocked(b)
	s.mu.Unlock()

	writeJSON(w, http.StatusOK, dto)
}

func (s *ArenaServer) createGameFromRequest(w http.ResponseWriter, req CreateGameRequest) {
	battleID := strings.TrimSpace(req.BattleID)
	if battleID == "" {
		battleID = "default"
	}

	participant := strings.TrimSpace(req.Participant)
	if participant == "" {
		participant = strings.TrimSpace(req.ClientName)
	}

	s.mu.RLock()
	existingBattle := s.battles[battleID]
	var battleParticipant, battleOpp string
	if existingBattle != nil {
		battleParticipant = existingBattle.participant
		battleOpp = existingBattle.opponentAgent
	}
	s.mu.RUnlock()

	if participant == "" {
		if battleParticipant != "" {
			participant = battleParticipant
		} else {
			participant = "Client Player"
		}
	}

	oppAgentName := strings.TrimSpace(req.OpponentAgent)
	if oppAgentName == "" && battleOpp != "" {
		oppAgentName = battleOpp
	}

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

	humanSide := req.HumanPlayer
	if humanSide == 0 && (req.ClientPlayer == 1 || req.ClientPlayer == 2) {
		humanSide = req.ClientPlayer
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

	hasCustomParticipant := participant != "" && participant != "Client Player"
	whiteName := strings.TrimSpace(req.WhiteName)
	if whiteName == "" {
		if oppAgent != nil && humanSide == uint8(engine.PlayerBlack) {
			whiteName = fmt.Sprintf("AI %s (White)", oppAgentName)
		} else if hasCustomParticipant {
			whiteName = fmt.Sprintf("%s (White)", participant)
		} else {
			whiteName = "Player 1 (White)"
		}
	}
	blackName := strings.TrimSpace(req.BlackName)
	if blackName == "" {
		if oppAgent != nil && humanSide != uint8(engine.PlayerBlack) {
			blackName = fmt.Sprintf("AI %s (Black)", oppAgentName)
		} else if hasCustomParticipant && humanSide == uint8(engine.PlayerBlack) {
			blackName = fmt.Sprintf("%s (Black)", participant)
		} else {
			blackName = "Player 2 (Black)"
		}
	}
	serverStartsWhite := isServerWhitePlayer(whiteName, oppAgentName, humanSide)
	matchName = formatMatchName(matchName, serverStartsWhite)

	clientPlayer := uint8(1)
	if req.ClientPlayer == 1 || req.ClientPlayer == 2 {
		clientPlayer = req.ClientPlayer
	} else if humanSide == 1 || humanSide == 2 {
		clientPlayer = humanSide
	} else if serverStartsWhite {
		clientPlayer = 2
	}

	g := engine.NewGame()
	// If external player chose Black (2) and an opponent model is configured, let the AI make White's opening move
	if oppAgent != nil && humanSide == uint8(engine.PlayerBlack) {
		if act, err := oppAgent.SelectAction(g); err == nil {
			_, _ = g.ApplyMove(act)
		}
	}

	s.mu.Lock()
	b := s.ensureBattleLocked(battleID, participant, oppAgentName)
	sess := &gameSession{
		id:            gameID,
		battleID:      b.id,
		name:          matchName,
		participant:   participant,
		clientPlayer:  clientPlayer,
		whiteName:     whiteName,
		blackName:     blackName,
		opponentAgent: oppAgentName,
		game:          g,
	}
	s.sessions[gameID] = sess
	dto := s.buildStateLocked(sess)
	s.mu.Unlock()

	writeJSON(w, http.StatusCreated, dto)
}

func (s *ArenaServer) handleGamesRoot(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		battleFilter := strings.TrimSpace(r.URL.Query().Get("battle_id"))
		s.mu.RLock()
		summaries := make([]GameSummaryDTO, 0, len(s.sessions))
		for _, sess := range s.sessions {
			sessBattle := sess.battleID
			if sessBattle == "" {
				sessBattle = "default"
			}
			if battleFilter != "" && sessBattle != battleFilter {
				continue
			}
			summaries = append(summaries, s.buildGameSummaryLocked(sess))
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
		s.createGameFromRequest(w, req)

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
			dto := s.buildStateLocked(sess)
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
				id:           gameID,
				battleID:     "default",
				name:         formatMatchName(gameID, false),
				participant:  "Client Player",
				clientPlayer: 1,
				whiteName:    "Player 1 (White)",
				blackName:    "Player 2 (Black)",
				game:         engine.NewGame(),
			}
			s.sessions[gameID] = sess
		} else {
			sess.game.Reset()
		}
		dto := s.buildStateLocked(sess)
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
	if req.BattleID != nil {
		bid := strings.TrimSpace(*req.BattleID)
		if bid != "" {
			b := s.ensureBattleLocked(bid, sess.participant, sess.opponentAgent)
			sess.battleID = b.id
		}
	}
	if req.Participant != nil {
		p := strings.TrimSpace(*req.Participant)
		if p != "" {
			sess.participant = p
		}
	} else if req.ClientName != nil {
		p := strings.TrimSpace(*req.ClientName)
		if p != "" {
			sess.participant = p
		}
	}
	if req.ClientPlayer != nil && (*req.ClientPlayer == 1 || *req.ClientPlayer == 2) {
		sess.clientPlayer = *req.ClientPlayer
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
		if req.ClientPlayer == nil {
			if serverStartsWhite {
				sess.clientPlayer = 2
			} else {
				sess.clientPlayer = 1
			}
		}
	}
	dto := s.buildStateLocked(sess)
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
		State: s.buildStateLocked(sess),
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
		State:    s.buildStateLocked(sess),
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
			id:           gameID,
			battleID:     "default",
			name:         formatMatchName(gameID, false),
			participant:  "Client Player",
			clientPlayer: 1,
			whiteName:    "Player 1 (White)",
			blackName:    "Player 2 (Black)",
			game:         engine.NewGame(),
		}
		s.sessions[gameID] = sess
	} else {
		sess.game.Reset()
	}
	for _, a := range moves {
		_, _ = sess.game.ApplyMove(a)
	}
	dto := s.buildStateLocked(sess)
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

	battleID := strings.TrimSpace(req.BattleID)
	if battleID == "" {
		battleID = "default"
	}
	participant := strings.TrimSpace(req.Participant)
	if participant == "" {
		participant = strings.TrimSpace(req.ClientName)
	}

	clientPlayer := uint8(1)
	if req.ClientPlayer == 1 || req.ClientPlayer == 2 {
		clientPlayer = req.ClientPlayer
	} else if serverStartsWhite {
		clientPlayer = 2
	}

	oppAgent := blackAgentName
	if serverStartsWhite && whiteAgentName != "" {
		oppAgent = whiteAgentName
	}

	s.mu.Lock()
	b := s.ensureBattleLocked(battleID, participant, oppAgent)
	if participant == "" {
		participant = b.participant
	}
	sess := &gameSession{
		id:            gameID,
		battleID:      b.id,
		name:          matchName,
		participant:   participant,
		clientPlayer:  clientPlayer,
		whiteName:     whiteName,
		blackName:     blackName,
		opponentAgent: oppAgent,
		game:          engine.NewGame(),
	}
	s.sessions[gameID] = sess
	s.mu.Unlock()

	timeout := 5 * time.Second
	if req.TimeoutMs > 0 {
		timeout = time.Duration(req.TimeoutMs) * time.Millisecond
	}

	for sess.game.Status == engine.StatusOngoing {
		s.mu.RLock()
		statePayload := s.buildStateLocked(sess)
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
	finalDTO := s.buildStateLocked(sess)
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
