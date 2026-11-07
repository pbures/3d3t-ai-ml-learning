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

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

//go:embed web/index.html
var webAssets embed.FS

// CoordDTO represents a 3D coordinate or move record in JSON payloads.
type CoordDTO struct {
	Action     uint8  `json:"action"`
	X          uint8  `json:"x"`
	Y          uint8  `json:"y"`
	Z          uint8  `json:"z"`
	Player     uint8  `json:"player,omitempty"`
	MoveNumber uint8  `json:"move_number,omitempty"`
}

// GameStateDTO is the full JSON representation of a 3T3D game session for remote bots and the 3D Web UI.
type GameStateDTO struct {
	GameID          string             `json:"game_id"`
	WhiteName       string             `json:"white_name"`
	BlackName       string             `json:"black_name"`
	WhiteBitboard   uint64             `json:"white_bitboard"`
	BlackBitboard   uint64             `json:"black_bitboard"`
	CurrentPlayer   uint8              `json:"current_player"`
	Status          string             `json:"status"`
	StatusCode      uint8              `json:"status_code"`
	Winner          uint8              `json:"winner"`
	WinLineMask     uint64             `json:"win_line_mask"`
	WinLineIndex    int                `json:"win_line_index"`
	WinLineCategory string             `json:"win_line_category,omitempty"`
	WinLineCoords   []CoordDTO         `json:"win_line_coords,omitempty"`
	ValidMovesMask  uint16             `json:"valid_moves_mask"`
	ValidActions    []uint8            `json:"valid_actions"`
	MoveCount       uint8              `json:"move_count"`
	Board           [4][4][4]uint8     `json:"board"` // [z][y][x]
	CanonicalTensor []uint8            `json:"canonical_tensor"`
	History         []CoordDTO         `json:"history"`
}

// GameSummaryDTO is a compact descriptor returned when listing games (`GET /api/v1/games`).
type GameSummaryDTO struct {
	GameID        string `json:"game_id"`
	WhiteName     string `json:"white_name"`
	BlackName     string `json:"black_name"`
	Status        string `json:"status"`
	Winner        uint8  `json:"winner"`
	MoveCount     uint8  `json:"move_count"`
	CurrentPlayer uint8  `json:"current_player"`
}

// CreateGameRequest is the optional request body for `POST /api/v1/games`.
type CreateGameRequest struct {
	GameID    string `json:"game_id"`
	WhiteName string `json:"white_name"`
	BlackName string `json:"black_name"`
}

// MoveRequest is the JSON payload for `POST /api/v1/games/{id}/move` (and the response from remote webhook bots).
type MoveRequest struct {
	Player    uint8 `json:"player,omitempty"`     // Optional: if non-zero (1 or 2), verifies turn order
	Action    *int  `json:"action,omitempty"`     // Action in 0..15
	UseCoords bool  `json:"use_coords,omitempty"` // If true, uses (X, Y) instead of Action
	X         int   `json:"x,omitempty"`
	Y         int   `json:"y,omitempty"`
}

// MoveResponseDTO is returned after a valid move is applied via `POST /api/v1/games/{id}/move`.
type MoveResponseDTO struct {
	State    GameStateDTO `json:"state"`
	LastMove CoordDTO     `json:"last_move"`
	Reward   float32      `json:"reward"`
	Done     bool         `json:"done"`
}

// PresetRequest is the JSON payload for `POST /api/v1/games/{id}/preset`.
type PresetRequest struct {
	Type      string `json:"type"`       // "draw" or "win"
	LineIndex int    `json:"line_index"` // 0..75 when type == "win"
	Winner    uint8  `json:"winner"`     // 1 (White) or 2 (Black) when type == "win"
}

// RunMatchRequest is the JSON payload for `POST /api/v1/matches/run` to orchestrate a remote bot-vs-bot match.
type RunMatchRequest struct {
	GameID      string `json:"game_id"`
	WhiteName   string `json:"white_name"`
	BlackName   string `json:"black_name"`
	WhiteBotURL string `json:"white_bot_url"`
	BlackBotURL string `json:"black_bot_url"`
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
	id        string
	whiteName string
	blackName string
	game      *engine.Game
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
		whiteName: "Player 1 (White)",
		blackName: "Player 2 (Black)",
		game:      engine.NewGame(),
	}
	return s
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
	validActions := make([]uint8, 0, 16)
	for a := uint8(0); a < 16; a++ {
		if (validMask & (uint16(1) << a)) != 0 {
			validActions = append(validActions, a)
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
	tensorSlice := make([]uint8, len(tensor))
	copy(tensorSlice, tensor[:])

	dto := GameStateDTO{
		GameID:          sess.id,
		WhiteName:       sess.whiteName,
		BlackName:       sess.blackName,
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

// Handler returns the root http.Handler serving both the 3D Web Replay UI and `/api/v1/*` endpoints.
func (s *ArenaServer) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", s.handleRootUI)
	mux.HandleFunc("/api/v1/health", s.handleHealth)
	mux.HandleFunc("/api/v1/winlines", s.handleWinLines)
	mux.HandleFunc("/api/v1/games", s.handleGamesRoot)
	mux.HandleFunc("/api/v1/games/", s.handleGameByID)
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

func (s *ArenaServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
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
			summaries = append(summaries, GameSummaryDTO{
				GameID:        sess.id,
				WhiteName:     sess.whiteName,
				BlackName:     sess.blackName,
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
		gameID := strings.TrimSpace(req.GameID)
		if gameID == "" {
			seq := atomic.AddUint64(&s.seqCounter, 1)
			gameID = fmt.Sprintf("game-%d", seq)
		}
		whiteName := strings.TrimSpace(req.WhiteName)
		if whiteName == "" {
			whiteName = "Player 1 (White)"
		}
		blackName := strings.TrimSpace(req.BlackName)
		if blackName == "" {
			blackName = "Player 2 (Black)"
		}

		s.mu.Lock()
		sess := &gameSession{
			id:        gameID,
			whiteName: whiteName,
			blackName: blackName,
			game:      engine.NewGame(),
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
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "method not allowed")
			return
		}
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

	default:
		writeError(w, http.StatusNotFound, fmt.Sprintf("unknown game sub-route %q", subAction))
	}
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

	s.mu.Lock()
	defer s.mu.Unlock()

	sess, exists := s.sessions[gameID]
	if !exists {
		writeError(w, http.StatusNotFound, fmt.Sprintf("game %q not found", gameID))
		return
	}

	if sess.game.Status != engine.StatusOngoing {
		writeError(w, http.StatusConflict, engine.ErrGameAlreadyOver.Error())
		return
	}

	if req.Player != 0 {
		if req.Player != uint8(engine.PlayerWhite) && req.Player != uint8(engine.PlayerBlack) {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid player ID %d (must be 1 or 2)", req.Player))
			return
		}
		if req.Player != uint8(sess.game.CurrentPlayer) {
			writeError(w, http.StatusConflict, fmt.Sprintf("out of turn: it is player %d's (%s) turn, not player %d",
				sess.game.CurrentPlayer, sess.game.CurrentPlayer.String(), req.Player))
			return
		}
	}

	rec, err := sess.game.ApplyMove(action)
	if err != nil {
		if errors.Is(err, engine.ErrColumnFull) || errors.Is(err, engine.ErrInvalidAction) || errors.Is(err, engine.ErrInvalidCoords) {
			writeError(w, http.StatusBadRequest, err.Error())
			return
		}
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
	if strings.TrimSpace(req.WhiteBotURL) == "" || strings.TrimSpace(req.BlackBotURL) == "" {
		writeError(w, http.StatusBadRequest, "both white_bot_url and black_bot_url are required")
		return
	}

	gameID := strings.TrimSpace(req.GameID)
	if gameID == "" {
		seq := atomic.AddUint64(&s.seqCounter, 1)
		gameID = fmt.Sprintf("match-%d", seq)
	}
	whiteName := strings.TrimSpace(req.WhiteName)
	if whiteName == "" {
		whiteName = "Remote Bot (White)"
	}
	blackName := strings.TrimSpace(req.BlackName)
	if blackName == "" {
		blackName = "Remote Bot (Black)"
	}

	sess := &gameSession{
		id:        gameID,
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
		activePlayer := sess.game.CurrentPlayer
		s.mu.RUnlock()

		botURL := req.WhiteBotURL
		if activePlayer == engine.PlayerBlack {
			botURL = req.BlackBotURL
		}

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

		action, err := resolveMoveAction(moveReq)
		if err != nil {
			writeError(w, http.StatusBadGateway, fmt.Sprintf("bot %q returned invalid move: %v", botURL, err))
			return
		}

		s.mu.Lock()
		_, moveErr := sess.game.ApplyMove(action)
		s.mu.Unlock()
		if moveErr != nil {
			writeError(w, http.StatusBadGateway, fmt.Sprintf("bot %q attempted illegal move %d: %v", botURL, action, moveErr))
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
