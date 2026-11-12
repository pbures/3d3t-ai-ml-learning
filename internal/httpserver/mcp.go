package httpserver

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
)

// MCPJSONRPCRequest represents a JSON-RPC 2.0 request or notification in the Model Context Protocol.
type MCPJSONRPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// MCPJSONRPCResponse represents a JSON-RPC 2.0 response in the Model Context Protocol.
type MCPJSONRPCResponse struct {
	JSONRPC string        `json:"jsonrpc"`
	ID      interface{}   `json:"id"`
	Result  interface{}   `json:"result,omitempty"`
	Error   *MCPRPCError  `json:"error,omitempty"`
}

// MCPRPCError represents a JSON-RPC 2.0 error object.
type MCPRPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// MCPTool describes a callable MCP tool exposed by the 3T3D Arena MCP Server.
type MCPTool struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

// MCPContentBlock represents a text content item returned by `tools/call`.
type MCPContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// MCPCallToolResult is the result payload for `tools/call`.
type MCPCallToolResult struct {
	Content           []MCPContentBlock `json:"content"`
	StructuredContent interface{}       `json:"structuredContent,omitempty"`
	IsError           bool              `json:"isError,omitempty"`
}

var mcpToolsCatalog = []MCPTool{
	{
		Name:        "list_models",
		Description: "List all implemented 3T3D AI models (ttt3d-deep-gray1, ttt3d-deep-gray1-pure, deep, ttt3dnet, ttt3dnet-pure, mcts, alphabeta, heuristic, random) available on the Arena Server.",
		InputSchema: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	},
	{
		Name:        "list_battles",
		Description: "List all 3T3D battles with their name, date, participant (client player), overall battle winner (CLIENT, SERVER, TIE, or ONGOING), and aggregate statistics.",
		InputSchema: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	},
	{
		Name:        "create_battle",
		Description: "Create a new 3T3D battle with a name, date, and participant (client player name) so the client can play multiple games in the battle and track aggregate battle statistics.",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"battle_id":      map[string]interface{}{"type": "string", "description": "Optional custom battle ID slug"},
				"name":           map[string]interface{}{"type": "string", "description": "Human-friendly battle title/name"},
				"date":           map[string]interface{}{"type": "string", "description": "Battle date/timestamp (ISO-8601/RFC3339 or date string; defaults to current UTC time)"},
				"participant":    map[string]interface{}{"type": "string", "description": "Name of the client player participating in the battle"},
				"client_name":    map[string]interface{}{"type": "string", "description": "Alias for participant (client player name)"},
				"opponent_agent": map[string]interface{}{"type": "string", "enum": []string{"", "ttt3d-deep-gray1", "ttt3d-deep-gray1-pure", "deep", "ttt3dnet", "ttt3dnet-pure", "mcts", "alphabeta", "heuristic", "random"}, "description": "Optional default server AI model for games in this battle"},
			},
		},
	},
	{
		Name:        "get_battle",
		Description: "Get battle metadata, statistics (total games, client wins, server wins, draws, win rates), battle winner (CLIENT or SERVER), and all games in the battle.",
		InputSchema: map[string]interface{}{
			"type":     "object",
			"required": []string{"battle_id"},
			"properties": map[string]interface{}{
				"battle_id": map[string]interface{}{"type": "string", "description": "Battle ID (e.g. 'default' or 'battle-1')"},
			},
		},
	},
	{
		Name:        "update_battle",
		Description: "Update a battle's name, date, participant (client player name), or default opponent AI model.",
		InputSchema: map[string]interface{}{
			"type":     "object",
			"required": []string{"battle_id"},
			"properties": map[string]interface{}{
				"battle_id":      map[string]interface{}{"type": "string", "description": "Battle ID"},
				"name":           map[string]interface{}{"type": "string", "description": "Updated battle name"},
				"date":           map[string]interface{}{"type": "string", "description": "Updated battle date"},
				"participant":    map[string]interface{}{"type": "string", "description": "Updated client participant name"},
				"client_name":    map[string]interface{}{"type": "string", "description": "Alias for participant"},
				"opponent_agent": map[string]interface{}{"type": "string", "description": "Updated default AI opponent model"},
			},
		},
	},
	{
		Name:        "create_battle_game",
		Description: "Create a new game inside an existing battle so the client participant can play another game in that battle.",
		InputSchema: map[string]interface{}{
			"type":     "object",
			"required": []string{"battle_id"},
			"properties": map[string]interface{}{
				"battle_id":      map[string]interface{}{"type": "string", "description": "Battle ID to add the game to"},
				"game_id":        map[string]interface{}{"type": "string", "description": "Optional custom game ID slug"},
				"name":           map[string]interface{}{"type": "string", "description": "Human-friendly match title/name"},
				"participant":    map[string]interface{}{"type": "string", "description": "Optional override for client player name"},
				"white_name":     map[string]interface{}{"type": "string", "description": "Display name for White (Player 1)"},
				"black_name":     map[string]interface{}{"type": "string", "description": "Display name for Black (Player 2)"},
				"opponent_agent": map[string]interface{}{"type": "string", "enum": []string{"", "ttt3d-deep-gray1", "ttt3d-deep-gray1-pure", "deep", "ttt3dnet", "ttt3dnet-pure", "mcts", "alphabeta", "heuristic", "random"}, "description": "Optional AI model to play against automatically"},
				"human_player":   map[string]interface{}{"type": "integer", "enum": []int{1, 2}, "description": "1 for Client to start first as White, 2 for Client to play second as Black (Server AI makes the opening White move)"},
				"client_player":  map[string]interface{}{"type": "integer", "enum": []int{1, 2}, "description": "Which side the client controls (1=White, 2=Black)"},
			},
		},
	},
	{
		Name:        "list_games",
		Description: "List all active and completed 3T3D game sessions with their battle_id, match names, players, status, winner_side (CLIENT or SERVER), and move counts.",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"battle_id": map[string]interface{}{"type": "string", "description": "Optional battle_id to filter games belonging to a specific battle"},
			},
		},
	},
	{
		Name:        "create_game",
		Description: "Create a new 3T3D match (optionally associated with a battle_id and participant/client_name) with a custom match name, player names, opponent AI model, and starting side (human_player: 1 for White/first, 2 for Black/second where AI starts).",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"game_id":        map[string]interface{}{"type": "string", "description": "Optional custom game ID slug"},
				"battle_id":      map[string]interface{}{"type": "string", "description": "Optional battle ID to organize this game under (defaults to 'default')"},
				"name":           map[string]interface{}{"type": "string", "description": "Human-friendly match title/name"},
				"participant":    map[string]interface{}{"type": "string", "description": "Client player name"},
				"client_name":    map[string]interface{}{"type": "string", "description": "Alias for participant (client player name)"},
				"white_name":     map[string]interface{}{"type": "string", "description": "Display name for White (Player 1)"},
				"black_name":     map[string]interface{}{"type": "string", "description": "Display name for Black (Player 2)"},
				"opponent_agent": map[string]interface{}{"type": "string", "enum": []string{"", "ttt3d-deep-gray1", "ttt3d-deep-gray1-pure", "deep", "ttt3dnet", "ttt3dnet-pure", "mcts", "alphabeta", "heuristic", "random"}, "description": "Optional AI model to play against automatically"},
				"human_player":   map[string]interface{}{"type": "integer", "enum": []int{1, 2}, "description": "1 to start first as White, 2 to play second as Black (AI makes the opening White move immediately)"},
				"client_player":  map[string]interface{}{"type": "integer", "enum": []int{1, 2}, "description": "Which side the client player controls (1=White, 2=Black)"},
			},
		},
	},
	{
		Name:        "get_game",
		Description: "Get the complete 3D board state, bitboards, canonical tensor, legal actions, winner_side (CLIENT or SERVER), and move history for a game session.",
		InputSchema: map[string]interface{}{
			"type":     "object",
			"required": []string{"game_id"},
			"properties": map[string]interface{}{
				"game_id": map[string]interface{}{"type": "string", "description": "Game session ID (e.g. 'default')"},
			},
		},
	},
	{
		Name:        "rename_game",
		Description: "Rename an existing match or update its battle_id, participant, player names, or configured opponent AI model.",
		InputSchema: map[string]interface{}{
			"type":     "object",
			"required": []string{"game_id"},
			"properties": map[string]interface{}{
				"game_id":        map[string]interface{}{"type": "string", "description": "Game session ID"},
				"battle_id":      map[string]interface{}{"type": "string", "description": "Updated battle ID"},
				"name":           map[string]interface{}{"type": "string", "description": "New match title/name"},
				"participant":    map[string]interface{}{"type": "string", "description": "Updated client player name"},
				"white_name":     map[string]interface{}{"type": "string", "description": "Updated White player name"},
				"black_name":     map[string]interface{}{"type": "string", "description": "Updated Black player name"},
				"opponent_agent": map[string]interface{}{"type": "string", "description": "Updated AI opponent model (or empty string to disable auto-reply)"},
			},
		},
	},
	{
		Name:        "play_move",
		Description: "Submit a move in a 3T3D game by column action (0..15, where action = 4*y + x) or (x, y) coordinates. If an opponent_agent is configured or provided, the AI replies automatically in the same call.",
		InputSchema: map[string]interface{}{
			"type":     "object",
			"required": []string{"game_id"},
			"properties": map[string]interface{}{
				"game_id":        map[string]interface{}{"type": "string", "description": "Game session ID"},
				"action":         map[string]interface{}{"type": "integer", "minimum": 0, "maximum": 15, "description": "Column index 0..15 (action = 4*y + x)"},
				"use_coords":     map[string]interface{}{"type": "boolean", "description": "Set true to use (x, y) coordinates instead of action"},
				"x":              map[string]interface{}{"type": "integer", "minimum": 0, "maximum": 3, "description": "Column X coordinate (0..3) when use_coords is true"},
				"y":              map[string]interface{}{"type": "integer", "minimum": 0, "maximum": 3, "description": "Column Y coordinate (0..3) when use_coords is true"},
				"player":         map[string]interface{}{"type": "integer", "enum": []int{1, 2}, "description": "Optional turn verification (1=White, 2=Black)"},
				"opponent_agent": map[string]interface{}{"type": "string", "enum": []string{"", "ttt3d-deep-gray1", "ttt3d-deep-gray1-pure", "deep", "ttt3dnet", "ttt3dnet-pure", "mcts", "alphabeta", "heuristic", "random"}, "description": "Optional AI model to reply immediately after this move"},
				"auto_reply":     map[string]interface{}{"type": "boolean", "description": "Optional override for automatic AI reply"},
			},
		},
	},
	{
		Name:        "trigger_ai_move",
		Description: "Have one of the built-in AI models (ttt3d-deep-gray1, ttt3d-deep-gray1-pure, deep, ttt3dnet, ttt3dnet-pure, mcts, alphabeta, heuristic, random) make the next move in a game.",
		InputSchema: map[string]interface{}{
			"type":     "object",
			"required": []string{"game_id"},
			"properties": map[string]interface{}{
				"game_id": map[string]interface{}{"type": "string", "description": "Game session ID"},
				"agent":   map[string]interface{}{"type": "string", "enum": []string{"ttt3d-deep-gray1", "ttt3d-deep-gray1-pure", "deep", "ttt3dnet", "ttt3dnet-pure", "mcts", "alphabeta", "heuristic", "random"}, "description": "AI model to step (defaults to ttt3d-deep-gray1)"},
			},
		},
	},
	{
		Name:        "reset_game",
		Description: "Reset a game session back to an empty 4x4x4 board.",
		InputSchema: map[string]interface{}{
			"type":     "object",
			"required": []string{"game_id"},
			"properties": map[string]interface{}{
				"game_id": map[string]interface{}{"type": "string", "description": "Game session ID"},
			},
		},
	},
	{
		Name:        "run_match",
		Description: "Orchestrate a full automated match from start to finish between built-in AI models or remote bot webhook URLs.",
		InputSchema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"game_id":       map[string]interface{}{"type": "string", "description": "Optional custom game ID"},
				"battle_id":     map[string]interface{}{"type": "string", "description": "Optional battle ID to organize this match under"},
				"name":          map[string]interface{}{"type": "string", "description": "Human-readable match title"},
				"participant":   map[string]interface{}{"type": "string", "description": "Client player name"},
				"client_player": map[string]interface{}{"type": "integer", "enum": []int{1, 2}, "description": "Which side the client player controls (1=White, 2=Black)"},
				"white_name":    map[string]interface{}{"type": "string", "description": "White player display name"},
				"black_name":    map[string]interface{}{"type": "string", "description": "Black player display name"},
				"white_agent":   map[string]interface{}{"type": "string", "enum": []string{"", "ttt3d-deep-gray1", "ttt3d-deep-gray1-pure", "deep", "ttt3dnet", "ttt3dnet-pure", "mcts", "alphabeta", "heuristic", "random"}, "description": "Built-in model for White"},
				"black_agent":   map[string]interface{}{"type": "string", "enum": []string{"", "ttt3d-deep-gray1", "ttt3d-deep-gray1-pure", "deep", "ttt3dnet", "ttt3dnet-pure", "mcts", "alphabeta", "heuristic", "random"}, "description": "Built-in model for Black"},
				"white_bot_url": map[string]interface{}{"type": "string", "description": "Webhook URL for White if not using white_agent"},
				"black_bot_url": map[string]interface{}{"type": "string", "description": "Webhook URL for Black if not using black_agent"},
				"timeout_ms":    map[string]interface{}{"type": "integer", "description": "Per-move webhook timeout in ms"},
			},
		},
	},
}

func (s *ArenaServer) invokeInternalAPI(method, path string, body interface{}) (int, []byte) {
	var bodyReader io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		bodyReader = bytes.NewReader(b)
	}
	req := httptest.NewRequest(method, path, bodyReader)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	return rec.Code, rec.Body.Bytes()
}

func (s *ArenaServer) handleMCP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Accept, Mcp-Session-Id")

	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return
	}

	if r.Method == http.MethodGet {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"protocol":        "mcp",
			"protocolVersion": "2024-11-05",
			"serverInfo": map[string]string{
				"name":    "3t3d-arena-mcp",
				"version": "1.0.0",
			},
			"tools": mcpToolsCatalog,
		})
		return
	}

	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}

	var rpcReq MCPJSONRPCRequest
	if err := json.NewDecoder(r.Body).Decode(&rpcReq); err != nil {
		writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
			JSONRPC: "2.0",
			ID:      nil,
			Error: &MCPRPCError{
				Code:    -32700,
				Message: "Parse error: invalid JSON-RPC payload",
			},
		})
		return
	}

	// JSON-RPC notifications (no ID or notifications/*)
	if strings.HasPrefix(rpcReq.Method, "notifications/") {
		w.WriteHeader(http.StatusAccepted)
		return
	}

	switch rpcReq.Method {
	case "initialize":
		writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
			JSONRPC: "2.0",
			ID:      rpcReq.ID,
			Result: map[string]interface{}{
				"protocolVersion": "2024-11-05",
				"capabilities": map[string]interface{}{
					"tools":     map[string]interface{}{},
					"resources": map[string]interface{}{},
				},
				"serverInfo": map[string]string{
					"name":    "3t3d-arena-mcp",
					"version": "1.0.0",
				},
			},
		})

	case "ping":
		writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
			JSONRPC: "2.0",
			ID:      rpcReq.ID,
			Result:  map[string]interface{}{},
		})

	case "tools/list":
		writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
			JSONRPC: "2.0",
			ID:      rpcReq.ID,
			Result: map[string]interface{}{
				"tools": mcpToolsCatalog,
			},
		})

	case "resources/list":
		writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
			JSONRPC: "2.0",
			ID:      rpcReq.ID,
			Result: map[string]interface{}{
				"resources": []map[string]string{
					{
						"uri":         "openapi://3t3d/openapi.json",
						"name":        "3T3D Arena OpenAPI 3.1 Specification",
						"mimeType":    "application/json",
						"description": "Complete OpenAPI 3.1 specification for the 3T3D Arena REST API on https://3t3d.buresovi.net",
					},
				},
			},
		})

	case "resources/read":
		var params struct {
			URI string `json:"uri"`
		}
		_ = json.Unmarshal(rpcReq.Params, &params)
		if params.URI != "openapi://3t3d/openapi.json" {
			writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
				JSONRPC: "2.0",
				ID:      rpcReq.ID,
				Error:   &MCPRPCError{Code: -32602, Message: fmt.Sprintf("unknown resource URI %q", params.URI)},
			})
			return
		}
		data, _ := webAssets.ReadFile("web/openapi.json")
		writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
			JSONRPC: "2.0",
			ID:      rpcReq.ID,
			Result: map[string]interface{}{
				"contents": []map[string]string{
					{
						"uri":      params.URI,
						"mimeType": "application/json",
						"text":     string(data),
					},
				},
			},
		})

	case "tools/call":
		s.handleMCPToolCall(w, rpcReq)

	default:
		writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
			JSONRPC: "2.0",
			ID:      rpcReq.ID,
			Error: &MCPRPCError{
				Code:    -32601,
				Message: fmt.Sprintf("Method not found: %s", rpcReq.Method),
			},
		})
	}
}

func (s *ArenaServer) handleMCPToolCall(w http.ResponseWriter, rpcReq MCPJSONRPCRequest) {
	var callParams struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	}
	if err := json.Unmarshal(rpcReq.Params, &callParams); err != nil || strings.TrimSpace(callParams.Name) == "" {
		writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
			JSONRPC: "2.0",
			ID:      rpcReq.ID,
			Error:   &MCPRPCError{Code: -32602, Message: "Invalid params for tools/call"},
		})
		return
	}
	args := callParams.Arguments
	if args == nil {
		args = map[string]interface{}{}
	}

	strArg := func(key, fallback string) string {
		if v, ok := args[key]; ok && v != nil {
			if sVal, ok := v.(string); ok && strings.TrimSpace(sVal) != "" {
				return strings.TrimSpace(sVal)
			}
		}
		return fallback
	}

	var status int
	var respBytes []byte

	switch callParams.Name {
	case "list_models":
		status, respBytes = s.invokeInternalAPI(http.MethodGet, "/api/v1/models", nil)

	case "list_battles":
		status, respBytes = s.invokeInternalAPI(http.MethodGet, "/api/v1/battles", nil)

	case "create_battle":
		status, respBytes = s.invokeInternalAPI(http.MethodPost, "/api/v1/battles", args)

	case "get_battle":
		bid := strArg("battle_id", "default")
		status, respBytes = s.invokeInternalAPI(http.MethodGet, "/api/v1/battles/"+bid, nil)

	case "update_battle":
		bid := strArg("battle_id", "")
		if bid == "" {
			s.writeMCPToolResult(w, rpcReq.ID, http.StatusBadRequest, []byte(`{"error":"battle_id is required"}`))
			return
		}
		status, respBytes = s.invokeInternalAPI(http.MethodPatch, "/api/v1/battles/"+bid, args)

	case "create_battle_game":
		bid := strArg("battle_id", "")
		if bid == "" {
			s.writeMCPToolResult(w, rpcReq.ID, http.StatusBadRequest, []byte(`{"error":"battle_id is required"}`))
			return
		}
		status, respBytes = s.invokeInternalAPI(http.MethodPost, "/api/v1/battles/"+bid+"/games", args)

	case "list_games":
		bid := strArg("battle_id", "")
		path := "/api/v1/games"
		if bid != "" {
			path += "?battle_id=" + bid
		}
		status, respBytes = s.invokeInternalAPI(http.MethodGet, path, nil)

	case "create_game":
		status, respBytes = s.invokeInternalAPI(http.MethodPost, "/api/v1/games", args)

	case "get_game":
		gid := strArg("game_id", "default")
		status, respBytes = s.invokeInternalAPI(http.MethodGet, "/api/v1/games/"+gid, nil)

	case "rename_game":
		gid := strArg("game_id", "")
		if gid == "" {
			s.writeMCPToolResult(w, rpcReq.ID, http.StatusBadRequest, []byte(`{"error":"game_id is required"}`))
			return
		}
		status, respBytes = s.invokeInternalAPI(http.MethodPatch, "/api/v1/games/"+gid, args)

	case "play_move":
		gid := strArg("game_id", "default")
		status, respBytes = s.invokeInternalAPI(http.MethodPost, "/api/v1/games/"+gid+"/move", args)

	case "trigger_ai_move":
		gid := strArg("game_id", "default")
		status, respBytes = s.invokeInternalAPI(http.MethodPost, "/api/v1/games/"+gid+"/ai-move", args)

	case "reset_game":
		gid := strArg("game_id", "default")
		status, respBytes = s.invokeInternalAPI(http.MethodPost, "/api/v1/games/"+gid+"/reset", map[string]interface{}{})

	case "run_match":
		status, respBytes = s.invokeInternalAPI(http.MethodPost, "/api/v1/matches/run", args)

	default:
		writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
			JSONRPC: "2.0",
			ID:      rpcReq.ID,
			Error:   &MCPRPCError{Code: -32602, Message: fmt.Sprintf("Unknown MCP tool: %s", callParams.Name)},
		})
		return
	}

	s.writeMCPToolResult(w, rpcReq.ID, status, respBytes)
}

func (s *ArenaServer) writeMCPToolResult(w http.ResponseWriter, id interface{}, httpStatus int, rawJSON []byte) {
	trimmed := strings.TrimSpace(string(rawJSON))
	var structured interface{}
	_ = json.Unmarshal(rawJSON, &structured)

	writeJSON(w, http.StatusOK, MCPJSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result: MCPCallToolResult{
			Content: []MCPContentBlock{
				{
					Type: "text",
					Text: trimmed,
				},
			},
			StructuredContent: structured,
			IsError:           httpStatus >= 400,
		},
	})
}

// ServeMCPStdio runs the Model Context Protocol JSON-RPC 2.0 server over newline-delimited stdio (`r` -> `w`).
func (s *ArenaServer) ServeMCPStdio(r io.Reader, w io.Writer) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	enc := json.NewEncoder(w)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		httpReq := httptest.NewRequest(http.MethodPost, "/mcp", bytes.NewReader(line))
		httpReq.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		s.handleMCP(rec, httpReq)
		if rec.Code == http.StatusAccepted || rec.Body.Len() == 0 {
			continue
		}
		var out map[string]interface{}
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err == nil {
			_ = enc.Encode(out)
		}
	}
	return scanner.Err()
}

