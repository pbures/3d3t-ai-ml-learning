package httpserver

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestUnitHandlerMethodsAndErrors(t *testing.T) {
	arena := NewArenaServer(nil)
	h := arena.Handler()

	// 1. Root UI route and 404/405 checks
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "3T3D") {
		t.Fatalf("expected 200 OK with HTML UI, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on POST /, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/nonexistent", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 on /nonexistent, got %d", rec.Code)
	}

	// 2. Health, Models, OpenAPI, Docs, and WinLines method checks
	req = httptest.NewRequest(http.MethodGet, "/api/v1/health", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 on GET /api/v1/health, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/health", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on POST /api/v1/health, got %d", rec.Code)
	}

	for _, openAPIRoute := range []string{"/openapi.json", "/api/v1/openapi.json", "/openapi.yaml", "/api/v1/openapi.yaml", "/docs", "/api/v1/docs", "/api/v1/models"} {
		req = httptest.NewRequest(http.MethodGet, openAPIRoute, nil)
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("expected 200 on GET %s, got %d", openAPIRoute, rec.Code)
		}

		req = httptest.NewRequest(http.MethodPost, openAPIRoute, nil)
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusMethodNotAllowed {
			t.Fatalf("expected 405 on POST %s, got %d", openAPIRoute, rec.Code)
		}
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/winlines", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on POST /api/v1/winlines, got %d", rec.Code)
	}

	// 3. Games collection bad JSON, invalid opponent_agent, and 405
	req = httptest.NewRequest(http.MethodPost, "/api/v1/games", strings.NewReader("{bad-json"))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on invalid JSON in POST /api/v1/games, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/games", strings.NewReader(`{"opponent_agent":"invalid-bot"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on unknown opponent_agent in POST /api/v1/games, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodDelete, "/api/v1/games", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on DELETE /api/v1/games, got %d", rec.Code)
	}

	// 4. Game by ID & PATCH edge cases
	req = httptest.NewRequest(http.MethodGet, "/api/v1/games/", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on empty game_id, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/games/default", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on POST /api/v1/games/default, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPatch, "/api/v1/games/default", strings.NewReader("{bad"))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on bad JSON in PATCH /api/v1/games/default, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPatch, "/api/v1/games/default", strings.NewReader(`{"opponent_agent":"unknown"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on unknown opponent_agent in PATCH, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPatch, "/api/v1/games/missing-id", strings.NewReader(`{"name":"Renamed"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 on PATCH missing game, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPatch, "/api/v1/games/default", strings.NewReader(`{"name":"Renamed Default","white_name":"Alice","black_name":"Bob","opponent_agent":"heuristic"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Renamed Default") {
		t.Fatalf("expected 200 on PATCH /api/v1/games/default, got %d (%s)", rec.Code, rec.Body.String())
	}
	// Reset default opponent_agent back to empty so subsequent move tests on "default" don't auto-reply unless requested
	req = httptest.NewRequest(http.MethodPatch, "/api/v1/games/default", strings.NewReader(`{"opponent_agent":""}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	req = httptest.NewRequest(http.MethodGet, "/api/v1/games/missing-id", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 on missing game, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/api/v1/games/default/reset", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on GET /reset, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/api/v1/games/default/move", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on GET /move, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/api/v1/games/default/preset", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on GET /preset, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/api/v1/games/default/unknown", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 on unknown sub-route, got %d", rec.Code)
	}

	// 5. Move bad payloads & missing game
	for _, body := range []string{
		`{bad`,
		`{}`,
		`{"player": 99, "action": 0}`,
		`{"action": 0, "opponent_agent": "invalid-agent"}`,
	} {
		req = httptest.NewRequest(http.MethodPost, "/api/v1/games/default/move", strings.NewReader(body))
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for body %s, got %d", body, rec.Code)
		}
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/games/missing/move", strings.NewReader(`{"action": 0}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 on move for missing game, got %d", rec.Code)
	}

	// 6. Preset endpoint validation & success paths
	for _, body := range []string{
		`{bad`,
		`{"type": "unknown"}`,
		`{"type": "win", "line_index": 99}`,
	} {
		req = httptest.NewRequest(http.MethodPost, "/api/v1/games/default/preset", strings.NewReader(body))
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for preset body %s, got %d", body, rec.Code)
		}
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/games/preset-game/preset", strings.NewReader(`{"type": "win", "line_index": 72, "winner": 2}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 on win preset, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/games/preset-game/preset", strings.NewReader(`{"type": "draw"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 on draw preset, got %d", rec.Code)
	}

	// 7. AI-Move & Built-in Bot Webhook validation
	req = httptest.NewRequest(http.MethodGet, "/api/v1/games/default/ai-move", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on GET /ai-move, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/games/default/ai-move", strings.NewReader(`{bad`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on bad JSON in /ai-move, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/games/missing/ai-move", strings.NewReader(`{"agent":"mcts"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 on missing game in /ai-move, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/games/default/ai-move", strings.NewReader(`{"agent":"unknown"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on unknown agent in /ai-move, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/games/preset-game/ai-move", strings.NewReader(`{"agent":"mcts"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("expected 409 on finished game in /ai-move, got %d", rec.Code)
	}

	// Bot webhook error paths
	req = httptest.NewRequest(http.MethodGet, "/api/v1/bots/mcts/move", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on GET /api/v1/bots/mcts/move, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/bots/mcts/invalid", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 on invalid bot route, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/bots/mcts/move", strings.NewReader(`{bad`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on bad JSON in bot webhook, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/bots/unknown/move", strings.NewReader(`{}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on unknown agent in bot webhook, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/v1/bots/mcts/move", strings.NewReader(`{"status_code":3}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusConflict {
		t.Fatalf("expected 409 on terminal state in bot webhook, got %d", rec.Code)
	}

	// 8. StartHTTPServer lifecycle & invalid address test
	_, httpURL, stopHTTP, err := StartHTTPServer("127.0.0.1:0")
	if err != nil {
		t.Fatalf("StartHTTPServer failed: %v", err)
	}
	resp, err := http.Get(httpURL + "/api/v1/health")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s/api/v1/health failed: %v", httpURL, err)
	}
	_ = resp.Body.Close()
	stopHTTP()

	if _, _, _, err := StartHTTPServer("invalid-addr:999999"); err == nil {
		t.Fatalf("expected error on invalid address in StartHTTPServer")
	}
	if _, _, _, _, err := StartTLSServer("invalid-addr:999999"); err == nil {
		t.Fatalf("expected error on invalid address in StartTLSServer")
	}
}

func TestUnitRunMatchErrorPaths(t *testing.T) {
	arena := NewArenaServer(nil)
	h := arena.Handler()

	// 405 on GET
	req := httptest.NewRequest(http.MethodGet, "/api/v1/matches/run", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on GET /api/v1/matches/run, got %d", rec.Code)
	}

	// Bad JSON, missing URLs/agents, or unknown agent names
	for _, body := range []string{
		`{bad`,
		`{"white_bot_url": "http://localhost"}`,
		`{"white_agent": "unknown", "black_agent": "random"}`,
		`{"white_agent": "random", "black_agent": "unknown"}`,
	} {
		req = httptest.NewRequest(http.MethodPost, "/api/v1/matches/run", strings.NewReader(body))
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for body %s, got %d", body, rec.Code)
		}
	}

	// Built-in agent vs built-in agent match via white_agent / black_agent
	req = httptest.NewRequest(http.MethodPost, "/api/v1/matches/run",
		strings.NewReader(`{"name":"Agent Showdown","white_agent":"heuristic","black_agent":"random"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "Agent Showdown") {
		t.Fatalf("expected 200 for built-in agent match, got %d (%s)", rec.Code, rec.Body.String())
	}

	// Unreachable bot URL -> 502 Bad Gateway
	req = httptest.NewRequest(http.MethodPost, "/api/v1/matches/run",
		bytes.NewBufferString(`{"white_bot_url":"http://127.0.0.1:1","black_bot_url":"http://127.0.0.1:1","timeout_ms":100}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 for unreachable bot, got %d", rec.Code)
	}

	// Bot returning HTTP 500
	errBot := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer errBot.Close()

	req = httptest.NewRequest(http.MethodPost, "/api/v1/matches/run",
		bytes.NewBufferString(`{"white_bot_url":"`+errBot.URL+`","black_bot_url":"`+errBot.URL+`"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 for bot returning 500, got %d", rec.Code)
	}

	// Bot returning invalid action (99)
	badMoveBot := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"action": 99}`))
	}))
	defer badMoveBot.Close()

	req = httptest.NewRequest(http.MethodPost, "/api/v1/matches/run",
		bytes.NewBufferString(`{"white_bot_url":"`+badMoveBot.URL+`","black_bot_url":"`+badMoveBot.URL+`"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 for bot returning action 99, got %d", rec.Code)
	}

	// Bot repeatedly playing column 0 even after column 0 is full (5th move in col 0) -> 502
	colZeroBot := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"action": 0}`))
	}))
	defer colZeroBot.Close()

	req = httptest.NewRequest(http.MethodPost, "/api/v1/matches/run",
		bytes.NewBufferString(`{"white_bot_url":"`+colZeroBot.URL+`","black_bot_url":"`+colZeroBot.URL+`"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 for bot playing into full column, got %d", rec.Code)
	}
}

func TestUnitMCPServerHTTPAndStdio(t *testing.T) {
	arena := NewArenaServer(nil)
	h := arena.Handler()

	// 1. OPTIONS & GET /mcp discovery
	req := httptest.NewRequest(http.MethodOptions, "/mcp", nil)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusNoContent {
		t.Fatalf("expected 204 on OPTIONS /mcp, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/mcp", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "3t3d-arena-mcp") {
		t.Fatalf("expected 200 on GET /mcp, got %d (%s)", rec.Code, rec.Body.String())
	}

	req = httptest.NewRequest(http.MethodDelete, "/mcp", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on DELETE /mcp, got %d", rec.Code)
	}

	// 2. Parse error & notifications
	req = httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader("{bad"))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if !strings.Contains(rec.Body.String(), "-32700") {
		t.Fatalf("expected JSON-RPC parse error -32700, got %s", rec.Body.String())
	}

	req = httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(`{"jsonrpc":"2.0","method":"notifications/initialized"}`))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("expected 202 on notifications/initialized, got %d", rec.Code)
	}

	// 3. Stdio transport and all MCP methods/tools
	stdioInput := strings.Join([]string{
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}`,
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","id":2,"method":"ping"}`,
		`{"jsonrpc":"2.0","id":3,"method":"tools/list"}`,
		`{"jsonrpc":"2.0","id":4,"method":"resources/list"}`,
		`{"jsonrpc":"2.0","id":5,"method":"resources/read","params":{"uri":"openapi://3t3d/openapi.json"}}`,
		`{"jsonrpc":"2.0","id":6,"method":"resources/read","params":{"uri":"unknown://uri"}}`,
		`{"jsonrpc":"2.0","id":7,"method":"unknown/method"}`,
		`{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{}}`,
		`{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"unknown_tool"}}`,
		`{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"list_models"}}`,
		`{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"create_game","arguments":{"game_id":"mcp-g1","name":"MCP Match 1","opponent_agent":"heuristic","human_player":1}}}`,
		`{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"play_move","arguments":{"game_id":"mcp-g1","action":0}}}`,
		`{"jsonrpc":"2.0","id":13,"method":"tools/call","params":{"name":"rename_game","arguments":{"game_id":"mcp-g1","name":"MCP Renamed"}}}`,
		`{"jsonrpc":"2.0","id":14,"method":"tools/call","params":{"name":"rename_game","arguments":{}}}`,
		`{"jsonrpc":"2.0","id":15,"method":"tools/call","params":{"name":"get_game","arguments":{"game_id":"mcp-g1"}}}`,
		`{"jsonrpc":"2.0","id":16,"method":"tools/call","params":{"name":"trigger_ai_move","arguments":{"game_id":"mcp-g1","agent":"random"}}}`,
		`{"jsonrpc":"2.0","id":17,"method":"tools/call","params":{"name":"reset_game","arguments":{"game_id":"mcp-g1"}}}`,
		`{"jsonrpc":"2.0","id":18,"method":"tools/call","params":{"name":"list_games"}}`,
		`{"jsonrpc":"2.0","id":19,"method":"tools/call","params":{"name":"run_match","arguments":{"name":"MCP Auto Match","white_agent":"heuristic","black_agent":"random"}}}`,
	}, "\n") + "\n"

	var stdioOut bytes.Buffer
	if err := arena.ServeMCPStdio(strings.NewReader(stdioInput), &stdioOut); err != nil {
		t.Fatalf("ServeMCPStdio failed: %v", err)
	}
	outLines := strings.Split(strings.TrimSpace(stdioOut.String()), "\n")
	if len(outLines) != 19 {
		t.Fatalf("expected 19 JSON-RPC responses from ServeMCPStdio (notification skipped), got %d", len(outLines))
	}
}

