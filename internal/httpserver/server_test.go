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

	// 2. Health and WinLines method checks
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

	req = httptest.NewRequest(http.MethodPost, "/api/v1/winlines", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on POST /api/v1/winlines, got %d", rec.Code)
	}

	// 3. Games collection bad JSON and 405
	req = httptest.NewRequest(http.MethodPost, "/api/v1/games", strings.NewReader("{bad-json"))
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 on invalid JSON in POST /api/v1/games, got %d", rec.Code)
	}

	req = httptest.NewRequest(http.MethodDelete, "/api/v1/games", nil)
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected 405 on DELETE /api/v1/games, got %d", rec.Code)
	}

	// 4. Game by ID edge cases
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

	// Bad JSON & missing URLs
	for _, body := range []string{`{bad`, `{"white_bot_url": "http://localhost"}`} {
		req = httptest.NewRequest(http.MethodPost, "/api/v1/matches/run", strings.NewReader(body))
		rec = httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected 400 for body %s, got %d", body, rec.Code)
		}
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
