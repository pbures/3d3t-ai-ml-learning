package httpserver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

func startBotWebhookServer(t *testing.T, scriptedMoves []uint8) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var state GameStateDTO
		if err := json.NewDecoder(r.Body).Decode(&state); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		act := int(scriptedMoves[state.MoveCount])
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(MoveRequest{Action: &act})
	}))
	t.Cleanup(srv.Close)
	return srv.URL
}

func postJSON(t *testing.T, client *http.Client, url string, payload interface{}, out interface{}) int {
	t.Helper()
	var bodyBytes []byte
	if payload != nil {
		var err error
		bodyBytes, err = json.Marshal(payload)
		if err != nil {
			t.Fatalf("failed to marshal JSON: %v", err)
		}
	}
	resp, err := client.Post(url, "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		t.Fatalf("POST %s failed: %v", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if out != nil && resp.StatusCode < 300 {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			t.Fatalf("failed to decode response from %s: %v", url, err)
		}
	}
	return resp.StatusCode
}

func getJSON(t *testing.T, client *http.Client, url string, out interface{}) int {
	t.Helper()
	resp, err := client.Get(url)
	if err != nil {
		t.Fatalf("GET %s failed: %v", url, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if out != nil && resp.StatusCode < 300 {
		if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
			t.Fatalf("failed to decode response from %s: %v", url, err)
		}
	}
	return resp.StatusCode
}

// TestHTTPSIntegration_AllGameRules verifies every game rule defined in SPECIFICATION.md
// over a live TLS 1.3 (https://) connection.
func TestHTTPSIntegration_AllGameRules(t *testing.T) {
	_, baseURL, client, cleanup, err := StartTLSServer("127.0.0.1:0")
	if err != nil {
		t.Fatalf("StartTLSServer failed: %v", err)
	}
	defer cleanup()

	// -------------------------------------------------------------------------
	// Rule 2.1: 4x4x4 Grid (64 places) & Coordinate/Action Boundaries
	// -------------------------------------------------------------------------
	t.Run("Rule_2.1_Grid_And_Bounds_Over_HTTPS", func(t *testing.T) {
		var created GameStateDTO
		code := postJSON(t, client, baseURL+"/api/v1/games", CreateGameRequest{GameID: "https-rule-2.1"}, &created)
		if code != http.StatusCreated {
			t.Fatalf("expected 201 Created, got %d", code)
		}
		if created.ValidMovesMask != 0xFFFF || len(created.ValidActions) != 16 {
			t.Fatalf("expected 16 valid actions on empty 4x4x4 grid, got mask=%04x len=%d",
				created.ValidMovesMask, len(created.ValidActions))
		}

		// Out-of-bounds action (>= 16 or < 0) must be rejected with 400 Bad Request
		act16 := 16
		if c := postJSON(t, client, baseURL+"/api/v1/games/https-rule-2.1/move", MoveRequest{Action: &act16}, nil); c != http.StatusBadRequest {
			t.Fatalf("expected 400 Bad Request for action=16, got %d", c)
		}

		// Out-of-bounds (x, y) coordinates (>= 4 or < 0) must be rejected with 400 Bad Request
		if c := postJSON(t, client, baseURL+"/api/v1/games/https-rule-2.1/move", MoveRequest{UseCoords: true, X: 4, Y: 0}, nil); c != http.StatusBadRequest {
			t.Fatalf("expected 400 Bad Request for x=4, got %d", c)
		}
		if c := postJSON(t, client, baseURL+"/api/v1/games/https-rule-2.1/move", MoveRequest{UseCoords: true, X: 0, Y: -1}, nil); c != http.StatusBadRequest {
			t.Fatalf("expected 400 Bad Request for y=-1, got %d", c)
		}

		// Verify GET /api/v1/winlines over HTTPS returns all 76 winning lines
		var wlPayload struct {
			Lines []WinLineDTO `json:"lines"`
		}
		if c := getJSON(t, client, baseURL+"/api/v1/winlines", &wlPayload); c != http.StatusOK || len(wlPayload.Lines) != 76 {
			t.Fatalf("expected 200 OK and 76 winlines, got code=%d len=%d", c, len(wlPayload.Lines))
		}
	})

	// -------------------------------------------------------------------------
	// Rule 2.2 & Rule 2.3a/b: Turn Alternation, Out-of-Turn Rejection,
	// Gravity (z=0..3), and Full Column Rejection
	// -------------------------------------------------------------------------
	t.Run("Rule_2.2_And_2.3_Turns_Gravity_And_FullColumn_Over_HTTPS", func(t *testing.T) {
		var st GameStateDTO
		if c := postJSON(t, client, baseURL+"/api/v1/games/https-rule-2.3/reset", nil, &st); c != http.StatusOK {
			t.Fatalf("expected 200 OK on reset, got %d", c)
		}
		if st.CurrentPlayer != uint8(engine.PlayerWhite) {
			t.Fatalf("expected WHITE (1) to start first, got %d", st.CurrentPlayer)
		}

		// Rule 2.2: Black (player 2) attempting to move on White's turn must be rejected with 409 Conflict
		act0 := 0
		if c := postJSON(t, client, baseURL+"/api/v1/games/https-rule-2.3/move",
			MoveRequest{Player: uint8(engine.PlayerBlack), Action: &act0}, nil); c != http.StatusConflict {
			t.Fatalf("expected 409 Conflict when Black moves on White's turn, got %d", c)
		}

		// Rule 2.3a: Drop 4 balls into column (x=1, y=3) using coordinate mode
		for expectedZ := uint8(0); expectedZ < 4; expectedZ++ {
			expectedMover := uint8(engine.PlayerWhite)
			if expectedZ%2 == 1 {
				expectedMover = uint8(engine.PlayerBlack)
			}

			var mvResp MoveResponseDTO
			c := postJSON(t, client, baseURL+"/api/v1/games/https-rule-2.3/move",
				MoveRequest{Player: expectedMover, UseCoords: true, X: 1, Y: 3}, &mvResp)
			if c != http.StatusOK {
				t.Fatalf("move at z=%d failed with status %d", expectedZ, c)
			}
			if mvResp.LastMove.X != 1 || mvResp.LastMove.Y != 3 || mvResp.LastMove.Z != expectedZ {
				t.Fatalf("gravity mismatch at z=%d: got (%d,%d,%d)",
					expectedZ, mvResp.LastMove.X, mvResp.LastMove.Y, mvResp.LastMove.Z)
			}
			if mvResp.State.Board[expectedZ][3][1] != expectedMover {
				t.Fatalf("expected 3D board[%d][3][1]=%d, got %d", expectedZ, expectedMover, mvResp.State.Board[expectedZ][3][1])
			}
		}

		// Rule 2.3b: 5th ball in column (x=1, y=3) must be rejected with 400 Bad Request
		if c := postJSON(t, client, baseURL+"/api/v1/games/https-rule-2.3/move",
			MoveRequest{UseCoords: true, X: 1, Y: 3}, nil); c != http.StatusBadRequest {
			t.Fatalf("expected 400 Bad Request on full column (1,3), got %d", c)
		}
	})

	// -------------------------------------------------------------------------
	// Rule 2.4a: All 76 Winning Lines (Orthogonal, Planar Diagonal, 3D Space Diagonal)
	// for both White and Black over HTTPS, plus Post-Win Rejection
	// -------------------------------------------------------------------------
	t.Run("Rule_2.4a_All_76_Winning_Lines_Over_HTTPS", func(t *testing.T) {
		for lineIdx := 0; lineIdx < engine.TotalWinLines; lineIdx++ {
			wl := engine.WinLines[lineIdx]
			for _, winner := range []engine.Player{engine.PlayerWhite, engine.PlayerBlack} {
				gameID := fmt.Sprintf("https-win-%d-p%d", lineIdx, winner)
				postJSON(t, client, baseURL+"/api/v1/games", CreateGameRequest{GameID: gameID}, nil)

				moves := engine.GenerateWinSequence(lineIdx, winner)
				var lastResp MoveResponseDTO
				for mIdx, a := range moves {
					act := int(a)
					c := postJSON(t, client, baseURL+"/api/v1/games/"+gameID+"/move", MoveRequest{Action: &act}, &lastResp)
					if c != http.StatusOK {
						t.Fatalf("game %s (%s) move %d failed with HTTP %d", gameID, wl.Category, mIdx, c)
					}
				}

				if !lastResp.Done || lastResp.Reward != 1.0 {
					t.Fatalf("game %s: expected Done=true Reward=1.0, got Done=%v Reward=%v", gameID, lastResp.Done, lastResp.Reward)
				}
				if lastResp.State.Winner != uint8(winner) || lastResp.State.WinLineIndex != lineIdx || lastResp.State.WinLineMask != wl.Mask {
					t.Fatalf("game %s: win metadata mismatch: winner=%d idx=%d mask=%016x",
						gameID, lastResp.State.Winner, lastResp.State.WinLineIndex, lastResp.State.WinLineMask)
				}
				if len(lastResp.State.WinLineCoords) != 4 || lastResp.State.ValidMovesMask != 0 {
					t.Fatalf("game %s: expected 4 win_line_coords and valid_moves_mask=0", gameID)
				}

				// Post-win move must be rejected with 409 Conflict
				act0 := 0
				if c := postJSON(t, client, baseURL+"/api/v1/games/"+gameID+"/move", MoveRequest{Action: &act0}, nil); c != http.StatusConflict {
					t.Fatalf("game %s: expected 409 Conflict after game won, got %d", gameID, c)
				}
			}
		}
	})

	// -------------------------------------------------------------------------
	// Rule 2.4b: Full 64-Move Non-Winning Draw Game over HTTPS
	// -------------------------------------------------------------------------
	t.Run("Rule_2.4b_64_Move_Draw_Over_HTTPS", func(t *testing.T) {
		gameID := "https-draw-64"
		postJSON(t, client, baseURL+"/api/v1/games", CreateGameRequest{GameID: gameID}, nil)

		drawSeq := engine.GenerateDrawSequence()
		var lastResp MoveResponseDTO
		for i, a := range drawSeq {
			act := int(a)
			c := postJSON(t, client, baseURL+"/api/v1/games/"+gameID+"/move", MoveRequest{Action: &act}, &lastResp)
			if c != http.StatusOK {
				t.Fatalf("draw move %d failed with HTTP %d", i+1, c)
			}
		}

		if !lastResp.Done || lastResp.Reward != 0.0 || lastResp.State.Status != "DRAW" || lastResp.State.Winner != 0 {
			t.Fatalf("expected DRAW state after 64 moves, got %+v", lastResp.State)
		}
		if lastResp.State.MoveCount != 64 || len(lastResp.State.History) != 64 || lastResp.State.ValidMovesMask != 0 {
			t.Fatalf("expected 64 history items and 0 valid moves on draw", )
		}
	})

	// -------------------------------------------------------------------------
	// Remote Bot Webhook Match Orchestration & Replay Inspection over HTTPS
	// -------------------------------------------------------------------------
	t.Run("Remote_Bot_Arena_Match_And_StepByStep_Replay", func(t *testing.T) {
		// Launch a mock remote Bot Server that plays a scripted 3D Space Diagonal win (line 72)
		scriptedMoves := engine.GenerateWinSequence(72, engine.PlayerWhite)

		botMux := http.NewServeMux()
		botMux.HandleFunc("/bot-move", func(w http.ResponseWriter, r *http.Request) {
			var state GameStateDTO
			if err := json.NewDecoder(r.Body).Decode(&state); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			act := int(scriptedMoves[state.MoveCount])
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(MoveRequest{Action: &act})
		})

		// Spin up a separate HTTPS server for the mock bots or use httptest
		_, botBaseURL, _, stopBotSrv, err := StartTLSServer("127.0.0.1:0")
		if err != nil {
			t.Fatalf("failed to start bot TLS server: %v", err)
		}
		defer stopBotSrv()
		_ = botBaseURL

		// Use a local HTTP test server for the bot webhook endpoint (trustedClient handles both http and https)
		botSrv := http.Server{Handler: botMux}
		botLis, err := (&http.Server{}), error(nil)
		_ = botSrv
		_ = botLis

		// Create an httptest server for the webhook bot
		webhookURL := startBotWebhookServer(t, scriptedMoves)

		var matchResult GameStateDTO
		code := postJSON(t, client, baseURL+"/api/v1/matches/run", RunMatchRequest{
			GameID:      "bot-battle-72",
			WhiteName:   "SpaceDiagBot-White",
			BlackName:   "SparringBot-Black",
			WhiteBotURL: webhookURL,
			BlackBotURL: webhookURL,
			TimeoutMs:   2000,
		}, &matchResult)
		if code != http.StatusOK {
			t.Fatalf("expected 200 OK from /api/v1/matches/run, got %d", code)
		}
		if matchResult.Status != "WHITE_WON" || matchResult.WinLineIndex != 72 {
			t.Fatalf("expected WHITE_WON on line 72, got status=%s line=%d", matchResult.Status, matchResult.WinLineIndex)
		}

		// Fetch game replay via GET /api/v1/games/bot-battle-72 and verify step-by-step history
		var replayDTO GameStateDTO
		if c := getJSON(t, client, baseURL+"/api/v1/games/bot-battle-72", &replayDTO); c != http.StatusOK {
			t.Fatalf("GET /api/v1/games/bot-battle-72 failed: %d", c)
		}
		if len(replayDTO.History) != len(scriptedMoves) {
			t.Fatalf("expected %d moves in replay history, got %d", len(scriptedMoves), len(replayDTO.History))
		}
		for i, stepRec := range replayDTO.History {
			if stepRec.MoveNumber != uint8(i+1) || stepRec.Action != scriptedMoves[i] {
				t.Fatalf("replay step %d mismatch: got %+v, want action %d", i+1, stepRec, scriptedMoves[i])
			}
		}

		// Verify GET /api/v1/games includes the completed match
		var listPayload struct {
			Games []GameSummaryDTO `json:"games"`
		}
		if c := getJSON(t, client, baseURL+"/api/v1/games", &listPayload); c != http.StatusOK || len(listPayload.Games) < 2 {
			t.Fatalf("expected game list with >= 2 games, got code=%d len=%d", c, len(listPayload.Games))
		}
	})
}
