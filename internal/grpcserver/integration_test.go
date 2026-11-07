package grpcserver

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
	"github.com/pbures/3d3t-ai-ml-learning/internal/grpcserver/pb"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func setupLiveGRPC(t *testing.T, endpoint string) (pb.EnvServiceClient, func()) {
	t.Helper()
	_, resolvedTarget, stopSrv, err := StartServer(endpoint)
	if err != nil {
		t.Fatalf("failed to start gRPC server on %s: %v", endpoint, err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	client, conn, err := DialClient(ctx, resolvedTarget)
	cancel()
	if err != nil {
		stopSrv()
		t.Fatalf("failed to dial gRPC server on %s: %v", resolvedTarget, err)
	}

	cleanup := func() {
		_ = conn.Close()
		stopSrv()
	}
	return client, cleanup
}

// TestGRPCIntegration_AllGameRules verifies every game rule defined in SPECIFICATION.md
// over a live Unix Domain Socket gRPC connection.
func TestGRPCIntegration_AllGameRules(t *testing.T) {
	sockPath := fmt.Sprintf("/tmp/3t3d_rules_%d.sock", os.Getpid())
	client, cleanup := setupLiveGRPC(t, "unix://"+sockPath)
	defer cleanup()

	ctx := context.Background()

	// -------------------------------------------------------------------------
	// Rule 2.1: 4x4x4 Grid (64 places) & Action Boundaries
	// -------------------------------------------------------------------------
	t.Run("Rule_2.1_Grid_And_Bounds", func(t *testing.T) {
		st, err := client.Reset(ctx, &pb.ResetRequest{GameId: "rule-2.1"})
		if err != nil {
			t.Fatalf("Reset failed: %v", err)
		}
		if st.ValidMovesMask != 0xFFFF || len(st.CanonicalTensor) != engine.CanonicalTensorSize {
			t.Fatalf("unexpected initial grid state: mask=%04x tensor_len=%d", st.ValidMovesMask, len(st.CanonicalTensor))
		}

		// Out-of-bounds action (>= 16) must be rejected over gRPC
		_, err = client.Step(ctx, &pb.StepRequest{GameId: "rule-2.1", Action: 16})
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("expected InvalidArgument for action 16, got %v", err)
		}

		// Verify GetWinLines returns all 76 lines within {0..3}^3
		wlResp, err := client.GetWinLines(ctx, &pb.Empty{})
		if err != nil || len(wlResp.Lines) != 76 {
			t.Fatalf("expected 76 winning lines from GetWinLines, got len=%d err=%v", len(wlResp.Lines), err)
		}
		for _, line := range wlResp.Lines {
			if len(line.Coords) != 4 {
				t.Fatalf("line %d has %d coords, want 4", line.Index, len(line.Coords))
			}
			for _, c := range line.Coords {
				if c.X > 3 || c.Y > 3 || c.Z > 3 {
					t.Fatalf("line %d coordinate out of 4x4x4 bounds: %+v", line.Index, c)
				}
			}
		}
	})

	// -------------------------------------------------------------------------
	// Rule 2.2 & Rule 2.3a/b: Turn Alternation, Gravity (z=0..3), and Full Column Rejection
	// -------------------------------------------------------------------------
	t.Run("Rule_2.2_And_2.3_Turns_Gravity_And_FullColumn", func(t *testing.T) {
		st, err := client.Reset(ctx, &pb.ResetRequest{GameId: "rule-2.3"})
		if err != nil {
			t.Fatalf("Reset failed: %v", err)
		}
		if st.CurrentPlayer != uint32(engine.PlayerWhite) {
			t.Fatalf("expected WHITE (1) to start first, got %d", st.CurrentPlayer)
		}

		// Choose column (x=3, y=2) -> action = 4*2 + 3 = 11
		const targetAction = 11
		for expectedZ := uint32(0); expectedZ < 4; expectedZ++ {
			expectedMover := uint32(engine.PlayerWhite)
			expectedNext := uint32(engine.PlayerBlack)
			if expectedZ%2 == 1 {
				expectedMover = uint32(engine.PlayerBlack)
				expectedNext = uint32(engine.PlayerWhite)
			}

			resp, err := client.Step(ctx, &pb.StepRequest{GameId: "rule-2.3", Action: targetAction})
			if err != nil {
				t.Fatalf("Step at z=%d failed: %v", expectedZ, err)
			}
			if resp.LastMove.X != 3 || resp.LastMove.Y != 2 || resp.LastMove.Z != expectedZ {
				t.Fatalf("gravity mismatch at z=%d: got (%d,%d,%d)",
					expectedZ, resp.LastMove.X, resp.LastMove.Y, resp.LastMove.Z)
			}
			if resp.LastMove.Player != expectedMover {
				t.Fatalf("expected mover %d at z=%d, got %d", expectedMover, expectedZ, resp.LastMove.Player)
			}
			if resp.State.CurrentPlayer != expectedNext {
				t.Fatalf("expected turn to switch to %d, got %d", expectedNext, resp.State.CurrentPlayer)
			}
		}

		// Verify GetState reflects 4 moves and column 11 is now masked out in valid_moves_mask
		stAfter, err := client.GetState(ctx, &pb.GetStateRequest{GameId: "rule-2.3"})
		if err != nil {
			t.Fatalf("GetState failed: %v", err)
		}
		if stAfter.MoveCount != 4 || len(stAfter.History) != 4 {
			t.Fatalf("expected 4 moves in state history, got count=%d len=%d", stAfter.MoveCount, len(stAfter.History))
		}
		if (stAfter.ValidMovesMask & (1 << targetAction)) != 0 {
			t.Fatalf("expected full column %d to be cleared from valid_moves_mask (%04x)", targetAction, stAfter.ValidMovesMask)
		}

		// 5th ball in column 11 must be rejected as an illegal move over gRPC
		_, err = client.Step(ctx, &pb.StepRequest{GameId: "rule-2.3", Action: targetAction})
		if status.Code(err) != codes.InvalidArgument {
			t.Fatalf("expected InvalidArgument when playing into a full column, got %v", err)
		}
	})

	// -------------------------------------------------------------------------
	// Rule 2.4a: All 76 Winning Lines (Orthogonal, Planar Diagonal, 3D Space Diagonal)
	// and Post-Win Move Rejection
	// -------------------------------------------------------------------------
	t.Run("Rule_2.4a_All_76_Winning_Lines_White_And_Black", func(t *testing.T) {
		for lineIdx := 0; lineIdx < engine.TotalWinLines; lineIdx++ {
			wl := engine.WinLines[lineIdx]
			for _, winner := range []engine.Player{engine.PlayerWhite, engine.PlayerBlack} {
				gameID := fmt.Sprintf("win-line-%d-player-%d", lineIdx, winner)
				if _, err := client.Reset(ctx, &pb.ResetRequest{GameId: gameID}); err != nil {
					t.Fatalf("Reset(%s) failed: %v", gameID, err)
				}

				moves := engine.GenerateWinSequence(lineIdx, winner)
				var lastResp *pb.StepResponse
				for mIdx, action := range moves {
					resp, err := client.Step(ctx, &pb.StepRequest{GameId: gameID, Action: uint32(action)})
					if err != nil {
						t.Fatalf("game %s (%s) move %d failed: %v", gameID, wl.Category, mIdx, err)
					}
					lastResp = resp
				}

				if !lastResp.Done || lastResp.Reward != 1.0 {
					t.Fatalf("game %s (%s): expected Done=true Reward=1.0, got Done=%v Reward=%v",
						gameID, wl.Category, lastResp.Done, lastResp.Reward)
				}
				if lastResp.State.Winner != uint32(winner) {
					t.Fatalf("game %s (%s): expected Winner=%d, got %d", gameID, wl.Category, winner, lastResp.State.Winner)
				}
				if lastResp.State.WinLineMask != wl.Mask || lastResp.State.WinLineIndex != int32(lineIdx) {
					t.Fatalf("game %s (%s): expected win mask %016x (idx %d), got %016x (idx %d)",
						gameID, wl.Category, wl.Mask, lineIdx, lastResp.State.WinLineMask, lastResp.State.WinLineIndex)
				}
				if lastResp.State.ValidMovesMask != 0 {
					t.Fatalf("game %s: expected valid_moves_mask=0 after win, got %04x", gameID, lastResp.State.ValidMovesMask)
				}

				// Post-win move must be rejected
				if _, err := client.Step(ctx, &pb.StepRequest{GameId: gameID, Action: 0}); status.Code(err) != codes.InvalidArgument {
					t.Fatalf("game %s: expected InvalidArgument for move after game won, got %v", gameID, err)
				}
			}
		}
	})

	// -------------------------------------------------------------------------
	// Rule 2.4b: Full 64-Move Non-Winning Draw Game
	// -------------------------------------------------------------------------
	t.Run("Rule_2.4b_64_Move_Draw_Game", func(t *testing.T) {
		gameID := "rule-2.4b-draw"
		if _, err := client.Reset(ctx, &pb.ResetRequest{GameId: gameID}); err != nil {
			t.Fatalf("Reset(%s) failed: %v", gameID, err)
		}

		drawMoves := engine.GenerateDrawSequence()
		var lastResp *pb.StepResponse
		for i, a := range drawMoves {
			resp, err := client.Step(ctx, &pb.StepRequest{GameId: gameID, Action: uint32(a)})
			if err != nil {
				t.Fatalf("draw move %d (action %d) failed: %v", i+1, a, err)
			}
			if i < 63 && resp.Done {
				t.Fatalf("draw game ended prematurely at move %d", i+1)
			}
			lastResp = resp
		}

		if !lastResp.Done || lastResp.Reward != 0.0 {
			t.Fatalf("expected Done=true Reward=0.0 on draw, got Done=%v Reward=%v", lastResp.Done, lastResp.Reward)
		}
		if lastResp.State.Status != uint32(engine.StatusDraw) || lastResp.State.Winner != uint32(engine.PlayerNone) {
			t.Fatalf("expected StatusDraw (3) and Winner=0, got status=%d winner=%d",
				lastResp.State.Status, lastResp.State.Winner)
		}
		if lastResp.State.MoveCount != 64 || lastResp.State.ValidMovesMask != 0 {
			t.Fatalf("expected MoveCount=64 and ValidMovesMask=0, got moves=%d mask=%04x",
				lastResp.State.MoveCount, lastResp.State.ValidMovesMask)
		}
	})
}

// TestGRPCIntegration_VectorizedPoolAndTCP verifies vectorized BatchStep, InitPool, ResetPool,
// and auto-reset across 77 parallel games over a TCP gRPC connection (76 winning lines + 1 draw game).
func TestGRPCIntegration_VectorizedPoolAndTCP(t *testing.T) {
	client, cleanup := setupLiveGRPC(t, "tcp://127.0.0.1:0")
	defer cleanup()

	ctx := context.Background()
	const numEnvs = 77 // Envs 0..75 play all 76 winning lines; Env 76 plays the 64-move draw sequence

	initResp, err := client.InitPool(ctx, &pb.InitPoolRequest{NumEnvs: numEnvs})
	if err != nil {
		t.Fatalf("InitPool(%d) failed: %v", numEnvs, err)
	}
	if initResp.NumEnvs != numEnvs || len(initResp.Observations) != numEnvs*engine.CanonicalTensorSize {
		t.Fatalf("unexpected InitPool response sizes: num=%d obs=%d", initResp.NumEnvs, len(initResp.Observations))
	}

	// Test ResetPool for specific env IDs and all env IDs
	if _, err := client.ResetPool(ctx, &pb.ResetPoolRequest{EnvIds: []uint32{0, 1}}); err != nil {
		t.Fatalf("ResetPool([0,1]) failed: %v", err)
	}
	if _, err := client.ResetPool(ctx, &pb.ResetPoolRequest{}); err != nil {
		t.Fatalf("ResetPool(all) failed: %v", err)
	}

	// Precompute the move trajectories for all 77 environments
	trajectories := make([][]uint8, numEnvs)
	for i := 0; i < 76; i++ {
		trajectories[i] = engine.GenerateWinSequence(i, engine.PlayerWhite)
	}
	drawSeq := engine.GenerateDrawSequence()
	trajectories[76] = drawSeq[:]

	// Track which environments have completed their first episode
	completedWin := make([]bool, 76)
	completedDraw := false
	cursors := make([]int, numEnvs)

	for step := 0; step < 64; step++ {
		actions := make([]byte, numEnvs)
		for i := 0; i < numEnvs; i++ {
			seq := trajectories[i]
			actions[i] = seq[cursors[i]%len(seq)]
			cursors[i]++
		}

		stepResp, err := client.BatchStep(ctx, &pb.BatchStepRequest{Actions: actions})
		if err != nil {
			t.Fatalf("BatchStep at step %d failed: %v", step, err)
		}

		for i := 0; i < 76; i++ {
			if !completedWin[i] && cursors[i] == len(trajectories[i]) {
				if !stepResp.Dones[i] || stepResp.Rewards[i] != 1.0 || stepResp.Winners[i] != uint32(engine.PlayerWhite) {
					t.Fatalf("env %d: expected Done=true Reward=1.0 Winner=WHITE at step %d, got done=%v reward=%v winner=%d",
						i, step, stepResp.Dones[i], stepResp.Rewards[i], stepResp.Winners[i])
				}
				if stepResp.WinLineMasks[i] != engine.WinMasks[i] {
					t.Fatalf("env %d: expected win mask %016x, got %016x", i, engine.WinMasks[i], stepResp.WinLineMasks[i])
				}
				// Verify auto-reset happened: valid_mask is restored to 0xFFFF and current_player is WHITE
				if stepResp.ValidMasks[i] != 0xFFFF || stepResp.CurrentPlayers[i] != uint32(engine.PlayerWhite) {
					t.Fatalf("env %d: expected auto-reset to initial state after win", i)
				}
				completedWin[i] = true
			}
		}

		if step == 63 {
			if !stepResp.Dones[76] || stepResp.Rewards[76] != 0.0 || stepResp.Winners[76] != uint32(engine.PlayerNone) {
				t.Fatalf("env 76 (draw): expected Done=true Reward=0.0 Winner=NONE at step 63, got done=%v reward=%v winner=%d",
					stepResp.Dones[76], stepResp.Rewards[76], stepResp.Winners[76])
			}
			completedDraw = true
		}
	}

	for i, ok := range completedWin {
		if !ok {
			t.Fatalf("env %d never completed its winning trajectory", i)
		}
	}
	if !completedDraw {
		t.Fatalf("env 76 never completed its 64-move draw trajectory")
	}
}
