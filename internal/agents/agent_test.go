package agents

import (
	"errors"
	"math"
	"testing"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

func TestCellLineCountsGeometry(t *testing.T) {
	count7 := 0
	count4 := 0
	for bit := 0; bit < engine.GridCells; bit++ {
		switch CellLineCounts[bit] {
		case 7:
			count7++
		case 4:
			count4++
		default:
			t.Fatalf("unexpected line count %d at bit %d", CellLineCounts[bit], bit)
		}
	}
	if count7 != 16 || count4 != 48 {
		t.Fatalf("expected 16 cells with 7 lines and 48 cells with 4 lines, got 7s=%d 4s=%d", count7, count4)
	}
}

func TestAgentFactoryAndTerminalErrors(t *testing.T) {
	for _, name := range []string{"random", "heuristic", "alphabeta", "minimax", "deep", "alphabeta-deep", "mcts", "ttt3d-deep-gray1", "ttt3d-deep-gray1-pure", "ttt3dnet", "ttt3dnet-pure"} {
		ag, err := NewAgentByName(name, 42)
		if err != nil || ag == nil || ag.Name() == "" {
			t.Fatalf("NewAgentByName(%q) failed: %v", name, err)
		}
	}
	if _, err := NewAgentByName("unknown-bot", 1); !errors.Is(err, ErrUnknownAgent) {
		t.Fatalf("expected ErrUnknownAgent, got %v", err)
	}

	// Constructor default fallbacks for <= 0 depth/simulations
	abDefault := NewAlphaBetaAgent(0)
	if abDefault.MaxDepth != 3 {
		t.Fatalf("expected default MaxDepth=3, got %d", abDefault.MaxDepth)
	}
	mctsDefault := NewMCTSAgent(0, 1)
	if mctsDefault.Simulations != 250 {
		t.Fatalf("expected default Simulations=250, got %d", mctsDefault.Simulations)
	}

	// Finished game must return ErrNoValidMoves for all agents
	g := engine.NewGame()
	for _, a := range engine.GenerateDrawSequence() {
		_, _ = g.ApplyMove(a)
	}
	for _, ag := range []Agent{
		NewRandomAgent(1),
		NewHeuristicAgent(),
		NewAlphaBetaAgent(2),
		NewMCTSAgent(50, 1),
	} {
		if _, err := ag.SelectAction(g); !errors.Is(err, ErrNoValidMoves) {
			t.Fatalf("agent %s: expected ErrNoValidMoves on terminal game, got %v", ag.Name(), err)
		}
	}
}

func TestAll76LinesImmediateWinAndBlock(t *testing.T) {
	tacticalAgents := []Agent{
		NewHeuristicAgent(),
		NewAlphaBetaAgent(2),
		NewMCTSAgent(60, 7),
		NewTTT3DNetAgent(false),
	}

	for lineIdx := 0; lineIdx < engine.TotalWinLines; lineIdx++ {
		moves := engine.GenerateWinSequence(lineIdx, engine.PlayerWhite)
		winningAction := moves[len(moves)-1]

		// 1. State 1 ply before White wins: it is White's turn, and `winningAction` wins immediately
		gWin := engine.NewGame()
		for i := 0; i < len(moves)-1; i++ {
			_, _ = gWin.ApplyMove(moves[i])
		}
		for _, ag := range tacticalAgents {
			act, err := ag.SelectAction(gWin)
			if err != nil {
				t.Fatalf("line %d (%s): SelectAction failed: %v", lineIdx, ag.Name(), err)
			}
			// Verify the chosen action immediately wins for White
			testG := gWin.Clone()
			_, _ = testG.ApplyMove(act)
			if testG.Status != engine.StatusWhiteWon {
				t.Fatalf("line %d: agent %s failed to take immediate 1-ply win (picked %d, expected win like %d)",
					lineIdx, ag.Name(), act, winningAction)
			}
		}

		// 2. Flip CurrentPlayer to Black in the exact same board state:
		// If Black does not already have its own immediate 1-ply win, Black MUST block White's 1-ply win!
		gBlock := gWin.Clone()
		gBlock.CurrentPlayer = engine.PlayerBlack
		blackWins := FindImmediateWinningActions(gBlock, engine.PlayerBlack)
		oppWins := FindImmediateWinningActions(gBlock, engine.PlayerWhite)
		if len(blackWins) == 0 && len(oppWins) == 1 {
			for _, ag := range tacticalAgents {
				act, err := ag.SelectAction(gBlock)
				if err != nil {
					t.Fatalf("line %d (%s): block SelectAction failed: %v", lineIdx, ag.Name(), err)
				}
				if act != oppWins[0] {
					t.Fatalf("line %d: agent %s failed to block opponent 1-ply win at %d (picked %d)",
						lineIdx, ag.Name(), oppWins[0], act)
				}
			}
		}
	}
}

func TestPoisonColumnAvoidance(t *testing.T) {
	// Construct a board where Black has 3-in-a-row at z=1: (0,0,1), (1,0,1), (2,0,1)
	// and column (3,0) (action 3) is currently empty at z=0!
	// If White plays action 3 (landing at (3,0,0)), Black will immediately play action 3 (landing at (3,0,1)) and win!
	// Therefore, action 3 is a "poison column" that White must avoid.
	g := engine.NewGame()
	setup := []struct {
		col    uint8
		player engine.Player
	}{
		{0, engine.PlayerWhite}, // (0,0,0) W
		{0, engine.PlayerBlack}, // (0,0,1) B
		{1, engine.PlayerWhite}, // (1,0,0) W
		{1, engine.PlayerBlack}, // (1,0,1) B
		{4, engine.PlayerWhite}, // (0,1,0) W
		{2, engine.PlayerBlack}, // (2,0,0) B
		{5, engine.PlayerWhite}, // (1,1,0) W
		{2, engine.PlayerBlack}, // (2,0,1) B -> Black now has (0,0,1), (1,0,1), (2,0,1)!
	}
	for _, s := range setup {
		if g.CurrentPlayer != s.player {
			t.Fatalf("unexpected player turn in setup")
		}
		if _, err := g.ApplyMove(s.col); err != nil {
			t.Fatalf("setup move failed: %v", err)
		}
	}

	// Verify action 3 is indeed a poison column for White
	if !GivesOpponentImmediateWin(g, 3) {
		t.Fatalf("expected column 3 to be detected as a poison move giving Black an immediate win at (3,0,1)")
	}

	for _, ag := range []Agent{
		NewHeuristicAgent(),
		NewAlphaBetaAgent(3),
		NewMCTSAgent(150, 99),
		NewTTT3DNetAgent(true),
		NewTTT3DNetAgent(false),
	} {
		act, err := ag.SelectAction(g)
		if err != nil {
			t.Fatalf("%s SelectAction failed: %v", ag.Name(), err)
		}
		if act == 3 {
			t.Fatalf("agent %s blundered into poison column 3!", ag.Name())
		}
	}
}

func playMatch(white, black Agent) engine.Player {
	g := engine.NewGame()
	for g.Status == engine.StatusOngoing {
		var act uint8
		if g.CurrentPlayer == engine.PlayerWhite {
			act, _ = white.SelectAction(g)
		} else {
			act, _ = black.SelectAction(g)
		}
		_, _ = g.ApplyMove(act)
	}
	return g.Winner
}

func TestArenaWinRatesAgainstRandom(t *testing.T) {
	const gamesPerSide = 10 // 20 games total per agent (10 as White, 10 as Black)

	challengers := []Agent{
		NewHeuristicAgent(),
		NewAlphaBetaAgent(2),
		NewMCTSAgent(80, 12345),
	}

	for _, challenger := range challengers {
		wins := 0
		total := gamesPerSide * 2
		for i := 0; i < gamesPerSide; i++ {
			rndWhite := NewRandomAgent(int64(100 + i))
			rndBlack := NewRandomAgent(int64(200 + i))

			if winner := playMatch(challenger, rndBlack); winner == engine.PlayerWhite {
				wins++
			}
			if winner := playMatch(rndWhite, challenger); winner == engine.PlayerBlack {
				wins++
			}
		}

		winRate := float64(wins) / float64(total)
		if winRate < 0.90 {
			t.Fatalf("agent %s achieved only %.1f%% (%d/%d) win rate vs Random, expected >= 90%%",
				challenger.Name(), winRate*100, wins, total)
		}
	}

	// Verify pure TTT3DNetAgent wins 100% (4/4) against Random (kept concise for -race speed)
	pureNN := NewTTT3DNetAgent(false)
	for i := 0; i < 2; i++ {
		if winner := playMatch(pureNN, NewRandomAgent(int64(300+i))); winner != engine.PlayerWhite {
			t.Fatalf("pure TTT3DNetAgent failed to win as White vs Random (seed %d)", 300+i)
		}
		if winner := playMatch(NewRandomAgent(int64(400+i)), pureNN); winner != engine.PlayerBlack {
			t.Fatalf("pure TTT3DNetAgent failed to win as Black vs Random (seed %d)", 400+i)
		}
	}
}

func TestTTT3DModelParityAndTacticalPlay(t *testing.T) {
	model, err := DefaultTTT3DModel()
	if err != nil {
		t.Fatalf("DefaultTTT3DModel failed: %v", err)
	}
	if model.Channels != 32 || model.NumBlocks != 3 {
		t.Fatalf("expected Channels=32, NumBlocks=3, got %d, %d", model.Channels, model.NumBlocks)
	}

	// Verify exact parity with PyTorch reference forward pass on sample tensor
	var input [3 * 64]float32
	input[0*64+0] = 1.0 // obs[0, 0, 0, 0] = 1.0
	input[1*64+5] = 1.0 // obs[1, 0, 1, 1] = 1.0
	for i := 0; i < 16; i++ {
		input[2*64+i] = 1.0 // gravity surface z=0
	}
	input[2*64+0] = 0.0
	input[2*64+16+0] = 1.0
	input[2*64+5] = 0.0
	input[2*64+16+5] = 1.0

	logits, val := model.Forward(input)
	expectedLogits := [4]float32{0.498583, -1.392628, -1.506592, 1.749612}
	for i := 0; i < 4; i++ {
		if math.Abs(float64(logits[i]-expectedLogits[i])) > 1e-4 {
			t.Fatalf("logit[%d] mismatch: got %.6f, expected %.6f", i, logits[i], expectedLogits[i])
		}
	}
	if math.Abs(float64(val - -0.028315)) > 1e-4 {
		t.Fatalf("value mismatch: got %.6f, expected -0.028315", val)
	}

	// Verify dormant stacked vertical fork detection (z=1 and z=2 in col 0 while col 0 is empty at h=0)
	gDormant := engine.NewGame()
	// White has 3-in-a-row at z=1: (1,0,1), (2,0,1), (3,0,1) -> missing (0,0,1) (bit 16)
	// White also has 3-in-a-row at z=2: (1,0,2), (2,0,2), (3,0,2) -> missing (0,0,2) (bit 32)
	gDormant.White = (uint64(1) << 17) | (uint64(1) << 18) | (uint64(1) << 19) |
		(uint64(1) << 33) | (uint64(1) << 34) | (uint64(1) << 35)
	gDormant.Black = (uint64(1) << 1) | (uint64(1) << 2) | (uint64(1) << 3)
	if !hasUnstoppableThreat(gDormant, engine.PlayerWhite) {
		t.Fatal("expected hasUnstoppableThreat to detect dormant stacked vertical fork at z=1 & z=2 above h=0")
	}

	// Test ParseFusedWeights error branches
	if _, err := ParseFusedWeights([]byte("short")); err == nil {
		t.Fatal("expected error on short weight buffer")
	}
	if _, err := ParseFusedWeights([]byte("BADMAGIC12345678")); err == nil {
		t.Fatal("expected error on bad magic header")
	}
	badVer := make([]byte, 20)
	copy(badVer[:4], "3T3D")
	badVer[4] = 99
	if _, err := ParseFusedWeights(badVer); err == nil {
		t.Fatal("expected error on unsupported version")
	}
	badDims := make([]byte, 20)
	copy(badDims[:4], "3T3D")
	badDims[4] = 1
	if _, err := ParseFusedWeights(badDims); err == nil {
		t.Fatal("expected error on zero dimensions")
	}
	truncated := make([]byte, 20)
	copy(truncated[:4], "3T3D")
	truncated[4] = 1
	truncated[8] = 32
	truncated[12] = 2
	if _, err := ParseFusedWeights(truncated); err == nil {
		t.Fatal("expected error on truncated payload")
	}

	// Test Pure vs Hybrid TTT3DNetAgent on immediate win, block, and ongoing board
	hybrid := NewTTT3DNetAgent(true)
	pure := NewTTT3DNetAgent(false)
	if hybrid.Name() != "ttt3dnet" || pure.Name() != "ttt3dnet-pure" {
		t.Fatalf("unexpected names: %s, %s", hybrid.Name(), pure.Name())
	}

	g := engine.NewGame()
	actPure, probsPure, valPure, err := pure.SelectActionWithPolicy(g)
	if err != nil || actPure >= 16 || valPure < -1.0 || valPure > 1.0 {
		t.Fatalf("pure SelectActionWithPolicy failed: act=%d val=%f err=%v", actPure, valPure, err)
	}
	var sumP float32
	for _, p := range probsPure {
		sumP += p
	}
	if math.Abs(float64(sumP-1.0)) > 1e-4 {
		t.Fatalf("expected probabilities to sum to 1.0, got %f", sumP)
	}

	// Immediate win on action 3
	gWin := engine.NewGame()
	for _, m := range []uint8{0, 4, 1, 5, 2, 6} {
		_, _ = gWin.ApplyMove(m)
	}
	if act, err := hybrid.SelectAction(gWin); err != nil || act != 3 {
		t.Fatalf("expected hybrid to take winning move 3, got %d (err=%v)", act, err)
	}

	// Immediate block on action 3
	gBlock := engine.NewGame()
	for _, m := range []uint8{0, 4, 1, 5, 2} {
		_, _ = gBlock.ApplyMove(m)
	}
	if act, err := hybrid.SelectAction(gBlock); err != nil || act != 3 {
		t.Fatalf("expected hybrid to block on move 3, got %d (err=%v)", act, err)
	}

	// Ongoing normal move selection via hybrid lookahead
	gMid := engine.NewGame()
	_, _ = gMid.ApplyMove(5)
	if act, err := hybrid.SelectAction(gMid); err != nil || act >= 16 {
		t.Fatalf("hybrid failed on mid-game state: act=%d err=%v", act, err)
	}

	// Terminal state returns ErrNoValidMoves
	gDone := engine.NewGame()
	for _, a := range engine.GenerateDrawSequence() {
		_, _ = gDone.ApplyMove(a)
	}
	if _, err := hybrid.SelectAction(gDone); !errors.Is(err, ErrNoValidMoves) {
		t.Fatalf("expected ErrNoValidMoves on finished game, got %v", err)
	}
}

func TestDeepSearchAgentAdaptiveDepthAndTiming(t *testing.T) {
	// 1. Default ~1.5s adaptive deep search agent reaches depth >= 15 in ~1.5s
	deepDefault := NewDeepSearchAgent(0, 0)
	if deepDefault.Name() != "deep" || deepDefault.MinDepth != 15 {
		t.Fatalf("unexpected default DeepSearchAgent config: name=%s minDepth=%d", deepDefault.Name(), deepDefault.MinDepth)
	}

	g := engine.NewGame()
	for _, m := range []uint8{5, 6, 9, 10, 0, 3} {
		_, _ = g.ApplyMove(m)
	}
	act, policy, val, err := deepDefault.SearchWithPolicy(g)
	if err != nil || act >= 16 {
		t.Fatalf("DeepSearchAgent.SearchWithPolicy failed: act=%d err=%v", act, err)
	}
	t.Logf("DeepSearchAgent default move: act=%d depth=%d nodes=%d elapsed=%v val=%.3f",
		act, deepDefault.LastDepthReached, deepDefault.LastNodesVisited, deepDefault.LastElapsed, val)
	if deepDefault.LastDepthReached < 15 {
		t.Fatalf("expected DeepSearchAgent to reach depth >= 15, got %d", deepDefault.LastDepthReached)
	}
	if deepDefault.LastElapsed < 1200_000_000 || deepDefault.LastElapsed > 2200_000_000 {
		t.Fatalf("expected DeepSearchAgent move computation around 1.5s, got %v", deepDefault.LastElapsed)
	}
	var pSum float32
	for _, p := range policy {
		pSum += p
	}
	if math.Abs(float64(pSum-1.0)) > 1e-4 {
		t.Fatalf("expected root policy to sum to 1.0, got %f", pSum)
	}

	// 2. Fast teacher mode (EnforceFullTime=false) also reaches depth >= 15 and solves 1-ply win/block/terminal
	fastDeep := NewDeepSearchAgent(100_000_000, 15)
	fastDeep.EnforceFullTime = false

	gWin := engine.NewGame()
	for _, m := range []uint8{0, 4, 1, 5, 2, 6} {
		_, _ = gWin.ApplyMove(m)
	}
	if wAct, _, wVal, err := fastDeep.SearchWithPolicy(gWin); err != nil || wAct != 3 || wVal != 1.0 {
		t.Fatalf("expected immediate win on 3 with val=1.0, got act=%d val=%f err=%v", wAct, wVal, err)
	}

	gBlock := engine.NewGame()
	for _, m := range []uint8{0, 4, 1, 5, 2} {
		_, _ = gBlock.ApplyMove(m)
	}
	if bAct, _, _, err := fastDeep.SearchWithPolicy(gBlock); err != nil || bAct != 3 {
		t.Fatalf("expected forced block on 3, got act=%d err=%v", bAct, err)
	}

	gDone := engine.NewGame()
	for _, a := range engine.GenerateDrawSequence() {
		_, _ = gDone.ApplyMove(a)
	}
	if _, err := fastDeep.SelectAction(gDone); !errors.Is(err, ErrNoValidMoves) {
		t.Fatalf("expected ErrNoValidMoves on terminal game, got %v", err)
	}
}
