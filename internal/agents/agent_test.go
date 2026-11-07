package agents

import (
	"errors"
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
	for _, name := range []string{"random", "heuristic", "alphabeta", "minimax", "mcts"} {
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
}
