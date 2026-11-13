package agents

import (
	"math"
	"math/bits"
	"math/rand"
	"testing"
	"time"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

// Positions taken from the Nov 2026 arena battle "Stupido vs deep (s6deep1113)" where the previous
// deep baseline turned a proven win/draw into a proven loss (verified with an exact solver).
var deepRegressionPositions = []struct {
	name    string
	moves   []uint8
	correct []uint8 // moves that preserve the proven game value
	value   int     // proven value for the side to move
	// heavy positions need more solver throughput than a race-instrumented build provides; they
	// are verified by the regular (non -race) test run.
	heavy bool
}{
	{"s6deep-b2-ply30-win", []uint8{6, 2, 7, 15, 6, 6, 12, 0, 5, 4, 5, 3, 1, 5, 1, 4, 9, 13, 9, 13, 9, 9, 4, 12, 8, 8, 0, 10, 8, 15}, []uint8{0, 1, 3, 4, 5, 7, 8, 11, 12, 13, 15}, 1, false},
	{"s6deep-w6-ply31-draw", []uint8{1, 0, 15, 12, 3, 7, 1, 4, 8, 5, 6, 4, 1, 1, 5, 4, 4, 5, 0, 0, 15, 14, 0, 15, 15, 7, 7, 6, 14, 5, 6}, []uint8{14}, 0, true},
	{"s6deep-w8-ply33-draw", []uint8{9, 9, 0, 6, 6, 5, 5, 6, 12, 8, 5, 2, 14, 13, 9, 2, 7, 8, 15, 3, 0, 15, 7, 0, 7, 7, 0, 2, 2, 8, 8, 1, 9}, []uint8{6, 12, 14}, 0, true},
}

func gameFromMoves(t *testing.T, moves []uint8) *engine.Game {
	t.Helper()
	g := engine.NewGame()
	for _, m := range moves {
		if _, err := g.ApplyMove(m); err != nil {
			t.Fatalf("invalid move %d: %v", m, err)
		}
	}
	return g
}

func sideBoards(g *engine.Game) (uint64, uint64) {
	if g.CurrentPlayer == engine.PlayerBlack {
		return g.Black, g.White
	}
	return g.White, g.Black
}

// referenceSolve is a deliberately naive exact solver (full threat recomputation, no TT, no
// poison/stacked-fork shortcuts) used to validate wdlSolver.
func referenceSolve(my, opp uint64, alpha, beta int) int {
	occ := my | opp
	if occ == engine64Full {
		return 0
	}
	sf := surfaceBitmaskFast(occ)
	mt, ot := threatCellsPairBitboard(my, opp)
	if mt&sf != 0 {
		return 1
	}
	oi := ot & sf
	if bits.OnesCount64(oi) >= 2 {
		return -1
	}
	if oi != 0 {
		return -referenceSolve(opp, my|oi, -beta, -alpha)
	}
	best := -2
	for m := sf; m != 0; m &= m - 1 {
		v := -referenceSolve(opp, my|(m&-m), -beta, -alpha)
		if v > best {
			best = v
		}
		if best > alpha {
			alpha = best
		}
		if alpha >= beta {
			break
		}
	}
	return best
}

// randomLatePosition plays random non-terminal moves until `stones` stones are on the board.
func randomLatePosition(r *rand.Rand, stones int) *engine.Game {
	for {
		g := engine.NewGame()
		for g.Status == engine.StatusOngoing && int(g.MoveCount) < stones {
			v := ValidActions(g)
			_, _ = g.ApplyMove(v[r.Intn(len(v))])
		}
		if g.Status == engine.StatusOngoing {
			return g
		}
	}
}

func TestWDLSolverMatchesReference(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	solver := newWDLSolver(time.Now().Add(time.Minute))
	defer solver.release()
	counts := map[int]int{}
	for i := 0; i < 60; i++ {
		g := randomLatePosition(r, 52+i%3)
		my, opp := sideBoards(g)
		mt, ot := threatCellsPairBitboard(my, opp)
		got, ok := solver.solveLimited(my, opp, mt, ot, colorHash(g.White, g.Black), g.MoveCount, 0)
		if !ok {
			t.Fatalf("solver aborted unexpectedly")
		}
		want := referenceSolve(my, opp, -1, 1)
		if got != want {
			t.Fatalf("position %d: solver=%d reference=%d", i, got, want)
		}
		counts[got]++
		// Null-window answers must agree with the exact value.
		if v, ok := solver.solveWindow(my, opp, mt, ot, colorHash(g.White, g.Black), g.MoveCount, 0, 1, 0); !ok || (v >= 1) != (want == 1) {
			t.Fatalf("position %d: win-test=%d exact=%d", i, v, want)
		}
		if v, ok := solver.solveWindow(my, opp, mt, ot, colorHash(g.White, g.Black), g.MoveCount, -1, 0, 0); !ok || (v <= -1) != (want == -1) {
			t.Fatalf("position %d: loss-test=%d exact=%d", i, v, want)
		}
	}
	t.Logf("value distribution: %v", counts)
}

func TestWDLSolverLimitsAndTimeout(t *testing.T) {
	g := gameFromMoves(t, deepRegressionPositions[0].moves)
	my, opp := sideBoards(g)
	mt, ot := threatCellsPairBitboard(my, opp)
	h := colorHash(g.White, g.Black)

	s := newWDLSolver(time.Now().Add(time.Minute))
	if _, ok := s.solveLimited(my, opp, mt, ot, h, g.MoveCount, 50); ok {
		t.Fatalf("expected a 50-node limited solve to be interrupted")
	}
	if s.timedOut {
		t.Fatalf("node limit must not be reported as timeout")
	}
	s.release()
	s.release() // idempotent

	expired := newWDLSolver(time.Now().Add(-time.Second))
	defer expired.release()
	expired.nodes = 1023 // next node triggers the clock check
	if _, ok := expired.solveLimited(my, opp, mt, ot, h, g.MoveCount, 0); ok || !expired.timedOut {
		t.Fatalf("expected the solve to time out")
	}
	if _, ok := expired.solveLimited(my, opp, mt, ot, h, g.MoveCount, 0); ok {
		t.Fatalf("a timed-out solver must refuse further work")
	}
}

func TestWDLSolverProvesRegressionPositions(t *testing.T) {
	for _, tc := range deepRegressionPositions {
		if tc.heavy && raceEnabled {
			t.Logf("%s: skipped under -race (tens of millions of instrumented nodes)", tc.name)
			continue
		}
		g := gameFromMoves(t, tc.moves)
		my, opp := sideBoards(g)
		mt, ot := threatCellsPairBitboard(my, opp)
		s := newWDLSolver(time.Now().Add(2 * time.Minute))
		v, ok := s.solveLimited(my, opp, mt, ot, colorHash(g.White, g.Black), g.MoveCount, 0)
		s.release()
		if !ok || v != tc.value {
			t.Fatalf("%s: expected proven value %d, got %d (ok=%v)", tc.name, tc.value, v, ok)
		}
	}
}

func TestDeepAgentFixesArenaBlunders(t *testing.T) {
	if raceEnabled {
		// Race instrumentation slows search by up to ~40x on CI runners, which makes a timed
		// 2.8s decision meaningless; the solver proofs are still checked by
		// TestWDLSolverProvesRegressionPositions and this test runs in regular builds.
		t.Skip("timed search check skipped under -race")
	}
	for _, tc := range deepRegressionPositions {
		g := gameFromMoves(t, tc.moves)
		ag := NewDeepSearchAgent(DefaultDeepTimeBudget, DefaultDeepMinDepth)
		ag.EnforceFullTime = false
		act, policy, val, err := ag.SearchWithPolicy(g)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		ok := false
		for _, c := range tc.correct {
			ok = ok || c == act
		}
		t.Logf("%s: act=%d val=%.2f proven=%d depth=%d nodes=%d solver=%d elapsed=%v",
			tc.name, act, val, ag.LastProvenValue, ag.LastDepthReached, ag.LastNodesVisited, ag.LastSolverNodes, ag.LastElapsed)
		if !ok {
			t.Fatalf("%s: deep chose %d, which loses; correct moves %v", tc.name, act, tc.correct)
		}
		if ag.LastElapsed > 3*time.Second {
			t.Fatalf("%s: move took %v, above the 3s cap", tc.name, ag.LastElapsed)
		}
		var sum float32
		for _, p := range policy {
			sum += p
		}
		if math.Abs(float64(sum-1)) > 1e-4 {
			t.Fatalf("%s: policy sums to %f", tc.name, sum)
		}
	}
}

func TestDeepAgentLostAndForcedPositions(t *testing.T) {
	// White to move with 0,1,2 on row y=0: immediate win at 3.
	g := gameFromMoves(t, []uint8{1, 5, 2, 6, 0, 9})
	ag := NewDeepSearchAgent(50*time.Millisecond, 4)
	ag.EnforceFullTime = false
	act, _, _, err := ag.SearchWithPolicy(g)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if act != 3 || ag.LastProvenValue != 1 {
		t.Fatalf("expected immediate win at 3, got %d (proven=%d)", act, ag.LastProvenValue)
	}

	// Black to move facing White's double threat (cells 3 and 12 at z=0).
	g2 := gameFromMoves(t, []uint8{0, 5, 1, 6, 2, 10, 4, 11, 8})
	act2, _, val2, err := ag.SearchWithPolicy(g2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if val2 != -1.0 || ag.LastProvenValue != -1 || (act2 != 3 && act2 != 12) {
		t.Fatalf("expected a hopeless block with val=-1, got act=%d val=%f proven=%d", act2, val2, ag.LastProvenValue)
	}

	// Single forced block is chosen and searched.
	g3 := gameFromMoves(t, []uint8{0, 5, 1, 6, 2})
	act3, _, _, err := ag.SearchWithPolicy(g3)
	if err != nil || act3 != 3 {
		t.Fatalf("expected forced block at 3, got %d err=%v", act3, err)
	}
}

func TestDeepAgentSolverToggleAndEnforcedTime(t *testing.T) {
	g := gameFromMoves(t, deepRegressionPositions[2].moves)
	ag := NewDeepSearchAgent(600*time.Millisecond, 4)
	if !ag.EnforceFullTime {
		t.Fatalf("budgets >= 500ms should enforce full time")
	}
	ag.SolverFromPly = -1
	if _, _, _, err := ag.SearchWithPolicy(g); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ag.LastSolverNodes != 0 || ag.LastProvenValue != wdlUnknown {
		t.Fatalf("solver should be disabled, got nodes=%d proven=%d", ag.LastSolverNodes, ag.LastProvenValue)
	}
	if ag.LastElapsed < 590*time.Millisecond || ag.LastElapsed > 900*time.Millisecond {
		t.Fatalf("expected ~600ms with EnforceFullTime, got %v", ag.LastElapsed)
	}
}

func TestParityThreatScore(t *testing.T) {
	if s := parityThreatScore(0, 0, 0xFFFF, 10); s != 0 {
		t.Fatalf("no threats should score 0, got %d", s)
	}
	surface := uint64(0xFFFF)          // all columns empty: surface is z=0
	whiteGood := uint64(1) << (32 + 5) // White threat at z=2 in column 5
	blackGood := uint64(1) << (16 + 6) // Black threat at z=1 in column 6
	// White to move (even move count): my=White.
	sW := parityThreatScore(whiteGood, 0, surface, 20)
	if sW <= 0 {
		t.Fatalf("White good-parity threat should favour White, got %d", sW)
	}
	// Same position, Black to move: score is negated.
	if sB := parityThreatScore(0, whiteGood, surface, 21); sB != -sW {
		t.Fatalf("expected perspective symmetry, got %d vs %d", sB, sW)
	}
	// Black good-parity threat favours Black (Black to move).
	if s := parityThreatScore(blackGood, 0, surface, 21); s <= 0 {
		t.Fatalf("Black good-parity threat should favour Black, got %d", s)
	}
	// Bad-parity threats count less than good-parity ones.
	whiteBad := uint64(1) << (16 + 5) // White at z=1
	if parityThreatScore(whiteBad, 0, surface, 20) >= sW {
		t.Fatalf("bad-parity threat should be worth less than a good-parity one")
	}
	blackBad := uint64(1) << (32 + 6) // Black at z=2
	if s := parityThreatScore(0, blackBad, surface, 20); s >= 0 {
		t.Fatalf("Black bad-parity threat should still slightly favour Black, got %d", s)
	}
	// Shared lowest threat cell is neutral.
	shared := uint64(1) << (32 + 9)
	if s := parityThreatScore(shared, shared, surface, 20); s != 0 {
		t.Fatalf("shared threat cell should be neutral, got %d", s)
	}
	// Both sides holding good threats: no exclusive bonus.
	if s := parityThreatScore(whiteGood, blackGood, surface, 20); s != 0 {
		t.Fatalf("balanced good threats should cancel, got %d", s)
	}
}

func TestDeepRootPolicyAndOrdering(t *testing.T) {
	var scores [16]int
	var proven [16]int
	for i := range proven {
		proven[i] = wdlUnknown
	}
	cands := []uint8{1, 2, 3, 4}
	scores[1], scores[2], scores[3], scores[4] = 10, 50, -20, 30
	order := orderByScore(cands, scores, 3)
	if order[0] != 3 || order[1] != 2 || order[2] != 4 || order[3] != 1 {
		t.Fatalf("unexpected order %v", order)
	}

	p := deepRootPolicy(cands, scores, proven, 2)
	if p[2] <= p[4] || p[4] <= p[1] || p[1] <= p[3] {
		t.Fatalf("policy should follow scores: %v", p)
	}

	proven[3] = 1
	p = deepRootPolicy(cands, scores, proven, 3)
	if p[3] != 1 {
		t.Fatalf("proven win should take all mass: %v", p)
	}

	proven[3] = -1
	proven[2] = 0
	scores[1], scores[4] = -500, -600
	p = deepRootPolicy(cands, scores, proven, 2)
	if p[2] < 0.99 || p[3] != 0 {
		t.Fatalf("proven draw should dominate negative unknowns and proven losses: %v", p)
	}

	// All candidates hopeless: falls back to the best column.
	for _, c := range cands {
		proven[c] = wdlUnknown
	}
	scores[1], scores[2], scores[3], scores[4] = ScoreLoss*2, ScoreLoss*2, ScoreLoss*2, ScoreLoss*2
	p = deepRootPolicy(cands, scores, proven, 4)
	var sum float32
	for _, v := range p {
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-4 {
		t.Fatalf("policy must stay normalised: %v", p)
	}
}

func TestDeepAgentPreservesSolvedValue(t *testing.T) {
	// On late positions the solver proves the root value; the chosen move must preserve it.
	r := rand.New(rand.NewSource(7))
	ag := NewDeepSearchAgent(300*time.Millisecond, 4)
	ag.EnforceFullTime = false
	ag.SolverFromPly = 0
	seen := map[int]int{}
	for i := 0; i < 24; i++ {
		g := randomLatePosition(r, 50+i%4)
		my, opp := sideBoards(g)
		root := referenceSolve(my, opp, -1, 1)
		act, _, _, err := ag.SearchWithPolicy(g)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		c := g.Clone()
		_, _ = c.ApplyMove(act)
		got := 1
		if c.Status == engine.StatusDraw {
			got = 0
		} else if c.Status == engine.StatusOngoing {
			cm, co := sideBoards(c)
			got = -referenceSolve(cm, co, -1, 1)
		}
		if got != root {
			t.Fatalf("position %d: root value %d but chosen move %d leads to %d", i, root, act, got)
		}
		seen[root]++
	}
	t.Logf("root values seen: %v", seen)
}
