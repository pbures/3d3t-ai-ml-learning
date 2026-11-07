package agents

import (
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"sync"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

var (
	ErrNoValidMoves = errors.New("no valid moves available in terminal or full board state")
	ErrUnknownAgent = errors.New("unknown agent type")
)

// Agent defines a decision-making player that selects a valid column action in {0..15} for a given 3T3D game state.
type Agent interface {
	Name() string
	SelectAction(g *engine.Game) (uint8, error)
}

// CellLineCounts stores the exact number of winning lines (either 7 or 4) passing through each bit index 0..63.
var CellLineCounts [engine.GridCells]int

func init() {
	for _, mask := range engine.WinMasks {
		for bit := uint8(0); bit < engine.GridCells; bit++ {
			if (mask & (uint64(1) << bit)) != 0 {
				CellLineCounts[bit]++
			}
		}
	}
}

// ValidActions returns a slice of all currently legal column actions in {0..15}.
func ValidActions(g *engine.Game) []uint8 {
	if g == nil || g.Status != engine.StatusOngoing {
		return nil
	}
	mask := g.ValidMovesMask()
	actions := make([]uint8, 0, engine.MaxActions)
	for a := uint8(0); a < engine.MaxActions; a++ {
		if (mask & (uint16(1) << a)) != 0 {
			actions = append(actions, a)
		}
	}
	return actions
}

// LandingMask returns a 64-bit bitmask with up to 16 bits set at the exact (x, y, z) landing
// positions for all currently valid column moves.
func LandingMask(g *engine.Game) uint64 {
	var surface uint64
	occupied := g.White | g.Black
	for a := uint8(0); a < engine.MaxActions; a++ {
		for z := uint8(0); z < engine.GridDim; z++ {
			bit := a + 16*z
			if (occupied & (uint64(1) << bit)) == 0 {
				surface |= uint64(1) << bit
				break
			}
		}
	}
	return surface
}

// FindImmediateWinningActions returns all valid column actions in {0..15} that would immediately
// complete a 4-in-a-row if played right now by `player`.
func FindImmediateWinningActions(g *engine.Game, player engine.Player) []uint8 {
	if g == nil || g.Status != engine.StatusOngoing {
		return nil
	}
	playerBoard := g.White
	if player == engine.PlayerBlack {
		playerBoard = g.Black
	}

	var wins []uint8
	for _, a := range ValidActions(g) {
		z, ok := g.LandingZ(a)
		if !ok {
			continue
		}
		bit := a + 16*z
		testBoard := playerBoard | (uint64(1) << bit)
		if won, _, _ := engine.CheckBitboardWin(testBoard); won {
			wins = append(wins, a)
		}
	}
	return wins
}

// GivesOpponentImmediateWin returns true if playing `action` for `g.CurrentPlayer` results in a non-winning
// state where the opponent now has at least one immediate 1-ply winning move (e.g. a poison column at z+1).
func GivesOpponentImmediateWin(g *engine.Game, action uint8) bool {
	clone := g.Clone()
	if _, err := clone.ApplyMove(action); err != nil {
		return true
	}
	if clone.Status != engine.StatusOngoing {
		return false
	}
	oppWins := FindImmediateWinningActions(clone, clone.CurrentPlayer)
	return len(oppWins) > 0
}

// RandomAgent selects uniformly at random among all legal column actions.
type RandomAgent struct {
	mu  sync.Mutex
	rng *rand.Rand
}

// NewRandomAgent creates a RandomAgent with a deterministic seed.
func NewRandomAgent(seed int64) *RandomAgent {
	return &RandomAgent{
		rng: rand.New(rand.NewSource(seed)),
	}
}

func (a *RandomAgent) Name() string {
	return "random"
}

func (a *RandomAgent) SelectAction(g *engine.Game) (uint8, error) {
	valid := ValidActions(g)
	if len(valid) == 0 {
		return 0, ErrNoValidMoves
	}
	a.mu.Lock()
	idx := a.rng.Intn(len(valid))
	a.mu.Unlock()
	return valid[idx], nil
}

// HeuristicAgent implements a fast tactical + positional agent:
// 1. Takes any immediate 1-ply winning move.
// 2. Blocks any immediate 1-ply opponent winning threat.
// 3. Filters out "poison column" moves that hand the opponent a 1-ply win.
// 4. Evaluates remaining moves using 76-line potential + cell centrality.
type HeuristicAgent struct{}

// NewHeuristicAgent creates a new HeuristicAgent.
func NewHeuristicAgent() *HeuristicAgent {
	return &HeuristicAgent{}
}

func (a *HeuristicAgent) Name() string {
	return "heuristic"
}

func (a *HeuristicAgent) SelectAction(g *engine.Game) (uint8, error) {
	valid := ValidActions(g)
	if len(valid) == 0 {
		return 0, ErrNoValidMoves
	}

	me := g.CurrentPlayer
	opp := me.Opponent()

	// 1. Immediate 1-ply win
	if myWins := FindImmediateWinningActions(g, me); len(myWins) > 0 {
		return myWins[0], nil
	}

	// 2. Immediate 1-ply forced block against opponent threat
	if oppWins := FindImmediateWinningActions(g, opp); len(oppWins) > 0 {
		for _, blockAction := range oppWins {
			if !GivesOpponentImmediateWin(g, blockAction) {
				return blockAction, nil
			}
		}
		return oppWins[0], nil
	}

	// 3. Filter out poison moves that hand the opponent an immediate win
	candidates := make([]uint8, 0, len(valid))
	for _, act := range valid {
		if !GivesOpponentImmediateWin(g, act) {
			candidates = append(candidates, act)
		}
	}
	if len(candidates) == 0 {
		candidates = valid
	}

	// 4. Score candidates by 76-line heuristic evaluation + cell line connectivity
	bestAction := candidates[0]
	bestScore := -1_000_000_000
	for _, act := range candidates {
		z, _ := g.LandingZ(act)
		bit := act + 16*z
		clone := g.Clone()
		_, _ = clone.ApplyMove(act)
		score := EvaluateBoard(clone, me) + CellLineCounts[bit]*2
		if score > bestScore {
			bestScore = score
			bestAction = act
		}
	}

	return bestAction, nil
}

// NewAgentByName instantiates one of the built-in agents ("random", "heuristic", "alphabeta", "mcts").
func NewAgentByName(name string, seed int64) (Agent, error) {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "random":
		return NewRandomAgent(seed), nil
	case "heuristic":
		return NewHeuristicAgent(), nil
	case "alphabeta", "minimax":
		return NewAlphaBetaAgent(3), nil
	case "mcts":
		return NewMCTSAgent(250, seed), nil
	default:
		return nil, fmt.Errorf("%w: %q (valid: random, heuristic, alphabeta, mcts)", ErrUnknownAgent, name)
	}
}
