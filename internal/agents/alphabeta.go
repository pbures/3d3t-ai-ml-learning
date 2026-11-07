package agents

import (
	"math/bits"
	"sort"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

const (
	ScoreWin  = 100000
	ScoreLoss = -100000
)

// EvaluateBoard computes a heuristic score for `g` from the perspective of `perspective` (PlayerWhite or PlayerBlack).
func EvaluateBoard(g *engine.Game, perspective engine.Player) int {
	if g.Status != engine.StatusOngoing {
		if g.Winner == perspective {
			return ScoreWin - int(g.MoveCount)
		}
		if g.Winner == perspective.Opponent() {
			return ScoreLoss + int(g.MoveCount)
		}
		return 0 // Draw
	}

	myBoard := g.White
	oppBoard := g.Black
	if perspective == engine.PlayerBlack {
		myBoard = g.Black
		oppBoard = g.White
	}

	surface := LandingMask(g)
	score := 0

	for _, mask := range engine.WinMasks {
		myBits := myBoard & mask
		oppBits := oppBoard & mask

		// If both players occupy at least one cell on this line, neither can win on it
		if myBits != 0 && oppBits != 0 {
			continue
		}

		if myBits != 0 {
			cnt := bits.OnesCount64(myBits)
			switch cnt {
			case 1:
				score += 1
			case 2:
				score += 6
			case 3:
				missingBit := mask &^ myBits
				if (missingBit & surface) != 0 {
					score += 65 // Playable immediate threat on gravity surface
				} else {
					score += 25 // Future floating threat
				}
			}
		} else if oppBits != 0 {
			cnt := bits.OnesCount64(oppBits)
			switch cnt {
			case 1:
				score -= 1
			case 2:
				score -= 6
			case 3:
				missingBit := mask &^ oppBits
				if (missingBit & surface) != 0 {
					score -= 80 // Urgent playable opponent threat
				} else {
					score -= 28
				}
			}
		}
	}

	return score
}

// AlphaBetaAgent uses Negamax tree search with Alpha-Beta pruning and move ordering.
type AlphaBetaAgent struct {
	MaxDepth int
}

// NewAlphaBetaAgent creates an AlphaBetaAgent with the given search depth (defaults to 3 if <= 0).
func NewAlphaBetaAgent(maxDepth int) *AlphaBetaAgent {
	if maxDepth <= 0 {
		maxDepth = 3
	}
	return &AlphaBetaAgent{MaxDepth: maxDepth}
}

func (a *AlphaBetaAgent) Name() string {
	return "alphabeta"
}

// orderedValidActions returns valid actions sorted by landing cell line connectivity (7-line cells first)
// to maximize Alpha-Beta pruning efficiency.
func orderedValidActions(g *engine.Game) []uint8 {
	actions := ValidActions(g)
	sort.Slice(actions, func(i, j int) bool {
		zi, _ := g.LandingZ(actions[i])
		zj, _ := g.LandingZ(actions[j])
		bi := actions[i] + 16*zi
		bj := actions[j] + 16*zj
		return CellLineCounts[bi] > CellLineCounts[bj]
	})
	return actions
}

func (a *AlphaBetaAgent) SelectAction(g *engine.Game) (uint8, error) {
	valid := orderedValidActions(g)
	if len(valid) == 0 {
		return 0, ErrNoValidMoves
	}

	me := g.CurrentPlayer
	opp := me.Opponent()

	// 1. Immediate 1-ply win short-circuit
	if myWins := FindImmediateWinningActions(g, me); len(myWins) > 0 {
		return myWins[0], nil
	}

	// 2. Immediate 1-ply forced block short-circuit
	if oppWins := FindImmediateWinningActions(g, opp); len(oppWins) > 0 {
		for _, blk := range oppWins {
			if !GivesOpponentImmediateWin(g, blk) {
				return blk, nil
			}
		}
		return oppWins[0], nil
	}

	bestAction := valid[0]
	alpha := -1_000_000
	beta := 1_000_000

	for _, act := range valid {
		child := g.Clone()
		_, _ = child.ApplyMove(act)
		val := -a.negamax(child, a.MaxDepth-1, -beta, -alpha)
		if val > alpha {
			alpha = val
			bestAction = act
		}
	}

	return bestAction, nil
}

func (a *AlphaBetaAgent) negamax(g *engine.Game, depth int, alpha int, beta int) int {
	if depth <= 0 || g.Status != engine.StatusOngoing {
		return EvaluateBoard(g, g.CurrentPlayer)
	}

	// Check if active player can win immediately on this ply
	if wins := FindImmediateWinningActions(g, g.CurrentPlayer); len(wins) > 0 {
		return ScoreWin - int(g.MoveCount+1)
	}

	valid := orderedValidActions(g)
	bestVal := -1_000_000

	for _, act := range valid {
		child := g.Clone()
		_, _ = child.ApplyMove(act)
		val := -a.negamax(child, depth-1, -beta, -alpha)
		if val > bestVal {
			bestVal = val
		}
		if bestVal > alpha {
			alpha = bestVal
		}
		if alpha >= beta {
			break // Beta cutoff
		}
	}

	return bestVal
}
