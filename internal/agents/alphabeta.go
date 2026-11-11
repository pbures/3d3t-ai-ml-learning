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

// SevenLineMask has bits set at the 16 cells of the 4x4x4 cube that belong to 7 winning lines
// (the 8 outer corners and the 8 inner 2x2x2 core cells).
var SevenLineMask uint64

func init() {
	for bit := uint8(0); bit < engine.GridCells; bit++ {
		if CellLineCounts[bit] == 7 {
			SevenLineMask |= uint64(1) << bit
		}
	}
}

// ThreatCellsMask returns a 64-bit bitmask of all currently empty cells (0..63) where `player`
// already occupies the other 3 cells of a winning line (both immediate at z=h and suspended at z>h).
func ThreatCellsMask(g *engine.Game, player engine.Player) uint64 {
	if g == nil {
		return 0
	}
	myBoard := g.White
	oppBoard := g.Black
	if player == engine.PlayerBlack {
		myBoard = g.Black
		oppBoard = g.White
	}
	var threats uint64
	for _, mask := range engine.WinMasks {
		if (oppBoard & mask) != 0 {
			continue
		}
		myBits := myBoard & mask
		if bits.OnesCount64(myBits) == 3 {
			threats |= (mask &^ myBits)
		}
	}
	return threats
}

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

	opp := perspective.Opponent()
	myBoard := g.White
	oppBoard := g.Black
	if perspective == engine.PlayerBlack {
		myBoard = g.Black
		oppBoard = g.White
	}

	surface := LandingMask(g)
	myThreats := ThreatCellsMask(g, perspective)
	oppThreats := ThreatCellsMask(g, opp)

	// Turn-aware immediate win detection
	myImmCount := bits.OnesCount64(myThreats & surface)
	oppImmCount := bits.OnesCount64(oppThreats & surface)
	if g.CurrentPlayer == perspective && myImmCount > 0 {
		return ScoreWin - int(g.MoveCount+1)
	}
	if g.CurrentPlayer == opp && oppImmCount > 0 {
		return ScoreLoss + int(g.MoveCount+1)
	}

	score := 0

	// Unstoppable multi-column or dormant stacked vertical forks (z and z+1 in the same column, h <= z)
	myStacked := myThreats & (myThreats >> 16) &^ oppThreats
	oppStacked := oppThreats & (oppThreats >> 16) &^ myThreats
	if myImmCount >= 2 || myStacked != 0 {
		score += 2500
	}
	if oppImmCount >= 2 || oppStacked != 0 {
		score -= 2600
	}

	// 7-line strategic cell control (outer corners + inner 2x2x2 core)
	score += 4 * (bits.OnesCount64(myBoard&SevenLineMask) - bits.OnesCount64(oppBoard&SevenLineMask))

	// Column-wise suspended threat ownership (prevents z=2/z=3 Zugzwang traps)
	mySuspended := myThreats &^ surface
	oppSuspended := oppThreats &^ surface
	if (mySuspended | oppSuspended) != 0 {
		for col := uint8(0); col < engine.MaxActions; col++ {
			lowestOwner := uint8(0) // 0=none, 1=me, 2=opp
			for z := uint8(1); z < engine.GridDim; z++ {
				bit := uint64(1) << (col + 16*z)
				mHas := (mySuspended & bit) != 0
				oHas := (oppSuspended & bit) != 0
				if !mHas && !oHas {
					continue
				}
				if lowestOwner == 0 {
					if mHas && !oHas {
						lowestOwner = 1
						score += 55
					} else if oHas && !mHas {
						lowestOwner = 2
						score -= 60
					} else {
						lowestOwner = 3
					}
				} else if lowestOwner == 1 && mHas {
					score += 35
				} else if lowestOwner == 2 && oHas {
					score -= 40
				}
			}
		}
	}

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
				score += 8
			case 3:
				missingBit := mask &^ myBits
				if (missingBit & surface) != 0 {
					score += 65 // Playable immediate threat on gravity surface
				} else {
					score += 32 // Future floating threat
				}
			}
		} else if oppBits != 0 {
			cnt := bits.OnesCount64(oppBits)
			switch cnt {
			case 1:
				score -= 1
			case 2:
				score -= 9
			case 3:
				missingBit := mask &^ oppBits
				if (missingBit & surface) != 0 {
					score -= 80 // Urgent playable opponent threat
				} else {
					score -= 36
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
