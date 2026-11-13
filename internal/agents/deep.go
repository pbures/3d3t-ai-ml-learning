package agents

import (
	"math"
	"math/bits"
	"time"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

const (
	// DefaultDeepTimeBudget is the per-move thinking time of the "deep" baseline (kept below the
	// arena's 3s per-move cap with margin for HTTP / scheduling overhead).
	DefaultDeepTimeBudget = 2800 * time.Millisecond
	// DefaultDeepMinDepth is the iterative-deepening depth that fast (non EnforceFullTime) searches
	// try to reach before returning.
	DefaultDeepMinDepth = 15
	DefaultDeepMaxDepth = 64
	// DefaultDeepSolverFromPly is the first ply (move count) at which the exact WDL endgame solver
	// is run in addition to PVS.
	DefaultDeepSolverFromPly = 20
	// deepPVSShareWithSolver is the fraction of the time budget given to PVS when the solver runs;
	// from deepLateSolverPly on, the solver usually decides the game, so it receives more time.
	deepPVSShareWithSolver = 0.45
	deepPVSShareLate       = 0.25
	deepLateSolverPly      = 26
	// deepSolverMargin is the safety margin kept free at the end of the budget.
	deepSolverMargin = 15 * time.Millisecond

	ttSize = 1 << 18 // 262,144 entries (~4 MB)
	ttMask = ttSize - 1

	ttFlagExact = 1
	ttFlagLower = 2
	ttFlagUpper = 3
)

var (
	cellWinLines     [64][7]uint64
	cellWinLineCount [64]uint8
	zobristPiece     [2][64]uint64
	zobristTurn      uint64
	baseColumnOrder  = [16]uint8{5, 6, 9, 10, 0, 3, 12, 15, 1, 2, 4, 7, 8, 11, 13, 14}
)

func init() {
	for _, mask := range engine.WinMasks {
		for bit := uint8(0); bit < 64; bit++ {
			if (mask & (uint64(1) << bit)) != 0 {
				idx := cellWinLineCount[bit]
				cellWinLines[bit][idx] = mask
				cellWinLineCount[bit] = idx + 1
			}
		}
	}

	// Deterministic 64-bit SplitMix64 generator for Zobrist hashing
	seed := uint64(0x9E3779B97F4A7C15)
	nextRand := func() uint64 {
		seed += 0x9E3779B97F4A7C15
		z := seed
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
		z = (z ^ (z >> 27)) * 0x94D049BB133111EB
		return z ^ (z >> 31)
	}
	for p := 0; p < 2; p++ {
		for b := 0; b < 64; b++ {
			zobristPiece[p][b] = nextRand()
		}
	}
	zobristTurn = nextRand()
}

type ttEntry struct {
	key      uint64
	score    int32
	depth    int8
	flag     uint8
	bestMove uint8
}

// surfaceBitmaskFast computes the 16-column 3D gravity landing mask in O(1) bit operations.
func surfaceBitmaskFast(occ uint64) uint64 {
	return (^occ) & ((occ << 16) | 0xFFFF)
}

// threatCellsPairBitboard computes both players' 3-in-a-row empty threat bitmasks in a single pass.
func threatCellsPairBitboard(myBB, oppBB uint64) (myThreats, oppThreats uint64) {
	for _, mask := range engine.WinMasks {
		myBits := myBB & mask
		oppBits := oppBB & mask
		if myBits != 0 && oppBits != 0 {
			continue
		}
		if oppBits == 0 {
			if bits.OnesCount64(myBits) == 3 {
				myThreats |= (mask &^ myBits)
			}
		} else if myBits == 0 {
			if bits.OnesCount64(oppBits) == 3 {
				oppThreats |= (mask &^ oppBits)
			}
		}
	}
	return myThreats, oppThreats
}

// evaluateBitboardFast computes the full 76-line + stacked-fork + staircase + suspended-threat score
// directly on raw 64-bit bitboards without allocating an engine.Game struct.
func evaluateBitboardFast(myBB, oppBB, myThreats, oppThreats uint64, moveCount uint8) int {
	occ := myBB | oppBB
	surface := surfaceBitmaskFast(occ)

	myImmCount := bits.OnesCount64(myThreats & surface)
	oppImmCount := bits.OnesCount64(oppThreats & surface)
	if myImmCount > 0 {
		return ScoreWin - int(moveCount+1)
	}
	if oppImmCount >= 2 {
		return ScoreLoss + int(moveCount+2)
	}

	score := 0

	// Unstoppable multi-column or dormant stacked vertical forks (z and z+1 in the same column)
	myStacked := myThreats & (myThreats >> 16) &^ oppThreats
	oppStacked := oppThreats & (oppThreats >> 16) &^ myThreats
	if myStacked != 0 {
		score += 2500
	}
	if oppStacked != 0 {
		score -= 2600
	}

	// 7-line strategic cell control (outer corners + inner 2x2x2 core) and center priority (B2,B3,C2,C3 on z=0,1)
	score += 4 * (bits.OnesCount64(myBB&SevenLineMask) - bits.OnesCount64(oppBB&SevenLineMask))
	score += 6 * (bits.OnesCount64(myBB&CenterZ0Z1Mask) - bits.OnesCount64(oppBB&CenterZ0Z1Mask))

	// 3-column staircase fork base detection (columns where player owns z=0 & z=1 and z=2 is empty)
	myVert2 := uint16((myBB & 0xFFFF) & (myBB >> 16) &^ (occ >> 32))
	oppVert2 := uint16((oppBB & 0xFFFF) & (oppBB >> 16) &^ (occ >> 32))
	myZ2 := uint16(myBB >> 32)
	oppZ2 := uint16(oppBB >> 32)
	for _, lm := range planarLineMasks16 {
		if (oppZ2 & lm) == 0 {
			v2 := bits.OnesCount16(myVert2 & lm)
			z2 := bits.OnesCount16(myZ2 & lm)
			if v2 >= 1 && v2+z2 >= 3 {
				score += 220
			} else if v2+z2 == 2 && v2 >= 1 {
				score += 45
			}
		}
		if (myZ2 & lm) == 0 {
			v2 := bits.OnesCount16(oppVert2 & lm)
			z2 := bits.OnesCount16(oppZ2 & lm)
			if v2 >= 1 && v2+z2 >= 3 {
				score -= 250
			} else if v2+z2 == 2 && v2 >= 1 {
				score -= 55
			}
		}
	}

	myVirtBoard := myBB | (uint64(myVert2) << 32)
	oppVirtBoard := oppBB | (uint64(oppVert2) << 32)

	// Column-wise suspended threat ownership (prevents z=2/z=3 Zugzwang traps)
	mySuspended := myThreats &^ surface
	oppSuspended := oppThreats &^ surface
	if (mySuspended | oppSuspended) != 0 {
		for col := uint8(0); col < engine.MaxActions; col++ {
			lowestOwner := uint8(0)
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

	var myTwoOnce, myForkIntersections uint64
	var oppTwoOnce, oppForkIntersections uint64
	myStackTargets := ((myThreats << 16) | (myThreats >> 16)) &^ occ &^ oppThreats
	oppStackTargets := ((oppThreats << 16) | (oppThreats >> 16)) &^ occ &^ myThreats

	for _, mask := range engine.WinMasks {
		myBits := myBB & mask
		oppBits := oppBB & mask
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
				emptyBits := mask &^ myBits
				myForkIntersections |= myTwoOnce & emptyBits
				myTwoOnce |= emptyBits
				if (emptyBits & myStackTargets) != 0 {
					if ((emptyBits &^ myStackTargets) & surface) != 0 {
						score += 180
					} else {
						score += 75
					}
				}
			case 3:
				missingBit := mask &^ myBits
				if (missingBit & surface) != 0 {
					score += 65
				} else {
					score += 32
				}
			}
		} else if oppBits != 0 {
			cnt := bits.OnesCount64(oppBits)
			switch cnt {
			case 1:
				score -= 1
			case 2:
				score -= 9
				emptyBits := mask &^ oppBits
				oppForkIntersections |= oppTwoOnce & emptyBits
				oppTwoOnce |= emptyBits
				if (emptyBits & oppStackTargets) != 0 {
					if ((emptyBits &^ oppStackTargets) & surface) != 0 {
						score -= 220
					} else {
						score -= 95
					}
				}
			case 3:
				missingBit := mask &^ oppBits
				if (missingBit & surface) != 0 {
					score -= 80
				} else {
					score -= 36
				}
			}
		}

		if myVirtBoard != myBB && (oppBB&mask) == 0 {
			vBits := myVirtBoard & mask
			if vBits != myBits && bits.OnesCount64(vBits) >= 2 {
				if ((mask &^ vBits) & myStackTargets) != 0 {
					score += 190
				}
			}
		}
		if oppVirtBoard != oppBB && (myBB&mask) == 0 {
			vBits := oppVirtBoard & mask
			if vBits != oppBits && bits.OnesCount64(vBits) >= 2 {
				if ((mask &^ vBits) & oppStackTargets) != 0 {
					score -= 240
				}
			}
		}
	}

	score += 110 * bits.OnesCount64(myForkIntersections&surface)
	score += 35 * bits.OnesCount64(myForkIntersections&^surface)
	score -= 135 * bits.OnesCount64(oppForkIntersections&surface)
	score -= 45 * bits.OnesCount64(oppForkIntersections&^surface)

	return score
}

func computeZobristHash(myBB, oppBB uint64) uint64 {
	var h uint64
	for b := myBB; b != 0; b &= b - 1 {
		bit := bits.TrailingZeros64(b)
		h ^= zobristPiece[0][bit]
	}
	for b := oppBB; b != 0; b &= b - 1 {
		bit := bits.TrailingZeros64(b)
		h ^= zobristPiece[1][bit]
	}
	return h
}

// DeepSearchAgent is the classical search baseline ("deep"). Each move it runs:
//  1. sound tactical shortcuts (immediate win, forced block, poison-column filtering),
//  2. iterative-deepening PVS (pvsSearcher) with Zobrist TT, logarithmic LMR with full-depth
//     re-search, killer/history ordering, check extensions and a zugzwang-parity aware evaluation,
//  3. from DefaultDeepSolverFromPly onwards, an exact Win/Draw/Loss solver (wdlSolver) over the root
//     moves that plays proven wins and vetoes proven losses.
//
// By default each move uses ~2.8 seconds. The agent is deterministic up to timing effects.
type DeepSearchAgent struct {
	TimeBudget      time.Duration
	MinDepth        int
	MaxDepth        int
	EnforceFullTime bool
	// SolverFromPly enables the exact endgame solver when the game's move count is >= this value.
	// A negative value disables the solver.
	SolverFromPly int

	LastDepthReached int
	LastNodesVisited uint64
	LastSolverNodes  uint64
	// LastProvenValue is the proven game value (+1 win, 0 draw, -1 loss) of the chosen move from
	// the mover's perspective, or wdlUnknown (2) if it was not proven.
	LastProvenValue int
	LastElapsed     time.Duration
}

// NewDeepSearchAgent creates a DeepSearchAgent with the specified per-move time budget and minimum depth.
func NewDeepSearchAgent(timeBudget time.Duration, minDepth int) *DeepSearchAgent {
	if timeBudget <= 0 {
		timeBudget = DefaultDeepTimeBudget
	}
	if minDepth <= 0 {
		minDepth = DefaultDeepMinDepth
	}
	return &DeepSearchAgent{
		TimeBudget:      timeBudget,
		MinDepth:        minDepth,
		MaxDepth:        DefaultDeepMaxDepth,
		EnforceFullTime: timeBudget >= 500*time.Millisecond,
		SolverFromPly:   DefaultDeepSolverFromPly,
	}
}

func (a *DeepSearchAgent) Name() string {
	return "deep"
}

// SearchWithPolicy runs the search and returns:
// - bestAction (0..15)
// - rootPolicy (16-element distribution over root moves derived from search scores / proofs)
// - rootValue (normalized scalar value in [-1.0, +1.0])
func (a *DeepSearchAgent) SearchWithPolicy(g *engine.Game) (uint8, [16]float32, float32, error) {
	var policy [16]float32
	valid := ValidActions(g)
	if len(valid) == 0 {
		return 0, policy, 0, ErrNoValidMoves
	}

	startTime := time.Now()
	budget := a.TimeBudget
	if budget <= 0 {
		budget = DefaultDeepTimeBudget
	}
	a.LastSolverNodes = 0
	a.LastNodesVisited = 0
	a.LastProvenValue = wdlUnknown

	myBB, oppBB := g.White, g.Black
	if g.CurrentPlayer == engine.PlayerBlack {
		myBB, oppBB = g.Black, g.White
	}
	occ := myBB | oppBB
	surface := surfaceBitmaskFast(occ)
	myThreats, oppThreats := threatCellsPairBitboard(myBB, oppBB)

	// 1. Immediate win.
	if myImm := myThreats & surface; myImm != 0 {
		var winCols []uint8
		for s := myImm; s != 0; s &= s - 1 {
			winCols = append(winCols, uint8(bits.TrailingZeros64(s))&15)
		}
		for _, c := range winCols {
			policy[c] = 1.0 / float32(len(winCols))
		}
		a.LastDepthReached = 1
		a.LastProvenValue = 1
		a.finish(startTime, budget)
		return winCols[0], policy, 1.0, nil
	}

	// 2. Forced block(s): the only sensible candidates are the blocking columns.
	var candidates []uint8
	if oppImm := oppThreats & surface; oppImm != 0 {
		var blkCols, safeBlkCols []uint8
		for s := oppImm; s != 0; s &= s - 1 {
			bit := uint8(bits.TrailingZeros64(s))
			blkCols = append(blkCols, bit&15)
			if bit >= 48 || (oppThreats&(uint64(1)<<(bit+16))) == 0 {
				safeBlkCols = append(safeBlkCols, bit&15)
			}
		}
		candidates = safeBlkCols
		if len(candidates) == 0 {
			candidates = blkCols
		}
		if len(blkCols) >= 2 || len(safeBlkCols) == 0 {
			// Already lost against correct play: block one threat and hope for a mistake.
			for _, c := range candidates {
				policy[c] = 1.0 / float32(len(candidates))
			}
			a.LastDepthReached = 1
			a.LastProvenValue = -1
			a.finish(startTime, budget)
			return candidates[0], policy, -1.0, nil
		}
	} else {
		// 3. Filter out poison columns (moving at z when the opponent threatens z+1).
		poisonMask := surface & (oppThreats >> 16)
		for _, col := range baseColumnOrder {
			bitMask := surface & (uint64(0x0001_0001_0001_0001) << col)
			if bitMask != 0 && bitMask&poisonMask == 0 {
				candidates = append(candidates, col)
			}
		}
		if len(candidates) == 0 {
			candidates = valid
		}
	}

	bestCol, scores, proven, bestScore := a.decide(myBB, oppBB, myThreats, oppThreats, g.MoveCount, candidates, startTime, budget)
	policy = deepRootPolicy(candidates, scores, proven, bestCol)

	var normVal float32
	switch {
	case proven[bestCol] == 1 || bestScore >= ScoreWin/2:
		normVal = 1.0
	case proven[bestCol] == -1 || bestScore <= ScoreLoss/2:
		normVal = -1.0
	case proven[bestCol] == 0:
		normVal = 0
	default:
		normVal = float32(math.Tanh(float64(bestScore) / 350.0))
	}
	a.finish(startTime, budget)
	return bestCol, policy, normVal, nil
}

// finish honours EnforceFullTime (constant per-move pacing) and records the elapsed time.
func (a *DeepSearchAgent) finish(startTime time.Time, budget time.Duration) {
	if a.EnforceFullTime {
		if rem := time.Until(startTime.Add(budget)); rem > 0 {
			time.Sleep(rem)
		}
	}
	a.LastElapsed = time.Since(startTime)
}

// decide runs PVS and (late in the game) the exact solver over `candidates`, returning the chosen
// column, the PVS root scores, the proven WDL value per column (wdlUnknown if unproven) and the
// effective score of the chosen move.
func (a *DeepSearchAgent) decide(
	myBB, oppBB, myT, oppT uint64,
	moveCount uint8,
	candidates []uint8,
	startTime time.Time,
	budget time.Duration,
) (uint8, [16]int, [16]int, int) {
	var proven [16]int
	for i := range proven {
		proven[i] = wdlUnknown
	}
	hardDeadline := startTime.Add(budget - deepSolverMargin)
	useSolver := a.SolverFromPly >= 0 && int(moveCount) >= a.SolverFromPly
	pvsDeadline := hardDeadline
	if useSolver {
		share := deepPVSShareWithSolver
		if moveCount >= deepLateSolverPly {
			share = deepPVSShareLate
		}
		pvsDeadline = startTime.Add(time.Duration(float64(budget) * share))
	}

	white, black := myBB, oppBB
	if moveCount&1 == 1 {
		white, black = oppBB, myBB
	}
	rootHash := colorHash(white, black)
	remaining := int(engine.GridCells) - int(moveCount)

	pvs := newPVSSearcher(pvsDeadline)
	bestCol, scores, depth, bestScore := a.iterativeDeepening(pvs, myBB, oppBB, myT, oppT, moveCount, rootHash, candidates, startTime, pvsDeadline, remaining)
	a.LastNodesVisited = pvs.nodes
	pvs.release()
	a.LastDepthReached = depth

	if !useSolver {
		return bestCol, scores, proven, bestScore
	}

	// Solve root moves in PVS order with iterative widening: every unresolved move gets a doubling
	// node budget per round (the TT is shared across rounds), so cheap proofs - typically forced
	// wins - surface before the solver sinks its whole budget into one hard-to-prove move.
	order := orderByScore(candidates, scores, bestCol)
	solver := newWDLSolver(hardDeadline)
	side := moveCount & 1
	surf := surfaceBitmaskFast(myBB | oppBB)
	// Pass 0 asks "does this move win?" (child lost: value <= -1 in window (-1,0)); pass 1 asks
	// "does this move lose?" (child won: value >= 1 in window (0,1)). A move that neither wins nor
	// loses is a proven draw.
	var notWin, notLoss [16]bool
	for pass := 0; pass < 2 && !solver.timedOut; pass++ {
		unresolved := 0
		for _, col := range order {
			if proven[col] == wdlUnknown {
				unresolved++
			}
		}
		for limit := uint64(20_000); unresolved > 0 && !solver.timedOut && limit < 1<<40; limit *= 4 {
			for _, col := range order {
				if proven[col] != wdlUnknown || (pass == 0 && notWin[col]) || (pass == 1 && notLoss[col]) {
					continue
				}
				bit := uint8(bits.TrailingZeros64(surf & (uint64(0x0001_0001_0001_0001) << col)))
				nm, nmt, not := playBit(myBB, oppBB, myT, oppT, bit)
				h := rootHash ^ zobristPiece[side][bit]
				var v int
				var ok bool
				if pass == 0 {
					v, ok = solver.solveWindow(oppBB, nm, not, nmt, h, moveCount+1, -1, 0, limit)
				} else {
					v, ok = solver.solveWindow(oppBB, nm, not, nmt, h, moveCount+1, 0, 1, limit)
				}
				if solver.timedOut {
					break
				}
				if !ok {
					continue
				}
				unresolved--
				if pass == 0 {
					if v <= -1 {
						proven[col] = 1
						unresolved = 0
						pass = 2 // a proven win ends the solver phase
						break
					}
					notWin[col] = true
				} else {
					if v >= 1 {
						proven[col] = -1
					} else {
						notLoss[col] = true
					}
				}
				if notWin[col] && notLoss[col] {
					proven[col] = 0
				}
			}
		}
	}
	a.LastSolverNodes = solver.nodes
	solver.release()

	// Choose: proven win > best of (unproven by PVS score, proven draw scored as 0) > proven loss.
	chosen := bestCol
	chosenKey := math.MinInt64
	for _, col := range order {
		var key int
		switch proven[col] {
		case 1:
			key = math.MaxInt32
		case 0:
			key = 0
		case -1:
			// Proven loss: only chosen if everything loses; prefer the longest PVS resistance.
			key = math.MinInt32 + (scores[col] - ScoreLoss*2)
		default:
			key = scores[col]
			if key <= ScoreLoss/2 && proven[col] == wdlUnknown {
				key = math.MinInt32/2 + (scores[col] - ScoreLoss*2)
			}
		}
		if key > chosenKey {
			chosenKey = key
			chosen = col
		}
	}
	a.LastProvenValue = proven[chosen]
	if proven[chosen] != wdlUnknown {
		a.LastDepthReached = remaining
	}
	chosenScore := scores[chosen]
	switch proven[chosen] {
	case 1:
		chosenScore = ScoreWin / 2
	case 0:
		chosenScore = 0
	case -1:
		chosenScore = ScoreLoss / 2
	}
	return chosen, scores, proven, chosenScore
}

// iterativeDeepening runs root PVS for increasing depths until the deadline, a proven result, the
// end of the game horizon, or (when EnforceFullTime is false) MinDepth is reached.
func (a *DeepSearchAgent) iterativeDeepening(
	s *pvsSearcher,
	myBB, oppBB, myT, oppT uint64,
	moveCount uint8,
	rootHash uint64,
	candidates []uint8,
	startTime, deadline time.Time,
	remaining int,
) (uint8, [16]int, int, int) {
	const inf = 1_000_000
	var scores [16]int
	for i := range scores {
		scores[i] = ScoreLoss * 2
	}
	maxDepth := a.MaxDepth
	if maxDepth <= 0 {
		maxDepth = DefaultDeepMaxDepth
	}
	if maxDepth > remaining {
		maxDepth = remaining
	}
	if maxDepth < 1 {
		maxDepth = 1
	}
	minDepth := a.MinDepth
	if minDepth <= 0 {
		minDepth = DefaultDeepMinDepth
	}

	occ := myBB | oppBB
	surface := surfaceBitmaskFast(occ)
	side := moveCount & 1
	order := append([]uint8(nil), candidates...)
	bestCol := order[0]
	bestScore := 0
	completed := 0

	for depth := 1; depth <= maxDepth; depth++ {
		if depth > 1 {
			now := time.Now()
			if now.After(deadline) {
				break
			}
			// Do not start an iteration that almost certainly cannot finish in time.
			if a.EnforceFullTime && now.Sub(startTime) > deadline.Sub(startTime)*6/10 {
				break
			}
		}
		// The first iteration always completes so that a legal, searched move is available.
		s.deadline = deadline
		if depth == 1 {
			s.deadline = startTime.Add(time.Hour)
		}
		for i := range s.history[0] {
			s.history[0][i] /= 2
			s.history[1][i] /= 2
		}

		iterScores := scores
		iterBest := -inf
		iterCol := order[0]
		alpha := -inf
		for i, col := range order {
			bit := uint8(bits.TrailingZeros64(surface & (uint64(0x0001_0001_0001_0001) << col)))
			nm, nmt, not := playBit(myBB, oppBB, myT, oppT, bit)
			nh := rootHash ^ zobristPiece[side][bit]
			var v int
			if i == 0 {
				v = -s.search(oppBB, nm, not, nmt, moveCount+1, 1, depth-1, -inf, inf, nh)
			} else {
				v = -s.search(oppBB, nm, not, nmt, moveCount+1, 1, depth-1, -alpha-1, -alpha, nh)
				if v > alpha && !s.aborted {
					v = -s.search(oppBB, nm, not, nmt, moveCount+1, 1, depth-1, -inf, -alpha, nh)
				}
			}
			if s.aborted {
				break
			}
			iterScores[col] = v
			if v > iterBest {
				iterBest = v
				iterCol = col
			}
			if v > alpha {
				alpha = v
			}
		}

		if s.aborted {
			// Partial iteration: completed root moves were searched deeper than before. Accept
			// them only if the previous best move (searched first) completed.
			if iterBest > -inf {
				bestCol, bestScore, scores = iterCol, iterBest, iterScores
			}
			break
		}
		bestCol, bestScore, scores, completed = iterCol, iterBest, iterScores, depth
		order = orderByScore(candidates, scores, bestCol)

		if bestScore >= ScoreWin/2 || bestScore <= ScoreLoss/2 {
			break // proven by search
		}
		if !a.EnforceFullTime && depth >= minDepth {
			break
		}
	}
	return bestCol, scores, completed, bestScore
}

// orderByScore returns candidates ordered by descending score with `first` in front (stable).
func orderByScore(candidates []uint8, scores [16]int, first uint8) []uint8 {
	out := make([]uint8, 0, len(candidates))
	out = append(out, first)
	for _, c := range candidates {
		if c == first {
			continue
		}
		pos := len(out)
		out = append(out, c)
		for pos > 1 && scores[out[pos-1]] < scores[c] {
			out[pos] = out[pos-1]
			pos--
		}
		out[pos] = c
	}
	return out
}

// deepRootPolicy converts root scores and proofs into a probability distribution over candidates.
func deepRootPolicy(candidates []uint8, scores [16]int, proven [16]int, bestCol uint8) [16]float32 {
	var policy [16]float32
	var wins []uint8
	for _, c := range candidates {
		if proven[c] == 1 || scores[c] >= ScoreWin/2 {
			wins = append(wins, c)
		}
	}
	if len(wins) > 0 {
		for _, c := range wins {
			policy[c] = 1.0 / float32(len(wins))
		}
		return policy
	}
	eff := func(c uint8) int {
		switch proven[c] {
		case 0:
			return 0
		case -1:
			return ScoreLoss
		}
		return scores[c]
	}
	maxS := math.MinInt32
	for _, c := range candidates {
		if v := eff(c); v > maxS {
			maxS = v
		}
	}
	var sum float64
	for _, c := range candidates {
		diff := float64(eff(c)-maxS) / 35.0
		if diff < -30.0 {
			continue
		}
		w := math.Exp(diff)
		policy[c] = float32(w)
		sum += w
	}
	if sum <= 0 {
		policy[bestCol] = 1.0
		return policy
	}
	for _, c := range candidates {
		policy[c] = float32(float64(policy[c]) / sum)
	}
	return policy
}

func (a *DeepSearchAgent) SelectAction(g *engine.Game) (uint8, error) {
	act, _, _, err := a.SearchWithPolicy(g)
	return act, err
}
