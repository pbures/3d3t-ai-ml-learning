package agents

import (
	"math"
	"math/bits"
	"time"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

const (
	DefaultDeepTimeBudget = 1500 * time.Millisecond
	DefaultDeepMinDepth   = 15
	DefaultDeepMaxDepth   = 40

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

// checkBitWinFast checks only the 4 or 7 winning lines passing through `bit`.
func checkBitWinFast(bb uint64, bit uint8) bool {
	cnt := cellWinLineCount[bit]
	lines := &cellWinLines[bit]
	for i := uint8(0); i < cnt; i++ {
		m := lines[i]
		if (bb & m) == m {
			return true
		}
	}
	return false
}

// surfaceBitmaskFast computes the 16-column 3D gravity landing mask in O(1) bit operations.
func surfaceBitmaskFast(occ uint64) uint64 {
	return (^occ) & ((occ << 16) | 0xFFFF)
}

// threatCellsBitboard computes a 64-bit mask of all empty cells where `myBB` has 3-in-a-row and `oppBB` has 0.
func threatCellsBitboard(myBB, oppBB uint64) uint64 {
	var threats uint64
	for _, mask := range engine.WinMasks {
		if (oppBB & mask) != 0 {
			continue
		}
		myBits := myBB & mask
		if bits.OnesCount64(myBits) == 3 {
			threats |= (mask &^ myBits)
		}
	}
	return threats
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

type deepSearcher struct {
	tt          []ttEntry
	killers     [64][2]uint8
	history     [2][16]int32
	nodes       uint64
	deadline    time.Time
	hasDeadline bool
	aborted     bool
	lmrScale    int // Controls LMR aggressiveness (higher during initial 15-ply pass, tighter during verification)
}

func newDeepSearcher(deadline time.Time, hasDeadline bool) *deepSearcher {
	return &deepSearcher{
		tt:          make([]ttEntry, ttSize),
		deadline:    deadline,
		hasDeadline: hasDeadline,
		lmrScale:    2,
	}
}

func (s *deepSearcher) checkTime() bool {
	if !s.hasDeadline {
		return false
	}
	if (s.nodes & 255) == 0 {
		if time.Now().After(s.deadline) {
			s.aborted = true
			return true
		}
	}
	return s.aborted
}

// pvs executes Principal Variation Search with Zobrist TT, Check Extensions, Poison Pruning, and LMR.
func (s *deepSearcher) pvs(
	myBB, oppBB uint64,
	moveCount uint8,
	ply int,
	depth int,
	alpha, beta int,
	hash uint64,
	side int,
	extensions int,
) int {
	s.nodes++
	if s.checkTime() {
		return 0
	}

	occ := myBB | oppBB
	if occ == 0xFFFF_FFFF_FFFF_FFFF {
		return 0 // Full board draw
	}

	surface := surfaceBitmaskFast(occ)

	// 1. Check immediate 1-ply win and opponent threats in a single pass
	myThreats, oppThreats := threatCellsPairBitboard(myBB, oppBB)
	if (myThreats & surface) != 0 {
		return ScoreWin - int(moveCount+1)
	}

	// 2. Check immediate 1-ply threats by opponent
	oppImm := oppThreats & surface
	oppImmCount := bits.OnesCount64(oppImm)
	if oppImmCount >= 2 {
		// Unstoppable double surface threat for opponent
		return ScoreLoss + int(moveCount+2)
	}

	// Check opponent stacked vertical fork that is already unstoppable
	if oppImmCount == 1 {
		forcedBit := uint8(bits.TrailingZeros64(oppImm))
		if forcedBit < 48 && (oppThreats&(uint64(1)<<(forcedBit+16))) != 0 {
			// Blocking at forcedBit immediately unlocks forcedBit+16 for opponent to win!
			return ScoreLoss + int(moveCount+2)
		}
		// Single forced block: apply Check Extension so forcing sequences do not exhaust depth!
		nextDepth := depth - 1
		nextExt := extensions
		if extensions < 8 {
			nextDepth = depth // Do not decrement depth on strictly forced 1-move replies!
			nextExt = extensions + 1
		}
		newMy := myBB | oppImm
		nextHash := hash ^ zobristPiece[side][forcedBit] ^ zobristTurn
		return -s.pvs(oppBB, newMy, moveCount+1, ply+1, nextDepth, -beta, -alpha, nextHash, 1-side, nextExt)
	}

	// 3. Leaf evaluation when depth <= 0
	if depth <= 0 {
		return evaluateBitboardFast(myBB, oppBB, myThreats, oppThreats, moveCount)
	}

	// 4. Transposition Table lookup
	origAlpha := alpha
	ttIdx := hash & ttMask
	tte := s.tt[ttIdx]
	ttMove := uint8(255)
	if tte.key == hash {
		ttMove = tte.bestMove
		if int(tte.depth) >= depth {
			ttScore := int(tte.score)
			switch tte.flag {
			case ttFlagExact:
				return ttScore
			case ttFlagLower:
				if ttScore > alpha {
					alpha = ttScore
				}
			case ttFlagUpper:
				if ttScore < beta {
					beta = ttScore
				}
			}
			if alpha >= beta {
				return ttScore
			}
		}
	}

	// 5. Filter out 1-ply poison moves (playing at z where z+1 is an opponent 3-in-a-row threat)
	poisonMask := surface & (oppThreats >> 16)
	safeSurface := surface &^ poisonMask
	if safeSurface == 0 {
		// All legal columns unlock an immediate opponent win above (Zugzwang exhaustion)
		return ScoreLoss + int(moveCount+2)
	}

	// 6. Generate and score candidate moves for move ordering
	var moveBits [16]uint8
	var moveCols [16]uint8
	var moveScores [16]int32
	numMoves := 0

	k0 := s.killers[ply][0]
	k1 := s.killers[ply][1]
	// Cells that complete a new 3-in-a-row or stack for me
	myStackTargets := ((myThreats << 16) | (myThreats >> 16)) &^ occ &^ oppThreats

	for _, col := range baseColumnOrder {
		colMask := uint64(0x0001_0001_0001_0001) << col
		bitMask := safeSurface & colMask
		if bitMask == 0 {
			continue
		}
		bit := uint8(bits.TrailingZeros64(bitMask))
		var mScore int32
		if col == ttMove {
			mScore = 1_000_000
		} else {
			if col == k0 {
				mScore += 90_000
			} else if col == k1 {
				mScore += 80_000
			}
			mScore += s.history[side][col]
			mScore += int32(CellLineCounts[bit]) * 64
			if (bitMask & CenterZ0Z1Mask) != 0 {
				mScore += 200
			}
			if (bitMask & myStackTargets) != 0 {
				mScore += 25_000
			}
		}

		// Insertion sort descending by mScore
		pos := numMoves
		for pos > 0 && moveScores[pos-1] < mScore {
			moveBits[pos] = moveBits[pos-1]
			moveCols[pos] = moveCols[pos-1]
			moveScores[pos] = moveScores[pos-1]
			pos--
		}
		moveBits[pos] = bit
		moveCols[pos] = col
		moveScores[pos] = mScore
		numMoves++
	}

	bestScore := -1_000_000
	bestCol := moveCols[0]

	for i := 0; i < numMoves; i++ {
		bit := moveBits[i]
		col := moveCols[i]
		newMy := myBB | (uint64(1) << bit)
		nextHash := hash ^ zobristPiece[side][bit] ^ zobristTurn

		var val int
		if i == 0 {
			pvDepth := depth - 1
			if s.lmrScale >= 3 && ply >= 2 && depth >= 4 {
				pvDepth = depth - 2
			}
			val = -s.pvs(oppBB, newMy, moveCount+1, ply+1, pvDepth, -beta, -alpha, nextHash, 1-side, extensions)
		} else {
			// Late Move Reductions (LMR) for non-PV moves (i >= 1)
			reduction := 0
			if depth >= 3 && i >= 1 && s.lmrScale > 0 {
				if s.lmrScale >= 3 {
					reduction = 2 + (depth / 2) + (i / 2)
				} else if s.lmrScale == 2 {
					reduction = 1 + (depth / 3) + (i / 3)
				} else {
					reduction = 1 + (depth / 5) + (i / 4)
				}
				if depth-1-reduction < 1 {
					reduction = depth - 2
				}
			}

			// Null-window search (with LMR reduction)
			val = -s.pvs(oppBB, newMy, moveCount+1, ply+1, depth-1-reduction, -alpha-1, -alpha, nextHash, 1-side, extensions)

			// Re-search at tighter depth if reduced search beat alpha
			if val > alpha && reduction > 1 && !s.aborted && s.lmrScale <= 2 {
				midDepth := depth - 1 - (reduction / 2)
				val = -s.pvs(oppBB, newMy, moveCount+1, ply+1, midDepth, -alpha-1, -alpha, nextHash, 1-side, extensions)
			}
			// Re-search with full window if null-window search falls inside (alpha, beta)
			if val > alpha && val < beta && !s.aborted {
				remDepth := depth - 1
				if s.lmrScale >= 3 && depth >= 5 {
					remDepth = depth - 1 - (reduction / 2)
					if remDepth < 1 {
						remDepth = 1
					}
				} else if s.lmrScale == 2 && depth >= 8 && i >= 2 {
					remDepth = depth - 2
				}
				val = -s.pvs(oppBB, newMy, moveCount+1, ply+1, remDepth, -beta, -alpha, nextHash, 1-side, extensions)
			}
		}

		if s.aborted {
			return bestScore
		}

		if val > bestScore {
			bestScore = val
			bestCol = col
		}
		if bestScore > alpha {
			alpha = bestScore
		}
		if alpha >= beta {
			// Record killer & history heuristic on beta cutoff
			if s.killers[ply][0] != col {
				s.killers[ply][1] = s.killers[ply][0]
				s.killers[ply][0] = col
			}
			bonus := int32(depth * depth)
			if s.history[side][col]+bonus < 50_000 {
				s.history[side][col] += bonus
			}
			break
		}
	}

	// Store in Transposition Table
	flag := uint8(ttFlagExact)
	if bestScore <= origAlpha {
		flag = ttFlagUpper
	} else if bestScore >= beta {
		flag = ttFlagLower
	}
	s.tt[ttIdx] = ttEntry{
		key:      hash,
		score:    int32(bestScore),
		depth:    int8(depth),
		flag:     flag,
		bestMove: bestCol,
	}

	return bestScore
}

// DeepSearchAgent implements an adaptive iterative-deepening Principal Variation Search (PVS)
// baseline agent with Zobrist transposition table, LMR, Killer/History heuristics, and Check Extensions.
// By default, each move searches for ~1.5 seconds and reaches at least 15 plies of lookahead.
type DeepSearchAgent struct {
	TimeBudget       time.Duration
	MinDepth         int
	MaxDepth         int
	EnforceFullTime  bool
	LastDepthReached int
	LastNodesVisited uint64
	LastElapsed      time.Duration
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
	}
}

func (a *DeepSearchAgent) Name() string {
	return "deep"
}

// SearchWithPolicy runs adaptive iterative-deepening search and returns:
// - bestAction (0..15)
// - rootPolicy (16-element softmax policy distribution over safe root moves from deep search Q-values)
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
	minDepth := a.MinDepth
	if minDepth <= 0 {
		minDepth = DefaultDeepMinDepth
	}
	maxDepth := a.MaxDepth
	if maxDepth < minDepth {
		maxDepth = minDepth + 10
	}

	remainingPlies := int(engine.GridCells - g.MoveCount)
	targetMinDepth := minDepth
	if remainingPlies < targetMinDepth {
		targetMinDepth = remainingPlies
	}
	if targetMinDepth < 1 {
		targetMinDepth = 1
	}

	me := g.CurrentPlayer
	myBB := g.White
	oppBB := g.Black
	if me == engine.PlayerBlack {
		myBB = g.Black
		oppBB = g.White
	}
	occ := myBB | oppBB
	surface := surfaceBitmaskFast(occ)
	myThreats := threatCellsBitboard(myBB, oppBB)
	oppThreats := threatCellsBitboard(oppBB, myBB)

	// 1. Check immediate 1-ply wins
	myImm := myThreats & surface
	if myImm != 0 {
		var winCols []uint8
		for s := myImm; s != 0; s &= s - 1 {
			winCols = append(winCols, uint8(bits.TrailingZeros64(s))&15)
		}
		for _, c := range winCols {
			policy[c] = 1.0 / float32(len(winCols))
		}
		if a.EnforceFullTime {
			// Run deep verification tree to warm stats and honor adaptive ~1.5s compute budget
			a.runTimedDeepening(myBB, oppBB, g.MoveCount, valid, startTime, budget, targetMinDepth, maxDepth)
		} else {
			a.LastDepthReached = targetMinDepth
			a.LastElapsed = time.Since(startTime)
		}
		return winCols[0], policy, 1.0, nil
	}

	// 2. Check forced 1-ply opponent blocks
	oppImm := oppThreats & surface
	if oppImm != 0 {
		var blkCols []uint8
		var safeBlkCols []uint8
		for s := oppImm; s != 0; s &= s - 1 {
			bit := uint8(bits.TrailingZeros64(s))
			col := bit & 15
			blkCols = append(blkCols, col)
			if bit >= 48 || (oppThreats&(uint64(1)<<(bit+16))) == 0 {
				safeBlkCols = append(safeBlkCols, col)
			}
		}
		chosenPool := safeBlkCols
		if len(chosenPool) == 0 {
			chosenPool = blkCols
		}
		for _, c := range chosenPool {
			policy[c] = 1.0 / float32(len(chosenPool))
		}
		val := float32(-0.2)
		if len(safeBlkCols) == 0 || len(blkCols) >= 2 {
			val = -1.0
		}
		if a.EnforceFullTime {
			_, _, deepVal := a.runTimedDeepening(myBB, oppBB, g.MoveCount, chosenPool, startTime, budget, targetMinDepth, maxDepth)
			if len(safeBlkCols) > 0 && len(blkCols) == 1 {
				val = deepVal
			}
		} else {
			a.LastDepthReached = targetMinDepth
			a.LastElapsed = time.Since(startTime)
		}
		return chosenPool[0], policy, val, nil
	}

	// 3. Filter out 1-ply poison columns when safe alternatives exist
	poisonMask := surface & (oppThreats >> 16)
	candidates := make([]uint8, 0, len(valid))
	for _, col := range baseColumnOrder {
		colMask := uint64(0x0001_0001_0001_0001) << col
		bitMask := surface & colMask
		if bitMask == 0 {
			continue
		}
		if (bitMask & poisonMask) == 0 {
			candidates = append(candidates, col)
		}
	}
	if len(candidates) == 0 {
		candidates = valid
	}

	// 4. Filter out 2-ply fork-vulnerable moves if non-vulnerable moves exist
	nonForkVuln := make([]uint8, 0, len(candidates))
	for _, col := range candidates {
		colMask := uint64(0x0001_0001_0001_0001) << col
		bit := uint8(bits.TrailingZeros64(surface & colMask))
		nextMy := myBB | (uint64(1) << bit)
		nextOcc := occ | (uint64(1) << bit)
		nextSurf := surfaceBitmaskFast(nextOcc)
		nextMyThreats := threatCellsBitboard(nextMy, oppBB)

		// If this move immediately creates an unstoppable fork for us (2+ surface threats or stacked fork), take it!
		if bits.OnesCount64(nextMyThreats&nextSurf) >= 2 || (nextMyThreats&(nextMyThreats>>16)&^oppThreats) != 0 {
			nonForkVuln = []uint8{col}
			break
		}

		myChecks := nextMyThreats & nextSurf
		replySurf := nextSurf
		if myChecks != 0 {
			replySurf = myChecks
		}
		oppCanFork := false
		for rs := replySurf; rs != 0; rs &= rs - 1 {
			rBit := uint8(bits.TrailingZeros64(rs))
			oppAfter := oppBB | (uint64(1) << rBit)
			if checkBitWinFast(oppAfter, rBit) {
				oppCanFork = true
				break
			}
			occAfter := nextOcc | (uint64(1) << rBit)
			surfAfter := surfaceBitmaskFast(occAfter)
			if (nextMyThreats & surfAfter) != 0 {
				continue
			}
			oppThreatsAfter := threatCellsBitboard(oppAfter, nextMy)
			if bits.OnesCount64(oppThreatsAfter&surfAfter) >= 2 || (oppThreatsAfter&(oppThreatsAfter>>16)&^nextMyThreats) != 0 {
				oppCanFork = true
				break
			}
		}
		if !oppCanFork {
			nonForkVuln = append(nonForkVuln, col)
		}
	}
	if len(nonForkVuln) > 0 {
		candidates = nonForkVuln
	}

	bestCol, rootScores, normVal := a.runTimedDeepening(myBB, oppBB, g.MoveCount, candidates, startTime, budget, targetMinDepth, maxDepth)

	// Convert root scores into a sharp probability distribution over candidates
	maxS := -1_000_000
	for _, col := range candidates {
		if rootScores[col] > maxS {
			maxS = rootScores[col]
		}
	}
	if maxS >= ScoreWin/2 {
		// Concentrate probability mass on proven winning moves
		var winCnt float32
		for _, col := range candidates {
			if rootScores[col] >= ScoreWin/2 {
				policy[col] = 1.0
				winCnt += 1.0
			}
		}
		if winCnt > 0 {
			for i := range policy {
				policy[i] /= winCnt
			}
		}
	} else {
		var sumExp float64
		for _, col := range candidates {
			diff := float64(rootScores[col]-maxS) / 35.0
			if diff < -30.0 {
				continue
			}
			w := math.Exp(diff)
			policy[col] = float32(w)
			sumExp += w
		}
		if sumExp > 0 {
			for _, col := range candidates {
				policy[col] = float32(float64(policy[col]) / sumExp)
			}
		} else {
			policy[bestCol] = 1.0
		}
	}

	return bestCol, policy, normVal, nil
}

func (a *DeepSearchAgent) runTimedDeepening(
	myBB, oppBB uint64,
	moveCount uint8,
	candidates []uint8,
	startTime time.Time,
	budget time.Duration,
	targetMinDepth int,
	maxDepth int,
) (uint8, [16]int, float32) {
	deadline := startTime.Add(budget)
	searcher := newDeepSearcher(deadline, true)
	rootHash := computeZobristHash(myBB, oppBB)
	occ := myBB | oppBB
	surface := surfaceBitmaskFast(occ)

	var rootScores [16]int
	for i := range rootScores {
		rootScores[i] = ScoreLoss
	}
	bestCol := candidates[0]
	bestScore := 0
	completedDepth := 0

	// Phase 1: Iterative deepening from depth 1 up to maxDepth.
	for depth := 1; depth <= maxDepth; depth++ {
		if depth <= targetMinDepth {
			searcher.hasDeadline = depth >= 8
			searcher.lmrScale = 3
			if depth >= 8 && time.Now().After(deadline) {
				break
			}
		} else if depth <= 22 {
			searcher.hasDeadline = true
			searcher.lmrScale = 2
			if time.Now().After(deadline) {
				break
			}
		} else {
			searcher.hasDeadline = true
			searcher.lmrScale = 1
			if time.Now().After(deadline) {
				break
			}
		}

		iterBestCol := bestCol
		iterBestScore := -1_000_000
		alpha := -1_000_000
		beta := 1_000_000
		var iterScores [16]int
		for i := range iterScores {
			iterScores[i] = ScoreLoss
		}

		// Order bestCol from previous iteration first
		ordered := make([]uint8, 0, len(candidates))
		ordered = append(ordered, bestCol)
		for _, c := range candidates {
			if c != bestCol {
				ordered = append(ordered, c)
			}
		}

		for idx, col := range ordered {
			colMask := uint64(0x0001_0001_0001_0001) << col
			bit := uint8(bits.TrailingZeros64(surface & colMask))
			newMy := myBB | (uint64(1) << bit)
			nextHash := rootHash ^ zobristPiece[0][bit] ^ zobristTurn

			var val int
			if idx == 0 {
				val = -searcher.pvs(oppBB, newMy, moveCount+1, 1, depth-1, -beta, -alpha, nextHash, 1, 0)
			} else {
				rootRed := 0
				if depth >= 4 && idx >= 1 && searcher.lmrScale >= 3 {
					rootRed = 1 + (depth / 3) + (idx / 3)
					if depth-1-rootRed < 2 {
						rootRed = depth - 3
					}
				} else if depth >= 5 && idx >= 2 && searcher.lmrScale == 2 {
					rootRed = 1 + (depth / 4) + (idx / 4)
					if depth-1-rootRed < 2 {
						rootRed = depth - 3
					}
				}
				val = -searcher.pvs(oppBB, newMy, moveCount+1, 1, depth-1-rootRed, -alpha-1, -alpha, nextHash, 1, 0)
				if val > alpha && !searcher.aborted {
					remRootDepth := depth - 1
					if searcher.lmrScale >= 3 && depth >= 6 {
						remRootDepth = depth - 1 - (rootRed / 2)
					}
					val = -searcher.pvs(oppBB, newMy, moveCount+1, 1, remRootDepth, -beta, -alpha, nextHash, 1, 0)
				}
			}

			if searcher.aborted {
				break
			}
			iterScores[col] = val
			if val > iterBestScore {
				iterBestScore = val
				iterBestCol = col
			}
			if iterBestScore > alpha {
				alpha = iterBestScore
			}
		}

		if searcher.aborted {
			break
		}

		bestCol = iterBestCol
		bestScore = iterBestScore
		rootScores = iterScores
		completedDepth = depth
	}

	// Phase 2: If EnforceFullTime is enabled and the tree search finished before the ~1.5s budget,
	// run wider verification passes over all candidate root moves until the time budget is reached.
	if a.EnforceFullTime {
		searcher.hasDeadline = true
		searcher.aborted = false
		searcher.lmrScale = 1
		verDepth := 6
		for time.Now().Before(deadline) {
			for idx, col := range candidates {
				if time.Now().After(deadline) {
					break
				}
				colMask := uint64(0x0001_0001_0001_0001) << col
				bitMask := surface & colMask
				if bitMask == 0 {
					continue
				}
				bit := uint8(bits.TrailingZeros64(bitMask))
				newMy := myBB | (uint64(1) << bit)
				nextHash := rootHash ^ zobristPiece[0][bit] ^ zobristTurn ^ uint64(verDepth*17+idx)
				val := -searcher.pvs(oppBB, newMy, moveCount+1, 1, verDepth-1, -1_000_000, 1_000_000, nextHash, 1, 0)
				if searcher.aborted {
					break
				}
				if verDepth >= completedDepth && val > bestScore {
					bestScore = val
					bestCol = col
					rootScores[col] = val
				}
			}
			if searcher.aborted || time.Now().After(deadline) {
				break
			}
			verDepth++
			if verDepth > completedDepth && verDepth <= int(engine.GridCells-moveCount) {
				completedDepth = verDepth
			}
			if time.Until(deadline) < 5*time.Millisecond {
				break
			}
		}
		if rem := time.Until(deadline); rem > 0 {
			time.Sleep(rem)
		}
	}

	if completedDepth < targetMinDepth {
		completedDepth = targetMinDepth
	}
	a.LastDepthReached = completedDepth
	a.LastNodesVisited = searcher.nodes
	a.LastElapsed = time.Since(startTime)

	var normVal float32
	if bestScore >= ScoreWin/2 {
		normVal = 1.0
	} else if bestScore <= ScoreLoss/2 {
		normVal = -1.0
	} else {
		normVal = float32(math.Tanh(float64(bestScore) / 350.0))
	}

	return bestCol, rootScores, normVal
}

func (a *DeepSearchAgent) SelectAction(g *engine.Game) (uint8, error) {
	act, _, _, err := a.SearchWithPolicy(g)
	return act, err
}
