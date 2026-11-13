package agents

import (
	"math/bits"
	"time"
)

// deepSearcher is the fixed-depth check-extended PVS used as the tactical lookahead of the hybrid
// neural agents (ttt3dnet / ttt3d-deep-gray1). Its behaviour is intentionally kept identical to the
// search those models were tuned and trained against; the standalone "deep" baseline agent uses
// the stronger pvsSearcher + wdlSolver pair defined in deep_pvs.go and deep_solver.go instead.
type deepSearcher struct {
	tt          []ttEntry
	killers     [64][2]uint8
	history     [2][16]int32
	nodes       uint64
	deadline    time.Time
	hasDeadline bool
	aborted     bool
}

func newDeepSearcher(deadline time.Time, hasDeadline bool) *deepSearcher {
	return &deepSearcher{
		tt:          make([]ttEntry, ttSize),
		deadline:    deadline,
		hasDeadline: hasDeadline,
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
			val = -s.pvs(oppBB, newMy, moveCount+1, ply+1, depth-1, -beta, -alpha, nextHash, 1-side, extensions)
		} else {
			// Late Move Reductions (LMR) for non-PV moves (i >= 1)
			reduction := 0
			if depth >= 3 {
				reduction = 1 + (depth / 3) + (i / 3)
				if depth-1-reduction < 1 {
					reduction = depth - 2
				}
			}

			// Null-window search (with LMR reduction)
			val = -s.pvs(oppBB, newMy, moveCount+1, ply+1, depth-1-reduction, -alpha-1, -alpha, nextHash, 1-side, extensions)

			// Re-search at tighter depth if reduced search beat alpha
			if val > alpha && reduction > 1 && !s.aborted {
				midDepth := depth - 1 - (reduction / 2)
				val = -s.pvs(oppBB, newMy, moveCount+1, ply+1, midDepth, -alpha-1, -alpha, nextHash, 1-side, extensions)
			}
			// Re-search with full window if null-window search falls inside (alpha, beta)
			if val > alpha && val < beta && !s.aborted {
				remDepth := depth - 1
				if depth >= 8 && i >= 2 {
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
