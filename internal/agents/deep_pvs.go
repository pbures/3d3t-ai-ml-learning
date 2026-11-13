package agents

import (
	"math"
	"math/bits"
	"sync"
	"time"
)

// pvsSearcher is the iterative-deepening Principal Variation Search used by the "deep" baseline.
//
// Compared to the hybrid deepSearcher it:
//   - maintains both players' threat sets incrementally (O(lines through the played cell) per node),
//   - uses conventional logarithmic Late Move Reductions that never reduce forcing moves and always
//     re-search at full depth when a reduced search beats alpha (so the reported depth is real),
//   - resolves forced replies (single immediate threat) without consuming depth (check extension),
//   - adds a zugzwang parity term (see parityThreatScore) on top of evaluateBitboardFast.
const (
	pvsTTBits = 20
	pvsTTSize = 1 << pvsTTBits
	pvsTTMask = pvsTTSize - 1
	pvsMaxPly = 72
)

type pvsEntry struct {
	key   uint64
	score int32
	depth int8
	flag  uint8
	move  uint8
}

var pvsTablePool = sync.Pool{New: func() interface{} {
	t := make([]pvsEntry, pvsTTSize)
	return &t
}}

// lmrTable[depth][moveIndex] holds the logarithmic late-move reduction.
var lmrTable [64][16]int8

func init() {
	for d := 1; d < 64; d++ {
		for i := 1; i < 16; i++ {
			r := 0.5 + math.Log(float64(d))*math.Log(float64(i))/2.25
			lmrTable[d][i] = int8(r)
		}
	}
}

type pvsSearcher struct {
	tt       []pvsEntry
	ttRef    *[]pvsEntry
	killers  [pvsMaxPly][2]uint8
	history  [2][16]int32
	nodes    uint64
	deadline time.Time
	aborted  bool
}

func newPVSSearcher(deadline time.Time) *pvsSearcher {
	ref := pvsTablePool.Get().(*[]pvsEntry)
	tt := *ref
	for i := range tt {
		tt[i] = pvsEntry{}
	}
	s := &pvsSearcher{tt: tt, ttRef: ref, deadline: deadline}
	for p := range s.killers {
		s.killers[p] = [2]uint8{255, 255}
	}
	return s
}

func (s *pvsSearcher) release() {
	if s.ttRef != nil {
		pvsTablePool.Put(s.ttRef)
		s.ttRef = nil
		s.tt = nil
	}
}

func (s *pvsSearcher) timeUp() bool {
	if s.aborted {
		return true
	}
	if s.nodes&1023 == 0 && time.Now().After(s.deadline) {
		s.aborted = true
	}
	return s.aborted
}

// parityThreatScore scores suspended (not yet playable) threats by Connect-4 style zugzwang parity,
// from the perspective of the side to move.
//
// All 16 columns have even height (4) and the board has an even number of cells, so under the
// "follow-up" strategy (always answering in the same column) the first player (White) ends up with
// the cells of rows z=0/z=2 and the second player (Black) with rows z=1/z=3. A suspended threat
// therefore tends to be cashed in by its owner when it sits on the owner's "good" row: z=2 for White
// (odd row 3 in Connect-4 terms) and z=1/z=3 for Black (even rows). Only the lowest suspended threat of each column matters for zugzwang,
// because play in that column stops there.
func parityThreatScore(myT, oppT, surface uint64, moveCount uint8) int {
	whiteToMove := moveCount&1 == 0
	wT, bT := myT, oppT
	if !whiteToMove {
		wT, bT = oppT, myT
	}
	susp := (wT | bT) &^ surface
	if susp == 0 {
		return 0
	}
	wGood, wBad, bGood, bBad := 0, 0, 0, 0
	for col := uint8(0); col < 16; col++ {
		colMask := uint64(0x0001_0001_0001_0001) << col
		cs := susp & colMask
		if cs == 0 {
			continue
		}
		low := cs & -cs
		z := bits.TrailingZeros64(low) >> 4
		wHas := wT&low != 0
		bHas := bT&low != 0
		switch {
		case wHas && bHas:
			// Shared threat cell: whoever is forced below it loses; treat as neutral.
		case wHas:
			if z == 2 {
				wGood++
			} else {
				wBad++
			}
		case bHas:
			if z == 1 || z == 3 {
				bGood++
			} else {
				bBad++
			}
		}
	}
	score := 90*(wGood-bGood) + 20*(wBad-bBad)
	if wGood > 0 && bGood == 0 {
		score += 140
	} else if bGood > 0 && wGood == 0 {
		score -= 140
	}
	if !whiteToMove {
		score = -score
	}
	return score
}

func (s *pvsSearcher) evaluate(my, opp, myT, oppT uint64, mc uint8) int {
	surface := surfaceBitmaskFast(my | opp)
	return evaluateBitboardFast(my, opp, myT, oppT, mc) + parityThreatScore(myT, oppT, surface, mc)
}

// search is a fail-soft negamax PVS. Scores are from the side to move's
// perspective; mate scores are encoded with the absolute move count so they are TT-safe.
func (s *pvsSearcher) search(my, opp, myT, oppT uint64, mc uint8, ply, depth, alpha, beta int, hash uint64) int {
	s.nodes++
	if s.timeUp() {
		return 0
	}
	occ := my | opp
	if occ == engine64Full {
		return 0
	}
	surf := surfaceBitmaskFast(occ)
	if myT&surf != 0 {
		return ScoreWin - int(mc+1)
	}
	side := mc & 1
	oppImm := oppT & surf
	if oppImm != 0 {
		if oppImm&(oppImm-1) != 0 {
			return ScoreLoss + int(mc+2)
		}
		bit := uint8(bits.TrailingZeros64(oppImm))
		if bit < 48 && oppT&(uint64(1)<<(bit+16)) != 0 {
			return ScoreLoss + int(mc+2)
		}
		// Forced block: check extension (depth is not consumed).
		nm, nmt, not := playBit(my, opp, myT, oppT, bit)
		return -s.search(opp, nm, not, nmt, mc+1, ply+1, depth, -beta, -alpha, hash^zobristPiece[side][bit])
	}
	if depth <= 0 || ply >= pvsMaxPly-1 {
		return s.evaluate(my, opp, myT, oppT, mc)
	}
	safe := surf &^ (oppT >> 16)
	if safe == 0 {
		return ScoreLoss + int(mc+2)
	}

	origAlpha := alpha
	idx := hash & pvsTTMask
	e := s.tt[idx]
	ttMove := uint8(255)
	if e.flag != 0 && e.key == hash {
		ttMove = e.move
		if int(e.depth) >= depth {
			sc := int(e.score)
			switch e.flag {
			case ttFlagExact:
				return sc
			case ttFlagLower:
				if sc >= beta {
					return sc
				}
			case ttFlagUpper:
				if sc <= alpha {
					return sc
				}
			}
		}
	}

	var mvBits [16]uint8
	var mvScore [16]int32
	var mvForcing [16]bool
	n := 0
	k0, k1 := s.killers[ply][0], s.killers[ply][1]
	for m := safe; m != 0; m &= m - 1 {
		bit := uint8(bits.TrailingZeros64(m))
		col := bit & 15
		b := uint64(1) << bit
		nt := newThreatsThrough(my|b, opp, bit)
		allT := (myT | nt) &^ b
		imm := bits.OnesCount64(allT & surfaceBitmaskFast(occ|b))
		var sc int32
		forcing := imm > 0
		switch {
		case col == ttMove:
			sc = 1 << 30
		case imm >= 2 || (allT&(allT>>16)&^oppT) != 0:
			sc = 1 << 28
		case forcing:
			sc = 1 << 24
		case col == k0:
			sc = 1 << 22
		case col == k1:
			sc = 1 << 21
		}
		sc += s.history[side][col] + int32(bits.OnesCount64(nt))*600 + int32(CellLineCounts[bit])*16
		if oppT&b != 0 {
			sc += 400
		}
		pos := n
		for pos > 0 && mvScore[pos-1] < sc {
			mvBits[pos] = mvBits[pos-1]
			mvScore[pos] = mvScore[pos-1]
			mvForcing[pos] = mvForcing[pos-1]
			pos--
		}
		mvBits[pos] = bit
		mvScore[pos] = sc
		mvForcing[pos] = forcing
		n++
	}

	best := -1_000_000
	bestCol := mvBits[0] & 15
	for i := 0; i < n; i++ {
		bit := mvBits[i]
		nm, nmt, not := playBit(my, opp, myT, oppT, bit)
		nh := hash ^ zobristPiece[side][bit]
		var v int
		if i == 0 {
			v = -s.search(opp, nm, not, nmt, mc+1, ply+1, depth-1, -beta, -alpha, nh)
		} else {
			r := 0
			if depth >= 3 && i >= 2 && !mvForcing[i] {
				d := depth
				if d > 63 {
					d = 63
				}
				r = int(lmrTable[d][i])
				if r > depth-2 {
					r = depth - 2
				}
			}
			v = -s.search(opp, nm, not, nmt, mc+1, ply+1, depth-1-r, -alpha-1, -alpha, nh)
			if v > alpha && r > 0 && !s.aborted {
				v = -s.search(opp, nm, not, nmt, mc+1, ply+1, depth-1, -alpha-1, -alpha, nh)
			}
			if v > alpha && v < beta && !s.aborted {
				v = -s.search(opp, nm, not, nmt, mc+1, ply+1, depth-1, -beta, -alpha, nh)
			}
		}
		if s.aborted {
			return 0
		}
		if v > best {
			best = v
			bestCol = bit & 15
		}
		if best > alpha {
			alpha = best
		}
		if alpha >= beta {
			col := bit & 15
			if !mvForcing[i] {
				if s.killers[ply][0] != col {
					s.killers[ply][1] = s.killers[ply][0]
					s.killers[ply][0] = col
				}
				if h := s.history[side][col] + int32(depth*depth); h < 1<<20 {
					s.history[side][col] = h
				}
			}
			break
		}
	}

	flag := uint8(ttFlagExact)
	if best <= origAlpha {
		flag = ttFlagUpper
	} else if best >= beta {
		flag = ttFlagLower
	}
	s.tt[idx] = pvsEntry{key: hash, score: int32(best), depth: int8(depth), flag: flag, move: bestCol}
	return best
}
