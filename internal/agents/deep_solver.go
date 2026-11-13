package agents

import (
	"math/bits"
	"sync"
	"time"
)

// Exact Win/Draw/Loss endgame solver for 4x4x4 gravity Tic-Tac-Toe.
//
// Arena analysis of the Stupido 6 vs deep battles (Nov 2026) showed that every lost game was already
// theoretically decided 20-30 plies before the end, through long zugzwang endgames that no
// heuristic evaluation horizon can see. An exact solver with sound tactical pruning proves positions
// with ~30-40 empty cells within a second or two, so the deep baseline uses it to (a) play proven
// wins and (b) veto root moves that are proven to lose.
//
// Values are from the perspective of the side to move: +1 win, 0 draw, -1 loss.

const (
	wdlTTBits = 23
	wdlTTSize = 1 << wdlTTBits
	wdlTTMask = wdlTTSize - 1

	// wdlUnknown marks a root move whose value could not be proven within the time budget.
	wdlUnknown = 2
)

// wdlEntry packs a solver TT entry into 64 bits:
//
//	bits  0..1  lower bound + 1      bits  2..3  upper bound + 1
//	bits  4..10 best cell + 1        bits 11..17 move count
//	bit  18     occupied flag        bits 23..63 hash verification bits
//
// The low wdlTTBits of the hash select the bucket, so only the high bits need to be stored.
type wdlEntry uint64

const (
	wdlKeyMask  = ^uint64(1<<wdlTTBits - 1)
	wdlOccupied = 1 << 18
)

func packWDL(hash uint64, lo, hi int, move uint8, mc uint8) wdlEntry {
	return wdlEntry(hash&wdlKeyMask | uint64(lo+1) | uint64(hi+1)<<2 | uint64(move)<<4 | uint64(mc)<<11 | wdlOccupied)
}

func (e wdlEntry) matches(hash uint64) bool {
	return uint64(e)&wdlOccupied != 0 && uint64(e)&wdlKeyMask == hash&wdlKeyMask
}
func (e wdlEntry) lo() int     { return int(e&3) - 1 }
func (e wdlEntry) hi() int     { return int(e>>2&3) - 1 }
func (e wdlEntry) move() uint8 { return uint8(e >> 4 & 127) }
func (e wdlEntry) mc() uint8   { return uint8(e >> 11 & 127) }
func (e wdlEntry) empty() bool { return uint64(e)&wdlOccupied == 0 }

var wdlTablePool = sync.Pool{New: func() interface{} {
	t := make([]wdlEntry, wdlTTSize)
	return &t
}}

type wdlSolver struct {
	tt       []wdlEntry
	ttRef    *[]wdlEntry
	nodes    uint64
	deadline time.Time
	aborted  bool
	timedOut bool
	// history[side][cell] accumulates cutoff credit for move ordering.
	history [2][64]int32
	// nodeLimit bounds the total node counter for the current solve call (0 = unlimited).
	nodeLimit uint64
}

func newWDLSolver(deadline time.Time) *wdlSolver {
	ref := wdlTablePool.Get().(*[]wdlEntry)
	tt := *ref
	for i := range tt {
		tt[i] = 0
	}
	return &wdlSolver{tt: tt, ttRef: ref, deadline: deadline}
}

// release returns the transposition table to the shared pool.
func (s *wdlSolver) release() {
	if s.ttRef != nil {
		wdlTablePool.Put(s.ttRef)
		s.ttRef = nil
		s.tt = nil
	}
}

func (s *wdlSolver) timeUp() bool {
	if s.aborted {
		return true
	}
	if s.nodeLimit != 0 && s.nodes >= s.nodeLimit {
		s.aborted = true
	} else if s.nodes&1023 == 0 && time.Now().After(s.deadline) {
		s.aborted = true
		s.timedOut = true
	}
	return s.aborted
}

// solveLimited runs an exact solve with at most `limit` additional nodes (0 = unlimited). It
// returns the value and whether the result is valid (not interrupted by the node limit or time).
func (s *wdlSolver) solveLimited(my, opp, myT, oppT, hash uint64, mc uint8, limit uint64) (int, bool) {
	return s.solveWindow(my, opp, myT, oppT, hash, mc, -1, 1, limit)
}

// solveWindow is solveLimited with an explicit (alpha, beta) window; null windows (-1,0) / (0,1)
// answer "is it lost?" / "is it won?" far more cheaply than an exact solve.
func (s *wdlSolver) solveWindow(my, opp, myT, oppT, hash uint64, mc uint8, alpha, beta int, limit uint64) (int, bool) {
	if s.timedOut {
		return 0, false
	}
	s.aborted = false
	s.nodeLimit = 0
	if limit != 0 {
		s.nodeLimit = s.nodes + limit
	}
	v := s.solve(my, opp, myT, oppT, hash, mc, alpha, beta)
	if s.aborted {
		return 0, false
	}
	return v, true
}

// newThreatsThrough returns the empty cells that become 3-in-a-row threats for `myBB` along the
// winning lines passing through `bit` (which `myBB` has just occupied).
func newThreatsThrough(myBB, oppBB uint64, bit uint8) uint64 {
	var t uint64
	cnt := cellWinLineCount[bit]
	lines := &cellWinLines[bit]
	for i := uint8(0); i < cnt; i++ {
		m := lines[i]
		if oppBB&m != 0 {
			continue
		}
		mb := myBB & m
		if bits.OnesCount64(mb) == 3 {
			t |= m &^ mb
		}
	}
	return t
}

// playBit applies `bit` for the side to move and returns the updated threat masks. Placing a
// stone on an empty cell can only kill the opponent's threat at that very cell (any opponent line
// through an empty cell with 3 opponent stones has that cell as its missing 4th), so threat sets
// can be maintained incrementally in O(lines through bit).
func playBit(my, opp, myT, oppT uint64, bit uint8) (newMy, newMyT, newOppT uint64) {
	b := uint64(1) << bit
	newMy = my | b
	newMyT = (myT | newThreatsThrough(newMy, opp, bit)) &^ b
	newOppT = oppT &^ b
	return
}

// colorHash computes the Zobrist hash of a position by stone colour (White=0, Black=1).
func colorHash(white, black uint64) uint64 {
	var h uint64
	for b := white; b != 0; b &= b - 1 {
		h ^= zobristPiece[0][bits.TrailingZeros64(b)]
	}
	for b := black; b != 0; b &= b - 1 {
		h ^= zobristPiece[1][bits.TrailingZeros64(b)]
	}
	return h
}

// solve returns the exact game value (side to move perspective) within window (alpha, beta).
// When the deadline expires s.aborted is set and the returned value is meaningless.
func (s *wdlSolver) solve(my, opp, myT, oppT, hash uint64, mc uint8, alpha, beta int) int {
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
		return 1
	}
	side := mc & 1
	oppImm := oppT & surf
	if oppImm != 0 {
		if oppImm&(oppImm-1) != 0 {
			return -1
		}
		bit := uint8(bits.TrailingZeros64(oppImm))
		if bit < 48 && oppT&(uint64(1)<<(bit+16)) != 0 {
			return -1
		}
		nm, nmt, not := playBit(my, opp, myT, oppT, bit)
		return -s.solve(opp, nm, not, nmt, hash^zobristPiece[side][bit], mc+1, -beta, -alpha)
	}
	moves := surf &^ (oppT >> 16)
	if moves == 0 {
		return -1
	}

	// Two-way bucket: slot 0 prefers entries closer to the root (more work behind them), slot 1
	// always accepts the newest entry.
	idx := hash & wdlTTMask &^ 1
	ttBit := uint8(255)
	lo, hi := -1, 1
	if e := s.tt[idx]; e.matches(hash) {
		lo, hi = e.lo(), e.hi()
		if m := e.move(); m != 0 {
			ttBit = m - 1
		}
	} else if e := s.tt[idx+1]; e.matches(hash) {
		lo, hi = e.lo(), e.hi()
		if m := e.move(); m != 0 {
			ttBit = m - 1
		}
	}
	if lo == hi || lo >= beta {
		return lo
	}
	if hi <= alpha {
		return hi
	}
	if lo > alpha {
		alpha = lo
	}
	if hi < beta {
		beta = hi
	}
	a0, b0 := alpha, beta

	var mvBits [16]uint8
	var mvScore [16]int32
	n := 0
	for m := moves; m != 0; m &= m - 1 {
		bit := uint8(bits.TrailingZeros64(m))
		b := uint64(1) << bit
		nt := newThreatsThrough(my|b, opp, bit)
		allT := (myT | nt) &^ b
		nsurf := surfaceBitmaskFast(occ | b)
		imm := bits.OnesCount64(allT & nsurf)
		sc := int32(CellLineCounts[bit])
		if bit == ttBit {
			sc += 1_000_000
		}
		if imm >= 2 || (allT&(allT>>16)&^oppT) != 0 {
			sc += 100_000
		}
		sc += int32(imm)*1000 + int32(bits.OnesCount64(nt))*100
		if oppT&b != 0 {
			sc += 50 // occupy an opponent's suspended threat cell
		}
		if bit < 48 && myT&(b<<16) != 0 {
			sc -= 400 // playing under our own threat lets the opponent block it
		}
		sc += s.history[side][bit]
		pos := n
		for pos > 0 && mvScore[pos-1] < sc {
			mvBits[pos] = mvBits[pos-1]
			mvScore[pos] = mvScore[pos-1]
			pos--
		}
		mvBits[pos] = bit
		mvScore[pos] = sc
		n++
	}

	best := -2
	bestBit := mvBits[0]
	for i := 0; i < n; i++ {
		bit := mvBits[i]
		nm, nmt, not := playBit(my, opp, myT, oppT, bit)
		v := -s.solve(opp, nm, not, nmt, hash^zobristPiece[side][bit], mc+1, -beta, -alpha)
		if s.aborted {
			return 0
		}
		if v > best {
			best = v
			bestBit = bit
		}
		if best > alpha {
			alpha = best
		}
		if alpha >= beta {
			if h := s.history[side][bit] + int32(64-mc); h < 900 {
				s.history[side][bit] = h
			}
			break
		}
	}

	if best <= a0 {
		hi = best
	} else if best >= b0 {
		lo = best
	} else {
		lo, hi = best, best
	}
	var mv uint8
	if best > a0 {
		mv = bestBit + 1
	}
	ne := packWDL(hash, lo, hi, mv, mc)
	switch e0 := s.tt[idx]; {
	case e0.matches(hash) || e0.empty() || mc <= e0.mc():
		s.tt[idx] = ne
	default:
		s.tt[idx+1] = ne
	}
	return best
}

const engine64Full = ^uint64(0)
