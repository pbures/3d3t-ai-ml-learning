package engine

import (
	"errors"
	"math/bits"
	"testing"
)

func TestPlayerAndStatusStrings(t *testing.T) {
	if PlayerWhite.String() != "WHITE" || PlayerBlack.String() != "BLACK" || PlayerNone.String() != "NONE" {
		t.Fatalf("unexpected Player.String() values")
	}
	if PlayerWhite.Opponent() != PlayerBlack || PlayerBlack.Opponent() != PlayerWhite || PlayerNone.Opponent() != PlayerNone {
		t.Fatalf("unexpected Player.Opponent() values")
	}
	if StatusOngoing.String() != "ONGOING" ||
		StatusWhiteWon.String() != "WHITE_WON" ||
		StatusBlackWon.String() != "BLACK_WON" ||
		StatusDraw.String() != "DRAW" ||
		GameStatus(99).String() != "UNKNOWN" {
		t.Fatalf("unexpected GameStatus.String() values")
	}
}

func TestCoordinateAndActionBijection(t *testing.T) {
	seenBits := make(map[uint8]bool)
	for z := uint8(0); z < 4; z++ {
		for y := uint8(0); y < 4; y++ {
			for x := uint8(0); x < 4; x++ {
				if !ValidCoord(x, y, z) {
					t.Fatalf("expected (%d,%d,%d) to be valid", x, y, z)
				}
				bit := BitIndex(x, y, z)
				if bit >= 64 {
					t.Fatalf("bit out of range: %d", bit)
				}
				if seenBits[bit] {
					t.Fatalf("duplicate bit index %d", bit)
				}
				seenBits[bit] = true

				rx, ry, rz := CoordsFromBit(bit)
				if rx != x || ry != y || rz != z {
					t.Fatalf("round-trip failed for (%d,%d,%d) -> bit %d -> (%d,%d,%d)", x, y, z, bit, rx, ry, rz)
				}
			}
		}
	}
	if len(seenBits) != 64 {
		t.Fatalf("expected 64 unique bits, got %d", len(seenBits))
	}

	if ValidCoord(4, 0, 0) || ValidCoord(0, 4, 0) || ValidCoord(0, 0, 4) {
		t.Fatalf("expected out-of-bounds coordinates to be invalid")
	}

	for a := uint8(0); a < 16; a++ {
		x, y, err := ActionToXY(a)
		if err != nil {
			t.Fatalf("unexpected error for action %d: %v", a, err)
		}
		ra, err := XYToAction(x, y)
		if err != nil || ra != a {
			t.Fatalf("action round-trip failed for %d -> (%d,%d) -> %d (err=%v)", a, x, y, ra, err)
		}
	}

	if _, _, err := ActionToXY(16); !errors.Is(err, ErrInvalidAction) {
		t.Fatalf("expected ErrInvalidAction for action 16, got %v", err)
	}
	if _, err := XYToAction(4, 0); !errors.Is(err, ErrInvalidCoords) {
		t.Fatalf("expected ErrInvalidCoords for x=4, got %v", err)
	}
	if _, err := XYToAction(0, 4); !errors.Is(err, ErrInvalidCoords) {
		t.Fatalf("expected ErrInvalidCoords for y=4, got %v", err)
	}
}

func TestWinLinesPrecomputation(t *testing.T) {
	if len(WinLines) != TotalWinLines {
		t.Fatalf("expected 76 winning lines, got %d", len(WinLines))
	}

	catCounts := make(map[LineCategory]int)
	uniqueMasks := make(map[uint64]bool)

	for i, wl := range WinLines {
		if wl.Index != i {
			t.Fatalf("expected index %d, got %d", i, wl.Index)
		}
		if wl.Mask != WinMasks[i] {
			t.Fatalf("WinLines[%d].Mask != WinMasks[%d]", i, i)
		}
		if bits.OnesCount64(wl.Mask) != 4 {
			t.Fatalf("expected 4 bits set in WinLines[%d].Mask, got %d", i, bits.OnesCount64(wl.Mask))
		}
		if uniqueMasks[wl.Mask] {
			t.Fatalf("duplicate winning line mask at index %d", i)
		}
		uniqueMasks[wl.Mask] = true
		catCounts[wl.Category]++
	}

	expectedCats := map[LineCategory]int{
		CategoryOrthogonalX: 16,
		CategoryOrthogonalY: 16,
		CategoryOrthogonalZ: 16,
		CategoryDiagXY:      8,
		CategoryDiagXZ:      8,
		CategoryDiagYZ:      8,
		CategoryDiagXYZ:     4,
	}
	for cat, want := range expectedCats {
		if got := catCounts[cat]; got != want {
			t.Fatalf("category %s: expected %d lines, got %d", cat, want, got)
		}
	}
}

func TestGravityAndFullColumnRules(t *testing.T) {
	g := NewGame()
	if g.CurrentPlayer != PlayerWhite || g.Status != StatusOngoing || g.ValidMovesMask() != 0xFFFF {
		t.Fatalf("unexpected initial game state")
	}

	// Drop 4 balls into column (2, 1) -> action = 4*1 + 2 = 6
	colAction, err := XYToAction(2, 1)
	if err != nil {
		t.Fatal(err)
	}

	for expectedZ := uint8(0); expectedZ < 4; expectedZ++ {
		if !g.IsActionValid(colAction) {
			t.Fatalf("expected action %d to be valid at z=%d", colAction, expectedZ)
		}
		expectedPlayer := PlayerWhite
		if expectedZ%2 == 1 {
			expectedPlayer = PlayerBlack
		}

		rec, err := g.ApplyMoveXY(2, 1)
		if err != nil {
			t.Fatalf("ApplyMoveXY failed at z=%d: %v", expectedZ, err)
		}
		if rec.X != 2 || rec.Y != 1 || rec.Z != expectedZ || rec.Player != expectedPlayer || rec.MoveNumber != expectedZ+1 {
			t.Fatalf("unexpected MoveRecord at z=%d: %+v", expectedZ, rec)
		}

		cellPlayer, err := g.CellAt(2, 1, expectedZ)
		if err != nil || cellPlayer != expectedPlayer {
			t.Fatalf("expected CellAt(2,1,%d)=%v, got %v (err=%v)", expectedZ, expectedPlayer, cellPlayer, err)
		}
	}

	// Column (2, 1) now has z=3 occupied, so it must be marked illegal
	if g.IsActionValid(colAction) {
		t.Fatalf("expected full column %d to be invalid", colAction)
	}
	if (g.ValidMovesMask() & (uint16(1) << colAction)) != 0 {
		t.Fatalf("expected bit %d to be 0 in ValidMovesMask", colAction)
	}
	if _, ok := g.LandingZ(colAction); ok {
		t.Fatalf("expected LandingZ(%d) to return false when full", colAction)
	}
	if _, ok := g.LandingZ(99); ok {
		t.Fatalf("expected LandingZ(99) to return false for invalid action")
	}

	// Attempting a 5th ball in column (2, 1) must return ErrColumnFull
	if _, err := g.ApplyMove(colAction); !errors.Is(err, ErrColumnFull) {
		t.Fatalf("expected ErrColumnFull, got %v", err)
	}
	if _, err := g.ApplyMoveXY(9, 9); !errors.Is(err, ErrInvalidCoords) {
		t.Fatalf("expected ErrInvalidCoords, got %v", err)
	}
	if _, err := g.CellAt(4, 0, 0); !errors.Is(err, ErrInvalidCoords) {
		t.Fatalf("expected ErrInvalidCoords from CellAt, got %v", err)
	}
}

// playWinLineViaGravity plays the sequence generated by GenerateWinSequence(wl.Index, winner).
func playWinLineViaGravity(t *testing.T, wl WinLine, winner Player) *Game {
	t.Helper()
	g := NewGame()
	moves := GenerateWinSequence(wl.Index, winner)
	for i, a := range moves {
		if _, err := g.ApplyMove(a); err != nil {
			t.Fatalf("line %d (%s) move %d (action %d) failed: %v", wl.Index, wl.Category, i, a, err)
		}
	}
	return g
}

func TestAll76WinningLinesWhiteAndBlack(t *testing.T) {
	for _, wl := range WinLines {
		// 1. Test White winning along line wl
		gw := playWinLineViaGravity(t, wl, PlayerWhite)
		if gw.Status != StatusWhiteWon || gw.Winner != PlayerWhite {
			t.Fatalf("line %d (%s): expected WHITE_WON, got status=%s winner=%s", wl.Index, wl.Category, gw.Status, gw.Winner)
		}
		if gw.WinLineMask != wl.Mask || gw.WinLineIndex != wl.Index {
			t.Fatalf("line %d (%s): expected mask %016x idx %d, got %016x idx %d",
				wl.Index, wl.Category, wl.Mask, wl.Index, gw.WinLineMask, gw.WinLineIndex)
		}
		if gw.ValidMovesMask() != 0 || gw.IsActionValid(0) {
			t.Fatalf("line %d: expected 0 valid moves after game over", wl.Index)
		}
		if _, err := gw.ApplyMove(0); !errors.Is(err, ErrGameAlreadyOver) {
			t.Fatalf("line %d: expected ErrGameAlreadyOver after win, got %v", wl.Index, err)
		}

		// 2. Test Black winning along line wl
		gb := playWinLineViaGravity(t, wl, PlayerBlack)
		if gb.Status != StatusBlackWon || gb.Winner != PlayerBlack {
			t.Fatalf("line %d (%s): expected BLACK_WON, got status=%s winner=%s", wl.Index, wl.Category, gb.Status, gb.Winner)
		}
		if gb.WinLineMask != wl.Mask || gb.WinLineIndex != wl.Index {
			t.Fatalf("line %d (%s): expected mask %016x idx %d, got %016x idx %d",
				wl.Index, wl.Category, wl.Mask, wl.Index, gb.WinLineMask, gb.WinLineIndex)
		}
	}
}

func TestFull64MoveDrawGame(t *testing.T) {
	g := NewGame()
	seq := GenerateDrawSequence()

	for i, action := range seq {
		if g.Status != StatusOngoing {
			t.Fatalf("game ended prematurely at move %d with status %s", i, g.Status)
		}
		rec, err := g.ApplyMove(action)
		if err != nil {
			t.Fatalf("move %d (action %d) failed: %v", i+1, action, err)
		}
		if rec.MoveNumber != uint8(i+1) {
			t.Fatalf("expected move number %d, got %d", i+1, rec.MoveNumber)
		}
	}

	if g.Status != StatusDraw {
		t.Fatalf("expected StatusDraw after 64 moves, got %s (winner=%s)", g.Status, g.Winner)
	}
	if g.Winner != PlayerNone || g.WinLineIndex != -1 || g.WinLineMask != 0 {
		t.Fatalf("expected no winner on draw, got winner=%s idx=%d", g.Winner, g.WinLineIndex)
	}
	if g.MoveCount != 64 || g.OccupiedBitboard() != FullBoardMask {
		t.Fatalf("expected 64 moves and full bitboard, got moves=%d occ=%016x", g.MoveCount, g.OccupiedBitboard())
	}
	if g.ValidMovesMask() != 0 {
		t.Fatalf("expected ValidMovesMask()==0 on draw, got %04x", g.ValidMovesMask())
	}
	if len(g.MovesHistory()) != 64 {
		t.Fatalf("expected 64 history records, got %d", len(g.MovesHistory()))
	}
}

func TestCloneAndCanonicalTensor(t *testing.T) {
	g := NewGame()
	tensor0 := g.CanonicalTensor()

	// On an empty board:
	// - Channel 0 (0..63) and Channel 1 (64..127) must be all 0.
	// - Channel 2 (128..191) must have 1s at z=0 (indices 128..143) and 0s at z=1..3 (indices 144..191).
	for i := 0; i < 128; i++ {
		if tensor0[i] != 0 {
			t.Fatalf("expected 0 at index %d on empty board, got %d", i, tensor0[i])
		}
	}
	for i := 128; i < 144; i++ {
		if tensor0[i] != 1 {
			t.Fatalf("expected 1 at z=0 gravity surface index %d, got %d", i, tensor0[i])
		}
	}
	for i := 144; i < 192; i++ {
		if tensor0[i] != 0 {
			t.Fatalf("expected 0 at z>=1 gravity surface index %d, got %d", i, tensor0[i])
		}
	}

	// White plays action 5 -> lands at bit 5 (x=1, y=1, z=0). Turn switches to Black.
	_, err := g.ApplyMove(5)
	if err != nil {
		t.Fatal(err)
	}

	cloned := g.Clone()
	if cloned.White != g.White || cloned.CurrentPlayer != PlayerBlack {
		t.Fatalf("Clone did not preserve state")
	}

	// Now it is Black's turn:
	// - Channel 0 (Black's pieces) = all 0
	// - Channel 1 (White's pieces) = 1 at bit 5 (offset 64 + 5 = 69)
	// - Channel 2 (Gravity surface) = for column 5, landing spot is now z=1 (bit 5 + 16 = 21 -> offset 128 + 21 = 149)
	tensor1 := g.CanonicalTensor()
	if tensor1[5] != 0 {
		t.Fatalf("expected Black's channel 0 at bit 5 to be 0")
	}
	if tensor1[64+5] != 1 {
		t.Fatalf("expected Opponent (White) channel 1 at bit 5 to be 1")
	}
	if tensor1[128+5] != 0 || tensor1[128+21] != 1 {
		t.Fatalf("expected column 5 gravity landing spot to shift from z=0 (bit 5) to z=1 (bit 21)")
	}
}

func BenchmarkApplyMoveAndCheckWin(b *testing.B) {
	seq := GenerateDrawSequence()
	g := NewGame()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Reset()
		for _, a := range seq {
			_, _ = g.ApplyMove(a)
		}
	}
}
