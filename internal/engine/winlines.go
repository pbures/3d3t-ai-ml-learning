package engine

// LineCategory describes the geometric orientation of a winning 4-in-a-row line.
type LineCategory string

const (
	CategoryOrthogonalX LineCategory = "Orthogonal-X" // 16 lines along X axis
	CategoryOrthogonalY LineCategory = "Orthogonal-Y" // 16 lines along Y axis
	CategoryOrthogonalZ LineCategory = "Orthogonal-Z" // 16 vertical lines along Z axis
	CategoryDiagXY      LineCategory = "Diag-XY"      // 8 planar diagonals in XY horizontal planes
	CategoryDiagXZ      LineCategory = "Diag-XZ"      // 8 planar diagonals in XZ vertical planes
	CategoryDiagYZ      LineCategory = "Diag-YZ"      // 8 planar diagonals in YZ vertical planes
	CategoryDiagXYZ     LineCategory = "Diag-XYZ"     // 4 3D space diagonals through cube center
)

// TotalWinLines is the exact number of 4-in-a-row lines in a 4x4x4 cube.
const TotalWinLines = 76

// Coord3D represents a discrete (X, Y, Z) position in {0..3}^3.
type Coord3D struct {
	X uint8
	Y uint8
	Z uint8
}

// WinLine holds the precomputed bitmask and metadata for one of the 76 winning lines.
type WinLine struct {
	Index    int
	Category LineCategory
	Mask     uint64
	Coords   [4]Coord3D
}

// WinLines holds all 76 precomputed winning lines, initialized at package load time.
var WinLines [TotalWinLines]WinLine

// WinMasks holds just the 76 uint64 bitmasks in a contiguous cache-friendly array for fast win checks.
var WinMasks [TotalWinLines]uint64

func init() {
	idx := 0
	addLine := func(cat LineCategory, x0, y0, z0, dx, dy, dz int) {
		var mask uint64
		var coords [4]Coord3D
		for step := 0; step < 4; step++ {
			x := uint8(x0 + step*dx)
			y := uint8(y0 + step*dy)
			z := uint8(z0 + step*dz)
			bit := BitIndex(x, y, z)
			mask |= uint64(1) << bit
			coords[step] = Coord3D{X: x, Y: y, Z: z}
		}
		WinLines[idx] = WinLine{
			Index:    idx,
			Category: cat,
			Mask:     mask,
			Coords:   coords,
		}
		WinMasks[idx] = mask
		idx++
	}

	// 1. Orthogonal-X lines (16): vary x in 0..3 for each fixed (y, z)
	for z := 0; z < 4; z++ {
		for y := 0; y < 4; y++ {
			addLine(CategoryOrthogonalX, 0, y, z, 1, 0, 0)
		}
	}

	// 2. Orthogonal-Y lines (16): vary y in 0..3 for each fixed (x, z)
	for z := 0; z < 4; z++ {
		for x := 0; x < 4; x++ {
			addLine(CategoryOrthogonalY, x, 0, z, 0, 1, 0)
		}
	}

	// 3. Orthogonal-Z vertical lines (16): vary z in 0..3 for each fixed (x, y)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			addLine(CategoryOrthogonalZ, x, y, 0, 0, 0, 1)
		}
	}

	// 4. Planar XY diagonals (8): 2 diagonals per horizontal plane z in 0..3
	for z := 0; z < 4; z++ {
		addLine(CategoryDiagXY, 0, 0, z, 1, 1, 0)
		addLine(CategoryDiagXY, 0, 3, z, 1, -1, 0)
	}

	// 5. Planar XZ diagonals (8): 2 diagonals per vertical plane y in 0..3
	for y := 0; y < 4; y++ {
		addLine(CategoryDiagXZ, 0, y, 0, 1, 0, 1)
		addLine(CategoryDiagXZ, 0, y, 3, 1, 0, -1)
	}

	// 6. Planar YZ diagonals (8): 2 diagonals per vertical plane x in 0..3
	for x := 0; x < 4; x++ {
		addLine(CategoryDiagYZ, x, 0, 0, 0, 1, 1)
		addLine(CategoryDiagYZ, x, 0, 3, 0, 1, -1)
	}

	// 7. 3D Space Diagonals (4): corner-to-opposite-corner
	addLine(CategoryDiagXYZ, 0, 0, 0, 1, 1, 1)
	addLine(CategoryDiagXYZ, 3, 0, 0, -1, 1, 1)
	addLine(CategoryDiagXYZ, 0, 3, 0, 1, -1, 1)
	addLine(CategoryDiagXYZ, 0, 0, 3, 1, 1, -1)
}

// CheckBitboardWin checks if the given player's 64-bit bitboard contains any of the 76 winning lines.
// Returns (true, winLineMask, winLineIndex) if won, or (false, 0, -1) otherwise.
func CheckBitboardWin(bitboard uint64) (bool, uint64, int) {
	for i, mask := range WinMasks {
		if (bitboard & mask) == mask {
			return true, mask, i
		}
	}
	return false, 0, -1
}

// GenerateWinSequence returns a deterministic sequence of valid gravity actions (0..15)
// that results in `winner` (PlayerWhite or PlayerBlack) winning specifically along `WinLines[lineIndex]`.
func GenerateWinSequence(lineIndex int, winner Player) []uint8 {
	wl := WinLines[lineIndex]

	isTarget := make(map[Coord3D]bool, 4)
	usedCols := make(map[uint8]bool, 4)
	for _, c := range wl.Coords {
		isTarget[c] = true
		usedCols[4*c.Y+c.X] = true
	}

	g := NewGame()
	var moves []uint8

	// Helper that plays a safe filler move in a column not belonging to the winning line
	playSafeFiller := func() {
		for a := uint8(0); a < MaxActions; a++ {
			if usedCols[a] || !g.IsActionValid(a) {
				continue
			}
			// Simulate move on a clone to ensure filler never accidentally triggers a win
			clone := g.Clone()
			if _, err := clone.ApplyMove(a); err == nil && clone.Status == StatusOngoing {
				_, _ = g.ApplyMove(a)
				moves = append(moves, a)
				return
			}
		}
	}

	// Build each of the 4 target coordinates in order (0, 1, 2, and finally 3 which triggers the win)
	for step := 0; step < 4; step++ {
		c := wl.Coords[step]
		colAction := 4*c.Y + c.X

		for g.Status == StatusOngoing {
			curZ, ok := g.LandingZ(colAction)
			if !ok || curZ > c.Z {
				break
			}
			here := Coord3D{X: c.X, Y: c.Y, Z: curZ}

			var desiredPlayer Player
			if isTarget[here] {
				desiredPlayer = winner
			} else {
				// Support cell beneath the winning line:
				// Place `winner` only when int(curZ) == step (at most 1 support cell per column and per layer),
				// and `winner.Opponent()` everywhere else, ensuring neither vertical columns nor horizontal
				// layers underneath `wl` can ever accumulate 4 of the same color.
				desiredPlayer = winner.Opponent()
				if int(curZ) == step {
					desiredPlayer = winner
				}
				clone := g.Clone()
				clone.CurrentPlayer = desiredPlayer
				if _, err := clone.ApplyMove(colAction); err != nil || clone.Status != StatusOngoing {
					desiredPlayer = desiredPlayer.Opponent()
				}
			}

			if g.CurrentPlayer != desiredPlayer {
				playSafeFiller()
			}

			if _, err := g.ApplyMove(colAction); err != nil {
				break
			}
			moves = append(moves, colAction)
			if curZ == c.Z {
				break
			}
		}
	}

	return moves
}

