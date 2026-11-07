package engine

import (
	"errors"
	"fmt"
)

// Player represents a participant or cell state in 3T3D.
type Player uint8

const (
	PlayerNone  Player = 0
	PlayerWhite Player = 1
	PlayerBlack Player = 2
)

// Opponent returns the opposite player (White <-> Black), or PlayerNone if invalid.
func (p Player) Opponent() Player {
	switch p {
	case PlayerWhite:
		return PlayerBlack
	case PlayerBlack:
		return PlayerWhite
	default:
		return PlayerNone
	}
}

// String returns the human-readable name of the player.
func (p Player) String() string {
	switch p {
	case PlayerWhite:
		return "WHITE"
	case PlayerBlack:
		return "BLACK"
	default:
		return "NONE"
	}
}

// GameStatus represents the lifecycle state of a 3T3D game.
type GameStatus uint8

const (
	StatusOngoing  GameStatus = 0
	StatusWhiteWon GameStatus = 1
	StatusBlackWon GameStatus = 2
	StatusDraw     GameStatus = 3
)

// String returns the human-readable name of the game status.
func (s GameStatus) String() string {
	switch s {
	case StatusOngoing:
		return "ONGOING"
	case StatusWhiteWon:
		return "WHITE_WON"
	case StatusBlackWon:
		return "BLACK_WON"
	case StatusDraw:
		return "DRAW"
	default:
		return "UNKNOWN"
	}
}

const (
	GridDim             = 4
	MaxActions          = 16
	GridCells           = 64
	CanonicalChannels   = 3
	CanonicalTensorSize = CanonicalChannels * GridCells // 192 bytes (3 x 4 x 4 x 4)
	FullBoardMask       = ^uint64(0)
)

var (
	ErrInvalidAction   = errors.New("action out of range: must be in 0..15")
	ErrInvalidCoords   = errors.New("coordinates out of range: x, y, z must be in 0..3")
	ErrColumnFull      = errors.New("selected column (x, y) is already full (z=3 occupied)")
	ErrGameAlreadyOver = errors.New("cannot apply move: game has already terminated")
)

// MoveRecord stores metadata for a single ball placement in the game history.
type MoveRecord struct {
	Action     uint8
	X          uint8
	Y          uint8
	Z          uint8
	Player     Player
	MoveNumber uint8 // 1-indexed (1..64)
}

// Game represents the complete state of a single 3T3D match using 64-bit bitboards.
type Game struct {
	White         uint64
	Black         uint64
	CurrentPlayer Player
	Status        GameStatus
	Winner        Player
	WinLineMask   uint64
	WinLineIndex  int
	MoveCount     uint8
	History       [GridCells]MoveRecord
}

// NewGame allocates and initializes a fresh 3T3D game with White to move first.
func NewGame() *Game {
	g := &Game{}
	g.Reset()
	return g
}

// Reset restores the game to an empty 4x4x4 board with White to move first.
func (g *Game) Reset() {
	g.White = 0
	g.Black = 0
	g.CurrentPlayer = PlayerWhite
	g.Status = StatusOngoing
	g.Winner = PlayerNone
	g.WinLineMask = 0
	g.WinLineIndex = -1
	g.MoveCount = 0
}

// Clone returns a deep copy of the Game state.
func (g *Game) Clone() *Game {
	copyGame := *g
	return &copyGame
}

// ValidCoord returns true if (x, y, z) lies inside the 4x4x4 cube {0..3}^3.
func ValidCoord(x, y, z uint8) bool {
	return x < GridDim && y < GridDim && z < GridDim
}

// BitIndex maps a 3D coordinate (x, y, z) in {0..3}^3 to a bit index in {0..63}.
// Formula: bitIndex(x, y, z) = 16*z + 4*y + x
func BitIndex(x, y, z uint8) uint8 {
	return 16*z + 4*y + x
}

// CoordsFromBit converts a bit index in {0..63} back to (x, y, z) in {0..3}^3.
func CoordsFromBit(bit uint8) (x, y, z uint8) {
	x = bit & 3
	y = (bit >> 2) & 3
	z = (bit >> 4) & 3
	return x, y, z
}

// XYToAction converts a 2D column coordinate (x, y) in {0..3}^2 into an action index in {0..15}.
func XYToAction(x, y uint8) (uint8, error) {
	if x >= GridDim || y >= GridDim {
		return 0, fmt.Errorf("%w: got x=%d, y=%d", ErrInvalidCoords, x, y)
	}
	return 4*y + x, nil
}

// ActionToXY converts an action index in {0..15} into column coordinates (x, y) in {0..3}^2.
func ActionToXY(action uint8) (x, y uint8, err error) {
	if action >= MaxActions {
		return 0, 0, fmt.Errorf("%w: got action=%d", ErrInvalidAction, action)
	}
	return action & 3, (action >> 2) & 3, nil
}

// OccupiedBitboard returns the combined bitmask of all placed balls (White | Black).
func (g *Game) OccupiedBitboard() uint64 {
	return g.White | g.Black
}

// ValidMovesMask returns a 16-bit integer where bit `a` is 1 iff action `a` in {0..15} is legal.
// When the game is over (Status != StatusOngoing), returns 0.
func (g *Game) ValidMovesMask() uint16 {
	if g.Status != StatusOngoing {
		return 0
	}
	occupied := g.White | g.Black
	// Top layer z=3 occupies bits 48..63. A column is playable iff its z=3 bit is 0.
	return uint16((^occupied) >> 48)
}

// IsActionValid returns true if the game is ongoing and column `action` in {0..15} is not full.
func (g *Game) IsActionValid(action uint8) bool {
	if action >= MaxActions || g.Status != StatusOngoing {
		return false
	}
	return (g.ValidMovesMask() & (uint16(1) << action)) != 0
}

// LandingZ computes the lowest unoccupied z coordinate in {0..3} for column `action` in {0..15}.
// Returns (z, true) if the column has space, or (0, false) if the column is full or action is invalid.
func (g *Game) LandingZ(action uint8) (uint8, bool) {
	if action >= MaxActions {
		return 0, false
	}
	occupied := g.White | g.Black
	for z := uint8(0); z < GridDim; z++ {
		bit := action + 16*z
		if (occupied & (uint64(1) << bit)) == 0 {
			return z, true
		}
	}
	return 0, false
}

// CellAt returns the Player occupying position (x, y, z), or an error if out of bounds.
func (g *Game) CellAt(x, y, z uint8) (Player, error) {
	if !ValidCoord(x, y, z) {
		return PlayerNone, fmt.Errorf("%w: (%d, %d, %d)", ErrInvalidCoords, x, y, z)
	}
	mask := uint64(1) << BitIndex(x, y, z)
	if (g.White & mask) != 0 {
		return PlayerWhite, nil
	}
	if (g.Black & mask) != 0 {
		return PlayerBlack, nil
	}
	return PlayerNone, nil
}

// ApplyMoveXY validates and applies a ball drop at column (x, y).
func (g *Game) ApplyMoveXY(x, y uint8) (MoveRecord, error) {
	action, err := XYToAction(x, y)
	if err != nil {
		return MoveRecord{}, err
	}
	return g.ApplyMove(action)
}

// ApplyMove validates and executes action `a` in {0..15} for the current player:
// 1. Verifies the game is ongoing and action is in {0..15}.
// 2. Resolves gravity to find the lowest empty z in {0..3}.
// 3. Updates the active player's bitboard and move history.
// 4. Checks for a 4-in-a-row win across all 76 winning lines or a 64-move draw.
// 5. Advances CurrentPlayer if the game remains ongoing.
func (g *Game) ApplyMove(action uint8) (MoveRecord, error) {
	if g.Status != StatusOngoing {
		return MoveRecord{}, ErrGameAlreadyOver
	}
	x, y, err := ActionToXY(action)
	if err != nil {
		return MoveRecord{}, err
	}
	z, ok := g.LandingZ(action)
	if !ok {
		return MoveRecord{}, fmt.Errorf("%w: column (%d, %d) action %d", ErrColumnFull, x, y, action)
	}

	bit := BitIndex(x, y, z)
	bitMask := uint64(1) << bit
	mover := g.CurrentPlayer

	var activeBoard uint64
	if mover == PlayerWhite {
		g.White |= bitMask
		activeBoard = g.White
	} else {
		g.Black |= bitMask
		activeBoard = g.Black
	}

	g.MoveCount++
	record := MoveRecord{
		Action:     action,
		X:          x,
		Y:          y,
		Z:          z,
		Player:     mover,
		MoveNumber: g.MoveCount,
	}
	g.History[g.MoveCount-1] = record

	// Check if this move completed any of the 76 winning lines
	if won, winMask, winIdx := CheckBitboardWin(activeBoard); won {
		g.WinLineMask = winMask
		g.WinLineIndex = winIdx
		g.Winner = mover
		if mover == PlayerWhite {
			g.Status = StatusWhiteWon
		} else {
			g.Status = StatusBlackWon
		}
		return record, nil
	}

	// Check if all 64 positions are now filled without a winner (Draw)
	if g.MoveCount == GridCells || (g.White|g.Black) == FullBoardMask {
		g.Status = StatusDraw
		g.Winner = PlayerNone
		return record, nil
	}

	// Switch turn to opponent
	g.CurrentPlayer = mover.Opponent()
	return record, nil
}

// MovesHistory returns a slice of all moves played so far in chronological order.
func (g *Game) MovesHistory() []MoveRecord {
	out := make([]MoveRecord, g.MoveCount)
	copy(out, g.History[:g.MoveCount])
	return out
}

// WriteCanonicalTensor populates a 192-byte slice (must have len >= 192) with the 3-channel
// canonical observation from the perspective of g.CurrentPlayer:
//   - Bytes 0..63   (Channel 0): 1 where g.CurrentPlayer has balls, 0 elsewhere
//   - Bytes 64..127 (Channel 1): 1 where g.CurrentPlayer.Opponent() has balls, 0 elsewhere
//   - Bytes 128..191 (Channel 2): 1 at the exact (x, y, z) gravity landing spot for each legal column move
func (g *Game) WriteCanonicalTensor(dst []byte) {
	for i := 0; i < CanonicalTensorSize; i++ {
		dst[i] = 0
	}

	myBoard := g.White
	oppBoard := g.Black
	if g.CurrentPlayer == PlayerBlack {
		myBoard = g.Black
		oppBoard = g.White
	}

	for bit := uint8(0); bit < GridCells; bit++ {
		mask := uint64(1) << bit
		if (myBoard & mask) != 0 {
			dst[bit] = 1
		}
		if (oppBoard & mask) != 0 {
			dst[64+bit] = 1
		}
	}

	if g.Status == StatusOngoing {
		occupied := g.White | g.Black
		for action := uint8(0); action < MaxActions; action++ {
			for z := uint8(0); z < GridDim; z++ {
				bit := action + 16*z
				if (occupied & (uint64(1) << bit)) == 0 {
					dst[128+bit] = 1
					break
				}
			}
		}
	}
}

// CanonicalTensor returns a new [192]byte array containing the 3-channel canonical observation.
func (g *Game) CanonicalTensor() [CanonicalTensorSize]byte {
	var tensor [CanonicalTensorSize]byte
	g.WriteCanonicalTensor(tensor[:])
	return tensor
}

// GenerateDrawSequence returns a deterministic sequence of 64 valid column actions (0..15)
// that fills the entire 4x4x4 grid without either White or Black ever forming a 4-in-a-row,
// terminating in StatusDraw on move 64.
func GenerateDrawSequence() [GridCells]uint8 {
	// Layer 0 (z=0) pattern where 1 = White, 0 = Black:
	//   y=0: 1 0 1 0
	//   y=1: 1 0 1 0
	//   y=2: 0 1 0 1
	//   y=3: 0 1 0 1
	// By assigning layers as:
	//   z=0: L0
	//   z=1: transpose(L0)
	//   z=2: NOT(L0)
	//   z=3: NOT(transpose(L0))
	// Every orthogonal line, planar diagonal, and 3D space diagonal has at most 3 of the same color!
	l0 := [4][4]Player{
		{PlayerWhite, PlayerBlack, PlayerWhite, PlayerBlack},
		{PlayerWhite, PlayerBlack, PlayerWhite, PlayerBlack},
		{PlayerBlack, PlayerWhite, PlayerBlack, PlayerWhite},
		{PlayerBlack, PlayerWhite, PlayerBlack, PlayerWhite},
	}

	var targetColor [4][4][4]Player // [z][y][x]
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			c0 := l0[y][x]
			c1 := l0[x][y] // transpose
			targetColor[0][y][x] = c0
			targetColor[1][y][x] = c1
			targetColor[2][y][x] = c0.Opponent()
			targetColor[3][y][x] = c1.Opponent()
		}
	}

	// Build the 64-move sequence layer-by-layer (z=0..3).
	// Within each layer z, there are exactly 8 White cells and 8 Black cells.
	// Alternating White and Black picks the next column in layer z matching the active player's color.
	var seq [GridCells]uint8
	moveIdx := 0
	for z := 0; z < 4; z++ {
		var whiteCols []uint8
		var blackCols []uint8
		for y := uint8(0); y < 4; y++ {
			for x := uint8(0); x < 4; x++ {
				action := 4*y + x
				if targetColor[z][y][x] == PlayerWhite {
					whiteCols = append(whiteCols, action)
				} else {
					blackCols = append(blackCols, action)
				}
			}
		}
		for i := 0; i < 8; i++ {
			seq[moveIdx] = whiteCols[i]
			moveIdx++
			seq[moveIdx] = blackCols[i]
			moveIdx++
		}
	}
	return seq
}
