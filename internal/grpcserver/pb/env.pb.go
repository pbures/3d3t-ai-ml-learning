package pb

import (
	"encoding/json"
	"google.golang.org/grpc/encoding"
)

// CodecName is registered with gRPC so clients and servers serialize messages deterministically.
const CodecName = "proto"

func init() {
	encoding.RegisterCodec(JSONProtoCodec{})
}

// JSONProtoCodec implements grpc/encoding.Codec for our pb message types.
type JSONProtoCodec struct{}

func (JSONProtoCodec) Name() string {
	return CodecName
}

func (JSONProtoCodec) Marshal(v interface{}) ([]byte, error) {
	return json.Marshal(v)
}

func (JSONProtoCodec) Unmarshal(data []byte, v interface{}) error {
	return json.Unmarshal(data, v)
}

type Empty struct{}

type ResetRequest struct {
	GameId string `json:"game_id"`
}

func (r *ResetRequest) GetGameIdStr() string {
	if r == nil {
		return ""
	}
	return r.GameId
}

type GetStateRequest struct {
	GameId string `json:"game_id"`
}

func (r *GetStateRequest) GetGameIdStr() string {
	if r == nil {
		return ""
	}
	return r.GameId
}

type StepRequest struct {
	GameId string `json:"game_id"`
	Action uint32 `json:"action"`
}

type MoveCoord struct {
	Action     uint32 `json:"action"`
	X          uint32 `json:"x"`
	Y          uint32 `json:"y"`
	Z          uint32 `json:"z"`
	Player     uint32 `json:"player"`
	MoveNumber uint32 `json:"move_number"`
}

type GameState struct {
	GameId          string       `json:"game_id"`
	WhiteBitboard   uint64       `json:"white_bitboard"`
	BlackBitboard   uint64       `json:"black_bitboard"`
	CurrentPlayer   uint32       `json:"current_player"`
	Status          uint32       `json:"status"`
	Winner          uint32       `json:"winner"`
	WinLineMask     uint64       `json:"win_line_mask"`
	WinLineIndex    int32        `json:"win_line_index"`
	ValidMovesMask  uint32       `json:"valid_moves_mask"`
	MoveCount       uint32       `json:"move_count"`
	CanonicalTensor []byte       `json:"canonical_tensor"`
	History         []*MoveCoord `json:"history"`
}

type StepResponse struct {
	State    *GameState `json:"state"`
	LastMove *MoveCoord `json:"last_move"`
	Reward   float32    `json:"reward"`
	Done     bool       `json:"done"`
}

type WinLineInfo struct {
	Index    int32        `json:"index"`
	Category string       `json:"category"`
	Mask     uint64       `json:"mask"`
	Coords   []*MoveCoord `json:"coords"`
}

type WinLinesResponse struct {
	Lines []*WinLineInfo `json:"lines"`
}

type InitPoolRequest struct {
	NumEnvs uint32 `json:"num_envs"`
}

type ResetPoolRequest struct {
	EnvIds []uint32 `json:"env_ids"`
}

type BatchStateResponse struct {
	NumEnvs        uint32   `json:"num_envs"`
	Observations   []byte   `json:"observations"`
	ValidMasks     []uint32 `json:"valid_masks"`
	CurrentPlayers []uint32 `json:"current_players"`
}

type BatchStepRequest struct {
	Actions []byte `json:"actions"`
}

func (r *BatchStepRequest) GetActionsSlice() []byte {
	if r == nil {
		return nil
	}
	return r.Actions
}

type BatchStepResponse struct {
	NumEnvs              uint32   `json:"num_envs"`
	Observations         []byte   `json:"observations"`
	TerminalObservations []byte   `json:"terminal_observations"`
	Rewards              []float32 `json:"rewards"`
	Dones                []bool   `json:"dones"`
	Winners              []uint32 `json:"winners"`
	ValidMasks           []uint32 `json:"valid_masks"`
	CurrentPlayers       []uint32 `json:"current_players"`
	WinLineMasks         []uint64 `json:"win_line_masks"`
}
