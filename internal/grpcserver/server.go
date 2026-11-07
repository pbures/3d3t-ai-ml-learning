package grpcserver

import (
	"context"
	"fmt"
	"net"
	"os"
	"strings"
	"sync"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
	"github.com/pbures/3d3t-ai-ml-learning/internal/grpcserver/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

const (
	DefaultGameID   = "default"
	MaxPoolSize     = 16384
	DefaultPoolSize = 64
)

// EnvServer implements pb.EnvServiceServer for both single-game rule inspection
// and high-speed vectorized RL environment stepping.
type EnvServer struct {
	pb.UnimplementedEnvServiceServer

	mu    sync.RWMutex
	games map[string]*engine.Game
	pool  []*engine.Game
}

// NewEnvServer creates a new EnvServer with an empty game registry and uninitialized pool.
func NewEnvServer() *EnvServer {
	s := &EnvServer{
		games: make(map[string]*engine.Game),
	}
	s.games[DefaultGameID] = engine.NewGame()
	return s
}

func resolveGameID(id string) string {
	if strings.TrimSpace(id) == "" {
		return DefaultGameID
	}
	return id
}

func toProtoState(gameID string, g *engine.Game) *pb.GameState {
	tensor := g.CanonicalTensor()
	history := make([]*pb.MoveCoord, g.MoveCount)
	for i := uint8(0); i < g.MoveCount; i++ {
		rec := g.History[i]
		history[i] = &pb.MoveCoord{
			Action:     uint32(rec.Action),
			X:          uint32(rec.X),
			Y:          uint32(rec.Y),
			Z:          uint32(rec.Z),
			Player:     uint32(rec.Player),
			MoveNumber: uint32(rec.MoveNumber),
		}
	}

	return &pb.GameState{
		GameId:          gameID,
		WhiteBitboard:   g.White,
		BlackBitboard:   g.Black,
		CurrentPlayer:   uint32(g.CurrentPlayer),
		Status:          uint32(g.Status),
		Winner:          uint32(g.Winner),
		WinLineMask:     g.WinLineMask,
		WinLineIndex:    int32(g.WinLineIndex),
		ValidMovesMask:  uint32(g.ValidMovesMask()),
		MoveCount:       uint32(g.MoveCount),
		CanonicalTensor: tensor[:],
		History:         history,
	}
}

// Reset creates or resets a single game session identified by req.GameId.
func (s *EnvServer) Reset(_ context.Context, req *pb.ResetRequest) (*pb.GameState, error) {
	gameID := resolveGameID(req.GetGameIdStr())
	s.mu.Lock()
	defer s.mu.Unlock()

	g, exists := s.games[gameID]
	if !exists {
		g = engine.NewGame()
		s.games[gameID] = g
	} else {
		g.Reset()
	}
	return toProtoState(gameID, g), nil
}

// GetState retrieves the current state of a game session identified by req.GameId.
func (s *EnvServer) GetState(_ context.Context, req *pb.GetStateRequest) (*pb.GameState, error) {
	gameID := resolveGameID(req.GetGameIdStr())
	s.mu.RLock()
	defer s.mu.RUnlock()

	g, exists := s.games[gameID]
	if !exists {
		return nil, status.Errorf(codes.NotFound, "game %q not found", gameID)
	}
	return toProtoState(gameID, g), nil
}

// Step validates and applies a single move (action in 0..15) to the specified game.
func (s *EnvServer) Step(_ context.Context, req *pb.StepRequest) (*pb.StepResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "nil StepRequest")
	}
	if req.Action >= engine.MaxActions {
		return nil, status.Errorf(codes.InvalidArgument, "%v: got %d", engine.ErrInvalidAction, req.Action)
	}

	gameID := resolveGameID(req.GameId)
	s.mu.Lock()
	defer s.mu.Unlock()

	g, exists := s.games[gameID]
	if !exists {
		return nil, status.Errorf(codes.NotFound, "game %q not found", gameID)
	}

	rec, err := g.ApplyMove(uint8(req.Action))
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "%v", err)
	}

	var reward float32
	done := g.Status != engine.StatusOngoing
	if g.Status == engine.StatusWhiteWon || g.Status == engine.StatusBlackWon {
		reward = 1.0
	}

	return &pb.StepResponse{
		State: toProtoState(gameID, g),
		LastMove: &pb.MoveCoord{
			Action:     uint32(rec.Action),
			X:          uint32(rec.X),
			Y:          uint32(rec.Y),
			Z:          uint32(rec.Z),
			Player:     uint32(rec.Player),
			MoveNumber: uint32(rec.MoveNumber),
		},
		Reward: reward,
		Done:   done,
	}, nil
}

// GetWinLines returns all 76 precomputed winning lines and their 3D coordinates.
func (s *EnvServer) GetWinLines(_ context.Context, _ *pb.Empty) (*pb.WinLinesResponse, error) {
	lines := make([]*pb.WinLineInfo, engine.TotalWinLines)
	for i, wl := range engine.WinLines {
		coords := make([]*pb.MoveCoord, 4)
		for k, c := range wl.Coords {
			coords[k] = &pb.MoveCoord{
				Action: uint32(4*c.Y + c.X),
				X:      uint32(c.X),
				Y:      uint32(c.Y),
				Z:      uint32(c.Z),
			}
		}
		lines[i] = &pb.WinLineInfo{
			Index:    int32(wl.Index),
			Category: string(wl.Category),
			Mask:     wl.Mask,
			Coords:   coords,
		}
	}
	return &pb.WinLinesResponse{Lines: lines}, nil
}

func (s *EnvServer) buildBatchStateLocked() *pb.BatchStateResponse {
	n := len(s.pool)
	obs := make([]byte, n*engine.CanonicalTensorSize)
	validMasks := make([]uint32, n)
	players := make([]uint32, n)

	for i, g := range s.pool {
		g.WriteCanonicalTensor(obs[i*engine.CanonicalTensorSize : (i+1)*engine.CanonicalTensorSize])
		validMasks[i] = uint32(g.ValidMovesMask())
		players[i] = uint32(g.CurrentPlayer)
	}

	return &pb.BatchStateResponse{
		NumEnvs:        uint32(n),
		Observations:   obs,
		ValidMasks:     validMasks,
		CurrentPlayers: players,
	}
}

// InitPool allocates N parallel 3T3D game environments for vectorized stepping.
func (s *EnvServer) InitPool(_ context.Context, req *pb.InitPoolRequest) (*pb.BatchStateResponse, error) {
	if req == nil || req.NumEnvs == 0 || req.NumEnvs > MaxPoolSize {
		return nil, status.Errorf(codes.InvalidArgument, "num_envs must be in 1..%d", MaxPoolSize)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.pool = make([]*engine.Game, req.NumEnvs)
	for i := range s.pool {
		s.pool[i] = engine.NewGame()
	}
	return s.buildBatchStateLocked(), nil
}

// ResetPool resets specified environment indices (or all environments if EnvIds is empty).
func (s *EnvServer) ResetPool(_ context.Context, req *pb.ResetPoolRequest) (*pb.BatchStateResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.pool) == 0 {
		return nil, status.Error(codes.FailedPrecondition, "vectorized pool not initialized; call InitPool first")
	}

	if req == nil || len(req.EnvIds) == 0 {
		for _, g := range s.pool {
			g.Reset()
		}
	} else {
		for _, idx := range req.EnvIds {
			if int(idx) >= len(s.pool) {
				return nil, status.Errorf(codes.InvalidArgument, "env_id %d out of range (pool size %d)", idx, len(s.pool))
			}
			s.pool[idx].Reset()
		}
	}
	return s.buildBatchStateLocked(), nil
}

// BatchStep executes one action per environment in the pool simultaneously.
// If an environment reaches a terminal state (Win or Draw), its terminal observation,
// reward, done=true,winner, and win_line_mask are recorded, and the environment is
// automatically reset so its returned `observations` slot holds the initial state of the next episode.
func (s *EnvServer) BatchStep(_ context.Context, req *pb.BatchStepRequest) (*pb.BatchStepResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	n := len(s.pool)
	if n == 0 {
		return nil, status.Error(codes.FailedPrecondition, "vectorized pool not initialized; call InitPool first")
	}
	if req == nil || len(req.Actions) != n {
		return nil, status.Errorf(codes.InvalidArgument, "expected %d actions, got %d", n, len(req.GetActionsSlice()))
	}

	// Validate all actions up front so a batch never partially applies on error
	for i, a := range req.Actions {
		if !s.pool[i].IsActionValid(a) {
			return nil, status.Errorf(codes.InvalidArgument, "env %d: invalid or full column action %d (valid_mask=0x%04x)",
				i, a, s.pool[i].ValidMovesMask())
		}
	}

	obs := make([]byte, n*engine.CanonicalTensorSize)
	termObs := make([]byte, n*engine.CanonicalTensorSize)
	rewards := make([]float32, n)
	dones := make([]bool, n)
	winners := make([]uint32, n)
	validMasks := make([]uint32, n)
	players := make([]uint32, n)
	winMasks := make([]uint64, n)

	for i, a := range req.Actions {
		g := s.pool[i]
		_, _ = g.ApplyMove(a)

		sliceStart := i * engine.CanonicalTensorSize
		sliceEnd := (i + 1) * engine.CanonicalTensorSize

		if g.Status != engine.StatusOngoing {
			dones[i] = true
			winners[i] = uint32(g.Winner)
			winMasks[i] = g.WinLineMask
			if g.Winner != engine.PlayerNone {
				rewards[i] = 1.0
			}
			// Capture terminal observation before auto-reset
			g.WriteCanonicalTensor(termObs[sliceStart:sliceEnd])
			// Auto-reset environment for the next episode
			g.Reset()
		}

		g.WriteCanonicalTensor(obs[sliceStart:sliceEnd])
		validMasks[i] = uint32(g.ValidMovesMask())
		players[i] = uint32(g.CurrentPlayer)
	}

	return &pb.BatchStepResponse{
		NumEnvs:              uint32(n),
		Observations:         obs,
		TerminalObservations: termObs,
		Rewards:              rewards,
		Dones:                dones,
		Winners:              winners,
		ValidMasks:           validMasks,
		CurrentPlayers:       players,
		WinLineMasks:         winMasks,
	}, nil
}

// ParseEndpoint splits an endpoint string such as "unix:///tmp/3t3d.sock" or "tcp://127.0.0.1:50051"
// into (network, address) suitable for net.Listen.
func ParseEndpoint(endpoint string) (network, addr string, err error) {
	switch {
	case strings.HasPrefix(endpoint, "unix://"):
		addr = strings.TrimPrefix(endpoint, "unix://")
		if addr == "" {
			return "", "", fmt.Errorf("empty unix socket path in %q", endpoint)
		}
		return "unix", addr, nil
	case strings.HasPrefix(endpoint, "tcp://"):
		addr = strings.TrimPrefix(endpoint, "tcp://")
		if addr == "" {
			return "", "", fmt.Errorf("empty tcp address in %q", endpoint)
		}
		return "tcp", addr, nil
	default:
		return "", "", fmt.Errorf("unsupported endpoint scheme %q (use unix:// or tcp://)", endpoint)
	}
}

// StartServer starts a gRPC server listening on the given endpoint ("unix:///path" or "tcp://host:port")
// and returns the grpc.Server, resolved dial target, and a cleanup stop function.
func StartServer(endpoint string) (*grpc.Server, string, func(), error) {
	network, addr, err := ParseEndpoint(endpoint)
	if err != nil {
		return nil, "", nil, err
	}
	if network == "unix" {
		_ = os.Remove(addr)
	}

	lis, err := net.Listen(network, addr)
	if err != nil {
		return nil, "", nil, err
	}

	grpcSrv := grpc.NewServer()
	pb.RegisterEnvServiceServer(grpcSrv, NewEnvServer())

	go func() {
		_ = grpcSrv.Serve(lis)
	}()

	actualTarget := endpoint
	if network == "tcp" {
		actualTarget = "tcp://" + lis.Addr().String()
	}

	cleanup := func() {
		grpcSrv.Stop()
		_ = lis.Close()
		if network == "unix" {
			_ = os.Remove(addr)
		}
	}

	return grpcSrv, actualTarget, cleanup, nil
}

// DialClient connects a gRPC EnvServiceClient to an endpoint ("unix:///path" or "tcp://host:port").
func DialClient(ctx context.Context, endpoint string) (pb.EnvServiceClient, *grpc.ClientConn, error) {
	network, addr, err := ParseEndpoint(endpoint)
	if err != nil {
		return nil, nil, err
	}

	dialTarget := addr
	if network == "unix" {
		dialTarget = "unix://" + addr
	}

	conn, err := grpc.DialContext(
		ctx,
		dialTarget,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.ForceCodec(pb.JSONProtoCodec{})),
	)
	if err != nil {
		return nil, nil, err
	}
	return pb.NewEnvServiceClient(conn), conn, nil
}
