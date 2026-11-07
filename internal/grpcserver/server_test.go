package grpcserver

import (
	"context"
	"testing"

	"github.com/pbures/3d3t-ai-ml-learning/internal/grpcserver/pb"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestParseEndpointUnit(t *testing.T) {
	netType, addr, err := ParseEndpoint("unix:///tmp/test_3t3d.sock")
	if err != nil || netType != "unix" || addr != "/tmp/test_3t3d.sock" {
		t.Fatalf("unexpected unix parse result: net=%s addr=%s err=%v", netType, addr, err)
	}

	netType, addr, err = ParseEndpoint("tcp://127.0.0.1:50051")
	if err != nil || netType != "tcp" || addr != "127.0.0.1:50051" {
		t.Fatalf("unexpected tcp parse result: net=%s addr=%s err=%v", netType, addr, err)
	}

	if _, _, err := ParseEndpoint("unix://"); err == nil {
		t.Fatalf("expected error for empty unix path")
	}
	if _, _, err := ParseEndpoint("tcp://"); err == nil {
		t.Fatalf("expected error for empty tcp address")
	}
	if _, _, err := ParseEndpoint("http://localhost:8080"); err == nil {
		t.Fatalf("expected error for unsupported scheme")
	}

	if _, _, _, err := StartServer("invalid://endpoint"); err == nil {
		t.Fatalf("expected StartServer error on invalid endpoint")
	}
	if _, _, err := DialClient(context.Background(), "invalid://endpoint"); err == nil {
		t.Fatalf("expected DialClient error on invalid endpoint")
	}
}

func TestEnvServerUnitErrorPaths(t *testing.T) {
	ctx := context.Background()
	srv := NewEnvServer()

	// Nil ResetRequest should default to "default" game ID
	st, err := srv.Reset(ctx, nil)
	if err != nil || st.GameId != DefaultGameID {
		t.Fatalf("expected default game reset, got st=%v err=%v", st, err)
	}

	// Nil GetStateRequest should return "default" game
	st, err = srv.GetState(ctx, nil)
	if err != nil || st.GameId != DefaultGameID {
		t.Fatalf("expected default game state, got st=%v err=%v", st, err)
	}

	// Non-existent game ID on GetState and Step
	if _, err := srv.GetState(ctx, &pb.GetStateRequest{GameId: "missing"}); status.Code(err) != codes.NotFound {
		t.Fatalf("expected NotFound for missing game, got %v", err)
	}
	if _, err := srv.Step(ctx, &pb.StepRequest{GameId: "missing", Action: 0}); status.Code(err) != codes.NotFound {
		t.Fatalf("expected NotFound for Step on missing game, got %v", err)
	}
	if _, err := srv.Step(ctx, nil); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument for nil StepRequest, got %v", err)
	}

	// Pool precondition errors before InitPool
	if _, err := srv.ResetPool(ctx, nil); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("expected FailedPrecondition on ResetPool before InitPool, got %v", err)
	}
	if _, err := srv.BatchStep(ctx, nil); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("expected FailedPrecondition on BatchStep before InitPool, got %v", err)
	}

	// Invalid InitPool sizes
	if _, err := srv.InitPool(ctx, nil); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument for nil InitPoolRequest")
	}
	if _, err := srv.InitPool(ctx, &pb.InitPoolRequest{NumEnvs: 0}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument for NumEnvs=0")
	}
	if _, err := srv.InitPool(ctx, &pb.InitPoolRequest{NumEnvs: MaxPoolSize + 1}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument for NumEnvs > MaxPoolSize")
	}

	// Initialize pool of 4 envs and test invalid ResetPool and BatchStep inputs
	if _, err := srv.InitPool(ctx, &pb.InitPoolRequest{NumEnvs: 4}); err != nil {
		t.Fatalf("InitPool(4) failed: %v", err)
	}
	if _, err := srv.ResetPool(ctx, &pb.ResetPoolRequest{EnvIds: []uint32{99}}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument for out-of-range env_id in ResetPool, got %v", err)
	}
	if _, err := srv.BatchStep(ctx, nil); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument for nil BatchStepRequest, got %v", err)
	}
	if _, err := srv.BatchStep(ctx, &pb.BatchStepRequest{Actions: []byte{0, 1}}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument for wrong action count in BatchStep, got %v", err)
	}
	if _, err := srv.BatchStep(ctx, &pb.BatchStepRequest{Actions: []byte{0, 1, 2, 99}}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument for illegal action 99 in BatchStep, got %v", err)
	}

	// SelectAction error paths
	if _, err := srv.SelectAction(ctx, nil); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument on nil SelectActionRequest, got %v", err)
	}
	if _, err := srv.SelectAction(ctx, &pb.SelectActionRequest{GameId: "missing"}); status.Code(err) != codes.NotFound {
		t.Fatalf("expected NotFound on SelectAction for missing game, got %v", err)
	}
	if _, err := srv.SelectAction(ctx, &pb.SelectActionRequest{GameId: DefaultGameID, Agent: "bad-agent"}); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("expected InvalidArgument on SelectAction for unknown agent, got %v", err)
	}
}

func TestUnimplementedServerAndCodec(t *testing.T) {
	ctx := context.Background()
	var unimp pb.UnimplementedEnvServiceServer

	if _, err := unimp.Reset(ctx, &pb.ResetRequest{}); status.Code(err) != codes.Unimplemented {
		t.Fatalf("expected Unimplemented")
	}
	if _, err := unimp.Step(ctx, &pb.StepRequest{}); status.Code(err) != codes.Unimplemented {
		t.Fatalf("expected Unimplemented")
	}
	if _, err := unimp.GetState(ctx, &pb.GetStateRequest{}); status.Code(err) != codes.Unimplemented {
		t.Fatalf("expected Unimplemented")
	}
	if _, err := unimp.GetWinLines(ctx, &pb.Empty{}); status.Code(err) != codes.Unimplemented {
		t.Fatalf("expected Unimplemented")
	}
	if _, err := unimp.InitPool(ctx, &pb.InitPoolRequest{}); status.Code(err) != codes.Unimplemented {
		t.Fatalf("expected Unimplemented")
	}
	if _, err := unimp.ResetPool(ctx, &pb.ResetPoolRequest{}); status.Code(err) != codes.Unimplemented {
		t.Fatalf("expected Unimplemented")
	}
	if _, err := unimp.BatchStep(ctx, &pb.BatchStepRequest{}); status.Code(err) != codes.Unimplemented {
		t.Fatalf("expected Unimplemented")
	}
	if _, err := unimp.SelectAction(ctx, &pb.SelectActionRequest{}); status.Code(err) != codes.Unimplemented {
		t.Fatalf("expected Unimplemented")
	}

	codec := pb.JSONProtoCodec{}
	if codec.Name() != pb.CodecName {
		t.Fatalf("unexpected codec name %s", codec.Name())
	}
	raw, err := codec.Marshal(&pb.ResetRequest{GameId: "g1"})
	if err != nil {
		t.Fatal(err)
	}
	var decoded pb.ResetRequest
	if err := codec.Unmarshal(raw, &decoded); err != nil || decoded.GameId != "g1" {
		t.Fatalf("codec round-trip failed: %v", err)
	}
}
