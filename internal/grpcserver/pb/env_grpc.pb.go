package pb

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// EnvServiceClient is the client API for EnvService service.
type EnvServiceClient interface {
	Reset(ctx context.Context, in *ResetRequest, opts ...grpc.CallOption) (*GameState, error)
	Step(ctx context.Context, in *StepRequest, opts ...grpc.CallOption) (*StepResponse, error)
	GetState(ctx context.Context, in *GetStateRequest, opts ...grpc.CallOption) (*GameState, error)
	GetWinLines(ctx context.Context, in *Empty, opts ...grpc.CallOption) (*WinLinesResponse, error)
	InitPool(ctx context.Context, in *InitPoolRequest, opts ...grpc.CallOption) (*BatchStateResponse, error)
	ResetPool(ctx context.Context, in *ResetPoolRequest, opts ...grpc.CallOption) (*BatchStateResponse, error)
	BatchStep(ctx context.Context, in *BatchStepRequest, opts ...grpc.CallOption) (*BatchStepResponse, error)
	SelectAction(ctx context.Context, in *SelectActionRequest, opts ...grpc.CallOption) (*SelectActionResponse, error)
}

type envServiceClient struct {
	cc grpc.ClientConnInterface
}

func NewEnvServiceClient(cc grpc.ClientConnInterface) EnvServiceClient {
	return &envServiceClient{cc}
}

func (c *envServiceClient) Reset(ctx context.Context, in *ResetRequest, opts ...grpc.CallOption) (*GameState, error) {
	out := new(GameState)
	err := c.cc.Invoke(ctx, "/env3t3d.v1.EnvService/Reset", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *envServiceClient) Step(ctx context.Context, in *StepRequest, opts ...grpc.CallOption) (*StepResponse, error) {
	out := new(StepResponse)
	err := c.cc.Invoke(ctx, "/env3t3d.v1.EnvService/Step", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *envServiceClient) GetState(ctx context.Context, in *GetStateRequest, opts ...grpc.CallOption) (*GameState, error) {
	out := new(GameState)
	err := c.cc.Invoke(ctx, "/env3t3d.v1.EnvService/GetState", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *envServiceClient) GetWinLines(ctx context.Context, in *Empty, opts ...grpc.CallOption) (*WinLinesResponse, error) {
	out := new(WinLinesResponse)
	err := c.cc.Invoke(ctx, "/env3t3d.v1.EnvService/GetWinLines", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *envServiceClient) InitPool(ctx context.Context, in *InitPoolRequest, opts ...grpc.CallOption) (*BatchStateResponse, error) {
	out := new(BatchStateResponse)
	err := c.cc.Invoke(ctx, "/env3t3d.v1.EnvService/InitPool", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *envServiceClient) ResetPool(ctx context.Context, in *ResetPoolRequest, opts ...grpc.CallOption) (*BatchStateResponse, error) {
	out := new(BatchStateResponse)
	err := c.cc.Invoke(ctx, "/env3t3d.v1.EnvService/ResetPool", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *envServiceClient) BatchStep(ctx context.Context, in *BatchStepRequest, opts ...grpc.CallOption) (*BatchStepResponse, error) {
	out := new(BatchStepResponse)
	err := c.cc.Invoke(ctx, "/env3t3d.v1.EnvService/BatchStep", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *envServiceClient) SelectAction(ctx context.Context, in *SelectActionRequest, opts ...grpc.CallOption) (*SelectActionResponse, error) {
	out := new(SelectActionResponse)
	err := c.cc.Invoke(ctx, "/env3t3d.v1.EnvService/SelectAction", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// EnvServiceServer is the server API for EnvService service.
type EnvServiceServer interface {
	Reset(context.Context, *ResetRequest) (*GameState, error)
	Step(context.Context, *StepRequest) (*StepResponse, error)
	GetState(context.Context, *GetStateRequest) (*GameState, error)
	GetWinLines(context.Context, *Empty) (*WinLinesResponse, error)
	InitPool(context.Context, *InitPoolRequest) (*BatchStateResponse, error)
	ResetPool(context.Context, *ResetPoolRequest) (*BatchStateResponse, error)
	BatchStep(context.Context, *BatchStepRequest) (*BatchStepResponse, error)
	SelectAction(context.Context, *SelectActionRequest) (*SelectActionResponse, error)
}

type UnimplementedEnvServiceServer struct{}

func (UnimplementedEnvServiceServer) Reset(context.Context, *ResetRequest) (*GameState, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Reset not implemented")
}
func (UnimplementedEnvServiceServer) Step(context.Context, *StepRequest) (*StepResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Step not implemented")
}
func (UnimplementedEnvServiceServer) GetState(context.Context, *GetStateRequest) (*GameState, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetState not implemented")
}
func (UnimplementedEnvServiceServer) GetWinLines(context.Context, *Empty) (*WinLinesResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method GetWinLines not implemented")
}
func (UnimplementedEnvServiceServer) InitPool(context.Context, *InitPoolRequest) (*BatchStateResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method InitPool not implemented")
}
func (UnimplementedEnvServiceServer) ResetPool(context.Context, *ResetPoolRequest) (*BatchStateResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method ResetPool not implemented")
}
func (UnimplementedEnvServiceServer) BatchStep(context.Context, *BatchStepRequest) (*BatchStepResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method BatchStep not implemented")
}
func (UnimplementedEnvServiceServer) SelectAction(context.Context, *SelectActionRequest) (*SelectActionResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SelectAction not implemented")
}

func RegisterEnvServiceServer(s grpc.ServiceRegistrar, srv EnvServiceServer) {
	s.RegisterService(&EnvService_ServiceDesc, srv)
}

func _EnvService_Reset_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ResetRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EnvServiceServer).Reset(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/env3t3d.v1.EnvService/Reset",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EnvServiceServer).Reset(ctx, req.(*ResetRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _EnvService_Step_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(StepRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EnvServiceServer).Step(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/env3t3d.v1.EnvService/Step",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EnvServiceServer).Step(ctx, req.(*StepRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _EnvService_GetState_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(GetStateRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EnvServiceServer).GetState(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/env3t3d.v1.EnvService/GetState",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EnvServiceServer).GetState(ctx, req.(*GetStateRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _EnvService_GetWinLines_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(Empty)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EnvServiceServer).GetWinLines(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/env3t3d.v1.EnvService/GetWinLines",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EnvServiceServer).GetWinLines(ctx, req.(*Empty))
	}
	return interceptor(ctx, in, info, handler)
}

func _EnvService_InitPool_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(InitPoolRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EnvServiceServer).InitPool(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/env3t3d.v1.EnvService/InitPool",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EnvServiceServer).InitPool(ctx, req.(*InitPoolRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _EnvService_ResetPool_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ResetPoolRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EnvServiceServer).ResetPool(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/env3t3d.v1.EnvService/ResetPool",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EnvServiceServer).ResetPool(ctx, req.(*ResetPoolRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _EnvService_BatchStep_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(BatchStepRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EnvServiceServer).BatchStep(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/env3t3d.v1.EnvService/BatchStep",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EnvServiceServer).BatchStep(ctx, req.(*BatchStepRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _EnvService_SelectAction_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(SelectActionRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(EnvServiceServer).SelectAction(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/env3t3d.v1.EnvService/SelectAction",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(EnvServiceServer).SelectAction(ctx, req.(*SelectActionRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var EnvService_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "env3t3d.v1.EnvService",
	HandlerType: (*EnvServiceServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Reset",
			Handler:    _EnvService_Reset_Handler,
		},
		{
			MethodName: "Step",
			Handler:    _EnvService_Step_Handler,
		},
		{
			MethodName: "GetState",
			Handler:    _EnvService_GetState_Handler,
		},
		{
			MethodName: "GetWinLines",
			Handler:    _EnvService_GetWinLines_Handler,
		},
		{
			MethodName: "InitPool",
			Handler:    _EnvService_InitPool_Handler,
		},
		{
			MethodName: "ResetPool",
			Handler:    _EnvService_ResetPool_Handler,
		},
		{
			MethodName: "BatchStep",
			Handler:    _EnvService_BatchStep_Handler,
		},
		{
			MethodName: "SelectAction",
			Handler:    _EnvService_SelectAction_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "proto/env.proto",
}
