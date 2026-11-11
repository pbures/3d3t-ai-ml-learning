package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/pbures/3d3t-ai-ml-learning/internal/grpcserver"
	"github.com/pbures/3d3t-ai-ml-learning/internal/httpserver"
)

func main() {
	httpAddr := flag.String("http-addr", "", "HTTP listen address for Kubernetes Ingress deployment (e.g. :8080)")
	httpsAddr := flag.String("https-addr", "127.0.0.1:8443", "HTTPS listen address for standalone TLS Battle API and 3D Web UI")
	grpcEndpoint := flag.String("grpc-endpoint", "unix:///tmp/3t3d_train.sock", "Local gRPC listen endpoint (empty to disable)")
	mcpStdio := flag.Bool("mcp-stdio", false, "Run Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio")
	flag.Parse()

	if *mcpStdio {
		arena := httpserver.NewArenaServer(nil)
		if err := arena.ServeMCPStdio(os.Stdin, os.Stdout); err != nil {
			log.Fatalf("MCP stdio error: %v", err)
		}
		return
	}

	if *grpcEndpoint != "" {
		_, resolvedGRPC, stopGRPC, err := grpcserver.StartServer(*grpcEndpoint)
		if err != nil {
			log.Fatalf("Failed to start gRPC server on %s: %v", *grpcEndpoint, err)
		}
		defer stopGRPC()
		fmt.Printf("3T3D Vectorized gRPC Server listening on %s\n", resolvedGRPC)
	}

	if *httpAddr != "" {
		_, baseHTTP, stopHTTP, err := httpserver.StartHTTPServer(*httpAddr)
		if err != nil {
			log.Fatalf("Failed to start HTTP Arena Server on %s: %v", *httpAddr, err)
		}
		defer stopHTTP()
		fmt.Printf("3T3D HTTP Battle API & 3D Web Replay UI running at %s/\n", baseHTTP)
	} else if *httpsAddr != "" {
		_, baseHTTPS, _, stopHTTPS, err := httpserver.StartTLSServer(*httpsAddr)
		if err != nil {
			log.Fatalf("Failed to start HTTPS Arena Server on %s: %v", *httpsAddr, err)
		}
		defer stopHTTPS()
		fmt.Printf("3T3D HTTPS Battle API & 3D Web Replay UI running at %s/\n", baseHTTPS)
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	fmt.Println("\nShutting down 3T3D Arena Server...")
}
