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
	httpsAddr := flag.String("https-addr", "127.0.0.1:8443", "HTTPS listen address for Battle API and 3D Web Replay UI")
	grpcEndpoint := flag.String("grpc-endpoint", "unix:///tmp/3t3d_train.sock", "Local gRPC listen endpoint (empty to disable)")
	flag.Parse()

	if *grpcEndpoint != "" {
		_, resolvedGRPC, stopGRPC, err := grpcserver.StartServer(*grpcEndpoint)
		if err != nil {
			log.Fatalf("Failed to start gRPC server on %s: %v", *grpcEndpoint, err)
		}
		defer stopGRPC()
		fmt.Printf("3T3D Vectorized gRPC Server listening on %s\n", resolvedGRPC)
	}

	_, baseURL, _, stopHTTPS, err := httpserver.StartTLSServer(*httpsAddr)
	if err != nil {
		log.Fatalf("Failed to start HTTPS Arena Server on %s: %v", *httpsAddr, err)
	}
	defer stopHTTPS()

	fmt.Printf("3T3D HTTPS Battle API & 3D Web Replay UI running at %s/\n", baseURL)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	fmt.Println("\nShutting down 3T3D Arena Server...")
}
