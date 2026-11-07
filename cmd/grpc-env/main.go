package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/pbures/3d3t-ai-ml-learning/internal/grpcserver"
)

func main() {
	endpoint := flag.String("endpoint", "unix:///tmp/3t3d_train.sock", "gRPC listen endpoint (unix:///path or tcp://host:port)")
	flag.Parse()

	_, resolvedTarget, cleanup, err := grpcserver.StartServer(*endpoint)
	if err != nil {
		log.Fatalf("Failed to start 3T3D gRPC environment server: %v", err)
	}
	defer cleanup()

	fmt.Printf("3T3D Vectorized gRPC Environment Server listening on %s\n", resolvedTarget)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	fmt.Println("\nShutting down 3T3D gRPC server...")
}
