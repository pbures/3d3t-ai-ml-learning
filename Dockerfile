# Stage 1: Build static Go binary
FROM golang:1.21-alpine AS builder

WORKDIR /workspace
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -ldflags="-s -w" -o /out/arena-server ./cmd/arena-server

# Stage 2: Minimal runtime image
FROM alpine:3.19

RUN apk --no-cache add ca-certificates && adduser -D -u 10001 appuser
USER appuser

COPY --from=builder /out/arena-server /usr/local/bin/arena-server

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/arena-server"]
CMD ["-http-addr", ":8080", "-grpc-endpoint", "unix:///tmp/3t3d_train.sock"]
