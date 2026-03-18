package ipc

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"sync"
	"time"
)

// Request is a JSON-RPC style request over UDS.
type Request struct {
	ID     string          `json:"id"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params,omitempty"`
}

// Response is a JSON-RPC style response.
type Response struct {
	ID     string      `json:"id"`
	Result interface{} `json:"result,omitempty"`
	Error  *RPCError   `json:"error,omitempty"`
}

// RPCError is a structured error.
type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// Event is a push notification from server to clients.
type Event struct {
	Type    string      `json:"type"`
	Payload interface{} `json:"payload"`
	Seq     int64       `json:"seq"`
	TS      int64       `json:"ts"`
}

// MethodHandler processes a request and returns a result.
type MethodHandler func(ctx context.Context, params json.RawMessage) (interface{}, error)

// Server is a Unix Domain Socket server with JSON-RPC protocol.
type Server struct {
	socketPath string
	listener   net.Listener
	handlers   map[string]MethodHandler
	clients    map[net.Conn]struct{}
	mu         sync.RWMutex
	seq        int64
}

// NewServer creates a UDS server at the given socket path.
func NewServer(socketPath string) *Server {
	return &Server{
		socketPath: socketPath,
		handlers:   make(map[string]MethodHandler),
		clients:    make(map[net.Conn]struct{}),
	}
}

// Handle registers a method handler.
func (s *Server) Handle(method string, h MethodHandler) {
	s.mu.Lock()
	s.handlers[method] = h
	s.mu.Unlock()
}

// Start begins accepting connections. Non-blocking — runs in background goroutines.
func (s *Server) Start(ctx context.Context) error {
	_ = os.Remove(s.socketPath)

	ln, err := net.Listen("unix", s.socketPath)
	if err != nil {
		return fmt.Errorf("listen %s: %w", s.socketPath, err)
	}
	s.listener = ln

	go func() {
		<-ctx.Done()
		ln.Close()
	}()

	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				select {
				case <-ctx.Done():
					return
				default:
					log.Printf("[ipc] accept error: %v", err)
					continue
				}
			}
			s.mu.Lock()
			s.clients[conn] = struct{}{}
			clientCount := len(s.clients)
			s.mu.Unlock()

			log.Printf("[ipc] client connected (clients=%d)", clientCount)
			go s.serveConn(ctx, conn)
		}
	}()

	return nil
}

func (s *Server) serveConn(ctx context.Context, conn net.Conn) {
	defer func() {
		conn.Close()
		s.mu.Lock()
		delete(s.clients, conn)
		clientCount := len(s.clients)
		s.mu.Unlock()
		log.Printf("[ipc] client disconnected (clients=%d)", clientCount)
	}()

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return
		default:
		}

		line := scanner.Bytes()
		var req Request
		if err := json.Unmarshal(line, &req); err != nil {
			s.writeResponse(conn, Response{Error: &RPCError{Code: -32700, Message: "parse error"}})
			continue
		}

		s.mu.RLock()
		handler, ok := s.handlers[req.Method]
		s.mu.RUnlock()

		if !ok {
			log.Printf("[ipc] method not found: %s (id=%s)", req.Method, req.ID)
			s.writeResponse(conn, Response{ID: req.ID, Error: &RPCError{Code: -32601, Message: "method not found: " + req.Method}})
			continue
		}

		log.Printf("[ipc] dispatch %s (id=%s)", req.Method, req.ID)
		callStart := time.Now()
		result, err := handler(ctx, req.Params)
		callDur := time.Since(callStart).Milliseconds()
		if err != nil {
			log.Printf("[ipc] dispatch %s error (id=%s, dur=%dms): %v", req.Method, req.ID, callDur, err)
			s.writeResponse(conn, Response{ID: req.ID, Error: &RPCError{Code: -32000, Message: err.Error()}})
		} else {
			log.Printf("[ipc] dispatch %s ok (id=%s, dur=%dms)", req.Method, req.ID, callDur)
			s.writeResponse(conn, Response{ID: req.ID, Result: result})
		}
	}
}

func (s *Server) writeResponse(conn net.Conn, resp Response) {
	data, err := json.Marshal(resp)
	if err != nil {
		return
	}
	data = append(data, '\n')
	conn.Write(data)
}

// Broadcast sends an event to all connected clients.
func (s *Server) Broadcast(evt Event) {
	s.mu.Lock()
	s.seq++
	evt.Seq = s.seq
	s.mu.Unlock()

	data, err := json.Marshal(evt)
	if err != nil {
		return
	}
	data = append(data, '\n')

	s.mu.RLock()
	defer s.mu.RUnlock()

	for conn := range s.clients {
		conn.Write(data)
	}
}

// ClientCount returns the number of active connections.
func (s *Server) ClientCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.clients)
}

// Close shuts down the server and cleans up the socket file.
func (s *Server) Close() error {
	if s.listener != nil {
		s.listener.Close()
	}
	s.mu.Lock()
	for conn := range s.clients {
		conn.Close()
	}
	s.clients = make(map[net.Conn]struct{})
	s.mu.Unlock()

	return os.Remove(s.socketPath)
}
