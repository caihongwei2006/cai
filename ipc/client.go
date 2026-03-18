package ipc

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"sync/atomic"
)

// Client connects to a UDS server for JSON-RPC communication.
type Client struct {
	conn    net.Conn
	scanner *bufio.Scanner
	mu      sync.Mutex
	nextID  atomic.Int64

	// EventHandler is called for server-pushed events (optional).
	EventHandler func(Event)
}

// Dial connects to a UDS server.
func Dial(socketPath string) (*Client, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, fmt.Errorf("dial %s: %w", socketPath, err)
	}

	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	return &Client{conn: conn, scanner: scanner}, nil
}

// Call sends a request and waits for the response.
func (c *Client) Call(method string, params interface{}) (*Response, error) {
	id := fmt.Sprintf("req_%d", c.nextID.Add(1))

	var rawParams json.RawMessage
	if params != nil {
		data, err := json.Marshal(params)
		if err != nil {
			return nil, fmt.Errorf("marshal params: %w", err)
		}
		rawParams = data
	}

	req := Request{ID: id, Method: method, Params: rawParams}
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	data = append(data, '\n')

	c.mu.Lock()
	_, err = c.conn.Write(data)
	c.mu.Unlock()
	if err != nil {
		return nil, fmt.Errorf("write: %w", err)
	}

	// Read response (blocking)
	if !c.scanner.Scan() {
		if err := c.scanner.Err(); err != nil {
			return nil, fmt.Errorf("read: %w", err)
		}
		return nil, fmt.Errorf("connection closed")
	}

	var resp Response
	if err := json.Unmarshal(c.scanner.Bytes(), &resp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	return &resp, nil
}

// Close disconnects from the server.
func (c *Client) Close() error {
	return c.conn.Close()
}
