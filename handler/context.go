package handler

import (
	"context"
	"sync"

	cai "github.com/anthropic/cai"
)

const abortIndex int8 = 127

// HandlerFunc is a single step in the handler chain.
type HandlerFunc func(c *Context)

// HandlersChain is an ordered slice of handlers.
type HandlersChain []HandlerFunc

// Context is a Gin-inspired execution context for a single PlanNode.
// Supports Next/Abort/JumpTo/InsertBefore for hot-reload orchestration.
type Context struct {
	handlers HandlersChain
	index    int8
	mu       sync.Mutex // protects handler chain mutation

	// Strictly isolated scope pools
	Brain      *cai.BrainScope
	Cerebellum *cai.CerebellumScope

	// Current execution state
	CurrentIntent cai.Intent
	CurrentSpanID string
	CurrentEpoch  int
	LastEnvelope  *cai.ExecutionEnvelope
	Script        string // generated script to execute

	// Framework references (not scopes — plumbing)
	MemDB     cai.MemoryDB
	Engines   cai.EngineRegistry
	Collector cai.DataCollector

	// Go context for cancellation
	Ctx    context.Context
	Cancel context.CancelFunc

	// User-defined key-value store (per-request)
	keys map[string]any
}

// New creates a Context with the given handler chain.
func New(handlers HandlersChain) *Context {
	return &Context{
		handlers: handlers,
		index:    -1,
	}
}

// Next advances to the next handler in the chain.
func (c *Context) Next() {
	c.index++
	for c.index < int8(len(c.handlers)) {
		c.handlers[c.index](c)
		c.index++
	}
}

// Abort stops the chain. No further handlers execute.
func (c *Context) Abort() {
	c.index = abortIndex
}

// IsAborted returns true if Abort was called.
func (c *Context) IsAborted() bool {
	return c.index >= abortIndex
}

// JumpTo sets the chain index to execute from a specific handler.
// Used for hot-reload: retry (same index) or re-orchestration (earlier index).
func (c *Context) JumpTo(target int8) {
	c.mu.Lock()
	c.index = target - 1 // -1 because Next() will increment
	c.mu.Unlock()
}

// InsertBefore dynamically inserts a handler before the current index.
// Used by Brain to inject pre-requisite steps (e.g., dependency install).
func (c *Context) InsertBefore(h HandlerFunc) {
	c.mu.Lock()
	idx := c.index
	if idx < 0 {
		idx = 0
	}
	newChain := make(HandlersChain, 0, len(c.handlers)+1)
	newChain = append(newChain, c.handlers[:idx]...)
	newChain = append(newChain, h)
	newChain = append(newChain, c.handlers[idx:]...)
	c.handlers = newChain
	c.index-- // step back to execute the inserted handler on Next()
	c.mu.Unlock()
}

// HandlerCount returns the number of handlers in the chain.
func (c *Context) HandlerCount() int8 {
	return int8(len(c.handlers))
}

// CurrentIndex returns the current handler index.
func (c *Context) CurrentIndex() int8 {
	return c.index
}

// Reset resets the context for reuse (pool-friendly).
func (c *Context) Reset() {
	c.index = -1
	c.Brain = nil
	c.Cerebellum = nil
	c.LastEnvelope = nil
	c.Script = ""
	c.CurrentSpanID = ""
	c.CurrentEpoch = 0
	c.keys = nil
}

// Set stores a key-value pair in the context.
func (c *Context) Set(key string, value any) {
	if c.keys == nil {
		c.keys = make(map[string]any)
	}
	c.keys[key] = value
}

// Get retrieves a value from the context.
func (c *Context) Get(key string) (any, bool) {
	if c.keys == nil {
		return nil, false
	}
	v, ok := c.keys[key]
	return v, ok
}
