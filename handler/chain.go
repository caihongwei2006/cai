package handler

import (
	"sync"
)

// ChainBuilder constructs a HandlersChain with middleware support.
type ChainBuilder struct {
	mu       sync.Mutex
	handlers HandlersChain
}

// NewChainBuilder creates a new builder.
func NewChainBuilder() *ChainBuilder {
	return &ChainBuilder{}
}

// Use appends handlers to the chain.
func (b *ChainBuilder) Use(handlers ...HandlerFunc) *ChainBuilder {
	b.mu.Lock()
	b.handlers = append(b.handlers, handlers...)
	b.mu.Unlock()
	return b
}

// Build returns the finalized handler chain.
func (b *ChainBuilder) Build() HandlersChain {
	b.mu.Lock()
	defer b.mu.Unlock()
	chain := make(HandlersChain, len(b.handlers))
	copy(chain, b.handlers)
	return chain
}

// StandardChain returns the default 5-handler chain indices.
// Developers replace each handler with their implementation.
//
// Index 0: IntentResolver   — Brain decomposes intent
// Index 1: ContextHydrator  — Framework builds CerebellumScope
// Index 2: ScriptGenerator  — Cerebellum generates script
// Index 3: SandboxExecutor  — Framework executes script
// Index 4: ResultRouter     — Routes SUCCESS/ERROR
const (
	IdxIntentResolver  int8 = 0
	IdxContextHydrator int8 = 1
	IdxScriptGenerator int8 = 2
	IdxSandboxExecutor int8 = 3
	IdxResultRouter    int8 = 4
)

// ContextPool pools Context objects for reuse across executions.
var ContextPool = sync.Pool{
	New: func() any {
		return &Context{
			index: -1,
		}
	},
}

// AcquireContext gets a Context from the pool.
func AcquireContext(handlers HandlersChain) *Context {
	c := ContextPool.Get().(*Context)
	c.handlers = handlers
	c.index = -1
	return c
}

// ReleaseContext returns a Context to the pool after resetting.
func ReleaseContext(c *Context) {
	c.Reset()
	c.handlers = nil
	ContextPool.Put(c)
}
