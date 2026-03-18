package engine

import (
	"fmt"
	"sync"

	cai "github.com/anthropic/cai"
)

// Registry maps EngineType → Engine implementation. Thread-safe.
type Registry struct {
	mu      sync.RWMutex
	engines map[cai.EngineType]cai.Engine
}

// NewRegistry creates an empty engine registry.
func NewRegistry() *Registry {
	return &Registry{
		engines: make(map[cai.EngineType]cai.Engine),
	}
}

// Register adds an engine to the registry.
func (r *Registry) Register(e cai.Engine) {
	r.mu.Lock()
	r.engines[e.Type()] = e
	r.mu.Unlock()
}

// Get retrieves an engine by type.
func (r *Registry) Get(t cai.EngineType) (cai.Engine, bool) {
	r.mu.RLock()
	e, ok := r.engines[t]
	r.mu.RUnlock()
	return e, ok
}

// MustGet retrieves an engine or panics.
func (r *Registry) MustGet(t cai.EngineType) cai.Engine {
	e, ok := r.Get(t)
	if !ok {
		panic(fmt.Sprintf("engine not registered: %s", t))
	}
	return e
}

// Close shuts down all registered engines.
func (r *Registry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	var lastErr error
	for _, e := range r.engines {
		if err := e.Close(); err != nil {
			lastErr = err
		}
	}
	return lastErr
}
