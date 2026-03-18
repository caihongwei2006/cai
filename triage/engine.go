package triage

import (
	"context"
	"fmt"

	cai "github.com/anthropic/cai"
)

// Config controls triage behavior.
type Config struct {
	MaxSelfIterations   int  // max retries before HITL (default: 3)
	EvictionThreshold   int  // consecutive failures to evict cache (default: 2)
	EnableBrainTakeover bool // if true, Brain generates script as last resort
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		MaxSelfIterations:   3,
		EvictionThreshold:   2,
		EnableBrainTakeover: false,
	}
}

// Engine is the independent Triage goroutine.
// It runs the same SOTA model config as Brain but with a separate lifecycle.
// Dual role: prompt optimization + HITL decision routing.
type Engine struct {
	config       Config
	memDB        cai.MemoryDB
	collector    cai.DataCollector
	errorChan    <-chan cai.ErrorPayload
	optimizeChan chan<- cai.OptimizePayload
	hitlChan     chan<- cai.HITLPayload

	// Developer-provided implementations
	optimizer    OptimizerFunc
	hitlResolver HITLResolverFunc
}

// OptimizerFunc is a function type matching Optimizer[T].Optimize for flexibility.
// Framework doesn't force generics on the triage goroutine.
type OptimizerFunc func(ctx context.Context, lastErr error, lastPrompt string, intent cai.Intent) (cai.OptimizationResult, error)

// HITLResolverFunc is a function type for HITL resolution.
type HITLResolverFunc func(ctx context.Context, payload cai.HITLPayload) (cai.Resolution, error)

// NewEngine constructs a Triage engine with all dependencies.
func NewEngine(
	config Config,
	memDB cai.MemoryDB,
	errorChan <-chan cai.ErrorPayload,
	optimizeChan chan<- cai.OptimizePayload,
	hitlChan chan<- cai.HITLPayload,
	optimizer OptimizerFunc,
	hitlResolver HITLResolverFunc,
	collector cai.DataCollector,
) *Engine {
	return &Engine{
		config:       config,
		memDB:        memDB,
		errorChan:    errorChan,
		optimizeChan: optimizeChan,
		hitlChan:     hitlChan,
		optimizer:    optimizer,
		hitlResolver: hitlResolver,
		collector:    collector,
	}
}

// Run starts the triage goroutine. Blocks until ctx is cancelled.
// This runs independently from the Brain main goroutine.
func (e *Engine) Run(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case errPayload, ok := <-e.errorChan:
			if !ok {
				return
			}
			e.handleError(ctx, errPayload)
		}
	}
}

func (e *Engine) handleError(ctx context.Context, ep cai.ErrorPayload) {
	action := ep.Intent.Action
	failCount := ep.EpochNum

	// Permission errors skip self-iteration entirely
	if ep.Envelope.Category == cai.ErrPermission {
		e.escalateToHITL(ctx, ep, "Permission denied — cannot self-iterate")
		return
	}

	// Within self-iteration budget?
	if failCount <= e.config.MaxSelfIterations && e.optimizer != nil {
		lastErr := fmt.Errorf("%s: %s", ep.Envelope.Category, ep.Envelope.RawStderr)
		optResult, optErr := e.optimizer(ctx, lastErr, ep.Prompt, ep.Intent)
		if optErr == nil && optResult.ShouldRetry {
			payload := cai.OptimizePayload{
				IntentAction:   action,
				NewSystemHints: optResult.CacheUpdate,
				ShouldRetry:    true,
				SpanID:         ep.SpanID,
			}
			select {
			case e.optimizeChan <- payload:
			case <-ctx.Done():
				return
			}
			return
		}
	}

	// Budget exhausted — escalate to HITL
	e.escalateToHITL(ctx, ep, fmt.Sprintf("Self-iteration exhausted after %d attempts", failCount))
}

func (e *Engine) escalateToHITL(ctx context.Context, ep cai.ErrorPayload, memo string) {
	payload := cai.HITLPayload{
		SpanID:     ep.SpanID,
		EpochNum:   ep.EpochNum,
		LastError:  ep.Envelope,
		TriageMemo: memo,
		Intent:     ep.Intent,
	}

	select {
	case e.hitlChan <- payload:
	case <-ctx.Done():
	}
}
