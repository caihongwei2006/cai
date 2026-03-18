package cai

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

// Option configures an Agent via functional options.
type Option func(*AgentConfig)

func WithPlanner(p Planner) Option             { return func(c *AgentConfig) { c.Planner = p } }
func WithExecutor(fn ExecutorFunc) Option      { return func(c *AgentConfig) { c.Executor = fn } }
func WithOptimizer(fn OptimizerAnyFunc) Option { return func(c *AgentConfig) { c.Optimizer = fn } }
func WithHITL(fn HITLResolverAnyFunc) Option   { return func(c *AgentConfig) { c.HITLResolver = fn } }
func WithDefaultSystemPrompt(prompt string) Option {
	return func(c *AgentConfig) { c.DefaultSystemPrompt = prompt }
}
func WithSystemPromptProvider(fn func(Intent) string) Option {
	return func(c *AgentConfig) { c.SystemPromptProvider = fn }
}

// WithInitialPrompts is deprecated compatibility plumbing.
// Active runtime execution no longer seeds or reads IntentMemory on first execution.
func WithInitialPrompts(m map[string]string) Option {
	return func(c *AgentConfig) { c.InitialPrompts = m }
}
func WithToolSchemas(t []ToolDef) Option { return func(c *AgentConfig) { c.ToolSchemas = t } }
func WithTools(tools ...Tool) Option {
	return func(c *AgentConfig) { c.Tools = append(c.Tools, tools...) }
}
func WithToolProviders(providers ...ToolProvider) Option {
	return func(c *AgentConfig) {
		for _, p := range providers {
			c.Tools = append(c.Tools, p.Tools()...)
		}
	}
}
func WithDocStore(ds WorkspaceDocStore) Option { return func(c *AgentConfig) { c.DocStore = ds } }
func WithModelConfig(mc *ModelConfig) Option   { return func(c *AgentConfig) { c.ModelConfig = mc } }
func WithOnTaskStart(fn func(spanID string, intent Intent)) Option {
	return func(c *AgentConfig) { c.OnTaskStart = fn }
}
func WithOnTaskComplete(fn func(spanID string, intent Intent, result string, err error)) Option {
	return func(c *AgentConfig) { c.OnTaskComplete = fn }
}
func WithOnIncident(fn func(event IncidentEvent)) Option {
	return func(c *AgentConfig) { c.OnIncident = fn }
}
func WithOnControllerState(fn func(traceID string, intent Intent, phase string)) Option {
	return func(c *AgentConfig) { c.OnControllerState = fn }
}
func WithOnLLMChunk(fn func(spanID string, intent Intent, phase string, text string)) Option {
	return func(c *AgentConfig) { c.OnLLMChunk = fn }
}
func WithOnLLMCall(fn func(spanID string, intent Intent, event LLMCallEvent)) Option {
	return func(c *AgentConfig) { c.OnLLMCall = fn }
}
func WithOnToolCall(fn func(spanID string, intent Intent, event ToolExecutionEvent)) Option {
	return func(c *AgentConfig) { c.OnToolCall = fn }
}
func WithPlannedChildRunner(fn func(ctx context.Context, req PlannedChildRequest) (string, error)) Option {
	return func(c *AgentConfig) { c.PlannedChildRunner = fn }
}
func WithWorkers(n int) Option             { return func(c *AgentConfig) { c.WorkerCount = n } }
func WithDBPath(path string) Option        { return func(c *AgentConfig) { c.dbPath = path } }
func WithCollector(d DataCollector) Option { return func(c *AgentConfig) { c.Collector = d } }
func WithStateStore(s StateStore) Option   { return func(c *AgentConfig) { c.StateStore = s } }
func WithMaxIterations(n int) Option       { return func(c *AgentConfig) { c.MaxSelfIterations = n } }
func WithMemDB(db MemoryDB) Option         { return func(c *AgentConfig) { c.MemDB = db } }

// WithPromptDir is deprecated compatibility plumbing.
// Active runtime execution no longer hydrates prompts from scaffold files.
func WithPromptDir(dir string) Option { return func(c *AgentConfig) { c.PromptDir = dir } }

// DBFactory creates a MemoryDB from a file path.
// Registered by memory package via init() to break circular import.
var DBFactory func(path string) (MemoryDB, error)

// PromptLoader is retained for backward-compatible tooling and exports.
// Active runtime execution no longer loads scaffold prompts on startup.
var PromptLoader func(dir string, db MemoryDB)

// PromptVersionWriter is retained for backward-compatible tooling and exports.
// Active runtime retry no longer writes optimized prompts back into IntentMemory.
var PromptVersionWriter func(dir string, action string, mem IntentMemory)

// SystemProber runs environment detection on a MemoryDB.
// Registered by memory package via init().
var SystemProber func(db MemoryDB) error

// WorkspaceDocLoader loads workspace .md files from a directory into DocStore.
// Registered by prompt package via init().
var WorkspaceDocLoader func(dir string, ds WorkspaceDocStore)

// WorkspaceDocWriter writes a versioned .md file after document evolution.
// Registered by prompt package via init().
var WorkspaceDocWriter func(dir string, name string, doc WorkspaceDocument)

// New creates an Agent with functional options.
// SQLite and system probe are managed automatically when memory package is imported.
//
//	import _ "github.com/anthropic/cai/memory" // register auto-DB
//
//	agent, err := cai.New(ctx,
//	    cai.WithPlanner(myPlanner),
//	    cai.WithExecutor(myExecutor),
//	)
func New(ctx context.Context, opts ...Option) (*Agent, error) {
	config := AgentConfig{
		WorkerCount:       4,
		MaxSelfIterations: 3,
		EvictionThreshold: 2,
	}
	for _, opt := range opts {
		opt(&config)
	}

	if config.Planner == nil {
		return nil, fmt.Errorf("cai.WithPlanner is required")
	}
	if config.Executor == nil && len(config.Tools) == 0 {
		return nil, fmt.Errorf("cai.WithExecutor or cai.WithTools is required")
	}

	if config.MemDB == nil {
		if DBFactory == nil {
			return nil, fmt.Errorf("MemDB not set and no DBFactory registered (import _ \"github.com/anthropic/cai/memory\")")
		}
		dbPath := config.dbPath
		if dbPath == "" {
			home, _ := os.UserHomeDir()
			dbPath = filepath.Join(home, ".cai", "agent.db")
		}
		db, err := DBFactory(dbPath)
		if err != nil {
			return nil, fmt.Errorf("auto-create db: %w", err)
		}
		if SystemProber != nil {
			SystemProber(db)
		}
		config.MemDB = db
		config.ownsDB = true
	}

	return newAgent(ctx, config)
}
