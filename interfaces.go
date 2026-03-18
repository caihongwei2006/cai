package cai

import (
	"context"
	"encoding/json"
)

// --- Core Framework Interfaces (Developer Implements) ---

// Planner decomposes a user objective into ordered intents.
type Planner interface {
	Plan(ctx context.Context, objective string, scope *BrainScope) ([]Intent, error)
}

// Executor is the "hands" of the small model. Framework doesn't care
// if it shells out, calls an API, or runs WASM.
type Executor[T any] interface {
	Execute(ctx context.Context, prompt string, meta T) (result string, err error)
}

// Optimizer is the "Prompt Compiler". Called by Triage on execution failure.
// Developer can use SOTA LLM, regex rules, or anything.
type Optimizer[T any] interface {
	Optimize(ctx context.Context, lastErr error, lastPrompt string, meta T) (OptimizationResult, error)
}

// OptimizationResult carries the outcome of prompt optimization.
type OptimizationResult struct {
	ShouldRetry bool
	NewPrompt   string
	CacheUpdate string // if non-empty, framework compiles it into retry-time experience context
}

// HITLResolver is the "human gate". Framework suspends the goroutine
// until Resolve() returns. Framework does not care who is on the other end.
type HITLResolver[T any] interface {
	Resolve(ctx context.Context, payload HITLPayload, meta T) (Resolution, error)
}

// --- Storage Interfaces ---

// MemoryDB abstracts all local persistence.
type MemoryDB interface {
	// IntentMemory (legacy compatibility storage; active runtime should avoid these)
	LoadIntentMemory(action string) (*IntentMemory, bool)
	SaveIntentMemory(mem IntentMemory) error
	EvictIntentMemory(action string, reason string) error
	IncrementFailure(action string) (int, error)
	ResetFailures(action string) error

	// SystemEvolution
	LoadSystemEvolution() SystemEvolutionMemory
	UpdateSystemEvolution(key, value string) error

	// Capability / Experience
	SaveCapabilityProfile(profile CapabilityProfile) error
	LoadCapabilityProfile(id CapabilityID) (*CapabilityProfile, bool)
	AppendExperienceEvent(event ExperienceEvent) error
	ListExperienceEvents(promptIdentity string, capabilityID CapabilityID, visibleOnly bool, limit int) ([]ExperienceEvent, error)
	SaveCompiledExperienceView(view CompiledExperienceView) error
	LoadCompiledExperienceView(promptIdentity string) (*CompiledExperienceView, bool)
	SaveRunPatch(patch RunPatch) error
	ListRunPatches(runID string, promptIdentity string, capabilityID CapabilityID, limit int) ([]RunPatch, error)
	ClearRunPatches(runID string) error

	// Epoch Log
	CreateTrace(trace AgentTrace) error
	UpdateTraceStatus(traceID, status string) error
	CreateSpan(span LogicalSpan) error
	UpdateSpanStatus(spanID, status string, finalEpoch int) error
	AppendEpoch(epoch ExecutionEpoch) error
	ActiveEpoch(spanID string) (*ExecutionEpoch, error)
	SpanHistory(spanID string) ([]ExecutionEpoch, error)
	SpanSummary(spanID string, depth int) ([]EpochSummary, error)

	// Lifecycle
	Close() error
}

// StateStore handles process-lifecycle persistence.
type StateStore interface {
	Hibernate(ctx context.Context, traceID string, state HibernationState) error
	Wake(ctx context.Context, traceID string) (*HibernationState, error)
	ListPending(ctx context.Context) ([]HibernationState, error)
	Clear(ctx context.Context, traceID string) error
}

// HibernationState survives process restarts.
type HibernationState struct {
	TraceID             string           `json:"trace_id"`
	RequestID           string           `json:"request_id"`
	PlanDAG             []PlanNode       `json:"plan_dag,omitempty"`
	CurrentIndex        int8             `json:"current_index,omitempty"`
	PendingSpanID       string           `json:"pending_span_id,omitempty"`
	PendingEpoch        int              `json:"pending_epoch,omitempty"`
	BrainScope          BrainScope       `json:"brain_scope,omitempty"`
	VisibleSteps        []VisibleStep    `json:"visible_steps,omitempty"`
	ChildRuns           []ChildRunState  `json:"child_runs,omitempty"`
	ActiveIncidents     []ActiveIncident `json:"active_incidents,omitempty"`
	PendingApproval     *PendingApproval `json:"pending_approval,omitempty"`
	CurrentSpans        []SpanCheckpoint `json:"current_spans,omitempty"`
	ControllerState     string           `json:"controller_state,omitempty"`
	ActivePlanRev       int              `json:"active_plan_rev,omitempty"`
	QuiescingStartedAt  int64            `json:"quiescing_started_at,omitempty"`
	LastSummaryAt       int64            `json:"last_summary_at,omitempty"`
	RevisionReason      string           `json:"revision_reason,omitempty"`
	LatestObservationAt int64            `json:"latest_observation_at"`
	CreatedAt           int64            `json:"created_at"`
}

// HistoryRetriever lets Brain expand beyond N(1) on demand.
type HistoryRetriever interface {
	RetrieveSpanHistory(ctx context.Context, spanID string, depth int) ([]EpochSummary, error)
}

// --- Tool Interface (Agent-Level Executable Tools) ---

// Tool is the unified contract for agent-callable tools.
// Implementations capture their own dependencies via factory closures.
//
//	bashTool := tools.NewBashTool(engine)   // closure captures engine
//	agent, _ := cai.New(ctx, cai.WithTools(bashTool, readTool, cronTool))
type Tool interface {
	Name() string
	Description() string
	InputSchema() json.RawMessage
	Execute(ctx context.Context, input json.RawMessage) (ToolResult, error)
}

// ToolResult is the structured output from a tool execution.
type ToolResult struct {
	Content string `json:"content"`
	IsError bool   `json:"is_error,omitempty"`
}

// ToolProvider groups related tools from a single source (MCP server, plugin, etc.).
type ToolProvider interface {
	Name() string
	Tools() []Tool
}

// --- Workspace Document Interface ---

// WorkspaceDocStore handles .md workspace documents with versioned iteration.
// Lifecycle: .md on disk -> SeedDocument -> SQLite -> EvolveDocument -> WriteBack .md_vN
type WorkspaceDocStore interface {
	LoadDocument(name string) (*WorkspaceDocument, bool)
	SaveDocument(doc WorkspaceDocument) error
	ListDocuments() ([]WorkspaceDocument, error)
	EvolveDocument(name string, newContent string, reason string) error
	DeleteDocument(name string) error
	Close() error
}

// --- Engine Interface ---

// Engine executes scripts in a specific runtime environment.
type Engine interface {
	Type() EngineType
	Execute(ctx context.Context, script string, env map[string]string) (*ExecutionEnvelope, error)
	Close() error
}

// EngineRegistry maps EngineType → Engine implementation.
type EngineRegistry interface {
	Register(e Engine)
	Get(t EngineType) (Engine, bool)
	Close() error
}

// --- Model Interface ---

// ModelEndpoint configures a model provider.
type ModelEndpoint struct {
	Provider    string  `json:"provider"`
	ModelID     string  `json:"model_id"`
	BaseURL     string  `json:"base_url"`
	APIKeyEnv   string  `json:"api_key_env"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	IsLocal     bool    `json:"is_local"`
}

// ModelConfig specifies models for each role.
type ModelConfig struct {
	Brain      ModelEndpoint `json:"brain"`
	Cerebellum ModelEndpoint `json:"cerebellum"`
}

// LocalExecutor supports KV cache prefix reuse for AOT mode.
type LocalExecutor interface {
	ExecuteWithPrefix(ctx context.Context, frozenPrefix string, variableSuffix string) (string, error)
}

// --- Data Collection Interface ---

// DataCollector receives framework events for flexible collection.
type DataCollector interface {
	OnEpoch(epoch ExecutionEpoch)
	OnSpanComplete(span LogicalSpan, epochs []ExecutionEpoch)
	OnTraceComplete(trace AgentTrace)
	OnCacheHit(action string)
	OnCacheEvict(action string, reason string)
	Close() error
}

// VisibilityFilter controls what data each actor can see.
type VisibilityFilter interface {
	FilterEpoch(epoch ExecutionEpoch, actor string) *ExecutionEpoch
	FilterSpan(span LogicalSpan, actor string) *LogicalSpan
}
