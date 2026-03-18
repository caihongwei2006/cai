package cai

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"
)

// EngineType identifies the execution environment for a script.
type EngineType string

const (
	EngineBash        EngineType = "bash"
	EngineAppleScript EngineType = "applescript"
	EngineTSBun       EngineType = "ts-bun"
	EnginePython      EngineType = "python"
	EngineNodeJS      EngineType = "nodejs"
)

// Intent is the Brain's primary output — minimal Apple-Intent-style dispatch.
// Serializes to ~20 tokens of JSON. The Meta field carries framework-internal
// orchestration metadata and is excluded from JSON serialization.
type Intent struct {
	Action string         `json:"action"`
	Target string         `json:"target"`
	Engine EngineType     `json:"engine"`
	Params map[string]any `json:"params,omitempty"`
	Meta   *IntentMeta    `json:"-"` // framework-side orchestration metadata, never serialized to LLM
}

// IntentMeta is framework-internal orchestration metadata.
// Not serialized into LLM context — populated by Planner post-processing.
type IntentMeta struct {
	PlanStepID     string   `json:"plan_step_id,omitempty"`
	PlanStepTitle  string   `json:"plan_step_title,omitempty"`
	ExecutionMode  string   `json:"execution_mode,omitempty"` // "inline" | "delegate" | "observe"
	Runtime        string   `json:"runtime,omitempty"`        // "root" | "subagent"
	Origin         string   `json:"origin,omitempty"`         // "planned" | "system"
	ParentStepID   string   `json:"parent_step_id,omitempty"`
	AllowedTools   []string `json:"allowed_tools,omitempty"`
	CapabilityID   string   `json:"capability_id,omitempty"`
	PromptIdentity string   `json:"prompt_identity,omitempty"`
	TaskClass      string   `json:"task_class,omitempty"`
	SessionKey     string   `json:"session_key,omitempty"`
	DoneWhen       string   `json:"done_when,omitempty"`
	Acceptance     string   `json:"acceptance,omitempty"`
	PlanRev        int      `json:"plan_rev,omitempty"`
}

// intentMeta returns a non-nil IntentMeta, falling back to reading from Params for backward compatibility.
func intentMeta(intent Intent) IntentMeta {
	if intent.Meta != nil {
		return *intent.Meta
	}
	// Backward compatibility: read from Params if Meta is not set
	m := IntentMeta{}
	if intent.Params == nil {
		return m
	}
	readStr := func(key string) string {
		if v, ok := intent.Params[key]; ok {
			if s, ok := v.(string); ok {
				return s
			}
		}
		return ""
	}
	m.PlanStepID = readStr("plan_step_id")
	m.PlanStepTitle = readStr("plan_step_title")
	m.ExecutionMode = readStr("execution_mode")
	m.Runtime = readStr("runtime")
	m.Origin = readStr("origin")
	m.ParentStepID = readStr("parent_step_id")
	m.CapabilityID = readStr("capability_id")
	m.PromptIdentity = readStr("prompt_identity")
	m.TaskClass = readStr("task_class")
	m.SessionKey = readStr("session_key")
	m.DoneWhen = readStr("done_when")
	m.Acceptance = readStr("acceptance")
	if m.DoneWhen == "" {
		m.DoneWhen = m.Acceptance
	}
	if m.Acceptance == "" {
		m.Acceptance = m.DoneWhen
	}
	if v, ok := intent.Params["plan_rev"]; ok {
		switch typed := v.(type) {
		case int:
			m.PlanRev = typed
		case float64:
			m.PlanRev = int(typed)
		case int64:
			m.PlanRev = int(typed)
		}
	}
	if v, ok := intent.Params["allowed_tools"]; ok {
		switch typed := v.(type) {
		case []string:
			m.AllowedTools = typed
		case []any:
			out := make([]string, 0, len(typed))
			for _, item := range typed {
				if s, ok := item.(string); ok {
					out = append(out, s)
				}
			}
			m.AllowedTools = out
		}
	}
	return m
}

// ErrorCategory classifies execution errors using RESTful semantics.
type ErrorCategory string

const (
	StatusSuccess ErrorCategory = "SUCCESS"
	ErrSyntax     ErrorCategory = "SYNTAX_ERROR"
	ErrDependency ErrorCategory = "DEPENDENCY_ERROR"
	ErrPermission ErrorCategory = "PERMISSION_ERROR"
	ErrLogic      ErrorCategory = "LOGIC_FORMAT_ERROR"
	ErrTimeout    ErrorCategory = "TIMEOUT"
	ErrUnknown    ErrorCategory = "UNKNOWN"
)

// RiskLevel aggregates the current threat to task completion.
type RiskLevel int

const (
	RiskNone   RiskLevel = 0
	RiskRetry  RiskLevel = 1
	RiskReplan RiskLevel = 2
	RiskHITL   RiskLevel = 3
)

// NodeStatus tracks a PlanNode's lifecycle.
type NodeStatus string

const (
	NodePending NodeStatus = "pending"
	NodeRunning NodeStatus = "running"
	NodeSuccess NodeStatus = "success"
	NodeFailed  NodeStatus = "failed"
	NodeSkipped NodeStatus = "skipped"
)

// ExecutionEnvelope unifies script output across all engines into RESTful semantics.
type ExecutionEnvelope struct {
	StatusCode int           `json:"status_code"`
	Category   ErrorCategory `json:"category"`
	RawStdout  string        `json:"raw_stdout"`
	RawStderr  string        `json:"raw_stderr"`
	ExitCode   int           `json:"exit_code"`
}

// ExecutionError is the framework's structured error type.
// Visibility is split: Root sees Summary+Attempts, Optimizer sees Detail+SystemPrompt.
type ExecutionError struct {
	Category     ErrorCategory `json:"category"`
	Code         int           `json:"code"`
	Summary      string        `json:"summary"`       // Root-visible: <=50 chars, no raw stderr
	Detail       string        `json:"detail"`        // Optimizer-only: full stderr
	Attempts     int           `json:"attempts"`      // Root-visible: consecutive failure count
	SystemPrompt string        `json:"system_prompt"` // Optimizer-only: current prompt being optimized
	SpanID       string        `json:"span_id"`
	EpochNum     int           `json:"epoch_num"`
	Intent       Intent        `json:"intent"`
}

// Error implements the error interface.
func (e *ExecutionError) Error() string {
	return fmt.Sprintf("[%s] %s (attempt %d)", e.Category, e.Summary, e.Attempts)
}

// ForRoot returns the Brain-visible subset: category, summary, attempt count.
func (e *ExecutionError) ForRoot() string {
	return fmt.Sprintf("%s: %s [attempt %d]", e.Category, e.Summary, e.Attempts)
}

// ForOptimizer returns the Triage-visible subset: full detail + current system prompt.
func (e *ExecutionError) ForOptimizer() (detail, currentPrompt string) {
	return e.Detail, e.SystemPrompt
}

// NewExecutionError constructs a structured error from an envelope + context.
func NewExecutionError(env ExecutionEnvelope, intent Intent, spanID string, epochNum int, attempts int, systemPrompt string) *ExecutionError {
	summary := env.RawStderr
	if len(summary) > 50 {
		summary = summary[:47] + "..."
	}
	return &ExecutionError{
		Category:     env.Category,
		Code:         env.StatusCode,
		Summary:      summary,
		Detail:       env.RawStderr,
		Attempts:     attempts,
		SystemPrompt: systemPrompt,
		SpanID:       spanID,
		EpochNum:     epochNum,
		Intent:       intent,
	}
}

// --- Channel Payloads (typed, zero-allocation-friendly) ---

type ImageInput struct {
	MimeType string `json:"mime_type"`
	Data     string `json:"data"`
}

type UserInput struct {
	Text   string       `json:"text"`
	Images []ImageInput `json:"images,omitempty"`
}

func (u UserInput) PlannerObjective() string {
	if trimmed := strings.TrimSpace(u.Text); trimmed != "" {
		return trimmed
	}
	if len(u.Images) > 0 {
		return "User sent an image attachment. Inspect the image and respond helpfully."
	}
	return ""
}

// TaskPayload flows Brain → Workers via Task_Chan.
type TaskPayload struct {
	SpanID    string
	EpochNum  int
	Scope     CerebellumScope
	Engine    EngineType
	Intent    Intent
	Ctx       context.Context
	UserInput UserInput
}

// ResultPayload flows Workers → Brain via Result_Chan.
type ResultPayload struct {
	SpanID   string
	EpochNum int
	Stdout   string
	Envelope ExecutionEnvelope
}

// ErrorPayload flows Workers → Triage via Error_Chan.
type ErrorPayload struct {
	SpanID       string
	EpochNum     int
	Envelope     ExecutionEnvelope
	Prompt       string // TaskPrompt that was used
	SystemPrompt string // current SystemPrompt — Triage needs this to know what to optimize
	Intent       Intent
}

// OptimizePayload flows Triage → Framework via Optimize_Chan.
type OptimizePayload struct {
	IntentAction   string
	PromptIdentity string
	CapabilityID   CapabilityID
	NewSystemHints string
	ShouldRetry    bool
	SpanID         string
	IncidentID     string
	CandidateActor string
}

// HITLPayload flows Triage → HITLResolver via HITL_Chan.
type HITLPayload struct {
	SpanID             string
	EpochNum           int
	LastError          ExecutionEnvelope
	TriageMemo         string
	Intent             Intent
	IncidentID         string
	CandidateActor     string
	ApprovalID         string
	AllowedActions     []string
	ResumeSystemPrompt string `json:"resume_system_prompt,omitempty"`
}

// Resolution is the HITL outcome.
type Resolution struct {
	Action    string `json:"action"` // "resume" | "abort" | "force_success"
	NewPrompt string `json:"new_prompt"`
	MockValue string `json:"mock_value"`
}

// LLMCallEvent captures one root/tool-loop model invocation.
type LLMCallEvent struct {
	Phase            string `json:"phase"`
	Iteration        int    `json:"iteration"`
	ModelID          string `json:"model_id,omitempty"`
	MessageCount     int    `json:"message_count,omitempty"`
	PromptTokens     int    `json:"prompt_tokens,omitempty"`
	CompletionTokens int    `json:"completion_tokens,omitempty"`
	TotalTokens      int    `json:"total_tokens,omitempty"`
	DurationMs       int64  `json:"duration_ms,omitempty"`
	FinishReason     string `json:"finish_reason,omitempty"`
	ToolCalls        int    `json:"tool_calls,omitempty"`
	Request          string `json:"request,omitempty"`
	Response         string `json:"response,omitempty"`
	Error            string `json:"error,omitempty"`
}

// ToolExecutionEvent captures one tool invocation produced by the root/tool loop.
type ToolExecutionEvent struct {
	ToolName   string `json:"tool_name"`
	ToolCallID string `json:"tool_call_id,omitempty"`
	Arguments  string `json:"arguments,omitempty"`
	Result     string `json:"result,omitempty"`
	IsError    bool   `json:"is_error,omitempty"`
	DurationMs int64  `json:"duration_ms,omitempty"`
}

// IncidentEvent describes append-only recovery lifecycle updates.
type IncidentEvent struct {
	EventType      string `json:"event_type"`
	TraceID        string `json:"trace_id"`
	RequestID      string `json:"request_id"`
	IncidentID     string `json:"incident_id"`
	SpanID         string `json:"span_id"`
	EpochNum       int    `json:"epoch_num"`
	CandidateActor string `json:"candidate_actor,omitempty"`
	WinnerActor    string `json:"winner_actor,omitempty"`
	ApprovalID     string `json:"approval_id,omitempty"`
	Reason         string `json:"reason,omitempty"`
	ErrorDigest    string `json:"error_digest,omitempty"`
	Status         string `json:"status,omitempty"`
}

// --- Scopes (strict isolation) ---

// BrainScope is visible only to Brain/Triage. Global plan and risk state.
type BrainScope struct {
	Objective     string
	PlanDAG       []PlanNode
	CurrentIndex  int8
	ErrorCounters map[string]int
	RiskLevel     RiskLevel
	GlobalMemory  map[string]string
}

// PlanNode is a single step in the Brain's execution plan.
type PlanNode struct {
	NodeID string
	Intent Intent
	Status NodeStatus
	SpanID string
}

// CerebellumScope is visible only to small model workers.
// Stateless, pure, minimal. Three segments:
//  1. SystemPrompt — user-defined initial prompt, iterable on fault
//  2. ToolSchema   — static tool/capability descriptions (like LangChain []tools)
//  3. TaskPrompt   — the single atomic instruction from Brain
//
// Beyond these, the small model sees NOTHING.
type CerebellumScope struct {
	SystemPrompt string    // user-defined, cached in IntentMemory, iterable on fault
	ToolSchema   []ToolDef // static tool definitions injected by framework
	TaskPrompt   string    // atomic instruction: "action: target"
	EnvMetadata  string    // "os=macOS_26, arch=arm64, shell=zsh"
}

// PlannedChildRequest is the root-orchestrated packet used to launch a planned
// child worker. It is internal runtime plumbing, not a peer LLM tool surface.
type PlannedChildRequest struct {
	TraceID string
	SpanID  string
	Intent  Intent
	Scope   CerebellumScope
}

// ToolDef describes a tool/capability available to the small model.
// Analogous to LangChain's Tool schema — declarative, not executable.
type ToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"` // JSON Schema, same source as Tool.InputSchema()
}

// --- Epoch-Based Trace Log ---

type AgentTrace struct {
	TraceID   string    `json:"trace_id"`
	Objective string    `json:"objective"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
}

type LogicalSpan struct {
	SpanID      string `json:"span_id"`
	TraceID     string `json:"trace_id"`
	PlanNodeID  string `json:"plan_node_id"`
	StepName    string `json:"step_name"`
	ActiveEpoch int    `json:"active_epoch"`
	FinalEpoch  int    `json:"final_epoch"`
	Status      string `json:"status"`
}

type ExecutionEpoch struct {
	EpochID     string        `json:"epoch_id"`
	SpanID      string        `json:"span_id"`
	EpochNumber int           `json:"epoch_number"`
	Prompt      string        `json:"prompt"`
	Result      string        `json:"result"`
	Error       string        `json:"error"`
	Category    ErrorCategory `json:"category"`
	IsDeadEnd   bool          `json:"is_dead_end"`
	CreatedAt   time.Time     `json:"created_at"`
	DurationMs  int64         `json:"duration_ms"`
}

// EpochSummary is the filtered view Brain gets from RetrieveSpanHistory.
// Never contains raw stderr.
type EpochSummary struct {
	EpochNumber int           `json:"epoch_number"`
	Category    ErrorCategory `json:"category"`
	ErrorDigest string        `json:"error_digest"`
	IsDeadEnd   bool          `json:"is_dead_end"`
}

// VisibleStep is the UI-facing projection of a span's latest status.
type VisibleStep struct {
	StepID        string `json:"step_id"`
	SpanID        string `json:"span_id"`
	Title         string `json:"title"`
	Detail        string `json:"detail,omitempty"`
	Status        string `json:"status"`
	SystemAdded   bool   `json:"system_added,omitempty"`
	ExecutionMode string `json:"execution_mode,omitempty"`
	Runtime       string `json:"runtime,omitempty"`
	Origin        string `json:"origin,omitempty"`
	ParentStepID  string `json:"parent_step_id,omitempty"`
	CapabilityID  string `json:"capability_id,omitempty"`
	PlanRev       int    `json:"plan_rev,omitempty"`
	UpdatedAt     int64  `json:"updated_at"`
}

type ChildRunState struct {
	ChildRunID   string `json:"child_run_id"`
	ParentRunID  string `json:"parent_run_id,omitempty"`
	ParentStepID string `json:"parent_step_id,omitempty"`
	Runtime      string `json:"runtime,omitempty"`
	Origin       string `json:"origin,omitempty"`
	Title        string `json:"title"`
	Summary      string `json:"summary,omitempty"`
	Status       string `json:"status"`
	TraceID      string `json:"trace_id,omitempty"`
	SessionKey   string `json:"session_key,omitempty"`
	StartedAt    int64  `json:"started_at,omitempty"`
	UpdatedAt    int64  `json:"updated_at"`
	EndedAt      int64  `json:"ended_at,omitempty"`
	LastError    string `json:"last_error,omitempty"`
}

// ActiveIncident captures a recoverable failure attached to a span.
type ActiveIncident struct {
	IncidentID     string `json:"incident_id"`
	SpanID         string `json:"span_id"`
	EpochNum       int    `json:"epoch_num"`
	Status         string `json:"status"`
	CandidateActor string `json:"candidate_actor,omitempty"`
	WinnerActor    string `json:"winner_actor,omitempty"`
	Reason         string `json:"reason,omitempty"`
	ErrorDigest    string `json:"error_digest,omitempty"`
	OpenedAt       int64  `json:"opened_at"`
	ResolvedAt     int64  `json:"resolved_at,omitempty"`
}

// PendingApproval captures a human decision that survives restarts.
type PendingApproval struct {
	ApprovalID string   `json:"approval_id"`
	IncidentID string   `json:"incident_id"`
	SpanID     string   `json:"span_id"`
	Reason     string   `json:"reason"`
	Actions    []string `json:"actions,omitempty"`
	OpenedAt   int64    `json:"opened_at"`
}

// SpanCheckpoint is the persisted pointer to the current span state.
type SpanCheckpoint struct {
	SpanID       string `json:"span_id"`
	IntentAction string `json:"intent_action"`
	StepName     string `json:"step_name"`
	Status       string `json:"status"`
	ActiveEpoch  int    `json:"active_epoch"`
	FinalEpoch   int    `json:"final_epoch"`
}

// --- Evolution Memory ---

type IntentMemory struct {
	IntentAction        string    `json:"intent_action"`
	SystemPromptHints   string    `json:"system_prompt_hints"`
	ContextHints        string    `json:"context_hints"`
	Frozen              bool      `json:"frozen"` // if true, prompt is locked — no self-iteration, no eviction
	SuccessStreak       int       `json:"success_streak"`
	ConsecutiveFailures int       `json:"consecutive_failures"`
	LastUsedAt          time.Time `json:"last_used_at"`
	LastEvictedAt       time.Time `json:"last_evicted_at"`
	Version             int       `json:"version"`
}

type CapabilityID string

type CapabilityProfile struct {
	ID          CapabilityID `json:"id"`
	Name        string       `json:"name"`
	Description string       `json:"description"`
	ToolAllow   []string     `json:"tool_allow,omitempty"`
	Tags        []string     `json:"tags,omitempty"`
	Version     int          `json:"version"`
}

type PromptIdentity struct {
	ID           string       `json:"id"`
	CapabilityID CapabilityID `json:"capability_id"`
	TaskClass    string       `json:"task_class"`
	ToolIntent   []string     `json:"tool_intent,omitempty"`
	StableHash   string       `json:"stable_hash,omitempty"`
	Version      int          `json:"version"`
}

type ExperienceEvent struct {
	EventID        string       `json:"event_id"`
	RunID          string       `json:"run_id"`
	TraceID        string       `json:"trace_id"`
	SpanID         string       `json:"span_id,omitempty"`
	EpochID        string       `json:"epoch_id,omitempty"`
	PromptIdentity string       `json:"prompt_identity"`
	CapabilityID   CapabilityID `json:"capability_id"`
	EventType      string       `json:"event_type"`
	ToolName       string       `json:"tool_name,omitempty"`
	ErrorCategory  string       `json:"error_category,omitempty"`
	ErrorSummary   string       `json:"error_summary,omitempty"`
	ErrorDetail    string       `json:"error_detail,omitempty"`
	PatchText      string       `json:"patch_text,omitempty"`
	VisibleToChild bool         `json:"visible_to_child"`
	CreatedAt      int64        `json:"created_at"`
}

type ToolHint struct {
	ToolName string `json:"tool_name"`
	Hint     string `json:"hint"`
}

type CompiledExperienceView struct {
	ViewID         string       `json:"view_id"`
	PromptIdentity string       `json:"prompt_identity"`
	CapabilityID   CapabilityID `json:"capability_id"`
	SystemPatch    string       `json:"system_patch"`
	ToolHints      []ToolHint   `json:"tool_hints,omitempty"`
	FailureGuards  []string     `json:"failure_guards,omitempty"`
	SourceEventIDs []string     `json:"source_event_ids,omitempty"`
	MaxTokens      int          `json:"max_tokens"`
	CompiledAt     int64        `json:"compiled_at"`
	Version        int          `json:"version"`
}

type RunPatch struct {
	PatchID        string       `json:"patch_id"`
	RunID          string       `json:"run_id"`
	SpanID         string       `json:"span_id,omitempty"`
	PromptIdentity string       `json:"prompt_identity"`
	CapabilityID   CapabilityID `json:"capability_id"`
	PatchText      string       `json:"patch_text"`
	Source         string       `json:"source"`
	ExpiresAt      int64        `json:"expires_at,omitempty"`
	CreatedAt      int64        `json:"created_at"`
}

type SystemEvolutionMemory struct {
	OSVersion           string            `json:"os_version"`
	Architecture        string            `json:"architecture"`
	Shell               string            `json:"shell"`
	PreferredPkgManager string            `json:"preferred_pkg_manager"`
	InstalledRuntimes   []string          `json:"installed_runtimes"`
	KnownQuirks         map[string]string `json:"known_quirks"`
	LastProbeAt         time.Time         `json:"last_probe_at"`
}

// --- Metrics ---

// TraceMetrics tracks per-trace performance data. All fields are atomic for concurrent access.
type TraceMetrics struct {
	TraceID            string
	TotalSOTATokens    atomic.Int64
	TotalFASTTokens    atomic.Int64
	CacheHits          atomic.Int64
	CacheMisses        atomic.Int64
	EvictionCount      atomic.Int64
	TotalRetries       atomic.Int64
	TotalEpochs        atomic.Int64
	TotalLatencyMs     atomic.Int64
	BrainTakeoverCount atomic.Int64
}

// CacheHitRate returns the fraction of intents served from cache.
func (m *TraceMetrics) CacheHitRate() float64 {
	hits := m.CacheHits.Load()
	total := hits + m.CacheMisses.Load()
	if total == 0 {
		return 0
	}
	return float64(hits) / float64(total)
}

// MetricsSnapshot is a non-atomic copy for serialization.
type MetricsSnapshot struct {
	TraceID            string  `json:"trace_id"`
	TotalSOTATokens    int64   `json:"total_sota_tokens"`
	TotalFASTTokens    int64   `json:"total_fast_tokens"`
	CacheHitRate       float64 `json:"cache_hit_rate"`
	EvictionCount      int64   `json:"eviction_count"`
	TotalRetries       int64   `json:"total_retries"`
	TotalEpochs        int64   `json:"total_epochs"`
	TotalLatencyMs     int64   `json:"total_latency_ms"`
	BrainTakeoverCount int64   `json:"brain_takeover_count"`
}

// Snapshot returns a serializable copy.
func (m *TraceMetrics) Snapshot() MetricsSnapshot {
	return MetricsSnapshot{
		TraceID:            m.TraceID,
		TotalSOTATokens:    m.TotalSOTATokens.Load(),
		TotalFASTTokens:    m.TotalFASTTokens.Load(),
		CacheHitRate:       m.CacheHitRate(),
		EvictionCount:      m.EvictionCount.Load(),
		TotalRetries:       m.TotalRetries.Load(),
		TotalEpochs:        m.TotalEpochs.Load(),
		TotalLatencyMs:     m.TotalLatencyMs.Load(),
		BrainTakeoverCount: m.BrainTakeoverCount.Load(),
	}
}

// --- Workspace Documents ---

// WorkspaceDocument represents a versioned .md file stored in SQLite.
// Supports the iteration lifecycle: .md -> SQLite -> evolve -> .md_vN write-back.
type WorkspaceDocument struct {
	Name         string    `json:"name"`           // e.g. "SOUL.md", "AGENTS.md"
	Path         string    `json:"path"`           // original disk path
	Content      string    `json:"content"`        // current content
	Version      int       `json:"version"`        // iteration counter
	Frozen       bool      `json:"frozen"`         // if true, no auto-evolution
	ContentHash  string    `json:"content_hash"`   // SHA256 for change detection
	LastSeedAt   time.Time `json:"last_seed_at"`   // when loaded from disk
	LastEvolveAt time.Time `json:"last_evolve_at"` // when SQLite diverged
}

// --- JIT/AOT ---

// FrozenInstruction is an AOT-compiled prompt. Once proven reliable,
// it enables zero-SOTA-token execution.
type FrozenInstruction struct {
	IntentAction   string    `json:"intent_action"`
	FixedPrompt    string    `json:"fixed_prompt"`
	ExpectedSchema string    `json:"expected_schema"`
	ModelConfig    string    `json:"model_config"`
	Temperature    float32   `json:"temperature"`
	MaxTokens      int       `json:"max_tokens"`
	Version        int       `json:"version"`
	CompiledAt     time.Time `json:"compiled_at"`
	SuccessRate    float64   `json:"success_rate"`
}
