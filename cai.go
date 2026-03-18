package cai

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/oklog/ulid/v2"
)

// AgentConfig is the developer-facing configuration for starting an agent.
// All business logic is injected via interfaces.
// For the simple API, use cai.New() with functional options instead.
type AgentConfig struct {
	Planner      Planner
	Executor     ExecutorFunc        // legacy: single-function executor (used when Tools is empty)
	Optimizer    OptimizerAnyFunc    // optional: nil disables self-iteration
	HITLResolver HITLResolverAnyFunc // optional: nil auto-aborts

	MemDB    MemoryDB          // optional: auto-created if nil
	Engines  EngineRegistry    // optional
	DocStore WorkspaceDocStore // optional: workspace .md document storage

	Tools                []Tool              // executable tools registered via WithTools (preferred over Executor)
	DefaultSystemPrompt  string              // base execution prompt assembled at runtime, not persisted as memory
	SystemPromptProvider func(Intent) string // optional dynamic provider for base execution prompt
	InitialPrompts       map[string]string   // deprecated compatibility field; active runtime no longer seeds or reads IntentMemory
	ToolSchemas          []ToolDef           // static tool definitions for small model (legacy, derived from Tools when empty)

	Collector   DataCollector // optional
	StateStore  StateStore    // optional
	IPCSocket   string        // optional: UDS socket path
	ModelConfig *ModelConfig  // optional
	PromptDir   string        // deprecated compatibility field; active runtime no longer hydrates prompts from scaffold files

	WorkerCount       int // default: 4
	MaxSelfIterations int // default: 3
	EvictionThreshold int // default: 2

	// Lifecycle hooks (optional, called from worker goroutines)
	OnTaskStart        func(spanID string, intent Intent)
	OnTaskComplete     func(spanID string, intent Intent, result string, err error)
	OnIncident         func(event IncidentEvent)
	OnControllerState  func(traceID string, intent Intent, phase string)
	OnLLMChunk         func(spanID string, intent Intent, phase string, text string)
	OnLLMCall          func(spanID string, intent Intent, event LLMCallEvent)
	OnToolCall         func(spanID string, intent Intent, event ToolExecutionEvent)
	PlannedChildRunner func(ctx context.Context, req PlannedChildRequest) (string, error)

	// internal
	dbPath  string          // set by WithDBPath
	ownsDB  bool            // if true, framework closes DB on Stop
	toolMap map[string]Tool // built from Tools at agent creation
}

// Function types for flexibility — avoids forcing generics on the orchestrator.
type ExecutorFunc func(ctx context.Context, prompt string, intent Intent) (string, error)

// OptimizerAnyFunc receives the current SystemPrompt (what to optimize) and the error.
// It returns an OptimizationResult with the improved prompt.
type OptimizerAnyFunc func(ctx context.Context, lastErr error, currentSystemPrompt string, intent Intent) (OptimizationResult, error)
type HITLResolverAnyFunc func(ctx context.Context, payload HITLPayload) (Resolution, error)

// Agent is the running framework instance.
type Agent struct {
	config AgentConfig
	cancel context.CancelFunc

	// Channels
	taskChan     chan TaskPayload
	resultChan   chan ResultPayload
	errorChan    chan ErrorPayload
	optimizeChan chan OptimizePayload
	hitlChan     chan HITLPayload

	// State
	traceID  string
	wg       sync.WaitGroup
	stateMu  sync.Mutex
	runs     map[string]*runState
	spanRun  map[string]string
	childSem chan struct{} // delegate child concurrency limiter
}

type runState struct {
	traceID             string
	requestID           string
	objective           string
	createdAt           int64
	latestObservationAt int64
	stepOrder           []string
	visibleSteps        map[string]VisibleStep
	currentSpans        map[string]SpanCheckpoint
	activeIncidents     map[string]ActiveIncident
	pendingApproval     *PendingApproval
	incidentBySpan      map[string]string
	childRuns           map[string]ChildRunState
	controllerState     string
	activePlanRev       int
	quiescingStartedAt  int64
	lastSummaryAt       int64
	revisionReason      string
}

// NewAgent creates an agent from an explicit AgentConfig.
// For the simple API, prefer cai.New() with functional options.
// If MemDB is nil, use cai.New() which auto-creates SQLite.
func NewAgent(ctx context.Context, config AgentConfig) (*Agent, error) {
	if config.MemDB == nil {
		return nil, fmt.Errorf("MemDB is required (use cai.New() for auto-creation)")
	}
	return newAgent(ctx, config)
}

func newAgent(ctx context.Context, config AgentConfig) (*Agent, error) {
	if config.Planner == nil {
		return nil, fmt.Errorf("Planner is required")
	}
	if config.Executor == nil && len(config.Tools) == 0 {
		return nil, fmt.Errorf("either Executor or Tools is required")
	}
	if config.MemDB == nil {
		return nil, fmt.Errorf("MemDB is required")
	}
	if config.StateStore == nil {
		if store, ok := config.MemDB.(StateStore); ok {
			config.StateStore = store
		}
	}

	if config.WorkerCount <= 0 {
		config.WorkerCount = 4
	}
	if config.MaxSelfIterations <= 0 {
		config.MaxSelfIterations = 3
	}
	if config.EvictionThreshold <= 0 {
		config.EvictionThreshold = 2
	}

	// Build internal toolMap from registered Tools
	if len(config.Tools) > 0 {
		config.toolMap = make(map[string]Tool, len(config.Tools))
		for _, t := range config.Tools {
			config.toolMap[t.Name()] = t
		}
		// Auto-derive ToolSchemas from Tools only when visibility was not explicitly set.
		// An explicit empty slice means tools exist internally but are hidden from the model.
		if config.ToolSchemas == nil {
			config.ToolSchemas = make([]ToolDef, 0, len(config.Tools))
			for _, t := range config.Tools {
				config.ToolSchemas = append(config.ToolSchemas, ToolDef{
					Name:        t.Name(),
					Description: t.Description(),
					Parameters:  t.InputSchema(),
				})
			}
		}
	}

	ctx, cancel := context.WithCancel(ctx)

	bufSize := config.WorkerCount * 4
	if bufSize < 8 {
		bufSize = 8
	}

	childSemSize := config.WorkerCount * 2
	if childSemSize < 4 {
		childSemSize = 4
	}

	a := &Agent{
		config:       config,
		cancel:       cancel,
		taskChan:     make(chan TaskPayload, bufSize),
		resultChan:   make(chan ResultPayload, bufSize),
		errorChan:    make(chan ErrorPayload, bufSize),
		optimizeChan: make(chan OptimizePayload, bufSize),
		hitlChan:     make(chan HITLPayload, config.WorkerCount),
		traceID:      "trace_" + ulid.Make().String(),
		runs:         make(map[string]*runState),
		spanRun:      make(map[string]string),
		childSem:     make(chan struct{}, childSemSize),
	}

	for i := 0; i < config.WorkerCount; i++ {
		a.wg.Add(1)
		go a.worker(ctx, i)
	}

	a.wg.Add(1)
	go a.triageLoop(ctx)

	return a, nil
}

// Run executes a user objective through the full Brain→Cerebellum pipeline.
func (a *Agent) Run(ctx context.Context, objective string) error {
	_, err := a.RunWithResult(ctx, objective)
	return err
}

// RunWithResult executes a user objective and returns the final text result.
func (a *Agent) RunWithResult(ctx context.Context, objective string) (string, error) {
	return a.RunInputWithResult(ctx, UserInput{Text: objective})
}

// RunInputWithResult executes a structured user input and returns the final text result.
func (a *Agent) RunInputWithResult(ctx context.Context, input UserInput) (string, error) {
	objective := input.PlannerObjective()
	runTraceID := "trace_" + ulid.Make().String()
	trace := AgentTrace{
		TraceID:   runTraceID,
		Objective: objective,
		Status:    "running",
		CreatedAt: time.Now().UTC(),
	}
	if err := a.config.MemDB.CreateTrace(trace); err != nil {
		return "", fmt.Errorf("create trace: %w", err)
	}

	intents, err := a.config.Planner.Plan(ctx, objective, &BrainScope{
		Objective:     objective,
		ErrorCounters: make(map[string]int),
		GlobalMemory:  make(map[string]string),
	})
	if err != nil {
		return "", fmt.Errorf("plan: %w", err)
	}

	oldTraceID := a.traceID
	a.traceID = runTraceID
	defer func() { a.traceID = oldTraceID }()

	return a.executeIntentsWithResult(ctx, input, intents)
}

func (a *Agent) executeIntents(ctx context.Context, objective string, intents []Intent) error {
	_, err := a.executeIntentsWithResult(ctx, UserInput{Text: objective}, intents)
	return err
}

type spanExecution struct {
	intent    Intent
	spanID    string
	stepName  string
	nextEpoch int
	userInput UserInput
}

type observedChildResult struct {
	StepID  string
	Title   string
	Status  string
	Detail  string
	Result  string
	Success bool
}

// hitlResolution carries the result of an async HITL goroutine back to the main loop.
type hitlResolution struct {
	spanID     string
	resolution Resolution
	hitl       HITLPayload
	err        error
}

// executeIntentsWithResult dispatches all intents concurrently and aggregates successful results.
func (a *Agent) executeIntentsWithResult(ctx context.Context, input UserInput, intents []Intent) (string, error) {
	objective := input.PlannerObjective()
	if len(intents) == 0 {
		_ = a.config.MemDB.UpdateTraceStatus(a.traceID, "success")
		if a.config.Collector != nil {
			a.config.Collector.OnTraceComplete(AgentTrace{
				TraceID:   a.traceID,
				Objective: objective,
				Status:    "success",
				CreatedAt: time.Now().UTC(),
			})
		}
		return "", nil
	}

	traceID := a.traceID
	traceCreatedAt := time.Now().UTC()
	hasDelegateChildren := false
	hasExplicitObserve := false
	observeCandidateIdx := -1
	fallbackObserveIdx := -1
	for i, intent := range intents {
		executionMode := firstNonEmpty(readIntentParam(intent, "execution_mode"), "inline")
		runtime := firstNonEmpty(readIntentParam(intent, "runtime"), "root")
		if executionMode == "observe" {
			hasExplicitObserve = true
		}
		if executionMode == "delegate" || runtime == "subagent" {
			hasDelegateChildren = true
		}
		if runtime == "root" && executionMode == "inline" {
			fallbackObserveIdx = i
			if looksLikeObserveBarrier(intent) && observeCandidateIdx < 0 {
				observeCandidateIdx = i
			}
		}
	}
	if hasDelegateChildren && !hasExplicitObserve && observeCandidateIdx < 0 {
		observeCandidateIdx = fallbackObserveIdx
	}
	run := &runState{
		traceID:             traceID,
		requestID:           traceID,
		objective:           objective,
		createdAt:           time.Now().UTC().UnixMilli(),
		latestObservationAt: time.Now().UTC().UnixMilli(),
		visibleSteps:        make(map[string]VisibleStep, len(intents)),
		currentSpans:        make(map[string]SpanCheckpoint, len(intents)),
		activeIncidents:     make(map[string]ActiveIncident),
		incidentBySpan:      make(map[string]string),
		childRuns:           make(map[string]ChildRunState),
		controllerState:     "executing",
		activePlanRev:       maxPlanRevision(intents),
	}

	pending := make(map[string]*spanExecution, len(intents))
	observePending := make(map[string]*spanExecution)
	observeStarted := make(map[string]bool)
	spanByStepID := make(map[string]string, len(intents))
	childrenByObserve := make(map[string][]string)
	childResults := make(map[string]observedChildResult)
	inlineResults := make(map[string]string)
	for i, intent := range intents {
		if i == observeCandidateIdx {
			if intent.Params == nil {
				intent.Params = make(map[string]any)
			}
			intent.Params["execution_mode"] = "observe"
			intent.Params["runtime"] = "root"
		}
		spanID := "span_" + ulid.Make().String()
		stepName := fmt.Sprintf("%s:%s", intent.Action, intent.Target)
		capabilityID := capabilityIDForIntent(intent)
		_ = a.config.MemDB.SaveCapabilityProfile(CapabilityProfile{
			ID:          capabilityID,
			Name:        string(capabilityID),
			Description: readIntentParam(intent, "plan_step_title"),
			ToolAllow:   toolIntentForIntent(intent),
			Tags:        []string{taskClassForIntent(intent)},
			Version:     1,
		})
		span := LogicalSpan{
			SpanID:      spanID,
			TraceID:     traceID,
			PlanNodeID:  fmt.Sprintf("node_%d", i),
			StepName:    stepName,
			ActiveEpoch: 0,
			Status:      "pending",
		}
		if err := a.config.MemDB.CreateSpan(span); err != nil {
			return "", fmt.Errorf("create span: %w", err)
		}

		pending[spanID] = &spanExecution{
			intent:    intent,
			spanID:    spanID,
			stepName:  stepName,
			nextEpoch: 1,
			userInput: inputForIntent(intent, objective, input),
		}

		run.stepOrder = append(run.stepOrder, spanID)
		planStepID := firstNonEmpty(readIntentParam(intent, "plan_step_id"), spanID)
		spanByStepID[planStepID] = spanID
		title := firstNonEmpty(readIntentParam(intent, "plan_step_title"), stepName)
		executionMode := firstNonEmpty(readIntentParam(intent, "execution_mode"), "inline")
		runtime := firstNonEmpty(readIntentParam(intent, "runtime"), "root")
		origin := firstNonEmpty(readIntentParam(intent, "origin"), "planned")
		parentStepID := readIntentParam(intent, "parent_step_id")
		planRev := readIntIntentParam(intent, "plan_rev")
		if planRev <= 0 {
			planRev = run.activePlanRev
		}
		run.visibleSteps[spanID] = VisibleStep{
			StepID:        planStepID,
			SpanID:        spanID,
			Title:         title,
			Detail:        intent.Target,
			Status:        "pending",
			ExecutionMode: executionMode,
			Runtime:       runtime,
			Origin:        origin,
			ParentStepID:  parentStepID,
			CapabilityID:  string(capabilityID),
			PlanRev:       planRev,
			UpdatedAt:     run.latestObservationAt,
		}
		run.currentSpans[spanID] = SpanCheckpoint{
			SpanID:       spanID,
			IntentAction: intent.Action,
			StepName:     stepName,
			Status:       "pending",
			ActiveEpoch:  0,
			FinalEpoch:   0,
		}
		if executionMode == "delegate" || runtime == "subagent" {
			run.childRuns[spanID] = ChildRunState{
				ChildRunID:   spanID,
				ParentRunID:  traceID,
				ParentStepID: firstNonEmpty(parentStepID, planStepID),
				Runtime:      firstNonEmpty(runtime, "subagent"),
				Origin:       origin,
				Title:        title,
				Summary:      intent.Target,
				Status:       "pending",
				TraceID:      traceID,
				SessionKey:   readIntentParam(intent, "session_key"),
				StartedAt:    run.createdAt,
				UpdatedAt:    run.latestObservationAt,
			}
		}
		if executionMode == "observe" {
			observePending[spanID] = pending[spanID]
		}
	}
	for spanID, exec := range pending {
		parentStepID := readIntentParam(exec.intent, "parent_step_id")
		if parentStepID == "" {
			continue
		}
		parentSpanID, ok := spanByStepID[parentStepID]
		if !ok {
			continue
		}
		childrenByObserve[parentSpanID] = append(childrenByObserve[parentSpanID], spanID)
	}
	if len(observePending) == 1 {
		var observeSpanID string
		for spanID := range observePending {
			observeSpanID = spanID
		}
		observeStepID := run.visibleSteps[observeSpanID].StepID
		for spanID, exec := range pending {
			if spanID == observeSpanID {
				continue
			}
			executionMode := firstNonEmpty(readIntentParam(exec.intent, "execution_mode"), "inline")
			runtime := firstNonEmpty(readIntentParam(exec.intent, "runtime"), "root")
			if executionMode != "delegate" && runtime != "subagent" {
				continue
			}
			step := run.visibleSteps[spanID]
			if strings.TrimSpace(step.ParentStepID) != "" {
				continue
			}
			step.ParentStepID = observeStepID
			run.visibleSteps[spanID] = step
			if child, ok := run.childRuns[spanID]; ok {
				child.ParentStepID = observeStepID
				run.childRuns[spanID] = child
			}
			childrenByObserve[observeSpanID] = append(childrenByObserve[observeSpanID], spanID)
		}
	}

	a.registerRunState(run)
	defer a.cleanupRunState(traceID)
	a.stateMu.Lock()
	initialState := a.snapshotLocked(traceID)
	a.stateMu.Unlock()
	a.persistSnapshot(context.Background(), traceID, initialState)
	finalizeTrace := func(status string) {
		_ = a.config.MemDB.ClearRunPatches(traceID)
		_ = a.config.MemDB.UpdateTraceStatus(traceID, status)
		if a.config.Collector != nil {
			a.config.Collector.OnTraceComplete(AgentTrace{
				TraceID:   traceID,
				Objective: objective,
				Status:    status,
				CreatedAt: traceCreatedAt,
			})
		}
	}

	dispatchExec := func(exec *spanExecution, scope CerebellumScope) error {
		executionMode := firstNonEmpty(readIntentParam(exec.intent, "execution_mode"), "inline")
		runtime := firstNonEmpty(readIntentParam(exec.intent, "runtime"), "root")
		switch {
		case executionMode == "observe":
			return a.dispatchTask(ctx, traceID, exec, scope)
		case executionMode == "delegate" || runtime == "subagent":
			if a.config.PlannedChildRunner != nil {
				return a.dispatchPlannedChild(ctx, traceID, exec, scope)
			}
			return a.dispatchTask(ctx, traceID, exec, scope)
		default:
			return a.dispatchTask(ctx, traceID, exec, scope)
		}
	}
	tryDispatchObserve := func() error {
		for observeSpanID, observeExec := range observePending {
			if observeStarted[observeSpanID] {
				continue
			}
			childSpanIDs := childrenByObserve[observeSpanID]
			if len(childSpanIDs) == 0 {
				observeStarted[observeSpanID] = true
				if err := dispatchExec(observeExec, CerebellumScope{
					TaskPrompt: buildObserveTaskPrompt(objective, observeExec.intent, nil),
				}); err != nil {
					return err
				}
				continue
			}

			ready := true
			observed := make([]observedChildResult, 0, len(childSpanIDs))
			for _, childSpanID := range childSpanIDs {
				if _, ok := pending[childSpanID]; ok {
					ready = false
					break
				}
				child, ok := childResults[childSpanID]
				if !ok || !child.Success {
					ready = false
					break
				}
				observed = append(observed, child)
			}
			if !ready {
				continue
			}
			observeStarted[observeSpanID] = true
			if err := dispatchExec(observeExec, CerebellumScope{
				TaskPrompt: buildObserveTaskPrompt(objective, observeExec.intent, observed),
			}); err != nil {
				return err
			}
		}
		return nil
	}

	for spanID, exec := range pending {
		if _, ok := observePending[spanID]; ok {
			continue
		}
		if err := dispatchExec(exec, CerebellumScope{}); err != nil {
			finalizeTrace("failed")
			return "", err
		}
	}
	for _, observeExec := range observePending {
		a.transitionControllerPhase(traceID, observeExec.intent, "observing")
		break
	}
	if err := tryDispatchObserve(); err != nil {
		finalizeTrace("failed")
		return "", err
	}

	var firstErr error
	var results []string
	hitlResultChan := make(chan hitlResolution, cap(a.hitlChan))
	for len(pending) > 0 {
		select {
		case <-ctx.Done():
			a.transitionControllerPhase(traceID, intents[0], "failed")
			finalizeTrace("failed")
			return "", ctx.Err()

		case result := <-a.resultChan:
			exec, ok := pending[result.SpanID]
			if !ok {
				continue
			}
			if result.Envelope.Category != StatusSuccess {
				continue
			}

			a.config.MemDB.UpdateSpanStatus(result.SpanID, "success", result.EpochNum)
			a.recordSpanSuccess(traceID, result.SpanID, result.EpochNum, result.Stdout)
			a.appendExperienceEvent(exec.intent, traceID, traceID, result.SpanID, "", "attempt_succeeded", string(result.Envelope.Category), result.Stdout, "", "", true)
			stepTitle := firstNonEmpty(readIntentParam(exec.intent, "plan_step_title"), exec.intent.Target)
			childResults[result.SpanID] = observedChildResult{
				StepID:  firstNonEmpty(readIntentParam(exec.intent, "plan_step_id"), result.SpanID),
				Title:   stepTitle,
				Status:  "success",
				Detail:  truncateText(result.Stdout, 280),
				Result:  result.Stdout,
				Success: true,
			}
			if firstNonEmpty(readIntentParam(exec.intent, "execution_mode"), "inline") != "delegate" &&
				firstNonEmpty(readIntentParam(exec.intent, "execution_mode"), "inline") != "observe" &&
				firstNonEmpty(readIntentParam(exec.intent, "runtime"), "root") == "root" {
				inlineResults[result.SpanID] = result.Stdout
			}
			if firstNonEmpty(readIntentParam(exec.intent, "execution_mode"), "inline") == "observe" {
				results = append(results, result.Stdout)
			}
			delete(pending, result.SpanID)
			if err := tryDispatchObserve(); err != nil {
				finalizeTrace("failed")
				return "", err
			}

		case opt := <-a.optimizeChan:
			exec, ok := pending[opt.SpanID]
			if !ok {
				continue
			}

			if opt.NewSystemHints != "" {
				a.saveRunPatch(exec.intent, traceID, exec.spanID, "optimizer", opt.NewSystemHints)
				a.appendExperienceEvent(exec.intent, traceID, traceID, exec.spanID, "", "optimizer_patch", "", "", "", opt.NewSystemHints, true)
			}

			scope := CerebellumScope{}
			if opt.NewSystemHints != "" {
				scope.SystemPrompt = opt.NewSystemHints
			}
			if err := dispatchExec(exec, scope); err != nil {
				finalizeTrace("failed")
				return "", err
			}

		case hitl := <-a.hitlChan:
			exec, ok := pending[hitl.SpanID]
			if !ok {
				continue
			}

			if a.config.HITLResolver == nil {
				a.recordIncidentResolved(traceID, hitl.IncidentID, "aborted", "human")
				a.config.MemDB.UpdateSpanStatus(hitl.SpanID, "failed", hitl.EpochNum)
				stepTitle := firstNonEmpty(readIntentParam(exec.intent, "plan_step_title"), exec.intent.Target)
				childResults[hitl.SpanID] = observedChildResult{
					StepID:  firstNonEmpty(readIntentParam(exec.intent, "plan_step_id"), hitl.SpanID),
					Title:   stepTitle,
					Status:  "fail",
					Detail:  fmt.Sprintf("intent %s aborted via HITL", exec.intent.Action),
					Result:  "",
					Success: false,
				}
				if firstErr == nil {
					firstErr = fmt.Errorf("HITL escalation with no resolver for %s", exec.intent.Action)
				}
				delete(pending, hitl.SpanID)
				continue
			}

			// Launch HITL resolution in a separate goroutine to avoid blocking
			// the main select loop while waiting for human input.
			go func(hitl HITLPayload, exec *spanExecution) {
				resolution, err := a.config.HITLResolver(ctx, hitl)
				if err != nil {
					// On resolver error, treat as abort
					a.recordIncidentResolved(traceID, hitl.IncidentID, "aborted", "human")
					a.config.MemDB.UpdateSpanStatus(hitl.SpanID, "failed", hitl.EpochNum)
					hitlResultChan <- hitlResolution{
						spanID:     hitl.SpanID,
						resolution: Resolution{Action: "abort"},
						err:        err,
					}
					return
				}
				hitlResultChan <- hitlResolution{
					spanID:     hitl.SpanID,
					resolution: resolution,
					hitl:       hitl,
				}
			}(hitl, exec)

		case hr := <-hitlResultChan:
			exec, ok := pending[hr.spanID]
			if !ok {
				continue
			}

			if hr.err != nil {
				stepTitle := firstNonEmpty(readIntentParam(exec.intent, "plan_step_title"), exec.intent.Target)
				childResults[hr.spanID] = observedChildResult{
					StepID:  firstNonEmpty(readIntentParam(exec.intent, "plan_step_id"), hr.spanID),
					Title:   stepTitle,
					Status:  "fail",
					Detail:  fmt.Sprintf("HITL resolver error: %v", hr.err),
					Result:  "",
					Success: false,
				}
				if firstErr == nil {
					firstErr = fmt.Errorf("HITL resolver: %w", hr.err)
				}
				delete(pending, hr.spanID)
				continue
			}

			switch hr.resolution.Action {
			case "abort":
				a.recordIncidentResolved(traceID, hr.hitl.IncidentID, "aborted", "human")
				a.config.MemDB.UpdateSpanStatus(hr.spanID, "failed", hr.hitl.EpochNum)
				stepTitle := firstNonEmpty(readIntentParam(exec.intent, "plan_step_title"), exec.intent.Target)
				childResults[hr.spanID] = observedChildResult{
					StepID:  firstNonEmpty(readIntentParam(exec.intent, "plan_step_id"), hr.spanID),
					Title:   stepTitle,
					Status:  "fail",
					Detail:  fmt.Sprintf("intent %s aborted via HITL", exec.intent.Action),
					Result:  "",
					Success: false,
				}
				if firstErr == nil {
					firstErr = fmt.Errorf("intent %s aborted via HITL", exec.intent.Action)
				}
				delete(pending, hr.spanID)

			case "force_success":
				forcedEpoch := exec.nextEpoch
				exec.nextEpoch++
				a.appendForcedSuccess(hr.spanID, forcedEpoch, hr.resolution.MockValue)
				a.config.MemDB.UpdateSpanStatus(hr.spanID, "success", forcedEpoch)
				a.recordIncidentResolved(traceID, hr.hitl.IncidentID, "resolved", "human")
				a.recordSpanSuccess(traceID, hr.spanID, forcedEpoch, hr.resolution.MockValue)
				stepTitle := firstNonEmpty(readIntentParam(exec.intent, "plan_step_title"), exec.intent.Target)
				childResults[hr.spanID] = observedChildResult{
					StepID:  firstNonEmpty(readIntentParam(exec.intent, "plan_step_id"), hr.spanID),
					Title:   stepTitle,
					Status:  "success",
					Detail:  truncateText(hr.resolution.MockValue, 280),
					Result:  hr.resolution.MockValue,
					Success: true,
				}
				if firstNonEmpty(readIntentParam(exec.intent, "execution_mode"), "inline") != "delegate" &&
					firstNonEmpty(readIntentParam(exec.intent, "execution_mode"), "inline") != "observe" &&
					firstNonEmpty(readIntentParam(exec.intent, "runtime"), "root") == "root" {
					inlineResults[hr.spanID] = hr.resolution.MockValue
				}
				if firstNonEmpty(readIntentParam(exec.intent, "execution_mode"), "inline") == "observe" {
					results = append(results, hr.resolution.MockValue)
				}
				delete(pending, hr.spanID)
				if err := tryDispatchObserve(); err != nil {
					finalizeTrace("failed")
					return "", err
				}

			case "resume":
				scope := CerebellumScope{}
				if hr.resolution.NewPrompt != "" {
					a.saveRunPatch(exec.intent, traceID, exec.spanID, "human", hr.resolution.NewPrompt)
					a.appendExperienceEvent(exec.intent, traceID, traceID, exec.spanID, "", "human_patch", "", "", "", hr.resolution.NewPrompt, true)
					scope.SystemPrompt = hr.resolution.NewPrompt
				}
				a.recordIncidentResolved(traceID, hr.hitl.IncidentID, "resolved", "human")
				if err := dispatchExec(exec, scope); err != nil {
					finalizeTrace("failed")
					return "", err
				}
			}
		}
	}

	if firstErr != nil {
		a.transitionControllerPhase(traceID, intents[0], "failed")
		finalizeTrace("failed")
		return "", firstErr
	}
	if err := a.enterStableCompletion(ctx, traceID, intents[0]); err != nil {
		a.transitionControllerPhase(traceID, intents[0], "failed")
		finalizeTrace("failed")
		return "", err
	}
	if a.config.StateStore != nil {
		_ = a.config.StateStore.Clear(ctx, traceID)
	}
	if len(results) == 0 && len(inlineResults) > 0 {
		for _, stepID := range run.stepOrder {
			if result, ok := inlineResults[stepID]; ok {
				results = append(results, result)
			}
		}
	}
	if len(results) == 1 {
		finalizeTrace("success")
		return results[0], nil
	}
	if len(results) > 1 {
		finalizeTrace("success")
		return strings.Join(results, "\n\n---\n\n"), nil
	}
	finalizeTrace("success")
	return "", nil
}

func buildObserveTaskPrompt(objective string, intent Intent, children []observedChildResult) string {
	var b strings.Builder
	b.WriteString("You are the root orchestrator. Child work has completed. Produce the final user-facing answer.\n\n")
	b.WriteString("Original objective:\n")
	b.WriteString(strings.TrimSpace(objective))
	b.WriteString("\n\n")
	if acceptance := firstNonEmpty(strings.TrimSpace(readIntentParam(intent, "done_when")), strings.TrimSpace(readIntentParam(intent, "acceptance"))); acceptance != "" {
		b.WriteString("Success criteria:\n")
		b.WriteString(acceptance)
		b.WriteString("\n\n")
	}
	if len(children) > 0 {
		b.WriteString("Child results:\n")
		for _, child := range children {
			b.WriteString(fmt.Sprintf("- %s [%s]\n", child.Title, child.Status))
			if child.Result != "" {
				b.WriteString(truncateText(child.Result, 2000))
			} else {
				b.WriteString(child.Detail)
			}
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}
	b.WriteString("Respond directly to the user. Synthesize the result; do not mention internal planning unless it is useful.")
	return strings.TrimSpace(b.String())
}

func (a *Agent) registerRunState(run *runState) {
	a.stateMu.Lock()
	defer a.stateMu.Unlock()

	a.runs[run.traceID] = run
	for _, stepID := range run.stepOrder {
		a.spanRun[stepID] = run.traceID
	}
}

func (a *Agent) cleanupRunState(traceID string) {
	a.stateMu.Lock()
	defer a.stateMu.Unlock()

	run := a.runs[traceID]
	if run != nil {
		for _, stepID := range run.stepOrder {
			delete(a.spanRun, stepID)
		}
	}
	delete(a.runs, traceID)
}

func readIntentParam(intent Intent, key string) string {
	if intent.Params == nil {
		return ""
	}
	value, ok := intent.Params[key]
	if !ok {
		return ""
	}
	asString, ok := value.(string)
	if !ok {
		return ""
	}
	return asString
}

func readIntIntentParam(intent Intent, key string) int {
	if intent.Params == nil {
		return 0
	}
	value, ok := intent.Params[key]
	if !ok {
		return 0
	}
	switch typed := value.(type) {
	case int:
		return typed
	case int64:
		return int(typed)
	case float64:
		return int(typed)
	default:
		return 0
	}
}

func readStringListIntentParam(intent Intent, key string) ([]string, bool) {
	if intent.Params == nil {
		return nil, false
	}
	value, ok := intent.Params[key]
	if !ok {
		return nil, false
	}
	if value == nil {
		return []string{}, true
	}
	switch typed := value.(type) {
	case []string:
		return typed, true
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if asString, ok := item.(string); ok {
				out = append(out, asString)
			}
		}
		return out, true
	default:
		return nil, false
	}
}

func capabilityIDForIntent(intent Intent) CapabilityID {
	if value := readIntentParam(intent, "capability_id"); value != "" {
		return CapabilityID(value)
	}
	if intent.Action == "agent_turn" {
		return CapabilityID("general.agent_turn")
	}
	return CapabilityID("general." + intent.Action)
}

func promptIdentityForIntent(intent Intent) string {
	if value := readIntentParam(intent, "prompt_identity"); value != "" {
		return value
	}
	return "cap:" + string(capabilityIDForIntent(intent))
}

func toolIntentForIntent(intent Intent) []string {
	if intent.Params == nil {
		return nil
	}
	value, ok := intent.Params["allowed_tools"]
	if !ok {
		return nil
	}
	switch typed := value.(type) {
	case []string:
		return typed
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if asString, ok := item.(string); ok && asString != "" {
				out = append(out, asString)
			}
		}
		return out
	default:
		return nil
	}
}

func taskClassForIntent(intent Intent) string {
	if value := readIntentParam(intent, "task_class"); value != "" {
		return value
	}
	return "execute"
}

func (a *Agent) toolsForIntent(intent Intent) []Tool {
	names, ok := readStringListIntentParam(intent, "allowed_tools")
	if !ok {
		return a.config.Tools
	}
	allowed := make(map[string]bool, len(names))
	for _, name := range names {
		if strings.TrimSpace(name) != "" {
			allowed[name] = true
		}
	}
	filtered := make([]Tool, 0, len(a.config.Tools))
	for _, tool := range a.config.Tools {
		if allowed[tool.Name()] {
			filtered = append(filtered, tool)
		}
	}
	return filtered
}

func looksLikeObserveBarrier(intent Intent) bool {
	candidates := []string{
		readIntentParam(intent, "capability_id"),
		readIntentParam(intent, "plan_step_title"),
		intent.Target,
	}
	for _, value := range candidates {
		lower := strings.ToLower(strings.TrimSpace(value))
		if lower == "" {
			continue
		}
		keywords := []string{
			"orchestration.observe_aggregate",
			"observe",
			"aggregate",
			"synthesi",
			"summary",
			"summar",
			"report back",
			"final answer",
			"汇总",
			"总结",
			"聚合",
			"汇报",
		}
		for _, keyword := range keywords {
			if strings.Contains(lower, keyword) {
				return true
			}
		}
	}
	return false
}

func truncateText(value string, limit int) string {
	if limit <= 0 || len(value) <= limit {
		return value
	}
	return value[:limit-3] + "..."
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}

func maxPlanRevision(intents []Intent) int {
	maxRev := 1
	for _, intent := range intents {
		if rev := readIntIntentParam(intent, "plan_rev"); rev > maxRev {
			maxRev = rev
		}
	}
	return maxRev
}

func sanitizePromptFileKey(value string) string {
	replacer := strings.NewReplacer(":", "_", "/", "_", "\\", "_", " ", "_")
	return replacer.Replace(value)
}

func dedupeNonEmpty(values []string, limit int) []string {
	if limit <= 0 {
		return nil
	}
	out := make([]string, 0, limit)
	seen := make(map[string]struct{}, limit)
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		out = append(out, trimmed)
		if len(out) >= limit {
			break
		}
	}
	return out
}

func (a *Agent) compileExperienceView(intent Intent, runID, spanID string) *CompiledExperienceView {
	promptIdentity := promptIdentityForIntent(intent)
	if promptIdentity == "" {
		return nil
	}
	capabilityID := capabilityIDForIntent(intent)

	runPatches, err := a.config.MemDB.ListRunPatches(runID, promptIdentity, capabilityID, 8)
	if err != nil {
		return nil
	}
	events, err := a.config.MemDB.ListExperienceEvents(promptIdentity, capabilityID, true, 16)
	if err != nil {
		return nil
	}

	var patchTexts []string
	var sourceEventIDs []string
	for _, patch := range runPatches {
		patchTexts = append(patchTexts, patch.PatchText)
		sourceEventIDs = append(sourceEventIDs, patch.PatchID)
	}

	var successText string
	for _, event := range events {
		switch event.EventType {
		case "optimizer_patch", "human_patch", "root_patch":
			patchTexts = append(patchTexts, event.PatchText)
			sourceEventIDs = append(sourceEventIDs, event.EventID)
		case "attempt_succeeded":
			if successText == "" && event.ErrorSummary != "" {
				successText = event.ErrorSummary
				sourceEventIDs = append(sourceEventIDs, event.EventID)
			}
		}
	}

	patches := dedupeNonEmpty(patchTexts, 5)
	successes := dedupeNonEmpty([]string{successText}, 1)
	if len(patches) == 0 && len(successes) == 0 {
		return nil
	}

	var b strings.Builder
	b.WriteString("## Root-Compiled Experience\n")
	b.WriteString("This bounded view is compiled from append-only audit history. Apply it, but do not restate it.\n")
	if len(patches) > 0 {
		b.WriteString("\nRecent corrective patches:\n")
		for i, patch := range patches {
			b.WriteString(fmt.Sprintf("%d. %s\n", i+1, truncateText(patch, 280)))
		}
	}
	if len(successes) > 0 {
		b.WriteString("\nRecent successful pattern:\n")
		for _, success := range successes {
			b.WriteString("- " + truncateText(success, 280) + "\n")
		}
	}

	view := CompiledExperienceView{
		ViewID:         "view:" + promptIdentity,
		PromptIdentity: promptIdentity,
		CapabilityID:   capabilityID,
		SystemPatch:    strings.TrimSpace(b.String()),
		SourceEventIDs: dedupeNonEmpty(sourceEventIDs, 6),
		MaxTokens:      320,
		CompiledAt:     time.Now().UTC().UnixMilli(),
		Version:        1,
	}
	_ = a.config.MemDB.SaveCompiledExperienceView(view)
	return &view
}

func (a *Agent) appendExperienceEvent(intent Intent, runID, traceID, spanID, epochID, eventType, errorCategory, errorSummary, errorDetail, patchText string, visibleToChild bool) {
	event := ExperienceEvent{
		EventID:        "exp_" + ulid.Make().String(),
		RunID:          runID,
		TraceID:        traceID,
		SpanID:         spanID,
		EpochID:        epochID,
		PromptIdentity: promptIdentityForIntent(intent),
		CapabilityID:   capabilityIDForIntent(intent),
		EventType:      eventType,
		ErrorCategory:  errorCategory,
		ErrorSummary:   truncateText(errorSummary, 280),
		ErrorDetail:    truncateText(errorDetail, 2000),
		PatchText:      truncateText(patchText, 600),
		VisibleToChild: visibleToChild,
		CreatedAt:      time.Now().UTC().UnixMilli(),
	}
	_ = a.config.MemDB.AppendExperienceEvent(event)
}

func (a *Agent) saveRunPatch(intent Intent, runID, spanID, source, patchText string) {
	patchText = strings.TrimSpace(patchText)
	if patchText == "" {
		return
	}
	patch := RunPatch{
		PatchID:        "patch_" + ulid.Make().String(),
		RunID:          runID,
		SpanID:         spanID,
		PromptIdentity: promptIdentityForIntent(intent),
		CapabilityID:   capabilityIDForIntent(intent),
		PatchText:      patchText,
		Source:         source,
		CreatedAt:      time.Now().UTC().UnixMilli(),
	}
	_ = a.config.MemDB.SaveRunPatch(patch)
}

func (a *Agent) dispatchTask(ctx context.Context, traceID string, exec *spanExecution, scope CerebellumScope) error {
	epoch := exec.nextEpoch
	exec.nextEpoch++
	a.markSpanQueued(traceID, exec.spanID, epoch)

	select {
	case a.taskChan <- TaskPayload{
		SpanID:    exec.spanID,
		EpochNum:  epoch,
		Scope:     scope,
		Engine:    exec.intent.Engine,
		Intent:    exec.intent,
		Ctx:       ctx,
		UserInput: exec.userInput,
	}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (a *Agent) appendForcedSuccess(spanID string, epochNum int, result string) {
	epoch := ExecutionEpoch{
		EpochID:     "epoch_" + ulid.Make().String(),
		SpanID:      spanID,
		EpochNumber: epochNum,
		Prompt:      "force_success",
		Result:      result,
		Category:    StatusSuccess,
		CreatedAt:   time.Now().UTC(),
	}
	_ = a.config.MemDB.AppendEpoch(epoch)
}

func (a *Agent) markSpanQueued(traceID, spanID string, epochNum int) {
	a.stateMu.Lock()
	defer a.stateMu.Unlock()

	run := a.runs[traceID]
	if run == nil {
		return
	}
	now := time.Now().UTC().UnixMilli()
	run.latestObservationAt = now
	if step, ok := run.visibleSteps[spanID]; ok {
		step.Status = "pending"
		step.UpdatedAt = now
		run.visibleSteps[spanID] = step
	}
	if span, ok := run.currentSpans[spanID]; ok {
		span.Status = "pending"
		span.ActiveEpoch = epochNum
		run.currentSpans[spanID] = span
	}
	if child, ok := run.childRuns[spanID]; ok {
		child.Status = "pending"
		child.UpdatedAt = now
		run.childRuns[spanID] = child
	}
}

func (a *Agent) markSpanRunning(spanID string, epochNum int) {
	a.stateMu.Lock()
	defer a.stateMu.Unlock()

	traceID := a.spanRun[spanID]
	run := a.runs[traceID]
	if run == nil {
		return
	}
	now := time.Now().UTC().UnixMilli()
	run.latestObservationAt = now
	if step, ok := run.visibleSteps[spanID]; ok {
		step.Status = "running"
		step.UpdatedAt = now
		run.visibleSteps[spanID] = step
	}
	if span, ok := run.currentSpans[spanID]; ok {
		span.Status = "running"
		span.ActiveEpoch = epochNum
		run.currentSpans[spanID] = span
	}
	if child, ok := run.childRuns[spanID]; ok {
		child.Status = "running"
		child.UpdatedAt = now
		run.childRuns[spanID] = child
	}
}

func (a *Agent) recordSpanSuccess(traceID, spanID string, epochNum int, result string) {
	var shouldEmit bool
	var incidentEvent IncidentEvent

	a.stateMu.Lock()
	run := a.runs[traceID]
	if run != nil {
		now := time.Now().UTC().UnixMilli()
		run.latestObservationAt = now
		if step, ok := run.visibleSteps[spanID]; ok {
			step.Status = "success"
			step.Detail = result
			step.UpdatedAt = now
			run.visibleSteps[spanID] = step
		}
		if span, ok := run.currentSpans[spanID]; ok {
			span.Status = "success"
			span.ActiveEpoch = epochNum
			span.FinalEpoch = epochNum
			run.currentSpans[spanID] = span
		}
		if child, ok := run.childRuns[spanID]; ok {
			child.Status = "success"
			child.UpdatedAt = now
			child.EndedAt = now
			run.childRuns[spanID] = child
		}
		if incidentID := run.incidentBySpan[spanID]; incidentID != "" {
			if incident, ok := run.activeIncidents[incidentID]; ok {
				incident.Status = "resolved"
				if incident.WinnerActor == "" {
					incident.WinnerActor = incident.CandidateActor
					if incident.WinnerActor == "" {
						incident.WinnerActor = "big"
					}
				}
				incident.ResolvedAt = now
				incidentEvent = IncidentEvent{
					EventType:      "incident_resolved",
					TraceID:        traceID,
					RequestID:      run.requestID,
					IncidentID:     incidentID,
					SpanID:         spanID,
					EpochNum:       epochNum,
					WinnerActor:    incident.WinnerActor,
					Status:         incident.Status,
					Reason:         incident.Reason,
					CandidateActor: incident.CandidateActor,
				}
				delete(run.activeIncidents, incidentID)
				delete(run.incidentBySpan, spanID)
				run.pendingApproval = nil
				shouldEmit = true
			}
		}
	}
	state := a.snapshotLocked(traceID)
	a.stateMu.Unlock()

	if shouldEmit && a.config.OnIncident != nil {
		a.config.OnIncident(incidentEvent)
	}
	a.persistSnapshot(context.Background(), traceID, state)
}

func (a *Agent) openIncident(ep ErrorPayload, candidateActor string, status string, approval bool) ActiveIncident {
	traceID := a.traceIDForSpan(ep.SpanID)
	now := time.Now().UTC().UnixMilli()
	incident := ActiveIncident{
		IncidentID:     NewID("incident"),
		SpanID:         ep.SpanID,
		EpochNum:       ep.EpochNum,
		Status:         status,
		CandidateActor: candidateActor,
		Reason:         ep.Envelope.RawStderr,
		ErrorDigest:    ep.Envelope.RawStderr,
		OpenedAt:       now,
	}

	var event IncidentEvent
	a.stateMu.Lock()
	run := a.runs[traceID]
	if run != nil {
		run.latestObservationAt = now
		run.activeIncidents[incident.IncidentID] = incident
		run.incidentBySpan[ep.SpanID] = incident.IncidentID
		if step, ok := run.visibleSteps[ep.SpanID]; ok {
			step.Status = "error"
			step.Detail = ep.Envelope.RawStderr
			step.UpdatedAt = now
			run.visibleSteps[ep.SpanID] = step
		}
		if span, ok := run.currentSpans[ep.SpanID]; ok {
			span.Status = "error"
			span.ActiveEpoch = ep.EpochNum
			run.currentSpans[ep.SpanID] = span
		}
		if approval {
			run.pendingApproval = &PendingApproval{
				ApprovalID: NewID("approval"),
				IncidentID: incident.IncidentID,
				SpanID:     ep.SpanID,
				Reason:     ep.Envelope.RawStderr,
				Actions:    []string{"resume", "abort", "force_success"},
				OpenedAt:   now,
			}
		}
		event = IncidentEvent{
			EventType:      "incident_opened",
			TraceID:        traceID,
			RequestID:      run.requestID,
			IncidentID:     incident.IncidentID,
			SpanID:         ep.SpanID,
			EpochNum:       ep.EpochNum,
			CandidateActor: candidateActor,
			ApprovalID:     approvalID(run.pendingApproval),
			Reason:         incident.Reason,
			ErrorDigest:    incident.ErrorDigest,
			Status:         incident.Status,
		}
	}
	state := a.snapshotLocked(traceID)
	a.stateMu.Unlock()

	if a.config.OnIncident != nil {
		a.config.OnIncident(event)
	}
	a.persistSnapshot(context.Background(), traceID, state)
	return incident
}

func (a *Agent) recordIncidentProgress(spanID, incidentID, candidateActor, status string) {
	traceID := a.traceIDForSpan(spanID)
	var event IncidentEvent

	a.stateMu.Lock()
	run := a.runs[traceID]
	if run != nil {
		now := time.Now().UTC().UnixMilli()
		run.latestObservationAt = now
		if incident, ok := run.activeIncidents[incidentID]; ok {
			incident.CandidateActor = candidateActor
			incident.Status = status
			run.activeIncidents[incidentID] = incident
			event = IncidentEvent{
				EventType:      "incident_progress",
				TraceID:        traceID,
				RequestID:      run.requestID,
				IncidentID:     incidentID,
				SpanID:         spanID,
				EpochNum:       incident.EpochNum,
				CandidateActor: candidateActor,
				Status:         status,
				Reason:         incident.Reason,
				ErrorDigest:    incident.ErrorDigest,
				ApprovalID:     approvalID(run.pendingApproval),
			}
		}
	}
	state := a.snapshotLocked(traceID)
	a.stateMu.Unlock()

	if a.config.OnIncident != nil && event.EventType != "" {
		a.config.OnIncident(event)
	}
	a.persistSnapshot(context.Background(), traceID, state)
}

func (a *Agent) recordIncidentResolved(traceID, incidentID, status, winnerActor string) {
	var event IncidentEvent

	a.stateMu.Lock()
	run := a.runs[traceID]
	if run != nil {
		now := time.Now().UTC().UnixMilli()
		run.latestObservationAt = now
		if incident, ok := run.activeIncidents[incidentID]; ok {
			incident.Status = status
			incident.WinnerActor = winnerActor
			incident.ResolvedAt = now
			event = IncidentEvent{
				EventType:      "incident_resolved",
				TraceID:        traceID,
				RequestID:      run.requestID,
				IncidentID:     incidentID,
				SpanID:         incident.SpanID,
				EpochNum:       incident.EpochNum,
				CandidateActor: incident.CandidateActor,
				WinnerActor:    winnerActor,
				Status:         status,
				Reason:         incident.Reason,
				ErrorDigest:    incident.ErrorDigest,
			}
			delete(run.activeIncidents, incidentID)
			delete(run.incidentBySpan, incident.SpanID)
			if run.pendingApproval != nil && run.pendingApproval.IncidentID == incidentID {
				run.pendingApproval = nil
			}
		}
	}
	state := a.snapshotLocked(traceID)
	a.stateMu.Unlock()

	if a.config.OnIncident != nil && event.EventType != "" {
		a.config.OnIncident(event)
	}
	a.persistSnapshot(context.Background(), traceID, state)
}

func (a *Agent) traceIDForSpan(spanID string) string {
	a.stateMu.Lock()
	defer a.stateMu.Unlock()
	return a.spanRun[spanID]
}

func (a *Agent) snapshotLocked(traceID string) *HibernationState {
	run := a.runs[traceID]
	if run == nil {
		return nil
	}

	state := &HibernationState{
		TraceID:             traceID,
		RequestID:           run.requestID,
		LatestObservationAt: run.latestObservationAt,
		CreatedAt:           run.createdAt,
		PendingApproval:     run.pendingApproval,
		ControllerState:     run.controllerState,
		ActivePlanRev:       run.activePlanRev,
		QuiescingStartedAt:  run.quiescingStartedAt,
		LastSummaryAt:       run.lastSummaryAt,
		RevisionReason:      run.revisionReason,
	}
	for _, stepID := range run.stepOrder {
		if step, ok := run.visibleSteps[stepID]; ok {
			state.VisibleSteps = append(state.VisibleSteps, step)
		}
		if child, ok := run.childRuns[stepID]; ok {
			state.ChildRuns = append(state.ChildRuns, child)
		}
		if span, ok := run.currentSpans[stepID]; ok {
			state.CurrentSpans = append(state.CurrentSpans, span)
		}
		if incidentID := run.incidentBySpan[stepID]; incidentID != "" {
			if incident, ok := run.activeIncidents[incidentID]; ok {
				state.ActiveIncidents = append(state.ActiveIncidents, incident)
			}
		}
	}
	return state
}

func (a *Agent) persistSnapshot(ctx context.Context, traceID string, state *HibernationState) {
	if a.config.StateStore == nil {
		return
	}
	if state == nil {
		_ = a.config.StateStore.Clear(ctx, traceID)
		return
	}
	switch state.ControllerState {
	case "done", "failed", "estopped":
		_ = a.config.StateStore.Clear(ctx, traceID)
		return
	}
	if len(state.ActiveIncidents) == 0 && state.PendingApproval == nil && state.ControllerState == "" && len(state.VisibleSteps) == 0 {
		_ = a.config.StateStore.Clear(ctx, traceID)
		return
	}
	_ = a.config.StateStore.Hibernate(ctx, traceID, *state)
}

func (a *Agent) transitionControllerPhase(traceID string, intent Intent, phase string) {
	a.stateMu.Lock()
	run := a.runs[traceID]
	if run != nil {
		now := time.Now().UTC().UnixMilli()
		run.controllerState = phase
		run.latestObservationAt = now
		switch phase {
		case "quiescing":
			run.quiescingStartedAt = now
		case "summarizing":
			run.lastSummaryAt = now
		}
	}
	state := a.snapshotLocked(traceID)
	a.stateMu.Unlock()

	if a.config.OnControllerState != nil {
		a.config.OnControllerState(traceID, intent, phase)
	}
	a.persistSnapshot(context.Background(), traceID, state)
}

func (a *Agent) enterStableCompletion(ctx context.Context, traceID string, intent Intent) error {
	a.transitionControllerPhase(traceID, intent, "quiescing")
	a.transitionControllerPhase(traceID, intent, "summarizing")
	return nil
}

func approvalID(approval *PendingApproval) string {
	if approval == nil {
		return ""
	}
	return approval.ApprovalID
}

func (a *Agent) pendingApprovalID(traceID string) string {
	a.stateMu.Lock()
	defer a.stateMu.Unlock()

	run := a.runs[traceID]
	if run == nil {
		return ""
	}
	return approvalID(run.pendingApproval)
}

// worker runs as a Cerebellum goroutine. Stateless, pure execution.
func (a *Agent) worker(ctx context.Context, id int) {
	defer a.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case task, ok := <-a.taskChan:
			if !ok {
				return
			}
			a.executeTask(ctx, task)
		}
	}
}

func (a *Agent) executeTask(ctx context.Context, task TaskPayload) {
	start := time.Now()
	a.markSpanRunning(task.SpanID, task.EpochNum)

	execCtx := task.Ctx
	if execCtx == nil {
		execCtx = ctx
	}

	// Lifecycle hook: task starting
	if a.config.OnTaskStart != nil {
		a.config.OnTaskStart(task.SpanID, task.Intent)
	}

	// --- Three-segment scope hydration (framework work, zero model cost) ---
	scope := task.Scope
	runID := a.traceIDForSpan(task.SpanID)

	// Segment 1: SystemPrompt — assembled at runtime, not loaded from Intent memory.
	if scope.SystemPrompt == "" {
		if a.config.SystemPromptProvider != nil {
			scope.SystemPrompt = strings.TrimSpace(a.config.SystemPromptProvider(task.Intent))
		}
		if scope.SystemPrompt == "" {
			scope.SystemPrompt = a.config.DefaultSystemPrompt
		}
	}

	// Segment 2: ToolSchema — static, from AgentConfig (never modified by framework)
	if scope.ToolSchema == nil {
		scope.ToolSchema = a.config.ToolSchemas
	}

	// Segment 3: TaskPrompt + EnvMetadata — assembled per invocation
	if scope.TaskPrompt == "" {
		if task.Intent.Action == "agent_turn" {
			// agent_turn: Target is the raw user message, used directly
			scope.TaskPrompt = task.Intent.Target
		} else {
			scope.TaskPrompt = fmt.Sprintf("%s: %s", task.Intent.Action, task.Intent.Target)
		}
	}
	if scope.EnvMetadata == "" {
		env := a.config.MemDB.LoadSystemEvolution()
		scope.EnvMetadata = fmt.Sprintf("os=%s, arch=%s, shell=%s",
			env.OSVersion, env.Architecture, env.Shell)
	}
	if view := a.compileExperienceView(task.Intent, runID, task.SpanID); view != nil && view.SystemPatch != "" {
		if scope.SystemPrompt != "" {
			scope.SystemPrompt += "\n\n"
		}
		scope.SystemPrompt += view.SystemPatch
	}
	a.appendExperienceEvent(task.Intent, runID, runID, task.SpanID, "", "attempt_started", "", scope.TaskPrompt, "", "", false)

	var result string
	var err error

	if len(a.config.toolMap) > 0 && a.config.Executor == nil {
		result, err = a.executeToolLoop(execCtx, scope, task)
	} else {
		// Legacy path: single ExecutorFunc call
		result, err = a.config.Executor(execCtx, scope.SystemPrompt+"\n"+scope.TaskPrompt, task.Intent)
	}

	if err != nil && contextCause(execCtx) != nil {
		if a.config.OnTaskComplete != nil {
			a.config.OnTaskComplete(task.SpanID, task.Intent, result, err)
		}
		return
	}

	duration := time.Since(start).Milliseconds()

	epochID := "epoch_" + ulid.Make().String()
	epoch := ExecutionEpoch{
		EpochID:     epochID,
		SpanID:      task.SpanID,
		EpochNumber: task.EpochNum,
		Prompt:      scope.TaskPrompt,
		CreatedAt:   time.Now().UTC(),
		DurationMs:  duration,
	}

	if err != nil {
		envelope := ExecutionEnvelope{
			StatusCode: 500,
			Category:   ErrUnknown,
			RawStdout:  result,
			RawStderr:  err.Error(),
			ExitCode:   1,
		}

		epoch.Error = err.Error()
		epoch.Category = envelope.Category
		epoch.IsDeadEnd = true
		epoch.Result = result
		a.config.MemDB.AppendEpoch(epoch)
		a.appendExperienceEvent(task.Intent, runID, runID, task.SpanID, epochID, "attempt_failed", string(envelope.Category), envelope.RawStderr, envelope.RawStderr, "", false)

		a.errorChan <- ErrorPayload{
			SpanID:       task.SpanID,
			EpochNum:     task.EpochNum,
			Envelope:     envelope,
			Prompt:       scope.TaskPrompt,
			SystemPrompt: scope.SystemPrompt,
			Intent:       task.Intent,
		}

		if a.config.OnTaskComplete != nil {
			a.config.OnTaskComplete(task.SpanID, task.Intent, result, err)
		}
		return
	}

	epoch.Result = result
	epoch.Category = StatusSuccess
	a.config.MemDB.AppendEpoch(epoch)

	a.resultChan <- ResultPayload{
		SpanID:   task.SpanID,
		EpochNum: task.EpochNum,
		Stdout:   result,
		Envelope: ExecutionEnvelope{
			StatusCode: 200,
			Category:   StatusSuccess,
			RawStdout:  result,
			ExitCode:   0,
		},
	}

	if a.config.Collector != nil {
		a.config.Collector.OnEpoch(epoch)
	}

	if a.config.OnTaskComplete != nil {
		a.config.OnTaskComplete(task.SpanID, task.Intent, result, nil)
	}
}

func (a *Agent) dispatchPlannedChild(ctx context.Context, traceID string, exec *spanExecution, scope CerebellumScope) error {
	if a.config.PlannedChildRunner == nil {
		return a.dispatchTask(ctx, traceID, exec, scope)
	}
	task := TaskPayload{
		SpanID:   exec.spanID,
		EpochNum: exec.nextEpoch,
		Scope:    scope,
		Intent:   exec.intent,
	}
	exec.nextEpoch++
	a.wg.Add(1)
	go func() {
		defer a.wg.Done()
		a.childSem <- struct{}{}        // acquire semaphore
		defer func() { <-a.childSem }() // release semaphore
		start := time.Now()
		a.markSpanRunning(task.SpanID, task.EpochNum)
		if a.config.OnTaskStart != nil {
			a.config.OnTaskStart(task.SpanID, task.Intent)
		}

		runID := a.traceIDForSpan(task.SpanID)
		scope := task.Scope
		if scope.SystemPrompt == "" {
			if a.config.SystemPromptProvider != nil {
				scope.SystemPrompt = strings.TrimSpace(a.config.SystemPromptProvider(task.Intent))
			}
			if scope.SystemPrompt == "" {
				scope.SystemPrompt = a.config.DefaultSystemPrompt
			}
		}
		if scope.TaskPrompt == "" {
			scope.TaskPrompt = task.Intent.Target
		}
		a.appendExperienceEvent(task.Intent, runID, runID, task.SpanID, "", "attempt_started", "", scope.TaskPrompt, "", "", false)

		result, err := a.config.PlannedChildRunner(ctx, PlannedChildRequest{
			TraceID: traceID,
			SpanID:  task.SpanID,
			Intent:  task.Intent,
			Scope:   scope,
		})

		duration := time.Since(start).Milliseconds()
		epochID := "epoch_" + ulid.Make().String()
		epoch := ExecutionEpoch{
			EpochID:     epochID,
			SpanID:      task.SpanID,
			EpochNumber: task.EpochNum,
			Prompt:      scope.TaskPrompt,
			CreatedAt:   time.Now().UTC(),
			DurationMs:  duration,
		}
		if err != nil {
			envelope := ExecutionEnvelope{
				StatusCode: 500,
				Category:   ErrUnknown,
				RawStdout:  result,
				RawStderr:  err.Error(),
				ExitCode:   1,
			}
			epoch.Error = err.Error()
			epoch.Category = envelope.Category
			epoch.IsDeadEnd = true
			epoch.Result = result
			a.config.MemDB.AppendEpoch(epoch)
			a.appendExperienceEvent(task.Intent, runID, runID, task.SpanID, epochID, "attempt_failed", string(envelope.Category), envelope.RawStderr, envelope.RawStderr, "", false)
			a.errorChan <- ErrorPayload{
				SpanID:       task.SpanID,
				EpochNum:     task.EpochNum,
				Envelope:     envelope,
				Prompt:       scope.TaskPrompt,
				SystemPrompt: scope.SystemPrompt,
				Intent:       task.Intent,
			}
			if a.config.OnTaskComplete != nil {
				a.config.OnTaskComplete(task.SpanID, task.Intent, result, err)
			}
			return
		}

		epoch.Result = result
		epoch.Category = StatusSuccess
		a.config.MemDB.AppendEpoch(epoch)
		a.resultChan <- ResultPayload{
			SpanID:   task.SpanID,
			EpochNum: task.EpochNum,
			Stdout:   result,
			Envelope: ExecutionEnvelope{
				StatusCode: 200,
				Category:   StatusSuccess,
				RawStdout:  result,
				ExitCode:   0,
			},
		}
		if a.config.OnTaskComplete != nil {
			a.config.OnTaskComplete(task.SpanID, task.Intent, result, nil)
		}
	}()
	return nil
}

// executeToolLoop implements the LLM -> tool_use -> execute -> tool_result -> LLM loop.
// This is the preferred execution path when Tools are registered via WithTools().
// The loop continues until the LLM produces a final text response with no tool_use blocks.
func (a *Agent) executeToolLoop(ctx context.Context, scope CerebellumScope, task TaskPayload) (string, error) {
	if a.config.ModelConfig == nil {
		return "", fmt.Errorf("ModelConfig is required for tool-use loop")
	}

	// Build tool schemas for LLM request
	availableTools := a.toolsForIntent(task.Intent)
	toolSchemas := make([]ToolCallSchema, 0, len(availableTools))
	toolMap := make(map[string]Tool, len(availableTools))
	runID := a.traceIDForSpan(task.SpanID)
	for _, t := range availableTools {
		toolMap[t.Name()] = t
		toolSchemas = append(toolSchemas, ToolCallSchema{
			Type: "function",
			Function: ToolCallFunction{
				Name:        t.Name(),
				Description: t.Description(),
				Parameters:  t.InputSchema(),
			},
		})
	}

	messages := []ChatMessage{
		{Role: "system", Content: TextContent(scope.SystemPrompt)},
		{Role: "user", Content: buildUserChatMessageContent(task, scope.TaskPrompt)},
	}

	const maxLoopIterations = 64
	for i := 0; i < maxLoopIterations; i++ {
		if err := contextCause(ctx); err != nil {
			return "", err
		}

		requestBody := ToolCompletionRequest{
			Model:       a.config.ModelConfig.Cerebellum.ModelID,
			Messages:    messages,
			Tools:       toolSchemas,
			MaxTokens:   a.config.ModelConfig.Cerebellum.MaxTokens,
			Temperature: a.config.ModelConfig.Cerebellum.Temperature,
			Stream:      true,
		}
		requestJSON, _ := json.Marshal(requestBody)
		llmStart := time.Now()
		resp, err := a.callLLMWithTools(ctx, messages, toolSchemas, func(text string) {
			if a.config.OnLLMChunk != nil && text != "" {
				a.config.OnLLMChunk(task.SpanID, task.Intent, "executing", text)
			}
		})
		llmDuration := time.Since(llmStart)
		if err != nil {
			if a.config.OnLLMCall != nil {
				a.config.OnLLMCall(task.SpanID, task.Intent, LLMCallEvent{
					Phase:        "executing",
					Iteration:    i,
					ModelID:      a.config.ModelConfig.Cerebellum.ModelID,
					MessageCount: len(messages),
					DurationMs:   llmDuration.Milliseconds(),
					Request:      string(requestJSON),
					Error:        err.Error(),
				})
			}
			log.Printf("[llm] call failed (iter=%d, span=%s, dur=%dms): %v", i, task.SpanID, llmDuration.Milliseconds(), err)
			return "", fmt.Errorf("llm call (iteration %d): %w", i, err)
		}
		if a.config.OnLLMCall != nil {
			responsePreview, _ := json.Marshal(map[string]any{
				"content":       resp.Content,
				"tool_calls":    resp.ToolCalls,
				"finish_reason": resp.FinishReason,
				"usage": map[string]int{
					"prompt_tokens":     resp.Usage.PromptTokens,
					"completion_tokens": resp.Usage.CompletionTokens,
					"total_tokens":      resp.Usage.TotalTokens,
				},
			})
			a.config.OnLLMCall(task.SpanID, task.Intent, LLMCallEvent{
				Phase:            "executing",
				Iteration:        i,
				ModelID:          a.config.ModelConfig.Cerebellum.ModelID,
				MessageCount:     len(messages),
				PromptTokens:     resp.Usage.PromptTokens,
				CompletionTokens: resp.Usage.CompletionTokens,
				TotalTokens:      resp.Usage.TotalTokens,
				DurationMs:       llmDuration.Milliseconds(),
				FinishReason:     resp.FinishReason,
				ToolCalls:        len(resp.ToolCalls),
				Request:          string(requestJSON),
				Response:         string(responsePreview),
			})
		}

		log.Printf("[llm] call ok (iter=%d, span=%s, model=%s, prompt_tok=%d, comp_tok=%d, total_tok=%d, finish=%s, tool_calls=%d, dur=%dms)",
			i, task.SpanID, a.config.ModelConfig.Cerebellum.ModelID,
			resp.Usage.PromptTokens, resp.Usage.CompletionTokens, resp.Usage.TotalTokens,
			resp.FinishReason, len(resp.ToolCalls), llmDuration.Milliseconds())

		if len(resp.ToolCalls) == 0 {
			log.Printf("[llm] final response (iter=%d, span=%s, len=%d)", i, task.SpanID, len(resp.Content))
			return resp.Content, nil
		}

		// Append assistant message with tool_use
		messages = append(messages, ChatMessage{
			Role:      "assistant",
			Content:   TextContent(resp.Content),
			ToolCalls: resp.ToolCalls,
		})

		// Execute each tool call and collect results
		for _, tc := range resp.ToolCalls {
			if err := contextCause(ctx); err != nil {
				return "", err
			}
			tool, ok := toolMap[tc.Function.Name]
			if !ok {
				log.Printf("[tool] unknown tool %q (span=%s, call_id=%s)", tc.Function.Name, task.SpanID, tc.ID)
				messages = append(messages, ChatMessage{
					Role:       "tool",
					Content:    TextContent(fmt.Sprintf("error: unknown tool %q", tc.Function.Name)),
					ToolCallID: tc.ID,
				})
				continue
			}

			log.Printf("[tool] exec start %s (span=%s, call_id=%s, args=%s)", tc.Function.Name, task.SpanID, tc.ID, truncate(tc.Function.Arguments, 200))
			toolStart := time.Now()
			toolCtx := WithToolExecutionMeta(ctx, ToolExecutionMeta{
				TraceID:      runID,
				SpanID:       task.SpanID,
				StepID:       firstNonEmpty(readIntentParam(task.Intent, "plan_step_id"), task.SpanID),
				IntentAction: task.Intent.Action,
			})
			toolResult, toolErr := tool.Execute(toolCtx, []byte(tc.Function.Arguments))
			toolDuration := time.Since(toolStart)
			resultContent := toolResult.Content
			if toolErr != nil {
				resultContent = fmt.Sprintf("error: %v", toolErr)
				log.Printf("[tool] exec error %s (span=%s, call_id=%s, dur=%dms): %v", tc.Function.Name, task.SpanID, tc.ID, toolDuration.Milliseconds(), toolErr)
			} else {
				log.Printf("[tool] exec done %s (span=%s, call_id=%s, dur=%dms, result_len=%d, is_error=%v)", tc.Function.Name, task.SpanID, tc.ID, toolDuration.Milliseconds(), len(resultContent), toolResult.IsError)
			}
			if a.config.OnToolCall != nil {
				a.config.OnToolCall(task.SpanID, task.Intent, ToolExecutionEvent{
					ToolName:   tc.Function.Name,
					ToolCallID: tc.ID,
					Arguments:  tc.Function.Arguments,
					Result:     resultContent,
					IsError:    toolErr != nil || toolResult.IsError,
					DurationMs: toolDuration.Milliseconds(),
				})
			}
			if err := contextCause(ctx); err != nil {
				return "", err
			}

			messages = append(messages, ChatMessage{
				Role:       "tool",
				Content:    TextContent(resultContent),
				ToolCallID: tc.ID,
			})
		}
	}

	return "", fmt.Errorf("tool-use loop exceeded %d iterations", maxLoopIterations)
}

func contextCause(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	select {
	case <-ctx.Done():
		if cause := context.Cause(ctx); cause != nil {
			return cause
		}
		return ctx.Err()
	default:
		return nil
	}
}

func inputForIntent(intent Intent, objective string, input UserInput) UserInput {
	if len(input.Images) == 0 {
		return UserInput{Text: intent.Target}
	}
	if strings.TrimSpace(intent.Target) == strings.TrimSpace(objective) {
		return input
	}
	return UserInput{Text: intent.Target}
}

func buildUserChatMessageContent(task TaskPayload, taskPrompt string) ChatMessageContent {
	if len(task.UserInput.Images) == 0 {
		return TextContent(taskPrompt)
	}
	parts := make([]ChatContentPart, 0, len(task.UserInput.Images)+1)
	if text := strings.TrimSpace(taskPrompt); text != "" {
		parts = append(parts, ChatContentPart{
			Type: "text",
			Text: text,
		})
	}
	for _, image := range task.UserInput.Images {
		parts = append(parts, ChatContentPart{
			Type: "image_url",
			ImageURL: &ChatImageURLPart{
				URL: fmt.Sprintf("data:%s;base64,%s", image.MimeType, image.Data),
			},
		})
	}
	if len(parts) == 0 {
		return TextContent("")
	}
	return PartsContent(parts...)
}

// callLLMWithTools sends a chat completion request with tool definitions.
// Returns the assistant's response including any tool_use blocks.
func (a *Agent) callLLMWithTools(ctx context.Context, messages []ChatMessage, tools []ToolCallSchema, onDelta func(text string)) (*LLMToolResponse, error) {
	return a.callLLMWithToolsMode(ctx, messages, tools, onDelta, true)
}

func (a *Agent) callLLMWithToolsMode(ctx context.Context, messages []ChatMessage, tools []ToolCallSchema, onDelta func(text string), stream bool) (*LLMToolResponse, error) {
	cfg := a.config.ModelConfig.Cerebellum

	reqBody := ToolCompletionRequest{
		Model:       cfg.ModelID,
		Messages:    messages,
		Tools:       tools,
		MaxTokens:   cfg.MaxTokens,
		Temperature: cfg.Temperature,
		Stream:      stream,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	url := cfg.BaseURL + "/v1/chat/completions"
	if url == "/v1/chat/completions" {
		return nil, fmt.Errorf("ModelConfig.Cerebellum.BaseURL is required for tool-use loop")
	}

	req, err := newHTTPRequest(ctx, "POST", url, data, cfg.APIKeyEnv)
	if err != nil {
		return nil, err
	}

	resp, err := toolHTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, err := readBody(resp.Body)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncate(string(body), 500))
	}

	if stream && strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
		result, err := parseToolCompletionStream(resp.Body, onDelta)
		if err != nil {
			return nil, err
		}
		if result != nil && (len(result.ToolCalls) > 0 || result.Content != "") {
			return result, nil
		}
		return result, nil
	}

	body, err := readBody(resp.Body)
	if err != nil {
		return nil, err
	}

	var result ToolCompletionResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}
	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("empty choices")
	}

	choice := result.Choices[0]
	llmResp := &LLMToolResponse{
		Content:      choice.Message.Content,
		ToolCalls:    choice.Message.ToolCalls,
		FinishReason: choice.FinishReason,
	}
	llmResp.Usage.PromptTokens = result.Usage.PromptTokens
	llmResp.Usage.CompletionTokens = result.Usage.CompletionTokens
	llmResp.Usage.TotalTokens = result.Usage.TotalTokens
	return llmResp, nil
}

// triageLoop runs the Triage goroutine — independent lifecycle from Brain.
func (a *Agent) triageLoop(ctx context.Context) {
	defer a.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case ep, ok := <-a.errorChan:
			if !ok {
				return
			}
			a.handleTriageError(ctx, ep)
		}
	}
}

func (a *Agent) handleTriageError(ctx context.Context, ep ErrorPayload) {
	action := ep.Intent.Action
	failCount := ep.EpochNum

	// Permission errors → immediate HITL
	if ep.Envelope.Category == ErrPermission {
		incident := a.openIncident(ep, "human", "pending_human", true)
		a.recordIncidentProgress(ep.SpanID, incident.IncidentID, "human", "pending_human")
		traceID := a.traceIDForSpan(ep.SpanID)
		a.hitlChan <- HITLPayload{
			SpanID:         ep.SpanID,
			EpochNum:       ep.EpochNum,
			LastError:      ep.Envelope,
			TriageMemo:     "Permission denied — cannot self-iterate",
			Intent:         ep.Intent,
			IncidentID:     incident.IncidentID,
			CandidateActor: "human",
			ApprovalID:     a.pendingApprovalID(traceID),
			AllowedActions: []string{"resume", "abort", "force_success"},
		}
		return
	}

	// Within self-iteration budget?
	// Optimizer receives a structured ExecutionError with full detail + current SystemPrompt.
	if failCount <= a.config.MaxSelfIterations && a.config.Optimizer != nil {
		incident := a.openIncident(ep, "big", "debugging", false)
		a.recordIncidentProgress(ep.SpanID, incident.IncidentID, "big", "debugging")
		execErr := NewExecutionError(ep.Envelope, ep.Intent, ep.SpanID, ep.EpochNum, failCount, ep.SystemPrompt)
		optResult, optErr := a.config.Optimizer(ctx, execErr, ep.SystemPrompt, ep.Intent)
		if optErr == nil && optResult.ShouldRetry {
			newHints := firstNonEmpty(optResult.CacheUpdate, optResult.NewPrompt)
			a.optimizeChan <- OptimizePayload{
				IntentAction:   action,
				PromptIdentity: promptIdentityForIntent(ep.Intent),
				CapabilityID:   capabilityIDForIntent(ep.Intent),
				NewSystemHints: newHints,
				ShouldRetry:    true,
				SpanID:         ep.SpanID,
				IncidentID:     incident.IncidentID,
				CandidateActor: "big",
			}
			return
		}
	}

	// Escalate to HITL
	incident := a.openIncident(ep, "human", "pending_human", true)
	a.recordIncidentProgress(ep.SpanID, incident.IncidentID, "human", "pending_human")
	traceID := a.traceIDForSpan(ep.SpanID)
	a.hitlChan <- HITLPayload{
		SpanID:         ep.SpanID,
		EpochNum:       ep.EpochNum,
		LastError:      ep.Envelope,
		TriageMemo:     fmt.Sprintf("Self-iteration exhausted after %d attempts", failCount),
		Intent:         ep.Intent,
		IncidentID:     incident.IncidentID,
		CandidateActor: "human",
		ApprovalID:     a.pendingApprovalID(traceID),
		AllowedActions: []string{"resume", "abort", "force_success"},
	}
}

// Stop gracefully shuts down the agent and releases resources.
func (a *Agent) Stop() {
	a.cancel()
	close(a.taskChan)
	a.wg.Wait()
	if a.config.ownsDB && a.config.MemDB != nil {
		a.config.MemDB.Close()
	}
}

// TraceID returns the current trace identifier.
func (a *Agent) TraceID() string {
	return a.traceID
}

// Hibernate persists current state to disk for process restart recovery.
func (a *Agent) Hibernate(ctx context.Context) error {
	if a.config.StateStore == nil {
		return fmt.Errorf("no StateStore configured")
	}
	a.stateMu.Lock()
	state := a.snapshotLocked(a.traceID)
	a.stateMu.Unlock()
	if state == nil {
		return fmt.Errorf("no pending state for trace %s", a.traceID)
	}
	return a.config.StateStore.Hibernate(ctx, a.traceID, *state)
}

// ListPendingStates returns persisted incident snapshots in newest-first order.
func (a *Agent) ListPendingStates(ctx context.Context) ([]HibernationState, error) {
	if a.config.StateStore == nil {
		return nil, fmt.Errorf("no StateStore configured")
	}
	return a.config.StateStore.ListPending(ctx)
}

// LatestPendingState returns the newest persisted snapshot, if any.
func (a *Agent) LatestPendingState(ctx context.Context) (*HibernationState, error) {
	states, err := a.ListPendingStates(ctx)
	if err != nil {
		return nil, err
	}
	if len(states) == 0 {
		return nil, nil
	}
	return &states[0], nil
}

// ULID generation helper (re-exported for convenience).
func NewID(prefix string) string {
	return prefix + "_" + ulid.Make().String()
}

// MarshalState serializes HibernationState to JSON.
func MarshalState(state HibernationState) ([]byte, error) {
	return json.Marshal(state)
}
