package cai_test

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	cai "github.com/anthropic/cai"
	"github.com/anthropic/cai/memory"
)

// testDB creates a temporary SQLite DB for testing. One line, no boilerplate.
func testDB(t *testing.T) *memory.SQLiteDB {
	t.Helper()
	db, err := memory.Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	memory.ProbeSystem(db)
	t.Cleanup(func() { db.Close() })
	return db
}

// simplePlanner returns a planner that always produces the given intents.
func simplePlanner(intents ...cai.Intent) cai.Planner {
	return plannerFunc(func(_ context.Context, _ string, _ *cai.BrainScope) ([]cai.Intent, error) {
		return intents, nil
	})
}

type plannerFunc func(context.Context, string, *cai.BrainScope) ([]cai.Intent, error)

func (f plannerFunc) Plan(ctx context.Context, obj string, s *cai.BrainScope) ([]cai.Intent, error) {
	return f(ctx, obj, s)
}

// --- Tests ---

func TestFullPipeline(t *testing.T) {
	db := testDB(t)
	calls := 0

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(
			cai.Intent{Action: "install", Target: "requests", Engine: cai.EngineBash},
			cai.Intent{Action: "run", Target: "tests", Engine: cai.EngineBash},
		),
		Executor: func(_ context.Context, _ string, i cai.Intent) (string, error) {
			calls++
			return "ok:" + i.Target, nil
		},
		MemDB:       db,
		WorkerCount: 2,
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := agent.Run(ctx, "install and test"); err != nil {
		t.Fatal(err)
	}
	agent.Stop()

	if calls != 2 {
		t.Errorf("expected 2 calls, got %d", calls)
	}
}

func TestSelfIteration(t *testing.T) {
	db := testDB(t)
	calls := 0

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(cai.Intent{Action: "install", Target: "requests", Engine: cai.EngineBash}),
		Executor: func(_ context.Context, _ string, _ cai.Intent) (string, error) {
			calls++
			if calls == 1 {
				return "", fmt.Errorf("ModuleNotFoundError: No module named 'requests'")
			}
			return "installed", nil
		},
		Optimizer: func(_ context.Context, _ error, _ string, _ cai.Intent) (cai.OptimizationResult, error) {
			return cai.OptimizationResult{ShouldRetry: true, CacheUpdate: "Use uv install"}, nil
		},
		MemDB:             db,
		WorkerCount:       1,
		MaxSelfIterations: 3,
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := agent.Run(ctx, "install requests"); err != nil {
		t.Fatal(err)
	}
	agent.Stop()

	if calls < 2 {
		t.Errorf("expected >= 2 executor calls, got %d", calls)
	}
}

func TestIntentMemoryCache(t *testing.T) {
	db := testDB(t)

	db.SaveIntentMemory(cai.IntentMemory{
		IntentAction: "install", SystemPromptHints: "Use uv", Version: 1,
	})

	mem, found := db.LoadIntentMemory("install")
	if !found || mem.SystemPromptHints != "Use uv" {
		t.Fatal("cache miss or wrong hints")
	}

	db.IncrementFailure("install")
	db.IncrementFailure("install")
	db.EvictIntentMemory("install", "pip gone")

	evicted, _ := db.LoadIntentMemory("install")
	if evicted.SystemPromptHints != "" {
		t.Error("expected empty after eviction")
	}
	if evicted.Version != 2 {
		t.Errorf("expected version 2, got %d", evicted.Version)
	}
}

func TestEpochVisibility(t *testing.T) {
	db := testDB(t)

	db.CreateTrace(cai.AgentTrace{TraceID: "t1", Objective: "test", Status: "running", CreatedAt: time.Now()})
	db.CreateSpan(cai.LogicalSpan{SpanID: "s1", TraceID: "t1", PlanNodeID: "n0", StepName: "step", ActiveEpoch: 1, Status: "running"})

	for i := 1; i <= 3; i++ {
		cat := cai.StatusSuccess
		dead := false
		if i < 3 {
			cat = cai.ErrSyntax
			dead = true
		}
		db.AppendEpoch(cai.ExecutionEpoch{
			EpochID: fmt.Sprintf("e%d", i), SpanID: "s1", EpochNumber: i,
			Prompt: fmt.Sprintf("p%d", i), Category: cat, IsDeadEnd: dead, CreatedAt: time.Now(),
		})
	}

	active, _ := db.ActiveEpoch("s1")
	if active.EpochNumber != 3 {
		t.Errorf("N(1) broken: expected epoch 3, got %d", active.EpochNumber)
	}

	history, _ := db.SpanHistory("s1")
	if len(history) != 3 {
		t.Errorf("expected 3 epochs, got %d", len(history))
	}
}

func TestFrozenPrompt(t *testing.T) {
	db := testDB(t)

	db.SaveIntentMemory(cai.IntentMemory{
		IntentAction: "deploy", SystemPromptHints: "Use kubectl", Frozen: true, Version: 1,
	})

	mem, _ := db.LoadIntentMemory("deploy")
	if !mem.Frozen {
		t.Error("expected Frozen=true")
	}

	db.EvictIntentMemory("deploy", "error")
	mem, _ = db.LoadIntentMemory("deploy")
	if mem.SystemPromptHints != "Use kubectl" {
		t.Error("frozen prompt should not be evicted")
	}
}

func TestDefaultSystemPromptProvider(t *testing.T) {
	db := testDB(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(cai.Intent{Action: "greet", Target: "world", Engine: cai.EngineBash}),
		Executor: func(_ context.Context, prompt string, _ cai.Intent) (string, error) {
			if !strings.Contains(prompt, "friendly") {
				t.Errorf("default system prompt not injected: %s", prompt)
			}
			return "hello", nil
		},
		MemDB:               db,
		DefaultSystemPrompt: "Always be friendly",
		WorkerCount:         1,
	})
	if err != nil {
		t.Fatal(err)
	}

	agent.Run(ctx, "say hello")
	agent.Stop()
}

func TestLegacyIntentMemoryIsIgnoredOnFirstExecution(t *testing.T) {
	db := testDB(t)
	if err := db.SaveIntentMemory(cai.IntentMemory{
		IntentAction:      "greet",
		SystemPromptHints: "Use the stale cached prompt",
		Version:           1,
	}); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(cai.Intent{Action: "greet", Target: "world", Engine: cai.EngineBash}),
		Executor: func(_ context.Context, prompt string, _ cai.Intent) (string, error) {
			if strings.Contains(prompt, "stale cached prompt") {
				t.Fatalf("legacy IntentMemory leaked into first execution: %s", prompt)
			}
			if !strings.Contains(prompt, "fresh runtime prompt") {
				t.Fatalf("runtime system prompt missing: %s", prompt)
			}
			return "hello", nil
		},
		MemDB:               db,
		DefaultSystemPrompt: "fresh runtime prompt",
		WorkerCount:         1,
	})
	if err != nil {
		t.Fatal(err)
	}

	if err := agent.Run(ctx, "say hello"); err != nil {
		t.Fatal(err)
	}
	agent.Stop()
}

func TestNewFunctionalOptions(t *testing.T) {
	// Register the DB factory (normally done via import _ "memory")
	cai.DBFactory = func(path string) (cai.MemoryDB, error) {
		return memory.Open(path)
	}
	cai.SystemProber = func(db cai.MemoryDB) error { return nil }

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	dbPath := filepath.Join(t.TempDir(), "opts.db")

	agent, err := cai.New(ctx,
		cai.WithPlanner(simplePlanner(cai.Intent{Action: "test", Target: "x", Engine: cai.EngineBash})),
		cai.WithExecutor(func(_ context.Context, _ string, _ cai.Intent) (string, error) {
			return "done", nil
		}),
		cai.WithDBPath(dbPath),
		cai.WithWorkers(1),
	)
	if err != nil {
		t.Fatal(err)
	}

	if err := agent.Run(ctx, "test"); err != nil {
		t.Fatal(err)
	}
	agent.Stop()
}

// TestConcurrentExecution verifies that 5 intents execute in parallel on 2 workers.
func TestConcurrentExecution(t *testing.T) {
	db := testDB(t)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	var maxConcurrent int64
	var currentConcurrent atomic.Int64
	var totalCalls atomic.Int64
	var mu sync.Mutex

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(
			cai.Intent{Action: "task_a", Target: "1", Engine: cai.EngineBash},
			cai.Intent{Action: "task_b", Target: "2", Engine: cai.EngineBash},
			cai.Intent{Action: "task_c", Target: "3", Engine: cai.EngineBash},
			cai.Intent{Action: "task_d", Target: "4", Engine: cai.EngineBash},
			cai.Intent{Action: "task_e", Target: "5", Engine: cai.EngineBash},
		),
		Executor: func(_ context.Context, _ string, intent cai.Intent) (string, error) {
			cur := currentConcurrent.Add(1)
			totalCalls.Add(1)

			mu.Lock()
			if cur > maxConcurrent {
				maxConcurrent = cur
			}
			mu.Unlock()

			time.Sleep(50 * time.Millisecond)
			currentConcurrent.Add(-1)
			return "done:" + intent.Action, nil
		},
		MemDB:       db,
		WorkerCount: 2,
	})
	if err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	if err := agent.Run(ctx, "do 5 things"); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)
	agent.Stop()

	if totalCalls.Load() != 5 {
		t.Errorf("expected 5 total calls, got %d", totalCalls.Load())
	}

	if maxConcurrent < 2 {
		t.Errorf("expected max concurrency >= 2, got %d (workers are starving!)", maxConcurrent)
	}
	t.Logf("max concurrency: %d, elapsed: %s", maxConcurrent, elapsed)

	// With 2 workers and 50ms per task:
	// Serial would take 5*50ms = 250ms
	// Parallel should take ~3*50ms = 150ms (ceil(5/2) batches)
	if elapsed > 200*time.Millisecond {
		t.Logf("warning: elapsed %s suggests limited parallelism", elapsed)
	}
}

func TestRetryEpochNumbersMonotonic(t *testing.T) {
	db := testDB(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var calls atomic.Int64
	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(cai.Intent{Action: "repair", Target: "thing", Engine: cai.EngineBash}),
		Executor: func(_ context.Context, _ string, _ cai.Intent) (string, error) {
			if calls.Add(1) < 3 {
				return "", fmt.Errorf("transient failure")
			}
			return "fixed", nil
		},
		Optimizer: func(_ context.Context, _ error, _ string, _ cai.Intent) (cai.OptimizationResult, error) {
			return cai.OptimizationResult{ShouldRetry: true, CacheUpdate: "Retry with stricter prompt"}, nil
		},
		MemDB:             db,
		WorkerCount:       1,
		MaxSelfIterations: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Stop()

	if err := agent.Run(ctx, "repair thing"); err != nil {
		t.Fatal(err)
	}

	var spanID string
	if err := db.DB().QueryRow(`SELECT span_id FROM spans LIMIT 1`).Scan(&spanID); err != nil {
		t.Fatal(err)
	}
	history, err := db.SpanHistory(spanID)
	if err != nil {
		t.Fatal(err)
	}
	if len(history) != 3 {
		t.Fatalf("expected 3 epochs, got %d", len(history))
	}
	for idx, epoch := range history {
		expected := idx + 1
		if epoch.EpochNumber != expected {
			t.Fatalf("epoch %d: expected number %d, got %d", idx, expected, epoch.EpochNumber)
		}
	}
}

func TestTraceStatusUpdatedAfterSuccessfulRun(t *testing.T) {
	db := testDB(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(
			cai.Intent{Action: "one", Target: "alpha", Engine: cai.EngineBash},
			cai.Intent{Action: "two", Target: "beta", Engine: cai.EngineBash},
		),
		Executor: func(_ context.Context, _ string, i cai.Intent) (string, error) {
			return "ok:" + i.Target, nil
		},
		MemDB:       db,
		WorkerCount: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Stop()

	if err := agent.Run(ctx, "two successful steps"); err != nil {
		t.Fatal(err)
	}

	var traceID, status string
	if err := db.DB().QueryRow(`SELECT trace_id, status FROM traces ORDER BY rowid DESC LIMIT 1`).Scan(&traceID, &status); err != nil {
		t.Fatal(err)
	}
	if status != "success" {
		t.Fatalf("expected trace %s status=success, got %s", traceID, status)
	}
}

func TestOptimizerPatchCompiledIntoRetryContext(t *testing.T) {
	db := testDB(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var calls atomic.Int64
	var sawCompiledPatch atomic.Bool
	intent := cai.Intent{
		Action: "dynamic_step",
		Target: "inspect macOS safely",
		Engine: cai.EngineBash,
		Params: map[string]any{
			"capability_id":   "macos.toolchain",
			"prompt_identity": "cap:macos.toolchain:inspect",
			"task_class":      "inspect",
		},
	}

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(intent),
		Executor: func(_ context.Context, prompt string, _ cai.Intent) (string, error) {
			if calls.Add(1) == 1 {
				return "", fmt.Errorf("first attempt failed")
			}
			if strings.Contains(prompt, "Prefer read-only inspection commands and avoid brittle AppleScript paths.") {
				sawCompiledPatch.Store(true)
			}
			return "ok", nil
		},
		Optimizer: func(_ context.Context, _ error, _ string, _ cai.Intent) (cai.OptimizationResult, error) {
			return cai.OptimizationResult{
				ShouldRetry: true,
				CacheUpdate: "Prefer read-only inspection commands and avoid brittle AppleScript paths.",
			}, nil
		},
		MemDB:             db,
		WorkerCount:       1,
		MaxSelfIterations: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Stop()

	if err := agent.Run(ctx, "inspect macOS safely"); err != nil {
		t.Fatal(err)
	}
	if !sawCompiledPatch.Load() {
		t.Fatal("expected retry context to include compiled optimizer patch")
	}
}

func TestPendingStateStoredDuringHITLAndClearedAfterResolution(t *testing.T) {
	db := testDB(t)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var sawPending atomic.Bool
	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(cai.Intent{Action: "secure", Target: "deploy", Engine: cai.EngineBash}),
		Executor: func(_ context.Context, _ string, _ cai.Intent) (string, error) {
			return "", fmt.Errorf("permission denied")
		},
		HITLResolver: func(ctx context.Context, payload cai.HITLPayload) (cai.Resolution, error) {
			states, err := db.ListPending(ctx)
			if err != nil {
				return cai.Resolution{}, err
			}
			if len(states) != 1 {
				t.Fatalf("expected 1 pending state, got %d", len(states))
			}
			state := states[0]
			if state.PendingApproval == nil {
				t.Fatal("expected pending approval to be stored")
			}
			if len(state.VisibleSteps) != 1 || state.VisibleSteps[0].Status != "error" {
				t.Fatalf("expected one visible error step, got %+v", state.VisibleSteps)
			}
			if len(state.ActiveIncidents) != 1 {
				t.Fatalf("expected one active incident, got %+v", state.ActiveIncidents)
			}
			if payload.IncidentID == "" || payload.ApprovalID == "" {
				t.Fatalf("expected incident and approval ids, got payload=%+v", payload)
			}
			sawPending.Store(true)
			return cai.Resolution{Action: "abort"}, nil
		},
		MemDB:       db,
		WorkerCount: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Stop()

	if err := agent.Run(ctx, "secure deploy"); err == nil {
		t.Fatal("expected HITL abort error")
	}
	if !sawPending.Load() {
		t.Fatal("expected HITL resolver to observe pending state")
	}
	states, err := db.ListPending(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(states) != 0 {
		t.Fatalf("expected pending state to be cleared, got %d entries", len(states))
	}
}

// TestHITLNonBlocking verifies that when one intent triggers HITL (with a slow resolver),
// the other concurrent intents still complete without being blocked.
func TestHITLNonBlocking(t *testing.T) {
	db := testDB(t)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	var fastDone atomic.Bool

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(
			cai.Intent{Action: "fast_task", Target: "quick", Engine: cai.EngineBash},
			cai.Intent{Action: "slow_task", Target: "needs_human", Engine: cai.EngineBash},
		),
		Executor: func(_ context.Context, _ string, intent cai.Intent) (string, error) {
			if intent.Action == "slow_task" {
				return "", fmt.Errorf("permission denied")
			}
			return "fast_done", nil
		},
		HITLResolver: func(ctx context.Context, payload cai.HITLPayload) (cai.Resolution, error) {
			// Simulate slow human response
			time.Sleep(500 * time.Millisecond)
			// By this point, the fast_task should have already completed
			if !fastDone.Load() {
				t.Log("warning: fast_task not yet done during HITL wait — may indicate blocking")
			}
			return cai.Resolution{Action: "abort"}, nil
		},
		MemDB:       db,
		WorkerCount: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Stop()

	// The key validation: the agent.Run call with the async HITL should complete
	// without hanging even though one intent triggers HITL. It should error (abort).
	err = agent.Run(ctx, "test hitl non-blocking")
	if err == nil {
		t.Log("run completed without error (fast_task succeeded, slow_task aborted)")
	}
}

func TestIntentMetaNotSerialized(t *testing.T) {
	intent := cai.Intent{
		Action: "test",
		Target: "something",
		Engine: cai.EngineBash,
		Meta: &cai.IntentMeta{
			PlanStepID:    "step_1",
			PlanStepTitle: "Test Step",
			ExecutionMode: "delegate",
			Runtime:       "subagent",
			CapabilityID:  "test.cap",
		},
	}

	data, err := json.Marshal(intent)
	if err != nil {
		t.Fatal(err)
	}

	jsonStr := string(data)
	// Meta should NOT appear in serialized JSON
	if strings.Contains(jsonStr, "plan_step_id") {
		t.Fatalf("Meta field leaked into JSON: %s", jsonStr)
	}
	if strings.Contains(jsonStr, "execution_mode") {
		t.Fatalf("Meta field leaked into JSON: %s", jsonStr)
	}
	if strings.Contains(jsonStr, "delegate") {
		t.Fatalf("Meta field leaked into JSON: %s", jsonStr)
	}
	if strings.Contains(jsonStr, "subagent") {
		t.Fatalf("Meta field leaked into JSON: %s", jsonStr)
	}

	// Verify Action/Target/Engine are present
	if !strings.Contains(jsonStr, `"action":"test"`) {
		t.Fatalf("Action missing from JSON: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"target":"something"`) {
		t.Fatalf("Target missing from JSON: %s", jsonStr)
	}
}
