package cai_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	cai "github.com/anthropic/cai"
)

type echoTool struct{}

type cancelTool struct {
	cancel context.CancelCauseFunc
	cause  error
}

func (echoTool) Name() string        { return "echo_tool" }
func (echoTool) Description() string { return "echoes the provided value" }
func (echoTool) InputSchema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}`)
}
func (echoTool) Execute(_ context.Context, input json.RawMessage) (cai.ToolResult, error) {
	var payload struct {
		Value string `json:"value"`
	}
	if err := json.Unmarshal(input, &payload); err != nil {
		return cai.ToolResult{}, err
	}
	return cai.ToolResult{Content: payload.Value}, nil
}

func (t cancelTool) Name() string        { return "cancel_tool" }
func (t cancelTool) Description() string { return "cancels the current task context" }
func (t cancelTool) InputSchema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{}}`)
}
func (t cancelTool) Execute(_ context.Context, _ json.RawMessage) (cai.ToolResult, error) {
	t.cancel(t.cause)
	return cai.ToolResult{Content: "cancelled"}, nil
}

func TestLLMAndToolHooksFireDuringToolLoop(t *testing.T) {
	db := testDB(t)
	var (
		mu         sync.Mutex
		llmEvents  []cai.LLMCallEvent
		toolEvents []cai.ToolExecutionEvent
		callCount  int
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		callCount++
		current := callCount
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if current == 1 {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{
					{
						"message": map[string]any{
							"content": "",
							"tool_calls": []map[string]any{
								{
									"id":   "call_1",
									"type": "function",
									"function": map[string]any{
										"name":      "echo_tool",
										"arguments": `{"value":"hello"}`,
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
				"usage": map[string]any{
					"prompt_tokens":     10,
					"completion_tokens": 4,
					"total_tokens":      14,
				},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "done",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     7,
				"completion_tokens": 3,
				"total_tokens":      10,
			},
		})
	}))
	defer server.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner: simplePlanner(cai.Intent{
			Action: "agent_turn",
			Target: "use the echo tool",
			Params: map[string]any{
				"execution_mode": "inline",
				"runtime":        "root",
				"capability_id":  "general.execute",
			},
		}),
		Tools: []cai.Tool{echoTool{}},
		MemDB: db,
		ModelConfig: &cai.ModelConfig{
			Cerebellum: cai.ModelEndpoint{
				BaseURL:   server.URL,
				ModelID:   "test-model",
				MaxTokens: 128,
			},
		},
		DefaultSystemPrompt: "Be concise",
		WorkerCount:         1,
		OnLLMCall: func(_ string, _ cai.Intent, event cai.LLMCallEvent) {
			mu.Lock()
			defer mu.Unlock()
			llmEvents = append(llmEvents, event)
		},
		OnToolCall: func(_ string, _ cai.Intent, event cai.ToolExecutionEvent) {
			mu.Lock()
			defer mu.Unlock()
			toolEvents = append(toolEvents, event)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Stop()

	result, err := agent.RunWithResult(ctx, "use the echo tool")
	if err != nil {
		t.Fatalf("RunWithResult: %v", err)
	}
	if result != "done" {
		t.Fatalf("expected final result done, got %q", result)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(llmEvents) != 2 {
		t.Fatalf("expected 2 llm events, got %d", len(llmEvents))
	}
	if llmEvents[0].ToolCalls != 1 {
		t.Fatalf("expected first llm event to include tool call, got %d", llmEvents[0].ToolCalls)
	}
	if llmEvents[0].TotalTokens != 14 || llmEvents[1].TotalTokens != 10 {
		t.Fatalf("unexpected llm usage: %#v", llmEvents)
	}
	if len(toolEvents) != 1 {
		t.Fatalf("expected 1 tool event, got %d", len(toolEvents))
	}
	if toolEvents[0].ToolName != "echo_tool" || toolEvents[0].Result != "hello" {
		t.Fatalf("unexpected tool event: %#v", toolEvents[0])
	}
}

func TestToolLoopUsesTaskContextCancellation(t *testing.T) {
	db := testDB(t)
	stopErr := errors.New("stop after dispatch")
	var (
		mu        sync.Mutex
		callCount int
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		callCount++
		current := callCount
		mu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if current == 1 {
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{
					{
						"message": map[string]any{
							"content": "",
							"tool_calls": []map[string]any{
								{
									"id":   "call_cancel",
									"type": "function",
									"function": map[string]any{
										"name":      "cancel_tool",
										"arguments": `{}`,
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
				"usage": map[string]any{
					"prompt_tokens":     6,
					"completion_tokens": 2,
					"total_tokens":      8,
				},
			})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{
					"message": map[string]any{
						"content": "unexpected second llm turn",
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]any{
				"prompt_tokens":     4,
				"completion_tokens": 2,
				"total_tokens":      6,
			},
		})
	}))
	defer server.Close()

	rootCtx, rootCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer rootCancel()
	taskCtx, taskCancel := context.WithCancelCause(rootCtx)

	agent, err := cai.NewAgent(rootCtx, cai.AgentConfig{
		Planner: simplePlanner(cai.Intent{
			Action: "agent_turn",
			Target: "cancel after first tool",
			Params: map[string]any{
				"execution_mode": "inline",
				"runtime":        "root",
				"capability_id":  "general.execute",
			},
		}),
		Tools: []cai.Tool{
			cancelTool{cancel: taskCancel, cause: stopErr},
		},
		MemDB: db,
		ModelConfig: &cai.ModelConfig{
			Cerebellum: cai.ModelEndpoint{
				BaseURL:   server.URL,
				ModelID:   "test-model",
				MaxTokens: 128,
			},
		},
		DefaultSystemPrompt: "Be concise",
		WorkerCount:         1,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer agent.Stop()

	_, err = agent.RunWithResult(taskCtx, "cancel after first tool")
	if !errors.Is(context.Cause(taskCtx), stopErr) {
		t.Fatalf("expected task context cause %v, got %v", stopErr, context.Cause(taskCtx))
	}
	if err == nil {
		t.Fatal("expected run to stop after task cancellation")
	}

	mu.Lock()
	defer mu.Unlock()
	if callCount != 1 {
		t.Fatalf("expected exactly one llm call before cancellation, got %d", callCount)
	}
}
