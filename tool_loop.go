package cai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"strings"
	"time"
)

// --- Tool-Use Loop LLM Types (OpenAI-compatible function calling) ---

type ChatImageURLPart struct {
	URL string `json:"url"`
}

type ChatContentPart struct {
	Type     string            `json:"type"`
	Text     string            `json:"text,omitempty"`
	ImageURL *ChatImageURLPart `json:"image_url,omitempty"`
}

type ChatMessageContent struct {
	text  *string
	parts []ChatContentPart
}

func TextContent(text string) ChatMessageContent {
	return ChatMessageContent{text: &text}
}

func PartsContent(parts ...ChatContentPart) ChatMessageContent {
	return ChatMessageContent{parts: append([]ChatContentPart(nil), parts...)}
}

func (c ChatMessageContent) MarshalJSON() ([]byte, error) {
	if c.text != nil {
		return json.Marshal(*c.text)
	}
	if c.parts != nil {
		return json.Marshal(c.parts)
	}
	return json.Marshal("")
}

// ChatMessage extends llm.Message with tool-calling fields.
type ChatMessage struct {
	Role       string             `json:"role"`
	Content    ChatMessageContent `json:"content"`
	ToolCalls  []ToolCall         `json:"tool_calls,omitempty"`
	ToolCallID string             `json:"tool_call_id,omitempty"`
}

// ToolCall represents a single tool invocation requested by the LLM.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFuncCall `json:"function"`
}

// ToolCallFuncCall is the function name + arguments from LLM.
type ToolCallFuncCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolCallSchema is the tool definition sent to the LLM.
type ToolCallSchema struct {
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction describes a tool's function schema for the LLM.
type ToolCallFunction struct {
	Name          string            `json:"name"`
	Description   string            `json:"description"`
	Parameters    json.RawMessage   `json:"parameters"`
	InputExamples []json.RawMessage `json:"input_examples,omitempty"`
}

type ToolChoice struct {
	Type     string              `json:"type"`
	Function *ToolChoiceFunction `json:"function,omitempty"`
}

type ToolChoiceFunction struct {
	Name string `json:"name"`
}

// ToolCompletionRequest is the request body for tool-enabled completions.
type ToolCompletionRequest struct {
	Model             string           `json:"model"`
	Messages          []ChatMessage    `json:"messages"`
	Tools             []ToolCallSchema `json:"tools,omitempty"`
	ToolChoice        *ToolChoice      `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool            `json:"parallel_tool_calls,omitempty"`
	MaxTokens         int              `json:"max_tokens,omitempty"`
	Temperature       float64          `json:"temperature"`
	Stream            bool             `json:"stream,omitempty"`
}

// ToolCompletionResponse is the response body.
type ToolCompletionResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message struct {
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls,omitempty"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// LLMToolResponse is the parsed result from callLLMWithTools.
type LLMToolResponse struct {
	Content      string
	ToolCalls    []ToolCall
	FinishReason string
	Usage        struct {
		PromptTokens     int
		CompletionTokens int
		TotalTokens      int
	}
}

type ToolCompletionStreamResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Delta struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// --- HTTP helpers for tool-use loop ---

var toolHTTPClient = &http.Client{Timeout: 600 * time.Second}

func newHTTPRequest(ctx context.Context, method, url string, body []byte, apiKeyEnv string) (*http.Request, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, strings.NewReader(string(body)))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	apiKey := os.Getenv(apiKeyEnv)
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	return req, nil
}

func readBody(r io.Reader) ([]byte, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}
	return data, nil
}

func readSSEDataBlock(reader *bufio.Reader) (string, error) {
	var lines []string
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF && len(lines) > 0 {
				return strings.Join(lines, "\n"), nil
			}
			return "", err
		}

		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			if len(lines) == 0 {
				continue
			}
			return strings.Join(lines, "\n"), nil
		}
		if strings.HasPrefix(line, "data:") {
			lines = append(lines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
}

func parseToolCompletionStream(r io.Reader, onDelta func(text string)) (*LLMToolResponse, error) {
	reader := bufio.NewReader(r)
	resp := &LLMToolResponse{}
	toolCalls := map[int]*ToolCall{}

	for {
		block, err := readSSEDataBlock(reader)
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("read stream: %w", err)
		}
		if block == "" {
			continue
		}
		if block == "[DONE]" {
			break
		}

		var chunk ToolCompletionStreamResponse
		if err := json.Unmarshal([]byte(block), &chunk); err != nil {
			return nil, fmt.Errorf("unmarshal stream chunk: %w", err)
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		choice := chunk.Choices[0]
		if choice.Delta.Content != "" {
			resp.Content += choice.Delta.Content
			if onDelta != nil {
				onDelta(choice.Delta.Content)
			}
		}
		for _, deltaCall := range choice.Delta.ToolCalls {
			call, ok := toolCalls[deltaCall.Index]
			if !ok {
				call = &ToolCall{}
				toolCalls[deltaCall.Index] = call
			}
			if deltaCall.ID != "" {
				call.ID += deltaCall.ID
			}
			if deltaCall.Type != "" {
				call.Type = deltaCall.Type
			}
			if deltaCall.Function.Name != "" {
				call.Function.Name += deltaCall.Function.Name
			}
			if deltaCall.Function.Arguments != "" {
				call.Function.Arguments += deltaCall.Function.Arguments
			}
		}
		if choice.FinishReason != "" {
			resp.FinishReason = choice.FinishReason
		}
		if chunk.Usage.TotalTokens > 0 {
			resp.Usage.PromptTokens = chunk.Usage.PromptTokens
			resp.Usage.CompletionTokens = chunk.Usage.CompletionTokens
			resp.Usage.TotalTokens = chunk.Usage.TotalTokens
		}
	}

	if len(toolCalls) > 0 {
		indexes := make([]int, 0, len(toolCalls))
		for idx := range toolCalls {
			indexes = append(indexes, idx)
		}
		sort.Ints(indexes)
		resp.ToolCalls = make([]ToolCall, 0, len(indexes))
		for _, idx := range indexes {
			resp.ToolCalls = append(resp.ToolCalls, *toolCalls[idx])
		}
	}

	return resp, nil
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
