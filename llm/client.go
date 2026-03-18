package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client is a minimal OpenAI-compatible LLM client.
// Works with Azure OpenAI, LiteLLM proxy, or any OpenAI-compatible endpoint.
type Client struct {
	endpoint   string
	apiKey     string
	apiVersion string
	deployment string
	httpClient *http.Client
}

// Config holds LLM client configuration.
type Config struct {
	Endpoint   string // e.g. "https://api.openai.com"
	APIKey     string
	APIVersion string // e.g. "2024-12-01-preview"
	Deployment string // e.g. "openai/gpt-4.1"
}

// NewClient creates an LLM client.
func NewClient(cfg Config) *Client {
	return &Client{
		endpoint:   cfg.Endpoint,
		apiKey:     cfg.APIKey,
		apiVersion: cfg.APIVersion,
		deployment: cfg.Deployment,
		httpClient: &http.Client{Timeout: 120 * time.Second},
	}
}

// Message is a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CompletionRequest is the request body.
type CompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature float64   `json:"temperature"`
}

// CompletionResponse is the response body.
type CompletionResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// Complete sends a chat completion request and returns the assistant's response.
func (c *Client) Complete(ctx context.Context, messages []Message, maxTokens int, temperature float64) (*CompletionResponse, error) {
	reqBody := CompletionRequest{
		Model:       c.deployment,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
	}

	data, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", c.endpoint)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	if c.apiVersion != "" {
		req.Header.Set("api-version", c.apiVersion)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http do: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result CompletionResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if len(result.Choices) == 0 {
		return nil, fmt.Errorf("empty choices in response")
	}

	return &result, nil
}

// SimpleComplete is a convenience method for single-turn completion.
func (c *Client) SimpleComplete(ctx context.Context, systemPrompt, userPrompt string, maxTokens int) (string, int, error) {
	msgs := []Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	resp, err := c.Complete(ctx, msgs, maxTokens, 0.0)
	if err != nil {
		return "", 0, err
	}

	return resp.Choices[0].Message.Content, resp.Usage.TotalTokens, nil
}
