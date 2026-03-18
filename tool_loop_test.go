package cai

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestChatMessageMarshalKeepsEmptyContentField(t *testing.T) {
	data, err := json.Marshal(ChatMessage{
		Role:    "system",
		Content: TextContent(""),
	})
	if err != nil {
		t.Fatalf("marshal chat message: %v", err)
	}

	got := string(data)
	if !strings.Contains(got, `"content":""`) {
		t.Fatalf("expected empty content field to be preserved, got %s", got)
	}
}

func TestChatMessageMarshalSupportsMultipartUserContent(t *testing.T) {
	data, err := json.Marshal(ChatMessage{
		Role: "user",
		Content: PartsContent(
			ChatContentPart{Type: "text", Text: "Inspect this"},
			ChatContentPart{
				Type: "image_url",
				ImageURL: &ChatImageURLPart{
					URL: "data:image/png;base64,abc123",
				},
			},
		),
	})
	if err != nil {
		t.Fatalf("marshal multipart chat message: %v", err)
	}

	got := string(data)
	if !strings.Contains(got, `"type":"image_url"`) {
		t.Fatalf("expected image content part in payload, got %s", got)
	}
	if !strings.Contains(got, `"type":"text"`) {
		t.Fatalf("expected text content part in payload, got %s", got)
	}
}
