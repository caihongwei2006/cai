package data

import (
	"encoding/json"
	"log"
	"os"
	"sync"

	cai "github.com/anthropic/cai"
)

// FileCollector writes framework events to a JSON Lines file for data collection.
type FileCollector struct {
	mu   sync.Mutex
	file *os.File
	enc  *json.Encoder
}

// NewFileCollector creates a collector that appends to the given file.
func NewFileCollector(path string) (*FileCollector, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if err != nil {
		return nil, err
	}
	return &FileCollector{
		file: f,
		enc:  json.NewEncoder(f),
	}, nil
}

type eventRecord struct {
	Type string      `json:"type"`
	Data interface{} `json:"data"`
}

func (c *FileCollector) OnEpoch(epoch cai.ExecutionEpoch) {
	c.write(eventRecord{Type: "epoch", Data: epoch})
}

func (c *FileCollector) OnSpanComplete(span cai.LogicalSpan, epochs []cai.ExecutionEpoch) {
	c.write(eventRecord{Type: "span_complete", Data: map[string]interface{}{
		"span":   span,
		"epochs": epochs,
	}})
}

func (c *FileCollector) OnTraceComplete(trace cai.AgentTrace) {
	c.write(eventRecord{Type: "trace_complete", Data: trace})
}

func (c *FileCollector) OnCacheHit(action string) {
	c.write(eventRecord{Type: "cache_hit", Data: map[string]string{"action": action}})
}

func (c *FileCollector) OnCacheEvict(action string, reason string) {
	c.write(eventRecord{Type: "cache_evict", Data: map[string]string{"action": action, "reason": reason}})
}

func (c *FileCollector) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.file.Close()
}

func (c *FileCollector) write(record eventRecord) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.enc.Encode(record); err != nil {
		log.Printf("[collector] write error: %v", err)
	}
}

// NopCollector discards all events. Used when no data collection is needed.
type NopCollector struct{}

func (NopCollector) OnEpoch(cai.ExecutionEpoch)                             {}
func (NopCollector) OnSpanComplete(cai.LogicalSpan, []cai.ExecutionEpoch)   {}
func (NopCollector) OnTraceComplete(cai.AgentTrace)                         {}
func (NopCollector) OnCacheHit(string)                                      {}
func (NopCollector) OnCacheEvict(string, string)                            {}
func (NopCollector) Close() error                                           { return nil }

// ChannelCollector sends events to a Go channel for custom processing pipelines.
type ChannelCollector struct {
	ch chan<- eventRecord
}

// NewChannelCollector creates a collector that sends to the given channel.
func NewChannelCollector(ch chan<- eventRecord) *ChannelCollector {
	return &ChannelCollector{ch: ch}
}

func (c *ChannelCollector) OnEpoch(epoch cai.ExecutionEpoch) {
	c.ch <- eventRecord{Type: "epoch", Data: epoch}
}
func (c *ChannelCollector) OnSpanComplete(span cai.LogicalSpan, epochs []cai.ExecutionEpoch) {
	c.ch <- eventRecord{Type: "span_complete", Data: map[string]interface{}{"span": span, "epochs": epochs}}
}
func (c *ChannelCollector) OnTraceComplete(trace cai.AgentTrace) {
	c.ch <- eventRecord{Type: "trace_complete", Data: trace}
}
func (c *ChannelCollector) OnCacheHit(action string) {
	c.ch <- eventRecord{Type: "cache_hit", Data: map[string]string{"action": action}}
}
func (c *ChannelCollector) OnCacheEvict(action string, reason string) {
	c.ch <- eventRecord{Type: "cache_evict", Data: map[string]string{"action": action, "reason": reason}}
}
func (c *ChannelCollector) Close() error { return nil }
