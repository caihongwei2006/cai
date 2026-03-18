package cai

import "context"

type toolExecutionMetaKey struct{}

// ToolExecutionMeta carries execution metadata into tool calls so tools can
// relate their work back to the active span/step without depending on planner
// internals.
type ToolExecutionMeta struct {
	TraceID      string
	SpanID       string
	StepID       string
	IntentAction string
}

func WithToolExecutionMeta(ctx context.Context, meta ToolExecutionMeta) context.Context {
	return context.WithValue(ctx, toolExecutionMetaKey{}, meta)
}

func ToolExecutionMetaFromContext(ctx context.Context) ToolExecutionMeta {
	if ctx == nil {
		return ToolExecutionMeta{}
	}
	if meta, ok := ctx.Value(toolExecutionMetaKey{}).(ToolExecutionMeta); ok {
		return meta
	}
	return ToolExecutionMeta{}
}
