package hydrator

import (
	"fmt"
	"strings"

	cai "github.com/anthropic/cai"
)

// DefaultSystemPrompts provides minimal defaults for runtime assembly.
var DefaultSystemPrompts = map[cai.EngineType]string{
	cai.EngineBash:        "You are a bash command generator. Output ONLY executable commands. No markdown. No explanation.",
	cai.EngineAppleScript: "You are an AppleScript generator. Output ONLY raw executable AppleScript. No markdown.",
	cai.EngineTSBun:       "You are a TypeScript code generator for Bun runtime. Output ONLY executable code. No explanation.",
	cai.EnginePython:      "You are a Python script generator. Output ONLY executable Python code. No explanation.",
	cai.EngineNodeJS:      "You are a Node.js script generator. Output ONLY executable JavaScript. No explanation.",
}

// Hydrate assembles a minimal three-segment CerebellumScope from Intent + runtime state.
//
// Segment 1: SystemPrompt — engine default
// Segment 2: ToolSchema   — static tool definitions, never modified by framework
// Segment 3: TaskPrompt   — atomic instruction from Brain
func Hydrate(intent cai.Intent, memDB cai.MemoryDB, tools []cai.ToolDef) cai.CerebellumScope {
	systemPrompt := defaultPrompt(intent.Engine)

	// Segment 2: ToolSchema (static, passed through)
	// Segment 3: TaskPrompt + EnvMetadata
	env := memDB.LoadSystemEvolution()

	return cai.CerebellumScope{
		SystemPrompt: systemPrompt,
		ToolSchema:   tools,
		TaskPrompt:   FormatTaskPrompt(intent),
		EnvMetadata:  formatEnvMetadata(env),
	}
}

// FormatTaskPrompt produces an extremely terse instruction from an Intent.
func FormatTaskPrompt(intent cai.Intent) string {
	s := fmt.Sprintf("%s: %s", intent.Action, intent.Target)
	if len(intent.Params) > 0 {
		for k, v := range intent.Params {
			s += fmt.Sprintf(" [%s=%v]", k, v)
		}
	}
	return s
}

// FormatToolSchema renders ToolDefs into a compact string for prompt injection.
func FormatToolSchema(tools []cai.ToolDef) string {
	if len(tools) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Available tools:\n")
	for _, t := range tools {
		b.WriteString(fmt.Sprintf("- %s: %s", t.Name, t.Description))
		if len(t.Parameters) > 0 {
			b.WriteString(" ")
			b.Write(t.Parameters)
		}
		b.WriteByte('\n')
	}
	return b.String()
}

func formatEnvMetadata(env cai.SystemEvolutionMemory) string {
	s := fmt.Sprintf("os=%s, arch=%s", env.OSVersion, env.Architecture)
	if env.Shell != "" {
		s += ", shell=" + env.Shell
	}
	if env.PreferredPkgManager != "" {
		s += ", pkg=" + env.PreferredPkgManager
	}
	return s
}

func defaultPrompt(engine cai.EngineType) string {
	if p, ok := DefaultSystemPrompts[engine]; ok {
		return p
	}
	return "Output ONLY executable code. No explanation."
}
