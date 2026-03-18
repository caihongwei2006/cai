package cai

import (
	"context"
	"fmt"
)

// --- Default HITL Resolvers ---

// AutoContinueResolver always resumes with no changes.
func AutoContinueResolver(_ context.Context, _ HITLPayload) (Resolution, error) {
	return Resolution{Action: "resume"}, nil
}

// AutoAbortResolver always aborts on HITL escalation.
func AutoAbortResolver(_ context.Context, _ HITLPayload) (Resolution, error) {
	return Resolution{Action: "abort"}, nil
}

// CLIResolver prompts the user in the terminal for a decision.
func CLIResolver(_ context.Context, payload HITLPayload) (Resolution, error) {
	fmt.Printf("\n[HITL] Escalation for %s:%s\n", payload.Intent.Action, payload.Intent.Target)
	fmt.Printf("  Error: %s %s\n", payload.LastError.Category, payload.LastError.RawStderr)
	fmt.Printf("  Triage: %s\n", payload.TriageMemo)
	fmt.Print("\nAction (resume/abort/force): ")

	var action string
	fmt.Scanln(&action)

	switch action {
	case "force":
		fmt.Print("Mock value: ")
		var mock string
		fmt.Scanln(&mock)
		return Resolution{Action: "force_success", MockValue: mock}, nil
	case "resume":
		fmt.Print("New prompt (empty to keep): ")
		var prompt string
		fmt.Scanln(&prompt)
		return Resolution{Action: "resume", NewPrompt: prompt}, nil
	default:
		return Resolution{Action: "abort"}, nil
	}
}

// --- Default Escalation ---

// EscalationPolicy controls the retry ladder.
type EscalationPolicy struct {
	MaxCerebellumRetries int
	MaxTriageRetries     int
	OnAllExhausted       string // "hitl" | "abort" | "brain_takeover"
}

// DefaultEscalation returns sensible defaults.
func DefaultEscalation() EscalationPolicy {
	return EscalationPolicy{
		MaxCerebellumRetries: 3,
		MaxTriageRetries:     2,
		OnAllExhausted:       "hitl",
	}
}
