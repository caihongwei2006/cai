package cai

import "context"

type staticPlanner struct {
	intents []Intent
}

func (p staticPlanner) Plan(_ context.Context, _ string, _ *BrainScope) ([]Intent, error) {
	out := make([]Intent, len(p.intents))
	copy(out, p.intents)
	return out, nil
}

// StaticPlanner returns a Planner that always emits the provided intents.
func StaticPlanner(intents ...Intent) Planner {
	out := make([]Intent, len(intents))
	copy(out, intents)
	return staticPlanner{intents: out}
}
