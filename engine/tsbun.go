package engine

import (
	"bytes"
	"context"
	"os/exec"

	cai "github.com/anthropic/cai"
	"github.com/anthropic/cai/envelope"
)

// TSBunEngine executes TypeScript via Bun runtime.
// For complex scripts with dependencies, use the embedded JS runtime or UDS worker.
type TSBunEngine struct {
	classifier *envelope.Classifier
	runtime    string
}

// NewTSBunEngine creates a TS/Bun execution engine.
// Falls back to "npx tsx" if bun is not installed.
func NewTSBunEngine() *TSBunEngine {
	runtime := "bun"
	if _, err := exec.LookPath("bun"); err != nil {
		runtime = "node"
	}
	return &TSBunEngine{
		classifier: envelope.DefaultClassifier(),
		runtime:    runtime,
	}
}

func (e *TSBunEngine) Type() cai.EngineType { return cai.EngineTSBun }

func (e *TSBunEngine) Execute(ctx context.Context, script string, env map[string]string) (*cai.ExecutionEnvelope, error) {
	var cmd *exec.Cmd
	if e.runtime == "bun" {
		cmd = exec.CommandContext(ctx, "bun", "eval", script)
	} else {
		cmd = exec.CommandContext(ctx, "node", "-e", script)
	}

	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		}
	}

	result := e.classifier.Classify(exitCode, stdout.String(), stderr.String())
	return &result, nil
}

func (e *TSBunEngine) Close() error { return nil }
