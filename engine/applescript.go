package engine

import (
	"bytes"
	"context"
	"os/exec"

	cai "github.com/anthropic/cai"
	"github.com/anthropic/cai/envelope"
)

// AppleScriptEngine executes AppleScript via osascript.
type AppleScriptEngine struct {
	classifier *envelope.Classifier
}

func NewAppleScriptEngine() *AppleScriptEngine {
	return &AppleScriptEngine{classifier: envelope.DefaultClassifier()}
}

func (e *AppleScriptEngine) Type() cai.EngineType { return cai.EngineAppleScript }

func (e *AppleScriptEngine) Execute(ctx context.Context, script string, env map[string]string) (*cai.ExecutionEnvelope, error) {
	cmd := exec.CommandContext(ctx, "osascript", "-e", script)
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

func (e *AppleScriptEngine) Close() error { return nil }
