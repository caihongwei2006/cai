package engine

import (
	"bytes"
	"context"
	"os/exec"
	"time"

	cai "github.com/anthropic/cai"
	"github.com/anthropic/cai/envelope"
)

// BashEngine executes scripts via /bin/bash.
type BashEngine struct {
	classifier *envelope.Classifier
	shell      string
}

// NewBashEngine creates a bash execution engine.
func NewBashEngine() *BashEngine {
	shell := "/bin/bash"
	if _, err := exec.LookPath("bash"); err != nil {
		shell = "/bin/sh"
	}
	return &BashEngine{
		classifier: envelope.DefaultClassifier(),
		shell:      shell,
	}
}

func (e *BashEngine) Type() cai.EngineType { return cai.EngineBash }

func (e *BashEngine) Execute(ctx context.Context, script string, env map[string]string) (*cai.ExecutionEnvelope, error) {
	cmd := exec.CommandContext(ctx, e.shell, "-c", script)

	for k, v := range env {
		cmd.Env = append(cmd.Env, k+"="+v)
	}

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	start := time.Now()
	err := cmd.Run()
	_ = time.Since(start)

	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else if ctx.Err() != nil {
			result := e.classifier.Classify(124, stdout.String(), "timeout: context deadline exceeded")
			return &result, nil
		}
	}

	result := e.classifier.Classify(exitCode, stdout.String(), stderr.String())
	return &result, nil
}

func (e *BashEngine) Close() error { return nil }
