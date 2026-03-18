package engine

import (
	"bytes"
	"context"
	"os/exec"

	cai "github.com/anthropic/cai"
	"github.com/anthropic/cai/envelope"
)

// PythonEngine executes Python scripts.
type PythonEngine struct {
	classifier *envelope.Classifier
	python     string
}

// NewPythonEngine creates a Python execution engine.
func NewPythonEngine() *PythonEngine {
	python := "python3"
	if _, err := exec.LookPath("python3"); err != nil {
		python = "python"
	}
	return &PythonEngine{
		classifier: envelope.DefaultClassifier(),
		python:     python,
	}
}

func (e *PythonEngine) Type() cai.EngineType { return cai.EnginePython }

func (e *PythonEngine) Execute(ctx context.Context, script string, env map[string]string) (*cai.ExecutionEnvelope, error) {
	cmd := exec.CommandContext(ctx, e.python, "-c", script)
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
		} else if ctx.Err() != nil {
			result := e.classifier.Classify(124, stdout.String(), "timeout: context deadline exceeded")
			return &result, nil
		}
	}

	result := e.classifier.Classify(exitCode, stdout.String(), stderr.String())
	return &result, nil
}

func (e *PythonEngine) Close() error { return nil }
