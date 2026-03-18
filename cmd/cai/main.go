package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	cai "github.com/anthropic/cai"
	"github.com/anthropic/cai/memory"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	homeDir, _ := os.UserHomeDir()
	dbPath := filepath.Join(homeDir, ".cai", "cai.db")

	switch os.Args[1] {
	case "run":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: cai run <objective>")
			os.Exit(1)
		}
		objective := strings.Join(os.Args[2:], " ")
		runAgent(dbPath, objective)

	case "status":
		showStatus(dbPath)

	case "probe":
		probeEnv(dbPath)

	case "history":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: cai history <span_id>")
			os.Exit(1)
		}
		showHistory(dbPath, os.Args[2])

	default:
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`cai - Concurrent Agent Intelligence Framework

Usage:
  cai run <objective>    Run an agent with the given objective
  cai status             Show system evolution memory
  cai probe              Probe and update system environment
  cai history <span_id>  Show epoch history for a span`)
}

func runAgent(dbPath, objective string) {
	db, err := memory.Open(dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	// Probe system environment
	if err := memory.ProbeSystem(db); err != nil {
		log.Printf("probe: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nShutting down...")
		cancel()
	}()

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner:  &echoPlan{},
		Executor: echoExecutor,
		MemDB:    db,
		Optimizer: func(ctx context.Context, lastErr error, lastPrompt string, intent cai.Intent) (cai.OptimizationResult, error) {
			return cai.OptimizationResult{
				ShouldRetry: true,
				NewPrompt:   lastPrompt + "\n# Fix: " + lastErr.Error(),
			}, nil
		},
		HITLResolver: func(ctx context.Context, payload cai.HITLPayload) (cai.Resolution, error) {
			fmt.Printf("[HITL] Error in %s: %s\n", payload.Intent.Action, payload.TriageMemo)
			fmt.Print("Action (resume/abort/force): ")
			var action string
			fmt.Scanln(&action)
			return cai.Resolution{Action: action}, nil
		},
		WorkerCount: 2,
	})
	if err != nil {
		log.Fatalf("create agent: %v", err)
	}

	fmt.Printf("Agent started [trace=%s]\n", agent.TraceID())
	fmt.Printf("Objective: %s\n\n", objective)

	if err := agent.Run(ctx, objective); err != nil {
		log.Printf("agent error: %v", err)
	}

	agent.Stop()
	fmt.Println("Done.")
}

// echoPlan is a demo planner that creates a single bash intent.
type echoPlan struct{}

func (p *echoPlan) Plan(ctx context.Context, objective string, scope *cai.BrainScope) ([]cai.Intent, error) {
	return []cai.Intent{
		{
			Action: "execute",
			Target: objective,
			Engine: cai.EngineBash,
		},
	}, nil
}

// echoExecutor is a demo executor that runs bash commands.
func echoExecutor(ctx context.Context, prompt string, intent cai.Intent) (string, error) {
	fmt.Printf("[executor] %s: %s\n", intent.Action, intent.Target)
	return fmt.Sprintf("Executed: %s", intent.Target), nil
}

func showStatus(dbPath string) {
	db, err := memory.Open(dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	env := db.LoadSystemEvolution()
	fmt.Printf("OS:        %s\n", env.OSVersion)
	fmt.Printf("Arch:      %s\n", env.Architecture)
	fmt.Printf("Shell:     %s\n", env.Shell)
	fmt.Printf("Pkg Mgr:   %s\n", env.PreferredPkgManager)
	fmt.Printf("Runtimes:  %v\n", env.InstalledRuntimes)
	fmt.Printf("Probed:    %s\n", env.LastProbeAt.Format("2006-01-02 15:04:05"))
}

func probeEnv(dbPath string) {
	db, err := memory.Open(dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	if err := memory.ProbeSystem(db); err != nil {
		log.Fatalf("probe: %v", err)
	}
	fmt.Println("Environment probed successfully.")
	showStatus(dbPath)
}

func showHistory(dbPath, spanID string) {
	db, err := memory.Open(dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	epochs, err := db.SpanHistory(spanID)
	if err != nil {
		log.Fatalf("history: %v", err)
	}

	if len(epochs) == 0 {
		fmt.Println("No epochs found for span:", spanID)
		return
	}

	for _, e := range epochs {
		status := "SUCCESS"
		if e.IsDeadEnd {
			status = "DEAD_END"
		}
		fmt.Printf("  Epoch %d [%s] %s\n", e.EpochNumber, status, e.Category)
		fmt.Printf("    Prompt: %s\n", e.Prompt)
		if e.Result != "" {
			fmt.Printf("    Result: %s\n", e.Result)
		}
		if e.Error != "" {
			fmt.Printf("    Error:  %s\n", e.Error)
		}
		fmt.Printf("    Duration: %dms\n\n", e.DurationMs)
	}
}
