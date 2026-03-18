package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	cai "github.com/anthropic/cai"
	"github.com/anthropic/cai/llm"
	"github.com/anthropic/cai/memory"
)

var (
	totalSOTA    int
	totalFAST    int
	iterationLog []string
)

func main() {
	client := llm.NewClient(llm.Config{
		Endpoint:   envOr("AZURE_ENDPOINT", "https://api.openai.com"),
		APIKey:     envOr("AZURE_OPENAI_API_KEY", "your-api-key-here"),
		APIVersion: envOr("AZURE_API_VERSION", "2024-12-01-preview"),
		Deployment: envOr("LLM_AZURE_DEPLOYMENT", "openai/gpt-4.1"),
	})

	homeDir, _ := os.UserHomeDir()
	dbPath := filepath.Join(homeDir, ".cai", "demo.db")

	db, err := memory.Open(dbPath)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	memory.ProbeSystem(db)

	envInfo := db.LoadSystemEvolution()
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║     CAI — Context Continuous Optimization Demo              ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Printf("System: %s %s | shell=%s | pkg=%s\n", envInfo.OSVersion, envInfo.Architecture, envInfo.Shell, envInfo.PreferredPkgManager)
	fmt.Printf("Model:  %s\n\n", envOr("LLM_AZURE_DEPLOYMENT", "openai/gpt-4.1"))

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() { <-sigCh; cancel() }()

	// --- RUN 1: Cold start (JIT mode) ---
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  RUN 1: JIT Mode (cold start — Brain compiles prompts)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	run1SOTA, run1FAST := runOnce(ctx, client, db, "List all .go files in current directory and count total lines of code")

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  RUN 2: AOT Mode (warm — cached system prompts)")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	// Reset counters
	totalSOTA = 0
	totalFAST = 0
	iterationLog = nil

	run2SOTA, run2FAST := runOnce(ctx, client, db, "List all .go files in current directory and count total lines of code")

	// --- Summary ---
	fmt.Println("\n╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║                    Token Economics                          ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Printf("║  RUN 1 (JIT):  SOTA=%4d  FAST=%4d  Total=%4d              ║\n", run1SOTA, run1FAST, run1SOTA+run1FAST)
	fmt.Printf("║  RUN 2 (AOT):  SOTA=%4d  FAST=%4d  Total=%4d              ║\n", run2SOTA, run2FAST, run2SOTA+run2FAST)
	if run1SOTA > 0 {
		saving := float64(run1SOTA-run2SOTA) / float64(run1SOTA) * 100
		fmt.Printf("║  SOTA Saving:  %.0f%%                                         ║\n", saving)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")

	// Show cached prompts
	fmt.Println("\n--- Cached IntentMemory (evolved through use) ---")
	showCachedMemory(db)
}

func runOnce(ctx context.Context, client *llm.Client, db *memory.SQLiteDB, objective string) (int, int) {
	totalSOTA = 0
	totalFAST = 0
	iterationLog = nil

	planner := &llmPlanner{client: client}
	executorFn := makeExecutor(client, db)
	optimizerFn := makeOptimizer(client, db)

	agent, err := cai.NewAgent(ctx, cai.AgentConfig{
		Planner:           planner,
		Executor:          executorFn,
		Optimizer:         optimizerFn,
		MemDB:             db,
		WorkerCount:       1,
		MaxSelfIterations: 3,
		EvictionThreshold: 3,
		HITLResolver: func(ctx context.Context, payload cai.HITLPayload) (cai.Resolution, error) {
			fmt.Printf("  [HITL] Auto-abort: %s\n", payload.TriageMemo)
			return cai.Resolution{Action: "abort"}, nil
		},
	})
	if err != nil {
		log.Fatalf("new agent: %v", err)
	}

	fmt.Printf("\nObjective: %s\n", objective)
	fmt.Printf("Trace: %s\n\n", agent.TraceID())

	if err := agent.Run(ctx, objective); err != nil {
		fmt.Printf("  Result: error — %v\n", err)
	}
	agent.Stop()

	fmt.Printf("\n  Tokens: SOTA=%d  FAST=%d\n", totalSOTA, totalFAST)
	if len(iterationLog) > 0 {
		fmt.Println("  Self-iterations:")
		for i, entry := range iterationLog {
			fmt.Printf("    %d. %s\n", i+1, entry)
		}
	}

	return totalSOTA, totalFAST
}

// --- Brain: Planner ---

type llmPlanner struct {
	client *llm.Client
}

func (p *llmPlanner) Plan(ctx context.Context, objective string, scope *cai.BrainScope) ([]cai.Intent, error) {
	fmt.Println("  [BRAIN] Decomposing objective into intents...")

	systemPrompt := `You are an agent planner. Decompose a user objective into minimal atomic intents.
Output a JSON array. Each intent: {"action":"verb_noun","target":"object","engine":"bash","params":{}}
Rules: each intent = ONE atomic bash operation. Max 3 intents. Output ONLY valid JSON array.`

	result, tokens, err := p.client.SimpleComplete(ctx, systemPrompt, "Objective: "+objective, 300)
	totalSOTA += tokens
	if err != nil {
		return nil, fmt.Errorf("planner: %w", err)
	}

	result = stripMarkdown(result)

	var intents []cai.Intent
	if err := json.Unmarshal([]byte(result), &intents); err != nil {
		log.Printf("  [BRAIN] JSON parse error, using fallback. Raw: %s", trunc(result, 200))
		intents = []cai.Intent{{Action: "execute", Target: objective, Engine: cai.EngineBash}}
	}

	fmt.Printf("  [BRAIN] %d intents (%d tokens):\n", len(intents), tokens)
	for i, intent := range intents {
		fmt.Printf("    %d. %s: %s [%s]\n", i+1, intent.Action, intent.Target, intent.Engine)
	}
	return intents, nil
}

// --- Cerebellum: Executor ---

func makeExecutor(client *llm.Client, db *memory.SQLiteDB) cai.ExecutorFunc {
	return func(ctx context.Context, prompt string, intent cai.Intent) (string, error) {
		systemPrompt := "You are a bash command generator. Output ONLY executable commands. No markdown. No explanation. No code fences."

		if mem, found := db.LoadIntentMemory(intent.Action); found && mem.SystemPromptHints != "" {
			systemPrompt = mem.SystemPromptHints
			fmt.Printf("  [CACHE HIT] '%s' v%d (streak=%d)\n", intent.Action, mem.Version, mem.SuccessStreak)
		} else {
			fmt.Printf("  [CACHE MISS] '%s'\n", intent.Action)
		}

		env := db.LoadSystemEvolution()
		taskPrompt := fmt.Sprintf("Task: %s\nTarget: %s\nEnv: os=%s arch=%s shell=%s",
			intent.Action, intent.Target, env.OSVersion, env.Architecture, env.Shell)

		fmt.Printf("  [CEREBELLUM] Generating script...\n")
		script, tokens, err := client.SimpleComplete(ctx, systemPrompt, taskPrompt, 100)
		totalFAST += tokens
		if err != nil {
			return "", fmt.Errorf("cerebellum: %w", err)
		}

		script = stripMarkdown(script)
		fmt.Printf("  [CEREBELLUM] Script (%d tok): %s\n", tokens, trunc(script, 120))

		// Actually execute the generated script
		fmt.Printf("  [EXECUTE] Running bash...\n")
		cmd := exec.CommandContext(ctx, "bash", "-c", script)
		cmd.Dir = "."
		out, execErr := cmd.CombinedOutput()
		output := strings.TrimSpace(string(out))

		if execErr != nil {
			fmt.Printf("  [EXECUTE] FAILED: %s\n", trunc(output, 200))
			return output, fmt.Errorf("%s\n%s", execErr.Error(), output)
		}

		fmt.Printf("  [EXECUTE] OK: %s\n", trunc(output, 200))

		// On success, freeze the prompt into IntentMemory (JIT→AOT)
		mem := cai.IntentMemory{
			IntentAction:      intent.Action,
			SystemPromptHints: systemPrompt,
			Version:           1,
		}
		if existing, found := db.LoadIntentMemory(intent.Action); found {
			mem.Version = existing.Version + 1
			mem.SuccessStreak = existing.SuccessStreak + 1
		}
		db.SaveIntentMemory(mem)

		return output, nil
	}
}

// --- Triage: Optimizer ---

func makeOptimizer(client *llm.Client, db *memory.SQLiteDB) cai.OptimizerAnyFunc {
	return func(ctx context.Context, lastErr error, lastPrompt string, intent cai.Intent) (cai.OptimizationResult, error) {
		fmt.Printf("\n  [TRIAGE] Error: %s\n", trunc(lastErr.Error(), 150))
		fmt.Printf("  [TRIAGE] Optimizing system prompt with SOTA model...\n")

		env := db.LoadSystemEvolution()
		analysisPrompt := fmt.Sprintf(`A script generator produced a bash command that failed.
Intent: %s (target: %s)
Environment: os=%s, arch=%s, shell=%s
Error: %s

Write an improved system prompt (under 80 words) that prevents this error.
The prompt must instruct the model to generate ONLY raw executable bash commands.
Output ONLY the improved system prompt text.`,
			intent.Action, intent.Target,
			env.OSVersion, env.Architecture, env.Shell,
			trunc(lastErr.Error(), 300))

		newHints, tokens, err := client.SimpleComplete(ctx, "You are a prompt compiler.", analysisPrompt, 200)
		totalSOTA += tokens
		if err != nil {
			return cai.OptimizationResult{}, err
		}

		newHints = strings.TrimSpace(newHints)
		fmt.Printf("  [TRIAGE] New prompt (%d tok): %s\n\n", tokens, trunc(newHints, 200))

		iterationLog = append(iterationLog, fmt.Sprintf("'%s' → %s", intent.Action, trunc(newHints, 80)))

		return cai.OptimizationResult{
			ShouldRetry: true,
			NewPrompt:   newHints,
			CacheUpdate: newHints,
		}, nil
	}
}

func showCachedMemory(db *memory.SQLiteDB) {
	// Query all intent_memory entries
	actions := []string{"list_files", "count_lines", "list_go_files", "count_loc", "execute", "find_files", "count_total_lines"}
	for _, action := range actions {
		if mem, found := db.LoadIntentMemory(action); found && mem.SystemPromptHints != "" {
			fmt.Printf("  [%s] v%d streak=%d failures=%d\n    > %s\n",
				action, mem.Version, mem.SuccessStreak, mem.ConsecutiveFailures,
				trunc(mem.SystemPromptHints, 120))
		}
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func trunc(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ↵ ")
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}

func stripMarkdown(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "```bash\n")
	s = strings.TrimPrefix(s, "```sh\n")
	s = strings.TrimPrefix(s, "```json\n")
	s = strings.TrimPrefix(s, "```json")
	s = strings.TrimPrefix(s, "```\n")
	s = strings.TrimPrefix(s, "```")
	s = strings.TrimSuffix(s, "\n```")
	s = strings.TrimSuffix(s, "```")
	return strings.TrimSpace(s)
}
