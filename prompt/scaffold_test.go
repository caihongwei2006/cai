package prompt_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	cai "github.com/anthropic/cai"
	"github.com/anthropic/cai/memory"
	"github.com/anthropic/cai/prompt"
)

func TestScaffoldAndSeed(t *testing.T) {
	dir := t.TempDir()

	prompt.Scaffold(dir, "install_pkg", cai.EngineBash)

	// User fills in system_prompt
	pf := prompt.PromptFile{
		Action:       "install_pkg",
		SystemPrompt: "Use uv. Output ONLY: uv install <pkg>",
		Engine:       cai.EngineBash,
	}
	writeFile(t, filepath.Join(dir, "install_pkg.json"), pf)

	prompts, _ := prompt.LoadAll(dir)
	if len(prompts) != 1 {
		t.Fatalf("expected 1, got %d", len(prompts))
	}

	db, _ := memory.Open(filepath.Join(t.TempDir(), "t.db"))
	defer db.Close()

	if n := prompt.SeedToDB(prompts, db); n != 1 {
		t.Errorf("expected 1 seeded, got %d", n)
	}

	mem, found := db.LoadIntentMemory("install_pkg")
	if !found || mem.SystemPromptHints != pf.SystemPrompt {
		t.Error("seed failed")
	}

	// Re-seed skips (local evolution wins)
	if n := prompt.SeedToDB(prompts, db); n != 0 {
		t.Error("should not re-seed")
	}
}

func TestVersionRetention(t *testing.T) {
	dir := t.TempDir()

	for v := 0; v <= 6; v++ {
		prompt.WriteVersion(dir, "deploy", cai.IntentMemory{
			IntentAction:      "deploy",
			SystemPromptHints: "v" + string(rune('0'+v)),
			Version:           v,
		})
	}

	// v0 must survive
	if _, err := os.Stat(filepath.Join(dir, "deploy_v0.json")); err != nil {
		t.Error("v0 must be permanent")
	}

	// Count versioned files (excluding active)
	entries, _ := os.ReadDir(dir)
	versions := 0
	for _, e := range entries {
		if e.Name() != "deploy.json" {
			versions++
		}
	}
	if versions > prompt.MaxVersions {
		t.Errorf("expected max %d versions, got %d", prompt.MaxVersions, versions)
	}
}

func TestLoadSkipsVersioned(t *testing.T) {
	dir := t.TempDir()

	writeFile(t, filepath.Join(dir, "git.json"), prompt.PromptFile{Action: "git", SystemPrompt: "active"})
	writeFile(t, filepath.Join(dir, "git_v0.json"), prompt.PromptFile{Action: "git", SystemPrompt: "old"})
	writeFile(t, filepath.Join(dir, "git_v1.json"), prompt.PromptFile{Action: "git", SystemPrompt: "v1"})

	prompts, _ := prompt.LoadAll(dir)
	if len(prompts) != 1 || prompts["git"].SystemPrompt != "active" {
		t.Error("should load only active file")
	}
}

func writeFile(t *testing.T, path string, v interface{}) {
	t.Helper()
	data, _ := json.MarshalIndent(v, "", "  ")
	os.WriteFile(path, data, 0644)
}
