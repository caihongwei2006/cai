package prompt

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	cai "github.com/anthropic/cai"
)

func sha256Hash(data []byte) [32]byte {
	return sha256.Sum256(data)
}

const MaxVersions = 4 // v0 (initial) + 3 most recent

// PromptFile is the JSON structure for prompt scaffold files.
type PromptFile struct {
	Action       string         `json:"action"`
	Version      int            `json:"version"`
	Frozen       bool           `json:"frozen"`
	SystemPrompt string         `json:"system_prompt"`
	ContextHints string         `json:"context_hints,omitempty"`
	Engine       cai.EngineType `json:"engine,omitempty"`
	Metadata     *PromptMeta    `json:"metadata,omitempty"`
}

// PromptMeta holds runtime statistics (written by framework, read-only for user).
type PromptMeta struct {
	SuccessStreak       int    `json:"success_streak"`
	ConsecutiveFailures int    `json:"consecutive_failures"`
	LastUsed            string `json:"last_used,omitempty"`
}

// Dir returns the default prompts directory.
func Dir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cai", "prompts")
}

// LoadAll reads all active prompt files (non-versioned) from the given directory.
// Returns a map of action → PromptFile.
func LoadAll(dir string) (map[string]*PromptFile, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	result := make(map[string]*PromptFile)
	versionRe := regexp.MustCompile(`_v\d+\.json$`)

	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".json") {
			continue
		}
		if versionRe.MatchString(name) {
			continue // skip versioned files
		}

		pf, err := loadFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		if pf.Action == "" {
			pf.Action = strings.TrimSuffix(name, ".json")
		}
		result[pf.Action] = pf
	}

	return result, nil
}

// SeedToDB writes prompt files into MemoryDB as IntentMemory entries.
// Only seeds if the action doesn't already exist in DB (local evolution wins).
func SeedToDB(prompts map[string]*PromptFile, db cai.MemoryDB) int {
	seeded := 0
	for action, pf := range prompts {
		if _, exists := db.LoadIntentMemory(action); exists {
			continue
		}
		db.SaveIntentMemory(cai.IntentMemory{
			IntentAction:      action,
			SystemPromptHints: pf.SystemPrompt,
			ContextHints:      pf.ContextHints,
			Frozen:            pf.Frozen,
			Version:           1,
		})
		seeded++
	}
	return seeded
}

// WriteVersion creates a versioned copy of a prompt file.
// Maintains retention: v0 (initial, permanent) + most recent 3 versions.
func WriteVersion(dir, action string, mem cai.IntentMemory) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	pf := PromptFile{
		Action:       action,
		Version:      mem.Version,
		Frozen:       mem.Frozen,
		SystemPrompt: mem.SystemPromptHints,
		ContextHints: mem.ContextHints,
		Metadata: &PromptMeta{
			SuccessStreak:       mem.SuccessStreak,
			ConsecutiveFailures: mem.ConsecutiveFailures,
			LastUsed:            mem.LastUsedAt.Format(time.RFC3339),
		},
	}

	// Write active file
	activePath := filepath.Join(dir, action+".json")
	if err := writeJSON(activePath, pf); err != nil {
		return fmt.Errorf("write active: %w", err)
	}

	// Write versioned copy
	versionPath := filepath.Join(dir, fmt.Sprintf("%s_v%d.json", action, mem.Version))
	if err := writeJSON(versionPath, pf); err != nil {
		return fmt.Errorf("write version: %w", err)
	}

	// Ensure v0 exists (initial version, permanent)
	v0Path := filepath.Join(dir, action+"_v0.json")
	if _, err := os.Stat(v0Path); os.IsNotExist(err) {
		writeJSON(v0Path, pf)
	}

	// Enforce retention: keep v0 + most recent (MaxVersions-1) versions
	pruneVersions(dir, action)

	return nil
}

// Scaffold generates an empty prompt file for a new action.
func Scaffold(dir, action string, engine cai.EngineType) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	path := filepath.Join(dir, action+".json")
	if _, err := os.Stat(path); err == nil {
		return nil // already exists
	}

	pf := PromptFile{
		Action:       action,
		Version:      0,
		Frozen:       false,
		SystemPrompt: "",
		Engine:       engine,
	}
	return writeJSON(path, pf)
}

func pruneVersions(dir, action string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}

	prefix := action + "_v"
	var versioned []struct {
		path    string
		version int
	}

	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, ".json") {
			continue
		}
		numStr := strings.TrimPrefix(strings.TrimSuffix(name, ".json"), prefix)
		var v int
		if _, err := fmt.Sscanf(numStr, "%d", &v); err != nil {
			continue
		}
		versioned = append(versioned, struct {
			path    string
			version int
		}{filepath.Join(dir, name), v})
	}

	if len(versioned) <= MaxVersions {
		return
	}

	sort.Slice(versioned, func(i, j int) bool {
		return versioned[i].version < versioned[j].version
	})

	// Keep v0 (index 0) and the last (MaxVersions-1) entries
	toKeep := make(map[int]bool)
	toKeep[0] = true // v0 always kept
	for i := len(versioned) - (MaxVersions - 1); i < len(versioned); i++ {
		if i >= 0 {
			toKeep[i] = true
		}
	}

	for i, v := range versioned {
		if !toKeep[i] {
			os.Remove(v.path)
		}
	}
}

func loadFile(path string) (*PromptFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var pf PromptFile
	if err := json.Unmarshal(data, &pf); err != nil {
		return nil, err
	}
	return &pf, nil
}

func writeJSON(path string, v interface{}) error {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// --- Workspace Document (.md) Support ---

const (
	DocMaxVersions = 4 // v0 (initial) + 3 most recent
	MaxDocBytes    = 2 * 1024 * 1024 // 2MB per file guard
)

// WorkspaceDocNames lists the standard workspace bootstrap files.
var WorkspaceDocNames = []string{
	"AGENTS.md", "SOUL.md", "TOOLS.md", "IDENTITY.md",
	"USER.md", "HEARTBEAT.md", "BOOTSTRAP.md", "MEMORY.md",
}

// LoadWorkspaceDocs reads .md files from a workspace directory.
// Returns a map of filename -> WorkspaceDocument. Only files that exist are included.
func LoadWorkspaceDocs(dir string) (map[string]*cai.WorkspaceDocument, error) {
	result := make(map[string]*cai.WorkspaceDocument)

	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return result, nil
		}
		return nil, err
	}

	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".md") {
			continue
		}
		// Skip versioned files (e.g. SOUL_v1.md)
		if versionedMdRe.MatchString(name) {
			continue
		}

		filePath := filepath.Join(dir, name)
		info, err := entry.Info()
		if err != nil || info.Size() > int64(MaxDocBytes) {
			continue
		}

		content, err := os.ReadFile(filePath)
		if err != nil {
			continue
		}

		hash := hashContent(content)
		result[name] = &cai.WorkspaceDocument{
			Name:        name,
			Path:        filePath,
			Content:     string(content),
			Version:     1,
			ContentHash: hash,
		}
	}

	return result, nil
}

var versionedMdRe = regexp.MustCompile(`_v\d+\.md$`)

// SeedDocsToDB writes workspace documents into WorkspaceDocStore.
// Only seeds if the document doesn't already exist or if the disk hash has changed.
func SeedDocsToDB(docs map[string]*cai.WorkspaceDocument, ds cai.WorkspaceDocStore) int {
	seeded := 0
	for name, doc := range docs {
		existing, found := ds.LoadDocument(name)
		if found && existing.ContentHash == doc.ContentHash {
			continue // unchanged on disk
		}
		if found && existing.Version > 1 {
			// Document has been evolved in SQLite; don't overwrite with disk version
			// unless disk hash is newer than the last seed
			if existing.ContentHash == doc.ContentHash {
				continue
			}
		}
		ds.SaveDocument(*doc)
		seeded++
	}
	return seeded
}

// WriteDocumentVersion creates a versioned copy of a workspace document (.md_vN).
func WriteDocumentVersion(dir, name string, doc cai.WorkspaceDocument) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	baseName := strings.TrimSuffix(name, ".md")

	// Write active file
	activePath := filepath.Join(dir, name)
	if err := os.WriteFile(activePath, []byte(doc.Content), 0644); err != nil {
		return fmt.Errorf("write active: %w", err)
	}

	// Write versioned copy
	versionPath := filepath.Join(dir, fmt.Sprintf("%s_v%d.md", baseName, doc.Version))
	if err := os.WriteFile(versionPath, []byte(doc.Content), 0644); err != nil {
		return fmt.Errorf("write version: %w", err)
	}

	// Ensure v0 exists (initial version, permanent)
	v0Path := filepath.Join(dir, baseName+"_v0.md")
	if _, err := os.Stat(v0Path); os.IsNotExist(err) {
		os.WriteFile(v0Path, []byte(doc.Content), 0644)
	}

	pruneDocVersions(dir, baseName)
	return nil
}

func pruneDocVersions(dir, baseName string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}

	prefix := baseName + "_v"
	var versioned []struct {
		path    string
		version int
	}

	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasPrefix(name, prefix) || !strings.HasSuffix(name, ".md") {
			continue
		}
		numStr := strings.TrimPrefix(strings.TrimSuffix(name, ".md"), prefix)
		var v int
		if _, err := fmt.Sscanf(numStr, "%d", &v); err != nil {
			continue
		}
		versioned = append(versioned, struct {
			path    string
			version int
		}{filepath.Join(dir, name), v})
	}

	if len(versioned) <= DocMaxVersions {
		return
	}

	sort.Slice(versioned, func(i, j int) bool {
		return versioned[i].version < versioned[j].version
	})

	toKeep := make(map[int]bool)
	toKeep[0] = true
	for i := len(versioned) - (DocMaxVersions - 1); i < len(versioned); i++ {
		if i >= 0 {
			toKeep[i] = true
		}
	}

	for i, v := range versioned {
		if !toKeep[i] {
			os.Remove(v.path)
		}
	}
}

func hashContent(data []byte) string {
	h := sha256Hash(data)
	return fmt.Sprintf("%x", h)
}
