package skill

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	cai "github.com/anthropic/cai"
	"gopkg.in/yaml.v3"
)

// Meta holds parsed SKILL.md metadata.
type Meta struct {
	Name          string           `json:"name"`
	Version       string           `json:"version"`
	Author        string           `json:"author"`
	Engines       []cai.EngineType `json:"engines"`
	MinCAIVersion string           `json:"min_cai_version"`
	Description   string           `json:"description"`
	IntentNames   []string         `json:"intent_names"`
	Tags          []string         `json:"tags"`
	LoadedAt      time.Time        `json:"loaded_at"`
	SkillDir      string           `json:"skill_dir"`
}

// IntentYAML is the structure of an intent YAML file inside a skill.
type IntentYAML struct {
	IntentAction      string `yaml:"intent_action"`
	SystemPromptHints string `yaml:"system_prompt_hints"`
	ContextHints      string `yaml:"context_hints"`
	Version           int    `yaml:"version"`
	Frozen            bool   `yaml:"frozen"`
	Engine            string `yaml:"engine"`
}

// Loader loads SKILL.md directories and registers IntentMemory entries.
type Loader struct {
	memDB cai.MemoryDB
}

// NewLoader creates a skill loader.
func NewLoader(memDB cai.MemoryDB) *Loader {
	return &Loader{memDB: memDB}
}

// Load reads a skill directory and populates IntentMemory.
// Returns the number of intents loaded (excludes those already evolved locally).
func (l *Loader) Load(ctx context.Context, skillDir string) (*Meta, int, error) {
	manifestPath := filepath.Join(skillDir, "SKILL.md")
	if _, err := os.Stat(manifestPath); err != nil {
		return nil, 0, fmt.Errorf("SKILL.md not found in %s", skillDir)
	}

	meta, err := parseManifest(manifestPath)
	if err != nil {
		return nil, 0, fmt.Errorf("parse SKILL.md: %w", err)
	}
	meta.LoadedAt = time.Now().UTC()
	meta.SkillDir = skillDir

	loaded := 0
	intentsDir := filepath.Join(skillDir, "intents")
	entries, err := os.ReadDir(intentsDir)
	if err != nil {
		return meta, 0, nil
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !strings.HasSuffix(name, ".yaml") && !strings.HasSuffix(name, ".yml") {
			continue
		}

		intentPath := filepath.Join(intentsDir, name)
		intent, err := parseIntentYAML(intentPath)
		if err != nil {
			continue
		}

		if intent.IntentAction == "" {
			intent.IntentAction = strings.TrimSuffix(strings.TrimSuffix(name, ".yaml"), ".yml")
		}
		meta.IntentNames = append(meta.IntentNames, intent.IntentAction)

		if l.memDB == nil {
			loaded++
			continue
		}

		// Local evolution wins: if DB has a higher version, skip
		if existing, found := l.memDB.LoadIntentMemory(intent.IntentAction); found {
			if existing.Version >= intent.Version {
				continue
			}
		}

		l.memDB.SaveIntentMemory(cai.IntentMemory{
			IntentAction:      intent.IntentAction,
			SystemPromptHints: intent.SystemPromptHints,
			ContextHints:      intent.ContextHints,
			Frozen:            intent.Frozen,
			Version:           intent.Version,
		})
		loaded++
	}

	return meta, loaded, nil
}

func parseIntentYAML(path string) (*IntentYAML, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var intent IntentYAML
	if err := yaml.Unmarshal(data, &intent); err != nil {
		return nil, err
	}
	if intent.Version == 0 {
		intent.Version = 1
	}
	return &intent, nil
}

// LoadSkillDirs loads all skill directories from a parent directory.
func LoadSkillDirs(ctx context.Context, loader *Loader, parentDir string) ([]*Meta, error) {
	entries, err := os.ReadDir(parentDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var skills []*Meta
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		skillDir := filepath.Join(parentDir, entry.Name())
		if _, err := os.Stat(filepath.Join(skillDir, "SKILL.md")); err != nil {
			continue
		}
		meta, _, err := loader.Load(ctx, skillDir)
		if err != nil {
			continue
		}
		skills = append(skills, meta)
	}
	return skills, nil
}

func parseManifest(path string) (*Meta, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	meta := &Meta{}
	scanner := bufio.NewScanner(f)
	inMeta := false

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if line == "## Meta" {
			inMeta = true
			continue
		}
		if strings.HasPrefix(line, "## ") && inMeta {
			inMeta = false
		}

		if inMeta && strings.HasPrefix(line, "- ") {
			parts := strings.SplitN(line[2:], ":", 2)
			if len(parts) != 2 {
				continue
			}
			key := strings.TrimSpace(parts[0])
			val := strings.TrimSpace(parts[1])

			switch key {
			case "name":
				meta.Name = val
			case "version":
				meta.Version = val
			case "author":
				meta.Author = val
			case "min_cai_version":
				meta.MinCAIVersion = val
			case "engines":
				val = strings.Trim(val, "[]")
				for _, e := range strings.Split(val, ",") {
					meta.Engines = append(meta.Engines, cai.EngineType(strings.TrimSpace(e)))
				}
			}
		}

		if strings.HasPrefix(line, "## Description") {
			if scanner.Scan() {
				meta.Description = strings.TrimSpace(scanner.Text())
			}
		}

		if strings.HasPrefix(line, "## Tags") {
			for scanner.Scan() {
				tagLine := strings.TrimSpace(scanner.Text())
				if tagLine == "" || strings.HasPrefix(tagLine, "##") {
					break
				}
				if strings.HasPrefix(tagLine, "- ") {
					meta.Tags = append(meta.Tags, strings.TrimSpace(tagLine[2:]))
				}
			}
		}
	}

	return meta, scanner.Err()
}
