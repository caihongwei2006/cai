package skill

import (
	"fmt"
	"path/filepath"
	"strings"
)

// BuildAvailableSkillsPrompt generates the <available_skills> XML block
// that is injected into the system prompt. The agent uses the read tool
// to load the full SKILL.md on demand.
func BuildAvailableSkillsPrompt(skills []*Meta) string {
	if len(skills) == 0 {
		return ""
	}

	var b strings.Builder
	b.WriteString("<available_skills description=\"Skills the agent can use. Use the read tool with the provided path to fetch full contents.\">\n")

	for _, s := range skills {
		skillMdPath := filepath.Join(s.SkillDir, "SKILL.md")
		b.WriteString(fmt.Sprintf("<agent_skill fullPath=%q>", skillMdPath))
		desc := s.Description
		if desc == "" {
			desc = s.Name
		}
		b.WriteString(desc)
		b.WriteString("</agent_skill>\n")
	}

	b.WriteString("</available_skills>")
	return b.String()
}

// BuildSkillsSection returns the full Skills section for the system prompt,
// including instructions for the agent on how to use skills.
func BuildSkillsSection(skills []*Meta, readToolName string) string {
	skillsPrompt := BuildAvailableSkillsPrompt(skills)
	if skillsPrompt == "" {
		return ""
	}

	if readToolName == "" {
		readToolName = "read"
	}

	var b strings.Builder
	b.WriteString("## Skills (mandatory)\n")
	b.WriteString("Before replying: scan <available_skills> <description> entries.\n")
	b.WriteString(fmt.Sprintf("- If exactly one skill clearly applies: read its SKILL.md at <location> with `%s`, then follow it.\n", readToolName))
	b.WriteString("- If multiple could apply: choose the most specific one, then read/follow it.\n")
	b.WriteString("- If none clearly apply: do not read any SKILL.md.\n")
	b.WriteString("Constraints: never read more than one skill up front; only read after selecting.\n\n")
	b.WriteString(skillsPrompt)
	b.WriteString("\n")
	return b.String()
}
