package envelope

import (
	"regexp"
	"strings"

	cai "github.com/anthropic/cai"
)

// ClassifierRule maps a regex pattern to an ErrorCategory.
type ClassifierRule struct {
	Pattern  *regexp.Regexp
	Category cai.ErrorCategory
	Code     int
}

// Classifier categorizes raw stderr/exit codes into ErrorCategory.
type Classifier struct {
	rules []ClassifierRule
}

// DefaultClassifier returns a classifier with standard OS/script error patterns.
func DefaultClassifier() *Classifier {
	return &Classifier{
		rules: []ClassifierRule{
			// Dependency errors (HTTP 424)
			{regexp.MustCompile(`(?i)(ModuleNotFoundError|ImportError|no module named|cannot find module|package .* is not installed|command not found.*pip|command not found.*npm)`), cai.ErrDependency, 424},
			{regexp.MustCompile(`(?i)(Could not find a version|No matching distribution|ENOENT.*node_modules)`), cai.ErrDependency, 424},

			// Permission errors (HTTP 403)
			{regexp.MustCompile(`(?i)(permission denied|operation not permitted|EACCES|sudo required|access denied)`), cai.ErrPermission, 403},

			// Syntax errors (HTTP 400)
			{regexp.MustCompile(`(?i)(SyntaxError|IndentationError|TabError|unexpected token|parse error|unexpected end)`), cai.ErrSyntax, 400},
			{regexp.MustCompile(`(?i)(compilation error|compile error|unterminated string|invalid syntax)`), cai.ErrSyntax, 400},

			// Logic/format errors (HTTP 422)
			{regexp.MustCompile(`(?i)(AssertionError|ValueError|TypeError|KeyError|IndexError|AttributeError)`), cai.ErrLogic, 422},
			{regexp.MustCompile(`(?i)(invalid json|json parse|unexpected end of json|cannot unmarshal)`), cai.ErrLogic, 422},

			// Timeout (HTTP 408)
			{regexp.MustCompile(`(?i)(timeout|timed out|deadline exceeded|context deadline)`), cai.ErrTimeout, 408},
		},
	}
}

// Classify examines exit code and stderr to produce an ExecutionEnvelope.
func (c *Classifier) Classify(exitCode int, stdout, stderr string) cai.ExecutionEnvelope {
	if exitCode == 0 && stderr == "" {
		return cai.ExecutionEnvelope{
			StatusCode: 200,
			Category:   cai.StatusSuccess,
			RawStdout:  stdout,
			ExitCode:   0,
		}
	}

	combined := stderr
	if combined == "" {
		combined = stdout
	}

	for _, rule := range c.rules {
		if rule.Pattern.MatchString(combined) {
			return cai.ExecutionEnvelope{
				StatusCode: rule.Code,
				Category:   rule.Category,
				RawStdout:  stdout,
				RawStderr:  stderr,
				ExitCode:   exitCode,
			}
		}
	}

	if exitCode == 0 {
		return cai.ExecutionEnvelope{
			StatusCode: 200,
			Category:   cai.StatusSuccess,
			RawStdout:  stdout,
			RawStderr:  stderr,
			ExitCode:   0,
		}
	}

	return cai.ExecutionEnvelope{
		StatusCode: 500,
		Category:   cai.ErrUnknown,
		RawStdout:  stdout,
		RawStderr:  stderr,
		ExitCode:   exitCode,
	}
}

// AddRule appends a custom classification rule.
func (c *Classifier) AddRule(pattern string, category cai.ErrorCategory, code int) error {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return err
	}
	c.rules = append(c.rules, ClassifierRule{Pattern: re, Category: category, Code: code})
	return nil
}

// DigestStderr produces a ≤100 char summary for EpochSummary (Brain never sees raw stderr).
func DigestStderr(stderr string, maxLen int) string {
	if maxLen <= 0 {
		maxLen = 100
	}
	s := strings.TrimSpace(stderr)
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\r", "")
	if len(s) > maxLen {
		s = s[:maxLen-3] + "..."
	}
	return s
}
