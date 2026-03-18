package memory

import (
	"encoding/json"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// ProbeSystem detects the current machine environment and writes to SystemEvolution.
func ProbeSystem(db *SQLiteDB) error {
	now := time.Now().UTC().Format(time.RFC3339)

	pairs := []struct{ key, value string }{
		{"os_version", probeOS()},
		{"architecture", runtime.GOARCH},
		{"shell", probeShell()},
		{"preferred_pkg_manager", probePkgManager()},
		{"installed_runtimes", probeRuntimes()},
		{"last_probe_at", now},
	}

	for _, p := range pairs {
		if err := db.UpdateSystemEvolution(p.key, p.value); err != nil {
			return err
		}
	}
	return nil
}

func probeOS() string {
	switch runtime.GOOS {
	case "darwin":
		out, err := exec.Command("sw_vers", "-productVersion").Output()
		if err == nil {
			return "macOS_" + strings.TrimSpace(string(out))
		}
		return "macOS"
	case "linux":
		out, err := os.ReadFile("/etc/os-release")
		if err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				if strings.HasPrefix(line, "PRETTY_NAME=") {
					return strings.Trim(strings.TrimPrefix(line, "PRETTY_NAME="), "\"")
				}
			}
		}
		return "Linux"
	default:
		return runtime.GOOS
	}
}

func probeShell() string {
	if sh := os.Getenv("SHELL"); sh != "" {
		parts := strings.Split(sh, "/")
		return parts[len(parts)-1]
	}
	return "sh"
}

func probePkgManager() string {
	candidates := []string{"uv", "pip3", "pip", "brew", "apt", "dnf", "pacman"}
	for _, c := range candidates {
		if _, err := exec.LookPath(c); err == nil {
			return c
		}
	}
	return ""
}

func probeRuntimes() string {
	var runtimes []string
	checks := map[string]string{
		"python3": "--version",
		"node":    "--version",
		"bun":     "--version",
		"deno":    "--version",
		"go":      "version",
	}
	for cmd, flag := range checks {
		if out, err := exec.Command(cmd, flag).Output(); err == nil {
			ver := strings.TrimSpace(string(out))
			if idx := strings.LastIndex(ver, " "); idx > 0 {
				ver = ver[idx+1:]
			}
			runtimes = append(runtimes, cmd+"/"+strings.TrimPrefix(ver, "v"))
		}
	}
	data, _ := json.Marshal(runtimes)
	return string(data)
}
