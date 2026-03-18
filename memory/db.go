package memory

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	cai "github.com/anthropic/cai"
	"github.com/anthropic/cai/envelope"
	_ "modernc.org/sqlite"
)

// SQLiteDB implements cai.MemoryDB with local SQLite persistence.
type SQLiteDB struct {
	db   *sql.DB
	mu   sync.RWMutex
	path string
}

// Open creates or opens a SQLite database at the given path.
func Open(dbPath string) (*SQLiteDB, error) {
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("mkdir %s: %w", dir, err)
	}
	db, err := sql.Open("sqlite", dbPath+"?_journal=WAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	db.SetMaxOpenConns(1) // SQLite single-writer
	db.SetMaxIdleConns(2)

	s := &SQLiteDB{db: db, path: dbPath}
	if err := s.migrate(); err != nil {
		db.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return s, nil
}

func (s *SQLiteDB) migrate() error {
	_, err := s.db.Exec(schema)
	return err
}

const schema = `
CREATE TABLE IF NOT EXISTS intent_memory (
    intent_action        TEXT PRIMARY KEY,
    system_prompt_hints  TEXT NOT NULL DEFAULT '',
    context_hints        TEXT DEFAULT '',
    frozen               INTEGER DEFAULT 0,
    success_streak       INTEGER DEFAULT 0,
    consecutive_failures INTEGER DEFAULT 0,
    last_used_at         TEXT,
    last_evicted_at      TEXT,
    version              INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS intent_memory_history (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    intent_action        TEXT NOT NULL,
    old_hints            TEXT NOT NULL,
    eviction_reason      TEXT NOT NULL,
    evicted_at           TEXT NOT NULL,
    version              INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS system_evolution (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS capability_profiles (
    capability_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    tool_allow    TEXT NOT NULL DEFAULT '[]',
    tags          TEXT NOT NULL DEFAULT '[]',
    version       INTEGER DEFAULT 1,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experience_events (
    event_id         TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    trace_id         TEXT NOT NULL,
    span_id          TEXT DEFAULT '',
    epoch_id         TEXT DEFAULT '',
    prompt_identity  TEXT NOT NULL,
    capability_id    TEXT NOT NULL,
    event_type       TEXT NOT NULL,
    tool_name        TEXT DEFAULT '',
    error_category   TEXT DEFAULT '',
    error_summary    TEXT DEFAULT '',
    error_detail     TEXT DEFAULT '',
    patch_text       TEXT DEFAULT '',
    visible_to_child INTEGER NOT NULL DEFAULT 0,
    created_at       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experience_prompt_created
ON experience_events(prompt_identity, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_experience_capability_created
ON experience_events(capability_id, created_at DESC);

CREATE TABLE IF NOT EXISTS compiled_experience_views (
    prompt_identity TEXT PRIMARY KEY,
    capability_id   TEXT NOT NULL,
    system_patch    TEXT NOT NULL DEFAULT '',
    tool_hints      TEXT NOT NULL DEFAULT '[]',
    failure_guards  TEXT NOT NULL DEFAULT '[]',
    source_event_ids TEXT NOT NULL DEFAULT '[]',
    max_tokens      INTEGER NOT NULL DEFAULT 0,
    compiled_at     INTEGER NOT NULL,
    version         INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS run_patches (
    patch_id         TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    span_id          TEXT DEFAULT '',
    prompt_identity  TEXT NOT NULL,
    capability_id    TEXT NOT NULL,
    patch_text       TEXT NOT NULL,
    source           TEXT NOT NULL,
    expires_at       INTEGER DEFAULT 0,
    created_at       INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_run_patches_prompt_created
ON run_patches(run_id, prompt_identity, created_at DESC);

CREATE TABLE IF NOT EXISTS traces (
    trace_id   TEXT PRIMARY KEY,
    objective  TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'running',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS spans (
    span_id      TEXT PRIMARY KEY,
    trace_id     TEXT NOT NULL REFERENCES traces(trace_id),
    plan_node_id TEXT NOT NULL,
    step_name    TEXT NOT NULL,
    active_epoch INTEGER DEFAULT 1,
    final_epoch  INTEGER DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS epochs (
    epoch_id     TEXT PRIMARY KEY,
    span_id      TEXT NOT NULL REFERENCES spans(span_id),
    epoch_number INTEGER NOT NULL,
    prompt       TEXT NOT NULL,
    result       TEXT DEFAULT '',
    error        TEXT DEFAULT '',
    category     TEXT NOT NULL DEFAULT 'SUCCESS',
    is_dead_end  INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    duration_ms  INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_epochs_span ON epochs(span_id, epoch_number);

CREATE TABLE IF NOT EXISTS hibernation (
    trace_id   TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_documents (
    name          TEXT PRIMARY KEY,
    path          TEXT NOT NULL DEFAULT '',
    content       TEXT NOT NULL DEFAULT '',
    version       INTEGER DEFAULT 1,
    frozen        INTEGER DEFAULT 0,
    content_hash  TEXT DEFAULT '',
    last_seed_at  TEXT,
    last_evolve_at TEXT
);

CREATE TABLE IF NOT EXISTS workspace_documents_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    old_content     TEXT NOT NULL,
    evolve_reason   TEXT NOT NULL,
    evolved_at      TEXT NOT NULL,
    version         INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    filepath  TEXT PRIMARY KEY,
    filename  TEXT NOT NULL,
    extension TEXT DEFAULT '',
    scope     INTEGER DEFAULT 0,
    is_dir    INTEGER DEFAULT 0,
    mod_time  TEXT,
    size      INTEGER DEFAULT 0
);

CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
    filepath, filename,
    tokenize='unicode61 remove_diacritics 1'
);
`

// --- IntentMemory ---

func (s *SQLiteDB) LoadIntentMemory(action string) (*cai.IntentMemory, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var m cai.IntentMemory
	var lastUsed, lastEvicted sql.NullString
	var frozen int
	err := s.db.QueryRow(
		`SELECT intent_action, system_prompt_hints, context_hints, frozen, success_streak, consecutive_failures, last_used_at, last_evicted_at, version FROM intent_memory WHERE intent_action = ?`,
		action,
	).Scan(&m.IntentAction, &m.SystemPromptHints, &m.ContextHints, &frozen, &m.SuccessStreak, &m.ConsecutiveFailures, &lastUsed, &lastEvicted, &m.Version)

	if err != nil {
		return nil, false
	}
	m.Frozen = frozen != 0
	if lastUsed.Valid {
		m.LastUsedAt, _ = time.Parse(time.RFC3339, lastUsed.String)
	}
	if lastEvicted.Valid {
		m.LastEvictedAt, _ = time.Parse(time.RFC3339, lastEvicted.String)
	}
	return &m, true
}

func (s *SQLiteDB) SaveIntentMemory(mem cai.IntentMemory) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		`INSERT INTO intent_memory (intent_action, system_prompt_hints, context_hints, frozen, success_streak, consecutive_failures, last_used_at, version)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(intent_action) DO UPDATE SET
		   system_prompt_hints = excluded.system_prompt_hints,
		   context_hints = excluded.context_hints,
		   frozen = excluded.frozen,
		   success_streak = excluded.success_streak,
		   consecutive_failures = excluded.consecutive_failures,
		   last_used_at = excluded.last_used_at,
		   version = excluded.version`,
		mem.IntentAction, mem.SystemPromptHints, mem.ContextHints,
		boolToInt(mem.Frozen), mem.SuccessStreak, mem.ConsecutiveFailures,
		time.Now().UTC().Format(time.RFC3339), mem.Version,
	)
	return err
}

func (s *SQLiteDB) EvictIntentMemory(action string, reason string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var oldHints string
	var version, frozen int
	err := s.db.QueryRow(`SELECT system_prompt_hints, version, frozen FROM intent_memory WHERE intent_action = ?`, action).Scan(&oldHints, &version, &frozen)
	if err != nil {
		return err
	}
	if frozen != 0 {
		return nil // frozen prompts are never evicted
	}

	now := time.Now().UTC().Format(time.RFC3339)
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(
		`INSERT INTO intent_memory_history (intent_action, old_hints, eviction_reason, evicted_at, version) VALUES (?, ?, ?, ?, ?)`,
		action, oldHints, envelope.DigestStderr(reason, 200), now, version,
	); err != nil {
		return err
	}

	if _, err := tx.Exec(
		`UPDATE intent_memory SET system_prompt_hints = '', consecutive_failures = 0, last_evicted_at = ?, version = version + 1 WHERE intent_action = ?`,
		now, action,
	); err != nil {
		return err
	}

	return tx.Commit()
}

func (s *SQLiteDB) IncrementFailure(action string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Upsert: create entry if not exists, then increment
	_, err := s.db.Exec(
		`INSERT INTO intent_memory (intent_action, consecutive_failures, success_streak)
		 VALUES (?, 1, 0)
		 ON CONFLICT(intent_action) DO UPDATE SET
		   consecutive_failures = consecutive_failures + 1,
		   success_streak = 0`,
		action,
	)
	if err != nil {
		return 0, err
	}

	var count int
	err = s.db.QueryRow(`SELECT consecutive_failures FROM intent_memory WHERE intent_action = ?`, action).Scan(&count)
	return count, err
}

func (s *SQLiteDB) ResetFailures(action string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		`UPDATE intent_memory SET consecutive_failures = 0, success_streak = success_streak + 1, last_used_at = ? WHERE intent_action = ?`,
		time.Now().UTC().Format(time.RFC3339), action,
	)
	return err
}

// --- SystemEvolution ---

func (s *SQLiteDB) LoadSystemEvolution() cai.SystemEvolutionMemory {
	s.mu.RLock()
	defer s.mu.RUnlock()

	mem := cai.SystemEvolutionMemory{
		KnownQuirks: make(map[string]string),
	}

	rows, err := s.db.Query(`SELECT key, value FROM system_evolution`)
	if err != nil {
		return mem
	}
	defer rows.Close()

	for rows.Next() {
		var k, v string
		if err := rows.Scan(&k, &v); err != nil {
			continue
		}
		switch k {
		case "os_version":
			mem.OSVersion = v
		case "architecture":
			mem.Architecture = v
		case "shell":
			mem.Shell = v
		case "preferred_pkg_manager":
			mem.PreferredPkgManager = v
		case "installed_runtimes":
			_ = json.Unmarshal([]byte(v), &mem.InstalledRuntimes)
		case "last_probe_at":
			mem.LastProbeAt, _ = time.Parse(time.RFC3339, v)
		default:
			if len(k) > 6 && k[:6] == "quirk:" {
				mem.KnownQuirks[k[6:]] = v
			}
		}
	}
	return mem
}

func (s *SQLiteDB) UpdateSystemEvolution(key, value string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		`INSERT INTO system_evolution (key, value, updated_at) VALUES (?, ?, ?)
		 ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`,
		key, value, time.Now().UTC().Format(time.RFC3339),
	)
	return err
}

func (s *SQLiteDB) SaveCapabilityProfile(profile cai.CapabilityProfile) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	toolAllow, _ := json.Marshal(profile.ToolAllow)
	tags, _ := json.Marshal(profile.Tags)
	if profile.Version <= 0 {
		profile.Version = 1
	}
	_, err := s.db.Exec(
		`INSERT INTO capability_profiles (capability_id, name, description, tool_allow, tags, version, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(capability_id) DO UPDATE SET
		   name = excluded.name,
		   description = excluded.description,
		   tool_allow = excluded.tool_allow,
		   tags = excluded.tags,
		   version = excluded.version,
		   updated_at = excluded.updated_at`,
		string(profile.ID), profile.Name, profile.Description, string(toolAllow), string(tags), profile.Version, time.Now().UTC().Format(time.RFC3339),
	)
	return err
}

func (s *SQLiteDB) LoadCapabilityProfile(id cai.CapabilityID) (*cai.CapabilityProfile, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var profile cai.CapabilityProfile
	var toolAllowJSON, tagsJSON string
	err := s.db.QueryRow(
		`SELECT capability_id, name, description, tool_allow, tags, version FROM capability_profiles WHERE capability_id = ?`,
		string(id),
	).Scan(&profile.ID, &profile.Name, &profile.Description, &toolAllowJSON, &tagsJSON, &profile.Version)
	if err != nil {
		return nil, false
	}
	_ = json.Unmarshal([]byte(toolAllowJSON), &profile.ToolAllow)
	_ = json.Unmarshal([]byte(tagsJSON), &profile.Tags)
	return &profile, true
}

func (s *SQLiteDB) AppendExperienceEvent(event cai.ExperienceEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		`INSERT INTO experience_events
		 (event_id, run_id, trace_id, span_id, epoch_id, prompt_identity, capability_id, event_type, tool_name, error_category, error_summary, error_detail, patch_text, visible_to_child, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		event.EventID, event.RunID, event.TraceID, event.SpanID, event.EpochID, event.PromptIdentity,
		string(event.CapabilityID), event.EventType, event.ToolName, event.ErrorCategory, event.ErrorSummary,
		event.ErrorDetail, event.PatchText, boolToInt(event.VisibleToChild), event.CreatedAt,
	)
	return err
}

func (s *SQLiteDB) ListExperienceEvents(promptIdentity string, capabilityID cai.CapabilityID, visibleOnly bool, limit int) ([]cai.ExperienceEvent, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if limit <= 0 {
		limit = 20
	}

	query := `SELECT event_id, run_id, trace_id, span_id, epoch_id, prompt_identity, capability_id, event_type, tool_name, error_category, error_summary, error_detail, patch_text, visible_to_child, created_at
		FROM experience_events
		WHERE (prompt_identity = ? OR capability_id = ?)`
	args := []any{promptIdentity, string(capabilityID)}
	if visibleOnly {
		query += ` AND visible_to_child = 1`
	}
	query += ` ORDER BY created_at DESC LIMIT ?`
	args = append(args, limit)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var events []cai.ExperienceEvent
	for rows.Next() {
		var event cai.ExperienceEvent
		var visible int
		if err := rows.Scan(
			&event.EventID, &event.RunID, &event.TraceID, &event.SpanID, &event.EpochID,
			&event.PromptIdentity, &event.CapabilityID, &event.EventType, &event.ToolName,
			&event.ErrorCategory, &event.ErrorSummary, &event.ErrorDetail, &event.PatchText,
			&visible, &event.CreatedAt,
		); err != nil {
			return nil, err
		}
		event.VisibleToChild = visible != 0
		events = append(events, event)
	}
	return events, rows.Err()
}

func (s *SQLiteDB) SaveCompiledExperienceView(view cai.CompiledExperienceView) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	toolHints, _ := json.Marshal(view.ToolHints)
	failureGuards, _ := json.Marshal(view.FailureGuards)
	sourceEventIDs, _ := json.Marshal(view.SourceEventIDs)
	if view.Version <= 0 {
		view.Version = 1
	}

	_, err := s.db.Exec(
		`INSERT INTO compiled_experience_views
		 (prompt_identity, capability_id, system_patch, tool_hints, failure_guards, source_event_ids, max_tokens, compiled_at, version)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(prompt_identity) DO UPDATE SET
		   capability_id = excluded.capability_id,
		   system_patch = excluded.system_patch,
		   tool_hints = excluded.tool_hints,
		   failure_guards = excluded.failure_guards,
		   source_event_ids = excluded.source_event_ids,
		   max_tokens = excluded.max_tokens,
		   compiled_at = excluded.compiled_at,
		   version = excluded.version`,
		view.PromptIdentity, string(view.CapabilityID), view.SystemPatch, string(toolHints), string(failureGuards),
		string(sourceEventIDs), view.MaxTokens, view.CompiledAt, view.Version,
	)
	return err
}

func (s *SQLiteDB) LoadCompiledExperienceView(promptIdentity string) (*cai.CompiledExperienceView, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var view cai.CompiledExperienceView
	var toolHintsJSON, failureGuardsJSON, sourceEventIDsJSON string
	err := s.db.QueryRow(
		`SELECT prompt_identity, capability_id, system_patch, tool_hints, failure_guards, source_event_ids, max_tokens, compiled_at, version
		 FROM compiled_experience_views WHERE prompt_identity = ?`,
		promptIdentity,
	).Scan(&view.PromptIdentity, &view.CapabilityID, &view.SystemPatch, &toolHintsJSON, &failureGuardsJSON, &sourceEventIDsJSON, &view.MaxTokens, &view.CompiledAt, &view.Version)
	if err != nil {
		return nil, false
	}
	view.ViewID = "view:" + view.PromptIdentity
	_ = json.Unmarshal([]byte(toolHintsJSON), &view.ToolHints)
	_ = json.Unmarshal([]byte(failureGuardsJSON), &view.FailureGuards)
	_ = json.Unmarshal([]byte(sourceEventIDsJSON), &view.SourceEventIDs)
	return &view, true
}

func (s *SQLiteDB) SaveRunPatch(patch cai.RunPatch) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		`INSERT INTO run_patches (patch_id, run_id, span_id, prompt_identity, capability_id, patch_text, source, expires_at, created_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		patch.PatchID, patch.RunID, patch.SpanID, patch.PromptIdentity, string(patch.CapabilityID),
		patch.PatchText, patch.Source, patch.ExpiresAt, patch.CreatedAt,
	)
	return err
}

func (s *SQLiteDB) ListRunPatches(runID string, promptIdentity string, capabilityID cai.CapabilityID, limit int) ([]cai.RunPatch, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if limit <= 0 {
		limit = 20
	}

	rows, err := s.db.Query(
		`SELECT patch_id, run_id, span_id, prompt_identity, capability_id, patch_text, source, expires_at, created_at
		 FROM run_patches
		 WHERE run_id = ? AND (prompt_identity = ? OR capability_id = ?)
		 ORDER BY created_at DESC LIMIT ?`,
		runID, promptIdentity, string(capabilityID), limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var patches []cai.RunPatch
	for rows.Next() {
		var patch cai.RunPatch
		if err := rows.Scan(
			&patch.PatchID, &patch.RunID, &patch.SpanID, &patch.PromptIdentity, &patch.CapabilityID,
			&patch.PatchText, &patch.Source, &patch.ExpiresAt, &patch.CreatedAt,
		); err != nil {
			return nil, err
		}
		patches = append(patches, patch)
	}
	return patches, rows.Err()
}

func (s *SQLiteDB) ClearRunPatches(runID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(`DELETE FROM run_patches WHERE run_id = ?`, runID)
	return err
}

// --- Traces / Spans / Epochs ---

func (s *SQLiteDB) CreateTrace(trace cai.AgentTrace) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		`INSERT INTO traces (trace_id, objective, status, created_at) VALUES (?, ?, ?, ?)`,
		trace.TraceID, trace.Objective, trace.Status, trace.CreatedAt.UTC().Format(time.RFC3339),
	)
	return err
}

func (s *SQLiteDB) UpdateTraceStatus(traceID, status string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		`UPDATE traces SET status = ? WHERE trace_id = ?`,
		status, traceID,
	)
	return err
}

func (s *SQLiteDB) CreateSpan(span cai.LogicalSpan) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		`INSERT INTO spans (span_id, trace_id, plan_node_id, step_name, active_epoch, status) VALUES (?, ?, ?, ?, ?, ?)`,
		span.SpanID, span.TraceID, span.PlanNodeID, span.StepName, span.ActiveEpoch, span.Status,
	)
	return err
}

func (s *SQLiteDB) UpdateSpanStatus(spanID, status string, finalEpoch int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(
		`UPDATE spans SET status = ?, final_epoch = ?, active_epoch = ? WHERE span_id = ?`,
		status, finalEpoch, finalEpoch, spanID,
	)
	return err
}

func (s *SQLiteDB) AppendEpoch(epoch cai.ExecutionEpoch) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(
		`INSERT INTO epochs (epoch_id, span_id, epoch_number, prompt, result, error, category, is_dead_end, created_at, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		epoch.EpochID, epoch.SpanID, epoch.EpochNumber, epoch.Prompt, epoch.Result,
		epoch.Error, string(epoch.Category), boolToInt(epoch.IsDeadEnd),
		epoch.CreatedAt.UTC().Format(time.RFC3339), epoch.DurationMs,
	); err != nil {
		return err
	}

	if _, err := tx.Exec(
		`UPDATE spans SET active_epoch = ? WHERE span_id = ?`,
		epoch.EpochNumber, epoch.SpanID,
	); err != nil {
		return err
	}

	return tx.Commit()
}

// ActiveEpoch returns only the current active epoch for a span (N(1) visibility).
func (s *SQLiteDB) ActiveEpoch(spanID string) (*cai.ExecutionEpoch, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var activeEpoch int
	if err := s.db.QueryRow(`SELECT active_epoch FROM spans WHERE span_id = ?`, spanID).Scan(&activeEpoch); err != nil {
		return nil, err
	}

	return s.scanEpoch(
		`SELECT epoch_id, span_id, epoch_number, prompt, result, error, category, is_dead_end, created_at, duration_ms
		 FROM epochs WHERE span_id = ? AND epoch_number = ?`,
		spanID, activeEpoch,
	)
}

// SpanHistory returns the complete epoch chain (for UI/audit).
func (s *SQLiteDB) SpanHistory(spanID string) ([]cai.ExecutionEpoch, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(
		`SELECT epoch_id, span_id, epoch_number, prompt, result, error, category, is_dead_end, created_at, duration_ms
		 FROM epochs WHERE span_id = ? ORDER BY epoch_number ASC`, spanID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var epochs []cai.ExecutionEpoch
	for rows.Next() {
		e, err := scanEpochRow(rows)
		if err != nil {
			return nil, err
		}
		epochs = append(epochs, e)
	}
	return epochs, rows.Err()
}

// SpanSummary returns categorized summaries (for Brain's on-demand history retrieval).
func (s *SQLiteDB) SpanSummary(spanID string, depth int) ([]cai.EpochSummary, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(
		`SELECT epoch_number, category, error, is_dead_end FROM epochs
		 WHERE span_id = ? ORDER BY epoch_number DESC LIMIT ?`,
		spanID, depth,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var summaries []cai.EpochSummary
	for rows.Next() {
		var es cai.EpochSummary
		var rawErr string
		var deadEnd int
		if err := rows.Scan(&es.EpochNumber, &es.Category, &rawErr, &deadEnd); err != nil {
			return nil, err
		}
		es.ErrorDigest = envelope.DigestStderr(rawErr, 100)
		es.IsDeadEnd = deadEnd != 0
		summaries = append(summaries, es)
	}
	return summaries, rows.Err()
}

// --- Hibernation (implements cai.StateStore) ---

func (s *SQLiteDB) Hibernate(_ context.Context, traceID string, state cai.HibernationState) error {
	state.TraceID = traceID
	if state.RequestID == "" {
		state.RequestID = traceID
	}
	if state.CreatedAt == 0 {
		state.CreatedAt = time.Now().UTC().UnixMilli()
	}
	if state.LatestObservationAt == 0 {
		state.LatestObservationAt = state.CreatedAt
	}

	stateJSON, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal state: %w", err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	_, err = s.db.Exec(
		`INSERT INTO hibernation (trace_id, state_json, created_at) VALUES (?, ?, ?)
		 ON CONFLICT(trace_id) DO UPDATE SET state_json = excluded.state_json, created_at = excluded.created_at`,
		traceID, string(stateJSON), time.Now().UTC().Format(time.RFC3339),
	)
	return err
}

func (s *SQLiteDB) Wake(_ context.Context, traceID string) (*cai.HibernationState, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var stateJSON string
	err := s.db.QueryRow(`SELECT state_json FROM hibernation WHERE trace_id = ?`, traceID).Scan(&stateJSON)
	if err != nil {
		return nil, err
	}

	var state cai.HibernationState
	if err := json.Unmarshal([]byte(stateJSON), &state); err != nil {
		return nil, fmt.Errorf("unmarshal state: %w", err)
	}
	return &state, nil
}

func (s *SQLiteDB) ListPending(_ context.Context) ([]cai.HibernationState, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(`SELECT state_json FROM hibernation ORDER BY created_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var states []cai.HibernationState
	for rows.Next() {
		var raw string
		if err := rows.Scan(&raw); err != nil {
			return nil, err
		}
		var state cai.HibernationState
		if err := json.Unmarshal([]byte(raw), &state); err != nil {
			return nil, fmt.Errorf("unmarshal state: %w", err)
		}
		states = append(states, state)
	}
	return states, rows.Err()
}

func (s *SQLiteDB) Clear(_ context.Context, traceID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(`DELETE FROM hibernation WHERE trace_id = ?`, traceID)
	return err
}

func (s *SQLiteDB) Close() error {
	return s.db.Close()
}

// --- WorkspaceDocStore implementation ---

func (s *SQLiteDB) LoadDocument(name string) (*cai.WorkspaceDocument, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var doc cai.WorkspaceDocument
	var frozen int
	var lastSeed, lastEvolve sql.NullString
	err := s.db.QueryRow(
		`SELECT name, path, content, version, frozen, content_hash, last_seed_at, last_evolve_at FROM workspace_documents WHERE name = ?`,
		name,
	).Scan(&doc.Name, &doc.Path, &doc.Content, &doc.Version, &frozen, &doc.ContentHash, &lastSeed, &lastEvolve)
	if err != nil {
		return nil, false
	}
	doc.Frozen = frozen != 0
	if lastSeed.Valid {
		doc.LastSeedAt, _ = time.Parse(time.RFC3339, lastSeed.String)
	}
	if lastEvolve.Valid {
		doc.LastEvolveAt, _ = time.Parse(time.RFC3339, lastEvolve.String)
	}
	return &doc, true
}

func (s *SQLiteDB) SaveDocument(doc cai.WorkspaceDocument) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	now := time.Now().UTC().Format(time.RFC3339)
	_, err := s.db.Exec(
		`INSERT INTO workspace_documents (name, path, content, version, frozen, content_hash, last_seed_at)
		 VALUES (?, ?, ?, ?, ?, ?, ?)
		 ON CONFLICT(name) DO UPDATE SET
		   path = excluded.path,
		   content = excluded.content,
		   version = excluded.version,
		   frozen = excluded.frozen,
		   content_hash = excluded.content_hash,
		   last_seed_at = excluded.last_seed_at`,
		doc.Name, doc.Path, doc.Content, doc.Version, boolToInt(doc.Frozen), doc.ContentHash, now,
	)
	return err
}

func (s *SQLiteDB) ListDocuments() ([]cai.WorkspaceDocument, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.Query(`SELECT name, path, content, version, frozen, content_hash, last_seed_at, last_evolve_at FROM workspace_documents ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var docs []cai.WorkspaceDocument
	for rows.Next() {
		var doc cai.WorkspaceDocument
		var frozen int
		var lastSeed, lastEvolve sql.NullString
		if err := rows.Scan(&doc.Name, &doc.Path, &doc.Content, &doc.Version, &frozen, &doc.ContentHash, &lastSeed, &lastEvolve); err != nil {
			return nil, err
		}
		doc.Frozen = frozen != 0
		if lastSeed.Valid {
			doc.LastSeedAt, _ = time.Parse(time.RFC3339, lastSeed.String)
		}
		if lastEvolve.Valid {
			doc.LastEvolveAt, _ = time.Parse(time.RFC3339, lastEvolve.String)
		}
		docs = append(docs, doc)
	}
	return docs, rows.Err()
}

func (s *SQLiteDB) EvolveDocument(name string, newContent string, reason string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	var oldContent string
	var version, frozen int
	err := s.db.QueryRow(`SELECT content, version, frozen FROM workspace_documents WHERE name = ?`, name).Scan(&oldContent, &version, &frozen)
	if err != nil {
		return fmt.Errorf("document %q not found: %w", name, err)
	}
	if frozen != 0 {
		return nil
	}

	now := time.Now().UTC().Format(time.RFC3339)
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(
		`INSERT INTO workspace_documents_history (name, old_content, evolve_reason, evolved_at, version) VALUES (?, ?, ?, ?, ?)`,
		name, oldContent, reason, now, version,
	); err != nil {
		return err
	}

	if _, err := tx.Exec(
		`UPDATE workspace_documents SET content = ?, version = version + 1, last_evolve_at = ? WHERE name = ?`,
		newContent, now, name,
	); err != nil {
		return err
	}

	return tx.Commit()
}

func (s *SQLiteDB) DeleteDocument(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(`DELETE FROM workspace_documents WHERE name = ?`, name)
	return err
}

// --- File Index (FTS5) ---

// UpsertFile inserts or updates a file entry and its FTS index.
func (s *SQLiteDB) UpsertFile(filepath, filename, ext string, isDir bool, modTime string, size int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	isDirInt := 0
	if isDir {
		isDirInt = 1
	}

	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.Exec(
		`INSERT INTO files (filepath, filename, extension, is_dir, mod_time, size) VALUES (?, ?, ?, ?, ?, ?)
		 ON CONFLICT(filepath) DO UPDATE SET filename=excluded.filename, extension=excluded.extension, is_dir=excluded.is_dir, mod_time=excluded.mod_time, size=excluded.size`,
		filepath, filename, ext, isDirInt, modTime, size,
	); err != nil {
		return err
	}

	// Sync FTS: delete old then insert new
	tx.Exec(`DELETE FROM files_fts WHERE filepath = ?`, filepath)
	if _, err := tx.Exec(`INSERT INTO files_fts (filepath, filename) VALUES (?, ?)`, filepath, filename); err != nil {
		return err
	}

	return tx.Commit()
}

// SearchFilesFTS queries the FTS5 index. Returns matching filepaths.
func (s *SQLiteDB) SearchFilesFTS(query string, limit int) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if limit <= 0 {
		limit = 50
	}

	// FTS5 match with prefix for fuzzy
	ftsQuery := query + "*"
	rows, err := s.db.Query(
		`SELECT f.filepath FROM files_fts fts JOIN files f ON fts.filepath = f.filepath WHERE files_fts MATCH ? LIMIT ?`,
		ftsQuery, limit,
	)
	if err != nil {
		// Fallback to LIKE if FTS fails
		rows, err = s.db.Query(
			`SELECT filepath FROM files WHERE filepath LIKE ? OR filename LIKE ? LIMIT ?`,
			"%"+query+"%", "%"+query+"%", limit,
		)
		if err != nil {
			return nil, err
		}
	}
	defer rows.Close()

	var results []string
	for rows.Next() {
		var fp string
		if err := rows.Scan(&fp); err != nil {
			continue
		}
		results = append(results, fp)
	}
	return results, rows.Err()
}

// ClearFileIndex removes all entries from the files table and FTS index.
func (s *SQLiteDB) ClearFileIndex() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.db.Exec(`DELETE FROM files_fts`)
	_, err := s.db.Exec(`DELETE FROM files`)
	return err
}

// FileCount returns the number of indexed files.
func (s *SQLiteDB) FileCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var count int
	s.db.QueryRow(`SELECT COUNT(*) FROM files`).Scan(&count)
	return count
}

// DB returns the underlying sql.DB for direct access (e.g., by file_index.go).
func (s *SQLiteDB) DB() *sql.DB {
	return s.db
}

// --- helpers ---

func (s *SQLiteDB) scanEpoch(query string, args ...interface{}) (*cai.ExecutionEpoch, error) {
	row := s.db.QueryRow(query, args...)
	var e cai.ExecutionEpoch
	var cat string
	var deadEnd int
	var createdAt string
	if err := row.Scan(&e.EpochID, &e.SpanID, &e.EpochNumber, &e.Prompt, &e.Result, &e.Error, &cat, &deadEnd, &createdAt, &e.DurationMs); err != nil {
		return nil, err
	}
	e.Category = cai.ErrorCategory(cat)
	e.IsDeadEnd = deadEnd != 0
	e.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
	return &e, nil
}

type scannable interface {
	Scan(dest ...interface{}) error
}

func scanEpochRow(row scannable) (cai.ExecutionEpoch, error) {
	var e cai.ExecutionEpoch
	var cat string
	var deadEnd int
	var createdAt string
	err := row.Scan(&e.EpochID, &e.SpanID, &e.EpochNumber, &e.Prompt, &e.Result, &e.Error, &cat, &deadEnd, &createdAt, &e.DurationMs)
	e.Category = cai.ErrorCategory(cat)
	e.IsDeadEnd = deadEnd != 0
	e.CreatedAt, _ = time.Parse(time.RFC3339, createdAt)
	return e, err
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
