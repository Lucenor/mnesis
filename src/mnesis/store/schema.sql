-- LCM SQLite Schema
-- Applied idempotently via ImmutableStore.initialize()

PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;

-- ── Sessions ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    parent_id     TEXT REFERENCES sessions(id),
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    model_id      TEXT NOT NULL DEFAULT '',
    provider_id   TEXT NOT NULL DEFAULT '',
    agent         TEXT NOT NULL DEFAULT 'default',
    title         TEXT,
    is_active     INTEGER NOT NULL DEFAULT 1,
    metadata      TEXT,
    system_prompt TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_sessions_parent_id
    ON sessions(parent_id) WHERE parent_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_sessions_created_at
    ON sessions(created_at);

-- ── Messages ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    parent_id       TEXT REFERENCES messages(id),
    role            TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    created_at      INTEGER NOT NULL,
    agent           TEXT NOT NULL DEFAULT 'default',
    model_id        TEXT NOT NULL DEFAULT '',
    provider_id     TEXT NOT NULL DEFAULT '',
    is_summary      INTEGER NOT NULL DEFAULT 0,
    tokens_input    INTEGER DEFAULT 0,
    tokens_output   INTEGER DEFAULT 0,
    tokens_cache_read  INTEGER DEFAULT 0,
    tokens_cache_write INTEGER DEFAULT 0,
    tokens_total    INTEGER DEFAULT 0,
    cost            REAL DEFAULT 0.0,
    finish_reason   TEXT,
    error_code      TEXT,
    error_message_text TEXT,
    error_retriable INTEGER DEFAULT 0,
    mode            TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id
    ON messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_messages_session_summary
    ON messages(session_id, is_summary)
    WHERE is_summary = 1;

-- ── Message Parts ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS message_parts (
    id              TEXT PRIMARY KEY,
    message_id      TEXT NOT NULL REFERENCES messages(id),
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    part_type       TEXT NOT NULL,
    part_index      INTEGER NOT NULL DEFAULT 0,
    content         TEXT NOT NULL DEFAULT '',
    tool_name       TEXT,
    tool_call_id    TEXT,
    tool_state      TEXT,
    compacted_at    INTEGER,
    started_at      INTEGER,
    completed_at    INTEGER,
    token_estimate  INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_parts_message_id
    ON message_parts(message_id, part_index);

CREATE INDEX IF NOT EXISTS idx_parts_session_tool
    ON message_parts(session_id, part_type, compacted_at)
    WHERE part_type = 'tool';

-- ── File References ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS file_references (
    content_id          TEXT PRIMARY KEY,
    path                TEXT NOT NULL,
    file_type           TEXT NOT NULL DEFAULT 'unknown',
    token_count         INTEGER NOT NULL DEFAULT 0,
    exploration_summary TEXT NOT NULL DEFAULT '',
    created_at          INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_file_refs_path
    ON file_references(path);

-- ── Summary Nodes ─────────────────────────────────────────────────────────────
-- span_start/end_message_id have no FK constraint — they may reference pruned messages.
-- kind, parent_node_ids, superseded added in Phase 3 (DAG persistence).
-- For existing databases these columns are added via Python ALTER TABLE migration
-- in ImmutableStore.initialize().

CREATE TABLE IF NOT EXISTS summary_nodes (
    id                    TEXT PRIMARY KEY,
    session_id            TEXT NOT NULL REFERENCES sessions(id),
    level                 INTEGER NOT NULL DEFAULT 0,
    span_start_message_id TEXT NOT NULL DEFAULT '',
    span_end_message_id   TEXT NOT NULL DEFAULT '',
    content               TEXT NOT NULL DEFAULT '',
    token_count           INTEGER NOT NULL DEFAULT 0,
    created_at            INTEGER NOT NULL,
    parent_node_id        TEXT,
    model_id              TEXT NOT NULL DEFAULT '',
    provider_id           TEXT NOT NULL DEFAULT '',
    compaction_level      INTEGER NOT NULL DEFAULT 1,
    is_active             INTEGER NOT NULL DEFAULT 1,
    kind                  TEXT NOT NULL DEFAULT 'leaf',
    parent_node_ids       TEXT NOT NULL DEFAULT '[]',
    superseded            INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_summary_nodes_session
    ON summary_nodes(session_id, is_active, level);

CREATE INDEX IF NOT EXISTS idx_summary_nodes_span
    ON summary_nodes(session_id, span_start_message_id, span_end_message_id);

-- ── Context Items (Phase 3 — O(1) context assembly) ──────────────────────────
--
-- Tracks exactly what is "currently in context" for each session.
-- Each row is either a 'message' or a 'summary' that belongs in the assembled
-- context window. When compaction runs it atomically removes compacted message
-- rows and inserts the replacement summary row, so context assembly becomes a
-- single ordered SELECT rather than an O(n) backward scan.
--
-- position: monotonically increasing per-session; determines display order.

CREATE TABLE IF NOT EXISTS context_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL,
    item_type  TEXT    NOT NULL CHECK(item_type IN ('message', 'summary')),
    item_id    TEXT    NOT NULL,
    position   INTEGER NOT NULL,
    created_at TEXT    NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_context_items_session_pos
    ON context_items(session_id, position);

CREATE INDEX IF NOT EXISTS idx_context_items_session
    ON context_items(session_id);

-- Covering index for DELETE … WHERE session_id=? AND item_id IN (…)
-- used by swap_context_items() during compaction.
CREATE INDEX IF NOT EXISTS idx_context_items_session_item
    ON context_items(session_id, item_id);

-- idx_summary_nodes_active is created in ImmutableStore.initialize() after the
-- Phase 3 ALTER TABLE migration so that it is safe for existing databases where
-- the superseded column may not yet exist when executescript() runs.
