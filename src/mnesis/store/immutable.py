"""Append-only SQLite-backed message store."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import structlog

from mnesis.models.config import StoreConfig

if TYPE_CHECKING:
    from mnesis.store.pool import StorePool
from mnesis.models.message import (
    Message,
    MessageError,
    MessagePart,
    MessageWithParts,
    TokenUsage,
)
from mnesis.models.summary import FileReference

# ── Exceptions ─────────────────────────────────────────────────────────────────


class MnesisStoreError(Exception):
    """Base class for store errors."""


class SessionNotFoundError(MnesisStoreError):
    """Raised when a session_id does not exist in the store."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session not found: {session_id!r}")
        self.session_id = session_id


class MessageNotFoundError(MnesisStoreError):
    """Raised when a message_id does not exist in the store."""

    def __init__(self, message_id: str) -> None:
        super().__init__(f"Message not found: {message_id!r}")
        self.message_id = message_id


class PartNotFoundError(MnesisStoreError):
    """Raised when a part_id does not exist in the store."""

    def __init__(self, part_id: str) -> None:
        super().__init__(f"Part not found: {part_id!r}")
        self.part_id = part_id


class DuplicateIDError(MnesisStoreError):
    """Raised when attempting to insert a record with a duplicate primary key."""

    def __init__(self, record_id: str) -> None:
        super().__init__(f"Duplicate ID: {record_id!r}")
        self.record_id = record_id


class ImmutableFieldError(MnesisStoreError):
    """Raised when attempting to modify an immutable field."""


# ── Internal raw storage model ─────────────────────────────────────────────────


class Session:
    """Thin data class for session rows (not Pydantic — avoids heavy validation on reads)."""

    __slots__ = (
        "agent",
        "created_at",
        "id",
        "is_active",
        "metadata",
        "model_id",
        "parent_id",
        "provider_id",
        "system_prompt",
        "title",
        "updated_at",
    )

    def __init__(
        self,
        id: str,
        parent_id: str | None,
        created_at: int,
        updated_at: int,
        model_id: str,
        provider_id: str,
        agent: str,
        title: str | None,
        is_active: bool,
        metadata: dict[str, Any] | None,
        system_prompt: str = "",
    ) -> None:
        self.id = id
        self.parent_id = parent_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.model_id = model_id
        self.provider_id = provider_id
        self.agent = agent
        self.title = title
        self.is_active = is_active
        self.metadata = metadata
        self.system_prompt = system_prompt


class RawMessagePart:
    """Internal storage model for a message part row."""

    __slots__ = (
        "compacted_at",
        "completed_at",
        "content",
        "id",
        "message_id",
        "part_index",
        "part_type",
        "session_id",
        "started_at",
        "token_estimate",
        "tool_call_id",
        "tool_name",
        "tool_state",
    )

    def __init__(
        self,
        id: str,
        message_id: str,
        session_id: str,
        part_type: str,
        content: str,
        part_index: int = 0,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_state: str | None = None,
        compacted_at: int | None = None,
        started_at: int | None = None,
        completed_at: int | None = None,
        token_estimate: int = 0,
    ) -> None:
        self.id = id
        self.message_id = message_id
        self.session_id = session_id
        self.part_type = part_type
        self.part_index = part_index
        self.content = content
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.tool_state = tool_state
        self.compacted_at = compacted_at
        self.started_at = started_at
        self.completed_at = completed_at
        self.token_estimate = token_estimate


# ── ImmutableStore ─────────────────────────────────────────────────────────────


class ImmutableStore:
    """
    Append-only, SQLite-backed message log.

    All writes use transactions. Part content is immutable — only status
    metadata (tool_state, compacted_at, timing, output) can be updated
    via ``update_part_status()``.

    When a ``StorePool`` is supplied the store borrows a shared connection
    from it — ``close()`` releases any local-only connection but leaves
    pooled connections open (the pool owns their lifetime).  This lets many
    concurrent ``MnesisSession`` objects share one physical SQLite connection
    without ``database is locked`` errors, and maps cleanly to a PostgreSQL
    connection-pool in the future.

    Usage (standalone)::

        store = ImmutableStore(StoreConfig())
        await store.initialize()
        try:
            await store.create_session(session)
            await store.append_message(msg)
        finally:
            await store.close()   # closes the private connection

    Usage (with pool)::

        pool = StorePool()
        store_a = ImmutableStore(config, pool=pool)
        store_b = ImmutableStore(config, pool=pool)
        await store_a.initialize()   # opens the shared connection once
        await store_b.initialize()   # reuses it
        await store_a.close()        # no-op — pool owns the connection
        await store_b.close()        # no-op
        await pool.close_all()       # actually closes the connection
    """

    def __init__(self, config: StoreConfig, pool: StorePool | None = None) -> None:
        self._config = config
        self._db_path = str(Path(config.db_path).expanduser())
        self._pool = pool
        self._conn: aiosqlite.Connection | None = None
        # Serialises transactions on the connection (see ``_transaction``). With a
        # pool it is replaced in ``initialize`` by the pool's per-connection lock,
        # so every store sharing a connection shares one lock.
        self._lock = asyncio.Lock()
        self._logger = structlog.get_logger("mnesis.store")

    async def initialize(self) -> None:
        """
        Open (or borrow) a database connection and apply the schema.

        When a pool is set, the connection is acquired from the pool and the
        schema is applied idempotently (``CREATE TABLE IF NOT EXISTS``).
        When no pool is set, a private connection is opened; the caller is
        responsible for calling ``close()`` to release it.

        Raises:
            aiosqlite.Error: If the database cannot be opened or the schema fails.
        """
        if self._pool is not None:
            conn = await self._pool.acquire(
                self._db_path,
                wal_mode=self._config.wal_mode,
                connection_timeout=self._config.connection_timeout,
            )
            self._lock = self._pool.write_lock(self._db_path)
        else:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(self._db_path, timeout=self._config.connection_timeout)
            try:
                conn.row_factory = aiosqlite.Row
                if self._config.wal_mode:
                    await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA foreign_keys=ON")
                await conn.execute("PRAGMA synchronous=NORMAL")
            except Exception:
                await conn.close()
                raise

        try:
            async with self._lock:
                await self._apply_schema(conn)
        except BaseException:
            # ``_conn`` is not set yet, so ``close()`` could not release this
            # connection later: a private one is closed here (a pooled one is
            # owned by the pool).
            if self._pool is None:
                await conn.close()
            raise

        self._conn = conn
        self._logger.info("store_initialized", db_path=self._db_path)

    async def close(self) -> None:
        """
        Release the database connection.

        If the connection is owned by a pool this is a no-op — the pool
        manages the connection lifetime.  If the connection is private
        (no pool was supplied) it is closed and set to ``None``.
        """
        if self._conn is None:
            return
        if self._pool is None:
            # Private connection — we own it, so close it
            await self._conn.close()
        # Pool-managed connection — the pool owns it; do nothing
        self._conn = None

    async def _apply_schema(self, conn: aiosqlite.Connection) -> None:
        """Create the schema and run the idempotent column/index migrations."""
        schema_path = Path(__file__).parent / "schema.sql"
        schema = schema_path.read_text()
        # executescript() handles multiple statements, comments, and semicolons correctly
        await conn.executescript(schema)
        await conn.commit()

        # Wave 1 migration: persist system_prompt on the sessions table.
        try:
            await conn.execute(
                "ALTER TABLE sessions ADD COLUMN system_prompt TEXT NOT NULL DEFAULT ''"
            )
        except aiosqlite.OperationalError:
            pass  # Column already exists

        # Phase 3 migration: add DAG columns to summary_nodes for existing databases.
        # CREATE TABLE IF NOT EXISTS is a no-op for pre-existing tables, so we must
        # ALTER TABLE to add new columns.  Each ADD COLUMN is wrapped in try/except
        # because SQLite raises OperationalError if the column already exists.
        for col_ddl in [
            "ALTER TABLE summary_nodes ADD COLUMN kind TEXT NOT NULL DEFAULT 'leaf'",
            "ALTER TABLE summary_nodes ADD COLUMN parent_node_ids TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE summary_nodes ADD COLUMN superseded INTEGER NOT NULL DEFAULT 0",
        ]:
            try:
                await conn.execute(col_ddl)
            except aiosqlite.OperationalError:
                pass  # Column already exists — "duplicate column name" is safe to ignore
        # Create the superseded index after the migration so that the column is
        # guaranteed to exist for both fresh and pre-Phase-3 databases.
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_summary_nodes_active"
            " ON summary_nodes(session_id, superseded)"
        )
        await conn.commit()

    @contextlib.asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Run a statement sequence as one transaction, alone on its connection.

        A pooled ``aiosqlite`` connection is shared by every coroutine (and every
        ``ImmutableStore``) on that database, and each ``await`` yields to the
        others. Python's sqlite3 opens an implicit transaction at the first DML
        statement and any ``commit()`` on the connection ends it, so without
        serialisation another coroutine's statements are swept into this
        transaction, its ``commit()`` can publish this one half-done (e.g. the
        ``DELETE`` of a compaction swap without its ``INSERT``), and a reader on
        the same connection sees the uncommitted state. The connection's lock is
        held for the whole sequence; the block's work is committed on success and
        rolled back on any exception (including cancellation), so a failed or
        cancelled sequence never leaves partial writes for a later, unrelated
        ``commit()`` to publish.

        The lock is not reentrant: never enter a transaction, or call a store
        method that does, from inside one.
        """
        conn = self._conn_or_raise()
        async with self._lock:
            try:
                yield conn
            except BaseException:
                await conn.rollback()
                raise
            await conn.commit()

    def _conn_or_raise(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise MnesisStoreError("Store is not initialized. Call initialize() first.")
        return self._conn

    # ── Session Methods ────────────────────────────────────────────────────────

    async def create_session(
        self,
        id: str,
        *,
        model_id: str = "",
        provider_id: str = "",
        agent: str = "default",
        parent_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        system_prompt: str = "",
    ) -> Session:
        """
        Insert a new session row.

        Args:
            id: ULID-based session ID (e.g. ``sess_01JXYZ...``).
            model_id: The model string used by this session.
            provider_id: The provider (e.g. ``anthropic``, ``openai``).
            agent: Agent role name (default ``"default"``).
            parent_id: Optional parent session ID for sub-sessions.
            title: Optional human-readable title.
            metadata: Optional JSON-serializable metadata dict.
            system_prompt: System prompt for this session. Persisted so that
                ``load()`` can restore the original prompt on resume.

        Returns:
            The created Session.

        Raises:
            DuplicateIDError: If a session with this ID already exists.
        """
        now = int(time.time() * 1000)
        meta_json = json.dumps(metadata) if metadata else None
        try:
            async with self._transaction() as conn:
                await conn.execute(
                    """
                    INSERT INTO sessions
                        (id, parent_id, created_at, updated_at,
                         model_id, provider_id, agent, title, is_active, metadata, system_prompt)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        id,
                        parent_id,
                        now,
                        now,
                        model_id,
                        provider_id,
                        agent,
                        title,
                        meta_json,
                        system_prompt,
                    ),
                )
        except aiosqlite.IntegrityError as exc:
            raise DuplicateIDError(id) from exc

        return Session(
            id=id,
            parent_id=parent_id,
            created_at=now,
            updated_at=now,
            model_id=model_id,
            provider_id=provider_id,
            agent=agent,
            title=title,
            is_active=True,
            metadata=metadata,
            system_prompt=system_prompt,
        )

    async def get_session(self, session_id: str) -> Session:
        """
        Fetch a session by ID.

        Raises:
            SessionNotFoundError: If no session with this ID exists.
        """
        conn = self._conn_or_raise()
        async with conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise SessionNotFoundError(session_id)
        return self._row_to_session(row)

    async def list_sessions(
        self,
        *,
        parent_id: str | None = None,
        active_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Session]:
        """
        List sessions, newest first.

        Args:
            parent_id: Filter to sessions with this parent (sub-sessions).
            active_only: Exclude soft-deleted sessions when True. Applies
                independently of ``parent_id``.
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip (for pagination).

        Returns:
            List of Session objects ordered by created_at DESC.
        """
        conn = self._conn_or_raise()
        conditions: list[str] = []
        params: list[Any] = []

        if parent_id is not None:
            conditions.append("parent_id = ?")
            params.append(parent_id)
        if active_only:
            conditions.append("is_active = 1")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        async with conn.execute(
            f"SELECT * FROM sessions {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_session(r) for r in rows]

    async def soft_delete_session(self, session_id: str) -> None:
        """Mark a session as inactive (is_active=0). Messages and parts are retained."""
        now = int(time.time() * 1000)
        async with self._transaction() as conn:
            await conn.execute(
                "UPDATE sessions SET is_active=0, updated_at=? WHERE id=?",
                (now, session_id),
            )

    # ── Message Methods ────────────────────────────────────────────────────────

    @staticmethod
    def _map_message_integrity_error(
        exc: aiosqlite.IntegrityError, message: Message
    ) -> MnesisStoreError:
        """Translate a failed message INSERT into the store's public error."""
        if "FOREIGN KEY" in str(exc):
            return SessionNotFoundError(message.session_id)
        return DuplicateIDError(message.id)

    async def _append_message_on(self, conn: aiosqlite.Connection, message: Message) -> None:
        """INSERT a message (and its context item) on *conn*; the caller owns the transaction."""
        tokens = message.tokens or TokenUsage()
        error = message.error
        now_str = str(int(time.time() * 1000))
        await conn.execute(
            """
            INSERT INTO messages (
                id, session_id, parent_id, role, created_at, agent, model_id,
                provider_id, is_summary, tokens_input, tokens_output,
                tokens_cache_read, tokens_cache_write, tokens_total, cost,
                finish_reason, error_code, error_message_text, error_retriable, mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.session_id,
                message.parent_id,
                message.role,
                message.created_at,
                message.agent,
                message.model_id,
                message.provider_id,
                int(message.is_summary),
                tokens.input,
                tokens.output,
                tokens.cache_read,
                tokens.cache_write,
                tokens.effective_total(),
                message.cost,
                message.finish_reason,
                error.code if error else None,
                error.message if error else None,
                int(error.retriable) if error else 0,
                message.mode,
            ),
        )
        # Track non-summary messages in the context_items view.
        # Summary items are inserted by the compaction engine atomically
        # alongside the DELETE of the compacted messages.
        # The position is computed atomically inside the INSERT via a
        # correlated subquery so concurrent appends within the same
        # transaction cannot race on the same MAX(position) value.
        if not message.is_summary:
            await conn.execute(
                """
                INSERT INTO context_items
                    (session_id, item_type, item_id, position, created_at)
                SELECT ?, 'message', ?, COALESCE(MAX(position), 0) + 1, ?
                FROM context_items
                WHERE session_id = ?
                """,
                (message.session_id, message.id, now_str, message.session_id),
            )

    async def append_message(self, message: Message) -> Message:
        """
        Append a message to the log.

        For non-summary messages a ``context_items`` row is inserted atomically
        in the same transaction so that the context-assembly view stays
        consistent with the message log.  Summary messages are NOT inserted
        here — the compaction engine inserts their ``context_items`` row
        separately as part of the atomic context swap.

        Args:
            message: The Message to persist. Must have a pre-generated id.

        Returns:
            The stored message (unmodified).

        Raises:
            SessionNotFoundError: If session_id does not exist.
            DuplicateIDError: If a message with this ID already exists.
        """
        try:
            async with self._transaction() as conn:
                await self._append_message_on(conn, message)
        except aiosqlite.IntegrityError as exc:
            raise self._map_message_integrity_error(exc, message) from exc

        return message

    async def append_part(self, part: RawMessagePart) -> RawMessagePart:
        """
        Append a single part to a message.

        Assigns ``part_index`` as ``max(existing_parts) + 1`` inside the INSERT
        statement itself (``INSERT ... SELECT``), so allocation and insertion are
        a single atomic SQLite statement.  Concurrent appends to the same message
        on the shared connection therefore can never read the same maximum.

        Args:
            part: The RawMessagePart to persist.

        Returns:
            The stored part with its assigned part_index.

        Raises:
            MessageNotFoundError: If message_id does not exist.
        """
        try:
            async with self._transaction() as conn:
                await self._append_part_on(conn, part)
        except aiosqlite.IntegrityError as exc:
            if "FOREIGN KEY" in str(exc):
                raise MessageNotFoundError(part.message_id) from exc
            raise
        return part

    async def _append_part_on(self, conn: aiosqlite.Connection, part: RawMessagePart) -> None:
        """INSERT a part on *conn* and set its ``part_index``; the caller owns the transaction."""
        await conn.execute(
            """
            INSERT INTO message_parts
                (id, message_id, session_id, part_type, part_index, content,
                 tool_name, tool_call_id, tool_state, compacted_at,
                 started_at, completed_at, token_estimate)
            SELECT ?, ?, ?, ?, COALESCE(MAX(part_index), -1) + 1, ?, ?, ?, ?, ?, ?, ?, ?
            FROM message_parts WHERE message_id = ?
            """,
            (
                part.id,
                part.message_id,
                part.session_id,
                part.part_type,
                part.content,
                part.tool_name,
                part.tool_call_id,
                part.tool_state,
                part.compacted_at,
                part.started_at,
                part.completed_at,
                part.token_estimate,
                part.message_id,
            ),
        )
        # Read back the allocated index; the part id is unique, so this cannot
        # race with other appends (indexes are never rewritten).
        async with conn.execute(
            "SELECT part_index FROM message_parts WHERE id = ?", (part.id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:  # pragma: no cover - the INSERT above just succeeded
            raise RuntimeError(f"part {part.id} missing immediately after INSERT")
        part.part_index = row[0]

    async def update_part_status(
        self,
        part_id: str,
        *,
        tool_state: str | None = None,
        compacted_at: int | None = None,
        started_at: int | None = None,
        completed_at: int | None = None,
        output: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """
        Update mutable status fields on a part.

        This is the **only** permitted mutation of persisted data.

        Args:
            part_id: The part to update.
            tool_state: New tool lifecycle state.
            compacted_at: Unix ms timestamp — sets the pruning tombstone.
            started_at: Tool execution start timestamp.
            completed_at: Tool execution end timestamp.
            output: Tool result output string (merged into content JSON).
            error_message: Tool error message (merged into content JSON).

        The content merge is done in SQL with ``json_set``, which requires
        SQLite JSON1 (built in since SQLite 3.38; the project supports
        Python 3.12+, whose bundled SQLite is newer). SQLite re-serializes the
        whole content JSON, so non-standard literals elsewhere in it are
        normalized (``NaN`` becomes ``null``, ``Infinity`` becomes ``9e999``);
        on SQLite < 3.42 such content fails with "malformed JSON".

        Raises:
            PartNotFoundError: If the part does not exist.
            TypeError: If ``output``/``error_message`` is given and the part's
                content is not a JSON object (nothing is updated).
        """

        # Build dynamic SET clause
        set_clauses: list[str] = []
        params: list[Any] = []

        if tool_state is not None:
            set_clauses.append("tool_state = ?")
            params.append(tool_state)
        if compacted_at is not None:
            set_clauses.append("compacted_at = ?")
            params.append(compacted_at)
        if started_at is not None:
            set_clauses.append("started_at = ?")
            params.append(started_at)
        if completed_at is not None:
            set_clauses.append("completed_at = ?")
            params.append(completed_at)

        # Merge into the content JSON inside the UPDATE itself (json_set), so
        # concurrent updates to different fields cannot lose each other's write
        # via a read-modify-write race. Values are bound as ``json(?)`` of
        # ``json.dumps(value)`` so they are stored as JSON strings with the same
        # escaping as before (e.g. lone surrogates become ``\udXXX`` escapes).
        json_updates: list[str] = []
        if output is not None:
            json_updates.append("'$.output', json(?)")
            params.append(json.dumps(output))
        if error_message is not None:
            json_updates.append("'$.error_message', json(?)")
            params.append(json.dumps(error_message))
        if json_updates:
            set_clauses.append(f"content = json_set(content, {', '.join(json_updates)})")

        if not set_clauses:
            return

        params.append(part_id)
        # json_set on non-object content would silently no-op; only match object
        # content so that case can be told apart from a missing part below.
        object_only = " AND json_type(content) = 'object'" if json_updates else ""
        async with self._transaction() as conn:
            result = await conn.execute(
                f"UPDATE message_parts SET {', '.join(set_clauses)} WHERE id = ?{object_only}",
                params,
            )
        if result.rowcount == 0:
            if json_updates:
                async with conn.execute(
                    "SELECT 1 FROM message_parts WHERE id = ?", (part_id,)
                ) as cursor:
                    if await cursor.fetchone() is not None:
                        raise TypeError(
                            f"Part {part_id} content is not a JSON object; cannot merge fields"
                        )
            raise PartNotFoundError(part_id)

    async def batch_set_compacted_at(self, part_ids: list[str], compacted_at: int) -> None:
        """Set ``compacted_at`` on multiple parts in a single query and commit.

        Args:
            part_ids: IDs of the parts to mark as compacted.
            compacted_at: Unix ms timestamp to write into ``compacted_at``.
        """
        if not part_ids:
            return
        placeholders = ",".join("?" * len(part_ids))
        async with self._transaction() as conn:
            result = await conn.execute(
                f"UPDATE message_parts SET compacted_at = ? WHERE id IN ({placeholders})",
                [compacted_at, *part_ids],
            )
        if result.rowcount == 0:
            raise PartNotFoundError(part_ids[0])

    async def update_message_tokens(
        self,
        message_id: str,
        tokens: TokenUsage,
        cost: float,
        finish_reason: str,
    ) -> None:
        """
        Update token usage fields on an assistant message after streaming completes.

        Args:
            message_id: The message to update.
            tokens: Final token usage from the provider response.
            cost: Dollar cost of the response.
            finish_reason: Provider finish reason (e.g. ``"stop"``, ``"max_tokens"``).
        """
        async with self._transaction() as conn:
            await conn.execute(
                """
                UPDATE messages SET
                    tokens_input=?, tokens_output=?, tokens_cache_read=?,
                    tokens_cache_write=?, tokens_total=?, cost=?, finish_reason=?
                WHERE id=?
                """,
                (
                    tokens.input,
                    tokens.output,
                    tokens.cache_read,
                    tokens.cache_write,
                    tokens.effective_total(),
                    cost,
                    finish_reason,
                    message_id,
                ),
            )

    # ── Query Methods ──────────────────────────────────────────────────────────

    async def get_message(self, message_id: str) -> Message:
        """
        Fetch a single message by ID.

        Raises:
            MessageNotFoundError: If no message with this ID exists.
        """
        conn = self._conn_or_raise()
        async with conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise MessageNotFoundError(message_id)
        return self._row_to_message(row)

    async def sum_token_usage(self, session_id: str) -> TokenUsage:
        """
        Aggregate token usage over a session's non-summary messages in SQL.

        Matches ``TokenUsage.__add__`` semantics: each row contributes its
        explicit ``tokens_total`` when positive, otherwise the sum of its parts.
        Summary messages are excluded (``send()``/``record()`` never count them).

        Args:
            session_id: The session to aggregate.

        Returns:
            The summed :class:`TokenUsage` (all zeros for an empty session).
        """
        conn = self._conn_or_raise()
        async with conn.execute(
            """
            SELECT
                COALESCE(SUM(tokens_input), 0),
                COALESCE(SUM(tokens_output), 0),
                COALESCE(SUM(tokens_cache_read), 0),
                COALESCE(SUM(tokens_cache_write), 0),
                COALESCE(SUM(CASE WHEN COALESCE(tokens_total, 0) > 0 THEN tokens_total
                    ELSE COALESCE(tokens_input, 0) + COALESCE(tokens_output, 0)
                       + COALESCE(tokens_cache_read, 0) + COALESCE(tokens_cache_write, 0)
                    END), 0)
            FROM messages WHERE session_id = ? AND is_summary = 0
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return TokenUsage()
        return TokenUsage(
            input=row[0], output=row[1], cache_read=row[2], cache_write=row[3], total=row[4]
        )

    async def get_messages(
        self,
        session_id: str,
        *,
        since_message_id: str | None = None,
    ) -> list[Message]:
        """
        Fetch all messages for a session in chronological order.

        Args:
            session_id: The session to query.
            since_message_id: If provided, returns only messages after the
                message with this ID, in insertion order (``created_at`` with
                insertion order as tie-breaker, so same-millisecond messages
                inserted after the boundary are included).

        Returns:
            List of Message objects ordered by ``created_at`` then insertion order.

        Raises:
            MessageNotFoundError: If ``since_message_id`` is not a message of
                ``session_id`` (including IDs belonging to another session).
        """
        conn = self._conn_or_raise()

        if since_message_id is not None:
            # Compare (created_at, rowid) as a pair: created_at has millisecond
            # resolution, so rowid (insertion order) breaks ties and keeps
            # same-millisecond messages inserted after the boundary.
            async with conn.execute(
                "SELECT m.* FROM messages m, messages b"
                " WHERE b.id=? AND b.session_id=? AND m.session_id=?"
                " AND (m.created_at, m.rowid) > (b.created_at, b.rowid)"
                " ORDER BY m.created_at ASC, m.rowid ASC",
                (since_message_id, session_id, session_id),
            ) as cursor:
                rows = await cursor.fetchall()
            if not rows:
                # Distinguish "nothing after the boundary" from "no such boundary".
                async with conn.execute(
                    "SELECT 1 FROM messages WHERE id=? AND session_id=?",
                    (since_message_id, session_id),
                ) as cursor:
                    if await cursor.fetchone() is None:
                        raise MessageNotFoundError(since_message_id)
        else:
            async with conn.execute(
                "SELECT * FROM messages WHERE session_id=? ORDER BY created_at ASC, rowid ASC",
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()

        return [self._row_to_message(r) for r in rows]

    async def get_parts(self, message_id: str) -> list[RawMessagePart]:
        """Fetch all parts for a message, ordered by part_index ASC."""
        conn = self._conn_or_raise()
        async with conn.execute(
            "SELECT * FROM message_parts WHERE message_id=? ORDER BY part_index ASC",
            (message_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_raw_part(r) for r in rows]

    async def get_raw_parts_for_messages(self, message_ids: list[str]) -> list[RawMessagePart]:
        """Fetch raw parts for a set of messages in a single IN query (no N+1).

        Args:
            message_ids: The message IDs whose parts should be fetched.

        Returns:
            All parts for the given messages, ordered by message_id then part_index.
        """
        if not message_ids:
            return []
        conn = self._conn_or_raise()
        placeholders = ",".join("?" * len(message_ids))
        async with conn.execute(
            f"SELECT * FROM message_parts WHERE message_id IN ({placeholders})"
            " ORDER BY message_id, part_index ASC",
            message_ids,
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_raw_part(r) for r in rows]

    async def get_messages_with_parts(
        self,
        session_id: str,
        *,
        since_message_id: str | None = None,
    ) -> list[MessageWithParts]:
        """
        Efficiently fetch messages and their parts in two queries (no N+1).

        Algorithm:
        1. Fetch all matching messages.
        2. Fetch all parts for those message IDs in a single IN query.
        3. Group parts by message ID and join in Python.
        4. Deserialize part JSON into typed MessagePart objects.

        The two queries are not one snapshot: on a connection shared with a
        concurrent writer a part appended between them can be missed (picked up by
        the next read). Rows are append-only, so a read never sees a half-written
        row; callers that need a consistent view of a message's parts re-read.

        Args:
            session_id: The session to query.
            since_message_id: If provided, returns only messages after this boundary.

        Returns:
            List of MessageWithParts in chronological order.

        Raises:
            MessageNotFoundError: If ``since_message_id`` is not in ``session_id``.
        """
        messages = await self.get_messages(session_id, since_message_id=since_message_id)
        if not messages:
            return []

        conn = self._conn_or_raise()
        message_ids = [m.id for m in messages]
        placeholders = ",".join("?" * len(message_ids))
        async with conn.execute(
            f"SELECT * FROM message_parts WHERE message_id IN ({placeholders})"
            " ORDER BY message_id, part_index ASC",
            message_ids,
        ) as cursor:
            part_rows = await cursor.fetchall()

        # Group by message_id
        parts_by_message: dict[str, list[MessagePart]] = {m.id: [] for m in messages}
        for row in part_rows:
            raw = self._row_to_raw_part(row)
            typed_part = self._deserialize_part(raw)
            if typed_part is not None:
                parts_by_message[raw.message_id].append(typed_part)

        return [
            MessageWithParts(message=msg, parts=parts_by_message.get(msg.id, []))
            for msg in messages
        ]

    async def get_messages_with_parts_by_ids(
        self,
        message_ids: list[str],
    ) -> list[MessageWithParts]:
        """
        Fetch a specific set of messages (and their parts) by ID.

        Returns only the requested messages in the same order as ``message_ids``,
        making context assembly O(k) in the number of context items rather than
        O(n) in the total session length.

        Args:
            message_ids: Ordered list of message IDs to fetch.

        Returns:
            List of MessageWithParts in the order of ``message_ids``.
        """
        if not message_ids:
            return []

        conn = self._conn_or_raise()
        placeholders = ",".join("?" * len(message_ids))
        async with conn.execute(
            f"SELECT * FROM messages WHERE id IN ({placeholders})",
            message_ids,
        ) as cursor:
            msg_rows = await cursor.fetchall()

        if not msg_rows:
            return []

        msg_by_id = {row["id"]: self._row_to_message(row) for row in msg_rows}

        async with conn.execute(
            f"SELECT * FROM message_parts WHERE message_id IN ({placeholders})"
            " ORDER BY message_id, part_index ASC",
            message_ids,
        ) as cursor:
            part_rows = await cursor.fetchall()

        parts_by_message: dict[str, list[MessagePart]] = {mid: [] for mid in message_ids}
        for row in part_rows:
            raw = self._row_to_raw_part(row)
            typed_part = self._deserialize_part(raw)
            if typed_part is not None:
                parts_by_message[raw.message_id].append(typed_part)

        # Return in the caller-specified order, silently skipping missing IDs.
        result: list[MessageWithParts] = []
        for mid in message_ids:
            msg = msg_by_id.get(mid)
            if msg is not None:
                result.append(MessageWithParts(message=msg, parts=parts_by_message.get(mid, [])))
        return result

    async def get_last_summary_message(self, session_id: str) -> Message | None:
        """
        Return the most recent message with is_summary=True, or None.

        Uses the partial index on (session_id, is_summary) for O(log n) lookup.
        """
        conn = self._conn_or_raise()
        async with conn.execute(
            """
            SELECT * FROM messages
            WHERE session_id=? AND is_summary=1
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_message(row)

    # ── File Reference Methods ─────────────────────────────────────────────────

    async def store_file_reference(self, ref: FileReference) -> FileReference:
        """
        Insert or replace a file reference (upsert by content_id).

        Args:
            ref: The FileReference to persist.

        Returns:
            The stored reference.
        """
        async with self._transaction() as conn:
            await conn.execute(
                """
                INSERT OR REPLACE INTO file_references
                    (content_id, path, file_type, token_count, exploration_summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ref.content_id,
                    ref.path,
                    ref.file_type,
                    ref.token_count,
                    ref.exploration_summary,
                    ref.created_at,
                ),
            )
        return ref

    async def get_file_reference(self, content_id: str) -> FileReference | None:
        """Fetch a file reference by content hash. Returns None if not found."""
        conn = self._conn_or_raise()
        async with conn.execute(
            "SELECT * FROM file_references WHERE content_id=?", (content_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_file_reference(row)

    async def get_file_reference_by_path(self, path: str) -> FileReference | None:
        """Fetch the most recently stored file reference for a given path."""
        conn = self._conn_or_raise()
        async with conn.execute(
            "SELECT * FROM file_references WHERE path=? ORDER BY created_at DESC LIMIT 1",
            (path,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_file_reference(row)

    # ── Context Items Methods ──────────────────────────────────────────────────

    async def get_context_items(self, session_id: str) -> list[tuple[str, str]]:
        """
        Return the ordered context items for a session.

        Each item is a ``(item_type, item_id)`` pair in ascending position
        order — the canonical sequence that ``ContextBuilder`` assembles into
        an LLM message list.

        Args:
            session_id: The session to query.

        Returns:
            List of ``(item_type, item_id)`` tuples ordered by position ASC.
        """
        conn = self._conn_or_raise()
        # Under the connection lock: a swap in progress on this shared connection
        # has deleted rows that are not yet replaced, and a read must not see that.
        async with (
            self._lock,
            conn.execute(
                "SELECT item_type, item_id FROM context_items"
                " WHERE session_id = ? ORDER BY position ASC",
                (session_id,),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [(row["item_type"], row["item_id"]) for row in rows]

    async def swap_context_items(
        self,
        session_id: str,
        remove_item_ids: list[str],
        summary_id: str,
    ) -> None:
        """
        Atomically replace a set of context items with a single summary row.

        This is the core of the O(1) compaction commit: within a single
        transaction, all rows whose ``item_id`` is in ``remove_item_ids`` are
        deleted and one new ``'summary'`` row is inserted at the minimum
        position of the removed items (i.e. it occupies the slot of the first
        item being removed); items outside the compacted span keep their
        original positions — no shifting required.

        Used in two scenarios:

        - **Leaf summarisation**: ``remove_item_ids`` contains the message IDs
          that were compacted (``item_type='message'``).
        - **Condensation**: ``remove_item_ids`` contains the parent summary
          node IDs (``item_type='summary'``) being merged.

        Args:
            session_id: The session being compacted.
            remove_item_ids: Context item IDs to remove (any item_type).
            summary_id: ID of the SummaryNode/condensed node that replaces them.
        """
        if not remove_item_ids:
            return
        # The summary occupies the position slot of the first item being removed.
        # Items after the compacted span retain their original positions —
        # no shifting required because positions are monotonically increasing
        # but need not be contiguous.
        async with self._transaction() as conn:
            await self._swap_context_items_on(conn, session_id, remove_item_ids, summary_id)

    async def _swap_context_items_on(
        self,
        conn: aiosqlite.Connection,
        session_id: str,
        remove_item_ids: list[str],
        summary_id: str,
    ) -> None:
        """The swap of :meth:`swap_context_items` on *conn*; the caller owns the transaction.

        *remove_item_ids* must be non-empty (both callers check).
        """
        now_str = str(int(time.time() * 1000))
        placeholders = ",".join("?" * len(remove_item_ids))
        async with conn.execute(
            f"SELECT MIN(position) FROM context_items"
            f" WHERE session_id = ? AND item_id IN ({placeholders})",
            (session_id, *remove_item_ids),
        ) as cursor:
            row = await cursor.fetchone()
        # MIN(position) is NULL when none of the remove_item_ids exist in
        # context_items (e.g. called with stale IDs after a crash-recovery).
        # In that case the swap is a no-op: nothing to remove and nowhere
        # deterministic to place the summary row.
        summary_pos: int | None = row[0] if row else None
        if summary_pos is None:
            return
        await conn.execute(
            f"DELETE FROM context_items WHERE session_id = ? AND item_id IN ({placeholders})",
            (session_id, *remove_item_ids),
        )
        await conn.execute(
            "INSERT INTO context_items (session_id, item_type, item_id, position, created_at)"
            " VALUES (?, 'summary', ?, ?, ?)",
            (session_id, summary_id, summary_pos, now_str),
        )

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _row_to_session(self, row: aiosqlite.Row) -> Session:
        meta = json.loads(row["metadata"]) if row["metadata"] else None
        # system_prompt column added in Wave 1 migration; guard for pre-migration DBs.
        try:
            system_prompt: str = row["system_prompt"] or ""
        except KeyError:
            system_prompt = ""
        return Session(
            id=row["id"],
            parent_id=row["parent_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            model_id=row["model_id"],
            provider_id=row["provider_id"],
            agent=row["agent"],
            title=row["title"],
            is_active=bool(row["is_active"]),
            metadata=meta,
            system_prompt=system_prompt,
        )

    def _row_to_message(self, row: aiosqlite.Row) -> Message:
        tokens = TokenUsage(
            input=row["tokens_input"] or 0,
            output=row["tokens_output"] or 0,
            cache_read=row["tokens_cache_read"] or 0,
            cache_write=row["tokens_cache_write"] or 0,
            total=row["tokens_total"] or 0,
        )
        error = None
        if row["error_code"]:
            error = MessageError(
                code=row["error_code"],
                message=row["error_message_text"] or "",
                retriable=bool(row["error_retriable"]),
            )
        return Message(
            id=row["id"],
            session_id=row["session_id"],
            parent_id=row["parent_id"],
            role=row["role"],
            created_at=row["created_at"],
            agent=row["agent"],
            model_id=row["model_id"],
            provider_id=row["provider_id"],
            is_summary=bool(row["is_summary"]),
            tokens=tokens,
            cost=row["cost"] or 0.0,
            finish_reason=row["finish_reason"],
            error=error,
            mode=row["mode"],
        )

    def _row_to_raw_part(self, row: aiosqlite.Row) -> RawMessagePart:
        return RawMessagePart(
            id=row["id"],
            message_id=row["message_id"],
            session_id=row["session_id"],
            part_type=row["part_type"],
            part_index=row["part_index"],
            content=row["content"],
            tool_name=row["tool_name"],
            tool_call_id=row["tool_call_id"],
            tool_state=row["tool_state"],
            compacted_at=row["compacted_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            token_estimate=row["token_estimate"] or 0,
        )

    def _deserialize_part(self, raw: RawMessagePart) -> MessagePart | None:
        """Deserialize a RawMessagePart's content JSON into a typed MessagePart."""
        try:
            data = json.loads(raw.content)
            # Apply live status fields that may differ from the serialized snapshot
            if raw.part_type == "tool":
                if "status" not in data:
                    data["status"] = {}
                data["status"]["compacted_at"] = raw.compacted_at
                data["status"]["state"] = raw.tool_state or data["status"].get("state", "pending")
                data["status"]["started_at"] = raw.started_at
                data["status"]["completed_at"] = raw.completed_at
            # Use Pydantic discriminated union adapter
            from pydantic import TypeAdapter

            from mnesis.models.message import MessagePart as MPType

            adapter: TypeAdapter[MPType] = TypeAdapter(MPType)
            return adapter.validate_python(data)
        except Exception as exc:
            self._logger.warning(
                "part_deserialize_failed",
                part_id=raw.id,
                part_type=raw.part_type,
                error=str(exc),
            )
            return None

    def _row_to_file_reference(self, row: aiosqlite.Row) -> FileReference:
        return FileReference(
            content_id=row["content_id"],
            path=row["path"],
            file_type=row["file_type"],
            token_count=row["token_count"],
            exploration_summary=row["exploration_summary"],
            created_at=row["created_at"],
        )
