"""
Shared connection pool for ImmutableStore.

A single ``StorePool`` instance manages one ``aiosqlite.Connection`` per
database path.  All ``ImmutableStore`` objects pointing at the same path
share that connection, so concurrent sub-agent sessions never fight over
SQLite's single-writer lock.

When migrating to PostgreSQL (or another backend), replace this module with
one that wraps ``asyncpg.create_pool`` / ``sqlalchemy.ext.asyncio`` — the
``ImmutableStore`` interface stays identical.

Usage::

    pool = StorePool()

    store_a = ImmutableStore(config, pool=pool)
    store_b = ImmutableStore(config, pool=pool)   # same DB path → same connection

    await store_a.initialize()   # opens the connection (idempotent on 2nd call)
    await store_b.initialize()   # reuses existing connection

    # … use stores …

    await pool.close_all()       # close all managed connections once at shutdown
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
import structlog

if TYPE_CHECKING:
    pass

_logger = structlog.get_logger("mnesis.store.pool")


class StorePool:
    """
    Process-scoped registry of open ``aiosqlite.Connection`` objects.

    Thread-safety: only safe to use from a single asyncio event loop — do not
    share a ``StorePool`` across threads.

    For each unique *resolved* database path the pool holds exactly one
    connection.  Callers may call ``acquire()`` concurrently; only the first
    caller opens the connection, subsequent callers receive the same object.

    The pool also manages a per-path ``asyncio.Lock`` that
    ``ImmutableStore`` uses to serialise write transactions.  SQLite in WAL
    mode allows unlimited concurrent readers but only one writer; the lock
    prevents competing writes from getting ``database is locked`` errors on
    busy concurrent workloads.
    """

    def __init__(self) -> None:
        self._connections: dict[str, aiosqlite.Connection] = {}
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._open_locks: dict[str, asyncio.Lock] = {}  # per-path open guards

    # ── Public API ─────────────────────────────────────────────────────────────

    async def acquire(
        self,
        db_path: str,
        *,
        wal_mode: bool = True,
        connection_timeout: float = 30.0,
    ) -> aiosqlite.Connection:
        """
        Return the shared connection for *db_path*, opening it if needed.

        Args:
            db_path: Resolved (~ expanded) absolute path to the database file.
            wal_mode: Enable WAL journal mode on first open.
            connection_timeout: SQLite busy timeout in seconds.

        Returns:
            The shared ``aiosqlite.Connection`` for this path.
        """
        resolved = str(Path(db_path).expanduser().resolve())  # noqa: ASYNC240

        # Fast path — connection already open
        if resolved in self._connections:
            return self._connections[resolved]

        # Slow path — open once, then cache.  Guard with a per-path lock so
        # concurrent coroutines don't race to open the same file.
        if resolved not in self._open_locks:
            self._open_locks[resolved] = asyncio.Lock()

        async with self._open_locks[resolved]:
            # Double-check after acquiring the lock
            if resolved in self._connections:
                return self._connections[resolved]

            Path(resolved).parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(resolved, timeout=connection_timeout)
            try:
                conn.row_factory = aiosqlite.Row
                if wal_mode:
                    await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA foreign_keys=ON")
                await conn.execute("PRAGMA synchronous=NORMAL")
            except Exception:
                await conn.close()
                raise

            self._connections[resolved] = conn
            self._write_locks[resolved] = asyncio.Lock()
            _logger.debug("pool_connection_opened", db_path=resolved)
            return conn

    def write_lock(self, db_path: str) -> asyncio.Lock:
        """
        Return the write-serialisation lock for *db_path*.

        The lock must already exist (i.e. ``acquire()`` must have been called
        for this path).  Raises ``KeyError`` if called before ``acquire()``.
        """
        resolved = str(Path(db_path).expanduser().resolve())
        return self._write_locks[resolved]

    async def close_path(self, db_path: str) -> None:
        """Close and remove the connection for a single path."""
        resolved = str(Path(db_path).expanduser().resolve())  # noqa: ASYNC240
        conn = self._connections.pop(resolved, None)
        self._write_locks.pop(resolved, None)
        self._open_locks.pop(resolved, None)
        if conn is not None:
            await conn.close()
            _logger.debug("pool_connection_closed", db_path=resolved)

    async def close_all(self) -> None:
        """Close every connection managed by this pool."""
        paths = list(self._connections.keys())
        for path in paths:
            await self.close_path(path)

    # ── Convenience: process-level default pool ─────────────────────────────────

    @staticmethod
    def default() -> StorePool:
        """
        Return the process-level default pool.

        The default pool is created lazily on first access and lives for the
        lifetime of the process.  It is suitable for production use; tests
        should create their own ``StorePool()`` instances to get full isolation.
        """
        global _default_pool
        if _default_pool is None:
            _default_pool = StorePool()
        return _default_pool


_default_pool: StorePool | None = None
