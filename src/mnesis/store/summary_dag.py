"""Logical DAG adapter over ImmutableStore for summary node management."""

from __future__ import annotations

import json
import time as time_mod
from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog

from mnesis.models.message import Message
from mnesis.models.summary import MessageSpan, SummaryNode
from mnesis.store.immutable import ImmutableStore, RawMessagePart

if TYPE_CHECKING:
    import aiosqlite


class SummaryDAGStore:
    """
    Manages the logical DAG of summary nodes.

    Summary nodes are persisted to two places:

    1. The ``messages`` table (``is_summary=True``) — read by ``ContextBuilder``
       to inject summaries into the active context window.
    2. The ``summary_nodes`` table — stores the full DAG: ``kind``,
       ``parent_node_ids`` (JSON array), and ``superseded`` flag.  This is the
       source of truth for DAG relationships and survives session restarts.

    Superseded nodes are tracked in an in-memory set (``_superseded_ids``) for
    fast within-session filtering, and persisted to ``summary_nodes.superseded``
    so the state is recoverable after a process restart.
    """

    def __init__(self, store: ImmutableStore) -> None:
        self._store = store
        self._logger = structlog.get_logger("mnesis.summary_dag")
        self._superseded_ids: set[str] = set()

    async def mark_superseded(self, node_ids: list[str]) -> None:
        """
        Mark one or more summary nodes as superseded by a condensed node.

        Updates both the in-memory set (for fast within-session filtering) and
        the ``summary_nodes`` table (so state survives a restart).

        Args:
            node_ids: IDs of the summary nodes consumed by a condensation.
        """
        self._superseded_ids.update(node_ids)
        conn = self._store._conn_or_raise()
        for node_id in node_ids:
            await conn.execute(
                "UPDATE summary_nodes SET superseded=1 WHERE id=?",
                (node_id,),
            )
        await conn.commit()
        self._logger.debug(
            "nodes_marked_superseded",
            node_ids=node_ids,
            total_superseded=len(self._superseded_ids),
        )

    async def get_active_nodes(self, session_id: str) -> list[SummaryNode]:
        """
        Return all active (non-superseded) summary nodes for a session.

        Queries ``summary_nodes WHERE superseded=0`` to recover DAG state after
        a restart, then populates each node's content from the corresponding
        message parts.  Also applies the in-memory superseded set for nodes
        marked superseded within the current session but not yet flushed (should
        not happen in practice, but guards against concurrent writes).

        Args:
            session_id: The session to query.

        Returns:
            List of SummaryNode objects in creation order (created_at ASC).
        """
        rows = await self._query_summary_nodes(session_id, superseded=False)
        if not rows:
            return []

        all_messages = await self._store.get_messages(session_id)
        nodes: list[SummaryNode] = []
        for row in rows:
            node_id = row["id"]
            if node_id in self._superseded_ids:
                continue
            node = await self._build_node_from_row(row, all_messages)
            if node is not None:
                nodes.append(node)
        return nodes

    async def get_latest_node(self, session_id: str) -> SummaryNode | None:
        """
        Return the most recently created active summary node, or None.

        Reads from ``summary_nodes`` to respect DAG supersession correctly,
        including across restarts.

        Args:
            session_id: The session to query.

        Returns:
            The latest active SummaryNode, or None.
        """
        rows = await self._query_summary_nodes(session_id, superseded=False)
        if not rows:
            return None
        # rows are ordered by created_at ASC; take the last
        latest_row = rows[-1]
        if latest_row["id"] in self._superseded_ids:
            # Scan backwards to find the most recent non-superseded node
            for row in reversed(rows):
                if row["id"] not in self._superseded_ids:
                    latest_row = row
                    break
            else:
                return None
        all_messages = await self._store.get_messages(session_id)
        return await self._build_node_from_row(latest_row, all_messages)

    async def get_coverage_gaps(self, session_id: str) -> list[MessageSpan]:
        """
        Return spans of messages not covered by any summary node.

        In Phase 1, there is at most one summary node (the most recent), so
        the only gap is the tail of messages after that node.

        Args:
            session_id: The session to query.

        Returns:
            List of MessageSpan objects representing uncovered message ranges.
        """
        messages = await self._store.get_messages(session_id)
        non_summary = [m for m in messages if not m.is_summary]
        if not non_summary:
            return []

        latest = await self._store.get_last_summary_message(session_id)
        if latest is None:
            # No summary at all — the entire history is uncovered
            if non_summary:
                return [
                    MessageSpan(
                        start_message_id=non_summary[0].id,
                        end_message_id=non_summary[-1].id,
                        message_count=len(non_summary),
                        estimated_tokens=0,
                    )
                ]
            return []

        # Find messages after the latest summary
        after_summary = [m for m in messages if m.created_at > latest.created_at]
        non_summary_after = [m for m in after_summary if not m.is_summary]
        if not non_summary_after:
            return []

        return [
            MessageSpan(
                start_message_id=non_summary_after[0].id,
                end_message_id=non_summary_after[-1].id,
                message_count=len(non_summary_after),
                estimated_tokens=0,
            )
        ]

    async def insert_node(
        self,
        node: SummaryNode,
        *,
        id_generator: Callable[[], str],
    ) -> SummaryNode:
        """
        Persist a new summary node.

        Writes:
        - A ``Message`` with ``is_summary=True`` (read by ``ContextBuilder``).
        - A ``TextPart`` with the summary content.
        - A ``CompactionMarkerPart`` with metadata.
        - A row in ``summary_nodes`` with ``kind``, ``parent_node_ids`` (JSON),
          and ``superseded=0`` (the DAG persistence added in Phase 3).

        Args:
            node: The SummaryNode to persist. ``node.id`` must be a valid
                message ID generated by ``make_id("msg")``.
            id_generator: Callable that returns new unique part IDs.

        Returns:
            The persisted node (unmodified).
        """
        summary_message = Message(
            id=node.id,
            session_id=node.session_id,
            role="assistant",
            created_at=int(time_mod.time() * 1000),
            is_summary=True,
            model_id=node.model_id,
            provider_id=node.provider_id,
            mode="compaction",
        )
        await self._store.append_message(summary_message)

        # Text part with summary content
        text_raw = RawMessagePart(
            id=id_generator(),
            message_id=node.id,
            session_id=node.session_id,
            part_type="text",
            content=json.dumps({"type": "text", "text": node.content}),
            token_estimate=node.token_count,
        )
        await self._store.append_part(text_raw)

        # Compaction marker part
        marker_raw = RawMessagePart(
            id=id_generator(),
            message_id=node.id,
            session_id=node.session_id,
            part_type="compaction",
            content=json.dumps(
                {
                    "type": "compaction",
                    "summary_node_id": node.id,
                    "compacted_message_count": 0,
                    "compacted_token_count": 0,
                    "level": node.compaction_level,
                }
            ),
        )
        await self._store.append_part(marker_raw)

        # Persist DAG metadata to summary_nodes table
        await self._upsert_summary_node_row(node)

        self._logger.info(
            "summary_node_inserted",
            session_id=node.session_id,
            node_id=node.id,
            kind=node.kind,
            level=node.compaction_level,
            token_count=node.token_count,
            parent_count=len(node.parent_node_ids),
        )
        return node

    async def get_node_by_id(self, node_id: str) -> SummaryNode | None:
        """
        Fetch a specific summary node by ID.

        Args:
            node_id: The message ID of the summary node.

        Returns:
            The SummaryNode, or None if not found or not a summary.
        """
        try:
            msg = await self._store.get_message(node_id)
        except Exception:
            return None
        if not msg.is_summary:
            return None

        row = await self._get_summary_node_row(node_id)
        all_messages = await self._store.get_messages(msg.session_id)
        if row is not None:
            return await self._build_node_from_row(row, all_messages)

        # Fallback: reconstruct from message history (pre-Phase-3 nodes)
        summary_messages = [m for m in all_messages if m.is_summary]
        summary_index = next((i for i, m in enumerate(summary_messages) if m.id == node_id), 0)
        return await self._build_node_from_message(msg, all_messages, summary_index)

    # ── Private Helpers ────────────────────────────────────────────────────────

    async def _upsert_summary_node_row(self, node: SummaryNode) -> None:
        """Insert or replace a row in ``summary_nodes`` for DAG persistence."""
        conn = self._store._conn_or_raise()
        span_start = node.span_start_message_id
        span_end = node.span_end_message_id
        await conn.execute(
            """
            INSERT OR REPLACE INTO summary_nodes (
                id, session_id, level, span_start_message_id, span_end_message_id,
                content, token_count, created_at,
                parent_node_id, model_id, provider_id, compaction_level, is_active,
                kind, parent_node_ids, superseded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 0)
            """,
            (
                node.id,
                node.session_id,
                node.level,
                span_start,
                span_end,
                node.content,
                node.token_count,
                node.created_at,
                node.parent_node_id,
                node.model_id,
                node.provider_id,
                node.compaction_level,
                node.kind,
                json.dumps(node.parent_node_ids),
            ),
        )
        await conn.commit()

    async def _query_summary_nodes(
        self, session_id: str, *, superseded: bool
    ) -> list[aiosqlite.Row]:
        """Query summary_nodes rows for a session, filtered by superseded flag."""
        conn = self._store._conn_or_raise()
        async with conn.execute(
            """
            SELECT * FROM summary_nodes
            WHERE session_id=? AND superseded=?
            ORDER BY created_at ASC
            """,
            (session_id, int(superseded)),
        ) as cursor:
            return list(await cursor.fetchall())

    async def _get_summary_node_row(self, node_id: str) -> aiosqlite.Row | None:
        """Fetch a single summary_nodes row by ID, or None."""
        conn = self._store._conn_or_raise()
        async with conn.execute(
            "SELECT * FROM summary_nodes WHERE id=?",
            (node_id,),
        ) as cursor:
            return await cursor.fetchone()

    async def _build_node_from_row(
        self,
        row: aiosqlite.Row,
        all_messages: list[Message],
    ) -> SummaryNode | None:
        """
        Build a SummaryNode from a ``summary_nodes`` table row.

        ``parent_node_ids`` and ``kind`` are read directly from the row —
        no reconstruction required.  Content is read from the associated
        message parts.
        """
        node_id: str = row["id"]
        parent_node_ids: list[str] = json.loads(row["parent_node_ids"] or "[]")
        kind: str = row["kind"] or "leaf"

        # Read content and compaction_level from message parts
        parts = await self._store.get_parts(node_id)
        content = row["content"] or ""
        token_count: int = row["token_count"] or 0
        compaction_level: int = row["compaction_level"] or 1

        for raw in parts:
            if raw.part_type == "text":
                try:
                    d = json.loads(raw.content)
                    content = d.get("text", content)
                    if raw.token_estimate:
                        token_count = raw.token_estimate
                except Exception:
                    pass
            elif raw.part_type == "compaction":
                try:
                    d = json.loads(raw.content)
                    compaction_level = d.get("level", compaction_level)
                except Exception:
                    pass

        # Find the message for model/provider metadata
        msg = next((m for m in all_messages if m.id == node_id), None)
        model_id = msg.model_id if msg else row["model_id"] or ""
        provider_id = msg.provider_id if msg else row["provider_id"] or ""
        created_at = msg.created_at if msg else row["created_at"]

        return SummaryNode(
            id=node_id,
            session_id=row["session_id"],
            level=row["level"] or 0,
            kind=kind,
            span_start_message_id=row["span_start_message_id"],
            span_end_message_id=row["span_end_message_id"],
            content=content,
            token_count=token_count,
            created_at=created_at,
            parent_node_id=row["parent_node_id"],
            parent_node_ids=parent_node_ids,
            model_id=model_id,
            provider_id=provider_id,
            compaction_level=compaction_level,
        )

    async def _build_node_from_message(
        self,
        summary_msg: Message,
        all_messages: list[Message],
        summary_index: int,
    ) -> SummaryNode | None:
        """
        Reconstruct a SummaryNode from a summary message and message history.

        Fallback path for nodes created before Phase 3 (no ``summary_nodes``
        row exists).  ``parent_node_ids`` defaults to ``[]``.
        """
        summary_messages = [m for m in all_messages if m.is_summary]

        # Determine span: from message after previous summary to this summary
        if summary_index == 0:
            non_summary_before = [
                m
                for m in all_messages
                if not m.is_summary and m.created_at <= summary_msg.created_at
            ]
        else:
            prev_summary = summary_messages[summary_index - 1]
            non_summary_before = [
                m
                for m in all_messages
                if not m.is_summary
                and m.created_at > prev_summary.created_at
                and m.created_at <= summary_msg.created_at
            ]

        span_start = non_summary_before[0].id if non_summary_before else summary_msg.id
        span_end = non_summary_before[-1].id if non_summary_before else summary_msg.id

        parts = await self._store.get_parts(summary_msg.id)
        content = ""
        token_count = 0
        compaction_level = 1
        for raw in parts:
            if raw.part_type == "text":
                try:
                    d = json.loads(raw.content)
                    content = d.get("text", "")
                    token_count = raw.token_estimate
                except Exception:
                    pass
            elif raw.part_type == "compaction":
                try:
                    d = json.loads(raw.content)
                    compaction_level = d.get("level", 1)
                except Exception:
                    pass

        return SummaryNode(
            id=summary_msg.id,
            session_id=summary_msg.session_id,
            level=0,
            span_start_message_id=span_start,
            span_end_message_id=span_end,
            content=content,
            token_count=token_count,
            created_at=summary_msg.created_at,
            model_id=summary_msg.model_id,
            provider_id=summary_msg.provider_id,
            compaction_level=compaction_level,
        )
