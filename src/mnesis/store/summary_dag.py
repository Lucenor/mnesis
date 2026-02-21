"""Logical DAG adapter over ImmutableStore for summary node management."""

from __future__ import annotations

import structlog

from mnesis.models.message import (
    Message,
)
from mnesis.models.summary import MessageSpan, SummaryNode
from mnesis.store.immutable import ImmutableStore, RawMessagePart


class SummaryDAGStore:
    """
    Manages the logical DAG of summary nodes.

    **MVP (Phase 1):** Summary nodes are stored as assistant messages with
    ``is_summary=True`` in the ImmutableStore. This class acts as a query
    adapter, providing the DAG API without a separate table.

    **Phase 2:** Will use the ``summary_nodes`` table for multi-level DAG support
    with hierarchical summarisation.

    This store never writes to the database directly — all writes go through
    ``ImmutableStore.append_message()`` and ``ImmutableStore.append_part()``.

    Superseded nodes are tracked in an in-memory set (``_superseded_ids``).
    When condensation produces a new node that replaces its parents, those
    parent IDs are recorded here so that ``get_active_nodes()`` excludes them
    from subsequent rounds.
    """

    def __init__(self, store: ImmutableStore) -> None:
        self._store = store
        self._logger = structlog.get_logger("mnesis.summary_dag")
        self._superseded_ids: set[str] = set()

    def mark_superseded(self, node_ids: list[str]) -> None:
        """
        Mark one or more summary nodes as superseded by a condensed node.

        Superseded nodes are excluded from future ``get_active_nodes()`` calls
        so that the condensation loop does not re-process already-merged nodes.

        Args:
            node_ids: IDs of the summary nodes that have been consumed by a
                condensation operation.
        """
        self._superseded_ids.update(node_ids)
        self._logger.debug(
            "nodes_marked_superseded",
            node_ids=node_ids,
            total_superseded=len(self._superseded_ids),
        )

    async def get_active_nodes(self, session_id: str) -> list[SummaryNode]:
        """
        Return all active summary nodes for a session, ordered chronologically.

        In Phase 1 these are constructed from ``is_summary=True`` messages.

        Args:
            session_id: The session to query.

        Returns:
            List of SummaryNode objects in creation order.
        """
        messages = await self._store.get_messages(session_id)
        summary_messages = [m for m in messages if m.is_summary]
        nodes = []
        for i, summary_msg in enumerate(summary_messages):
            if summary_msg.id in self._superseded_ids:
                continue
            node = await self._build_node_from_message(summary_msg, messages, i)
            if node is not None:
                nodes.append(node)
        return nodes

    async def get_latest_node(self, session_id: str) -> SummaryNode | None:
        """
        Return the most recent summary node, or None if no compaction has occurred.

        Args:
            session_id: The session to query.

        Returns:
            The latest SummaryNode, or None.
        """
        summary_msg = await self._store.get_last_summary_message(session_id)
        if summary_msg is None:
            return None

        messages = await self._store.get_messages(session_id)
        summary_messages = [m for m in messages if m.is_summary]
        summary_index = next(
            (i for i, m in enumerate(summary_messages) if m.id == summary_msg.id),
            0,
        )
        return await self._build_node_from_message(summary_msg, messages, summary_index)

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

        Writes a ``Message`` with ``is_summary=True``, a ``TextPart`` with the
        summary content, and a ``CompactionMarkerPart`` with metadata.

        Args:
            node: The SummaryNode to persist. ``node.id`` must be a valid
                message ID generated by ``make_id("msg")``.
            id_generator: Callable that returns new unique part IDs.

        Returns:
            The persisted node (unmodified).
        """
        import json
        import time as time_mod

        from mnesis.models.message import Message as Msg

        summary_message = Msg(
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
        text_part_data = {"type": "text", "text": node.content}
        text_raw = RawMessagePart(
            id=id_generator(),
            message_id=node.id,
            session_id=node.session_id,
            part_type="text",
            content=json.dumps(text_part_data),
            token_estimate=node.token_count,
        )
        await self._store.append_part(text_raw)

        # Compaction marker part
        marker_data = {
            "type": "compaction",
            "summary_node_id": node.id,
            "compacted_message_count": 0,
            "compacted_token_count": 0,
            "level": node.compaction_level,
        }
        marker_raw = RawMessagePart(
            id=id_generator(),
            message_id=node.id,
            session_id=node.session_id,
            part_type="compaction",
            content=json.dumps(marker_data),
        )
        await self._store.append_part(marker_raw)

        self._logger.info(
            "summary_node_inserted",
            session_id=node.session_id,
            node_id=node.id,
            level=node.compaction_level,
            token_count=node.token_count,
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
        messages = await self._store.get_messages(msg.session_id)
        summary_messages = [m for m in messages if m.is_summary]
        summary_index = next((i for i, m in enumerate(summary_messages) if m.id == node_id), 0)
        return await self._build_node_from_message(msg, messages, summary_index)

    # ── Private Helpers ────────────────────────────────────────────────────────

    async def _build_node_from_message(
        self,
        summary_msg: Message,
        all_messages: list[Message],
        summary_index: int,
    ) -> SummaryNode | None:
        """Reconstruct a SummaryNode from a summary message and message history."""
        summary_messages = [m for m in all_messages if m.is_summary]

        # Determine span: from message after previous summary to this summary
        if summary_index == 0:
            # First summary covers from session start
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

        # Get summary text from parts
        parts = await self._store.get_parts(summary_msg.id)
        content = ""
        token_count = 0
        compaction_level = 1
        for raw in parts:
            if raw.part_type == "text":
                import json

                try:
                    d = json.loads(raw.content)
                    content = d.get("text", "")
                    token_count = raw.token_estimate
                except Exception:
                    pass
            elif raw.part_type == "compaction":
                import json

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


# Import needed for type hint in insert_node
from collections.abc import Callable  # noqa: E402
