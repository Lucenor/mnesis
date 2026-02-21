"""Context window assembly algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from mnesis.models.config import MnesisConfig, ModelInfo
from mnesis.models.message import (
    ContextBudget,
    FileRefPart,
    MessageWithParts,
    TextPart,
    ToolPart,
)
from mnesis.store.immutable import ImmutableStore
from mnesis.store.summary_dag import SummaryDAGStore
from mnesis.tokens.estimator import TokenEstimator


@dataclass
class LLMMessage:
    """A single message formatted for the LLM provider API."""

    role: str
    content: str | list[dict[str, Any]]


@dataclass
class BuiltContext:
    """The assembled context window ready for an LLM call."""

    messages: list[LLMMessage]
    system_prompt: str
    token_estimate: int
    budget: ContextBudget
    has_summary: bool
    oldest_included_message_id: str | None = None
    summary_token_count: int = 0


class ContextBuilder:
    """
    Assembles the exact message list sent to the LLM on each turn.

    Invariants:
    1. The most recent raw messages are always included.
    2. Messages before the most recent summary are excluded (replaced by the summary).
    3. Tool parts with ``compacted_at`` set have output replaced by tombstone strings.
    4. FileRefPart objects are rendered as structured ``[FILE: ...]`` blocks.
    5. Total assembled token count fits within ``ContextBudget.usable``.
    6. The system prompt is counted against the token budget.

    Context assembly is O(1) via the ``context_items`` table: a single ordered
    SELECT returns the current context snapshot, and each item is loaded by ID.
    This replaces the previous O(n) backward scan that inferred context from
    message timestamps and summary span IDs.
    """

    def __init__(
        self,
        store: ImmutableStore,
        dag_store: SummaryDAGStore,
        token_estimator: TokenEstimator,
    ) -> None:
        self._store = store
        self._dag_store = dag_store
        self._estimator = token_estimator
        self._logger = structlog.get_logger("mnesis.context_builder")

    async def build(
        self,
        session_id: str,
        model: ModelInfo,
        system_prompt: str,
        config: MnesisConfig,
    ) -> BuiltContext:
        """
        Build the context window for the next LLM call.

        Uses the ``context_items`` table to determine what is currently in
        context (O(1) lookup), then loads each item by ID and assembles the
        LLM message list in position order, respecting the token budget.

        Args:
            session_id: The session to build context for.
            model: Model metadata for budget calculations and tokenisation.
            system_prompt: The system prompt to include (counted against budget).
            config: Mnesis configuration providing the compaction buffer value.

        Returns:
            BuiltContext with the assembled messages, budget, and token estimate.
        """
        # Step 1: Compute budget
        budget = ContextBudget(
            model_context_limit=model.context_limit,
            reserved_output_tokens=model.max_output_tokens,
            compaction_buffer=config.compaction.buffer,
        )
        system_tokens = self._estimator.estimate(system_prompt, model)
        available = budget.usable - system_tokens

        # Step 2: Fetch the ordered context snapshot from context_items.
        # Each row is (item_type, item_id) in ascending position order.
        context_items = await self._store.get_context_items(session_id)

        if not context_items:
            # Backward-compatibility: on a database that was upgraded from a
            # pre-context_items schema, the table exists but has no rows for
            # sessions whose messages were written before the migration.
            # Fall back to the full session scan so those sessions still work.
            all_msgs_fallback = await self._store.get_messages_with_parts(session_id)
            non_summary_fallback = [m for m in all_msgs_fallback if not m.is_summary]
            if not non_summary_fallback:
                return BuiltContext(
                    messages=[],
                    system_prompt=system_prompt,
                    token_estimate=system_tokens,
                    budget=budget,
                    has_summary=False,
                    oldest_included_message_id=None,
                    summary_token_count=0,
                )
            # Treat all non-summary messages as the context snapshot, newest→oldest.
            includable_fb: list[MessageWithParts] = []
            tokens_used_fb = 0
            for msg_with_parts in reversed(non_summary_fallback):
                msg_tokens = self._estimator.estimate_message(msg_with_parts, model)
                if tokens_used_fb + msg_tokens > available:
                    break
                includable_fb.append(msg_with_parts)
                tokens_used_fb += msg_tokens
            includable_fb.reverse()
            llm_messages_fb = [self._convert_message(m) for m in includable_fb]
            return BuiltContext(
                messages=llm_messages_fb,
                system_prompt=system_prompt,
                token_estimate=system_tokens + tokens_used_fb,
                budget=budget,
                has_summary=False,
                oldest_included_message_id=includable_fb[0].id if includable_fb else None,
                summary_token_count=0,
            )

        # Step 3: Separate summaries from messages and compute summary tokens
        # to reserve budget space.
        # Load each summary node once and cache it to avoid repeated round-trips
        # (get_node_by_id internally calls get_messages for the full session,
        # so deduplication matters for sessions with multiple summaries).
        has_summary = any(item_type == "summary" for item_type, _ in context_items)
        summary_cache: dict[str, Any] = {}
        summary_token_count = 0
        if has_summary:
            for item_type, item_id in context_items:
                if item_type == "summary" and item_id not in summary_cache:
                    node = await self._dag_store.get_node_by_id(item_id)
                    summary_cache[item_id] = node  # store None on miss to skip later
                    if node is not None:
                        summary_token_count += node.token_count
            available -= summary_token_count

        # Step 4: Collect message items in order, then walk newest→oldest to
        # fit within the token budget (same greedy approach as before, but now
        # the candidate set comes from the DB rather than a backward scan).
        message_ids_ordered = [iid for t, iid in context_items if t == "message"]

        # Fetch only the required messages by ID — O(k) where k is the number
        # of context items, not O(n) in the total session length.
        candidate_messages: list[MessageWithParts] = []
        if message_ids_ordered:
            candidate_messages = await self._store.get_messages_with_parts_by_ids(
                message_ids_ordered
            )
            if len(candidate_messages) < len(message_ids_ordered):
                fetched_ids = {m.id for m in candidate_messages}
                missing = [mid for mid in message_ids_ordered if mid not in fetched_ids]
                self._logger.warning(
                    "context_items_referenced_missing_messages",
                    session_id=session_id,
                    missing_message_ids=missing,
                )

        # Walk newest→oldest, include as many messages as fit.
        includable: list[MessageWithParts] = []
        tokens_used = 0
        for msg_with_parts in reversed(candidate_messages):
            msg_tokens = self._estimator.estimate_message(msg_with_parts, model)
            if tokens_used + msg_tokens > available:
                self._logger.debug(
                    "context_budget_reached",
                    session_id=session_id,
                    excluded_count=len(candidate_messages) - len(includable),
                )
                break
            includable.append(msg_with_parts)
            tokens_used += msg_tokens

        includable.reverse()  # Restore chronological order

        # Step 5: Convert to LLM message format
        llm_messages: list[LLMMessage] = []

        # 5a: Inject summary nodes in position order (before raw messages),
        # reusing the cache populated in Step 3.
        for item_type, item_id in context_items:
            if item_type == "summary":
                node = summary_cache.get(item_id)
                if node is not None:
                    llm_messages.append(LLMMessage(role="assistant", content=node.content))

        # 5b: Convert raw messages
        for msg_with_parts in includable:
            llm_msg = self._convert_message(msg_with_parts)
            llm_messages.append(llm_msg)

        total_tokens = system_tokens + summary_token_count + tokens_used

        self._logger.debug(
            "context_built",
            session_id=session_id,
            message_count=len(llm_messages),
            token_estimate=total_tokens,
            has_summary=has_summary,
        )

        return BuiltContext(
            messages=llm_messages,
            system_prompt=system_prompt,
            token_estimate=total_tokens,
            budget=budget,
            has_summary=has_summary,
            oldest_included_message_id=includable[0].id if includable else None,
            summary_token_count=summary_token_count,
        )

    def _convert_message(self, msg_with_parts: MessageWithParts) -> LLMMessage:
        """Convert a MessageWithParts to an LLMMessage for the provider API."""
        role = msg_with_parts.role
        content_parts: list[str] = []

        for part in msg_with_parts.parts:
            if isinstance(part, TextPart):
                content_parts.append(part.text)
            elif isinstance(part, ToolPart):
                if part.compacted_at is not None:
                    # Tombstone replacement
                    tombstone = f"[Tool '{part.tool_name}' output compacted at {part.compacted_at}]"
                    content_parts.append(tombstone)
                else:
                    tool_text = self._render_tool_part(part)
                    content_parts.append(tool_text)
            elif isinstance(part, FileRefPart):
                content_parts.append(self._render_file_ref(part))
            else:
                # Other part types: use text representation if available
                text = getattr(part, "text", None)
                if text:
                    content_parts.append(str(text))

        content = "\n".join(content_parts)
        return LLMMessage(role=role, content=content)

    @staticmethod
    def _render_tool_part(part: ToolPart) -> str:
        """Render a tool call and result as readable text."""
        import json

        lines = [
            f"[Tool: {part.tool_name}]",
            f"Input: {json.dumps(part.input, indent=2)}",
        ]
        if part.output:
            lines.append(f"Output: {part.output}")
        if part.error_message:
            lines.append(f"Error: {part.error_message}")
        lines.append("[/Tool]")
        return "\n".join(lines)

    @staticmethod
    def _render_file_ref(part: FileRefPart) -> str:
        """Render a FileRefPart as a structured text block."""
        return (
            f"[FILE: {part.path}]\n"
            f"Content-ID: {part.content_id}\n"
            f"Type: {part.file_type}\n"
            f"Tokens: {part.token_count:,}\n"
            f"Exploration Summary:\n{part.exploration_summary}\n"
            f"[/FILE]"
        )
