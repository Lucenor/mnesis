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

        # Step 2: Find compaction boundary
        latest_summary = await self._dag_store.get_latest_node(session_id)
        summary_token_count = 0

        if latest_summary is not None:
            boundary_id = latest_summary.span_end_message_id
            raw_messages = await self._store.get_messages_with_parts(
                session_id, since_message_id=boundary_id
            )
            # Exclude other summary messages from the raw tail
            raw_messages = [m for m in raw_messages if not m.is_summary]
            summary_token_count = latest_summary.token_count
            available -= summary_token_count
        else:
            raw_messages = await self._store.get_messages_with_parts(session_id)
            raw_messages = [m for m in raw_messages if not m.is_summary]

        # Step 3: Walk newest→oldest to fit within budget
        includable: list[MessageWithParts] = []
        tokens_used = 0
        for msg_with_parts in reversed(raw_messages):
            msg_tokens = self._estimator.estimate_message(msg_with_parts, model)
            if tokens_used + msg_tokens > available:
                # Stop — cannot fit this message without breaking coherence
                self._logger.debug(
                    "context_budget_reached",
                    session_id=session_id,
                    excluded_count=len(raw_messages) - len(includable),
                )
                break
            includable.append(msg_with_parts)
            tokens_used += msg_tokens

        includable.reverse()  # Restore chronological order

        # Step 4: Convert to LLM message format
        llm_messages: list[LLMMessage] = []

        # 4a: Prepend summary as assistant turn
        if latest_summary is not None:
            llm_messages.append(LLMMessage(role="assistant", content=latest_summary.content))

        # 4b: Convert raw messages
        for msg_with_parts in includable:
            llm_msg = self._convert_message(msg_with_parts)
            llm_messages.append(llm_msg)

        total_tokens = system_tokens + summary_token_count + tokens_used

        self._logger.debug(
            "context_built",
            session_id=session_id,
            message_count=len(llm_messages),
            token_estimate=total_tokens,
            has_summary=latest_summary is not None,
        )

        return BuiltContext(
            messages=llm_messages,
            system_prompt=system_prompt,
            token_estimate=total_tokens,
            budget=budget,
            has_summary=latest_summary is not None,
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
