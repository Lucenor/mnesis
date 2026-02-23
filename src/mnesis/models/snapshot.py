"""Per-turn context snapshot models for session.history()."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mnesis.models.message import CompactionResult


class ContextBreakdown(BaseModel):
    """Token counts by component for a single context window state.

    All counts are estimates produced by :class:`~mnesis.tokens.estimator.TokenEstimator`.
    They reflect the context window as assembled immediately after a turn is
    persisted — that is, the exact token budget breakdown that would be used
    on the *next* LLM call if one were made immediately after this turn.

    Attributes:
        system_prompt: Tokens consumed by the system prompt.
        summary: Tokens consumed by compaction summaries injected into context.
        messages: Tokens consumed by raw (non-summary) message history.
        tool_outputs: Subset of ``messages`` attributable to tool call parts.
            This is always ``<= messages``.
        total: Grand total — equals ``system_prompt + summary + messages``.
    """

    system_prompt: int = Field(
        ge=0,
        description="Tokens consumed by the system prompt.",
    )
    summary: int = Field(
        ge=0,
        description="Tokens consumed by compaction summaries.",
    )
    messages: int = Field(
        ge=0,
        description="Tokens consumed by raw (non-summary) message history.",
    )
    tool_outputs: int = Field(
        ge=0,
        description="Subset of messages tokens attributable to tool call parts.",
    )
    total: int = Field(
        ge=0,
        description="Grand total context tokens (system_prompt + summary + messages).",
    )


class TurnSnapshot(BaseModel):
    """Per-turn record of context window state and compaction activity.

    One ``TurnSnapshot`` is appended to the internal list after every
    :meth:`~mnesis.session.MnesisSession.send` and every
    :meth:`~mnesis.session.MnesisSession.record` call.  Retrieve the full list
    with :meth:`~mnesis.session.MnesisSession.history`.

    Typical use-cases:

    - **Debugging** — inspect exact context composition at each turn.
    - **Research** — plot sawtooth token-usage curves, measure compaction
      level distribution, or analyse information-retention across compaction
      boundaries.

    Attributes:
        turn_index: 0-based counter incremented on every ``send()`` /
            ``record()`` call.
        role: The role of the turn that was just completed.  Always
            ``"assistant"`` because a snapshot is captured after the full
            user + assistant exchange is persisted.
        context_tokens: Token breakdown of the context window after this turn.
        compaction_triggered: ``True`` if compaction was scheduled (background
            or foreground) during or immediately after this turn.
        compact_result: The :class:`~mnesis.models.message.CompactionResult`
            from the most recent explicit
            :meth:`~mnesis.session.MnesisSession.compact` call that completed
            before this snapshot was captured.  Background compactions
            triggered automatically by :meth:`~mnesis.session.MnesisSession.send`
            or :meth:`~mnesis.session.MnesisSession.record` are not reflected
            here.  ``None`` when no explicit ``compact()`` call has been made
            yet or when one was still in flight when the snapshot was captured.
    """

    turn_index: int = Field(
        ge=0,
        description="0-based turn counter. Increments on every send()/record().",
    )
    role: Literal["user", "assistant"] = Field(
        description="Role of the turn just completed. Always 'assistant' for full turns.",
    )
    context_tokens: ContextBreakdown = Field(
        description="Context window token breakdown after this turn.",
    )
    compaction_triggered: bool = Field(
        default=False,
        description="True if compaction was triggered during or after this turn.",
    )
    compact_result: CompactionResult | None = Field(
        default=None,
        description=(
            "Result of the most recent explicit compact() call that completed "
            "before this snapshot was captured. Background compactions are not "
            "reflected here. None when no explicit compact() call has been made "
            "yet or when one was still in flight."
        ),
    )
