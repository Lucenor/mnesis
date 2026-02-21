"""Typed payload definitions for each MnesisEvent.

Each event published by Mnesis components carries a payload dict. This module
defines a ``TypedDict`` for every event so handlers can use static type
checkers and IDE auto-complete rather than guessing key names at runtime.

Usage example::

    from mnesis.events.bus import EventBus, MnesisEvent
    from mnesis.events.payloads import CompactionCompletedPayload

    def on_compaction(event: MnesisEvent, payload: CompactionCompletedPayload) -> None:
        print(
            f"Compacted {payload['compacted_message_count']} messages "
            f"at level {payload['level_used']} "
            f"({payload['tokens_before']} → {payload['tokens_after']} tokens)"
        )

    bus.subscribe(MnesisEvent.COMPACTION_COMPLETED, on_compaction)  # type: ignore[arg-type]
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

# ── Session lifecycle ─────────────────────────────────────────────────────────


class SessionCreatedPayload(TypedDict):
    """Payload for :attr:`MnesisEvent.SESSION_CREATED`."""

    session_id: str
    """The newly created session's ID."""
    model: str
    """The LLM model string the session was created with."""


class SessionClosedPayload(TypedDict):
    """Payload for :attr:`MnesisEvent.SESSION_CLOSED`."""

    session_id: str
    """The session that was closed."""


# SESSION_UPDATED and SESSION_DELETED are reserved for future use.
# They are defined in MnesisEvent but not currently published.


# ── Message lifecycle ─────────────────────────────────────────────────────────


class MessageCreatedPayload(TypedDict):
    """Payload for :attr:`MnesisEvent.MESSAGE_CREATED`."""

    message_id: str
    """The newly persisted message's ID."""
    role: str
    """``"user"`` or ``"assistant"``."""


# PART_CREATED and PART_UPDATED are reserved for future streaming events.
# They are defined in MnesisEvent but not currently published.


# ── Compaction lifecycle ──────────────────────────────────────────────────────


class CompactionTriggeredPayload(TypedDict):
    """Payload for :attr:`MnesisEvent.COMPACTION_TRIGGERED`."""

    session_id: str
    """The session whose compaction was triggered."""
    tokens: int
    """Current cumulative token count that crossed the threshold."""


class CompactionCompletedPayload(TypedDict):
    """Payload for :attr:`MnesisEvent.COMPACTION_COMPLETED`.

    This is the ``model_dump()`` of a :class:`mnesis.models.message.CompactionResult`.
    All fields from ``CompactionResult`` are present.
    """

    session_id: str
    summary_message_id: str
    """ID of the newly created summary message, or ``""`` on failure."""
    level_used: int
    """Escalation level: 1 = selective LLM, 2 = aggressive LLM, 3 = deterministic."""
    compacted_message_count: int
    summary_token_count: int
    tokens_before: int
    tokens_after: int
    elapsed_ms: float
    pruned_tool_outputs: int
    """Number of tool outputs tombstoned by the pruner during this run."""
    pruned_tokens: int
    """Tokens reclaimed by pruning tool outputs."""


class CompactionFailedPayload(TypedDict):
    """Payload for :attr:`MnesisEvent.COMPACTION_FAILED`."""

    session_id: str
    error: str
    """Human-readable error description."""


# ── Pruning ───────────────────────────────────────────────────────────────────

# PRUNE_COMPLETED is reserved for future use.
# It is defined in MnesisEvent but not currently published.


# ── Safety ────────────────────────────────────────────────────────────────────


class DoomLoopDetectedPayload(TypedDict):
    """Payload for :attr:`MnesisEvent.DOOM_LOOP_DETECTED`."""

    session_id: str
    tool: str
    """Name of the tool call that was repeated past the doom-loop threshold."""


# ── Operator events ───────────────────────────────────────────────────────────
#
# MAP_* events fire on the *operator's own EventBus*, NOT the session bus.
# To receive them via a session bus, pass the session bus instance when
# constructing the operator:
#
#     llm_map = LLMMap(config.operators, event_bus=session.event_bus)
#
# Each operator (LLMMap, AgenticMap) creates its own internal EventBus by
# default. Subscribing to MAP_* events on a MnesisSession's bus will never
# fire unless you explicitly inject that bus into the operator.


class MapStartedPayload(TypedDict):
    """Payload for :attr:`MnesisEvent.MAP_STARTED` (operator bus only).

    ``model`` is present only for :class:`~mnesis.operators.LLMMap`.
    ``type`` is present only for :class:`~mnesis.operators.AgenticMap` (value: ``"agentic"``).
    """

    total: int
    """Total number of items to process."""
    model: NotRequired[str]
    """LLM model string. Present for LLMMap; absent for AgenticMap."""
    type: NotRequired[str]
    """Operator type string. Present for AgenticMap (``"agentic"``); absent for LLMMap."""


class MapItemCompletedPayload(TypedDict):
    """Payload for :attr:`MnesisEvent.MAP_ITEM_COMPLETED` (operator bus only)."""

    completed: int
    """Number of items completed so far (including this one)."""
    total: int
    """Total number of items."""
    success: bool
    """Whether this item succeeded."""


class MapCompletedPayload(TypedDict):
    """Payload for :attr:`MnesisEvent.MAP_COMPLETED` (operator bus only).

    ``completed`` is present only for :class:`~mnesis.operators.LLMMap`.
    :class:`~mnesis.operators.AgenticMap` publishes only ``total``.
    """

    total: int
    """Total number of items processed."""
    completed: NotRequired[int]
    """Number of items that completed. Present for LLMMap; absent for AgenticMap."""
