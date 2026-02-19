"""In-process pub/sub event bus for Mnesis session lifecycle events."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Any, Awaitable, Callable

import structlog

Handler = Callable[["MnesisEvent", dict[str, Any]], None | Awaitable[None]]


class MnesisEvent(StrEnum):
    """All event types published by Mnesis components."""

    # Session lifecycle
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SESSION_CLOSED = "session.closed"
    SESSION_DELETED = "session.deleted"

    # Message lifecycle
    MESSAGE_CREATED = "message.created"
    MESSAGE_UPDATED = "message.updated"

    # Streaming part events (granular, high-frequency)
    PART_CREATED = "part.created"
    PART_UPDATED = "part.updated"

    # Compaction lifecycle
    COMPACTION_TRIGGERED = "compaction.triggered"
    COMPACTION_COMPLETED = "compaction.completed"
    COMPACTION_FAILED = "compaction.failed"

    # Pruning
    PRUNE_COMPLETED = "prune.completed"

    # Safety
    DOOM_LOOP_DETECTED = "doom_loop.detected"

    # Operator events
    MAP_STARTED = "map.started"
    MAP_ITEM_COMPLETED = "map.item_completed"
    MAP_COMPLETED = "map.completed"


class EventBus:
    """
    Simple in-process pub/sub event bus.

    Design decisions:
    - Sync handlers are called inline within ``publish()``.
    - Async handlers are scheduled via ``asyncio.create_task()`` (fire-and-forget).
    - Handler exceptions are logged but never propagate to the publisher.
    - Each ``MnesisSession`` owns its own ``EventBus`` instance for isolation.
      Share one instance across sessions for cross-session monitoring.

    Example::

        bus = EventBus()

        def on_compaction(event, payload):
            print(f"Compacted {payload['compacted_message_count']} messages")

        bus.subscribe(MnesisEvent.COMPACTION_COMPLETED, on_compaction)
        bus.publish(MnesisEvent.COMPACTION_COMPLETED, {"compacted_message_count": 10})
    """

    def __init__(self, logger: structlog.BoundLogger | None = None) -> None:
        self._handlers: dict[MnesisEvent, list[Handler]] = {}
        self._global_handlers: list[Handler] = []
        self._logger = logger or structlog.get_logger("mnesis.events")

    def subscribe(self, event: MnesisEvent, handler: Handler) -> None:
        """
        Register a handler for a specific event type.

        Args:
            event: The event type to listen for.
            handler: Callable accepting ``(event, payload)``. May be sync or async.
        """
        self._handlers.setdefault(event, []).append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        """
        Register a handler for ALL event types.

        Args:
            handler: Callable accepting ``(event, payload)``. May be sync or async.
        """
        self._global_handlers.append(handler)

    def unsubscribe(self, event: MnesisEvent, handler: Handler) -> None:
        """
        Remove a previously registered handler. No-op if not found.

        Args:
            event: The event type the handler was registered for.
            handler: The handler callable to remove.
        """
        handlers = self._handlers.get(event, [])
        try:
            handlers.remove(handler)
        except ValueError:
            pass

    def publish(self, event: MnesisEvent, payload: dict[str, Any]) -> None:
        """
        Publish an event to all registered handlers.

        Sync handlers are called immediately in registration order.
        Async handlers are scheduled as background tasks (non-blocking).
        Exceptions from any handler are logged and swallowed.

        Args:
            event: The event type to publish.
            payload: Event-specific data dictionary.
        """
        all_handlers = list(self._handlers.get(event, [])) + list(self._global_handlers)
        for handler in all_handlers:
            try:
                result = handler(event, payload)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        # No running event loop — skip async handler
                        pass
            except Exception as exc:
                self._logger.error(
                    "event_handler_error",
                    event=str(event),
                    handler=getattr(handler, "__qualname__", repr(handler)),
                    error=str(exc),
                )
