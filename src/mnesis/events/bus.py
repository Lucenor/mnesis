"""In-process pub/sub event bus for Mnesis session lifecycle events."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from enum import StrEnum
from functools import partial
from typing import Any

import structlog

Handler = Callable[["MnesisEvent", dict[str, Any]], Awaitable[None] | None]


async def _drive_awaitable(awaitable: Awaitable[Any]) -> None:
    """Await a non-coroutine awaitable so it can run as a task."""
    await awaitable


class MnesisEvent(StrEnum):
    """All event types published by Mnesis components.

    Typed payload definitions for each event live in
    :mod:`mnesis.events.payloads`. Import them for static type checking::

        from mnesis.events.payloads import CompactionCompletedPayload

    **Payload schemas by event:**

    ``SESSION_CREATED``
        :class:`~mnesis.events.payloads.SessionCreatedPayload` —
        ``session_id: str``, ``model: str``

    ``SESSION_CLOSED``
        :class:`~mnesis.events.payloads.SessionClosedPayload` —
        ``session_id: str``

    ``SESSION_UPDATED``, ``SESSION_DELETED``
        *Reserved — not yet published.*

    ``MESSAGE_CREATED``
        :class:`~mnesis.events.payloads.MessageCreatedPayload` —
        ``message_id: str``, ``role: str`` (``"user"`` or ``"assistant"``)

    ``MESSAGE_UPDATED``, ``PART_CREATED``, ``PART_UPDATED``
        *Reserved — not yet published.*

    ``COMPACTION_TRIGGERED``
        :class:`~mnesis.events.payloads.CompactionTriggeredPayload` —
        ``session_id: str``, ``tokens: int``

    ``COMPACTION_COMPLETED``
        :class:`~mnesis.events.payloads.CompactionCompletedPayload` —
        all fields from :class:`~mnesis.models.message.CompactionResult`
        serialized via ``model_dump()``.

    ``COMPACTION_FAILED``
        :class:`~mnesis.events.payloads.CompactionFailedPayload` —
        ``session_id: str``, ``error: str``

    ``PRUNE_COMPLETED``
        :class:`~mnesis.events.payloads.PruneCompletedPayload` —
        ``session_id: str``, ``pruned_count: int``, ``pruned_tokens: int``.
        Published by the compaction engine after pruning runs and at least one
        tool output was tombstoned.

    ``LLM_RETRY``
        :class:`~mnesis.events.payloads.LlmRetryPayload` —
        ``session_id: str``, ``attempt: int``, ``max_retries: int``,
        ``error_type: str``, ``error_message: str``, ``delay_seconds: float``

    ``DOOM_LOOP_DETECTED``
        :class:`~mnesis.events.payloads.DoomLoopDetectedPayload` —
        ``session_id: str``, ``tool: str``

    ``MAP_STARTED``, ``MAP_ITEM_COMPLETED``, ``MAP_COMPLETED``
        :class:`~mnesis.events.payloads.MapStartedPayload`,
        :class:`~mnesis.events.payloads.MapItemCompletedPayload`,
        :class:`~mnesis.events.payloads.MapCompletedPayload`.

        **Important:** These events fire on the *operator's own EventBus*,
        not the session bus. Subscribe via the session bus only if you
        injected it into the operator at construction time::

            llm_map = LLMMap(config.operators, event_bus=session.event_bus)
    """

    # Session lifecycle
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"  # Reserved — not yet published.
    SESSION_CLOSED = "session.closed"
    SESSION_DELETED = "session.deleted"  # Reserved — not yet published.

    # Message lifecycle
    MESSAGE_CREATED = "message.created"
    MESSAGE_UPDATED = "message.updated"  # Reserved — not yet published.

    # Streaming part events (granular, high-frequency; reserved — not yet published)
    PART_CREATED = "part.created"
    PART_UPDATED = "part.updated"

    # Compaction lifecycle
    COMPACTION_TRIGGERED = "compaction.triggered"
    COMPACTION_COMPLETED = "compaction.completed"
    COMPACTION_FAILED = "compaction.failed"

    # Pruning
    PRUNE_COMPLETED = "prune.completed"

    # LLM retry
    LLM_RETRY = "llm.retry"

    # Safety
    DOOM_LOOP_DETECTED = "doom_loop.detected"

    # Operator events — fire on the operator's EventBus, not the session bus.
    # Inject the session bus via LLMMap(config, event_bus=session.event_bus)
    # or AgenticMap(config, event_bus=session.event_bus) to receive them.
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
        # Strong references to in-flight async handler tasks: asyncio only holds
        # weak references, so an unreferenced task can be garbage-collected mid-flight.
        self._handler_tasks: set[asyncio.Task[Any]] = set()

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
        Remove a previously registered handler.

        **Silent no-op** if ``handler`` is not registered for ``event`` — this
        matches the behaviour of most event systems and makes teardown code safe
        to call unconditionally without tracking registration state.

        Args:
            event: The event type the handler was registered for.
            handler: The handler callable to remove.
        """
        handlers = self._handlers.get(event, [])
        try:
            handlers.remove(handler)
        except ValueError:
            # Handler not registered for this event — silent no-op by design.
            pass

    def unsubscribe_all(self, handler: Handler) -> None:
        """
        Remove ``handler`` from all event types and from the global handler list.

        Mirrors :meth:`subscribe_all` for symmetric cleanup. **Silent no-op**
        for any event type or global slot where ``handler`` is not registered.

        Args:
            handler: The handler callable to remove everywhere.
        """
        for handlers in self._handlers.values():
            try:
                handlers.remove(handler)
            except ValueError:
                # Handler not registered for this event type — silent no-op by design.
                pass
        try:
            self._global_handlers.remove(handler)
        except ValueError:
            # Handler not in global list — silent no-op by design.
            pass

    def _on_handler_done(
        self, event: MnesisEvent, handler: Handler, task: asyncio.Task[Any]
    ) -> None:
        """Release an async handler task and log any exception it raised."""
        self._handler_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._logger.error(
                "event_handler_error",
                mnesis_event=str(event),
                handler=getattr(handler, "__qualname__", repr(handler)),
                error=str(exc),
                exc_info=exc,
            )

    def publish(self, event: MnesisEvent, payload: dict[str, Any]) -> None:
        """
        Publish an event to all registered handlers.

        Sync handlers are called immediately in registration order.
        Async handlers (any awaitable return: coroutine, Future, or custom
        ``__await__`` object) are scheduled as background tasks (non-blocking).
        Exceptions from any handler are logged and swallowed.

        Args:
            event: The event type to publish.
            payload: Event-specific data dictionary.
        """
        all_handlers = list(self._handlers.get(event, [])) + list(self._global_handlers)
        for handler in all_handlers:
            try:
                result = handler(event, payload)
                if inspect.isawaitable(result):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # No running event loop — skip async handler
                        if asyncio.iscoroutine(result):
                            result.close()
                    else:
                        # create_task requires a coroutine; wrap Futures and
                        # custom awaitables so they are awaited and errors logged.
                        coro = result if asyncio.iscoroutine(result) else _drive_awaitable(result)
                        task = loop.create_task(coro)
                        self._handler_tasks.add(task)
                        task.add_done_callback(partial(self._on_handler_done, event, handler))
            except Exception as exc:
                self._logger.error(
                    "event_handler_error",
                    mnesis_event=str(event),
                    handler=getattr(handler, "__qualname__", repr(handler)),
                    error=str(exc),
                )
