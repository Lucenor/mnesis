"""Tests for EventBus async handler task management."""

from __future__ import annotations

import asyncio
import gc

from mnesis.events.bus import EventBus, MnesisEvent


class _CapturingLogger:
    """Minimal structlog stand-in recording error calls."""

    def __init__(self) -> None:
        self.errors: list[dict] = []

    def error(self, msg: str, **kw: object) -> None:
        self.errors.append({"msg": msg, **kw})


class TestAsyncHandlerTasks:
    async def test_async_handler_runs_and_task_reference_released(self) -> None:
        bus = EventBus()
        seen: list[str] = []

        async def handler(event: MnesisEvent, payload: dict) -> None:
            await asyncio.sleep(0)
            seen.append(payload["k"])

        bus.subscribe(MnesisEvent.SESSION_CREATED, handler)
        bus.publish(MnesisEvent.SESSION_CREATED, {"k": "v"})
        # The bus holds a strong reference while the task is in flight, even across GC.
        assert len(bus._handler_tasks) == 1
        gc.collect()
        await asyncio.sleep(0.01)

        assert seen == ["v"]
        assert bus._handler_tasks == set()

    async def test_async_handler_exception_is_logged_not_raised(self) -> None:
        logger = _CapturingLogger()
        bus = EventBus(logger=logger)  # type: ignore[arg-type]

        async def bad(event: MnesisEvent, payload: dict) -> None:
            raise ValueError("boom")

        bus.subscribe(MnesisEvent.SESSION_CREATED, bad)
        bus.publish(MnesisEvent.SESSION_CREATED, {})  # must not raise
        await asyncio.sleep(0.01)

        assert len(logger.errors) == 1
        assert logger.errors[0]["msg"] == "event_handler_error"
        assert logger.errors[0]["error"] == "boom"
        assert "bad" in logger.errors[0]["handler"]
        assert bus._handler_tasks == set()

    async def test_cancelled_async_handler_is_not_logged_as_error(self) -> None:
        logger = _CapturingLogger()
        bus = EventBus(logger=logger)  # type: ignore[arg-type]

        async def slow(event: MnesisEvent, payload: dict) -> None:
            await asyncio.sleep(10)

        bus.subscribe(MnesisEvent.SESSION_CREATED, slow)
        bus.publish(MnesisEvent.SESSION_CREATED, {})
        (task,) = bus._handler_tasks
        task.cancel()
        await asyncio.sleep(0.01)

        assert logger.errors == []
        assert bus._handler_tasks == set()

    def test_async_handler_without_running_loop_is_skipped_cleanly(self) -> None:
        """No loop: the coroutine is closed (no 'never awaited' warning) and nothing leaks."""
        bus = EventBus()
        ran: list[bool] = []

        async def handler(event: MnesisEvent, payload: dict) -> None:
            ran.append(True)

        bus.subscribe(MnesisEvent.SESSION_CREATED, handler)
        bus.publish(MnesisEvent.SESSION_CREATED, {})

        assert ran == []
        assert bus._handler_tasks == set()
