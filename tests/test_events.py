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


class TestNonCoroutineAwaitables:
    async def test_future_returning_handler_is_awaited(self) -> None:
        bus = EventBus()
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()

        def handler(event: MnesisEvent, payload: dict) -> asyncio.Future[None]:
            return fut

        bus.subscribe(MnesisEvent.SESSION_CREATED, handler)
        bus.publish(MnesisEvent.SESSION_CREATED, {})
        assert len(bus._handler_tasks) == 1
        fut.set_result(None)
        await asyncio.sleep(0.01)
        assert bus._handler_tasks == set()

    async def test_custom_awaitable_handler_runs(self) -> None:
        bus = EventBus()
        seen: list[str] = []

        class Custom:
            def __await__(self):  # type: ignore[no-untyped-def]
                seen.append("ran")
                yield from asyncio.sleep(0).__await__()

        def handler(event: MnesisEvent, payload: dict) -> Custom:
            return Custom()

        bus.subscribe(MnesisEvent.SESSION_CREATED, handler)
        bus.publish(MnesisEvent.SESSION_CREATED, {})
        await asyncio.sleep(0.01)
        assert seen == ["ran"]
        assert bus._handler_tasks == set()

    async def test_failing_future_is_logged(self) -> None:
        logger = _CapturingLogger()
        bus = EventBus(logger=logger)  # type: ignore[arg-type]
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        fut.set_exception(RuntimeError("fut boom"))

        bus.subscribe(MnesisEvent.SESSION_CREATED, lambda e, p: fut)
        bus.publish(MnesisEvent.SESSION_CREATED, {})
        await asyncio.sleep(0.01)
        assert len(logger.errors) == 1
        assert logger.errors[0]["error"] == "fut boom"
        assert logger.errors[0]["mnesis_event"] == str(MnesisEvent.SESSION_CREATED)

    def test_non_coroutine_awaitable_without_loop_is_skipped(self) -> None:
        bus = EventBus()

        class Custom:
            def __await__(self):  # type: ignore[no-untyped-def]
                raise AssertionError("must not be awaited without a loop")
                yield

        bus.subscribe(MnesisEvent.SESSION_CREATED, lambda e, p: Custom())
        bus.publish(MnesisEvent.SESSION_CREATED, {})
        assert bus._handler_tasks == set()
