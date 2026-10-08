"""close() owner/waiter handoff, interruption and a seeded interleaving fuzzer."""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest

import mnesis.compaction.engine as engine_mod
import mnesis.session as session_mod
from mnesis import MnesisSession
from mnesis.events.bus import MnesisEvent
from mnesis.models.config import MnesisConfig, StoreConfig
from mnesis.store.pool import StorePool

MODEL = "anthropic/claude-haiku-4-5"


@pytest.fixture(autouse=True)
def _fast_mock(monkeypatch):
    """Mock LLMs with a small, seedable delay; a short in-flight wait bound."""
    monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
    monkeypatch.setattr(session_mod, "_CLOSE_INFLIGHT_TIMEOUT", 0.2)
    rnd_box = {"rnd": random.Random(0), "send_delay": 0.0}
    orig = MnesisSession._mock_response

    async def slow_mock(self, *args, **kwargs):
        await asyncio.sleep(rnd_box["send_delay"] or rnd_box["rnd"].uniform(0, 0.03))
        return await orig(self, *args, **kwargs)

    def llm_factory(model, **kw):
        async def _call(*, model=model, messages, max_tokens):
            await asyncio.sleep(rnd_box["rnd"].uniform(0.005, 0.04))
            return "## Goal\nX\n\n## Completed Work\n- did things\n"

        return _call

    monkeypatch.setattr(MnesisSession, "_mock_response", slow_mock)
    monkeypatch.setattr(engine_mod, "_make_llm_call", llm_factory)
    return rnd_box


async def _session(tmp_path, name="c.db", pool=None):
    cfg = MnesisConfig(store=StoreConfig(db_path=str(tmp_path / name)))
    session = await MnesisSession.create(model=MODEL, config=cfg, pool=pool)
    events: list[int] = []
    session.subscribe(MnesisEvent.SESSION_CLOSED, lambda e, p: events.append(1))
    session._test_events = events  # type: ignore[attr-defined]
    return session


def _state_ok_closed(session) -> None:
    assert session._closed
    assert len(session._test_events) == 1


async def _assert_reopened(session) -> None:
    assert not session._closed and not session._closing
    assert session._close_done is None and session._close_waiters == 0
    assert session._store._conn is not None
    assert session._test_events == []
    result = await asyncio.wait_for(session.send("after"), timeout=3)
    assert result.text
    await session.close()
    assert len(session._test_events) == 1


class TestHandoff:
    async def test_owner_interrupted_waiter_takes_over_and_closes_once(self, tmp_path, _fast_mock):
        _fast_mock["send_delay"] = 0.3
        session = await _session(tmp_path)
        send = asyncio.create_task(session.send("x"))
        await asyncio.sleep(0.05)
        owner = asyncio.create_task(session.close())
        await asyncio.sleep(0.01)
        waiter = asyncio.create_task(session.close())
        await asyncio.sleep(0.01)
        _ = owner.cancel()
        _ = await asyncio.wait({owner})
        _ = await asyncio.wait_for(waiter, timeout=5)  # took over and closed
        _ = await asyncio.wait({send})
        _state_ok_closed(session)
        await session.close()  # no-op
        assert len(session._test_events) == 1

    async def test_owner_and_waiter_both_interrupted_in_one_tick_reopen(self, tmp_path, _fast_mock):
        _fast_mock["send_delay"] = 0.3
        session = await _session(tmp_path)
        send = asyncio.create_task(session.send("x"))
        await asyncio.sleep(0.05)
        owner = asyncio.create_task(session.close())
        await asyncio.sleep(0.01)
        waiter = asyncio.create_task(session.close())
        await asyncio.sleep(0.01)
        _ = owner.cancel()
        await asyncio.sleep(0)  # the owner's finally runs: a waiter is still counted
        _ = waiter.cancel()  # ...and is cancelled before it can take over
        _ = await asyncio.wait({owner, waiter})
        _ = await asyncio.wait({send})
        await _assert_reopened(session)

    async def test_cancelled_gather_of_two_closes_reopens(self, tmp_path, _fast_mock):
        _fast_mock["send_delay"] = 0.3
        session = await _session(tmp_path)
        send = asyncio.create_task(session.send("x"))
        await asyncio.sleep(0.05)
        both = asyncio.ensure_future(asyncio.gather(session.close(), session.close()))
        await asyncio.sleep(0.05)
        _ = both.cancel()
        _ = await asyncio.wait({both})
        _ = await asyncio.wait({send})
        await _assert_reopened(session)

    async def test_timeout_around_gather_of_three_closes_reopens(self, tmp_path, _fast_mock):
        _fast_mock["send_delay"] = 0.3
        session = await _session(tmp_path)
        send = asyncio.create_task(session.send("x"))
        await asyncio.sleep(0.05)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.05):
                _ = await asyncio.gather(session.close(), session.close(), session.close())
        _ = await asyncio.wait({send})
        await _assert_reopened(session)

    async def test_two_tasks_with_the_same_deadline_reopen(self, tmp_path, _fast_mock):
        _fast_mock["send_delay"] = 0.3
        session = await _session(tmp_path)
        send = asyncio.create_task(session.send("x"))
        await asyncio.sleep(0.05)
        deadline = asyncio.get_running_loop().time() + 0.05

        async def closer() -> str:
            try:
                async with asyncio.timeout_at(deadline):
                    await session.close()
            except TimeoutError:
                return "timeout"
            return "ok"

        assert await asyncio.gather(closer(), closer()) == ["timeout", "timeout"]
        _ = await asyncio.wait({send})
        await _assert_reopened(session)


class _SlowConn:
    def __init__(self, conn: Any) -> None:
        self._c = conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._c, name)

    async def close(self) -> None:
        await asyncio.sleep(0.3)
        await self._c.close()


class TestInterruptedStoreClose:
    @pytest.mark.parametrize("with_waiter", [False, True])
    async def test_interruption_inside_store_close_keeps_the_session_closed(
        self, tmp_path, with_waiter
    ):
        session = await _session(tmp_path)
        _ = await session.send("warm")
        session._store._conn = _SlowConn(session._store._conn)
        owner = asyncio.create_task(session.close())
        waiter = asyncio.create_task(session.close()) if with_waiter else None
        await asyncio.sleep(0.1)  # the owner is inside store.close()
        _ = owner.cancel()
        _ = await asyncio.wait({owner})
        if waiter is not None:
            await asyncio.wait_for(waiter, timeout=5)
        await asyncio.sleep(0.4)
        _state_ok_closed(session)  # SESSION_CLOSED exactly once
        await session.close()  # a later close() is a no-op
        assert len(session._test_events) == 1


@pytest.mark.parametrize("seed_block", range(4))
async def test_seeded_close_interleaving_fuzz(tmp_path, _fast_mock, seed_block):
    """~30 seeded scenarios per block: never hang, leak, or end half-closed."""
    violations: list[Any] = []
    for seed in range(seed_block * 30, seed_block * 30 + 30):
        await _scenario(tmp_path, _fast_mock, seed, violations)
    assert violations == []


async def _scenario(tmp_path, rnd_box, seed: int, violations: list[Any]) -> None:
    rnd = random.Random(seed)
    rnd_box["rnd"] = rnd
    rnd_box["send_delay"] = 0.0
    pool = StorePool() if rnd.random() < 0.3 else None
    session = await _session(tmp_path, f"fz{seed}.db", pool=pool)
    for i in range(3):
        _ = await session.record(f"q{i} " * 30, f"a{i} " * 30)
    loop = asyncio.get_running_loop()
    background: dict[str, asyncio.Task[Any]] = {}
    inside: dict[str, str] = {}

    for j in range(rnd.randint(0, 2)):
        close_inside = rnd.random() < 0.35
        abort = rnd.random() < 0.5

        def make_on_part(name: str, close_inside=close_inside, abort=abort):
            fired = [False]

            async def on_part(_part: Any) -> None:
                if close_inside and not fired[0]:
                    fired[0] = True
                    inside[name] = "called"
                    await session.close(abort_compaction=abort)

            return on_part

        name = f"send{j}"
        background[name] = asyncio.create_task(
            session.send(f"hello {j}", on_part=make_on_part(name))
        )
    if rnd.random() < 0.5:
        background["compact"] = asyncio.create_task(session.compact())
    await asyncio.sleep(rnd.uniform(0, 0.02))

    closers: dict[str, asyncio.Task[Any]] = {}
    children: list[asyncio.Task[Any]] = []
    shared_deadline = loop.time() + rnd.uniform(0.0, 0.1)
    for i in range(rnd.randint(1, 4)):
        mode = rnd.choice(["plain", "timeout", "timeout", "cancel", "gather", "same_deadline"])
        abort = rnd.random() < 0.4
        delay = rnd.uniform(0, 0.03)
        interrupt_after = rnd.uniform(0.0, 0.1)

        async def closer(mode=mode, abort=abort, delay=delay, interrupt_after=interrupt_after):
            await asyncio.sleep(delay)
            if mode in ("plain", "cancel"):
                await session.close(abort_compaction=abort)
            elif mode == "timeout":
                async with asyncio.timeout(interrupt_after):
                    await session.close(abort_compaction=abort)
            elif mode == "same_deadline":
                async with asyncio.timeout_at(shared_deadline):
                    await session.close(abort_compaction=abort)
            else:
                first = asyncio.create_task(session.close(abort_compaction=abort))
                second = asyncio.create_task(session.close())
                children.extend([first, second])
                async with asyncio.timeout(interrupt_after):
                    _ = await asyncio.gather(first, second)

        task = asyncio.create_task(closer())
        closers[f"c{i}:{mode}"] = task
        if mode == "cancel":
            loop.call_later(delay + interrupt_after, task.cancel)

    everything = [*closers.values(), *background.values()]
    _, pending = await asyncio.wait(everything, timeout=4.0)
    if not pending and children:
        _, pending = await asyncio.wait(children, timeout=4.0)
    if pending:
        violations.append((seed, "hang", sorted(closers) + sorted(background)))
        for task in pending:
            _ = task.cancel()
        _ = await asyncio.gather(*pending, return_exceptions=True)
        return

    allowed = {
        "plain": {"ok"},
        "cancel": {"ok", "CancelledError"},
        "timeout": {"ok", "TimeoutError"},
        "same_deadline": {"ok", "TimeoutError"},
        "gather": {"ok", "TimeoutError"},
    }
    returned_normally = False
    for name, task in closers.items():
        outcome = (
            "CancelledError"
            if task.cancelled()
            else ("ok" if task.exception() is None else type(task.exception()).__name__)
        )
        if outcome not in allowed[name.split(":")[1]]:
            violations.append((seed, "closer-leaked", name, outcome))
        returned_normally = returned_normally or outcome == "ok"
    for task in children:
        if task.done() and not task.cancelled() and task.exception() is not None:
            violations.append((seed, "child-leaked", repr(task.exception())))
    for name, task in background.items():
        if task.cancelled():
            violations.append((seed, "bg-cancelled", name))
            continue
        exc = task.exception()
        if exc is None or type(exc).__name__ == "SessionClosedError" or name in inside:
            continue  # a send that closed the store under itself may fail (documented)
        violations.append((seed, "bg-leaked", name, repr(exc)[:80]))

    events = len(session._test_events)
    if returned_normally and not session._closed:
        violations.append((seed, "close-returned-but-open"))
    if session._closed:
        if events != 1 or (pool is None and session._store._conn is not None):
            violations.append((seed, "closed-but-bad", events, session._store._conn is not None))
        if session._compaction_engine.in_flight:
            violations.append((seed, "closed-with-compaction-in-flight"))
    else:
        _ = await session._compaction_engine.wait_for_pending()
        await asyncio.sleep(0)
        reopened = (
            not session._closing
            and session._close_done is None
            and session._store._conn is not None
            and events == 0
            and not session._compaction_abort.is_set()
        )
        try:
            _ = await asyncio.wait_for(session.send("after"), 3)
            usable = True
        except Exception as exc:
            usable = False
            violations.append((seed, "reopened-unusable", repr(exc)[:80]))
        if not (reopened and usable):
            violations.append(
                (seed, "half-closed", session._closing, session._store._conn is not None, events)
            )
        await session.close()
        if len(session._test_events) != 1:
            violations.append((seed, "events-after-final-close", len(session._test_events)))
    if pool is not None:
        await pool.close_all()
