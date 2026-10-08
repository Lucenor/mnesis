"""RetryConfig-driven retries for compaction LLM calls (shared with send())."""

from __future__ import annotations

import asyncio
import time
import types

import pytest
from litellm.exceptions import AuthenticationError, RateLimitError

import mnesis.compaction.engine as engine_mod
from mnesis.compaction.levels import condense_level1
from mnesis.events.bus import MnesisEvent
from mnesis.models.config import MnesisConfig, RetryConfig, SessionConfig, StoreConfig
from mnesis.models.message import ContextBudget
from mnesis.models.summary import SummaryNode
from mnesis.retry import RetriesExhaustedError, call_with_retry
from mnesis.tokens.estimator import TokenEstimator
from tests.conftest import make_message, make_raw_part

_SUMMARY = "## Goal\nretry summary\n\n## Completed Work\n- done\n"


def _rate_limit() -> RateLimitError:
    return RateLimitError("429", llm_provider="test", model="m")


@pytest.fixture
def retry_config(tmp_path) -> MnesisConfig:
    return MnesisConfig(
        store=StoreConfig(db_path=str(tmp_path / "retry.db")),
        session=SessionConfig(retry=RetryConfig(max_retries=2, base_delay=0.01, jitter=False)),
    )


async def _engine(
    session_id, store, dag_store, estimator, event_bus, config
) -> engine_mod.CompactionEngine:
    for i in range(8):
        msg = make_message(
            session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_retry_{i}"
        )
        await store.append_message(msg)
        await store.append_part(make_raw_part(msg.id, session_id, part_id=f"part_retry_{i}"))
    return engine_mod.CompactionEngine(
        store,
        dag_store,
        estimator,
        event_bus,
        config,
        session_model="anthropic/claude-haiku-4-5",
    )


class TestCompactionRetry:
    async def test_transient_error_retried_at_same_level(
        self, session_id, store, dag_store, estimator, event_bus, retry_config, monkeypatch
    ):
        """A 429 on L1 then success yields an L1 summary, with no escalation."""
        calls: list[int] = []

        async def flaky(**kwargs: object) -> str:
            calls.append(1)
            if len(calls) == 1:
                raise _rate_limit()
            return _SUMMARY

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: flaky)
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, retry_config)
        result = await engine.run_compaction(session_id)

        assert result.level_used == 1
        assert len(calls) == 2

    async def test_non_retryable_error_escalates_immediately(
        self, session_id, store, dag_store, estimator, event_bus, retry_config, monkeypatch
    ):
        """Non-retryable errors are not retried; the level fails and escalates."""
        calls: list[int] = []

        async def bad_auth(**kwargs: object) -> str:
            calls.append(1)
            raise AuthenticationError("nope", llm_provider="test", model="m")

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: bad_auth)
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, retry_config)
        result = await engine.run_compaction(session_id)

        assert result.level_used == 3
        # One attempt each for L1 and L2: no retry of the non-retryable error.
        assert len(calls) == 2

    async def test_retries_exhausted_skips_remaining_llm_levels(
        self, session_id, store, dag_store, estimator, event_bus, retry_config, monkeypatch
    ):
        calls: list[int] = []

        async def always_429(**kwargs: object) -> str:
            calls.append(1)
            raise _rate_limit()

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: always_429)
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, retry_config)
        result = await engine.run_compaction(session_id)

        assert result.level_used == 3
        # One retry sequence (1 + max_retries) on the outage; L2 is skipped, not retried.
        assert len(calls) == 3

    @staticmethod
    def _patch_litellm(monkeypatch) -> list[dict]:
        """Fail the first litellm call with a 429, then return a valid summary."""
        import litellm

        seen: list[dict] = []

        async def fake_acompletion(**kwargs):
            seen.append(kwargs)
            if len(seen) == 1:
                raise _rate_limit()
            msg = types.SimpleNamespace(content=_SUMMARY)
            choice = types.SimpleNamespace(message=msg, finish_reason="stop")
            return types.SimpleNamespace(choices=[choice])

        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        return seen

    async def test_default_config_leaves_litellm_retries_enabled(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        """max_retries == 0: Mnesis adds no retry and does not pass num_retries=0."""
        seen = self._patch_litellm(monkeypatch)
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, config)
        result = await engine.run_compaction(session_id)

        assert result.level_used == 2  # no Mnesis retry: L1's 429 escalated
        assert all("num_retries" not in kw for kw in seen)

    async def test_retries_enabled_disables_litellm_retries(
        self, session_id, store, dag_store, estimator, event_bus, retry_config, monkeypatch
    ):
        """max_retries > 0: Mnesis retries and passes num_retries=0 (no double retry)."""
        seen = self._patch_litellm(monkeypatch)
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, retry_config)
        result = await engine.run_compaction(session_id)

        assert result.level_used == 1
        assert len(seen) == 2
        assert all(kw["num_retries"] == 0 for kw in seen)

    async def test_abort_during_backoff_returns_stub(
        self, session_id, store, dag_store, estimator, event_bus, tmp_path, monkeypatch
    ):
        """Setting abort while backing off ends the run promptly with the stub."""
        cfg = MnesisConfig(
            store=StoreConfig(db_path=str(tmp_path / "retry_abort.db")),
            session=SessionConfig(retry=RetryConfig(max_retries=3, base_delay=30, jitter=False)),
        )
        first_call = asyncio.Event()

        async def always_429(**kwargs: object) -> str:
            first_call.set()
            raise _rate_limit()

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: always_429)
        failed: list[dict] = []
        event_bus.subscribe(MnesisEvent.COMPACTION_FAILED, lambda e, p: failed.append(p))
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, cfg)

        abort = asyncio.Event()
        task = asyncio.create_task(engine.run_compaction(session_id, abort=abort))
        await first_call.wait()
        await asyncio.sleep(0.05)  # now inside the 30s backoff
        started = time.monotonic()
        abort.set()
        result = await task

        assert time.monotonic() - started < 5
        assert result.level_used == 0
        assert result.summary_message_id == ""
        assert len(failed) == 1
        assert failed[0]["aborted"] is True

    async def test_failure_payload_not_aborted_on_error(
        self, session_id, store, dag_store, estimator, event_bus, config
    ):
        engine = engine_mod.CompactionEngine(
            store, dag_store, estimator, event_bus, config, session_model="m"
        )
        failed: list[dict] = []
        event_bus.subscribe(MnesisEvent.COMPACTION_FAILED, lambda e, p: failed.append(p))
        result = engine._failure_result(session_id, RuntimeError("boom"), time.time() * 1000)

        assert result.level_used == 0
        assert failed == [{"session_id": session_id, "error": "boom", "aborted": False}]


class TestIsRetryable:
    def test_classification(self):
        from mnesis.retry import is_retryable

        assert is_retryable(_rate_limit())
        assert not is_retryable(ValueError("x"))

    def test_without_litellm_nothing_is_retryable(self, monkeypatch):
        import sys

        from mnesis.retry import is_retryable

        monkeypatch.setitem(sys.modules, "litellm.exceptions", None)
        assert not is_retryable(_rate_limit())


class TestCallWithRetry:
    async def test_external_cancel_during_backoff_propagates(self):
        cfg = RetryConfig(max_retries=2, base_delay=30, jitter=False)
        started = asyncio.Event()

        async def fail() -> str:
            started.set()
            raise _rate_limit()

        task = asyncio.create_task(call_with_retry(fail, cfg, abort=asyncio.Event()))
        await started.wait()
        await asyncio.sleep(0.05)
        _ = task.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await task


def _node(node_id: str, content: str) -> SummaryNode:
    return SummaryNode(
        id=node_id,
        session_id="s",
        kind="leaf",
        span_start_message_id="a",
        span_end_message_id="b",
        content=content,
        token_count=max(1, len(content) // 4),
    )


_BUDGET = ContextBudget(model_context_limit=100_000, reserved_output_tokens=0, compaction_buffer=0)
_FAST = RetryConfig(max_retries=2, base_delay=0.01, jitter=False)


class TestCondensationRetry:
    async def test_condensation_429_then_success(self):
        """Condensation L1 retried after a 429 yields an L1 node (no escalation)."""
        calls: list[int] = []

        async def flaky(**kwargs: object) -> str:
            calls.append(1)
            if len(calls) == 1:
                raise _rate_limit()
            return _SUMMARY

        async def llm_call(**kwargs: object) -> str:
            return await call_with_retry(lambda: flaky(**kwargs), _FAST)

        nodes = [_node("a", "alpha " * 20), _node("b", "beta " * 20)]
        cond = await condense_level1(nodes, "m", _BUDGET, TokenEstimator(), llm_call)

        assert cond is not None
        assert cond.compaction_level == 1
        assert len(calls) == 2

    async def test_condensation_outage_skips_to_level3(
        self, store, dag_store, estimator, event_bus
    ):
        calls: list[int] = []

        async def outage(**kwargs: object) -> str:
            calls.append(1)
            raise RetriesExhaustedError("429")

        engine = engine_mod.CompactionEngine(
            store, dag_store, estimator, event_bus, MnesisConfig(), session_model="m"
        )
        nodes = [_node("a", "alpha " * 20), _node("b", "beta " * 20)]
        cond = await engine._run_condensation(nodes, "m", _BUDGET, outage, None)

        assert len(calls) == 1  # L2 skipped
        assert cond.compaction_level == 3


class TestCallWithRetryExhaustion:
    async def test_exhaustion_raises_distinct_error(self):
        async def fail() -> str:
            raise _rate_limit()

        with pytest.raises(RetriesExhaustedError) as info:
            _ = await call_with_retry(fail, _FAST)
        assert isinstance(info.value.__cause__, RateLimitError)

    async def test_no_retries_configured_propagates_original(self):
        async def fail() -> str:
            raise _rate_limit()

        with pytest.raises(RateLimitError):
            _ = await call_with_retry(fail, RetryConfig(max_retries=0))

    async def test_outage_fails_fast_for_later_calls_in_the_run(
        self, session_id, store, dag_store, estimator, event_bus, retry_config, monkeypatch
    ):
        """After one call exhausts its retries, no further LLM call is attempted."""
        calls: list[int] = []

        async def always_429(**kwargs: object) -> str:
            calls.append(1)
            raise _rate_limit()

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: always_429)
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, retry_config)
        t0 = time.monotonic()
        _ = await engine.run_compaction(session_id)
        assert len(calls) == 3
        assert time.monotonic() - t0 < 5


class TestOutageAndAbortBetweenLevels:
    async def test_later_llm_call_in_run_fails_fast_without_calling_provider(
        self, session_id, store, dag_store, estimator, event_bus, retry_config, monkeypatch
    ):
        """After an outage, a further llm_call in the same run re-raises it, no request."""
        calls: list[int] = []
        second: list[BaseException] = []

        async def always_429(**kwargs: object) -> str:
            calls.append(1)
            raise _rate_limit()

        async def two_calls(messages, model, budget, estimator, llm_call, **kw):
            with pytest.raises(RetriesExhaustedError):
                _ = await llm_call(messages=[], max_tokens=1)
            try:
                _ = await llm_call(messages=[], max_tokens=1)
            except RetriesExhaustedError as exc:
                second.append(exc)
            return None

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: always_429)
        monkeypatch.setattr(engine_mod, "level1_summarise", two_calls)
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, retry_config)
        result = await engine.run_compaction(session_id)

        assert len(calls) == 3  # only the first call hit the provider
        assert len(second) == 1
        assert result.level_used == 3

    async def test_abort_between_summarisation_levels(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        abort = asyncio.Event()

        async def empty_and_abort(**kwargs: object) -> str:
            abort.set()
            return ""

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: empty_and_abort)
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, config)
        result = await engine.run_compaction(session_id, abort=abort)

        assert result.level_used == 0  # stub: aborted before level 2
        assert result.summary_message_id == ""

    async def test_abort_between_condensation_levels(self, store, dag_store, estimator, event_bus):
        abort = asyncio.Event()

        async def empty_and_abort(**kwargs: object) -> str:
            abort.set()
            return ""

        engine = engine_mod.CompactionEngine(
            store, dag_store, estimator, event_bus, MnesisConfig(), session_model="m"
        )
        nodes = [_node("a", "alpha " * 20), _node("b", "beta " * 20)]
        with pytest.raises(asyncio.CancelledError):
            _ = await engine._run_condensation(nodes, "m", _BUDGET, empty_and_abort, abort)

    async def test_level2_outage_propagates_from_level_functions(self):
        from mnesis.compaction.levels import condense_level2, level2_summarise
        from tests.test_compaction import _make_messages_with_parts

        async def outage(**kwargs: object) -> str:
            raise RetriesExhaustedError("429")

        est = TokenEstimator()
        msgs = _make_messages_with_parts("sess_l2_outage", 6)
        with pytest.raises(RetriesExhaustedError):
            _ = await level2_summarise(msgs, "m", _BUDGET, est, outage)
        nodes = [_node("a", "alpha " * 20), _node("b", "beta " * 20)]
        with pytest.raises(RetriesExhaustedError):
            _ = await condense_level2(nodes, "m", _BUDGET, est, outage)


class TestRunAfterOutage:
    async def test_outage_state_does_not_leak_into_next_run(
        self, session_id, store, dag_store, estimator, event_bus, tmp_path, monkeypatch
    ):
        """Run 1 exhausts retries (L3); run 2 on the same engine calls the LLM again."""
        cfg = MnesisConfig(
            store=StoreConfig(db_path=str(tmp_path / "after_outage.db")),
            session=SessionConfig(retry=RetryConfig(max_retries=1, base_delay=0.0, jitter=False)),
        )
        calls: list[int] = []

        async def down(**kwargs: object) -> str:
            calls.append(1)
            raise _rate_limit()

        async def healthy(**kwargs: object) -> str:
            calls.append(1)
            return _SUMMARY

        current = [down]

        async def llm(**kwargs: object) -> str:
            return await current[0](**kwargs)

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: llm)
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, cfg)

        first = await engine.run_compaction(session_id)
        assert first.level_used == 3
        assert len(calls) == 2  # 1 + max_retries, then L2 skipped

        current[0] = healthy
        calls.clear()
        for i in range(8):
            msg = make_message(
                session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_after_{i}"
            )
            await store.append_message(msg)
            await store.append_part(make_raw_part(msg.id, session_id, part_id=f"part_after_{i}"))

        second = await engine.run_compaction(session_id)
        assert calls  # the LLM was called again; the outage did not persist
        assert second.level_used == 1


class TestCompactionRetryEvents:
    """B3: compaction retries publish LLM_RETRY with source/stage/level."""

    async def test_retry_publishes_event_with_source_and_level(
        self, session_id, store, dag_store, estimator, event_bus, retry_config, monkeypatch
    ):
        calls: list[int] = []

        async def flaky(**kwargs: object) -> str:
            calls.append(1)
            if len(calls) == 1:
                raise _rate_limit()
            return _SUMMARY

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: flaky)
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, retry_config)
        with structlog_capture() as logs:
            _ = await engine.run_compaction(session_id)

        retries = [p for e, p in event_bus.collected if e == MnesisEvent.LLM_RETRY]
        assert len(retries) == 1
        payload = retries[0]
        assert payload["source"] == "compaction"
        assert payload["stage"] == "summarisation"
        assert payload["level"] == 1
        assert payload["session_id"] == session_id
        assert payload["attempt"] == 1
        assert payload["max_retries"] == 2
        assert payload["error_type"].endswith("RateLimitError")
        assert payload["delay_seconds"] == pytest.approx(0.01)

        retry_logs = [entry for entry in logs if entry["event"] == "llm_call_retrying"]
        assert len(retry_logs) == 1
        assert retry_logs[0]["session_id"] == session_id
        assert retry_logs[0]["stage"] == "summarisation"
        assert retry_logs[0]["compaction_level"] == 1
        # The structlog ``level`` key stays the log level (not overwritten data).
        assert retry_logs[0]["log_level"] == "warning"

    async def test_stage_tag_set_around_condensation_calls(
        self, store, dag_store, estimator, event_bus
    ):
        engine = engine_mod.CompactionEngine(
            store, dag_store, estimator, event_bus, MnesisConfig(), session_model="m"
        )
        seen: list[tuple[str, int] | None] = []

        async def spy(**kwargs: object) -> str:
            seen.append(engine_mod._call_stage.get())
            return _SUMMARY

        nodes = [_node("a", "alpha " * 20), _node("b", "beta " * 20)]
        _ = await engine._run_condensation(nodes, "m", _BUDGET, spy, None)
        assert seen == [("condensation", 1)]
        assert engine_mod._call_stage.get() is None  # reset after the call

    async def test_callback_failure_does_not_fail_the_call(self):
        calls: list[int] = []

        async def flaky() -> str:
            calls.append(1)
            if len(calls) == 1:
                raise _rate_limit()
            return "ok"

        def boom(attempt: int, delay: float, exc: Exception) -> None:
            raise RuntimeError("subscriber bug")

        assert await call_with_retry(flaky, _FAST, on_retry=boom) == "ok"


def structlog_capture():
    from structlog.testing import capture_logs

    return capture_logs()


class TestWaitForPendingShield:
    """B1: cancelling a waiter must not cancel the background compaction."""

    async def test_cancelled_waiter_leaves_background_run_alive(
        self, store, dag_store, estimator, event_bus
    ):
        engine = engine_mod.CompactionEngine(
            store, dag_store, estimator, event_bus, MnesisConfig(), session_model="m"
        )
        release = asyncio.Event()

        async def slow_run(session_id: str, **kwargs: object):
            await release.wait()
            return engine._failure_result(session_id, RuntimeError("x"), 0.0)

        engine.run_compaction = slow_run  # type: ignore[method-assign]
        bg = asyncio.create_task(engine.run_compaction("s"))
        engine._pending_task = bg

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02):
                _ = await engine.wait_for_pending()

        assert not bg.cancelled() and not bg.done()
        assert engine.in_flight and engine.has_pending  # still tracked

        release.set()
        result = await engine.wait_for_pending()  # a later call collects it
        assert result is not None
        assert not engine.has_pending

    async def test_task_own_cancellation_still_propagates(
        self, store, dag_store, estimator, event_bus
    ):
        engine = engine_mod.CompactionEngine(
            store, dag_store, estimator, event_bus, MnesisConfig(), session_model="m"
        )

        async def forever() -> None:
            await asyncio.sleep(60)

        bg = asyncio.ensure_future(forever())
        engine._pending_task = bg  # type: ignore[assignment]
        waiter = asyncio.create_task(engine.wait_for_pending())
        await asyncio.sleep(0.01)
        _ = bg.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await waiter
        # The cancelled handle is reaped by the next call.
        assert await engine.wait_for_pending() is None
        assert not engine.has_pending


class TestCloseAbortCompaction:
    """B2: close(abort_compaction=True) ends a backoff wait promptly."""

    @staticmethod
    async def _session(tmp_path, monkeypatch, base_delay: float):
        from mnesis import MnesisSession

        cfg = MnesisConfig(
            store=StoreConfig(db_path=str(tmp_path / "abort.db")),
            session=SessionConfig(
                retry=RetryConfig(max_retries=2, base_delay=base_delay, jitter=False)
            ),
        )
        entered = asyncio.Event()

        async def always_429(**kwargs: object) -> str:
            entered.set()
            raise _rate_limit()

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: always_429)
        session = await MnesisSession.create(model="anthropic/claude-haiku-4-5", config=cfg)
        for i in range(4):
            _ = await session.record(f"question {i} " * 20, f"answer {i} " * 20)
        return session, entered

    async def test_abort_ends_backoff_and_returns_stub(self, tmp_path, monkeypatch):
        session, entered = await self._session(tmp_path, monkeypatch, base_delay=60.0)
        failed: list[dict] = []
        session.subscribe(MnesisEvent.COMPACTION_FAILED, lambda e, p: failed.append(p))
        compaction = asyncio.create_task(session.compact())
        await asyncio.wait_for(entered.wait(), timeout=5)
        await asyncio.sleep(0.05)  # now in the 60 s backoff

        t0 = time.monotonic()
        await session.close(abort_compaction=True)
        assert time.monotonic() - t0 < 5
        result = await asyncio.wait_for(compaction, timeout=5)
        assert result.level_used == 0 and result.summary_message_id == ""
        assert failed and failed[0]["aborted"] is True

    async def test_interrupted_close_clears_the_abort_flag(self, tmp_path, monkeypatch):
        """A close() that times out while draining must not leave every later run aborting."""
        session, _entered = await self._session(tmp_path, monkeypatch, base_delay=0.05)
        engine = session._compaction_engine
        real_wait = engine.wait_for_pending

        async def stuck() -> None:
            await asyncio.sleep(60)

        monkeypatch.setattr(engine, "wait_for_pending", stuck)
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.1):
                await session.close(abort_compaction=True)
        assert not session._compaction_abort.is_set()

        monkeypatch.setattr(engine, "wait_for_pending", real_wait)
        await session.close()

    async def test_default_close_waits_for_the_run(self, tmp_path, monkeypatch):
        """Without the option the run finishes (L3 after the outage) before close returns."""
        session, entered = await self._session(tmp_path, monkeypatch, base_delay=0.05)
        completed: list[dict] = []
        session.subscribe(MnesisEvent.COMPACTION_COMPLETED, lambda e, p: completed.append(p))
        engine = session._compaction_engine
        assert engine.check_and_trigger(
            session.id, 10**9, session._model_info, abort=session._compaction_abort
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        await session.close()
        assert completed and completed[0]["level_used"] == 3
        assert not session._compaction_abort.is_set()
