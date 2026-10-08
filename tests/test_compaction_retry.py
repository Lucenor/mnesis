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
