"""RetryConfig-driven retries for compaction LLM calls (shared with send())."""

from __future__ import annotations

import asyncio
import time
import types

import pytest
from litellm.exceptions import AuthenticationError, RateLimitError

import mnesis.compaction.engine as engine_mod
from mnesis.events.bus import MnesisEvent
from mnesis.models.config import MnesisConfig, RetryConfig, SessionConfig, StoreConfig
from mnesis.retry import call_with_retry
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

    async def test_retries_exhausted_escalates(
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
        assert len(calls) == 6  # (1 + max_retries) attempts for each of L1 and L2

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
