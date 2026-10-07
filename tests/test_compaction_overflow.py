"""Regression tests: compaction thresholds measure the current context.

Overflow decisions (soft and hard, in ``send()`` and ``record()``) must use the
size of the context window the next LLM call would carry, not lifetime billed
usage. Lifetime usage only ever grows; the current context shrinks when
compaction swaps messages for a summary.

Kept separate from ``test_session.py`` because it exercises session, engine and
context builder together.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from mnesis import MnesisConfig, MnesisSession, TokenUsage
from mnesis.compaction.engine import CompactionEngine
from mnesis.context.builder import ContextBuilder
from mnesis.events.bus import MnesisEvent
from mnesis.models.config import CompactionConfig, ModelInfo, StoreConfig
from mnesis.models.message import CompactionResult
from tests.conftest import make_message, make_raw_part

MODEL = "anthropic/claude-opus-4-6"


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setenv("MNESIS_MOCK_LLM", "1")


def _cfg(tmp_path, *, small: bool = True) -> MnesisConfig:
    # Small window: usable = 12000 - 1000 - 2000 = 9000 (soft threshold 5400).
    return MnesisConfig(
        store=StoreConfig(db_path=str(tmp_path / "test.db")),
        compaction=CompactionConfig(compaction_output_budget=2000),
        model_overrides={"context_limit": 12000, "max_output_tokens": 1000} if small else None,
    )


def _big(i: int) -> str:
    return f"turn {i} " + "lorem ipsum " * 130  # roughly 400 tokens


def _count_triggers(session: MnesisSession) -> list[int]:
    triggers: list[int] = []
    session.subscribe(MnesisEvent.COMPACTION_TRIGGERED, lambda e, p: triggers.append(1))
    return triggers


async def _drive_send(session: MnesisSession, n: int, start: int = 0) -> list[tuple[int, bool]]:
    """Run n send() turns; return (compactions_triggered, blocked_on_compaction) per turn."""
    triggers = _count_triggers(session)
    rows = []
    for i in range(start, start + n):
        before = len(triggers)
        result = await session.send(_big(i))
        rows.append((len(triggers) - before, result.compaction_result is not None))
        await session._compaction_engine.wait_for_pending()  # settle background run
    return rows


async def _drive_record(session: MnesisSession, n: int, start: int = 0) -> list[int]:
    triggers = _count_triggers(session)
    rows = []
    for i in range(start, start + n):
        before = len(triggers)
        await session.record(_big(i), _big(i))
        rows.append(len(triggers) - before)
        await session._compaction_engine.wait_for_pending()
    return rows


class TestSendCurrentContext:
    async def test_no_recompaction_after_context_fits(self, tmp_path):
        """Past the budget compaction runs, then later sends neither compact nor block."""
        cfg = _cfg(tmp_path)
        async with MnesisSession.open(model=MODEL, config=cfg) as s:
            rows = await _drive_send(s, 24)
            lifetime = s.token_usage.effective_total()
            usable = s._compaction_engine._usable_tokens(s._model_info)

        # Premise: lifetime usage is far past the budget, yet compaction is rare.
        assert lifetime > 3 * usable
        assert 1 <= sum(t for t, _ in rows) <= 5, rows
        assert all(t <= 1 for t, _ in rows), rows
        assert not any(blocked for _, blocked in rows), rows
        # The turn right after each compaction must be quiet.
        for idx, (t, _) in enumerate(rows[:-1]):
            if t:
                assert rows[idx + 1][0] == 0, rows

    async def test_reloaded_compacted_session_does_not_compact(self, tmp_path):
        """Lifetime usage > budget but compacted context is small: first send is quiet."""
        cfg = _cfg(tmp_path)
        async with MnesisSession.open(model=MODEL, config=cfg) as s:
            sid = s.id
            await _drive_send(s, 16)
            await s.compact()

        s2 = await MnesisSession.load(sid, config=cfg)
        try:
            usable = s2._compaction_engine._usable_tokens(s2._model_info)
            assert s2.token_usage.effective_total() > usable
            rows = await _drive_send(s2, 1, start=100)
        finally:
            await s2.close()
        assert rows == [(0, False)]

    async def test_reloaded_session_over_hard_threshold_compacts_blocking(self, tmp_path):
        """A reloaded session whose real context exceeds the hard limit compacts first."""
        # Write history under a roomy window (no compaction), reload with a tight one.
        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path, small=False)) as s:
            sid = s.id
            for i in range(30):
                await s.record(_big(i), _big(i))

        s2 = await MnesisSession.load(sid, config=_cfg(tmp_path))
        try:
            before = await s2._context_builder.build(
                sid, s2._model_info, s2._system_prompt, s2._config
            )
            assert s2._compaction_engine.is_hard_overflow(before.context_tokens, s2._model_info)
            result = await s2.send("continue")
            after = await s2._context_builder.build(
                sid, s2._model_info, s2._system_prompt, s2._config
            )
        finally:
            await s2.close()
        assert result.compaction_result is not None
        assert result.compaction_result.level_used > 0
        assert after.context_tokens < before.context_tokens


class TestRecordCurrentContext:
    async def test_no_recompaction_every_turn(self, tmp_path):
        """BYO-LLM path: lifetime usage keeps growing, compaction stays rare."""
        cfg = _cfg(tmp_path)
        async with MnesisSession.open(model=MODEL, config=cfg) as s:
            triggers = _count_triggers(s)
            per_turn = []
            for i in range(24):
                before = len(triggers)
                # Provider-reported usage carrying the full prompt each turn.
                await s.record(_big(i), _big(i), tokens=TokenUsage(input=6000, output=400))
                per_turn.append(len(triggers) - before)
                await s._compaction_engine.wait_for_pending()
            lifetime = s.token_usage.effective_total()
            usable = s._compaction_engine._usable_tokens(s._model_info)

        assert lifetime > 10 * usable
        assert 1 <= sum(per_turn) <= 9, per_turn  # vs. 24 when keyed on lifetime
        assert max(per_turn) <= 1

    async def test_reloaded_session_quiet_when_compacted_and_triggers_when_over(self, tmp_path):
        cfg = _cfg(tmp_path)
        async with MnesisSession.open(model=MODEL, config=cfg) as s:
            sid = s.id
            await _drive_record(s, 20)
            await s.compact()

        s2 = await MnesisSession.load(sid, config=cfg)
        try:
            assert await _drive_record(s2, 1, start=100) == [0]
            triggers = _count_triggers(s2)
            # One turn large enough to cross the soft threshold on its own.
            result = await s2.record("x " * 20000, "ok")
            await s2._compaction_engine.wait_for_pending()
        finally:
            await s2.close()
        assert result.compaction_triggered is True
        assert len(triggers) == 1

    async def test_rapid_overflowing_records_run_one_compaction(self, tmp_path):
        """Overflowing record() calls while a compaction is in flight schedule only one."""
        calls = 0
        gate = asyncio.Event()

        async def _slow(session_id: str, abort: object = None) -> CompactionResult:
            nonlocal calls
            calls += 1
            await gate.wait()
            return CompactionResult(
                session_id=session_id,
                summary_message_id="",
                level_used=0,
                compacted_message_count=0,
                summary_token_count=0,
                tokens_before=0,
                tokens_after=0,
                elapsed_ms=0.0,
            )

        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path)) as s:
            s._compaction_engine.run_compaction = _slow  # type: ignore[method-assign]
            first = await s.record("x " * 20000, "ok")
            second = await s.record("y " * 20000, "ok")
            await asyncio.sleep(0)
            gate.set()
            await s._compaction_engine.wait_for_pending()

        assert first.compaction_triggered is True
        assert second.compaction_triggered is False
        assert calls == 1


class TestEngineMeasure:
    def _engine(self, store, dag_store, estimator, event_bus, config) -> CompactionEngine:
        return CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_id"
        )

    def test_thresholds_accept_plain_int(self, estimator, event_bus, store, dag_store, config):
        engine = self._engine(store, dag_store, estimator, event_bus, config)
        model = ModelInfo(model_id="m", context_limit=100_000, max_output_tokens=4_000)
        # usable 76_000, soft 45_600
        assert engine.is_soft_overflow(45_599, model) is False
        assert engine.is_soft_overflow(45_600, model) is True
        assert engine.is_hard_overflow(75_999, model) is False
        assert engine.is_hard_overflow(76_000, model) is True

    async def test_check_and_trigger_is_noop_while_in_flight(
        self, estimator, event_bus, store, dag_store, config
    ):
        engine = self._engine(store, dag_store, estimator, event_bus, config)
        model = ModelInfo(model_id="m", context_limit=100_000, max_output_tokens=4_000)
        gate = asyncio.Event()
        runs = 0

        async def _fake(session_id: str, abort: object = None) -> CompactionResult:
            nonlocal runs
            runs += 1
            await gate.wait()
            return CompactionResult(
                session_id=session_id,
                summary_message_id="",
                level_used=0,
                compacted_message_count=0,
                summary_token_count=0,
                tokens_before=0,
                tokens_after=0,
                elapsed_ms=0.0,
            )

        engine.run_compaction = _fake  # type: ignore[method-assign]
        assert engine.check_and_trigger("s", 90_000, model) is True
        first_task = engine._pending_task
        assert engine.check_and_trigger("s", 90_000, model) is False
        assert engine._pending_task is first_task  # tracked task not orphaned
        gate.set()
        await engine.wait_for_pending()
        # Once finished, a new run may be scheduled again.
        gate.clear()
        assert engine.check_and_trigger("s", 90_000, model) is True
        gate.set()
        await engine.wait_for_pending()
        assert runs == 2


class TestBuilderContextTokens:
    @pytest.fixture
    def tight_model(self):
        return ModelInfo(model_id="m", context_limit=4_000, max_output_tokens=200)

    async def test_context_tokens_not_capped_by_budget(
        self, session_id, store, dag_store, estimator, tight_model
    ):
        """token_estimate is capped at the budget; context_tokens keeps counting."""
        cfg = MnesisConfig(compaction=CompactionConfig(compaction_output_budget=1000))
        for i in range(10):
            await store.append_message(
                make_message(session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"m{i}")
            )
            await store.append_part(
                make_raw_part(
                    f"m{i}",
                    session_id,
                    content=json.dumps({"type": "text", "text": "word " * 400}),
                    part_id=f"p{i}",
                )
            )
        builder = ContextBuilder(store, dag_store, estimator)
        ctx = await builder.build(session_id, tight_model, "sys", cfg)

        assert ctx.token_estimate <= ctx.budget.usable
        assert len(ctx.messages) < 10  # truncated by the budget
        assert ctx.context_tokens > ctx.budget.usable  # but the real size is reported
        assert ctx.context_tokens >= ctx.token_estimate

    async def test_context_tokens_defaults_to_estimate(self):
        from mnesis.context.builder import BuiltContext
        from mnesis.models.message import ContextBudget

        ctx = BuiltContext(
            messages=[],
            system_prompt="",
            token_estimate=42,
            budget=ContextBudget(model_context_limit=1000, reserved_output_tokens=0),
            has_summary=False,
        )
        assert ctx.context_tokens == 42


class TestEngineFitsMeasure:
    async def test_tokens_after_counts_older_summaries(self, tmp_path):
        """CompactionResult.tokens_after is the context size: tail plus all live summaries."""
        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path)) as s:
            for i in range(8):
                await s.record(_big(i), _big(i))
            first = await s.compact()
            for i in range(8, 16):
                await s.record(_big(i), _big(i))
            second = await s.compact()
            nodes = await s._dag_store.get_active_nodes(s.id)

        live_summary_tokens = sum(n.token_count for n in nodes)
        assert first.level_used > 0 and second.level_used > 0
        assert second.tokens_after >= live_summary_tokens
