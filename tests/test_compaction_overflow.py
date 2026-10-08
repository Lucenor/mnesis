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
import itertools
import json
import uuid

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


def _min_gap(triggers: list[int]) -> int:
    """Smallest number of turns between two compacting turns (large if < 2)."""
    idx = [i for i, t in enumerate(triggers) if t]
    return min((b - a for a, b in itertools.pairwise(idx)), default=len(triggers))


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
        assert 1 <= sum(t for t, _ in rows) <= 4, rows  # observed 3 in 24 turns
        assert _min_gap([t for t, _ in rows]) >= 4, rows
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
        assert 1 <= sum(per_turn) <= 7, per_turn  # observed 6; 24 when keyed on lifetime
        assert _min_gap(per_turn) >= 2, per_turn
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
        cfg = _cfg(tmp_path).model_copy(
            update={"compaction": CompactionConfig(compaction_output_budget=2000, auto=False)}
        )
        async with MnesisSession.open(model=MODEL, config=cfg) as s:
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


class TestOverflowCheckBuildFailure:
    async def test_record_survives_context_build_failure(self, tmp_path, monkeypatch):
        """A failing context build skips the overflow check and the snapshot, not the turn."""
        cfg = _cfg(tmp_path)
        async with MnesisSession.open(model=MODEL, config=cfg) as s:

            async def boom(*args, **kwargs):
                raise RuntimeError("build failed")

            monkeypatch.setattr(s._context_builder, "build", boom)
            result = await s.record("hello", "world", tokens=TokenUsage(input=10, output=5))

        assert result.compaction_triggered is False


class TestBuilderFallbackTruncation:
    async def test_fallback_path_truncates_but_reports_full_size(
        self, session_id, store, dag_store, estimator
    ):
        """Legacy sessions (no context_items rows) are budget-truncated too."""
        tight = ModelInfo(model_id="m", context_limit=4_000, max_output_tokens=200)
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
        conn = store._conn_or_raise()
        _ = await conn.execute("DELETE FROM context_items WHERE session_id = ?", (session_id,))
        await conn.commit()
        assert await store.get_context_items(session_id) == []

        ctx = await ContextBuilder(store, dag_store, estimator).build(session_id, tight, "sys", cfg)

        assert 0 < len(ctx.messages) < 10
        assert ctx.token_estimate <= ctx.budget.usable
        assert ctx.context_tokens > ctx.budget.usable


class TestLongRunSteadyState:
    @pytest.mark.parametrize("sys_tokens", [0, 900])
    async def test_long_run_no_compaction_streaks_no_blocking(self, tmp_path, sys_tokens):
        """100 turns with a non-trivial system prompt: no compaction streaks, never blocks."""
        system_prompt = (
            "You are helpful. " * (sys_tokens * 3 // 17) or "You are a helpful assistant."
        )
        async with MnesisSession.open(
            model=MODEL, config=_cfg(tmp_path), system_prompt=system_prompt
        ) as s:
            rows = await _drive_send(s, 100)
        triggers = [t for t, _ in rows]
        assert not any(blocked for _, blocked in rows), rows
        assert _min_gap(triggers) >= 2, triggers  # never compacts on consecutive turns
        assert sum(triggers) <= 25, triggers  # observed 15-19 (vs 40-70 before)


class TestEngineAgreesWithBuilder:
    @pytest.mark.parametrize("model", [MODEL, "gpt-4o"])
    async def test_tokens_before_after_match_builder_measure(self, tmp_path, model):
        """After compact(), the engine's own measure equals the builder's (system prompt too)."""
        cfg = MnesisConfig(
            store=StoreConfig(db_path=str(tmp_path / "t.db")),
            compaction=CompactionConfig(auto=False),
        )
        async with MnesisSession.open(model=model, config=cfg, system_prompt="Rules. " * 1500) as s:
            for i in range(12):
                await s.record(f"turn {i} " + "lorem ipsum dolor sit amet " * 100, "reply " * 200)
            builder_before = await s._measure_context(s.id)
            result = await s.compact()
            builder_after = await s._measure_context(s.id)

        assert result.tokens_before == builder_before
        assert result.tokens_after == builder_after
        assert result.tokens_after < result.tokens_before

    async def test_engine_without_measure_uses_estimator_basis(
        self, estimator, event_bus, store, dag_store, config, session_id
    ):
        """Without a measure callback both numbers are estimator-based (no system prompt)."""
        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            id_generator=lambda p: f"{p}_{uuid.uuid4().hex[:8]}",
            session_model=MODEL,
        )
        for i in range(8):
            await store.append_message(
                make_message(session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"m{i}")
            )
            await store.append_part(
                make_raw_part(
                    f"m{i}",
                    session_id,
                    content=json.dumps({"type": "text", "text": "word " * 200}),
                    part_id=f"p{i}",
                )
            )
        result = await engine.run_compaction(session_id)
        assert result.level_used > 0
        assert 0 < result.tokens_after < result.tokens_before


def _stub_result(session_id: str) -> CompactionResult:
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


async def _over_hard_session(tmp_path) -> MnesisSession:
    """Reload a session whose real context exceeds the hard limit of a tight window."""
    async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path, small=False)) as s:
        sid = s.id
        for i in range(30):
            await s.record(_big(i), _big(i))
    return await MnesisSession.load(sid, config=_cfg(tmp_path))


class TestHardLimitAfterInflightWait:
    async def test_stale_inflight_run_is_followed_by_one_fresh_run(self, tmp_path):
        """An in-flight run that leaves the context over hard does not let send() overflow."""
        s = await _over_hard_session(tmp_path)
        try:
            engine = s._compaction_engine
            real = engine.run_compaction
            real_calls = 0

            async def _counting(session_id: str, **kw):
                nonlocal real_calls
                if kw.get("until_under_hard"):  # the blocking hard-path run
                    real_calls += 1
                return await real(session_id, **kw)

            async def _noop() -> CompactionResult:
                return _stub_result(s.id)  # snapshotted earlier: shrinks nothing

            engine.run_compaction = _counting  # type: ignore[method-assign]
            engine._pending_task = asyncio.create_task(_noop())
            result = await s.send("continue")
            after = await s._measure_context(s.id)
        finally:
            await s.close()

        assert real_calls == 1
        assert result.compaction_result is not None and result.compaction_result.level_used > 0
        assert not engine.is_hard_overflow(after, s._model_info)

    async def test_retry_is_bounded_when_compaction_cannot_shrink(self, tmp_path):
        s = await _over_hard_session(tmp_path)
        try:
            engine = s._compaction_engine
            runs = 0

            async def _useless(session_id: str, **kw) -> CompactionResult:
                nonlocal runs
                runs += 1
                return _stub_result(session_id)

            engine.run_compaction = _useless  # type: ignore[method-assign]
            engine._pending_task = asyncio.create_task(_useless(s.id))
            result = await asyncio.wait_for(s.send("continue"), timeout=30)
        finally:
            await s.close()

        assert result.finish_reason in ("stop", "end_turn")
        # stale in-flight run + exactly one fresh blocking retry + the post-turn background
        # soft trigger (the context stays over soft since nothing can shrink it).
        assert runs == 3

    async def test_finished_background_run_is_remeasured_before_blocking(self, tmp_path):
        """If a background run already shrank the context, no fresh blocking run starts."""
        s = await _over_hard_session(tmp_path)
        try:
            engine = s._compaction_engine
            real = engine.run_compaction
            real_calls = 0

            async def _counting(session_id: str, **kw):
                nonlocal real_calls
                real_calls += 1
                return await real(session_id, **kw)

            engine.run_compaction = _counting  # type: ignore[method-assign]
            # Background run lands between the send's first measurement and its check.
            orig_build = s._context_builder.build
            first = True

            async def _build(*a, **k):
                nonlocal first
                ctx = await orig_build(*a, **k)
                if first:
                    first = False
                    engine._pending_task = asyncio.create_task(engine.run_compaction(s.id))
                    _ = await engine._pending_task  # swap committed after `ctx` was measured
                return ctx

            s._context_builder.build = _build  # type: ignore[method-assign]
            result = await s.send("continue")
        finally:
            await s.close()

        assert real_calls == 1  # only the background one
        assert result.finish_reason in ("stop", "end_turn")


class TestManualCompactionIsExclusive:
    async def test_compact_waits_for_background_run(self, tmp_path):
        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path)) as s:
            engine = s._compaction_engine
            active = 0
            max_active = 0
            order: list[str] = []
            gate = asyncio.Event()

            async def _fake(session_id: str, abort: object = None, **kw) -> CompactionResult:
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                order.append("start")
                if len(order) == 1:
                    await gate.wait()
                await asyncio.sleep(0)
                active -= 1
                order.append("end")
                return _stub_result(session_id)

            engine.run_compaction = _fake  # type: ignore[method-assign]
            model = s._model_info
            assert engine.check_and_trigger(s.id, model.context_limit, model) is True
            manual = asyncio.create_task(s.compact())
            await asyncio.sleep(0.05)
            assert not manual.done()  # waiting for the background run
            gate.set()
            _ = await manual

        assert max_active == 1
        assert order == ["start", "end", "start", "end"]


class TestConcurrentSendAndCompact:
    @pytest.mark.parametrize("compact_delay", [0, 0.001, 0.01, 0.05])
    async def test_send_and_compact_never_overlap(self, tmp_path, compact_delay):
        """compact() racing a send() that waits on a stale run: one run at a time."""
        s = await _over_hard_session(tmp_path)
        try:
            engine = s._compaction_engine
            usable = engine._usable_tokens(s._model_info)
            real = engine.run_compaction
            active = peak = 0

            async def _counting(session_id: str, **kw):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    return await real(session_id, **kw)
                finally:
                    active -= 1

            async def _stale() -> CompactionResult:
                await asyncio.sleep(0.1)
                return _stub_result(s.id)  # snapshotted earlier: shrinks nothing

            engine.run_compaction = _counting  # type: ignore[method-assign]
            engine._pending_task = asyncio.create_task(_stale())
            sent: list[int] = []
            orig = s._ensure_under_hard_limit

            async def _spy(ctx, sys_prompt):
                out = await orig(ctx, sys_prompt)
                sent.append(out[0].context_tokens)
                return out

            s._ensure_under_hard_limit = _spy  # type: ignore[method-assign]
            send_task = asyncio.create_task(s.send("continue"))
            await asyncio.sleep(compact_delay)
            compact_task = asyncio.create_task(s.compact())
            _ = await asyncio.wait_for(send_task, 30)
            _ = await asyncio.wait_for(compact_task, 30)
            _ = await engine.wait_for_pending()

            items = await s._store.get_context_items(s.id)
            in_context = {item_id for kind, item_id in items if kind == "summary"}
            live = {n.id for n in await s._dag_store.get_active_nodes(s.id)}
        finally:
            await s.close()

        assert peak == 1
        assert live <= in_context, "orphaned live summary node"
        assert sent and all(tokens < usable for tokens in sent), sent


class TestEngineMeasureAndTarget:
    def _cfg(self, store) -> MnesisConfig:
        return MnesisConfig(
            store=StoreConfig(db_path=str(store._config.db_path)),
            compaction=CompactionConfig(compaction_output_budget=1000),
        )

    async def _seed(self, store, session_id, n=8, words=600):
        for i in range(n):
            await store.append_message(
                make_message(session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"m{i}")
            )
            await store.append_part(
                make_raw_part(
                    f"m{i}",
                    session_id,
                    content=json.dumps({"type": "text", "text": "word " * words}),
                    part_id=f"p{i}",
                )
            )

    async def test_failing_measure_falls_back_to_estimator(
        self, estimator, event_bus, store, dag_store, session_id
    ):
        async def _boom(session_id: str) -> int:
            raise RuntimeError("measure failed")

        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            self._cfg(store),
            id_generator=lambda p: f"{p}_{uuid.uuid4().hex[:8]}",
            session_model=MODEL,
            context_measure=_boom,
        )
        await self._seed(store, session_id)
        result = await engine.run_compaction(session_id)
        assert result.level_used > 0
        assert 0 < result.tokens_after < result.tokens_before

    async def test_fallback_tokens_before_counts_only_in_context_messages(
        self, estimator, event_bus, store, dag_store, session_id
    ):
        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            self._cfg(store),
            id_generator=lambda p: f"{p}_{uuid.uuid4().hex[:8]}",
            session_model=MODEL,
        )
        await self._seed(store, session_id)
        first = await engine.run_compaction(session_id)
        await self._seed_more(store, session_id)
        context_ids = {i for k, i in await store.get_context_items(session_id) if k != "summary"}
        nodes = await dag_store.get_active_nodes(session_id)
        msgs = await store.get_messages_with_parts(session_id)
        expected = sum(
            estimator.estimate_message(m) for m in msgs if not m.is_summary and m.id in context_ids
        ) + sum(n.token_count for n in nodes)
        second = await engine.run_compaction(session_id)

        assert first.level_used > 0
        assert second.tokens_before == expected

    async def _seed_more(self, store, session_id):
        for i in range(8, 12):
            await store.append_message(
                make_message(session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"m{i}")
            )
            await store.append_part(
                make_raw_part(
                    f"m{i}",
                    session_id,
                    content=json.dumps({"type": "text", "text": "word " * 600}),
                    part_id=f"p{i}",
                )
            )

    @pytest.mark.parametrize(("measure_value", "condenses"), [(749, False), (750, True)])
    async def test_condensation_target_is_half_the_soft_threshold(
        self, estimator, event_bus, store, dag_store, session_id, measure_value, condenses
    ):
        """usable 2500 * soft 0.6 * 0.5 = 750: a measure below it stops, at it condenses."""
        from mnesis.compaction.levels import CondensationCandidate

        model_info = ModelInfo(model_id="m", context_limit=4_000, max_output_tokens=500)
        calls = 0

        async def _measure(session_id: str) -> int:
            return measure_value

        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            self._cfg(store),  # compaction_output_budget=1000 -> usable 2500
            id_generator=lambda p: f"{p}_{uuid.uuid4().hex[:8]}",
            session_model=MODEL,
            model_info=model_info,
            context_measure=_measure,
        )
        await self._seed(store, session_id)
        _ = await engine.run_compaction(session_id)  # first leaf
        await self._seed_more(store, session_id)

        async def _no_progress(nodes, *a, **k):
            nonlocal calls
            calls += 1
            tokens = sum(n.token_count for n in nodes)
            return CondensationCandidate(
                text="same",
                token_count=tokens,
                parent_node_ids=[n.id for n in nodes],
                compaction_level=1,
            )

        engine._run_condensation = _no_progress  # type: ignore[method-assign]
        _ = await engine.run_compaction(session_id)
        assert len(await dag_store.get_active_nodes(session_id)) >= 2
        assert calls == (1 if condenses else 0)
