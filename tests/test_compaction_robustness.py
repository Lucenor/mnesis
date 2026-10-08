"""Regression tests for the second round of review fixes on the compaction follow-ups.

Covers: file-ID footer priority, condensation retry suppression, once-per-episode pause
warning, silent empty compaction, fail-safe span handling and event-handler errors.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import structlog.testing

import mnesis.compaction.engine as engine_mod
from mnesis import MnesisConfig, MnesisSession
from mnesis.compaction.levels import (
    CondensationCandidate,
    SummaryCandidate,
    level1_summarise,
    level2_summarise,
)
from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.models.config import CompactionConfig, ModelInfo, StoreConfig
from mnesis.models.message import ContextBudget, MessageWithParts, TextPart
from mnesis.tokens.estimator import TokenEstimator
from tests.conftest import make_message, make_raw_part

MODEL = "anthropic/claude-opus-4-6"


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setenv("MNESIS_MOCK_LLM", "1")


def _cfg(tmp_path, **comp):
    return MnesisConfig(
        store=StoreConfig(db_path=str(tmp_path / "t.db")),
        compaction=CompactionConfig(compaction_output_budget=2_000, **comp),
        model_overrides={"context_limit": 12_000, "max_output_tokens": 1_000},
    )


def _turn(i: int, words: int = 130) -> str:
    return f"turn {i} " + "lorem ipsum " * words


async def _short_llm(**kwargs: object) -> str:
    return "## Goal\nshort"


class TestFooterPriority:
    async def test_footer_that_fits_alone_escalates_instead_of_dropping_ids(self, estimator):
        ids = [f"file_{i:016x}" for i in range(40)]
        msgs = []
        for i in range(8):
            chunk = " ".join(ids[i * 5 : (i + 1) * 5])
            msg = make_message(
                "sess_prio", role="user" if i % 2 == 0 else "assistant", msg_id=f"m{i}"
            )
            msgs.append(
                MessageWithParts(
                    message=msg, parts=[TextPart(text=f"t{i} {chunk} " + "word " * 400)]
                )
            )
        budget = ContextBudget(
            model_context_limit=600, reserved_output_tokens=100, compaction_buffer=100
        )

        async def long_llm(**kwargs: object) -> str:
            return "## Goal\n" + "word " * 330  # text + footer exceeds usable

        footer_alone = estimator.estimate("[LCM File IDs: " + ", ".join(ids) + "]")
        assert footer_alone < budget.usable
        assert await level1_summarise(msgs, "m", budget, estimator, long_llm) is None
        cand = await level2_summarise(msgs, "m", budget, estimator, _short_llm)
        assert cand is not None
        assert all(fid in cand.text for fid in ids[: 5 * cand.messages_covered])


class TestCompactionRobustness:
    async def _two_node_session(self, tmp_path):
        s = await MnesisSession.create(
            model=MODEL, config=_cfg(tmp_path, auto=False, condensation_enabled=False)
        )
        for i, words in enumerate((400, 1500, 1500)):
            _ = await s.record(_turn(i, words), "ok")
        _ = await s.compact()
        for i in (3, 4):
            _ = await s.record(_turn(i, 1500), "ok")
        _ = await s.compact()
        assert len(await s._dag_store.get_active_nodes(s.id)) == 2
        s._config.compaction.condensation_enabled = True
        return s

    async def test_failed_condensation_is_not_retried_for_the_same_nodes(self, tmp_path):
        s = await self._two_node_session(tmp_path)
        try:
            engine = s._compaction_engine
            attempts = 0

            async def no_progress(nodes, *args, **kwargs):
                nonlocal attempts
                attempts += 1
                return CondensationCandidate(
                    text="x", token_count=10**6, parent_node_ids=[n.id for n in nodes]
                )

            engine._run_condensation = no_progress  # type: ignore[method-assign]
            _ = await s.compact()
            _ = await s.compact()
            _ = await s.compact()
            assert attempts == 1
            # A new live node changes the set, so condensation is attempted again.
            _ = await s.record(_turn(5, 1500), "ok")
            _ = await s.compact()
            assert attempts == 2
        finally:
            await s.close()

    def test_pause_warning_logged_once_per_episode(self, store, dag_store, event_bus, estimator):
        info = ModelInfo(model_id="m", context_limit=12_000, max_output_tokens=1_000)
        engine = engine_mod.CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            MnesisConfig(compaction=CompactionConfig(compaction_output_budget=2_000)),
            session_model=MODEL,
            model_info=info,
        )

        def stall(after: int = 6000) -> None:
            engine._note_run_outcome(
                "s", tokens_before=9000, tokens_after=after, more_to_compact=False
            )

        with structlog.testing.capture_logs() as logs:
            stall()
            stall()
            stall(after=1000)  # a healthy run ends the episode
            stall()
        warned = [e for e in logs if e["event"] == "compaction_cannot_reduce_context"]
        assert len(warned) == 2

    async def test_empty_compaction_is_silent(self, tmp_path, store, dag_store, event_bus):
        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path)) as s:
            events: list[MnesisEvent] = []
            s.subscribe(MnesisEvent.COMPACTION_COMPLETED, lambda e, p: events.append(e))
            result = await s.compact()
            assert result.level_used == 0 and result.summary_message_id == ""
            assert not events
        bare = engine_mod.CompactionEngine(
            store, dag_store, TokenEstimator(), event_bus, MnesisConfig()
        )
        await store.create_session("sess_empty", model_id=MODEL, agent="t")
        bare_result = await bare.run_compaction("sess_empty")
        assert bare_result.level_used == 0
        assert not event_bus.collected

    async def test_unlocatable_span_never_touches_the_context(
        self, store, dag_store, event_bus, estimator
    ):
        sid = "sess_badspan"
        await store.create_session(sid, model_id=MODEL, agent="t")
        for i in range(8):
            mid = f"u{i}"
            _ = await store.append_message(
                make_message(sid, role="user" if i % 2 == 0 else "assistant", msg_id=mid)
            )
            _ = await store.append_part(
                make_raw_part(
                    mid,
                    sid,
                    content=json.dumps({"type": "text", "text": "hi"}),
                    part_id=f"p{i}",
                )
            )
        engine = engine_mod.CompactionEngine(
            store, dag_store, estimator, event_bus, MnesisConfig(), session_model=MODEL
        )

        async def bogus(*args, **kwargs):
            return SummaryCandidate(
                text="x",
                token_count=1,
                span_start_message_id="nope",
                span_end_message_id="nada",
                compaction_level=1,
                messages_covered=1,
            )

        engine._run_summarisation = bogus  # type: ignore[method-assign]
        before = await store.get_context_items(sid)
        _ = await engine.run_compaction(sid)
        assert await store.get_context_items(sid) == before
        assert await dag_store.get_active_nodes(sid) == []


class TestEventBusHandlerErrors:
    async def test_raising_handlers_do_not_break_publish(self):
        bus = EventBus()

        def sync_boom(event, payload):
            raise RuntimeError("sync boom")

        async def async_boom(event, payload):
            raise RuntimeError("async boom")

        bus.subscribe(MnesisEvent.COMPACTION_FAILED, sync_boom)
        bus.subscribe(MnesisEvent.COMPACTION_FAILED, async_boom)
        with structlog.testing.capture_logs() as logs:
            bus.publish(MnesisEvent.COMPACTION_FAILED, {"error": "x"})
            await asyncio.sleep(0.05)
        errors = [e for e in logs if e["event"] == "event_handler_error"]
        assert {e["error"] for e in errors} == {"sync boom", "async boom"}

    async def test_raising_failed_handler_does_not_make_run_compaction_raise(
        self, store, dag_store, estimator
    ):
        bus = EventBus()

        def boom(event, payload):
            raise RuntimeError("handler boom")

        bus.subscribe(MnesisEvent.COMPACTION_FAILED, boom)
        sid = "sess_cf"
        await store.create_session(sid, model_id=MODEL, agent="t")
        _ = await store.append_message(make_message(sid, role="user", msg_id="u0"))
        _ = await store.append_part(
            make_raw_part(
                "u0", sid, content=json.dumps({"type": "text", "text": "hi"}), part_id="p0"
            )
        )
        # No compaction model: the run fails and publishes COMPACTION_FAILED.
        engine = engine_mod.CompactionEngine(store, dag_store, estimator, bus, MnesisConfig())
        result = await engine.run_compaction(sid)
        assert result.level_used == 0


class TestCondensationFallbackAndWorkLeft:
    async def test_bloated_llm_condensation_falls_to_level3_in_the_same_run(
        self, tmp_path, monkeypatch
    ):
        s = await TestCompactionRobustness()._two_node_session(tmp_path)
        try:
            calls = 0

            async def bloated(nodes, *args, **kwargs):
                nonlocal calls
                calls += 1
                return CondensationCandidate(
                    text="x " * 5000,
                    token_count=10**6,
                    parent_node_ids=[n.id for n in nodes],
                    compaction_level=1,
                )

            def deterministic(nodes, estimator, budget):
                return CondensationCandidate(
                    text="[CONDENSED] det",
                    token_count=1,
                    parent_node_ids=[n.id for n in nodes],
                    compaction_level=3,
                )

            monkeypatch.setattr(engine_mod, "condense_level1", bloated)
            monkeypatch.setattr(engine_mod, "condense_level3_deterministic", deterministic)
            result = await s.compact()
            nodes = await s._dag_store.get_active_nodes(s.id)
            assert calls == 1
            assert len(nodes) == 1 and nodes[0].kind == "condensed"
            assert nodes[0].compaction_level == 3
            assert result.level_used == 3
        finally:
            await s.close()

    async def test_exception_path_resets_work_left(self, store, dag_store, event_bus, estimator):
        info = ModelInfo(model_id="m", context_limit=12_000, max_output_tokens=1_000)
        engine = engine_mod.CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            MnesisConfig(compaction=CompactionConfig(compaction_output_budget=2_000)),
            session_model=MODEL,
            model_info=info,
        )
        engine._note_run_outcome("s", tokens_before=9000, tokens_after=6000, more_to_compact=True)
        assert engine.more_to_compact

        async def boom(session_id: str):
            raise RuntimeError("store down")

        store.get_messages_with_parts = boom  # type: ignore[method-assign]
        await store.create_session("sess_boom", model_id=MODEL, agent="t")
        result = await engine.run_compaction("sess_boom")
        assert result.level_used == 0
        assert not engine.more_to_compact

    async def test_nothing_compactable_does_not_trigger_an_extra_run(self, tmp_path):
        async with MnesisSession.open(
            model=MODEL, config=_cfg(tmp_path), system_prompt="short"
        ) as s:
            engine = s._compaction_engine
            runs: list[bool] = []
            real = engine.run_compaction

            async def spy(session_id, **kw):
                runs.append(bool(kw.get("until_under_hard")))
                return await real(session_id, **kw)

            engine.run_compaction = spy  # type: ignore[method-assign]
            _ = await s.record("turn 0 hi", "ok")
            # The per-turn prompt alone is over the hard limit and nothing can shrink.
            _ = await s.send("turn 1 hi", system_prompt="word " * 9_000)
            assert runs == [True]  # the blocking run only; no useless full-drain rerun
