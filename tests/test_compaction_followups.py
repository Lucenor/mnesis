"""Regression tests for the compaction follow-ups.

Covers: capped-span history loss, re-summarising content already in a summary,
model-aware budgets/estimator, bounded condensation level 3, raw file-ID
extraction, stalled-compaction detection, tool-schema accounting and misc nits.
Session-level tests live here (rather than ``test_session.py``) because they
exercise session, engine, builder and store together.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import re
import uuid
from pathlib import Path

import pytest

import mnesis.compaction.engine as engine_mod
from mnesis import MnesisConfig, MnesisSession
from mnesis.compaction.file_ids import (
    extract_file_ids,
    extract_file_ids_from_messages,
    most_recent_file_ids,
    most_recent_file_ids_from_nodes,
    strip_file_ids_footer,
)
from mnesis.compaction.levels import (
    _CONDENSE_LEVEL3_MINIMAL_HEADER,
    MIN_MESSAGES_TO_SUMMARISE,
    _messages_to_summarise,
    _truncate_to_tokens,
    condense_level3_deterministic,
    level1_summarise,
    level2_summarise,
)
from mnesis.events.bus import MnesisEvent
from mnesis.models.config import (
    CompactionConfig,
    ModelInfo,
    StoreConfig,
    check_compaction_budget,
)
from mnesis.models.message import (
    ContextBudget,
    FileRefPart,
    MessageWithParts,
    TextPart,
    ToolPart,
    ToolStatus,
)
from mnesis.models.summary import SummaryNode
from mnesis.tokens.estimator import TokenEstimator
from tests.conftest import make_message, make_raw_part

MODEL = "anthropic/claude-opus-4-6"


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch):
    monkeypatch.setenv("MNESIS_MOCK_LLM", "1")


def _cfg(tmp_path, *, window: int = 12_000, out: int = 1_000, budget: int = 2_000, **comp):
    return MnesisConfig(
        store=StoreConfig(db_path=str(tmp_path / "t.db")),
        compaction=CompactionConfig(compaction_output_budget=budget, **comp),
        model_overrides={"context_limit": window, "max_output_tokens": out},
    )


def _turn(i: int, words: int = 130) -> str:
    return f"turn {i} " + "lorem ipsum " * words


def _chat(
    session_id: str, count: int, chars: int = 400, prefix: str = "m"
) -> list[MessageWithParts]:
    out = []
    for i in range(count):
        msg = make_message(
            session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"{prefix}_{i:04d}"
        )
        out.append(MessageWithParts(message=msg, parts=[TextPart(text=f"t{i} " + "y" * chars)]))
    return out


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


async def _short_llm(**kwargs: object) -> str:
    return "## Goal\nshort"


def _capture_llm(monkeypatch) -> list[set[int]]:
    """Patch the engine LLM so every summarisation call records which turns it saw."""
    seen: list[set[int]] = []
    orig = engine_mod._make_llm_call

    def mk(model: str):
        inner = orig(model)

        async def _call(**kw):
            content = kw["messages"][0]["content"]
            if "<conversation>" in content:
                conv = content.split("<conversation>")[1]
                seen.append({int(x) for x in re.findall(r"turn (\d+) ", conv)})
            return await inner(**kw)

        return _call

    monkeypatch.setattr(engine_mod, "_make_llm_call", mk)
    return seen


async def _raw_user_turns(s: MnesisSession) -> set[int]:
    ctx = await s._context_builder.build(s.id, s._model_info, s._system_prompt, s._config)
    return {
        int(x)
        for m in ctx.messages
        if m.role == "user"
        for x in re.findall(r"turn (\d+) ", str(m.content))
    }


# ── H1: the recorded span equals what was summarised ──────────────────────────


class TestCappedSpanMatchesSummarisedInput:
    async def test_level1_span_is_capped_prefix(self, estimator):
        budget = ContextBudget(
            model_context_limit=50_000, reserved_output_tokens=4_000, compaction_buffer=10_000
        )
        msgs = _chat("sess_h1", 40, chars=2000)  # ~500 tokens each
        seen: list[str] = []

        async def llm(**kwargs):
            seen.append(kwargs["messages"][0]["content"])
            return "## Goal\nshort"

        cand = await level1_summarise(msgs, "m", budget, estimator, llm, model_context_limit=4000)
        assert cand is not None
        covered = cand.messages_covered
        assert MIN_MESSAGES_TO_SUMMARISE <= covered < len(_messages_to_summarise(msgs))
        assert cand.span_start_message_id == msgs[0].id
        assert cand.span_end_message_id == msgs[covered - 1].id
        assert f"t{covered - 1} " in seen[0]
        assert f"t{covered} " not in seen[0]

    async def test_level2_span_is_capped_prefix(self, estimator):
        budget = ContextBudget(
            model_context_limit=50_000, reserved_output_tokens=4_000, compaction_buffer=10_000
        )
        msgs = _chat("sess_h1b", 40, chars=2000)
        cand = await level2_summarise(
            msgs, "m", budget, estimator, _short_llm, model_context_limit=4000
        )
        assert cand is not None
        assert cand.span_end_message_id == msgs[cand.messages_covered - 1].id
        assert cand.messages_covered < len(_messages_to_summarise(msgs))

    async def test_uncapped_span_covers_whole_summarisable_range(self, estimator):
        budget = ContextBudget(
            model_context_limit=50_000, reserved_output_tokens=4_000, compaction_buffer=10_000
        )
        msgs = _chat("sess_h1c", 12, chars=600)
        cand = await level1_summarise(msgs, "m", budget, estimator, _short_llm)
        assert cand is not None
        expected = _messages_to_summarise(msgs)
        assert cand.span_end_message_id == expected[-1].id
        assert cand.messages_covered == len(expected)

    async def test_no_turn_leaves_context_unsummarised(self, tmp_path, monkeypatch):
        """End to end: every turn is either raw in context or was given to a summariser."""
        seen = _capture_llm(monkeypatch)
        n = 40
        async with MnesisSession.open(
            model=MODEL, config=_cfg(tmp_path, auto=False, condensation_enabled=False)
        ) as s:
            for i in range(n):
                _ = await s.record(_turn(i), "reply " + "lorem ipsum " * 60)
            result = await s.compact()
            raw = await _raw_user_turns(s)
        summarised = set().union(*seen) if seen else set()
        assert len(seen) > 1, "a 12K window must need several capped summarisation passes"
        assert result.compacted_message_count > 0
        assert raw | summarised >= set(range(n)), sorted(set(range(n)) - raw - summarised)
        assert raw.isdisjoint(summarised)
        assert raw == {n - 2, n - 1}  # protected tail stays verbatim


# ── H2 / N1: only raw messages still in context are summarised ────────────────


class TestSummariseOnlyWhatIsInContext:
    async def test_second_compaction_does_not_resummarise(self, tmp_path, monkeypatch):
        seen = _capture_llm(monkeypatch)
        cfg = _cfg(tmp_path, window=200_000, out=8_192, budget=20_000, auto=False)
        cfg = cfg.model_copy(
            update={"compaction": cfg.compaction.model_copy(update={"condensation_enabled": False})}
        )
        async with MnesisSession.open(model=MODEL, config=cfg) as s:
            for i in range(6):
                _ = await s.record(_turn(i, 20), _turn(i, 20))
            _ = await s.compact()
            for i in range(6, 12):
                _ = await s.record(_turn(i, 20), _turn(i, 20))
            _ = await s.compact()
            msgs = await s._store.get_messages_with_parts(s.id)
            order = {m.id: i for i, m in enumerate(m for m in msgs if not m.is_summary)}
            nodes = sorted(
                await s._dag_store.get_active_nodes(s.id), key=lambda n: n.span_start_message_id
            )
        assert len(seen) == 2
        assert seen[0].isdisjoint(seen[1]), "old content summarised twice"
        spans = [(order[n.span_start_message_id], order[n.span_end_message_id]) for n in nodes]
        assert len(spans) == 2
        assert spans[0][1] < spans[1][0], spans

    async def test_repeat_compact_with_only_protected_tail_is_a_no_op(self, tmp_path):
        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path, auto=False)) as s:
            for i in range(8):
                _ = await s.record(_turn(i), _turn(i))
            first = await s.compact()
            assert first.level_used > 0
            before = await s._store.get_context_items(s.id)
            second = await s.compact()
            after = await s._store.get_context_items(s.id)
            raw = await _raw_user_turns(s)
        assert second.level_used == 0 and second.summary_message_id == ""
        assert before == after
        assert raw == {6, 7}  # tail not turned into a level-3 digest

    async def test_context_with_only_summaries_returns_no_op(self, store, dag_store, event_bus):
        """No raw rows in context_items (all compacted): not mistaken for a legacy database."""
        sid = "sess_onlysum"
        await store.create_session(sid, model_id=MODEL, agent="t")
        for i in range(2):
            _ = await store.append_message(make_message(sid, role="user", msg_id=f"u{i}"))
            _ = await store.append_part(
                make_raw_part(
                    f"u{i}",
                    sid,
                    content=json.dumps({"type": "text", "text": "hi"}),
                    part_id=f"p{i}",
                )
            )
        await store.swap_context_items(sid, ["u0", "u1"], "sum_x")
        engine = engine_mod.CompactionEngine(
            store,
            dag_store,
            TokenEstimator(),
            event_bus,
            MnesisConfig(),
            id_generator=lambda p: f"{p}_x",
            session_model=MODEL,
        )
        result = await engine.run_compaction(sid)
        assert result.level_used == 0 and result.summary_message_id == ""
        assert MnesisEvent.COMPACTION_FAILED not in [e for e, _ in event_bus.collected]

    async def test_legacy_database_without_context_items_summarises_everything(
        self, tmp_path, monkeypatch
    ):
        seen = _capture_llm(monkeypatch)
        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path, auto=False)) as s:
            for i in range(6):
                _ = await s.record(_turn(i, 20), _turn(i, 20))
            conn = s._store._conn_or_raise()
            _ = await conn.execute("DELETE FROM context_items")
            await conn.commit()
            result = await s.compact()
        assert result.level_used > 0
        assert seen and {0, 1, 2, 3} <= set().union(*seen)

    async def test_later_pass_failure_keeps_earlier_passes(self, tmp_path, monkeypatch):
        """If the LLM dies after the first capped pass, that pass is kept (no level 3 drain)."""
        calls = 0
        orig = engine_mod._make_llm_call

        def mk(model: str):
            inner = orig(model)

            async def _call(**kw):
                nonlocal calls
                calls += 1
                if calls > 1:
                    raise RuntimeError("llm down")
                return await inner(**kw)

            return _call

        monkeypatch.setattr(engine_mod, "_make_llm_call", mk)
        async with MnesisSession.open(
            model=MODEL, config=_cfg(tmp_path, auto=False, condensation_enabled=False)
        ) as s:
            for i in range(40):
                _ = await s.record(_turn(i), "reply " + "lorem ipsum " * 60)
            result = await s.compact()
            raw = await _raw_user_turns(s)
        assert result.level_used == 1
        assert 0 < len(raw) < 40 and {38, 39} <= raw  # later turns stay verbatim


# ── B1 / B2: budget and estimator follow the session model ────────────────────


class TestModelAwareBudget:
    async def _engine(self, store, dag_store, event_bus, info, cfg=None):
        return engine_mod.CompactionEngine(
            store,
            dag_store,
            TokenEstimator(),
            event_bus,
            cfg or MnesisConfig(compaction=CompactionConfig(compaction_output_budget=2_000)),
            id_generator=lambda p: f"{p}_x",
            session_model=info.model_id,
            model_info=info,
        )

    async def test_budget_comes_from_model_info(self, store, dag_store, event_bus):
        info = ModelInfo(
            model_id="tiny", context_limit=8_000, max_output_tokens=1_000, encoding="cl100k_base"
        )
        engine = await self._engine(store, dag_store, event_bus, info)
        assert engine._summary_budget().usable == 5_000  # 8000 - 1000 - 2000

    async def test_bare_engine_and_unknown_window_fall_back(self, store, dag_store, event_bus):
        bare = engine_mod.CompactionEngine(
            store, dag_store, TokenEstimator(), event_bus, MnesisConfig(), session_model=MODEL
        )
        assert bare._summary_budget().usable == 200_000 - 8_192 - 20_000
        zero = ModelInfo(model_id="z", context_limit=0, max_output_tokens=0)
        engine = await self._engine(store, dag_store, event_bus, zero)
        assert engine._summary_budget().model_context_limit == 200_000

    async def test_level3_summary_fits_small_session_window(self, tmp_path, monkeypatch):
        async def boom(**kw):
            raise RuntimeError("down")

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model: boom)
        cfg = _cfg(tmp_path, window=8_000, out=1_000, budget=2_000, auto=False)
        async with MnesisSession.open(model=MODEL, config=cfg) as s:
            for i in range(30):
                _ = await s.record(_turn(i, 60), _turn(i, 60))
            result = await s.compact()
            usable = s._compaction_engine._usable_tokens(s._model_info)
            ctx = await s._context_builder.build(s.id, s._model_info, s._system_prompt, s._config)
        assert result.level_used == 3
        assert result.summary_token_count <= usable == 5_000
        assert ctx.summary_token_count == result.summary_token_count

    @pytest.mark.parametrize("model", ["anthropic/claude-opus-4-6", "gpt-4o"])
    async def test_summary_token_count_matches_builder_estimator(self, tmp_path, model):
        cfg = MnesisConfig(store=StoreConfig(db_path=str(tmp_path / "e.db")))
        async with MnesisSession.open(model=model, config=cfg) as s:
            for i in range(6):
                _ = await s.record(_turn(i, 40), _turn(i, 40))
            _ = await s.compact()
            nodes = await s._dag_store.get_active_nodes(s.id)
            ctx = await s._context_builder.build(s.id, s._model_info, s._system_prompt, s._config)
            expect = s._estimator.estimate(nodes[0].content, s._model_info)
        assert nodes[0].token_count == expect
        assert ctx.summary_token_count == expect

    async def test_level3_sizing_uses_session_tokenizer(self):
        """A footer that fits under the heuristic can overflow o200k; sizing must use o200k."""
        from mnesis.compaction.levels import level3_deterministic

        info = ModelInfo(
            model_id="gpt-4o",
            context_limit=12_000,
            max_output_tokens=1_000,
            encoding="o200k_base",
        )
        est = TokenEstimator().for_model(info)
        msgs = [
            MessageWithParts(
                message=make_message("s", role="user", msg_id="m0"),
                parts=[TextPart(text=" ".join(f"file_{i:016x}" for i in range(1000)))],
            )
        ]
        budget = ContextBudget(
            model_context_limit=12_000, reserved_output_tokens=1_000, compaction_buffer=2_000
        )
        cand = level3_deterministic(msgs, budget, est)
        assert est.estimate(cand.text) == cand.token_count <= budget.usable
        assert TokenEstimator().estimate(cand.text) < cand.token_count  # heuristic undercounts


# ── Estimator view ────────────────────────────────────────────────────────────


class TestEstimatorForModel:
    def test_none_returns_self(self):
        e = TokenEstimator()
        assert e.for_model(None) is e

    def test_default_model_applies_and_explicit_wins(self):
        claude = ModelInfo(model_id="c", encoding="claude_heuristic")
        view = TokenEstimator().for_model(claude)
        assert view.estimate("x" * 30) == 10  # len // 3
        assert TokenEstimator().estimate("x" * 30) == 7  # len // 4
        other = ModelInfo(model_id="u", encoding="unknown")
        assert view.estimate("x" * 30, other) == 7

    def test_view_estimate_message_uses_default_model(self):
        claude = ModelInfo(model_id="c", encoding="claude_heuristic")
        msg = MessageWithParts(
            message=make_message("s", msg_id="mm"), parts=[TextPart(text="x" * 300)]
        )
        base = TokenEstimator()
        assert base.for_model(claude).estimate_message(msg) == base.estimate_message(msg, claude)
        assert base.for_model(claude).estimate_message(msg) > base.estimate_message(msg)

    def test_caches_are_isolated_but_encoders_shared(self):
        base = TokenEstimator()
        a = base.for_model(ModelInfo(model_id="a", encoding="claude_heuristic"))
        b = base.for_model(ModelInfo(model_id="b", encoding="unknown"))
        assert a.estimate_cached("x" * 30, "k") == 10
        assert b.estimate_cached("x" * 30, "k") == 7
        assert a._encoder_cache is base._encoder_cache

    def test_heuristic_only_is_preserved(self):
        view = TokenEstimator(heuristic_only=True).for_model(
            ModelInfo(model_id="o", encoding="o200k_base")
        )
        assert view.estimate("x" * 30) == 7

    def test_file_ref_tokens_rendered_with_thousands_separator(self):
        """The estimate renders FileRefPart exactly as ContextBuilder does."""
        from mnesis.context.builder import ContextBuilder

        part = FileRefPart(
            content_id="c" * 64,
            path="/p/x.py",
            file_type="python",
            token_count=12_345,
            exploration_summary="s",
        )
        msg = MessageWithParts(message=make_message("s", msg_id="fr"), parts=[part])
        est = TokenEstimator(heuristic_only=True)
        rendered = ContextBuilder._render_file_ref(part)
        assert est.estimate_message(msg) == 4 + est.estimate(rendered)


# ── B3: compaction_output_budget validation ───────────────────────────────────


class TestCompactionBudgetValidation:
    def test_static_upper_bound(self):
        CompactionConfig(compaction_output_budget=100_000)
        with pytest.raises(ValueError):
            CompactionConfig(compaction_output_budget=100_001)

    def test_raises_when_no_usable_context(self):
        info = ModelInfo(model_id="small", context_limit=8_000, max_output_tokens=4_000)
        with pytest.raises(ValueError, match="no usable context"):
            check_compaction_budget(CompactionConfig(compaction_output_budget=4_000), info)

    def test_warns_when_budget_exceeds_usable(self, capsys):
        info = ModelInfo(model_id="small", context_limit=10_000, max_output_tokens=1_000)
        check_compaction_budget(CompactionConfig(compaction_output_budget=5_000), info)
        assert "compaction_output_budget_exceeds_usable_window" in capsys.readouterr().out

    def test_healthy_and_unknown_window_pass_quietly(self, capsys):
        ok = ModelInfo(model_id="ok", context_limit=200_000, max_output_tokens=8_192)
        check_compaction_budget(CompactionConfig(), ok)
        check_compaction_budget(CompactionConfig(), ModelInfo(model_id="z", context_limit=0))
        assert "exceeds_usable" not in capsys.readouterr().out

    async def test_create_rejects_before_opening_store(self, tmp_path):
        cfg = _cfg(tmp_path, window=8_000, out=4_000, budget=4_000)
        with pytest.raises(ValueError, match="no usable context"):
            _ = await MnesisSession.create(model=MODEL, config=cfg)
        assert not Path(cfg.store.db_path).exists()  # noqa: ASYNC240

    async def test_load_rejects_and_closes_store(self, tmp_path):
        good = _cfg(tmp_path)
        async with MnesisSession.open(model=MODEL, config=good) as s:
            sid = s.id
        bad = _cfg(tmp_path, window=8_000, out=4_000, budget=4_000)
        with pytest.raises(ValueError, match="no usable context"):
            _ = await MnesisSession.load(sid, config=bad)


# ── L1 / L2 / L3: bounded condensation level 3, raw file IDs ──────────────────


class TestCondenseLevel3Bounded:
    def _many_id_nodes(self, n_ids: int, n_nodes: int = 3) -> list[SummaryNode]:
        per = n_ids // n_nodes
        nodes = []
        for k in range(n_nodes):
            ids = ", ".join(f"file_{k * per + i:016x}" for i in range(per))
            nodes.append(_node(f"node_{k}", f"Prose {k}.\n\n[LCM File IDs: {ids}]"))
        return nodes

    @pytest.mark.parametrize("encoding", ["claude_heuristic", "o200k_base"])
    def test_thousand_ids_fit_budget(self, encoding):
        info = ModelInfo(model_id="m", encoding=encoding)  # type: ignore[arg-type]
        est = TokenEstimator().for_model(info)
        budget = ContextBudget(
            model_context_limit=12_000, reserved_output_tokens=1_000, compaction_buffer=2_000
        )
        result = condense_level3_deterministic(self._many_id_nodes(3_000), est, budget)
        assert result.token_count <= budget.usable
        assert est.estimate(result.text) == result.token_count
        assert f"file_{2_999:016x}" in result.text  # most recently referenced kept

    def test_ids_dropped_only_when_footer_cannot_fit(self, estimator):
        budget = ContextBudget(
            model_context_limit=3_000, reserved_output_tokens=500, compaction_buffer=500
        )
        result = condense_level3_deterministic(self._many_id_nodes(900), estimator, budget)
        assert result.token_count <= budget.usable
        kept = extract_file_ids(result.text)
        assert 0 < len(kept) < 900
        assert f"file_{899:016x}" in kept
        assert f"file_{0:016x}" not in kept

    def test_no_prose_forced_in_when_footer_takes_the_allowance(self, estimator):
        budget = ContextBudget(
            model_context_limit=40_000, reserved_output_tokens=1_000, compaction_buffer=2_000
        )
        ids = ", ".join(f"file_{i:016x}" for i in range(150))
        node = _node("big", f"UNIQUE_PROSE_MARKER\n\n[LCM File IDs: {ids}]")
        result = condense_level3_deterministic([node], estimator, budget)
        assert "UNIQUE_PROSE_MARKER" not in result.text
        assert all(f"file_{i:016x}" in result.text for i in range(150))

    def test_oversized_chunk_is_truncated_to_the_prose_allowance(self, estimator):
        budget = ContextBudget(
            model_context_limit=40_000, reserved_output_tokens=1_000, compaction_buffer=2_000
        )
        result = condense_level3_deterministic([_node("long", "word " * 5000)], estimator, budget)
        assert 0 < estimator.estimate(result.text) <= 520  # 512 of prose budget + rounding

    def test_node_footers_are_not_duplicated_in_prose(self, estimator):
        budget = ContextBudget(
            model_context_limit=40_000, reserved_output_tokens=1_000, compaction_buffer=2_000
        )
        nodes = [
            _node("a", "A.\n\n[LCM File IDs: file_aa112233bb445566]"),
            _node("b", "B.\n\n[LCM File IDs: file_cc778899dd001122]"),
        ]
        result = condense_level3_deterministic(nodes, estimator, budget)
        assert result.text.count("file_aa112233bb445566") == 1
        assert result.text.count("file_cc778899dd001122") == 1

    def test_long_parent_list_uses_minimal_header(self, estimator):
        budget = ContextBudget(
            model_context_limit=40_000, reserved_output_tokens=1_000, compaction_buffer=2_000
        )
        nodes = [_node(f"msg_{'z' * 40}_{i}", f"c{i}") for i in range(60)]
        result = condense_level3_deterministic(nodes, estimator, budget)
        assert "Condensed from" not in result.text
        assert result.text.startswith(_CONDENSE_LEVEL3_MINIMAL_HEADER.strip())
        assert len(result.parent_node_ids) == 60

    def test_short_parent_list_keeps_full_header(self, estimator):
        budget = ContextBudget(
            model_context_limit=40_000, reserved_output_tokens=1_000, compaction_buffer=2_000
        )
        result = condense_level3_deterministic(
            [_node("n1", "one"), _node("n2", "two")], estimator, budget
        )
        assert "[Condensed from: n1, n2]" in result.text

    @pytest.mark.parametrize("usable", [1, 3, 6, 10, 20, 40, 80, 200])
    def test_tiny_budgets_are_bounded_or_lossless(self, estimator, usable):
        tiny = ContextBudget(
            model_context_limit=usable + 20, reserved_output_tokens=10, compaction_buffer=10
        )
        nodes = [
            _node("a", "alpha " * 100 + "\n\n[LCM File IDs: file_aa112233bb445566]"),
            _node("b", "beta " * 100),
        ]
        result = condense_level3_deterministic(nodes, estimator, tiny)
        if usable >= estimator.estimate(_CONDENSE_LEVEL3_MINIMAL_HEADER):
            assert result.token_count <= usable
        else:
            assert "file_aa112233bb445566" in result.text

    def test_joiner_overhead_is_shed(self):
        class JoinerEstimator:
            def estimate(self, text: str, model: ModelInfo | None = None) -> int:
                return len(text) // 4 + 5 * text.count("---")

        est = JoinerEstimator()
        nodes = [_node(f"n{i}", "w" * 100) for i in range(8)]
        probe = ContextBudget(
            model_context_limit=100_000, reserved_output_tokens=0, compaction_buffer=0
        )
        full = condense_level3_deterministic(nodes, est, probe)  # type: ignore[arg-type]
        assert full.text.count("---") == 7
        tight = ContextBudget(
            model_context_limit=full.token_count - 5,
            reserved_output_tokens=0,
            compaction_buffer=0,
        )
        shed = condense_level3_deterministic(nodes, est, tight)  # type: ignore[arg-type]
        assert shed.token_count <= tight.usable
        assert shed.text.count("---") < 7

    def test_truncate_to_tokens(self, estimator):
        assert _truncate_to_tokens("abc", 0, estimator) == ""
        assert _truncate_to_tokens("abcd", 5, estimator) == "abcd"
        cut = _truncate_to_tokens("x" * 400, 10, estimator)
        assert estimator.estimate(cut) <= 10 and len(cut) == 43


class TestFileIdExtractionIsRaw:
    def _msg(self, parts: list, mid: str = "raw_1") -> MessageWithParts:
        return MessageWithParts(
            message=make_message("s", role="assistant", msg_id=mid), parts=parts
        )

    def test_id_beyond_display_caps_is_found(self):
        far = "file_" + "ab" * 8
        text_msg = self._msg([TextPart(text="p" * 150_000 + f" {far}")], "raw_t")
        tool_msg = self._msg(
            [ToolPart(tool_name="r", tool_call_id="c1", output="o" * 5_000 + " file_" + "cd" * 8)],
            "raw_o",
        )
        assert extract_file_ids_from_messages([text_msg]) == [far]
        assert extract_file_ids_from_messages([tool_msg]) == ["file_" + "cd" * 8]

    def test_id_in_tool_input_and_error_is_found(self):
        msg = self._msg(
            [
                ToolPart(
                    tool_name="open",
                    tool_call_id="c2",
                    input={"file_id": "file_" + "12" * 8},
                    error_message="missing file_" + "34" * 8,
                )
            ]
        )
        assert extract_file_ids_from_messages([msg]) == ["file_" + "12" * 8, "file_" + "34" * 8]

    def test_id_only_in_pruned_tool_output_is_found(self):
        pruned = ToolPart(
            tool_name="read",
            tool_call_id="c3",
            output="x" * 3000 + " file_" + "56" * 8,
            status=ToolStatus(state="completed", compacted_at=1234),
        )
        assert extract_file_ids_from_messages([self._msg([pruned])]) == ["file_" + "56" * 8]

    def test_most_recent_is_by_last_occurrence_within_a_message(self):
        a, b = "file_" + "aa" * 8, "file_" + "bb" * 8
        msg = self._msg([TextPart(text=f"{a} {b} {a}")])
        assert most_recent_file_ids([msg]) == [a, b]  # a's last use is after b's
        older = self._msg([TextPart(text=b)], "raw_old")
        assert most_recent_file_ids([older, msg]) == [a, b]

    def test_most_recent_from_nodes(self):
        a, b = "file_" + "aa" * 8, "file_" + "bb" * 8
        nodes = [_node("n1", a), _node("n2", f"{b} {a}")]
        assert most_recent_file_ids_from_nodes(nodes) == [a, b]

    def test_strip_footer(self):
        assert strip_file_ids_footer("body\n\n[LCM File IDs: file_aa112233bb445566]") == "body"
        assert strip_file_ids_footer("no footer") == "no footer"

    async def test_pruned_output_id_reaches_footer_and_original_stays_in_store(self, tmp_path):
        """Pruning tombstones the output in context, but its ID is still preserved."""
        fid = "file_" + "7a" * 8
        cfg = _cfg(
            tmp_path,
            window=200_000,
            out=8_192,
            budget=20_000,
            auto=False,
            prune_protect_tokens=100,
            prune_minimum_tokens=10,
        )
        async with MnesisSession.open(model=MODEL, config=cfg) as s:
            _ = await s.record(
                "turn 0 start",
                [
                    ToolPart(
                        tool_name="read",
                        tool_call_id="c1",
                        output=f"{fid} " + "payload " * 2500,
                        status=ToolStatus(state="completed"),
                    ),
                    TextPart(text="done " * 50),
                ],
            )
            for i in range(1, 8):
                _ = await s.record(_turn(i, 40), _turn(i, 40))
            result = await s.compact()
            msgs = await s._store.get_messages_with_parts(s.id)
            tool = next(p for m in msgs for p in m.parts if isinstance(p, ToolPart))
            nodes = await s._dag_store.get_active_nodes(s.id)
        assert result.pruned_tool_outputs >= 1
        assert tool.compacted_at is not None and fid in (tool.output or "")
        assert fid in nodes[0].content


# ── T1: stalled compaction ────────────────────────────────────────────────────


class TestStalledCompaction:
    async def _stalled_session(self, tmp_path, monkeypatch, calls):
        """Open a session whose protected tail alone exceeds the soft threshold."""
        orig = engine_mod._make_llm_call

        def mk(model: str):
            inner = orig(model)

            async def _call(**kw):
                calls.append(1)
                return await inner(**kw)

            return _call

        monkeypatch.setattr(engine_mod, "_make_llm_call", mk)
        s = await MnesisSession.create(model=MODEL, config=_cfg(tmp_path, auto=False))
        # usable 9_000, soft 5_400. Turn 0 is summarisable; turns 1-2 (~6K tokens each) are not.
        _ = await s.record(_turn(0, 400), "ok")
        _ = await s.record(_turn(1, 1500), "ok")
        _ = await s.record(_turn(2, 1500), "ok")
        s._config.compaction.auto = True
        return s

    async def test_oversized_tail_stops_retriggering(self, tmp_path, monkeypatch):
        calls: list[int] = []
        s = await self._stalled_session(tmp_path, monkeypatch, calls)
        try:
            engine = s._compaction_engine
            triggers: list[int] = []
            s.subscribe(MnesisEvent.COMPACTION_TRIGGERED, lambda e, p: triggers.append(1))
            first = await s.compact()
            assert first.level_used == 1
            assert engine._stalled_until is not None
            llm_calls = len(calls)
            for i in range(3, 6):
                _ = await s.record(f"turn {i} x", "ok")
                _ = await engine.wait_for_pending()
            # Over the hard limit, send() must not block on a futile run either.
            result = await s.send("turn 9 y")
            assert result.compaction_result is None
            assert await s._measure_context(s.id) >= engine._usable_tokens(s._model_info)
        finally:
            await s.close()
        assert not triggers
        assert len(calls) == llm_calls

    async def test_rearms_after_growth_and_manual_compact_always_runs(self, tmp_path, monkeypatch):
        calls: list[int] = []
        s = await self._stalled_session(tmp_path, monkeypatch, calls)
        try:
            engine = s._compaction_engine
            info = s._model_info
            _ = await s.compact()
            rearm = engine._stalled_until
            assert rearm is not None
            assert engine.check_and_trigger(s.id, rearm - 1, info) is False
            assert engine.check_and_trigger(s.id, rearm, info) is True
            assert engine._stalled_until is None
            _ = await engine.wait_for_pending()
            before = len(calls)
            _ = await s.compact()  # explicit request is never gated
        finally:
            await s.close()
        assert len(calls) >= before

    async def test_progress_below_soft_clears_stall(self, tmp_path):
        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path)) as s:
            engine = s._compaction_engine
            engine._stalled_until = 10**9
            for i in range(12):
                _ = await s.record(_turn(i), _turn(i))
            engine._stalled_until = None  # as left by a healthy run
            _ = await s.compact()
            assert engine._stalled_until is None

    async def test_bare_engine_without_model_info_never_stalls(
        self, store, dag_store, event_bus, estimator
    ):
        engine = engine_mod.CompactionEngine(
            store, dag_store, estimator, event_bus, MnesisConfig(), session_model=MODEL
        )
        engine._note_run_outcome("s", tokens_before=10, tokens_after=10**9, more_to_compact=False)
        assert engine._stalled_until is None
        engine._model_info = ModelInfo(model_id="z", context_limit=0)
        engine._note_run_outcome("s", tokens_before=10, tokens_after=10**9, more_to_compact=False)
        assert engine._stalled_until is None

    async def test_progress_with_more_to_compact_is_not_a_stall(
        self, store, dag_store, event_bus, estimator
    ):
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
        assert engine._stalled_until is None
        engine._note_run_outcome("s", tokens_before=9000, tokens_after=6000, more_to_compact=False)
        assert engine._stalled_until == 6000 + (9000 - 5400)
        engine._note_run_outcome("s", tokens_before=9000, tokens_after=5000, more_to_compact=False)
        assert engine._stalled_until is None  # under soft: healthy


# ── T2: tool schemas count towards the thresholds ─────────────────────────────


class TestToolSchemaTokens:
    def _tools(self, n_chars: int) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "big",
                    "description": "d " * (n_chars // 2),
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def test_tools_push_context_over_soft_threshold(self, tmp_path):
        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path)) as s:
            engine = s._compaction_engine
            triggers: list[int] = []
            s.subscribe(MnesisEvent.COMPACTION_TRIGGERED, lambda e, p: triggers.append(1))
            for i in range(4):
                _ = await s.record(_turn(i, 40), _turn(i, 40))
            _ = await engine.wait_for_pending()
            assert not triggers  # history alone is under the soft threshold
            _ = await s.send("hello", tools=self._tools(20_000))
            _ = await engine.wait_for_pending()
            assert s._tool_schema_tokens > 3_000
        assert triggers, "tool schemas must count towards the soft threshold"

    async def test_no_tools_resets_schema_tokens(self, tmp_path):
        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path)) as s:
            _ = await s.send("a", tools=self._tools(200))
            assert s._tool_schema_tokens > 0
            _ = await s.send("b")
            assert s._tool_schema_tokens == 0


# ── D2: non-JSON-serialisable tool input ──────────────────────────────────────


class TestRecordNonJsonToolInput:
    async def test_datetime_uuid_path_and_objects_persist_and_reload(self, tmp_path):
        class Odd:
            def __str__(self) -> str:
                return "odd-object"

        when = datetime.datetime(2026, 1, 2, 3, 4, 5)
        uid = uuid.UUID(int=7)
        tool = ToolPart(
            tool_name="schedule",
            tool_call_id="c1",
            input={"at": when, "id": uid, "p": Path("/tmp/x"), "o": Odd()},
            output="ok",
            status=ToolStatus(state="completed"),
        )
        async with MnesisSession.open(model=MODEL, config=_cfg(tmp_path, window=200_000)) as s:
            result = await s.record("book it", [tool, TextPart(text="done")])
            msgs = await s.messages()
        stored = next(
            p
            for m in msgs
            if m.id == result.assistant_message_id
            for p in m.parts
            if isinstance(p, ToolPart)
        )
        assert stored.input == {
            "at": "2026-01-02T03:04:05",
            "id": str(uid),
            "p": "/tmp/x",
            "o": "odd-object",
        }
        assert result.tokens.output > 0


# ── N5: close() drains replaced pending runs ──────────────────────────────────


class TestCloseDrainsPending:
    async def test_close_awaits_run_scheduled_while_waiting(self, tmp_path):
        s = await MnesisSession.create(model=MODEL, config=_cfg(tmp_path))
        engine = s._compaction_engine
        done: list[str] = []

        async def second() -> None:
            await asyncio.sleep(0.05)
            done.append("second")

        async def first() -> None:
            await asyncio.sleep(0.01)
            engine._pending_task = asyncio.create_task(second())  # type: ignore[assignment]
            done.append("first")

        engine._pending_task = asyncio.create_task(first())  # type: ignore[assignment]
        await s.close()
        assert done == ["first", "second"]
        assert not engine.has_pending
