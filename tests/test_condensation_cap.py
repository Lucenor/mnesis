"""F1: condensation input is capped against the compaction model's window."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
import structlog.testing

from mnesis.compaction.engine import CompactionEngine
from mnesis.compaction.file_ids import extract_file_ids
from mnesis.compaction.levels import condense_level1, condense_level2
from mnesis.models.config import CompactionConfig, MnesisConfig, ModelInfo, StoreConfig
from mnesis.models.message import ContextBudget
from mnesis.models.summary import SummaryNode
from mnesis.session import make_id
from mnesis.tokens.estimator import TokenEstimator
from tests.conftest import make_message, make_raw_part

WINDOW = 8_192
MAX_OUT = 1_024


@pytest.fixture
def est() -> TokenEstimator:
    e = TokenEstimator()
    e._force_heuristic = True
    return e


@pytest.fixture
def budget() -> ContextBudget:
    return ContextBudget(
        model_context_limit=200_000, reserved_output_tokens=4_000, compaction_buffer=1_000
    )


def _big_node(i: int, est: TokenEstimator, tokens: int = 1_800) -> SummaryNode:
    content = (
        f"## Goal\nnode {i} "
        + ("detail " * (tokens * 4 // 7))
        + f"\n\n[LCM File IDs: file_{i:016x}]"
    )
    return SummaryNode(
        id=f"node_{i}",
        session_id="s",
        kind="leaf",
        span_start_message_id=f"a{i}",
        span_end_message_id=f"b{i}",
        content=content,
        token_count=est.estimate(content),
    )


class Recorder:
    def __init__(self, est: TokenEstimator) -> None:
        self.est = est
        self.calls: list[tuple[str, int]] = []

    async def __call__(self, **kwargs: Any) -> str:
        prompt = kwargs["messages"][0]["content"]
        self.calls.append((prompt, kwargs["max_tokens"]))
        return "## Goal\nmerged\n"

    def assert_within_window(self) -> None:
        assert self.calls
        for prompt, max_tokens in self.calls:
            assert self.est.estimate(prompt) + max_tokens <= WINDOW


class TestLevelFunctions:
    async def test_level1_condenses_oldest_subset_that_fits(self, est, budget):
        nodes = [_big_node(i, est) for i in range(6)]  # ~10.8K tokens > window
        assert sum(n.token_count for n in nodes) > WINDOW
        rec = Recorder(est)
        cand = await condense_level1(
            nodes,
            "m",
            budget,
            est,
            rec,
            model_max_output_tokens=MAX_OUT,
            model_context_limit=WINDOW,
        )
        assert cand is not None
        n_fit = len(cand.parent_node_ids)
        assert 2 <= n_fit < 6
        assert cand.parent_node_ids == [f"node_{i}" for i in range(n_fit)]  # oldest first
        rec.assert_within_window()
        # File ids of every condensed node are in the footer; none of the others.
        ids = set(extract_file_ids(cand.text))
        assert ids == {f"file_{i:016x}" for i in range(n_fit)}

    async def test_everything_fits_everything_is_condensed(self, est, budget):
        nodes = [_big_node(i, est, tokens=300) for i in range(4)]
        cand = await condense_level1(
            nodes,
            "m",
            budget,
            est,
            Recorder(est),
            model_max_output_tokens=MAX_OUT,
            model_context_limit=WINDOW,
        )
        assert cand is not None and cand.parent_node_ids == [n.id for n in nodes]

    async def test_too_few_fit_escalates_without_a_provider_call(self, est, budget):
        nodes = [_big_node(i, est, tokens=5_000) for i in range(3)]  # one alone ~ the window
        rec = Recorder(est)
        cand = await condense_level1(
            nodes,
            "m",
            budget,
            est,
            rec,
            model_max_output_tokens=MAX_OUT,
            model_context_limit=WINDOW,
        )
        assert cand is None
        assert rec.calls == []  # never sent an over-window request

    async def test_level2_caps_excerpt_input_too(self, est, budget):
        nodes = [_big_node(i, est, tokens=400) for i in range(40)]  # 40 x 800-char excerpts
        rec = Recorder(est)
        tiny_window = 1_500
        cand = await condense_level2(
            nodes,
            "m",
            budget,
            est,
            rec,
            model_max_output_tokens=256,
            model_context_limit=tiny_window,
        )
        assert cand is not None
        assert 2 <= len(cand.parent_node_ids) < 40
        assert cand.parent_node_ids == [n.id for n in nodes[: len(cand.parent_node_ids)]]
        prompt, max_tokens = rec.calls[0]
        assert est.estimate(prompt) + max_tokens <= tiny_window

    async def test_unknown_window_disables_the_cap(self, est, budget):
        nodes = [_big_node(i, est) for i in range(6)]
        cand = await condense_level1(nodes, "m", budget, est, Recorder(est), model_context_limit=0)
        assert cand is not None and cand.parent_node_ids == [n.id for n in nodes]

    async def test_level2_too_few_fit_escalates_without_a_provider_call(self, est, budget):
        nodes = [_big_node(i, est, tokens=400) for i in range(3)]
        rec = Recorder(est)
        cand = await condense_level2(
            nodes, "m", budget, est, rec, model_max_output_tokens=256, model_context_limit=300
        )
        assert cand is None and rec.calls == []

    async def test_empty_nodes_return_none(self, est, budget):
        rec = Recorder(est)
        assert await condense_level2([], "m", budget, est, rec) is None
        assert await condense_level1([], "m", budget, est, rec) is None
        assert rec.calls == []

    async def test_compaction_estimator_units_are_used(self, est, budget):
        """A denser compaction-model tokenizer makes the same nodes not fit."""

        class Dense(TokenEstimator):
            def estimate(self, text: str, model: Any = None) -> int:  # 3x the tokens
                return 3 * super().estimate(text, model)

        dense = Dense()
        dense._force_heuristic = True
        nodes = [_big_node(i, est, tokens=900) for i in range(4)]
        sparse_cand = await condense_level1(
            nodes,
            "m",
            budget,
            est,
            Recorder(est),
            model_max_output_tokens=MAX_OUT,
            model_context_limit=WINDOW,
        )
        dense_cand = await condense_level1(
            nodes,
            "m",
            budget,
            est,
            Recorder(est),
            model_max_output_tokens=MAX_OUT,
            model_context_limit=WINDOW,
            compaction_estimator=dense,
        )
        assert sparse_cand is not None and len(sparse_cand.parent_node_ids) == 4
        assert dense_cand is None or len(dense_cand.parent_node_ids) < 4


async def _engine_with_nodes(
    store, dag_store, est, event_bus, tmp_path, n_nodes, session_id, sizes=None, **compaction_kw
):
    cfg = MnesisConfig(
        store=StoreConfig(db_path=str(tmp_path / "cap.db")),
        compaction=CompactionConfig(
            compaction_output_budget=1_000, max_compaction_rounds=5, **compaction_kw
        ),
    )
    info = ModelInfo(
        model_id="anthropic/claude-haiku-4-5",
        provider_id="anthropic",
        context_limit=WINDOW,
        max_output_tokens=MAX_OUT,
    )
    engine = CompactionEngine(
        store,
        dag_store,
        est,
        event_bus,
        cfg,
        session_model="anthropic/claude-haiku-4-5",
        model_info=info,
    )
    # One raw message pair so the run has a protected tail, then n live leaf summaries.
    for i in range(4):
        msg = make_message(
            session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_cap_{i}"
        )
        await store.append_message(msg)
        await store.append_part(
            make_raw_part(
                msg.id,
                session_id,
                part_id=f"part_cap_{i}",
                content=json.dumps({"type": "text", "text": f"turn {i}"}),
            )
        )
    nodes = []
    for i in range(n_nodes):
        node = _big_node(i, est, sizes[i] if sizes else 1_800)
        node.session_id = session_id
        node.id = f"msg_node_{i}"
        node.span_start_message_id = f"msg_cap_{0}"
        node.span_end_message_id = f"msg_cap_{0}"
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))
        async with store._transaction() as conn:
            await conn.execute(
                "INSERT INTO context_items (session_id, item_type, item_id, position, created_at)"
                " VALUES (?, 'summary', ?, ?, '0')",
                (session_id, node.id, i - 10),  # before the raw messages
            )
        nodes.append(node)
    return engine, nodes


class TestEngineCondensesSubsets:
    async def test_over_window_node_set_is_condensed_in_subsets(
        self, session_id, store, dag_store, est, event_bus, tmp_path, monkeypatch
    ):
        engine, nodes = await _engine_with_nodes(
            store, dag_store, est, event_bus, tmp_path, 6, session_id
        )
        assert sum(n.token_count for n in nodes) > WINDOW
        rec = Recorder(est)
        monkeypatch.setattr("mnesis.compaction.engine._make_llm_call", lambda model, **kw: rec)

        result = await engine.run_compaction(session_id)

        assert result.level_used in (1, 2)
        rec.assert_within_window()  # no provider call over the compaction window
        assert len(rec.calls) >= 2  # needed more than one round for six big nodes
        active = await dag_store.get_active_nodes(session_id)
        # Everything condensed away is superseded exactly once; the live set shrank.
        assert len(active) < len(nodes)
        live_ids = {n.id for n in active}
        for n in nodes:
            if n.id not in live_ids:
                assert n.id not in {i for t, i in await store.get_context_items(session_id) if t}
        # Invariant 5: every file id survives somewhere in the live summaries.
        surviving = {fid for n in active for fid in extract_file_ids(n.content)}
        assert surviving >= {f"file_{i:016x}" for i in range(6)}

    async def test_uncondensed_nodes_stay_live_after_a_capped_round(
        self, session_id, store, dag_store, est, event_bus, tmp_path, monkeypatch
    ):
        engine, nodes = await _engine_with_nodes(
            store, dag_store, est, event_bus, tmp_path, 6, session_id
        )
        rec = Recorder(est)
        monkeypatch.setattr("mnesis.compaction.engine._make_llm_call", lambda model, **kw: rec)
        budget = engine._summary_budget()
        info = engine._model_info
        assert info is not None

        cond = await engine._run_condensation(
            nodes, "anthropic/claude-haiku-4-5", budget, rec, None, info
        )
        consumed = engine._consumed_nodes(nodes, cond)
        assert 2 <= len(consumed) < len(nodes)
        assert consumed == nodes[: len(consumed)]  # oldest first, contiguous
        assert set(cond.parent_node_ids) == {n.id for n in consumed}
        rec.assert_within_window()
        # Nothing was superseded by merely producing the candidate.
        assert {n.id for n in await dag_store.get_active_nodes(session_id)} == {n.id for n in nodes}


class _LabelRecorder(Recorder):
    """Replies ``merged-<n>`` so later prompts show which merge they contain."""

    async def __call__(self, **kwargs: Any) -> str:
        _ = await super().__call__(**kwargs)
        return f"## Goal\nmerged-{len(self.calls)}\n"


class TestContextOrder:
    """F1 review: subsets are the oldest *in the context*, not by creation time."""

    async def test_rounds_keep_chronological_order(
        self, session_id, store, dag_store, est, event_bus, tmp_path, monkeypatch
    ):
        engine, _nodes = await _engine_with_nodes(
            store,
            dag_store,
            est,
            event_bus,
            tmp_path,
            6,
            session_id,
            sizes=[1800, 1800, 1800, 2500, 2500, 4800],
        )
        rec = _LabelRecorder(est)
        monkeypatch.setattr("mnesis.compaction.engine._make_llm_call", lambda model, **kw: rec)

        _ = await engine.run_compaction(session_id)

        assert len(rec.calls) >= 2
        rows = await dag_store._query_summary_nodes(
            session_id, superseded=True
        ) + await dag_store._query_summary_nodes(session_id, superseded=False)
        parents_of = {r["id"]: json.loads(r["parent_node_ids"]) for r in rows}

        def leaves(node_id: str) -> set[int]:
            if not parents_of[node_id]:
                return {int(node_id.rsplit("_", 1)[1])}
            return set().union(*(leaves(p) for p in parents_of[node_id]))

        label_leaves: dict[str, set[int]] = {f"node {i} ": {i} for i in range(6)}
        for r in rows:
            m = re.search(r"merged-(\d+)", r["content"])
            if m:
                label_leaves[f"merged-{m.group(1)}"] = leaves(r["id"])

        # 1. Every round's prompt lists its inputs oldest first.
        for prompt, _max in rec.calls:
            found = re.findall(r"node \d+ |merged-\d+", prompt)
            firsts = [min(label_leaves[label]) for label in found]
            assert firsts == sorted(firsts), found
        # 2. Each condensed node merges a contiguous run of the original leaves.
        for r in rows:
            if parents_of[r["id"]]:
                covered = sorted(leaves(r["id"]))
                assert covered == list(range(covered[0], covered[-1] + 1)), covered
        # 3. The final context lists the summaries in chronological order.
        items = [i for t, i in await store.get_context_items(session_id) if t == "summary"]
        final = [min(leaves(i)) for i in items]
        assert final == sorted(final), final


class TestFitCountsFraming:
    def test_joiners_and_wrapper_count_against_the_cap(self, est):
        from mnesis.compaction.levels import _fit_condensation_nodes

        nodes = [_big_node(i, est, 100) for i in range(3)]
        texts = ["x" * 400] * 3  # 100 heuristic tokens each: 300 in all
        # 75% of 400 is 300: the bare texts would just fit, the framing tips it over.
        fitted = _fit_condensation_nodes(
            nodes, texts, est, 400, reserved_tokens=0, level=1, separator="\n\n---\n\n"
        )
        assert [n.id for n in fitted] == ["node_0", "node_1"]


class _L1Probe:
    """llm_call stub: level 1 condensation prompts fail as configured; level 2 succeeds."""

    def __init__(self, l1: str) -> None:
        self.l1 = l1  # "truncate" | "error" | "empty" | "ok"
        self.l1_calls = 0
        self.l2_calls = 0

    async def __call__(self, **kwargs: Any) -> str:
        from mnesis.compaction.engine import CompactionTruncatedError

        prompt = kwargs["messages"][0]["content"]
        if "condensing multiple context summaries" in prompt:
            self.l1_calls += 1
            if self.l1 == "truncate":
                raise CompactionTruncatedError("hit the output limit")
            if self.l1 == "error":
                raise RuntimeError("provider hiccup")
            if self.l1 == "empty":
                return "  "
            return "## Goal\nmerged\n"
        self.l2_calls += 1
        return "GOAL: merged"


class TestCondenseLevel1Skip:
    async def _setup(self, *args, **compaction_kw):
        session_id, store, dag_store, est, event_bus, tmp_path = args
        engine, nodes = await _engine_with_nodes(
            store,
            dag_store,
            est,
            event_bus,
            tmp_path,
            3,
            session_id,
            sizes=[300] * 3,
            **compaction_kw,
        )
        return engine, nodes, engine._summary_budget(), engine._model_info

    async def _run(self, engine, nodes, budget, info, probe):
        return await engine._run_condensation(
            nodes, "anthropic/claude-haiku-4-5", budget, probe, None, info
        )

    async def test_truncation_escalates_then_later_runs_skip_level1(
        self, session_id, store, dag_store, est, event_bus, tmp_path
    ):
        engine, nodes, budget, info = await self._setup(
            session_id, store, dag_store, est, event_bus, tmp_path
        )
        probe = _L1Probe("truncate")
        with structlog.testing.capture_logs() as logs:
            first = await self._run(engine, nodes, budget, info, probe)
            _ = await self._run(engine, nodes, budget, info, probe)
        events = [e for e in logs if e["event"] == "condense_level1_disabled_after_truncation"]
        assert len(events) == 1 and events[0]["model"] == "anthropic/claude-haiku-4-5"
        assert first.compaction_level == 2  # this run escalated
        assert engine._condense_l1_truncated
        assert (probe.l1_calls, probe.l2_calls) == (1, 2)  # the second run made no level 1 request

    async def test_config_flag_skips_level1_from_the_start(
        self, session_id, store, dag_store, est, event_bus, tmp_path
    ):
        engine, nodes, budget, info = await self._setup(
            session_id, store, dag_store, est, event_bus, tmp_path, condense_skip_level1=True
        )
        probe = _L1Probe("ok")
        cand = await self._run(engine, nodes, budget, info, probe)
        assert cand.compaction_level == 2
        assert (probe.l1_calls, probe.l2_calls) == (0, 1)

    @pytest.mark.parametrize("failure", ["error", "empty"])
    async def test_other_failures_do_not_disable_level1(
        self, failure, session_id, store, dag_store, est, event_bus, tmp_path
    ):
        engine, nodes, budget, info = await self._setup(
            session_id, store, dag_store, est, event_bus, tmp_path
        )
        probe = _L1Probe(failure)
        _ = await self._run(engine, nodes, budget, info, probe)
        _ = await self._run(engine, nodes, budget, info, probe)
        assert probe.l1_calls == 2
        assert not engine._condense_l1_truncated

    async def test_level1_kept_when_level2_is_disabled(
        self, session_id, store, dag_store, est, event_bus, tmp_path
    ):
        engine, nodes, budget, info = await self._setup(
            session_id,
            store,
            dag_store,
            est,
            event_bus,
            tmp_path,
            condense_skip_level1=True,
            level2_enabled=False,
        )
        probe = _L1Probe("ok")
        cand = await self._run(engine, nodes, budget, info, probe)
        assert cand.compaction_level == 1
        assert probe.l1_calls == 1
