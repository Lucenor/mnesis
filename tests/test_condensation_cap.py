"""F1: condensation input is capped against the compaction model's window."""

from __future__ import annotations

import json
from typing import Any

import pytest

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


async def _engine_with_nodes(store, dag_store, est, event_bus, tmp_path, n_nodes, session_id):
    cfg = MnesisConfig(
        store=StoreConfig(db_path=str(tmp_path / "cap.db")),
        compaction=CompactionConfig(compaction_output_budget=1_000, max_compaction_rounds=5),
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
        node = _big_node(i, est)
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
