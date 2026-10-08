"""Empty or truncated compaction completions must never become the summary."""

from __future__ import annotations

import sys
import types

import pytest

import mnesis.compaction.engine as engine_mod
from mnesis.compaction.levels import condense_level1, condense_level2
from mnesis.models.config import MnesisConfig
from mnesis.models.message import ContextBudget
from mnesis.models.summary import SummaryNode
from tests.conftest import make_message, make_raw_part

_GOOD = "## Goal\nreal summary\n\n## Completed Work\n- done\n"
_MODEL = "anthropic/claude-opus-4-6"


async def _engine(
    session_id, store, dag_store, estimator, event_bus, config
) -> engine_mod.CompactionEngine:
    for i in range(8):
        msg = make_message(
            session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_empty_{i}"
        )
        await store.append_message(msg)
        await store.append_part(make_raw_part(msg.id, session_id, part_id=f"part_empty_{i}"))
    return engine_mod.CompactionEngine(
        store, dag_store, estimator, event_bus, config, session_model="anthropic/claude-haiku-4-5"
    )


def _scripted(monkeypatch, replies: list[str]) -> list[int]:
    calls: list[int] = []

    async def llm(**kwargs: object) -> str:
        calls.append(1)
        return replies[min(len(calls), len(replies)) - 1]

    monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model: llm)
    return calls


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


class TestEmptyCompletion:
    @pytest.mark.parametrize("empty", ["", "   \n\t "])
    async def test_level1_empty_escalates_to_level2(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch, empty
    ):
        _scripted(monkeypatch, [empty, _GOOD])
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, config)
        result = await engine.run_compaction(session_id)

        assert result.level_used == 2
        assert result.summary_token_count > 0
        msgs = await store.get_messages(session_id)
        summaries = [m for m in msgs if m.is_summary]
        assert summaries
        parts = await store.get_parts(summaries[-1].id)
        assert "real summary" in parts[0].content

    async def test_level1_and_level2_empty_fall_to_level3(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        calls = _scripted(monkeypatch, [""])
        engine = await _engine(session_id, store, dag_store, estimator, event_bus, config)
        result = await engine.run_compaction(session_id)

        assert len(calls) == 2  # an empty reply is not retried at the same level
        assert result.level_used == 3
        assert result.summary_token_count > 0

    async def test_condensation_empty_falls_through_without_empty_node(self):
        budget = ContextBudget(
            model_context_limit=100_000, reserved_output_tokens=0, compaction_buffer=0
        )
        from mnesis.tokens.estimator import TokenEstimator

        est = TokenEstimator()
        nodes = [_node("a", "alpha " * 20), _node("b", "beta " * 20)]

        async def empty(**kwargs: object) -> str:
            return "  "

        assert await condense_level1(nodes, _MODEL, budget, est, empty) is None
        assert await condense_level2(nodes, _MODEL, budget, est, empty) is None

    async def test_engine_condensation_escalates_to_nonempty_level3(
        self, store, dag_store, event_bus, estimator
    ):
        engine = engine_mod.CompactionEngine(
            store, dag_store, estimator, event_bus, MnesisConfig(), session_model=_MODEL
        )
        budget = ContextBudget(
            model_context_limit=100_000, reserved_output_tokens=0, compaction_buffer=0
        )

        async def empty(**kwargs: object) -> str:
            return ""

        nodes = [_node("a", "alpha " * 20), _node("b", "beta " * 20)]
        cond = await engine._run_condensation(nodes, _MODEL, budget, empty, None)
        assert cond.text.strip()
        assert cond.compaction_level == 3


class TestTruncatedCompletion:
    @staticmethod
    def _fake_litellm(monkeypatch, finish_reason: str) -> None:
        async def fake_acompletion(**kwargs: object) -> object:
            msg = types.SimpleNamespace(content="partial summary cut o")
            choice = types.SimpleNamespace(message=msg, finish_reason=finish_reason)
            return types.SimpleNamespace(choices=[choice])

        fake = types.ModuleType("litellm")
        fake.acompletion = fake_acompletion  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "litellm", fake)
        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)

    async def test_length_finish_reason_raises(self, monkeypatch):
        self._fake_litellm(monkeypatch, "length")
        with pytest.raises(engine_mod.CompactionTruncatedError):
            await engine_mod._make_llm_call("m")(
                messages=[{"role": "user", "content": "x"}], max_tokens=10
            )

    async def test_stop_finish_reason_accepted(self, monkeypatch):
        self._fake_litellm(monkeypatch, "stop")
        out = await engine_mod._make_llm_call("m")(
            messages=[{"role": "user", "content": "x"}], max_tokens=10
        )
        assert out == "partial summary cut o"

    async def test_truncated_level1_is_not_accepted(self, monkeypatch):
        """A length-truncated L1 reply fails the level (-> escalation), not retried."""
        from mnesis.compaction.levels import level1_summarise
        from mnesis.retry import is_retryable
        from tests.test_compaction import _make_messages_with_parts

        assert not is_retryable(engine_mod.CompactionTruncatedError("x"))

        self._fake_litellm(monkeypatch, "length")
        llm = engine_mod._make_llm_call("m")
        from mnesis.tokens.estimator import TokenEstimator

        budget = ContextBudget(
            model_context_limit=100_000, reserved_output_tokens=0, compaction_buffer=0
        )
        msgs = _make_messages_with_parts("sess_trunc", 6)
        out = await level1_summarise(msgs, "m", budget, TokenEstimator(), llm)
        assert out is None
