"""Tests for CompactionEngine and three-level escalation."""

from __future__ import annotations

import pytest

from mnesis.compaction.engine import CompactionEngine
from mnesis.compaction.levels import (
    SummaryCandidate,
    level1_summarise,
    level2_summarise,
    level3_deterministic,
)
from mnesis.models.config import ModelInfo
from mnesis.models.message import ContextBudget, MessageWithParts, TextPart, TokenUsage
from tests.conftest import make_message, make_raw_part


def _make_messages_with_parts(session_id: str, count: int) -> list[MessageWithParts]:
    """Create a list of MessageWithParts for testing compaction levels."""
    result = []
    roles = ["user", "assistant"]
    for i in range(count):
        msg = make_message(session_id, role=roles[i % 2], msg_id=f"msg_cmp_{i:03d}")
        parts = [TextPart(text=f"Message {i} content: " + "x" * 200)]
        result.append(MessageWithParts(message=msg, parts=parts))
    return result


@pytest.fixture
def budget():
    return ContextBudget(
        model_context_limit=50_000,
        reserved_output_tokens=4_000,
        compaction_buffer=10_000,
    )


class TestCompactionLevels:
    async def test_level3_deterministic_always_fits(self, estimator, budget):
        """Level 3 always produces a result within budget."""
        messages = _make_messages_with_parts("sess_l3", 20)
        candidate = level3_deterministic(messages, budget, estimator)
        assert isinstance(candidate, SummaryCandidate)
        assert candidate.token_count <= budget.usable
        assert candidate.compaction_level == 3

    async def test_level3_empty_messages(self, estimator, budget):
        """Level 3 handles empty message list gracefully."""
        candidate = level3_deterministic([], budget, estimator)
        assert candidate.text != ""
        assert candidate.compaction_level == 3

    async def test_level1_fails_when_llm_errors(self, estimator, budget):
        """Level 1 returns None when the LLM call fails."""

        async def failing_llm(**kwargs):
            raise RuntimeError("LLM unavailable")

        messages = _make_messages_with_parts("sess_l1", 6)
        result = await level1_summarise(messages, "test-model", budget, estimator, failing_llm)
        assert result is None

    async def test_level1_returns_none_if_too_few_messages(self, estimator, budget):
        """Level 1 returns None when there are fewer than 4 messages (nothing to summarize)."""
        messages = _make_messages_with_parts("sess_l1_few", 2)
        # With only 2 messages, _messages_to_summarise returns empty
        result = await level1_summarise(
            messages,
            "test-model",
            budget,
            estimator,
            lambda **k: (_ for _ in ()).throw(RuntimeError("should not call")),
        )
        assert result is None

    async def test_level2_fails_when_llm_errors(self, estimator, budget):
        """Level 2 returns None when the LLM call fails."""

        async def failing_llm(**kwargs):
            raise RuntimeError("LLM unavailable")

        messages = _make_messages_with_parts("sess_l2", 6)
        result = await level2_summarise(messages, "test-model", budget, estimator, failing_llm)
        assert result is None

    async def test_level1_success(self, estimator, budget):
        """Level 1 returns a candidate when LLM succeeds."""
        summary_text = "## Goal\nTest summarization.\n\n## Completed Work\nDone."

        async def mock_llm(**kwargs):
            return summary_text

        messages = _make_messages_with_parts("sess_l1_ok", 6)
        result = await level1_summarise(messages, "test-model", budget, estimator, mock_llm)
        assert result is not None
        assert result.compaction_level == 1
        assert result.text == summary_text

    async def test_level1_returns_none_if_result_too_large(self, estimator):
        """Level 1 returns None when the summary is too large for the budget."""
        tiny_budget = ContextBudget(
            model_context_limit=1_000,
            reserved_output_tokens=100,
            compaction_buffer=100,
        )

        async def huge_llm(**kwargs):
            return "x" * 10_000  # Way too large

        messages = _make_messages_with_parts("sess_l1_big", 6)
        result = await level1_summarise(messages, "test-model", tiny_budget, estimator, huge_llm)
        assert result is None


class TestIsOverflow:
    def test_overflow_false_when_auto_disabled(
        self, estimator, event_bus, store, dag_store, config
    ):
        """is_overflow returns False when auto compaction is disabled."""
        from mnesis.models.config import CompactionConfig

        cfg = config.model_copy(update={"compaction": CompactionConfig(auto=False)})
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, cfg, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo.from_model_string("anthropic/claude-opus-4-6")
        tokens = TokenUsage(total=190_000)
        assert engine.is_overflow(tokens, model) is False

    def test_overflow_false_for_unlimited_model(
        self, estimator, event_bus, store, dag_store, config
    ):
        """is_overflow returns False when context_limit is 0 (unlimited)."""
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo(model_id="local-model", context_limit=0, max_output_tokens=4096)
        tokens = TokenUsage(total=999_999)
        assert engine.is_overflow(tokens, model) is False

    def test_overflow_true_at_threshold(self, estimator, event_bus, store, dag_store, config):
        """is_overflow returns True when tokens reach the usable threshold."""
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo(
            model_id="test-model",
            context_limit=100_000,
            max_output_tokens=4_000,
        )
        # usable = 100_000 - 4_000 - 20_000 = 76_000
        tokens_ok = TokenUsage(total=75_999)
        tokens_overflow = TokenUsage(total=76_001)
        assert engine.is_overflow(tokens_ok, model) is False
        assert engine.is_overflow(tokens_overflow, model) is True


class TestCompactionEngine:
    async def test_run_compaction_never_raises(
        self, session_id, store, dag_store, estimator, event_bus, config
    ):
        """run_compaction catches all errors and returns a CompactionResult."""
        # Use a non-existent model — should fall through to level 3
        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            id_generator=lambda p: f"{p}_never_raises",
        )
        # Populate with some messages
        for i in range(4):
            msg = make_message(
                session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_nr_{i:03d}"
            )
            await store.append_message(msg)
            part = make_raw_part(msg.id, session_id, part_id=f"part_nr_{i:03d}")
            await store.append_part(part)

        # Override model to force level 3 via unavailable LLM
        result = await engine.run_compaction(session_id, model_override="nonexistent-model-xyz")
        # Should return a result without crashing
        assert result.session_id == session_id

    async def test_compaction_event_published(
        self, session_id, store, dag_store, estimator, event_bus, config
    ):
        """COMPACTION_COMPLETED event is published after run_compaction."""
        from mnesis.events.bus import MnesisEvent

        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_event_test"
        )
        for i in range(4):
            msg = make_message(
                session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_ev_{i:03d}"
            )
            await store.append_message(msg)
            part = make_raw_part(msg.id, session_id, part_id=f"part_ev_{i:03d}")
            await store.append_part(part)

        await engine.run_compaction(session_id, model_override="nonexistent-model-xyz")

        published_events = [e for e, _ in event_bus.collected]
        assert (
            MnesisEvent.COMPACTION_COMPLETED in published_events
            or MnesisEvent.COMPACTION_FAILED in published_events
        )
