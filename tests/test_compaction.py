"""Tests for CompactionEngine and three-level escalation."""

from __future__ import annotations

import asyncio

import pytest

from mnesis.compaction.engine import CompactionEngine
from mnesis.compaction.levels import (
    SummaryCandidate,
    level1_summarise,
    level2_summarise,
    level3_deterministic,
)
from mnesis.models.config import ModelInfo
from mnesis.models.message import ContextBudget, MessageWithParts, TextPart, TokenUsage, ToolPart
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

    async def test_level2_returns_none_if_result_too_large(self, estimator):
        """Level 2 returns None when the summary is too large for the budget."""
        tiny_budget = ContextBudget(
            model_context_limit=1_000,
            reserved_output_tokens=100,
            compaction_buffer=100,
        )

        async def huge_llm(**kwargs):
            return "x" * 10_000  # Way too large

        messages = _make_messages_with_parts("sess_l2_big", 6)
        result = await level2_summarise(messages, "test-model", tiny_budget, estimator, huge_llm)
        assert result is None

    async def test_level2_success(self, estimator, budget):
        """Level 2 returns a SummaryCandidate when LLM succeeds and result fits budget."""
        summary_text = "GOAL: Test.\nCONSTRAINTS: none\nFILES: -\nNEXT: done\nCONTEXT: ok"

        async def mock_llm(**kwargs):
            return summary_text

        messages = _make_messages_with_parts("sess_l2_ok", 6)
        result = await level2_summarise(messages, "test-model", budget, estimator, mock_llm)
        assert result is not None
        assert result.compaction_level == 2
        assert result.text == summary_text

    def test_level3_breaks_when_budget_exhausted(self, estimator):
        """Level 3 breaks early when messages don't fit in the token budget."""
        # usable = 100 - 10 - 10 = 80, target = 68
        # Header (~14 tokens) + first message line (~57 tokens) > 68, so break fires
        tight_budget = ContextBudget(
            model_context_limit=100,
            reserved_output_tokens=10,
            compaction_buffer=10,
        )
        messages = _make_messages_with_parts("sess_l3_tight", 10)
        candidate = level3_deterministic(messages, tight_budget, estimator)
        assert isinstance(candidate, SummaryCandidate)
        assert candidate.compaction_level == 3
        assert candidate.token_count <= tight_budget.usable
        # Budget is so tight no messages fit — summary is only the truncation header
        assert "CONTEXT TRUNCATED" in candidate.text

    def test_extract_text_includes_tool_output(self, estimator, budget):
        """_extract_text includes ToolPart output when compacted_at is None."""
        tool_part = ToolPart(
            tool_name="read_file",
            tool_call_id="call_tool_001",
            output="important file content",
        )
        assistant_msg = MessageWithParts(
            message=make_message("sess_tool", role="assistant", msg_id="msg_tool_001"),
            parts=[tool_part],
        )
        user_msg = MessageWithParts(
            message=make_message("sess_tool", role="user", msg_id="msg_tool_000"),
            parts=[TextPart(text="read that file")],
        )
        candidate = level3_deterministic([user_msg, assistant_msg], budget, estimator)
        assert "important file content" in candidate.text


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

    def test_overflow_true_at_soft_threshold(self, estimator, event_bus, store, dag_store, config):
        """is_overflow (soft threshold) fires at soft_threshold_fraction * usable."""
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo(
            model_id="test-model",
            context_limit=100_000,
            max_output_tokens=4_000,
        )
        # usable = 100_000 - 4_000 - 20_000 = 76_000
        # soft_limit = 76_000 * 0.6 = 45_600
        tokens_below_soft = TokenUsage(total=45_599)
        tokens_above_soft = TokenUsage(total=45_601)
        assert engine.is_overflow(tokens_below_soft, model) is False
        assert engine.is_overflow(tokens_above_soft, model) is True

    def test_hard_overflow_at_usable_threshold(
        self, estimator, event_bus, store, dag_store, config
    ):
        """is_hard_overflow fires at the full usable budget (hard threshold)."""
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
        assert engine.is_hard_overflow(tokens_ok, model) is False
        assert engine.is_hard_overflow(tokens_overflow, model) is True


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
            session_model="nonexistent-model-xyz",
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
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            id_generator=lambda p: f"{p}_event_test",
            session_model="nonexistent-model-xyz",
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

    async def test_run_compaction_uses_level1_in_mock_mode(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        """run_compaction reaches level 1 (not level 3) when MNESIS_MOCK_LLM=1.

        Regression test for the bug where _make_llm_call always called litellm,
        causing AuthenticationError in CI/test environments before falling back to
        level 3.  With the fix, MNESIS_MOCK_LLM=1 short-circuits to a mock summary
        and the engine reports level_used=1.
        """
        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            session_model="anthropic/claude-haiku-4-5",
        )
        # level1_summarise requires >= 4 messages to have anything to summarise
        for i in range(6):
            msg = make_message(
                session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_ml1_{i:03d}"
            )
            await store.append_message(msg)
            part = make_raw_part(msg.id, session_id, part_id=f"part_ml1_{i:03d}")
            await store.append_part(part)

        result = await engine.run_compaction(session_id)
        assert result.level_used == 1

    def test_check_and_trigger_returns_false_when_no_overflow(
        self, estimator, event_bus, store, dag_store, config
    ):
        """check_and_trigger returns False and schedules nothing when under threshold."""
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo(model_id="test-model", context_limit=100_000, max_output_tokens=4_000)
        tokens = TokenUsage(total=1_000)  # well under the 76_000 threshold
        result = engine.check_and_trigger("test-sess", tokens, model)
        assert result is False
        assert engine._pending_task is None

    async def test_check_and_trigger_triggers_and_returns_true(
        self, session_id, estimator, event_bus, store, dag_store, config
    ):
        """check_and_trigger schedules a task and returns True on overflow."""
        from mnesis.events.bus import MnesisEvent

        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            id_generator=lambda p: f"{p}_trigger",
            session_model="anthropic/claude-haiku-4-5",
        )
        # usable = 100_000 - 4_000 - 20_000 = 76_000
        # soft_limit = 76_000 * 0.6 = 45_600; 80_000 is well above soft threshold
        model = ModelInfo(model_id="test-model", context_limit=100_000, max_output_tokens=4_000)
        tokens = TokenUsage(total=80_000)
        result = engine.check_and_trigger(session_id, tokens, model)
        assert result is True
        assert engine._pending_task is not None

        published = [e for e, _ in event_bus.collected]
        assert MnesisEvent.COMPACTION_TRIGGERED in published

        # Clean up background task
        await engine.wait_for_pending()

    async def test_wait_for_pending_cancels_in_flight_task(
        self, estimator, event_bus, store, dag_store, config
    ):
        """wait_for_pending cancels an in-flight task and clears _pending_task."""
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_wait"
        )
        task = asyncio.create_task(asyncio.sleep(60))
        engine._pending_task = task
        await engine.wait_for_pending()
        assert engine._pending_task is None
        assert task.cancelled()

    async def test_run_compaction_returns_zero_result_for_empty_session(
        self, session_id, store, dag_store, estimator, event_bus, config
    ):
        """run_compaction returns level_used=0 when the session has no messages."""
        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            id_generator=lambda p: f"{p}_empty",
            session_model="anthropic/claude-haiku-4-5",
        )
        result = await engine.run_compaction(session_id)
        assert result.session_id == session_id
        assert result.level_used == 0
        assert result.compacted_message_count == 0

    async def test_run_compaction_returns_failure_result_without_model(
        self, session_id, store, dag_store, estimator, event_bus, config
    ):
        """run_compaction returns level_used=0 when no compaction model is configured."""
        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            id_generator=lambda p: f"{p}_nomodel",
            # No session_model — will raise ValueError internally
        )
        for i in range(4):
            msg = make_message(
                session_id,
                role="user" if i % 2 == 0 else "assistant",
                msg_id=f"msg_nomodel_{i:03d}",
            )
            await store.append_message(msg)
            part = make_raw_part(msg.id, session_id, part_id=f"part_nomodel_{i:03d}")
            await store.append_part(part)
        result = await engine.run_compaction(session_id)
        assert result.level_used == 0

    async def test_run_compaction_inner_aborts_after_prune(
        self, session_id, store, dag_store, estimator, event_bus, config
    ):
        """_run_compaction_inner raises CancelledError when abort event is pre-set."""
        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            id_generator=lambda p: f"{p}_abort",
            session_model="anthropic/claude-haiku-4-5",
        )
        for i in range(4):
            msg = make_message(
                session_id,
                role="user" if i % 2 == 0 else "assistant",
                msg_id=f"msg_abort_{i:03d}",
            )
            await store.append_message(msg)
            part = make_raw_part(msg.id, session_id, part_id=f"part_abort_{i:03d}")
            await store.append_part(part)

        abort = asyncio.Event()
        abort.set()
        with pytest.raises(asyncio.CancelledError):
            await engine._run_compaction_inner(
                session_id, abort=abort, model_override="anthropic/claude-haiku-4-5"
            )

    async def test_run_compaction_inner_aborts_before_level1(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        """_run_compaction_inner raises CancelledError when abort is set after pruning."""
        abort = asyncio.Event()
        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            id_generator=lambda p: f"{p}_abort1",
            session_model="anthropic/claude-haiku-4-5",
        )
        for i in range(4):
            msg = make_message(
                session_id,
                role="user" if i % 2 == 0 else "assistant",
                msg_id=f"msg_abort1_{i:03d}",
            )
            await store.append_message(msg)
            part = make_raw_part(msg.id, session_id, part_id=f"part_abort1_{i:03d}")
            await store.append_part(part)

        # Set abort inside the pruner so abort is False at line 237 but True at line 279
        original_prune = engine._pruner.prune

        async def prune_then_abort(session_id: str):
            result = await original_prune(session_id)
            abort.set()
            return result

        monkeypatch.setattr(engine._pruner, "prune", prune_then_abort)

        with pytest.raises(asyncio.CancelledError):
            await engine._run_compaction_inner(
                session_id, abort=abort, model_override="anthropic/claude-haiku-4-5"
            )
