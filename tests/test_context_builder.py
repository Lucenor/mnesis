"""Tests for ContextBuilder."""

from __future__ import annotations

import json

import pytest

from mnesis.context.builder import ContextBuilder
from mnesis.models.config import MnesisConfig, ModelInfo
from mnesis.models.message import ContextBudget
from tests.conftest import make_message, make_raw_part


@pytest.fixture
def model_info():
    """Reasonably sized context limit for testing — big enough to hold messages."""
    return ModelInfo(
        model_id="test-model",
        context_limit=100_000,
        max_output_tokens=4_000,
        encoding="cl100k_base",
    )


@pytest.fixture
def small_config():
    """Config with small compaction buffer."""
    return MnesisConfig()


@pytest.fixture
def builder(store, dag_store, estimator):
    return ContextBuilder(store, dag_store, estimator)


class TestContextBuilder:
    async def test_build_empty_session(self, session_id, builder, model_info, small_config):
        """Empty session returns empty message list."""
        ctx = await builder.build(session_id, model_info, "You are helpful.", small_config)
        assert ctx.messages == []
        assert ctx.has_summary is False
        assert ctx.token_estimate > 0  # system prompt counted

    async def test_build_without_summary_includes_all(
        self, session_id, store, builder, model_info, small_config
    ):
        """Without compaction, all messages are included."""
        for i in range(3):
            msg_id = f"msg_cb_{i:03d}"
            msg = make_message(
                session_id, role="user" if i % 2 == 0 else "assistant", msg_id=msg_id
            )
            await store.append_message(msg)
            part = make_raw_part(msg_id, session_id, part_id=f"part_cb_{i:03d}")
            await store.append_part(part)

        ctx = await builder.build(session_id, model_info, "System.", small_config)
        assert len(ctx.messages) == 3
        assert ctx.has_summary is False

    async def test_build_with_summary_excludes_old_messages(
        self, session_id, store, dag_store, builder, model_info, small_config
    ):
        """Messages before the summary are excluded; summary prepended."""
        # 3 old messages
        for i in range(3):
            msg_id = f"msg_old_{i:03d}"
            msg = make_message(session_id, role="user", msg_id=msg_id)
            await store.append_message(msg)

        # Insert a summary via dag_store
        from mnesis.models.summary import SummaryNode

        summary_node = SummaryNode(
            id="msg_sum_001",
            session_id=session_id,
            level=0,
            span_start_message_id="msg_old_000",
            span_end_message_id="msg_old_002",
            content="Summary of the old messages.",
            token_count=50,
            compaction_level=1,
        )
        counter = [0]
        def gen():
            counter[0] += 1
            return f"part_sum_{counter[0]:03d}"
        await dag_store.insert_node(summary_node, id_generator=gen)

        # 2 new messages after the summary
        import asyncio
        await asyncio.sleep(0.01)
        for i in range(2):
            msg_id = f"msg_new_{i:03d}"
            msg = make_message(session_id, role="user", msg_id=msg_id)
            await store.append_message(msg)
            part = make_raw_part(msg_id, session_id, part_id=f"part_new_{i:03d}")
            await store.append_part(part)

        ctx = await builder.build(session_id, model_info, "System.", small_config)
        assert ctx.has_summary is True
        # First message should be the summary (assistant)
        assert ctx.messages[0].role == "assistant"
        assert "Summary" in ctx.messages[0].content
        # Only new messages after summary
        assert len(ctx.messages) == 3  # summary + 2 new

    async def test_build_pruned_tool_renders_tombstone(
        self, session_id, store, builder, model_info, small_config
    ):
        """Pruned tool parts (compacted_at set) render as tombstone strings."""
        msg_id = "msg_tool_001"
        msg = make_message(session_id, role="assistant", msg_id=msg_id)
        await store.append_message(msg)

        tool_content = json.dumps({
            "type": "tool",
            "tool_name": "read_file",
            "tool_call_id": "call_001",
            "input": {"path": "/tmp/test.py"},
            "output": "def hello(): pass",
            "status": {"state": "completed"},
        })
        part = make_raw_part(
            msg_id, session_id, part_type="tool",
            part_id="part_tool_001", content=tool_content,
            tool_call_id="call_001", tool_name="read_file", tool_state="completed"
        )
        await store.append_part(part)

        # Apply tombstone
        import time
        await store.update_part_status("part_tool_001", compacted_at=int(time.time() * 1000))

        ctx = await builder.build(session_id, model_info, "System.", small_config)
        assert len(ctx.messages) == 1
        assert "compacted" in ctx.messages[0].content.lower()
        # Original output should NOT appear
        assert "def hello" not in ctx.messages[0].content

    async def test_build_file_ref_renders_block(
        self, session_id, store, builder, model_info, small_config
    ):
        """FileRefPart is rendered as a [FILE: ...] block."""
        msg_id = "msg_file_001"
        msg = make_message(session_id, role="user", msg_id=msg_id)
        await store.append_message(msg)

        file_content = json.dumps({
            "type": "file_ref",
            "content_id": "abc123def",
            "path": "/src/main.py",
            "file_type": "python",
            "token_count": 15000,
            "exploration_summary": "Module with 5 classes.",
        })
        part = make_raw_part(
            msg_id, session_id, part_type="file_ref",
            part_id="part_file_001", content=file_content
        )
        await store.append_part(part)

        ctx = await builder.build(session_id, model_info, "System.", small_config)
        assert len(ctx.messages) == 1
        assert "[FILE:" in ctx.messages[0].content
        assert "/src/main.py" in ctx.messages[0].content
        assert "Module with 5 classes" in ctx.messages[0].content
        assert "[/FILE]" in ctx.messages[0].content

    async def test_build_respects_token_budget(
        self, session_id, store, builder, model_info, small_config
    ):
        """Context builder stops including messages when budget is exceeded."""
        # Create a tiny model with very small context
        tiny_model = ModelInfo(
            model_id="tiny",
            context_limit=500,
            max_output_tokens=100,
            encoding="cl100k_base",
        )
        # Create many messages with substantial content
        for i in range(20):
            msg_id = f"msg_budget_{i:03d}"
            msg = make_message(session_id, role="user", msg_id=msg_id)
            await store.append_message(msg)
            # Each part has 200+ chars of content
            big_content = json.dumps({"type": "text", "text": "x" * 300})
            part = make_raw_part(
                msg_id, session_id, part_id=f"part_budget_{i:03d}", content=big_content
            )
            await store.append_part(part)

        ctx = await builder.build(session_id, tiny_model, "S.", small_config)
        # Should have fewer messages than we created
        assert len(ctx.messages) < 20
        # Token estimate must not exceed usable budget
        assert ctx.token_estimate <= ctx.budget.usable + ctx.budget.compaction_buffer


class TestContextBudget:
    def test_usable_calculation(self):
        budget = ContextBudget(
            model_context_limit=200_000,
            reserved_output_tokens=8_192,
            compaction_buffer=20_000,
        )
        assert budget.usable == 200_000 - 8_192 - 20_000

    def test_fits(self):
        budget = ContextBudget(
            model_context_limit=10_000,
            reserved_output_tokens=1_000,
            compaction_buffer=1_000,
        )
        assert budget.fits(7_999) is True
        assert budget.fits(8_001) is False

    def test_remaining(self):
        budget = ContextBudget(
            model_context_limit=10_000,
            reserved_output_tokens=1_000,
            compaction_buffer=1_000,
        )
        assert budget.remaining(3_000) == budget.usable - 3_000
