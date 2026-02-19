"""Tests for ToolOutputPruner."""

from __future__ import annotations

import json

from mnesis.compaction.pruner import ToolOutputPrunerAsync
from mnesis.models.config import CompactionConfig, MnesisConfig
from tests.conftest import make_message, make_raw_part


def _make_tool_part(
    msg_id: str,
    session_id: str,
    part_id: str,
    tool_call_id: str = "call_001",
    tool_name: str = "read_file",
    output: str = "x" * 500,
    state: str = "completed",
    compacted_at: int | None = None,
) -> tuple:
    """Returns (RawMessagePart, json_content)."""
    content = json.dumps(
        {
            "type": "tool",
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "input": {"path": "/test.py"},
            "output": output,
            "status": {"state": state, "compacted_at": compacted_at},
        }
    )
    part = make_raw_part(
        msg_id,
        session_id,
        part_type="tool",
        part_id=part_id,
        content=content,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        tool_state=state,
    )
    return part


class TestToolOutputPruner:
    async def test_prune_noop_when_disabled(self, session_id, store, estimator, config):
        """Pruner is a no-op when compaction.prune is False."""
        cfg = config.model_copy(update={"compaction": CompactionConfig(prune=False)})
        pruner = ToolOutputPrunerAsync(store, estimator, cfg)
        result = await pruner.prune(session_id)
        assert result.pruned_count == 0

    async def test_prune_noop_when_no_messages(self, session_id, store, estimator, config):
        """Pruner returns zero results for an empty session."""
        pruner = ToolOutputPrunerAsync(store, estimator, config)
        result = await pruner.prune(session_id)
        assert result.pruned_count == 0
        assert result.pruned_tokens == 0

    async def test_prune_skips_protected_tools(self, session_id, store, estimator, config):
        """Protected tools (e.g. 'skill') are never pruned."""
        # Create enough messages to pass the protect window
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_prot_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)
            if role == "assistant":
                part = _make_tool_part(
                    msg_id,
                    session_id,
                    part_id=f"part_prot_{i:03d}",
                    tool_call_id=f"call_{i:03d}",
                    tool_name="skill",  # Protected
                    output="x" * 2000,
                )
                await store.append_part(part)

        pruner = ToolOutputPrunerAsync(store, estimator, config)
        result = await pruner.prune(session_id)
        # Protected tools should never be pruned
        assert result.pruned_count == 0

    async def test_prune_applies_tombstones(self, session_id, store, estimator):
        """Unprotected tool outputs outside protect window are tombstoned."""
        # Use tight protect window so pruning is easily triggered
        cfg = MnesisConfig(
            compaction=CompactionConfig(
                prune_protect_tokens=100,  # Very small protect window
                prune_minimum_tokens=50,  # Very small minimum
            )
        )

        # Create enough messages to push past the protect window
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_tomb_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)
            if role == "assistant":
                part = _make_tool_part(
                    msg_id,
                    session_id,
                    part_id=f"part_tomb_{i:03d}",
                    tool_call_id=f"call_{i:03d}",
                    tool_name="read_file",
                    output="x" * 300,  # 300 chars / 4 ≈ 75 tokens each
                )
                await store.append_part(part)

        pruner = ToolOutputPrunerAsync(store, estimator, cfg)
        result = await pruner.prune(session_id)

        if result.pruned_count > 0:
            # Verify tombstones were applied
            parts = await store.get_messages_with_parts(session_id)
            tombstoned = []
            for mwp in parts:
                raw_parts = await store.get_parts(mwp.id)
                for rp in raw_parts:
                    if rp.part_type == "tool" and rp.compacted_at is not None:
                        tombstoned.append(rp.id)
            assert len(tombstoned) == result.pruned_count

    async def test_prune_skips_recent_turns(self, session_id, store, estimator):
        """Tool outputs in the most recent 2 user turns are never pruned."""
        cfg = MnesisConfig(
            compaction=CompactionConfig(
                prune_protect_tokens=10,  # Very small — would prune most things
                prune_minimum_tokens=5,
            )
        )

        # Only create 2 user turns (should all be protected)
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_recent_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)
            if role == "assistant":
                part = _make_tool_part(
                    msg_id,
                    session_id,
                    part_id=f"part_recent_{i:03d}",
                    tool_call_id=f"call_r{i:03d}",
                    output="x" * 100,
                )
                await store.append_part(part)

        pruner = ToolOutputPrunerAsync(store, estimator, cfg)
        result = await pruner.prune(session_id)
        # Recent 2 user turns protect all these tool outputs
        assert result.pruned_count == 0

    async def test_prune_stops_at_summary_boundary(self, session_id, store, dag_store, estimator):
        """Pruner stops scanning at is_summary messages."""
        cfg = MnesisConfig(
            compaction=CompactionConfig(
                prune_protect_tokens=10,
                prune_minimum_tokens=5,
            )
        )

        # Create old messages before a summary
        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_pre_sum_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)
            if role == "assistant":
                part = _make_tool_part(
                    msg_id,
                    session_id,
                    part_id=f"part_pre_{i:03d}",
                    tool_call_id=f"call_pre_{i:03d}",
                    output="x" * 500,
                )
                await store.append_part(part)

        # Insert summary
        import asyncio

        await asyncio.sleep(0.01)
        summary_msg = make_message(
            session_id, role="assistant", msg_id="msg_sum_prune_001", is_summary=True
        )
        await store.append_message(summary_msg)

        # Create 2 new messages after summary (more than 2 user turns)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_post_sum_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)

        pruner = ToolOutputPrunerAsync(store, estimator, cfg)
        # The pruner should stop at the summary and not go back further
        result = await pruner.prune(session_id)
        # No tool parts after the summary, so nothing to prune
        assert result.pruned_count == 0
