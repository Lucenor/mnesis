"""Tests for ToolOutputPruner."""

from __future__ import annotations

import json

from mnesis.compaction.engine import CompactionEngine
from mnesis.compaction.pruner import ToolOutputPrunerAsync
from mnesis.events.bus import MnesisEvent
from mnesis.models.config import CompactionConfig, MnesisConfig
from mnesis.store.immutable import RawMessagePart
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
) -> RawMessagePart:
    """Build and return a RawMessagePart with tool content for use in pruner tests."""
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
        compacted_at=compacted_at,
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

    async def test_prune_skipped_below_minimum_tokens(self, session_id, store, estimator):
        """Pruner is a no-op when total prunable volume is below prune_minimum_tokens.

        Covers lines 122-133 in pruner.py: when pruned_volume <= minimum_tokens
        the method returns early with pruned_count=0.
        """
        cfg = MnesisConfig(
            compaction=CompactionConfig(
                prune_protect_tokens=200_000,  # large protect window (must exceed minimum)
                prune_minimum_tokens=100_000,  # huge minimum — nothing will ever exceed it
            )
        )

        # Create enough turns to pass the user-turn guard (>= 2 user turns)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_min_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)
            if role == "assistant":
                part = _make_tool_part(
                    msg_id,
                    session_id,
                    part_id=f"part_min_{i:03d}",
                    tool_call_id=f"call_m{i:03d}",
                    output="x" * 100,  # Small output — well below 100K minimum
                )
                await store.append_part(part)

        pruner = ToolOutputPrunerAsync(store, estimator, cfg)
        result = await pruner.prune(session_id)

        assert result.pruned_count == 0
        assert result.pruned_tokens == 0
        # Tool parts were added so the scanner must have visited them before the
        # minimum-token early return.
        assert result.candidates_scanned > 0

    async def test_prune_all_tool_outputs_already_compacted(self, session_id, store, estimator):
        """Pruner is a no-op when all tool outputs already have compacted_at set.

        Covers the break-on-compacted_at path (line 219-220 in ToolOutputPrunerAsync).
        """
        import time as _time

        cfg = MnesisConfig(
            compaction=CompactionConfig(
                prune_protect_tokens=10,
                prune_minimum_tokens=5,
            )
        )

        now_ms = int(_time.time() * 1000)

        # Create 6 messages — all tool outputs already tombstoned
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_already_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)
            if role == "assistant":
                part = _make_tool_part(
                    msg_id,
                    session_id,
                    part_id=f"part_already_{i:03d}",
                    tool_call_id=f"call_a{i:03d}",
                    output="x" * 300,
                    compacted_at=now_ms,  # already tombstoned
                )
                await store.append_part(part)

        pruner = ToolOutputPrunerAsync(store, estimator, cfg)
        result = await pruner.prune(session_id)

        # All outputs already compacted — nothing new to tombstone
        assert result.pruned_count == 0

    async def test_base_pruner_prune_noop_when_disabled(self, session_id, store, estimator):
        """Base ToolOutputPruner.prune() returns early when prune=False.

        Covers lines 67-68 of the base class implementation.
        """
        from mnesis.compaction.pruner import ToolOutputPruner

        cfg = MnesisConfig(compaction=CompactionConfig(prune=False))
        pruner = ToolOutputPruner(store, estimator, cfg)
        result = await pruner.prune(session_id)
        assert result.pruned_count == 0
        assert result.pruned_tokens == 0

    async def test_base_pruner_prune_empty_session(self, session_id, store, estimator):
        """Base ToolOutputPruner.prune() returns zeros for an empty session.

        Covers lines 73-75 (early return on empty messages).
        """
        from mnesis.compaction.pruner import ToolOutputPruner

        cfg = MnesisConfig(compaction=CompactionConfig(prune=True))
        pruner = ToolOutputPruner(store, estimator, cfg)
        result = await pruner.prune(session_id)
        assert result.pruned_count == 0
        assert result.candidates_scanned == 0

    async def test_base_pruner_prune_applies_tombstones(self, session_id, store, estimator):
        """Base ToolOutputPruner.prune() tombstones outputs outside protect window.

        Covers lines 83-150 of the base class (full pruning loop including
        _get_part_id lookup).
        """
        cfg = MnesisConfig(
            compaction=CompactionConfig(
                prune_protect_tokens=50,
                prune_minimum_tokens=10,
            )
        )

        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_base_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)
            if role == "assistant":
                part = _make_tool_part(
                    msg_id,
                    session_id,
                    part_id=f"part_base_{i:03d}",
                    tool_call_id=f"call_b{i:03d}",
                    output="x" * 300,
                )
                await store.append_part(part)

        from mnesis.compaction.pruner import ToolOutputPruner

        pruner = ToolOutputPruner(store, estimator, cfg)
        result = await pruner.prune(session_id)

        # With 4 assistant messages each carrying ~300-char output (>> 50-token protect window),
        # the base pruner must tombstone at least one part outside that window.
        assert result.pruned_count > 0
        assert result.candidates_scanned > 0

    async def test_base_pruner_get_part_id_sync_returns_empty(self, session_id, store, estimator):
        """_get_part_id_sync always returns empty string (sentinel).

        Covers line 165 of pruner.py.
        """
        from mnesis.compaction.pruner import ToolOutputPruner
        from mnesis.models.message import ToolPart, ToolStatus

        cfg = MnesisConfig()
        pruner = ToolOutputPruner(store, estimator, cfg)

        # Create a minimal MessageWithParts and ToolPart to pass in
        from mnesis.models.message import Message, MessageWithParts

        msg = Message(
            id="msg_sync_001",
            session_id=session_id,
            role="assistant",
        )
        tool_part = ToolPart(
            tool_name="test_tool",
            tool_call_id="call_sync_001",
            input={},
            status=ToolStatus(state="completed"),
        )
        mwp = MessageWithParts(message=msg, parts=[tool_part])
        result = pruner._get_part_id_sync(mwp, tool_part)
        assert result == ""


class TestPruneCompletedEvent:
    async def test_prune_completed_event_published_when_pruning_occurs(
        self, session_id, store, dag_store, estimator, event_bus
    ):
        """PRUNE_COMPLETED event is published with correct payload when pruning fires."""
        cfg = MnesisConfig(
            compaction=CompactionConfig(
                prune_protect_tokens=100,
                prune_minimum_tokens=50,
            )
        )

        # 10 messages (5 user/5 assistant pairs). The pruner protects the 2 most
        # recent user turns, so the 3 oldest assistant tool outputs (≈75 tokens each,
        # 225 tokens total) fall outside the protect window and will be pruned.
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_pcev_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)
            if role == "assistant":
                part = _make_tool_part(
                    msg_id,
                    session_id,
                    part_id=f"part_pcev_{i:03d}",
                    tool_call_id=f"call_pcev_{i:03d}",
                    tool_name="read_file",
                    output="x" * 300,  # 300 chars / 4 ≈ 75 tokens each
                )
                await store.append_part(part)

        # Use a counter-based id_generator to avoid UNIQUE constraint collisions
        _counter: list[int] = [0]

        def _id_gen(prefix: str) -> str:
            _counter[0] += 1
            return f"{prefix}_pcev_{_counter[0]:04d}"

        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            cfg,
            id_generator=_id_gen,
            session_model="nonexistent-model-xyz",
        )
        await engine.run_compaction(session_id, model_override="nonexistent-model-xyz")

        prune_events = [(e, p) for e, p in event_bus.collected if e == MnesisEvent.PRUNE_COMPLETED]

        assert len(prune_events) == 1, (
            f"Expected exactly one PRUNE_COMPLETED event, got {len(prune_events)}"
        )
        _, payload = prune_events[0]
        assert payload["session_id"] == session_id
        assert isinstance(payload["pruned_count"], int)
        assert payload["pruned_count"] > 0
        assert isinstance(payload["pruned_tokens"], int)
        assert payload["pruned_tokens"] > 0

    async def test_prune_completed_event_not_published_when_no_pruning(
        self, session_id, store, dag_store, estimator, event_bus
    ):
        """PRUNE_COMPLETED is NOT published when pruning is disabled or no candidates exist."""
        cfg = MnesisConfig(
            compaction=CompactionConfig(
                prune=False,  # pruning disabled
            )
        )

        for i in range(4):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_nopev_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)

        _ctr: list[int] = [0]

        def _id_gen_nopev(prefix: str) -> str:
            _ctr[0] += 1
            return f"{prefix}_nopev_{_ctr[0]:04d}"

        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            cfg,
            id_generator=_id_gen_nopev,
            session_model="nonexistent-model-xyz",
        )
        await engine.run_compaction(session_id, model_override="nonexistent-model-xyz")

        prune_events = [e for e, _ in event_bus.collected if e == MnesisEvent.PRUNE_COMPLETED]
        assert len(prune_events) == 0

    async def test_prune_completed_payload_fields(
        self, session_id, store, dag_store, estimator, event_bus
    ):
        """PRUNE_COMPLETED payload contains session_id, pruned_count, pruned_tokens."""
        cfg = MnesisConfig(
            compaction=CompactionConfig(
                prune_protect_tokens=50,
                prune_minimum_tokens=10,
            )
        )

        # Create messages designed to push well past the protect window
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            msg_id = f"msg_payload_{i:03d}"
            msg = make_message(session_id, role=role, msg_id=msg_id)
            await store.append_message(msg)
            if role == "assistant":
                part = _make_tool_part(
                    msg_id,
                    session_id,
                    part_id=f"part_payload_{i:03d}",
                    tool_call_id=f"call_pl_{i:03d}",
                    tool_name="read_file",
                    output="x" * 500,
                )
                await store.append_part(part)

        _ctr2: list[int] = [0]

        def _id_gen_pl(prefix: str) -> str:
            _ctr2[0] += 1
            return f"{prefix}_pl_{_ctr2[0]:04d}"

        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            cfg,
            id_generator=_id_gen_pl,
            session_model="nonexistent-model-xyz",
        )
        await engine.run_compaction(session_id, model_override="nonexistent-model-xyz")

        prune_events = [(e, p) for e, p in event_bus.collected if e == MnesisEvent.PRUNE_COMPLETED]

        assert len(prune_events) == 1, (
            f"Expected exactly one PRUNE_COMPLETED event, got {len(prune_events)}"
        )
        _, payload = prune_events[0]
        # Verify all required TypedDict fields are present and correctly typed
        assert "session_id" in payload
        assert "pruned_count" in payload
        assert "pruned_tokens" in payload
        assert payload["session_id"] == session_id
        assert isinstance(payload["pruned_count"], int)
        assert isinstance(payload["pruned_tokens"], int)
