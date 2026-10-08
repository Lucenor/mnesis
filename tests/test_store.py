"""Tests for ImmutableStore and SummaryDAGStore."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from mnesis.models.message import TokenUsage
from mnesis.models.summary import FileReference
from mnesis.store.immutable import (
    DuplicateIDError,
    ImmutableStore,
    PartNotFoundError,
    SessionNotFoundError,
)
from tests.conftest import make_message, make_raw_part


class TestImmutableStore:
    async def test_create_session(self, store):
        """Creating a session returns a Session with correct fields."""
        session = await store.create_session("sess_001", model_id="gpt-4o", agent="test")
        assert session.id == "sess_001"
        assert session.model_id == "gpt-4o"
        assert session.agent == "test"
        assert session.is_active is True

    async def test_create_session_duplicate_raises(self, store):
        """Duplicate session ID raises DuplicateIDError."""
        await store.create_session("sess_dup")
        with pytest.raises(DuplicateIDError):
            await store.create_session("sess_dup")

    async def test_get_session(self, store):
        """get_session returns the stored session."""
        await store.create_session("sess_get", model_id="claude-3")
        session = await store.get_session("sess_get")
        assert session.id == "sess_get"
        assert session.model_id == "claude-3"

    async def test_get_session_not_found(self, store):
        """get_session raises SessionNotFoundError for missing session."""
        with pytest.raises(SessionNotFoundError):
            await store.get_session("sess_nonexistent")

    async def test_list_sessions(self, store):
        """list_sessions returns sessions in reverse chronological order."""
        for i in range(3):
            await store.create_session(f"sess_list_{i}", model_id="gpt-4")
        sessions = await store.list_sessions()
        assert len(sessions) >= 3
        # Newest first
        timestamps = [s.created_at for s in sessions]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_soft_delete_session(self, store):
        """Soft-deleted sessions excluded from active listing, messages retained."""
        await store.create_session("sess_del")
        msg = make_message("sess_del", msg_id="msg_del_001")
        await store.append_message(msg)
        await store.soft_delete_session("sess_del")
        active_sessions = await store.list_sessions(active_only=True)
        assert all(s.id != "sess_del" for s in active_sessions)
        # Messages still retrievable
        loaded = await store.get_message("msg_del_001")
        assert loaded.id == "msg_del_001"

    async def test_append_message(self, session_id, store):
        """append_message stores a message and it can be retrieved."""
        msg = make_message(session_id, role="user", msg_id="msg_001")
        stored = await store.append_message(msg)
        assert stored.id == "msg_001"
        loaded = await store.get_message("msg_001")
        assert loaded.role == "user"
        assert loaded.session_id == session_id

    async def test_append_message_duplicate_raises(self, session_id, store):
        """Duplicate message ID raises DuplicateIDError."""
        msg = make_message(session_id, msg_id="msg_dup_001")
        await store.append_message(msg)
        with pytest.raises(DuplicateIDError):
            await store.append_message(msg)

    async def test_append_message_bad_session_raises(self, store):
        """Message with unknown session_id raises SessionNotFoundError."""
        msg = make_message("sess_nonexistent", msg_id="msg_bad_sess")
        with pytest.raises(SessionNotFoundError):
            await store.append_message(msg)

    async def test_append_part_assigns_index(self, session_id, store):
        """Parts receive sequential part_index values."""
        msg = make_message(session_id, role="assistant", msg_id="msg_parts_001")
        await store.append_message(msg)

        part1 = make_raw_part("msg_parts_001", session_id, part_id="part_001")
        part2 = make_raw_part("msg_parts_001", session_id, part_id="part_002")
        part3 = make_raw_part("msg_parts_001", session_id, part_id="part_003")

        await store.append_part(part1)
        await store.append_part(part2)
        await store.append_part(part3)

        parts = await store.get_parts("msg_parts_001")
        assert [p.part_index for p in parts] == [0, 1, 2]

    async def test_append_part_concurrent_unique_indexes(self, session_id, store):
        """Concurrent appends to one message get distinct, gap-free part indexes."""
        msg = make_message(session_id, role="assistant", msg_id="msg_conc_001")
        await store.append_message(msg)

        raw_parts = [
            make_raw_part("msg_conc_001", session_id, part_id=f"part_conc_{i:03d}")
            for i in range(25)
        ]
        stored = await asyncio.gather(*(store.append_part(p) for p in raw_parts))

        returned = sorted(p.part_index for p in stored)
        assert returned == list(range(25))
        persisted = await store.get_parts("msg_conc_001")
        assert [p.part_index for p in persisted] == list(range(25))
        # Each returned part reports the index it was actually stored with.
        by_id = {p.id: p.part_index for p in persisted}
        assert all(by_id[p.id] == p.part_index for p in stored)

    async def test_append_part_unknown_message_raises(self, session_id, store):
        """append_part for a nonexistent message raises MessageNotFoundError."""
        from mnesis.store.immutable import MessageNotFoundError

        part = make_raw_part("msg_missing", session_id, part_id="part_orphan")
        with pytest.raises(MessageNotFoundError):
            await store.append_part(part)

    async def test_update_part_status_compacted_at(self, session_id, store):
        """update_part_status sets the compacted_at tombstone."""
        msg = make_message(session_id, role="assistant", msg_id="msg_prune_001")
        await store.append_message(msg)
        part = make_raw_part(
            "msg_prune_001",
            session_id,
            part_type="tool",
            part_id="part_prune_001",
            tool_call_id="call_001",
            tool_name="read_file",
            tool_state="completed",
        )
        await store.append_part(part)

        now_ms = int(time.time() * 1000)
        await store.update_part_status("part_prune_001", compacted_at=now_ms)

        parts = await store.get_parts("msg_prune_001")
        assert parts[0].compacted_at == now_ms

    async def test_update_part_status_not_found(self, store):
        """update_part_status raises PartNotFoundError for missing part."""
        with pytest.raises(PartNotFoundError):
            await store.update_part_status("part_nonexistent", tool_state="running")

    async def test_batch_set_compacted_at_empty_is_noop(self, store):
        """batch_set_compacted_at with an empty list is a no-op (early-return path)."""
        # Should not raise and should return without touching the DB.
        await store.batch_set_compacted_at([], compacted_at=1_000_000)

    async def test_batch_set_compacted_at_marks_parts(self, session_id, store):
        """batch_set_compacted_at stamps compacted_at on every supplied part ID."""
        msg = make_message(session_id, role="assistant", msg_id="msg_batch_ca_001")
        await store.append_message(msg)
        for i in range(3):
            await store.append_part(
                make_raw_part(
                    "msg_batch_ca_001",
                    session_id,
                    part_type="tool",
                    part_id=f"part_batch_ca_{i:03d}",
                    tool_call_id=f"call_{i:03d}",
                )
            )

        now_ms = int(time.time() * 1000)
        await store.batch_set_compacted_at(
            ["part_batch_ca_000", "part_batch_ca_001", "part_batch_ca_002"],
            compacted_at=now_ms,
        )

        parts = await store.get_parts("msg_batch_ca_001")
        assert all(p.compacted_at == now_ms for p in parts)

    async def test_batch_set_compacted_at_not_found(self, store):
        """batch_set_compacted_at raises PartNotFoundError when no rows match."""
        with pytest.raises(PartNotFoundError):
            await store.batch_set_compacted_at(
                ["part_nonexistent_x", "part_nonexistent_y"],
                compacted_at=1_000_000,
            )

    async def test_get_raw_parts_for_messages_empty_is_noop(self, store):
        """get_raw_parts_for_messages with an empty list returns [] (early-return path)."""
        result = await store.get_raw_parts_for_messages([])
        assert result == []

    async def test_get_raw_parts_for_messages_returns_all_parts(self, session_id, store):
        """get_raw_parts_for_messages fetches parts for all supplied message IDs."""
        for i in range(2):
            msg_id = f"msg_bulk_parts_{i:03d}"
            await store.append_message(make_message(session_id, msg_id=msg_id))
            for j in range(2):
                await store.append_part(
                    make_raw_part(
                        msg_id,
                        session_id,
                        part_id=f"part_bulk_{i:03d}_{j:03d}",
                    )
                )

        results = await store.get_raw_parts_for_messages(
            ["msg_bulk_parts_000", "msg_bulk_parts_001"]
        )
        assert len(results) == 4
        msg_ids = {r.message_id for r in results}
        assert msg_ids == {"msg_bulk_parts_000", "msg_bulk_parts_001"}

    async def test_get_messages_with_parts_two_queries(self, session_id, store):
        """get_messages_with_parts uses batch loading (correctness check)."""
        for i in range(5):
            msg_id = f"msg_batch_{i:03d}"
            msg = make_message(
                session_id, role="user" if i % 2 == 0 else "assistant", msg_id=msg_id
            )
            await store.append_message(msg)
            part = make_raw_part(msg_id, session_id, part_id=f"part_batch_{i:03d}")
            await store.append_part(part)

        results = await store.get_messages_with_parts(session_id)
        assert len(results) == 5
        for mwp in results:
            assert len(mwp.parts) == 1

    async def test_get_last_summary_message(self, session_id, store):
        """get_last_summary_message returns the most recent is_summary message."""
        for i in range(3):
            msg = make_message(session_id, role="user", msg_id=f"msg_sum_{i:03d}")
            await store.append_message(msg)

        # Insert two summary messages
        sum1 = make_message(session_id, role="assistant", msg_id="msg_sum_s001", is_summary=True)
        sum2 = make_message(session_id, role="assistant", msg_id="msg_sum_s002", is_summary=True)
        await store.append_message(sum1)
        await asyncio.sleep(0.01)  # Ensure different timestamps
        await store.append_message(sum2)

        latest = await store.get_last_summary_message(session_id)
        assert latest is not None
        assert latest.id == "msg_sum_s002"

    async def test_update_message_tokens(self, session_id, store):
        """Token usage is updated correctly after streaming."""
        msg = make_message(session_id, role="assistant", msg_id="msg_tok_001")
        await store.append_message(msg)

        usage = TokenUsage(input=100, output=200, total=300)
        await store.update_message_tokens("msg_tok_001", usage, 0.05, "stop")

        loaded = await store.get_message("msg_tok_001")
        assert loaded.tokens is not None
        assert loaded.tokens.input == 100
        assert loaded.tokens.output == 200
        assert loaded.finish_reason == "stop"

    async def test_sum_token_usage_matches_python_aggregation(self, session_id, store):
        """SQL aggregation mirrors TokenUsage.__add__ and skips summary messages."""
        usages = [
            ("msg_sum_a", TokenUsage(input=10, output=20, total=0), False),
            ("msg_sum_b", TokenUsage(input=5, output=5, cache_read=7, total=100), False),
            ("msg_sum_c", TokenUsage(input=999, output=999, total=1998), True),
        ]
        expected = TokenUsage()
        for msg_id, usage, is_summary in usages:
            await store.append_message(
                make_message(session_id, role="assistant", msg_id=msg_id, is_summary=is_summary)
            )
            await store.update_message_tokens(msg_id, usage, 0.0, "stop")
            if not is_summary:
                expected = expected + usage

        got = await store.sum_token_usage(session_id)
        assert got == expected
        assert got.effective_total() == 130

    async def test_file_reference_upsert(self, store):
        """Storing a file reference twice updates the existing row."""
        ref1 = FileReference(
            content_id="abc123",
            path="/tmp/test.py",
            file_type="python",
            token_count=500,
            exploration_summary="Module with 3 classes.",
        )
        ref2 = FileReference(
            content_id="abc123",
            path="/tmp/test_v2.py",  # Path can change
            file_type="python",
            token_count=600,
            exploration_summary="Updated summary.",
        )
        await store.store_file_reference(ref1)
        await store.store_file_reference(ref2)

        fetched = await store.get_file_reference("abc123")
        assert fetched is not None
        assert fetched.token_count == 600
        assert fetched.exploration_summary == "Updated summary."

    async def test_get_file_reference_not_found(self, store):
        """get_file_reference returns None for unknown content_id."""
        result = await store.get_file_reference("nonexistent_hash")
        assert result is None

    async def test_get_messages_since_message_id(self, session_id, store):
        """since_message_id correctly filters to messages after the boundary."""
        messages = []
        for i in range(5):
            await asyncio.sleep(0.001)
            msg = make_message(session_id, msg_id=f"msg_since_{i:03d}")
            await store.append_message(msg)
            messages.append(msg)

        # Get messages after the second message
        result = await store.get_messages(session_id, since_message_id=messages[1].id)
        assert len(result) == 3
        assert result[0].id == messages[2].id


class TestSummaryDAGStore:
    async def test_get_latest_node_none_when_no_summary(self, session_id, store, dag_store):
        """Returns None when no summary messages exist."""
        node = await dag_store.get_latest_node(session_id)
        assert node is None

    async def test_get_active_nodes_empty(self, session_id, store, dag_store):
        """Returns empty list when no compaction has occurred."""
        nodes = await dag_store.get_active_nodes(session_id)
        assert nodes == []

    async def test_get_coverage_gaps_all_uncovered(self, session_id, store, dag_store):
        """All messages are in one gap when there's no summary."""
        for i in range(3):
            msg = make_message(session_id, msg_id=f"msg_gap_{i:03d}")
            await store.append_message(msg)

        gaps = await dag_store.get_coverage_gaps(session_id)
        assert len(gaps) == 1
        assert gaps[0].message_count == 3


# ── DAG Persistence Tests ──────────────────────────────────────────────────────


async def _insert_leaf_node(
    dag_store: object,
    session_id: str,
    node_id: str,
    content: str = "summary text",
    token_count: int = 50,
) -> object:
    """Helper: insert a leaf SummaryNode and return it."""
    from mnesis.models.summary import SummaryNode
    from mnesis.session import make_id

    node = SummaryNode(
        id=node_id,
        session_id=session_id,
        kind="leaf",
        span_start_message_id="span_start_placeholder",
        span_end_message_id="span_end_placeholder",
        content=content,
        token_count=token_count,
    )
    return await dag_store.insert_node(node, id_generator=lambda: make_id("part"))


class TestDAGPersistence:
    """Tests that verify DAG state is persisted to summary_nodes and survives restarts."""

    async def test_insert_leaf_node_persists_to_summary_nodes(self, session_id, store, dag_store):
        """insert_node writes kind and parent_node_ids to summary_nodes table."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        node = SummaryNode(
            id="node_leaf_01",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="span_a",
            span_end_message_id="span_b",
            content="leaf summary",
            token_count=42,
        )
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        # Verify the row was written to summary_nodes
        conn = store._conn_or_raise()
        async with conn.execute(
            "SELECT kind, parent_node_ids, superseded FROM summary_nodes WHERE id=?",
            ("node_leaf_01",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row["kind"] == "leaf"
        assert row["parent_node_ids"] == "[]"
        assert row["superseded"] == 0

    async def test_insert_condensed_node_persists_parent_node_ids(
        self, session_id, store, dag_store
    ):
        """Condensed node persists parent_node_ids as JSON array."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        parent_ids = ["parent_node_01", "parent_node_02"]
        node = SummaryNode(
            id="node_condensed_01",
            session_id=session_id,
            kind="condensed",
            span_start_message_id="span_a",
            span_end_message_id="span_b",
            content="condensed summary",
            token_count=100,
            parent_node_ids=parent_ids,
        )
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        conn = store._conn_or_raise()
        async with conn.execute(
            "SELECT kind, parent_node_ids FROM summary_nodes WHERE id=?",
            ("node_condensed_01",),
        ) as cursor:
            row = await cursor.fetchone()

        assert row is not None
        assert row["kind"] == "condensed"
        assert json.loads(row["parent_node_ids"]) == parent_ids

    async def test_get_active_nodes_restores_parent_node_ids_after_restart(self, config, pool):
        """Reopening the store returns condensed node with correct parent_node_ids."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id
        from mnesis.store.immutable import ImmutableStore
        from mnesis.store.summary_dag import SummaryDAGStore

        # ── First store instance ───────────────────────────────────────────────
        store1 = ImmutableStore(config.store, pool=pool)
        await store1.initialize()
        dag1 = SummaryDAGStore(store1)

        sid = "sess_restart_dag_01"
        await store1.create_session(sid, model_id="gpt-4o")

        # Insert a leaf node
        leaf = SummaryNode(
            id="node_leaf_r01",
            session_id=sid,
            kind="leaf",
            span_start_message_id="sm_a",
            span_end_message_id="sm_b",
            content="leaf text",
            token_count=30,
        )
        await dag1.insert_node(leaf, id_generator=lambda: make_id("part"))

        # Insert a condensed node that supersedes the leaf
        condensed = SummaryNode(
            id="node_condensed_r01",
            session_id=sid,
            kind="condensed",
            span_start_message_id="sm_a",
            span_end_message_id="sm_b",
            content="condensed text",
            token_count=20,
            parent_node_ids=["node_leaf_r01"],
        )
        await dag1.insert_node(condensed, id_generator=lambda: make_id("part"))
        await dag1.mark_superseded(["node_leaf_r01"])

        await store1.close()

        # ── Second store instance (simulates restart) ──────────────────────────
        store2 = ImmutableStore(config.store, pool=pool)
        await store2.initialize()
        dag2 = SummaryDAGStore(store2)

        active = await dag2.get_active_nodes(sid)

        assert len(active) == 1, f"Expected 1 active node, got {len(active)}"
        assert active[0].id == "node_condensed_r01"
        assert active[0].kind == "condensed"
        assert active[0].parent_node_ids == ["node_leaf_r01"]

        await store2.close()

    async def test_mark_superseded_persists_across_restart(self, config, pool):
        """mark_superseded sets superseded=1 in DB; a fresh store sees it as inactive."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id
        from mnesis.store.immutable import ImmutableStore
        from mnesis.store.summary_dag import SummaryDAGStore

        store1 = ImmutableStore(config.store, pool=pool)
        await store1.initialize()
        dag1 = SummaryDAGStore(store1)

        sid = "sess_sup_persist_01"
        await store1.create_session(sid, model_id="gpt-4o")

        node = SummaryNode(
            id="node_sup_p01",
            session_id=sid,
            kind="leaf",
            span_start_message_id="sm_x",
            span_end_message_id="sm_y",
            content="text",
            token_count=10,
        )
        await dag1.insert_node(node, id_generator=lambda: make_id("part"))

        # Verify active before superseding
        before = await dag1.get_active_nodes(sid)
        assert any(n.id == "node_sup_p01" for n in before)

        await dag1.mark_superseded(["node_sup_p01"])
        await store1.close()

        # Fresh store — no in-memory state
        store2 = ImmutableStore(config.store, pool=pool)
        await store2.initialize()
        dag2 = SummaryDAGStore(store2)

        after = await dag2.get_active_nodes(sid)
        assert not any(n.id == "node_sup_p01" for n in after)

        await store2.close()

    async def test_get_latest_node_ignores_superseded(self, session_id, store, dag_store):
        """get_latest_node returns the most recent non-superseded node."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        node_a = SummaryNode(
            id="node_latest_a",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_a",
            span_end_message_id="sm_b",
            content="older",
            token_count=10,
        )
        await dag_store.insert_node(node_a, id_generator=lambda: make_id("part"))

        await asyncio.sleep(0.01)  # ensure different created_at

        node_b = SummaryNode(
            id="node_latest_b",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_c",
            span_end_message_id="sm_d",
            content="newer",
            token_count=20,
        )
        await dag_store.insert_node(node_b, id_generator=lambda: make_id("part"))

        # Supersede the newer node
        await dag_store.mark_superseded(["node_latest_b"])

        latest = await dag_store.get_latest_node(session_id)
        assert latest is not None
        assert latest.id == "node_latest_a"

    async def test_get_active_nodes_content_restored(self, session_id, store, dag_store):
        """get_active_nodes correctly reconstructs node content from message parts."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        expected_content = "this is the summary content"
        node = SummaryNode(
            id="node_content_01",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_e",
            span_end_message_id="sm_f",
            content=expected_content,
            token_count=15,
        )
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        active = await dag_store.get_active_nodes(session_id)
        assert len(active) == 1
        assert active[0].content == expected_content
        assert active[0].token_count == 15
        assert active[0].kind == "leaf"
        assert active[0].parent_node_ids == []

    async def test_get_active_nodes_filters_in_memory_superseded(
        self, session_id, store, dag_store
    ):
        """get_active_nodes skips nodes in the in-memory _superseded_ids set."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        node = SummaryNode(
            id="node_mem_sup_01",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_a",
            span_end_message_id="sm_b",
            content="text",
            token_count=10,
        )
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        # Supersede in-memory (within same session instance); the DB row is
        # updated to superseded=1, so get_active_nodes must also skip it.
        await dag_store.mark_superseded(["node_mem_sup_01"])

        active = await dag_store.get_active_nodes(session_id)
        assert not any(n.id == "node_mem_sup_01" for n in active)

    async def test_get_latest_node_in_memory_superseded_scans_backwards(
        self, session_id, store, dag_store
    ):
        """get_latest_node falls back to older node when latest is in _superseded_ids."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        node_a = SummaryNode(
            id="node_scan_a",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_a",
            span_end_message_id="sm_b",
            content="older",
            token_count=10,
        )
        await dag_store.insert_node(node_a, id_generator=lambda: make_id("part"))

        await asyncio.sleep(0.01)

        node_b = SummaryNode(
            id="node_scan_b",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_c",
            span_end_message_id="sm_d",
            content="newer",
            token_count=20,
        )
        await dag_store.insert_node(node_b, id_generator=lambda: make_id("part"))

        # Mark newer in-memory; get_latest_node must scan backwards to node_a.
        await dag_store.mark_superseded(["node_scan_b"])

        latest = await dag_store.get_latest_node(session_id)
        assert latest is not None
        assert latest.id == "node_scan_a"

    async def test_get_latest_node_all_in_memory_superseded_returns_none(
        self, session_id, store, dag_store
    ):
        """get_latest_node returns None when all nodes are in _superseded_ids."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        node = SummaryNode(
            id="node_all_sup_01",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_a",
            span_end_message_id="sm_b",
            content="text",
            token_count=10,
        )
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))
        await dag_store.mark_superseded(["node_all_sup_01"])

        # After superseding, the DB row has superseded=1, so the query returns
        # no rows — get_latest_node returns None via the early-return path.
        latest = await dag_store.get_latest_node(session_id)
        assert latest is None

    async def test_get_coverage_gaps_no_non_summary_messages(self, session_id, store, dag_store):
        """get_coverage_gaps returns empty list when all messages are summaries."""
        # No non-summary messages added; get_coverage_gaps should return []
        gaps = await dag_store.get_coverage_gaps(session_id)
        assert gaps == []

    async def test_get_coverage_gaps_no_gap_after_summary(self, session_id, store, dag_store):
        """get_coverage_gaps returns [] when no non-summary messages follow the summary."""

        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        # Insert a non-summary message before the summary
        msg = make_message(session_id, msg_id="msg_before_sum")
        await store.append_message(msg)

        # Insert a summary node (which inserts an is_summary=True message)
        node = SummaryNode(
            id="node_gap_sum_01",
            session_id=session_id,
            kind="leaf",
            span_start_message_id=msg.id,
            span_end_message_id=msg.id,
            content="summary",
            token_count=5,
        )
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        # No non-summary messages added after the summary
        gaps = await dag_store.get_coverage_gaps(session_id)
        assert gaps == []

    async def test_get_node_by_id_with_summary_node_row(self, session_id, store, dag_store):
        """get_node_by_id returns a SummaryNode for a Phase-3 persisted node."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        node = SummaryNode(
            id="node_by_id_01",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_a",
            span_end_message_id="sm_b",
            content="content by id",
            token_count=25,
        )
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        fetched = await dag_store.get_node_by_id("node_by_id_01")
        assert fetched is not None
        assert fetched.id == "node_by_id_01"
        assert fetched.content == "content by id"
        assert fetched.kind == "leaf"

    async def test_get_node_by_id_nonexistent_returns_none(self, session_id, store, dag_store):
        """get_node_by_id returns None for an unknown node ID."""
        fetched = await dag_store.get_node_by_id("nonexistent_node_id")
        assert fetched is None

    async def test_get_node_by_id_non_summary_message_returns_none(
        self, session_id, store, dag_store
    ):
        """get_node_by_id returns None when the message exists but is not a summary."""
        msg = make_message(session_id, msg_id="non_sum_msg_01")
        await store.append_message(msg)

        fetched = await dag_store.get_node_by_id("non_sum_msg_01")
        assert fetched is None

    async def test_get_node_by_id_pre_phase3_fallback(self, config, pool):
        """get_node_by_id falls back to _build_node_from_message for pre-Phase-3 nodes."""

        from mnesis.models.message import Message
        from mnesis.session import make_id
        from mnesis.store.immutable import ImmutableStore, RawMessagePart
        from mnesis.store.summary_dag import SummaryDAGStore

        store = ImmutableStore(config.store, pool=pool)
        await store.initialize()
        dag = SummaryDAGStore(store)

        sid = "sess_pre_phase3_01"
        await store.create_session(sid, model_id="gpt-4o")

        # Insert a non-summary message so span reconstruction works
        non_sum = make_message(sid, msg_id="pre_p3_msg_01")
        await store.append_message(non_sum)

        # Manually insert a summary message WITHOUT a summary_nodes row
        # to simulate a pre-Phase-3 node.
        summary_msg_id = make_id("msg")
        import time as time_mod

        summary_msg = Message(
            id=summary_msg_id,
            session_id=sid,
            role="assistant",
            created_at=int(time_mod.time() * 1000) + 1000,
            is_summary=True,
        )
        await store.append_message(summary_msg)

        text_part = RawMessagePart(
            id=make_id("part"),
            message_id=summary_msg_id,
            session_id=sid,
            part_type="text",
            content=json.dumps({"type": "text", "text": "legacy summary"}),
            token_estimate=12,
        )
        await store.append_part(text_part)

        # No row inserted into summary_nodes — simulates pre-Phase-3 node.
        fetched = await dag.get_node_by_id(summary_msg_id)
        assert fetched is not None
        assert fetched.content == "legacy summary"
        assert fetched.parent_node_ids == []

        await store.close()

    @pytest.mark.parametrize(
        ("text_payload", "marker_payload"),
        [("not json{", "also not json"), ("[1, 2]", "[3]")],
        ids=["invalid-json", "non-object-json"],
    )
    async def test_get_node_by_id_pre_phase3_corrupt_parts(
        self, config, pool, text_payload, marker_payload
    ):
        """Unparseable legacy summary parts degrade to defaults instead of failing."""
        from mnesis.models.message import Message
        from mnesis.session import make_id
        from mnesis.store.immutable import ImmutableStore, RawMessagePart
        from mnesis.store.summary_dag import SummaryDAGStore

        store = ImmutableStore(config.store, pool=pool)
        await store.initialize()
        dag = SummaryDAGStore(store)

        sid = "sess_pre_phase3_corrupt"
        await store.create_session(sid, model_id="gpt-4o")
        await store.append_message(make_message(sid, msg_id="corrupt_msg_01"))

        summary_msg_id = make_id("msg")
        await store.append_message(
            Message(
                id=summary_msg_id,
                session_id=sid,
                role="assistant",
                created_at=int(time.time() * 1000) + 1000,
                is_summary=True,
            )
        )
        for part_type, payload in (("text", text_payload), ("compaction", marker_payload)):
            await store.append_part(
                RawMessagePart(
                    id=make_id("part"),
                    message_id=summary_msg_id,
                    session_id=sid,
                    part_type=part_type,
                    content=payload,
                    token_estimate=5,
                )
            )

        fetched = await dag.get_node_by_id(summary_msg_id)
        assert fetched is not None
        assert fetched.content == ""
        assert fetched.compaction_level == 1

        await store.close()

    async def test_get_node_by_id_pre_phase3_second_summary(self, config, pool):
        """get_node_by_id fallback for pre-Phase-3 node when it is not the first summary."""
        import time as time_mod

        from mnesis.models.message import Message
        from mnesis.session import make_id
        from mnesis.store.immutable import ImmutableStore, RawMessagePart
        from mnesis.store.summary_dag import SummaryDAGStore

        store = ImmutableStore(config.store, pool=pool)
        await store.initialize()
        dag = SummaryDAGStore(store)

        sid = "sess_pre_phase3_02"
        await store.create_session(sid, model_id="gpt-4o")

        base_ts = int(time_mod.time() * 1000)

        # Insert a non-summary message
        non_sum = make_message(sid, msg_id="pre_p3_msg_02a")
        await store.append_message(non_sum)

        # First pre-Phase-3 summary (no summary_nodes row)
        sum1_id = make_id("msg")
        sum1 = Message(
            id=sum1_id,
            session_id=sid,
            role="assistant",
            created_at=base_ts + 100,
            is_summary=True,
        )
        await store.append_message(sum1)
        await store.append_part(
            RawMessagePart(
                id=make_id("part"),
                message_id=sum1_id,
                session_id=sid,
                part_type="text",
                content=json.dumps({"type": "text", "text": "first summary"}),
                token_estimate=8,
            )
        )

        # Another non-summary message after first summary
        non_sum2 = make_message(sid, msg_id="pre_p3_msg_02b")
        await store.append_message(non_sum2)

        # Second pre-Phase-3 summary (no summary_nodes row)
        sum2_id = make_id("msg")
        sum2 = Message(
            id=sum2_id,
            session_id=sid,
            role="assistant",
            created_at=base_ts + 300,
            is_summary=True,
        )
        await store.append_message(sum2)
        await store.append_part(
            RawMessagePart(
                id=make_id("part"),
                message_id=sum2_id,
                session_id=sid,
                part_type="text",
                content=json.dumps({"type": "text", "text": "second summary"}),
                token_estimate=9,
            )
        )

        # Fetch second summary — summary_index > 0, exercises the else branch
        fetched = await dag.get_node_by_id(sum2_id)
        assert fetched is not None
        assert fetched.content == "second summary"
        assert fetched.parent_node_ids == []

        await store.close()

    async def test_get_coverage_gaps_with_messages_after_summary(
        self, session_id, store, dag_store
    ):
        """get_coverage_gaps returns a span for messages added after the latest summary."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        # Add a non-summary message before the summary
        msg_before = make_message(session_id, msg_id="msg_before_cg_01")
        await store.append_message(msg_before)

        await asyncio.sleep(0.01)

        # Insert a summary node
        node = SummaryNode(
            id="node_cg_01",
            session_id=session_id,
            kind="leaf",
            span_start_message_id=msg_before.id,
            span_end_message_id=msg_before.id,
            content="summary",
            token_count=5,
        )
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        await asyncio.sleep(0.01)

        # Add a non-summary message AFTER the summary
        msg_after = make_message(session_id, msg_id="msg_after_cg_01")
        await store.append_message(msg_after)

        gaps = await dag_store.get_coverage_gaps(session_id)
        assert len(gaps) == 1
        assert gaps[0].start_message_id == msg_after.id
        assert gaps[0].end_message_id == msg_after.id
        assert gaps[0].message_count == 1

    async def test_get_active_nodes_in_memory_superseded_guard(self, session_id, store, dag_store):
        """Directly inject into _superseded_ids to cover the in-memory guard on line 91."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        node = SummaryNode(
            id="node_guard_01",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_a",
            span_end_message_id="sm_b",
            content="text",
            token_count=10,
        )
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        # Inject directly into _superseded_ids without updating the DB.
        # This simulates the "concurrent write guard" path in get_active_nodes.
        dag_store._superseded_ids.add("node_guard_01")

        # The node is superseded=0 in DB, so _query_summary_nodes returns it,
        # but the in-memory check at line 91 skips it.
        active = await dag_store.get_active_nodes(session_id)
        assert not any(n.id == "node_guard_01" for n in active)

    async def test_get_latest_node_in_memory_guard_scan_backwards(
        self, session_id, store, dag_store
    ):
        """Directly inject into _superseded_ids to trigger backwards scan in get_latest_node."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        node_a = SummaryNode(
            id="node_guard_scan_a",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_a",
            span_end_message_id="sm_b",
            content="older",
            token_count=10,
        )
        await dag_store.insert_node(node_a, id_generator=lambda: make_id("part"))

        await asyncio.sleep(0.01)

        node_b = SummaryNode(
            id="node_guard_scan_b",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_c",
            span_end_message_id="sm_d",
            content="newer",
            token_count=20,
        )
        await dag_store.insert_node(node_b, id_generator=lambda: make_id("part"))

        # Inject directly — node_b is superseded=0 in DB so query returns it,
        # but the in-memory check at line 115 triggers the backwards scan.
        dag_store._superseded_ids.add("node_guard_scan_b")

        latest = await dag_store.get_latest_node(session_id)
        assert latest is not None
        assert latest.id == "node_guard_scan_a"

    async def test_get_latest_node_all_in_memory_guard_returns_none(
        self, session_id, store, dag_store
    ):
        """Backwards scan in get_latest_node returns None when all are in _superseded_ids."""
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        node = SummaryNode(
            id="node_guard_all_a",
            session_id=session_id,
            kind="leaf",
            span_start_message_id="sm_a",
            span_end_message_id="sm_b",
            content="text",
            token_count=10,
        )
        await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        # All nodes in _superseded_ids in-memory while DB still has superseded=0.
        dag_store._superseded_ids.add("node_guard_all_a")

        # Backwards scan exhausts all rows → for-else returns None (line 122).
        latest = await dag_store.get_latest_node(session_id)
        assert latest is None


class TestImmutableStoreCoverageGaps:
    """Additional tests targeting uncovered branches in store/immutable.py."""

    # ── conn_or_raise ─────────────────────────────────────────────────────────

    async def test_conn_or_raise_before_initialize(self, config):
        """_conn_or_raise() raises MnesisStoreError before initialize() is called."""
        from mnesis.store.immutable import ImmutableStore, MnesisStoreError

        store = ImmutableStore(config.store)
        with pytest.raises(MnesisStoreError, match="not initialized"):
            store._conn_or_raise()

    # ── wal_mode error path ───────────────────────────────────────────────────

    async def test_initialize_closes_conn_on_pragma_error(self, tmp_path):
        """initialize() closes the connection if a PRAGMA raises after connect.

        Covers lines 240-242: the except-block that closes and re-raises when
        the WAL pragma (or other post-connect setup) fails.
        """
        from mnesis.models.config import StoreConfig
        from mnesis.store.immutable import ImmutableStore

        cfg = StoreConfig(db_path=str(tmp_path / "wal_err.db"), wal_mode=True)
        store = ImmutableStore(cfg)

        # Patch aiosqlite.connect to return a connection whose execute raises
        import unittest.mock as mock

        fake_conn = mock.MagicMock()
        fake_conn.row_factory = None
        fake_conn.execute = mock.AsyncMock(side_effect=RuntimeError("simulated WAL failure"))
        fake_conn.close = mock.AsyncMock()
        fake_conn.__aenter__ = mock.AsyncMock(return_value=fake_conn)
        fake_conn.__aexit__ = mock.AsyncMock(return_value=False)

        with mock.patch("aiosqlite.connect", new=mock.AsyncMock(return_value=fake_conn)):
            with pytest.raises(RuntimeError, match="simulated WAL failure"):
                await store.initialize()

        fake_conn.close.assert_awaited_once()

    # ── append_message skips context_items for summary messages ──────────────

    async def test_append_summary_message_skips_context_items(self, session_id, store):
        """Summary messages are NOT inserted into context_items (line 508: if not is_summary).

        The compaction engine inserts context_items for summary nodes separately.
        This test verifies the guard works — summary rows do not appear in context_items.
        """
        from tests.conftest import make_message

        summary = make_message(
            session_id, role="assistant", msg_id="msg_no_ci_001", is_summary=True
        )
        await store.append_message(summary)

        items = await store.get_context_items(session_id)
        item_ids = [item_id for _, item_id in items]
        assert "msg_no_ci_001" not in item_ids

    # ── append_part: FK error ─────────────────────────────────────────────────

    async def test_append_part_unknown_message_raises(self, session_id, store):
        """append_part() raises MessageNotFoundError for unknown message_id.

        Covers line 578: the FK integrity error is translated to MessageNotFoundError.
        """
        from mnesis.store.immutable import MessageNotFoundError
        from tests.conftest import make_raw_part

        part = make_raw_part("msg_nonexistent_000", session_id, part_id="part_fk_001")
        with pytest.raises(MessageNotFoundError):
            await store.append_part(part)

    # ── update_part_status: output / error_message paths ─────────────────────

    async def test_update_part_status_output_field(self, session_id, store):
        """update_part_status() merges output into the content JSON (lines 629-643)."""
        from tests.conftest import make_message, make_raw_part

        msg = make_message(session_id, role="assistant", msg_id="msg_out_001")
        await store.append_message(msg)
        part = make_raw_part(
            "msg_out_001",
            session_id,
            part_type="tool",
            part_id="part_out_001",
            tool_call_id="call_out",
            tool_name="run_code",
            tool_state="running",
        )
        await store.append_part(part)

        await store.update_part_status("part_out_001", output="result data here")
        parts = await store.get_parts("msg_out_001")
        assert len(parts) == 1
        assert parts[0].compacted_at is None
        content = json.loads(parts[0].content)
        assert content["output"] == "result data here"

    async def test_update_part_status_error_message_field(self, session_id, store):
        """update_part_status() merges error_message into content JSON."""
        from tests.conftest import make_message, make_raw_part

        msg = make_message(session_id, role="assistant", msg_id="msg_err_001")
        await store.append_message(msg)
        part = make_raw_part(
            "msg_err_001",
            session_id,
            part_type="tool",
            part_id="part_err_001",
            tool_call_id="call_err",
            tool_name="bad_tool",
            tool_state="running",
        )
        await store.append_part(part)

        await store.update_part_status("part_err_001", error_message="something failed")
        parts = await store.get_parts("msg_err_001")
        assert len(parts) == 1
        content = json.loads(parts[0].content)
        assert content["error_message"] == "something failed"
        assert content["status"]["state"] == "running"

    async def test_update_part_status_lone_surrogate_is_escaped(self, session_id, store):
        """Lone surrogates in output/error_message are stored escaped, not rejected."""
        from tests.conftest import make_message, make_raw_part

        msg = make_message(session_id, role="assistant", msg_id="msg_sur_001")
        await store.append_message(msg)
        await store.append_part(
            make_raw_part("msg_sur_001", session_id, part_type="tool", part_id="part_sur_001")
        )

        await store.update_part_status(
            "part_sur_001", output="a\ud800b", error_message='q"\né\udc00'
        )
        content = json.loads((await store.get_parts("msg_sur_001"))[0].content)
        assert content["output"] == "a\ud800b"
        assert content["error_message"] == 'q"\né\udc00'

    @pytest.mark.parametrize("raw", ["[1, 2]", '"text"', "null", "42"])
    async def test_update_part_status_non_object_content_raises(self, session_id, store, raw):
        """Merging into non-object content raises (as before) and changes nothing."""
        from tests.conftest import make_message, make_raw_part

        msg = make_message(session_id, role="assistant", msg_id="msg_nonobj_001")
        await store.append_message(msg)
        part = make_raw_part("msg_nonobj_001", session_id, part_id="part_nonobj_001")
        part.content = raw
        await store.append_part(part)

        with pytest.raises(TypeError):
            await store.update_part_status("part_nonobj_001", output="x", tool_state="done")
        stored = (await store.get_parts("msg_nonobj_001"))[0]
        assert stored.content == raw
        assert stored.tool_state != "done"

    async def test_update_part_status_missing_part_with_output_raises_not_found(self, store):
        with pytest.raises(PartNotFoundError):
            await store.update_part_status("part_missing", output="x")

    async def test_update_part_status_no_fields_is_noop(self, session_id, store):
        """update_part_status() with no kwargs is a no-op (line 646: early return)."""
        from tests.conftest import make_message, make_raw_part

        msg = make_message(session_id, role="assistant", msg_id="msg_noop_001")
        await store.append_message(msg)
        part = make_raw_part("msg_noop_001", session_id, part_id="part_noop_001")
        await store.append_part(part)

        # Should not raise; nothing to update
        await store.update_part_status("part_noop_001")

    async def test_update_part_status_part_not_found_content_path(self, store):
        """update_part_status() raises PartNotFoundError when content RMW finds no row.

        Covers line 636: the row is None check inside the content RMW block.
        """
        from mnesis.store.immutable import PartNotFoundError

        with pytest.raises(PartNotFoundError):
            await store.update_part_status(
                "part_does_not_exist",
                output="will never land",
            )

    # ── get_messages_with_parts_by_ids ────────────────────────────────────────

    async def test_get_messages_with_parts_by_ids_returns_in_order(self, session_id, store):
        """get_messages_with_parts_by_ids() returns messages in caller-specified order."""
        from tests.conftest import make_message, make_raw_part

        ids = []
        for i in range(3):
            msg_id = f"msg_byids_{i:03d}"
            msg = make_message(session_id, msg_id=msg_id)
            await store.append_message(msg)
            part = make_raw_part(msg_id, session_id, part_id=f"part_byids_{i:03d}")
            await store.append_part(part)
            ids.append(msg_id)

        # Request in reverse order
        reversed_ids = list(reversed(ids))
        results = await store.get_messages_with_parts_by_ids(reversed_ids)
        assert [r.id for r in results] == reversed_ids

    async def test_get_messages_with_parts_by_ids_empty_list(self, store):
        """get_messages_with_parts_by_ids([]) returns empty list immediately."""
        results = await store.get_messages_with_parts_by_ids([])
        assert results == []

    async def test_get_messages_with_parts_by_ids_skips_missing(self, session_id, store):
        """get_messages_with_parts_by_ids() silently skips IDs not in the DB."""
        from tests.conftest import make_message

        msg = make_message(session_id, msg_id="msg_skip_001")
        await store.append_message(msg)

        results = await store.get_messages_with_parts_by_ids(["msg_skip_001", "msg_does_not_exist"])
        # Only the existing message is returned
        assert len(results) == 1
        assert results[0].id == "msg_skip_001"

    # ── get_file_reference_by_path ────────────────────────────────────────────

    async def test_get_file_reference_by_path_returns_latest(self, store):
        """get_file_reference_by_path() returns the most recently stored ref for a path."""
        from mnesis.models.summary import FileReference

        ref1 = FileReference(
            content_id="hash_v1",
            path="/shared/path.py",
            file_type="python",
            token_count=100,
            exploration_summary="Version 1",
            created_at=1000,
        )
        await store.store_file_reference(ref1)

        ref2 = FileReference(
            content_id="hash_v2",
            path="/shared/path.py",
            file_type="python",
            token_count=200,
            exploration_summary="Version 2",
            created_at=2000,
        )
        await store.store_file_reference(ref2)

        result = await store.get_file_reference_by_path("/shared/path.py")
        assert result is not None
        # ref2 has the higher created_at so it must be returned as the latest
        assert result.content_id == "hash_v2"
        assert result.exploration_summary == "Version 2"

    async def test_get_file_reference_by_path_not_found(self, store):
        """get_file_reference_by_path() returns None for unknown path."""
        result = await store.get_file_reference_by_path("/nonexistent/path.py")
        assert result is None

    # ── swap_context_items edge cases ─────────────────────────────────────────

    async def test_swap_context_items_empty_remove_list_is_noop(self, session_id, store):
        """swap_context_items() with empty remove_item_ids is a no-op (line 992)."""
        from tests.conftest import make_message

        msg = make_message(session_id, msg_id="msg_swap_noop_001")
        await store.append_message(msg)

        # Should not raise, nothing should change
        await store.swap_context_items(session_id, [], "summary_x")
        items = await store.get_context_items(session_id)
        assert any(item_id == "msg_swap_noop_001" for _, item_id in items)

    async def test_swap_context_items_stale_ids_is_noop(self, session_id, store):
        """swap_context_items() with IDs not in context_items is a no-op (lines 1013-1016).

        When MIN(position) is NULL because none of the remove_item_ids exist,
        the function returns without inserting a summary row.
        """
        from tests.conftest import make_message

        msg = make_message(session_id, msg_id="msg_swap_real_001")
        await store.append_message(msg)

        # Pass a stale ID that doesn't exist in context_items
        await store.swap_context_items(session_id, ["stale_id_xyz"], "summary_y")

        items = await store.get_context_items(session_id)
        item_ids = [item_id for _, item_id in items]
        # Original item untouched; no summary inserted
        assert "msg_swap_real_001" in item_ids
        assert "summary_y" not in item_ids

    async def test_swap_context_items_replaces_with_summary(self, session_id, store):
        """swap_context_items() atomically removes messages and inserts summary item."""
        from tests.conftest import make_message

        msg1 = make_message(session_id, msg_id="msg_swap_a_001")
        msg2 = make_message(session_id, msg_id="msg_swap_a_002")
        msg3 = make_message(session_id, msg_id="msg_swap_a_003")
        for m in [msg1, msg2, msg3]:
            await store.append_message(m)

        await store.swap_context_items(
            session_id,
            ["msg_swap_a_001", "msg_swap_a_002"],
            "sum_node_001",
        )

        items = await store.get_context_items(session_id)
        item_ids = [item_id for _, item_id in items]
        # Compacted messages removed; summary inserted; msg3 retained
        assert "msg_swap_a_001" not in item_ids
        assert "msg_swap_a_002" not in item_ids
        assert "sum_node_001" in item_ids
        assert "msg_swap_a_003" in item_ids

    # ── Session loading: empty-string model_id edge case ─────────────────────

    async def test_get_session_with_empty_model_id(self, store):
        """Sessions created with empty-string model_id are loadable via get_session()."""
        conn = store._conn_or_raise()
        await conn.execute(
            "INSERT INTO sessions (id, parent_id, created_at, updated_at, model_id, "
            "provider_id, agent, is_active) VALUES (?, NULL, 1, 1, '', '', 'default', 1)",
            ("sess_null_model",),
        )
        await conn.commit()

        session = await store.get_session("sess_null_model")
        assert session.id == "sess_null_model"
        assert session.model_id == ""

    # ── Concurrent writes via StorePool ──────────────────────────────────────

    async def test_concurrent_writes_via_store_pool(self, config, pool):
        """Two ImmutableStore instances sharing the same StorePool serialize writes.

        Verifies that concurrent appends via a shared connection pool do not
        produce 'database is locked' errors (StorePool invariant).
        """
        from mnesis.store.immutable import ImmutableStore
        from tests.conftest import make_message

        store_a = ImmutableStore(config.store, pool=pool)
        store_b = ImmutableStore(config.store, pool=pool)
        await store_a.initialize()
        await store_b.initialize()

        try:
            # Create a single session used by both stores
            sid = "sess_concurrent_001"
            await store_a.create_session(sid, model_id="gpt-4o")

            # Concurrently append messages from both store references
            async def _append_from_a(i: int) -> None:
                msg = make_message(sid, msg_id=f"msg_conc_a_{i:03d}")
                await store_a.append_message(msg)

            async def _append_from_b(i: int) -> None:
                msg = make_message(sid, msg_id=f"msg_conc_b_{i:03d}")
                await store_b.append_message(msg)

            tasks = [
                *[_append_from_a(i) for i in range(5)],
                *[_append_from_b(i) for i in range(5)],
            ]
            await asyncio.gather(*tasks)

            messages = await store_a.get_messages(sid)
            assert len(messages) == 10
        finally:
            await store_a.close()
            await store_b.close()

    # ── list_sessions with parent_id filter ──────────────────────────────────

    async def test_list_sessions_with_parent_id(self, store):
        """list_sessions(parent_id=...) filters to child sessions only."""
        await store.create_session("sess_parent_001")
        await store.create_session("sess_child_001", parent_id="sess_parent_001")
        await store.create_session("sess_child_002", parent_id="sess_parent_001")
        await store.create_session("sess_other_001")

        children = await store.list_sessions(parent_id="sess_parent_001")
        child_ids = {s.id for s in children}
        assert "sess_child_001" in child_ids
        assert "sess_child_002" in child_ids
        assert "sess_other_001" not in child_ids

    async def test_list_sessions_parent_id_respects_active_only(self, store):
        """active_only still excludes soft-deleted children when parent_id is given."""
        await store.create_session("sess_parent_ao")
        await store.create_session("sess_child_live", parent_id="sess_parent_ao")
        await store.create_session("sess_child_dead", parent_id="sess_parent_ao")
        await store.soft_delete_session("sess_child_dead")

        active = await store.list_sessions(parent_id="sess_parent_ao")
        assert {s.id for s in active} == {"sess_child_live"}

        everything = await store.list_sessions(parent_id="sess_parent_ao", active_only=False)
        assert {s.id for s in everything} == {"sess_child_live", "sess_child_dead"}

    # ── get_messages_with_parts_by_ids: no part rows ─────────────────────────

    async def test_get_messages_with_parts_by_ids_no_parts(self, session_id, store):
        """get_messages_with_parts_by_ids() returns messages with empty parts lists."""
        from tests.conftest import make_message

        msg = make_message(session_id, msg_id="msg_noparts_001")
        await store.append_message(msg)

        results = await store.get_messages_with_parts_by_ids(["msg_noparts_001"])
        assert len(results) == 1
        assert results[0].parts == []


class TestStoreRaceAndBoundary:
    async def test_concurrent_part_updates_to_different_fields_both_survive(
        self, session_id, store
    ):
        """output and error_message updated concurrently must not overwrite each other."""
        msg = make_message(session_id, role="assistant", msg_id="msg_race_001")
        await store.append_message(msg)
        part = make_raw_part("msg_race_001", session_id, part_type="tool", part_id="part_race_001")
        await store.append_part(part)

        for _ in range(20):
            _ = await asyncio.gather(
                store.update_part_status("part_race_001", output="the output"),
                store.update_part_status("part_race_001", error_message="the error"),
            )

        parts = await store.get_parts("msg_race_001")
        content = json.loads(parts[0].content)
        assert content["output"] == "the output"
        assert content["error_message"] == "the error"
        assert content["tool_name"] == "test_tool"  # untouched fields preserved

    async def test_update_part_output_and_error_in_one_call(self, session_id, store):
        msg = make_message(session_id, role="assistant", msg_id="msg_both_001")
        await store.append_message(msg)
        await store.append_part(
            make_raw_part("msg_both_001", session_id, part_type="tool", part_id="part_both_001")
        )
        await store.update_part_status(
            "part_both_001", tool_state="error", output='{"a": 1}', error_message="boom"
        )
        parts = await store.get_parts("msg_both_001")
        content = json.loads(parts[0].content)
        assert content["output"] == '{"a": 1}'  # stored as a string, not parsed JSON
        assert content["error_message"] == "boom"
        assert parts[0].tool_state == "error"

    async def test_update_part_content_not_found_raises(self, store):
        with pytest.raises(PartNotFoundError):
            await store.update_part_status("part_missing", output="x")

    async def test_since_message_id_from_other_session_rejected(self, session_id, store):
        from mnesis.store.immutable import MessageNotFoundError

        await store.create_session("sess_OTHER", model_id="m", agent="test")
        other = make_message("sess_OTHER", msg_id="msg_other_001")
        await store.append_message(other)
        await store.append_message(make_message(session_id, msg_id="msg_mine_001"))

        with pytest.raises(MessageNotFoundError):
            await store.get_messages(session_id, since_message_id=other.id)
        with pytest.raises(MessageNotFoundError):
            await store.get_messages(session_id, since_message_id="msg_nonexistent")

    async def test_since_message_id_includes_same_millisecond_messages(self, session_id, store):
        """Messages sharing the boundary's created_at but inserted later are returned."""
        ids = [f"msg_same_ms_{i}" for i in range(4)]
        for mid in ids:
            msg = make_message(session_id, msg_id=mid).model_copy(update={"created_at": 5_000})
            await store.append_message(msg)

        result = await store.get_messages(session_id, since_message_id=ids[1])
        assert [m.id for m in result] == ids[2:]
        assert [m.id for m in await store.get_messages(session_id)] == ids

    async def test_since_message_id_last_message_returns_empty(self, session_id, store):
        await store.append_message(make_message(session_id, msg_id="msg_last_001"))
        assert await store.get_messages(session_id, since_message_id="msg_last_001") == []


class TestSameMillisecondOrdering:
    """C1: rows sharing a created_at millisecond come back in insertion order."""

    async def test_get_last_summary_message_breaks_ties_by_insertion(self, session_id, store):
        ts = 1_700_000_000_000
        # Ids sort opposite to insertion order, so an id-ordered tie-break would fail.
        for msg_id in ("msg_z_first", "msg_m_second", "msg_a_third"):
            msg = make_message(session_id, role="assistant", msg_id=msg_id, is_summary=True)
            msg.created_at = ts
            await store.append_message(msg)

        latest = await store.get_last_summary_message(session_id)
        assert latest is not None
        assert latest.id == "msg_a_third"

    async def test_dag_nodes_ordered_by_insertion_within_a_millisecond(
        self, session_id, store, dag_store
    ):
        from mnesis.models.summary import SummaryNode
        from mnesis.session import make_id

        ts = 1_700_000_000_000
        ids = ["node_z", "node_m", "node_a"]
        for node_id in ids:
            node = SummaryNode(
                id=node_id,
                session_id=session_id,
                kind="leaf",
                span_start_message_id="s",
                span_end_message_id="e",
                content=f"content {node_id}",
                token_count=10,
                created_at=ts,
            )
            await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        active = await dag_store.get_active_nodes(session_id)
        assert [n.id for n in active] == ids
        latest = await dag_store.get_latest_node(session_id)
        assert latest is not None and latest.id == "node_a"


class TestInitializeFailureReleasesConnection:
    """C2: a schema failure on a private connection must not leak it."""

    async def test_private_connection_closed_when_schema_step_fails(self, tmp_path, monkeypatch):
        import aiosqlite

        from mnesis.models.config import StoreConfig
        from mnesis.store.immutable import ImmutableStore

        opened: list[aiosqlite.Connection] = []
        real_connect = aiosqlite.connect

        def tracking_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        async def failing_executescript(self, sql):
            raise aiosqlite.OperationalError("schema boom")

        monkeypatch.setattr(aiosqlite, "connect", tracking_connect)
        monkeypatch.setattr(aiosqlite.Connection, "executescript", failing_executescript)

        store = ImmutableStore(StoreConfig(db_path=str(tmp_path / "leak.db")))
        with pytest.raises(aiosqlite.OperationalError, match="schema boom"):
            await store.initialize()

        assert len(opened) == 1
        with pytest.raises(ValueError, match="no active connection"):
            _ = await opened[0].execute("SELECT 1")  # closed, not leaked
        await store.close()  # still a harmless no-op

    async def test_pooled_connection_is_left_to_the_pool(self, tmp_path, monkeypatch):
        import aiosqlite

        from mnesis.models.config import StoreConfig
        from mnesis.store.immutable import ImmutableStore
        from mnesis.store.pool import StorePool

        async def failing_executescript(self, sql):
            raise aiosqlite.OperationalError("schema boom")

        monkeypatch.setattr(aiosqlite.Connection, "executescript", failing_executescript)
        pool = StorePool()
        try:
            store = ImmutableStore(StoreConfig(db_path=str(tmp_path / "pooled.db")), pool=pool)
            with pytest.raises(aiosqlite.OperationalError):
                await store.initialize()
        finally:
            await pool.close_all()


class TestGetNodeByIdIsSingleNode:
    """C3: resolving one summary node must not load the whole session."""

    async def test_row_path_does_not_load_all_messages(self, session_id, store, dag_store):
        node = await _insert_leaf_node(dag_store, session_id, "node_single", content="alpha")
        for i in range(5):
            await store.append_message(make_message(session_id, msg_id=f"msg_extra_{i}"))

        calls: list[str] = []
        real = store.get_messages

        async def spy(sid, *a, **kw):
            calls.append(sid)
            return await real(sid, *a, **kw)

        store.get_messages = spy  # type: ignore[method-assign]
        loaded = await dag_store.get_node_by_id("node_single")

        assert calls == []
        assert loaded is not None
        assert loaded.id == node.id
        assert loaded.content == "alpha"
        assert loaded.kind == "leaf"
        assert loaded.model_id == node.model_id
        assert loaded.created_at == (await store.get_message("node_single")).created_at

    async def test_non_summary_and_missing_return_none(self, session_id, store, dag_store):
        msg = make_message(session_id, role="user", msg_id="msg_plain")
        await store.append_message(msg)
        assert await dag_store.get_node_by_id("msg_plain") is None
        assert await dag_store.get_node_by_id("does_not_exist") is None


class TestConnectionSerialization:
    """F2: multi-statement transactions on a shared pooled connection never interleave."""

    @staticmethod
    async def _seed(session_id, store, n=4):
        ids = []
        for i in range(n):
            msg = make_message(session_id, role="user", msg_id=f"msg_ser_{i}")
            await store.append_message(msg)
            ids.append(msg.id)
        return ids

    @staticmethod
    def _gate_insert(store, *, fail: bool = False):
        """Pause the swap between its DELETE and INSERT; optionally fail the INSERT."""
        conn = store._conn
        orig = conn.execute
        at_insert = asyncio.Event()
        release = asyncio.Event()

        def gated(sql, *args, **kwargs):
            if "INSERT INTO context_items" in sql and "'summary'" in sql:

                async def run():
                    at_insert.set()
                    await release.wait()
                    if fail:
                        raise RuntimeError("insert failed")
                    return await orig(sql, *args, **kwargs)

                return run()
            return orig(sql, *args, **kwargs)

        conn.execute = gated
        return at_insert, release, lambda: setattr(conn, "execute", orig)

    async def test_reader_never_sees_span_missing_during_swap(self, session_id, store):
        ids = await self._seed(session_id, store)
        at_insert, release, restore = self._gate_insert(store)
        try:
            swap = asyncio.create_task(store.swap_context_items(session_id, ids[:3], "sum_ser"))
            await asyncio.wait_for(at_insert.wait(), 5)  # DELETE done, INSERT pending
            reader = asyncio.create_task(store.get_context_items(session_id))
            await asyncio.sleep(0.05)
            assert not reader.done()  # blocked, not served the half-swapped state
            release.set()
            _ = await asyncio.wait({swap})
            swap.result()
            items = await reader
        finally:
            restore()
        assert items == [("summary", "sum_ser"), ("message", ids[3])]

    async def test_failed_swap_rolls_back_and_unrelated_write_is_unaffected(
        self, session_id, store, config, pool
    ):
        ids = await self._seed(session_id, store)
        other = ImmutableStore(config.store, pool=pool)
        await other.initialize()
        at_insert, release, restore = self._gate_insert(store, fail=True)
        try:
            swap = asyncio.create_task(store.swap_context_items(session_id, ids[:3], "sum_ser"))
            await asyncio.wait_for(at_insert.wait(), 5)
            unrelated = asyncio.create_task(
                other.append_message(make_message(session_id, role="user", msg_id="msg_other"))
            )
            await asyncio.sleep(0.05)
            assert not unrelated.done()  # queued behind the open swap transaction
            release.set()
            with pytest.raises(RuntimeError, match="insert failed"):
                _ = await asyncio.wait({swap})
                swap.result()
            _ = await unrelated  # commits only its own work
        finally:
            restore()
        items = await store.get_context_items(session_id)
        # The DELETE was rolled back (span intact) and the unrelated message landed.
        assert items == [("message", i) for i in [*ids, "msg_other"]]

    async def test_cancelled_swap_rolls_back(self, session_id, store):
        ids = await self._seed(session_id, store)
        at_insert, _release, restore = self._gate_insert(store)
        try:
            swap = asyncio.create_task(store.swap_context_items(session_id, ids[:3], "sum_ser"))
            await asyncio.wait_for(at_insert.wait(), 5)
            _ = swap.cancel()
            with pytest.raises(asyncio.CancelledError):
                _ = await asyncio.wait({swap})
                swap.result()
        finally:
            restore()
        # Nothing half-applied, and the lock was released for later writers.
        assert await store.get_context_items(session_id) == [("message", i) for i in ids]
        await store.append_message(make_message(session_id, role="user", msg_id="msg_after"))
        assert (await store.get_context_items(session_id))[-1] == ("message", "msg_after")

    async def test_stores_sharing_a_pool_share_one_lock(self, config, pool, store):
        other = ImmutableStore(config.store, pool=pool)
        await other.initialize()
        assert other._lock is store._lock
        assert store._lock is pool.write_lock(config.store.db_path)

    async def test_private_store_has_its_own_lock(self, tmp_path):
        from mnesis.models.config import StoreConfig

        a = ImmutableStore(StoreConfig(db_path=str(tmp_path / "a.db")))
        b = ImmutableStore(StoreConfig(db_path=str(tmp_path / "b.db")))
        assert a._lock is not b._lock

    async def test_concurrent_appends_across_stores_keep_positions_unique(
        self, session_id, store, config, pool
    ):
        other = ImmutableStore(config.store, pool=pool)
        await other.initialize()

        async def burst(s, tag):
            for i in range(15):
                await s.append_message(
                    make_message(session_id, role="user", msg_id=f"msg_{tag}_{i}")
                )

        await asyncio.gather(burst(store, "a"), burst(other, "b"))
        items = await store.get_context_items(session_id)
        assert len(items) == 30 and len({i for _, i in items}) == 30

    async def test_dag_writes_use_the_same_lock(self, session_id, store, dag_store):
        await _insert_leaf_node(dag_store, session_id, "node_a")
        await _insert_leaf_node(dag_store, session_id, "node_b")
        # Hold the lock: mark_superseded and node reads must wait for it.
        async with store._lock:
            marking = asyncio.create_task(dag_store.mark_superseded(["node_a"]))
            reading = asyncio.create_task(dag_store.get_active_nodes(session_id))
            await asyncio.sleep(0.05)
            assert not marking.done() and not reading.done()
        _ = await asyncio.wait({marking})
        marking.result()
        assert [n.id for n in await reading] in (["node_a", "node_b"], ["node_b"])
        assert [n.id for n in await dag_store.get_active_nodes(session_id)] == ["node_b"]

    async def test_single_node_lookups_use_the_same_lock(self, session_id, store, dag_store):
        _ = await _insert_leaf_node(dag_store, session_id, "node_a")

        class SpyLock:
            """Delegates to the real lock and counts acquisitions (no timing)."""

            def __init__(self, inner):
                self.inner = inner
                self.acquired = 0

            async def __aenter__(self):
                await self.inner.acquire()
                self.acquired += 1

            async def __aexit__(self, *exc):
                self.inner.release()

        spy = SpyLock(store._lock)
        store._lock = spy  # type: ignore[assignment]
        row = await dag_store._get_summary_node_row("node_a")
        assert row is not None and spy.acquired == 1
        assert await dag_store._session_has_any_summary_nodes(session_id) is True
        assert spy.acquired == 2
        node = await dag_store.get_node_by_id("node_a")
        assert node is not None and node.id == "node_a"
        assert spy.acquired >= 3  # its row lookup took the lock too

    async def test_file_reference_reads_do_not_see_uncommitted_rows(self, config, pool, store):
        """Dedup lookups skip another store's in-flight insert that then rolls back."""
        other = ImmutableStore(config.store, pool=pool)
        await other.initialize()
        ref = FileReference(
            content_id="rolled_back",
            path="/tmp/rb.py",
            file_type="python",
            token_count=100,
            exploration_summary="s",
        )
        inserted = asyncio.Event()
        seen: dict[str, object] = {}

        async def inserter():
            try:
                async with other._transaction() as conn:
                    await conn.execute(
                        "INSERT INTO file_references "
                        "(content_id, path, file_type, token_count, exploration_summary, "
                        "created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            ref.content_id,
                            ref.path,
                            ref.file_type,
                            ref.token_count,
                            ref.exploration_summary,
                            ref.created_at,
                        ),
                    )
                    inserted.set()
                    # Hold the transaction open until the readers have finished.
                    # With the lock they cannot, so this times out; without it they
                    # finish early (having seen the dirty row) and the test fails.
                    _ = await asyncio.wait({reader_task}, timeout=0.5)
                    raise RuntimeError("force rollback")
            except RuntimeError:
                pass

        async def readers():
            _ = await inserted.wait()
            seen["by_id"] = await store.get_file_reference("rolled_back")
            seen["by_path"] = await store.get_file_reference_by_path("/tmp/rb.py")

        reader_task = asyncio.create_task(readers())
        await inserter()
        _ = await reader_task
        assert seen == {"by_id": None, "by_path": None}


class TestAtomicNodeCommit:
    """Node insert + context swap + supersession commit as one transaction."""

    @staticmethod
    def _node(session_id, node_id, kind="condensed", parents=()):
        from mnesis.models.summary import SummaryNode

        return SummaryNode(
            id=node_id,
            session_id=session_id,
            kind=kind,
            span_start_message_id="a",
            span_end_message_id="b",
            content="merged",
            token_count=10,
            parent_node_ids=list(parents),
        )

    @staticmethod
    async def _setup(session_id, store, dag_store):
        from mnesis.session import make_id

        conn = store._conn
        for pos, nid in enumerate(("node_a", "node_b"), start=100):
            await _insert_leaf_node(dag_store, session_id, nid)
            await conn.execute(
                "INSERT INTO context_items (session_id, item_type, item_id, position, created_at)"
                " VALUES (?, 'summary', ?, ?, '0')",
                (session_id, nid, pos),
            )
        await conn.commit()
        return lambda: make_id("part")

    async def test_commit_swaps_and_supersedes_together(self, session_id, store, dag_store):
        gen = await self._setup(session_id, store, dag_store)
        node = self._node(session_id, "node_c", parents=["node_a", "node_b"])
        _ = await dag_store.commit_summary_node(
            node,
            id_generator=gen,
            remove_item_ids=["node_a", "node_b"],
            supersede_node_ids=["node_a", "node_b"],
        )
        assert await store.get_context_items(session_id) == [("summary", "node_c")]
        assert [n.id for n in await dag_store.get_active_nodes(session_id)] == ["node_c"]

    async def _interrupted(self, session_id, store, dag_store, *, cancel: bool):
        gen = await self._setup(session_id, store, dag_store)
        conn = store._conn
        orig = conn.execute
        at_update = asyncio.Event()

        def gated(sql, *args, **kwargs):
            if "SET superseded=1" in sql:

                async def run():
                    at_update.set()  # swap done, supersede pending
                    if cancel:
                        await asyncio.sleep(30)
                    raise RuntimeError("supersede failed")

                return run()
            return orig(sql, *args, **kwargs)

        conn.execute = gated
        node = self._node(session_id, "node_c", parents=["node_a", "node_b"])
        task = asyncio.create_task(
            dag_store.commit_summary_node(
                node,
                id_generator=gen,
                remove_item_ids=["node_a", "node_b"],
                supersede_node_ids=["node_a", "node_b"],
            )
        )
        try:
            await asyncio.wait_for(at_update.wait(), 5)
            if cancel:
                _ = task.cancel()
            _ = await asyncio.wait({task})
        finally:
            conn.execute = orig
        return task

    async def _assert_untouched(self, session_id, store, dag_store):
        assert await store.get_context_items(session_id) == [
            ("summary", "node_a"),
            ("summary", "node_b"),
        ]
        assert {n.id for n in await dag_store.get_active_nodes(session_id)} == {"node_a", "node_b"}
        assert dag_store._superseded_ids == set()
        from mnesis.store.immutable import MessageNotFoundError

        with pytest.raises(MessageNotFoundError):
            _ = await store.get_message("node_c")

    async def test_failure_after_swap_rolls_everything_back(self, session_id, store, dag_store):
        task = await self._interrupted(session_id, store, dag_store, cancel=False)
        with pytest.raises(RuntimeError, match="supersede failed"):
            _ = task.result()
        await self._assert_untouched(session_id, store, dag_store)

    async def test_cancel_between_swap_and_supersede_rolls_back(self, session_id, store, dag_store):
        task = await self._interrupted(session_id, store, dag_store, cancel=True)
        assert task.cancelled()
        await self._assert_untouched(session_id, store, dag_store)

    async def test_duplicate_node_id_raises_and_commits_nothing(self, session_id, store, dag_store):
        from mnesis.store.immutable import DuplicateIDError

        gen = await self._setup(session_id, store, dag_store)
        dup = self._node(session_id, "node_a")  # id already taken by a leaf
        with pytest.raises(DuplicateIDError):
            _ = await dag_store.commit_summary_node(
                dup, id_generator=gen, remove_item_ids=["node_b"], supersede_node_ids=["node_b"]
            )
        await self._assert_untouched(session_id, store, dag_store)

    async def test_duplicate_part_id_names_the_part_not_the_node(
        self, session_id, store, dag_store
    ):
        from mnesis.store.immutable import DuplicateIDError

        await self._setup(session_id, store, dag_store)
        await store.append_message(make_message(session_id, role="user", msg_id="msg_host"))
        await store.append_part(make_raw_part("msg_host", session_id, part_id="part_taken"))
        node = self._node(session_id, "node_new")
        with pytest.raises(DuplicateIDError) as info:
            _ = await dag_store.commit_summary_node(node, id_generator=lambda: "part_taken")
        assert info.value.record_id == "part_taken"
        assert "node_new" not in str(info.value)

    async def test_unknown_session_raises_session_not_found(self, store, dag_store):
        from mnesis.store.immutable import SessionNotFoundError

        node = self._node("sess_missing", "node_x")
        with pytest.raises(SessionNotFoundError):
            _ = await dag_store.commit_summary_node(node, id_generator=lambda: "part_x")

    async def test_existing_part_id_falls_back_to_the_first_part(self, store, dag_store):
        a = make_raw_part("m", "s", part_id="part_a")
        b = make_raw_part("m", "s", part_id="part_b")
        assert await dag_store._existing_part_id(a, b) == "part_a"


class TestMarkSupersededRollback:
    async def test_failed_transaction_leaves_nodes_visible(self, session_id, store, dag_store):
        await _insert_leaf_node(dag_store, session_id, "node_a")
        await _insert_leaf_node(dag_store, session_id, "node_b")
        conn = store._conn
        orig = conn.execute

        def boom(sql, *args, **kwargs):
            if "SET superseded=1" in sql:

                async def fail():
                    raise RuntimeError("injected")

                return fail()
            return orig(sql, *args, **kwargs)

        conn.execute = boom
        try:
            with pytest.raises(RuntimeError, match="injected"):
                await dag_store.mark_superseded(["node_a"])
        finally:
            conn.execute = orig
        assert dag_store._superseded_ids == set()
        assert {n.id for n in await dag_store.get_active_nodes(session_id)} == {"node_a", "node_b"}


async def test_store_close_twice_and_concurrently_is_a_noop(tmp_path):
    from mnesis.models.config import StoreConfig
    from mnesis.store.immutable import ImmutableStore

    store = ImmutableStore(StoreConfig(db_path=str(tmp_path / "twice.db")))
    await store.initialize()
    await asyncio.wait_for(asyncio.gather(store.close(), store.close()), timeout=5)
    await store.close()
    assert store._conn is None
