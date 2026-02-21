"""Tests for ImmutableStore and SummaryDAGStore."""

from __future__ import annotations

import asyncio
import time

import pytest

from mnesis.models.message import TokenUsage
from mnesis.models.summary import FileReference
from mnesis.store.immutable import (
    DuplicateIDError,
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

        import json

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
        import asyncio

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
