"""Tests for ImmutableStore and SummaryDAGStore."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from mnesis.models.message import Message, TextPart, TokenUsage
from mnesis.models.summary import FileReference
from mnesis.store.immutable import (
    DuplicateIDError,
    MessageNotFoundError,
    PartNotFoundError,
    SessionNotFoundError,
)
from tests.conftest import make_message, make_raw_part


class TestImmutableStore:
    async def test_create_session(self, store):
        """Creating a session returns a Session with correct fields."""
        session = await store.create_session(
            "sess_001", model_id="gpt-4o", agent="test"
        )
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
            "msg_prune_001", session_id, part_type="tool",
            part_id="part_prune_001", tool_call_id="call_001",
            tool_name="read_file", tool_state="completed"
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
            msg = make_message(session_id, role="user" if i % 2 == 0 else "assistant", msg_id=msg_id)
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
