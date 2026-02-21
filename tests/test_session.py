"""Integration tests for MnesisSession."""

from __future__ import annotations

import pytest

from mnesis.events.bus import MnesisEvent
from mnesis.models.message import TextPart


@pytest.fixture
def mock_llm_env(monkeypatch):
    """Enable mock LLM mode for send()-based session tests."""
    monkeypatch.setenv("MNESIS_MOCK_LLM", "1")


class TestMnesisSession:
    """Tests for send()-based session flows. Require MNESIS_MOCK_LLM."""

    async def test_create_returns_valid_session_id(self, tmp_path, mock_llm_env):
        """Session ID has correct ULID format with prefix."""
        from mnesis import MnesisSession

        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        )
        assert session.id.startswith("sess_")
        await session.close()

    async def test_send_appends_user_and_assistant_messages(self, tmp_path, mock_llm_env):
        """send() stores both user and assistant messages."""
        from mnesis import MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            await session.send("Hello!")
            messages = await session.messages()

        # Should have at least user + assistant
        assert len(messages) >= 2
        roles = [m.role for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    async def test_send_returns_turn_result(self, tmp_path, mock_llm_env):
        """send() returns a TurnResult with expected fields."""
        from mnesis import MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            result = await session.send("What is 2+2?")

        assert result.message_id.startswith("msg_")
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.finish_reason in ("stop", "end_turn", "error")

    async def test_send_streaming_calls_on_part(self, tmp_path, mock_llm_env):
        """on_part callback is called during streaming."""
        from mnesis import MnesisSession

        received_parts = []

        def on_part(part):
            received_parts.append(part)

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            await session.send("Hello!", on_part=on_part)

        assert len(received_parts) > 0
        assert any(isinstance(p, TextPart) for p in received_parts)

    async def test_load_restores_session(self, tmp_path, mock_llm_env):
        """load() can resume a session from the database."""
        from mnesis import MnesisSession

        db = str(tmp_path / "test.db")

        # Create and send a message
        session1 = await MnesisSession.create(model="anthropic/claude-opus-4-6", db_path=db)
        session_id = session1.id
        await session1.send("First message.")
        await session1.close()

        # Load and verify history
        session2 = await MnesisSession.load(session_id, db_path=db)
        messages = await session2.messages()
        assert len(messages) >= 2
        await session2.close()

    async def test_load_restores_system_prompt(self, tmp_path, mock_llm_env):
        """C-1: load() restores the original system_prompt, not a hardcoded fallback."""
        from mnesis import MnesisSession

        db = str(tmp_path / "test.db")
        custom_prompt = "You are a specialized coding assistant."

        session1 = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=db,
            system_prompt=custom_prompt,
        )
        session_id = session1.id
        await session1.close()

        session2 = await MnesisSession.load(session_id, db_path=db)
        assert session2._system_prompt == custom_prompt
        await session2.close()

    async def test_load_raises_when_model_id_missing(self, tmp_path):
        """M-9: load() raises ValueError when the stored session has no model_id."""
        from mnesis.models.config import MnesisConfig, StoreConfig
        from mnesis.store.immutable import ImmutableStore

        db = str(tmp_path / "test.db")
        cfg = MnesisConfig()
        cfg = cfg.model_copy(update={"store": StoreConfig(db_path=db)})
        store = ImmutableStore(cfg.store)
        await store.initialize()

        # Insert a session row with an empty model_id to simulate a legacy/corrupted record.
        await store._conn_or_raise().execute(
            "INSERT INTO sessions (id, parent_id, created_at, updated_at, model_id, "
            "provider_id, agent, is_active) VALUES (?, NULL, 1, 1, '', '', 'default', 1)",
            ("sess_no_model_id",),
        )
        await store._conn_or_raise().commit()
        await store.close()

        import pytest

        from mnesis import MnesisSession

        with pytest.raises(ValueError, match="has no stored model_id"):
            await MnesisSession.load("sess_no_model_id", db_path=db)

    async def test_context_manager_closes_on_exception(self, tmp_path, mock_llm_env):
        """Session is closed even when send() raises."""
        from mnesis import MnesisSession

        closed = []

        class TestSession(MnesisSession):
            async def close(self):
                closed.append(True)
                await super().close()

        # We can't easily inject TestSession, so just verify close() is idempotent
        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        )
        await session.close()
        # Calling close again should not raise
        await session.close()

    async def test_event_bus_session_created(self, tmp_path, mock_llm_env):
        """SESSION_CREATED event is published on create()."""
        from mnesis import MnesisSession

        events = []

        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        )
        session.event_bus.subscribe_all(lambda e, p: events.append(e))
        await session.send("Hi")
        await session.close()

        # Events after subscribe should include message events
        assert MnesisEvent.MESSAGE_CREATED in events

    async def test_messages_returns_full_history(self, tmp_path, mock_llm_env):
        """messages() includes all turns in chronological order."""
        from mnesis import MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            await session.send("First")
            await session.send("Second")
            await session.send("Third")
            msgs = await session.messages()

        # 3 user + 3 assistant = 6 messages
        assert len(msgs) >= 6

    async def test_token_usage_accumulates(self, tmp_path, mock_llm_env):
        """token_usage increases with each send() call."""
        from mnesis import MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            await session.send("Hello")
            usage_after_1 = session.token_usage.effective_total()
            await session.send("World")
            usage_after_2 = session.token_usage.effective_total()

        assert usage_after_2 > usage_after_1

    async def test_manual_compact_returns_result(self, tmp_path, mock_llm_env):
        """compact() runs synchronously and returns CompactionResult."""
        from mnesis import CompactionResult, MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            await session.send("Message one")
            await session.send("Message two")
            result = await session.compact()

        assert isinstance(result, CompactionResult)
        assert result.session_id != ""


class TestMnesisSessionRecord:
    """Tests for record() — no LLM calls, no MNESIS_MOCK_LLM required."""

    async def test_record_persists_user_and_assistant_messages(self, tmp_path):
        """record() stores both messages without making an LLM call."""
        from mnesis import MnesisSession, RecordResult

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            result = await session.record(
                user_message="What is the capital of France?",
                assistant_response="The capital of France is Paris.",
            )
            messages = await session.messages()

        assert isinstance(result, RecordResult)
        assert result.user_message_id.startswith("msg_")
        assert result.assistant_message_id.startswith("msg_")
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].text_content() == "The capital of France is Paris."

    async def test_record_accepts_explicit_token_usage(self, tmp_path):
        """record() uses provided token usage and accumulates it."""
        from mnesis import MnesisSession, TokenUsage

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            await session.record(
                user_message="Hello",
                assistant_response="Hi there!",
                tokens=TokenUsage(input=10, output=5),
            )
            usage = session.token_usage

        assert usage.input == 10
        assert usage.output == 5

    async def test_record_estimates_tokens_when_not_provided(self, tmp_path):
        """record() estimates token usage when tokens arg is omitted."""
        from mnesis import MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            result = await session.record(
                user_message="Tell me about the moon.",
                assistant_response="The moon is Earth's only natural satellite.",
            )

        assert result.tokens.input > 0
        assert result.tokens.output > 0

    async def test_record_publishes_message_created_events(self, tmp_path):
        """record() publishes MESSAGE_CREATED for both user and assistant."""
        from mnesis import MnesisSession
        from mnesis.events.bus import MnesisEvent

        events = []

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            session.event_bus.subscribe_all(lambda e, p: events.append((e, p)))
            await session.record(
                user_message="Ping",
                assistant_response="Pong",
            )

        message_events = [e for e, _ in events if e == MnesisEvent.MESSAGE_CREATED]
        assert len(message_events) == 2

    async def test_record_accepts_message_parts(self, tmp_path):
        """record() accepts list[MessagePart] for both arguments."""
        from mnesis import MnesisSession
        from mnesis.models.message import TextPart

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            result = await session.record(
                user_message=[TextPart(text="Hello from parts")],
                assistant_response=[TextPart(text="Reply from parts")],
            )

        assert result.user_message_id.startswith("msg_")
        assert result.assistant_message_id.startswith("msg_")

    async def test_record_tool_part_token_estimate_covers_input_output_error(self, tmp_path):
        """record() without explicit tokens accounts for ToolPart input/output/error_message.

        Regression test: previously token_estimate was computed from part.output only,
        and the session-level output token count was derived from text-only content.
        A ToolPart-only response therefore contributed 0 to tokens.output, preventing
        auto-compaction from triggering on tool-heavy sessions.
        """
        from mnesis import MnesisSession
        from mnesis.models.message import ToolPart, ToolStatus

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            # Completed tool with non-empty input and output
            completed = ToolPart(
                tool_name="read_file",
                tool_call_id="call_ok",
                input={"path": "/tmp/big_file.txt"},
                output="line1\nline2\nline3",
                status=ToolStatus(state="completed"),
            )
            result_ok = await session.record(
                user_message="Read the file.",
                assistant_response=[completed],
            )

            # Error tool with error_message but no output
            errored = ToolPart(
                tool_name="read_file",
                tool_call_id="call_err",
                input={"path": "/tmp/missing.txt"},
                error_message="FileNotFoundError: /tmp/missing.txt",
                status=ToolStatus(state="error"),
            )
            result_err = await session.record(
                user_message="Read the missing file.",
                assistant_response=[errored],
            )

        # Both turns must have non-zero output token estimates
        assert result_ok.tokens.output > 0
        assert result_err.tokens.output > 0

    async def test_record_tool_part_persists_tool_call_id(self, tmp_path):
        """record() with ToolPart writes tool_call_id/tool_name/tool_state to the DB row.

        Regression test: before the fix, record() built RawMessagePart without setting
        these fields, so the pruner's part_id_map lookup always failed and tool output
        tombstoning was silently broken for the BYO-LLM pattern.
        """
        from mnesis import MnesisSession
        from mnesis.models.message import ToolPart, ToolStatus

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            tool = ToolPart(
                tool_name="list_directory",
                tool_call_id="call_test_001",
                input={"path": "/tmp"},
                output="file1.txt  file2.txt",
                status=ToolStatus(state="completed"),
            )
            result = await session.record(
                user_message="List /tmp",
                assistant_response=[tool],
            )

            raw_parts = await session._store.get_parts(result.assistant_message_id)

        tool_raw = [p for p in raw_parts if p.part_type == "tool"]
        assert len(tool_raw) == 1
        assert tool_raw[0].tool_call_id == "call_test_001"
        assert tool_raw[0].tool_name == "list_directory"
        assert tool_raw[0].tool_state == "completed"

    async def test_record_tool_parts_are_prunable(self, tmp_path, monkeypatch):
        """ToolParts recorded via record() can be tombstoned by the pruner.

        Regression test: tombstoning requires tool_call_id on the raw DB row.
        Without the session.record() fix, no tombstones would ever be created.
        """
        from mnesis import MnesisConfig, MnesisSession
        from mnesis.models.config import CompactionConfig
        from mnesis.models.message import TextPart, ToolPart, ToolStatus

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        config = MnesisConfig(
            compaction=CompactionConfig(
                prune=True,
                prune_protect_tokens=50,
                prune_minimum_tokens=10,
            )
        )

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
            config=config,
        ) as session:
            for i in range(4):
                tool = ToolPart(
                    tool_name="read_file",
                    tool_call_id=f"call_{i:03d}",
                    input={"path": f"/tmp/file_{i}.txt"},
                    output=f"Content of file {i}: " + "x" * 200,
                    status=ToolStatus(state="completed"),
                )
                await session.record(
                    user_message=f"Read file {i}.",
                    assistant_response=[
                        tool,
                        TextPart(text=f"File {i} read successfully."),
                    ],
                )

            await session.compact()
            messages_after = await session.messages()

        tombstoned = sum(
            1
            for mwp in messages_after
            for part in mwp.parts
            if isinstance(part, ToolPart) and part.compacted_at is not None
        )
        assert tombstoned > 0, "Expected at least one tool output tombstoned after compaction"


class TestPublicAPIContracts:
    """Tests for public API correctness: exception exports, finish_reason typing."""

    def test_session_not_found_error_importable_from_mnesis(self):
        """H-1: SessionNotFoundError must be importable from the top-level mnesis package."""
        from mnesis import SessionNotFoundError
        from mnesis.store.immutable import SessionNotFoundError as _StoreImpl

        assert SessionNotFoundError is _StoreImpl

    def test_mnesis_store_error_importable_from_mnesis(self):
        """H-1: MnesisStoreError must be importable from the top-level mnesis package."""
        from mnesis import MnesisStoreError
        from mnesis.store.immutable import MnesisStoreError as _StoreImpl

        assert MnesisStoreError is _StoreImpl

    def test_session_not_found_error_in_all(self):
        """H-1: Both exceptions must appear in mnesis.__all__."""
        from mnesis import __all__ as mnesis_all

        assert "SessionNotFoundError" in mnesis_all
        assert "MnesisStoreError" in mnesis_all

    def test_session_not_found_error_is_mnesis_store_error_subclass(self):
        """SessionNotFoundError must be a subclass of MnesisStoreError for catch-hierarchy."""
        from mnesis import MnesisStoreError, SessionNotFoundError

        assert issubclass(SessionNotFoundError, MnesisStoreError)

    def test_turn_result_finish_reason_enum_annotation(self):
        """M-8: TurnResult.finish_reason must use FinishReason | str, not plain str."""
        import types
        from typing import Union, get_args, get_origin, get_type_hints

        from mnesis.models.message import FinishReason, TurnResult

        hints = get_type_hints(TurnResult)
        fr_type = hints["finish_reason"]
        # `FinishReason | str` uses the PEP 604 union syntax; get_origin returns
        # types.UnionType on Python 3.10-3.13 and typing.Union on older versions.
        origin = get_origin(fr_type)
        assert origin in (Union, types.UnionType), f"Expected Union, got {origin}"
        args = get_args(fr_type)
        assert FinishReason in args, f"FinishReason not in Union args: {args}"
        # Verify enum members carry the expected string values
        assert FinishReason.ERROR == "error"
        assert FinishReason.STOP == "stop"
        assert FinishReason.MAX_TOKENS == "max_tokens"

    def test_finish_reason_exported_from_top_level(self):
        """M-8: FinishReason must be importable from the top-level mnesis package."""
        from mnesis import FinishReason

        assert FinishReason.STOP == "stop"
        assert issubclass(FinishReason, str)


class TestSessionOpen:
    """Tests for M-1: MnesisSession.open() async context manager factory."""

    async def test_open_yields_session_and_closes(self, tmp_path):
        """open() yields a working session and closes it on exit."""
        from mnesis import MnesisSession

        async with MnesisSession.open(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            assert session.id.startswith("sess_")

    async def test_open_closes_on_exception(self, tmp_path):
        """open() calls close() even when the body raises."""
        from mnesis import MnesisSession

        closed_sessions: list[str] = []
        original_close = MnesisSession.close

        async def tracking_close(self):  # type: ignore[override]
            closed_sessions.append(self.id)
            await original_close(self)

        MnesisSession.close = tracking_close  # type: ignore[method-assign]
        try:
            with pytest.raises(ValueError, match="boom"):
                async with MnesisSession.open(
                    model="anthropic/claude-opus-4-6",
                    db_path=str(tmp_path / "test.db"),
                ) as session:
                    assert session.id.startswith("sess_")
                    raise ValueError("boom")
            assert len(closed_sessions) == 1
            assert closed_sessions[0].startswith("sess_")
        finally:
            MnesisSession.close = original_close  # type: ignore[method-assign]


class TestContextForNextTurn:
    """Tests for M-7: session.context_for_next_turn()."""

    async def test_returns_list_of_role_content_dicts(self, tmp_path):
        """context_for_next_turn() returns a list[dict] with role/content keys."""
        from mnesis import MnesisSession

        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        )
        try:
            # Empty context — no messages yet
            ctx = await session.context_for_next_turn()
            assert isinstance(ctx, list)

            # After recording a turn, messages appear in context
            await session.record(
                user_message="Hello",
                assistant_response="Hi there",
            )
            ctx = await session.context_for_next_turn()
            assert len(ctx) >= 1
            for item in ctx:
                assert "role" in item
                assert "content" in item
                assert item["role"] in ("user", "assistant")
        finally:
            await session.close()

    async def test_system_prompt_override(self, tmp_path):
        """context_for_next_turn() accepts an optional system_prompt override."""
        from mnesis import MnesisSession

        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            system_prompt="Default prompt.",
            db_path=str(tmp_path / "test.db"),
        )
        try:
            # Should not raise regardless of override value
            ctx = await session.context_for_next_turn(system_prompt="Override prompt.")
            assert isinstance(ctx, list)
        finally:
            await session.close()


class TestCompactionInProgress:
    """Tests for L-9: session.compaction_in_progress property."""

    async def test_compaction_in_progress_false_at_start(self, tmp_path):
        """compaction_in_progress is False when no compaction task is running."""
        from mnesis import MnesisSession

        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        )
        try:
            assert session.compaction_in_progress is False
        finally:
            await session.close()
