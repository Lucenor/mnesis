"""Integration tests for MnesisSession."""

from __future__ import annotations

import asyncio

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
        """The async context manager closes the session when the body raises."""
        from mnesis import MnesisSession
        from mnesis.store.immutable import MnesisStoreError

        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        )

        async def body_that_fails() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            async with session:
                _ = await body_that_fails()

        # The context manager closed the session: the store is no longer usable.
        with pytest.raises(MnesisStoreError):
            _ = await session.messages()

        # close() is idempotent, so a second call after the context manager
        # already closed the session must not raise.
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

            async def _body_raises() -> None:
                async with MnesisSession.open(
                    model="anthropic/claude-opus-4-6",
                    db_path=str(tmp_path / "test.db"),
                ) as session:
                    assert session.id.startswith("sess_")
                    raise ValueError("boom")

            with pytest.raises(ValueError, match="boom"):
                await _body_raises()
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


class TestConversationMessages:
    """Tests for L-8: session.conversation_messages() convenience method."""

    async def test_conversation_messages_excludes_summaries(self, tmp_path):
        """conversation_messages() must exclude is_summary messages."""
        from mnesis import MnesisSession

        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        )
        try:
            await session.record(user_message="Hello", assistant_response="Hi")
            await session.record(user_message="How are you?", assistant_response="Good")

            all_msgs = await session.messages()
            conv_msgs = await session.conversation_messages()

            # All conversation messages are non-summary
            assert all(not m.is_summary for m in conv_msgs)
            # conversation_messages is a subset of messages()
            conv_ids = {m.id for m in conv_msgs}
            all_ids = {m.id for m in all_msgs}
            assert conv_ids <= all_ids
            # With no compaction triggered, all messages are conversational
            assert len(conv_msgs) == len([m for m in all_msgs if not m.is_summary])
        finally:
            await session.close()


class TestEventBusUnsubscribeAll:
    """Tests for L-13: EventBus.unsubscribe_all()."""

    def test_unsubscribe_all_removes_from_specific_events(self):
        """unsubscribe_all removes handler from per-event registrations."""
        from mnesis.events.bus import EventBus, MnesisEvent

        bus = EventBus()
        calls: list[str] = []

        def handler(event, payload):
            calls.append("called")

        bus.subscribe(MnesisEvent.SESSION_CREATED, handler)
        bus.subscribe(MnesisEvent.SESSION_CLOSED, handler)
        bus.unsubscribe_all(handler)

        bus.publish(MnesisEvent.SESSION_CREATED, {})
        bus.publish(MnesisEvent.SESSION_CLOSED, {})
        assert calls == []

    def test_unsubscribe_all_removes_from_global_handlers(self):
        """unsubscribe_all removes handler from subscribe_all registrations."""
        from mnesis.events.bus import EventBus, MnesisEvent

        bus = EventBus()
        calls: list[str] = []

        def handler(event, payload):
            calls.append("called")

        bus.subscribe_all(handler)
        bus.unsubscribe_all(handler)
        bus.publish(MnesisEvent.SESSION_CREATED, {})
        assert calls == []

    def test_unsubscribe_all_is_silent_noop_when_not_registered(self):
        """unsubscribe_all is a no-op when handler is not registered anywhere."""
        from mnesis.events.bus import EventBus

        bus = EventBus()

        def handler(event, payload):
            pass

        # Should not raise
        bus.unsubscribe_all(handler)

    def test_unsubscribe_noop_documented(self):
        """unsubscribe is silent no-op when handler not registered."""
        from mnesis.events.bus import EventBus, MnesisEvent

        bus = EventBus()

        def handler(event, payload):
            pass

        # Not registered — should not raise
        bus.unsubscribe(MnesisEvent.SESSION_CREATED, handler)


class TestTokenEstimatorHeuristicOnly:
    """Tests for L-15: TokenEstimator(heuristic_only=True) constructor param."""

    def test_heuristic_only_skips_tiktoken(self):
        """heuristic_only=True uses character-based estimate regardless of model."""
        from mnesis.models.config import ModelInfo
        from mnesis.tokens.estimator import TokenEstimator

        estimator = TokenEstimator(heuristic_only=True)
        # cl100k_base encoding would normally trigger tiktoken; heuristic_only skips it
        model = ModelInfo(
            model_id="gpt-4o",
            provider_id="openai",
            encoding="cl100k_base",
            context_limit=128000,
            max_output_tokens=4096,
        )
        text = "hello world"
        result = estimator.estimate(text, model)
        # Heuristic: len // 4, minimum 1
        assert result == max(1, len(text) // 4)

    def test_heuristic_only_false_by_default(self):
        """Default constructor does not force heuristic mode."""
        from mnesis.tokens.estimator import TokenEstimator

        estimator = TokenEstimator()
        assert estimator._force_heuristic is False

    def test_heuristic_only_true(self):
        """heuristic_only=True sets _force_heuristic."""
        from mnesis.tokens.estimator import TokenEstimator

        estimator = TokenEstimator(heuristic_only=True)
        assert estimator._force_heuristic is True


class TestSessionHistory:
    """Tests for session.history() per-turn context snapshots."""

    async def test_history_accumulates_per_turn(self, tmp_path, monkeypatch):
        """history() grows by one entry per send() call."""
        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        from mnesis import MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            assert len(session.history()) == 0

            await session.send("First message.")
            assert len(session.history()) == 1

            await session.send("Second message.")
            assert len(session.history()) == 2

            await session.send("Third message.")
            assert len(session.history()) == 3

        # Each snapshot has a sequential turn_index
        history = session.history()
        for i, snap in enumerate(history):
            assert snap.turn_index == i

    async def test_history_token_breakdown(self, tmp_path, monkeypatch):
        """context_tokens fields are populated and consistent after send()."""
        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        from mnesis import MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
            system_prompt="You are a helpful assistant.",
        ) as session:
            await session.send("Tell me about the ocean.")
            await session.send("Tell me more.")

        history = session.history()
        assert len(history) == 2

        for snap in history:
            bd = snap.context_tokens
            # System prompt should be non-zero (we passed a real system prompt)
            assert bd.system_prompt > 0, "system_prompt tokens must be > 0"
            # Total should equal system_prompt + summary + messages
            assert bd.total == bd.system_prompt + bd.summary + bd.messages
            # Total is always positive after real turns
            assert bd.total > 0
            # Tool outputs are a subset of messages (can be 0 for text-only turns)
            assert 0 <= bd.tool_outputs <= bd.messages
            # Role is always 'assistant' for a completed turn
            assert snap.role == "assistant"

    async def test_history_compaction_captured(self, tmp_path, monkeypatch):
        """compact_result is populated on the snapshot following an explicit compact() call."""
        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        from mnesis import CompactionResult, MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            await session.send("Message one.")
            await session.send("Message two.")

            # Explicit manual compaction — result should appear on next snapshot
            await session.compact()

            await session.send("Message three, after compaction.")

        history = session.history()
        # 3 send() calls → 3 snapshots
        assert len(history) == 3

        # The third snapshot (turn_index=2) should carry the compact_result
        snap = history[2]
        assert snap.compact_result is not None
        assert isinstance(snap.compact_result, CompactionResult)
        # Snapshots before compaction should have None
        assert history[0].compact_result is None
        assert history[1].compact_result is None

    async def test_history_record_mode(self, tmp_path):
        """history() works in BYO-LLM mode (session.record())."""
        from mnesis import MnesisSession, TokenUsage

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
            system_prompt="You are a specialist.",
        ) as session:
            assert len(session.history()) == 0

            await session.record(
                user_message="What is 2+2?",
                assistant_response="Four.",
                tokens=TokenUsage(input=10, output=5),
            )
            await session.record(
                user_message="What is 3+3?",
                assistant_response="Six.",
                tokens=TokenUsage(input=12, output=5),
            )

        history = session.history()
        assert len(history) == 2

        for i, snap in enumerate(history):
            assert snap.turn_index == i
            assert snap.role == "assistant"
            # System prompt was non-empty, so system_prompt tokens > 0
            assert snap.context_tokens.system_prompt > 0
            # Total is sane
            assert snap.context_tokens.total > 0

        # history() returns a copy — mutating it does not affect internal state
        history.append(history[0])
        assert len(session.history()) == 2

    async def test_history_new_types_importable_from_top_level(self):
        """ContextBreakdown and TurnSnapshot are importable from top-level mnesis."""
        from mnesis import ContextBreakdown, TurnSnapshot

        assert ContextBreakdown.__name__ == "ContextBreakdown"
        assert TurnSnapshot.__name__ == "TurnSnapshot"

    # ── stream() tests ────────────────────────────────────────────────────────

    async def test_stream_yields_text_delta_events(self, tmp_path, mock_llm_env):
        """stream() yields at least one TextDelta event in mock mode."""
        from mnesis import MnesisSession, TextDelta, TurnComplete

        deltas = []
        turn_completes = []

        async with MnesisSession.open(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            async for event in session.stream("Hello!"):
                if isinstance(event, TextDelta):
                    deltas.append(event)
                elif isinstance(event, TurnComplete):
                    turn_completes.append(event)

        assert len(deltas) > 0
        assert all(isinstance(d.text, str) and len(d.text) > 0 for d in deltas)

    async def test_stream_yields_turn_complete_as_final_event(self, tmp_path, mock_llm_env):
        """stream() emits TurnComplete as the last event with a valid TurnResult."""
        from mnesis import MnesisSession, TurnComplete
        from mnesis.models.message import TurnResult

        events = []

        async with MnesisSession.open(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            async for event in session.stream("What is 2+2?"):
                events.append(event)

        assert len(events) > 0
        last = events[-1]
        assert isinstance(last, TurnComplete)
        assert isinstance(last.result, TurnResult)
        assert last.result.message_id.startswith("msg_")
        assert isinstance(last.result.text, str) and len(last.result.text) > 0

    async def test_stream_messages_persisted_after_completion(self, tmp_path, mock_llm_env):
        """After stream() completes, both user and assistant messages are in the store."""
        from mnesis import MnesisSession

        async with MnesisSession.open(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            async for _ in session.stream("Persist me."):
                pass

            messages = await session.messages()

        roles = [m.role for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    async def test_stream_compaction_triggered_when_threshold_crossed(self, tmp_path, mock_llm_env):
        """stream() triggers compaction if the current context exceeds the soft threshold.

        The context window is shrunk (``model_overrides``) so one large message
        puts the *current context* past the soft threshold; lifetime usage is
        deliberately not what drives the decision (see test_compaction_overflow).
        """
        from mnesis import MnesisConfig, MnesisSession
        from mnesis.models.config import CompactionConfig, StoreConfig

        cfg = MnesisConfig(
            store=StoreConfig(db_path=str(tmp_path / "test.db")),
            compaction=CompactionConfig(compaction_output_budget=2000),
            model_overrides={"context_limit": 12000, "max_output_tokens": 1000},
        )
        turn_complete_result = None

        async with MnesisSession.open(model="anthropic/claude-opus-4-6", config=cfg) as session:
            async for event in session.stream("lorem ipsum " * 2000):
                if event.type == "turn_complete":
                    turn_complete_result = event.result

        assert turn_complete_result is not None
        assert turn_complete_result.compaction_triggered is True

    async def test_stream_abandonment_still_persists_turn(self, tmp_path, mock_llm_env):
        """Breaking out of stream() early still results in full message persistence."""
        from mnesis import MnesisSession

        db = str(tmp_path / "test.db")
        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=db,
        )
        session_id = session.id

        try:
            # Consume only the first event then break — abandons the iterator
            async for _event in session.stream("Abandon me."):
                break  # immediately abandon

            # Wait deterministically for the background send() task to persist
            # the full turn rather than relying on a fixed-duration sleep.
            deadline = asyncio.get_running_loop().time() + 5.0
            while True:
                messages = await session.messages()
                roles = [m.role for m in messages]
                if "user" in roles and "assistant" in roles:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    pytest.fail(
                        "Timed out waiting for abandoned stream() to persist both user "
                        "and assistant messages"
                    )
                await asyncio.sleep(0.01)
        finally:
            await session.close()

        # Reload the session and verify both messages were persisted
        reloaded = await MnesisSession.load(session_id, db_path=db)
        messages = await reloaded.messages()
        await reloaded.close()

        roles = [m.role for m in messages]
        assert "user" in roles
        assert "assistant" in roles


class TestRetryResilience:
    """Tests for RetryConfig and the retry loop in send()."""

    # ------------------------------------------------------------------
    # Test 1: max_retries=0 (default) does not retry
    # ------------------------------------------------------------------
    async def test_no_retry_by_default(self, tmp_path):
        """max_retries=0 (default) means send() fails immediately on error."""
        import os
        from unittest.mock import AsyncMock, patch

        from litellm.exceptions import RateLimitError

        from mnesis import MnesisSession
        from mnesis.events.bus import MnesisEvent
        from mnesis.models.config import MnesisConfig, SessionConfig

        cfg = MnesisConfig(session=SessionConfig())  # max_retries=0 by default
        assert cfg.session.retry.max_retries == 0

        retry_events: list[dict] = []

        # Patch _stream_response to raise RateLimitError on every call
        async def _raise_rate_limit(*args, **kwargs):
            raise RateLimitError(
                message="rate limited",
                llm_provider="anthropic",
                model="claude-opus-4-6",
            )

        env = {"MNESIS_MOCK_LLM": "0"}
        with patch.dict(os.environ, env):
            async with await MnesisSession.create(
                model="anthropic/claude-opus-4-6",
                config=cfg,
                db_path=str(tmp_path / "test.db"),
            ) as session:
                session.event_bus.subscribe(
                    MnesisEvent.LLM_RETRY,
                    lambda e, p: retry_events.append(p),
                )
                with patch.object(
                    session, "_stream_response", new=AsyncMock(side_effect=_raise_rate_limit)
                ):
                    result = await session.send("Hello")

        # No retry events — failed immediately
        assert len(retry_events) == 0
        assert result.finish_reason == "error"
        assert "Error" in result.text

    # ------------------------------------------------------------------
    # Test 2: retry succeeds after 1 transient RateLimitError
    # ------------------------------------------------------------------
    async def test_retry_succeeds_after_transient_error(self, tmp_path):
        """send() retries after a RateLimitError and returns the successful response."""
        import os
        from unittest.mock import AsyncMock, patch

        from litellm.exceptions import RateLimitError

        from mnesis import MnesisSession
        from mnesis.events.bus import MnesisEvent
        from mnesis.models.config import MnesisConfig, RetryConfig, SessionConfig
        from mnesis.models.message import TokenUsage

        cfg = MnesisConfig(
            session=SessionConfig(retry=RetryConfig(max_retries=2, base_delay=0.0, jitter=False))
        )
        retry_events: list[dict] = []
        call_count = 0

        async def _flaky_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RateLimitError(
                    message="rate limited",
                    llm_provider="anthropic",
                    model="claude-opus-4-6",
                )
            return "success text", TokenUsage(input=10, output=5, total=15), "stop"

        env = {"MNESIS_MOCK_LLM": "0"}
        with patch.dict(os.environ, env):
            async with await MnesisSession.create(
                model="anthropic/claude-opus-4-6",
                config=cfg,
                db_path=str(tmp_path / "test.db"),
            ) as session:
                session.event_bus.subscribe(
                    MnesisEvent.LLM_RETRY,
                    lambda e, p: retry_events.append(p),
                )
                with patch.object(
                    session, "_stream_response", new=AsyncMock(side_effect=_flaky_stream)
                ):
                    result = await session.send("Hello")

        assert result.finish_reason == "stop"
        assert result.text == "success text"
        assert call_count == 2
        # One retry event published (for the first failure)
        assert len(retry_events) == 1
        assert retry_events[0]["attempt"] == 1
        assert retry_events[0]["max_retries"] == 2

    # ------------------------------------------------------------------
    # Test 3: retry exhausted returns finish_reason="error"
    # ------------------------------------------------------------------
    async def test_retry_exhausted_returns_error(self, tmp_path):
        """send() returns finish_reason='error' after all retries are exhausted."""
        import os
        from unittest.mock import AsyncMock, patch

        from litellm.exceptions import InternalServerError

        from mnesis import MnesisSession
        from mnesis.models.config import MnesisConfig, RetryConfig, SessionConfig

        cfg = MnesisConfig(
            session=SessionConfig(retry=RetryConfig(max_retries=2, base_delay=0.0, jitter=False))
        )

        async def _always_fail(*args, **kwargs):
            raise InternalServerError(
                message="server error",
                llm_provider="anthropic",
                model="claude-opus-4-6",
            )

        env = {"MNESIS_MOCK_LLM": "0"}
        with patch.dict(os.environ, env):
            async with await MnesisSession.create(
                model="anthropic/claude-opus-4-6",
                config=cfg,
                db_path=str(tmp_path / "test.db"),
            ) as session:
                with patch.object(
                    session, "_stream_response", new=AsyncMock(side_effect=_always_fail)
                ) as mock:
                    result = await session.send("Hello")
                    # Called max_retries+1 times total (initial + 2 retries)
                    assert mock.call_count == 3

        assert result.finish_reason == "error"
        assert "Error" in result.text

    # ------------------------------------------------------------------
    # Test 4: non-retryable error is NOT retried
    # ------------------------------------------------------------------
    async def test_non_retryable_error_not_retried(self, tmp_path):
        """AuthenticationError is non-retryable — send() fails immediately."""
        import os
        from unittest.mock import AsyncMock, patch

        from litellm.exceptions import AuthenticationError

        from mnesis import MnesisSession
        from mnesis.models.config import MnesisConfig, RetryConfig, SessionConfig

        cfg = MnesisConfig(
            session=SessionConfig(retry=RetryConfig(max_retries=3, base_delay=0.0, jitter=False))
        )

        async def _auth_fail(*args, **kwargs):
            raise AuthenticationError(
                message="invalid api key",
                llm_provider="anthropic",
                model="claude-opus-4-6",
            )

        env = {"MNESIS_MOCK_LLM": "0"}
        with patch.dict(os.environ, env):
            async with await MnesisSession.create(
                model="anthropic/claude-opus-4-6",
                config=cfg,
                db_path=str(tmp_path / "test.db"),
            ) as session:
                with patch.object(
                    session, "_stream_response", new=AsyncMock(side_effect=_auth_fail)
                ) as mock:
                    result = await session.send("Hello")
                    # Called only once — no retry for auth errors
                    assert mock.call_count == 1

        assert result.finish_reason == "error"

    # ------------------------------------------------------------------
    # Test 5: LLM_RETRY event published with correct payload
    # ------------------------------------------------------------------
    async def test_llm_retry_event_payload(self, tmp_path):
        """LLM_RETRY event carries correct session_id, attempt, error info, delay."""
        import os
        from unittest.mock import AsyncMock, patch

        from litellm.exceptions import ServiceUnavailableError

        from mnesis import MnesisSession
        from mnesis.events.bus import MnesisEvent
        from mnesis.models.config import MnesisConfig, RetryConfig, SessionConfig
        from mnesis.models.message import TokenUsage

        cfg = MnesisConfig(
            session=SessionConfig(retry=RetryConfig(max_retries=2, base_delay=0.0, jitter=False))
        )
        retry_payloads: list[dict] = []

        call_count = 0

        async def _fail_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ServiceUnavailableError(
                    message="service down",
                    llm_provider="anthropic",
                    model="claude-opus-4-6",
                )
            return "ok", TokenUsage(input=5, output=3, total=8), "stop"

        env = {"MNESIS_MOCK_LLM": "0"}
        with patch.dict(os.environ, env):
            async with await MnesisSession.create(
                model="anthropic/claude-opus-4-6",
                config=cfg,
                db_path=str(tmp_path / "test.db"),
            ) as session:
                session_id = session.id
                session.event_bus.subscribe(
                    MnesisEvent.LLM_RETRY,
                    lambda e, p: retry_payloads.append(p),
                )
                with patch.object(
                    session, "_stream_response", new=AsyncMock(side_effect=_fail_once)
                ):
                    await session.send("Hello")

        assert len(retry_payloads) == 1
        payload = retry_payloads[0]
        assert payload["session_id"] == session_id
        assert payload["attempt"] == 1
        assert payload["max_retries"] == 2
        assert "ServiceUnavailableError" in payload["error_type"]
        assert "service down" in payload["error_message"]
        assert isinstance(payload["delay_seconds"], float)
        # Additive key: send() retries are labelled, and carry no compaction-only keys.
        assert payload["source"] == "send"
        assert "stage" not in payload and "compaction_level" not in payload

    # ------------------------------------------------------------------
    # Test 6: close() during retry backoff cancels the sleep promptly
    # ------------------------------------------------------------------
    async def test_close_during_retry_cancels_sleep(self, tmp_path):
        """close() cancels the retry sleep so send() unblocks without waiting.

        The critical invariant is that the retry sleep is cancelled promptly —
        the task must complete well within the 60-second sleep window, even if
        the DB is already closed by the time send() tries to persist results.
        """
        import contextlib
        import os
        from unittest.mock import AsyncMock, patch

        from litellm.exceptions import RateLimitError

        from mnesis import MnesisSession
        from mnesis.models.config import MnesisConfig, RetryConfig, SessionConfig

        # Large base_delay so the test would hang if cancellation doesn't work.
        cfg = MnesisConfig(
            session=SessionConfig(retry=RetryConfig(max_retries=3, base_delay=60.0, jitter=False))
        )

        async def _always_rate_limited(*args, **kwargs):
            raise RateLimitError(
                message="rate limited",
                llm_provider="anthropic",
                model="claude-opus-4-6",
            )

        env = {"MNESIS_MOCK_LLM": "0"}
        with patch.dict(os.environ, env):
            session = await MnesisSession.create(
                model="anthropic/claude-opus-4-6",
                config=cfg,
                db_path=str(tmp_path / "test.db"),
            )

            async def _send_and_collect():
                with patch.object(
                    session, "_stream_response", new=AsyncMock(side_effect=_always_rate_limited)
                ):
                    return await session.send("Hello")

            # Run send() in background; it will hit the first error and go to sleep.
            send_task = asyncio.create_task(_send_and_collect())

            # Wait deterministically for the retry sleep task to be registered and
            # running, rather than relying on a fixed asyncio.sleep() delay that can
            # be too short on a slow CI runner.
            async def _wait_for_retry_sleep() -> None:
                while True:
                    t = getattr(session, "_retry_sleep_task", None)
                    if t is not None and not t.done():
                        return
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(_wait_for_retry_sleep(), timeout=5.0)

            # close() must cancel the sleep so send() unblocks.
            await session.close()

            # The send task should complete quickly (well under the 60s sleep).
            # It may raise an exception (e.g. DB closed) or return normally —
            # either is acceptable; what matters is it finishes promptly.
            try:
                await asyncio.wait_for(send_task, timeout=2.0)
            except TimeoutError:
                send_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    _ = await send_task
                pytest.fail("send() did not finish promptly after close() cancelled retry sleep")
            except Exception:
                pass  # DB-closed or other error after cancellation — expected

        # If we reach here without timing out, cancellation worked correctly.


class TestSessionCoverageGaps:
    """Additional tests targeting uncovered branches in session.py."""

    # ── _stream_response error path ─────────────────────────────────────────

    async def test_send_handles_llm_exception_gracefully(self, tmp_path, monkeypatch):
        """send() returns finish_reason='error' when the LLM call raises an exception.

        Covers the except-branch in send() (lines 478-481) that catches any
        exception from _stream_response/_mock_response and stores an error message.
        """
        from mnesis import MnesisSession

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        db = str(tmp_path / "test.db")

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=db,
        ) as session:
            # Patch _mock_response to raise so we exercise the error path without
            # requiring a real API key.
            async def _raise(*args, **kwargs):
                raise RuntimeError("simulated LLM failure")

            session._mock_response = _raise  # type: ignore[method-assign]
            result = await session.send("Trigger error")

        assert result.finish_reason == "error"
        assert "[Error:" in result.text

    # ── _stream_response async on_part callback ─────────────────────────────

    async def test_send_async_on_part_callback(self, tmp_path, monkeypatch):
        """send() awaits on_part when it returns a coroutine (line 564-565)."""
        from mnesis import MnesisSession

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        received: list[str] = []

        async def async_on_part(part):
            from mnesis.models.message import TextPart

            if isinstance(part, TextPart):
                received.append(part.text)

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            await session.send("Hello", on_part=async_on_part)

        assert len(received) > 0

    # ── close() with in-flight compaction ───────────────────────────────────

    async def test_close_waits_for_in_flight_compaction(self, tmp_path, monkeypatch):
        """close() calls wait_for_pending() before closing the DB.

        Exercises the wait_for_pending() call inside close() by spying on it
        and verifying it was invoked, regardless of whether compaction actually
        triggered (which is non-deterministic with mock responses).
        """

        from mnesis import MnesisConfig, MnesisSession
        from mnesis.models.config import CompactionConfig

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        cfg = MnesisConfig(
            compaction=CompactionConfig(
                auto=True,
                compaction_output_budget=1_000,
                soft_threshold_fraction=0.1,
            )
        )

        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
            config=cfg,
        )

        # Spy on wait_for_pending so we can assert it was called by close().
        wait_called = [0]
        original_wait = session._compaction_engine.wait_for_pending

        async def _spy_wait():
            wait_called[0] += 1
            return await original_wait()

        session._compaction_engine.wait_for_pending = _spy_wait  # type: ignore[method-assign]

        await session.send("Trigger compaction")
        await session.close()

        # close() must have called wait_for_pending() at least once.
        assert wait_called[0] >= 1

    # ── load() for existing session ─────────────────────────────────────────

    async def test_load_resumes_and_sends(self, tmp_path, monkeypatch):
        """load() reopens an existing session and send() appends more turns."""
        from mnesis import MnesisSession

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        db = str(tmp_path / "test.db")

        session1 = await MnesisSession.create(model="anthropic/claude-opus-4-6", db_path=db)
        sid = session1.id
        await session1.send("First turn")
        await session1.close()

        session2 = await MnesisSession.load(sid, db_path=db)
        result = await session2.send("Second turn")
        msgs = await session2.messages()
        await session2.close()

        assert result.finish_reason in ("stop", "end_turn")
        # 2 user + 2 assistant messages
        assert len(msgs) >= 4

    # ── load() raises on db_path + config.store.db_path conflict ────────────

    async def test_load_raises_on_db_path_conflict(self, tmp_path):
        """load() raises ValueError when db_path AND config.store.db_path are both set."""
        import pytest

        from mnesis import MnesisConfig, MnesisSession
        from mnesis.models.config import StoreConfig

        cfg = MnesisConfig(store=StoreConfig(db_path=str(tmp_path / "other.db")))
        with pytest.raises(ValueError, match="Specify db_path either via"):
            await MnesisSession.load(
                "sess_fake",
                config=cfg,
                db_path=str(tmp_path / "conflict.db"),
            )

    # ── create() raises on db_path conflict ─────────────────────────────────

    async def test_create_raises_on_db_path_conflict(self, tmp_path):
        """create() raises ValueError when both db_path and config.store.db_path are given."""
        import pytest

        from mnesis import MnesisConfig, MnesisSession
        from mnesis.models.config import StoreConfig

        cfg = MnesisConfig(store=StoreConfig(db_path=str(tmp_path / "other.db")))
        with pytest.raises(ValueError, match="Specify db_path either via"):
            await MnesisSession.create(
                model="anthropic/claude-opus-4-6",
                config=cfg,
                db_path=str(tmp_path / "conflict.db"),
            )

    # ── context_for_next_turn() after record() ───────────────────────────────

    async def test_context_for_next_turn_after_record(self, tmp_path):
        """context_for_next_turn() returns a non-empty message list after record()."""
        from mnesis import MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            await session.record(
                user_message="What is 1+1?",
                assistant_response="It is 2.",
            )
            messages = await session.context_for_next_turn()

        assert len(messages) >= 2
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    # ── context_for_next_turn() with custom system_prompt ───────────────────

    async def test_context_for_next_turn_custom_system_prompt(self, tmp_path):
        """context_for_next_turn() accepts an override system_prompt."""
        from mnesis import MnesisSession

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            # Works even on an empty session
            messages = await session.context_for_next_turn(system_prompt="Override prompt")

        # Empty session has no conversation messages — just an empty list
        assert isinstance(messages, list)

    # ── model_overrides applied during create() ──────────────────────────────

    async def test_create_applies_model_overrides(self, tmp_path, monkeypatch):
        """create() applies model_overrides from MnesisConfig to ModelInfo."""
        from mnesis import MnesisConfig, MnesisSession
        from mnesis.models.config import CompactionConfig

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        cfg = MnesisConfig(
            model_overrides={"context_limit": 8192, "max_output_tokens": 1024},
            compaction=CompactionConfig(compaction_output_budget=1_000),
        )
        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
            config=cfg,
        )
        try:
            assert session._model_info.context_limit == 8192
        finally:
            await session.close()

    # ── doom loop detection ──────────────────────────────────────────────────

    async def test_send_text_turn_resets_doom_window(self, tmp_path, monkeypatch):
        """send() turns carry no tracked tool calls, so they reset any stale window."""
        from mnesis import MnesisConfig, MnesisSession
        from mnesis.models.config import SessionConfig

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        cfg = MnesisConfig(session=SessionConfig(doom_loop_threshold=2))

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
            config=cfg,
        ) as session:
            session._recent_tool_calls = [
                ("read_file", '{"path":"/x"}'),
                ("read_file", '{"path":"/x"}'),
            ]
            result = await session.send("Do something")

        assert result.doom_loop_detected is False
        assert session._recent_tool_calls == []

    async def test_record_doom_loop_does_not_leak_into_send(self, tmp_path, monkeypatch):
        """After record() trips detection, later text-only send() turns are clean."""
        from mnesis import MnesisConfig, MnesisSession
        from mnesis.events.bus import MnesisEvent
        from mnesis.models.config import SessionConfig
        from mnesis.models.message import ToolPart, ToolStatus

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        cfg = MnesisConfig(session=SessionConfig(doom_loop_threshold=3))
        events: list[dict] = []

        def _tp(i: int) -> ToolPart:
            return ToolPart(
                tool_name="read_file",
                tool_call_id=f"c{i}",
                input={"path": "/x"},
                output="ok",
                status=ToolStatus(state="completed"),
            )

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6", db_path=str(tmp_path / "test.db"), config=cfg
        ) as session:
            session.subscribe(MnesisEvent.DOOM_LOOP_DETECTED, lambda e, p: events.append(p))
            recorded = [await session.record(f"t{i}", [_tp(i)]) for i in range(3)]
            assert recorded[-1].doom_loop_detected is True
            assert len(events) == 1
            sent = [await session.send(f"hello {i}") for i in range(2)]

        assert [r.doom_loop_detected for r in sent] == [False, False]
        assert len(events) == 1

    async def test_record_mixed_key_tool_input_does_not_crash(self, tmp_path, monkeypatch):
        """Nested dicts with unsortable mixed key types must not break record()."""
        from mnesis import MnesisSession
        from mnesis.models.message import ToolPart, ToolStatus

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        part = ToolPart(
            tool_name="search",
            tool_call_id="c1",
            input={"q": {1: "a", "b": "c"}},
            output="ok",
            status=ToolStatus(state="completed"),
        )
        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6", db_path=str(tmp_path / "test.db")
        ) as session:
            await session.record("q", [part])
            assert len(session._recent_tool_calls) == 1

    # ── subscribe() convenience wrapper ──────────────────────────────────────

    async def test_subscribe_registers_event_handler(self, tmp_path, monkeypatch):
        """subscribe() registers a handler that fires on subsequent events."""
        from mnesis import MnesisSession
        from mnesis.events.bus import MnesisEvent

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        received = []

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            session.subscribe(
                MnesisEvent.MESSAGE_CREATED,
                lambda event, payload: received.append(payload),
            )
            await session.send("Hi")

        assert len(received) > 0

    # ── hard overflow path ───────────────────────────────────────────────────

    async def test_send_hard_overflow_triggers_synchronous_compaction(self, tmp_path, monkeypatch):
        """Hard overflow causes blocking compaction before the LLM call (lines 429-435).

        We set the compaction budget so small that even a single send() pushes
        the cumulative token count past the hard limit on the *next* call, forcing
        the synchronous wait path.
        """
        from mnesis import MnesisConfig, MnesisSession
        from mnesis.models.config import CompactionConfig

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        # Absurdly tight budget: context limit == output budget so usable == 0
        # which means is_hard_overflow is always True after the first turn.
        cfg = MnesisConfig(
            compaction=CompactionConfig(
                auto=True,
                compaction_output_budget=1_000,
            )
        )

        session = await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
            config=cfg,
        )
        try:
            # Manually override is_hard_overflow to return True on second call
            call_count = [0]
            original = session._compaction_engine.is_hard_overflow
            original_wait = session._compaction_engine.wait_for_pending
            wait_calls = [0]

            def patched_overflow(tokens, model_info):
                call_count[0] += 1
                if call_count[0] >= 2:
                    return True
                return original(tokens, model_info)

            async def patched_wait():
                wait_calls[0] += 1
                return await original_wait()

            session._compaction_engine.is_hard_overflow = patched_overflow  # type: ignore[method-assign]
            session._compaction_engine.wait_for_pending = patched_wait  # type: ignore[method-assign]
            await session.send("First message")
            result = await session.send("Second message — should trigger sync compaction")
            assert result.finish_reason in ("stop", "end_turn", "error")
            # The synchronous compaction path must have awaited wait_for_pending
            assert wait_calls[0] >= 1
        finally:
            await session.close()

    async def test_send_hard_overflow_populates_compaction_result(self, tmp_path, monkeypatch):
        """TurnResult.compaction_result carries the result of the blocking hard-threshold run."""
        from mnesis import MnesisSession
        from mnesis.models.message import CompactionResult

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        expected = CompactionResult(
            session_id="s",
            summary_message_id="msg_summary",
            level_used=3,
            compacted_message_count=4,
            summary_token_count=10,
            tokens_before=1000,
            tokens_after=10,
            elapsed_ms=1.0,
        )

        async def _fake_compaction() -> CompactionResult:
            await asyncio.sleep(0)
            return expected

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6", db_path=str(tmp_path / "test.db")
        ) as session:
            engine = session._compaction_engine
            engine.is_hard_overflow = lambda tokens, model: True  # type: ignore[method-assign]
            engine._pending_task = asyncio.ensure_future(_fake_compaction())
            result = await session.send("Over the hard limit")

        assert result.compaction_result == expected

    async def test_send_hard_overflow_ignores_stale_finished_task(self, tmp_path, monkeypatch):
        """A finished earlier background task is not reported as this turn's result."""
        from mnesis import MnesisSession
        from mnesis.models.message import CompactionResult

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        def _result(msg_id: str) -> CompactionResult:
            return CompactionResult(
                session_id="s",
                summary_message_id=msg_id,
                level_used=3,
                compacted_message_count=1,
                summary_token_count=1,
                tokens_before=1,
                tokens_after=1,
                elapsed_ms=1.0,
            )

        async def _done() -> CompactionResult:
            return _result("msg_stale")

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6", db_path=str(tmp_path / "test.db")
        ) as session:
            engine = session._compaction_engine
            stale = asyncio.ensure_future(_done())
            await asyncio.wait({stale})
            engine._pending_task = stale
            engine.is_hard_overflow = lambda tokens, model: True  # type: ignore[method-assign]
            engine.is_overflow = lambda tokens, model: True  # type: ignore[method-assign]

            async def _fresh(
                session_id: str, abort: object = None, **kw: object
            ) -> CompactionResult:
                return _result("msg_fresh")

            engine.run_compaction = _fresh  # type: ignore[method-assign]
            result = await session.send("Over the hard limit")

        assert result.compaction_result is not None
        assert result.compaction_result.summary_message_id == "msg_fresh"

    async def test_send_without_hard_overflow_has_no_compaction_result(self, tmp_path, monkeypatch):
        """Background (soft) compaction never populates TurnResult.compaction_result."""
        from mnesis import MnesisSession

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6", db_path=str(tmp_path / "test.db")
        ) as session:
            result = await session.send("Hello")

        assert result.compaction_result is None

    # ── load() rebuilds cumulative token usage ──────────────────────────────

    async def test_load_rebuilds_cumulative_token_usage(self, tmp_path, monkeypatch):
        """load() restores token_usage from persisted turns so overflow checks see them."""
        from mnesis import MnesisSession
        from mnesis.models.message import TokenUsage

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        db = str(tmp_path / "test.db")

        s1 = await MnesisSession.create(model="anthropic/claude-opus-4-6", db_path=db)
        await s1.send("First message.")
        await s1.record("Second", "Reply", tokens=TokenUsage(input=40, output=2))
        before = s1.token_usage
        session_id = s1.id
        await s1.close()
        assert before.effective_total() > 0

        s2 = await MnesisSession.load(session_id, db_path=db)
        try:
            assert s2.token_usage == before
        finally:
            await s2.close()

    async def test_load_new_session_has_zero_token_usage(self, tmp_path, monkeypatch):
        """load() on a session with no turns yields empty usage."""
        from mnesis import MnesisSession

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        db = str(tmp_path / "test.db")
        s1 = await MnesisSession.create(model="anthropic/claude-opus-4-6", db_path=db)
        session_id = s1.id
        await s1.close()

        s2 = await MnesisSession.load(session_id, db_path=db)
        try:
            assert s2.token_usage.effective_total() == 0
        finally:
            await s2.close()

    # ── doom loop detection via record() ────────────────────────────────────

    @staticmethod
    def _tool_part(name: str = "read_file", path: str = "/x", call_id: str = "c1"):
        from mnesis.models.message import ToolPart, ToolStatus

        return ToolPart(
            tool_name=name,
            tool_call_id=call_id,
            input={"path": path},
            output="ok",
            status=ToolStatus(state="completed"),
        )

    async def test_record_detects_doom_loop(self, tmp_path, monkeypatch):
        """record() tracks tool calls so repeated identical calls trip doom-loop detection."""
        from mnesis import MnesisConfig, MnesisSession
        from mnesis.events.bus import MnesisEvent
        from mnesis.models.config import SessionConfig

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        cfg = MnesisConfig(session=SessionConfig(doom_loop_threshold=3))
        events: list[dict] = []

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6", db_path=str(tmp_path / "test.db"), config=cfg
        ) as session:
            session.subscribe(MnesisEvent.DOOM_LOOP_DETECTED, lambda e, p: events.append(p))
            results = [
                await session.record(f"turn {i}", [self._tool_part(call_id=f"c{i}")])
                for i in range(3)
            ]

        assert [r.doom_loop_detected for r in results] == [False, False, True]
        assert events and events[0]["tool"] == "read_file"

    async def test_record_different_tool_inputs_not_doom_loop(self, tmp_path, monkeypatch):
        """Calls with the same tool but different inputs are not a doom loop."""
        from mnesis import MnesisConfig, MnesisSession
        from mnesis.models.config import SessionConfig

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        cfg = MnesisConfig(session=SessionConfig(doom_loop_threshold=2))

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6", db_path=str(tmp_path / "test.db"), config=cfg
        ) as session:
            r1 = await session.record("a", [self._tool_part(path="/a", call_id="c1")])
            r2 = await session.record("b", [self._tool_part(path="/b", call_id="c2")])

        assert r1.doom_loop_detected is False
        assert r2.doom_loop_detected is False

    async def test_record_text_turn_breaks_doom_loop_run(self, tmp_path, monkeypatch):
        """A text-only turn between identical tool calls resets the consecutive run."""
        from mnesis import MnesisConfig, MnesisSession
        from mnesis.models.config import SessionConfig

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        cfg = MnesisConfig(session=SessionConfig(doom_loop_threshold=2))

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6", db_path=str(tmp_path / "test.db"), config=cfg
        ) as session:
            await session.record("a", [self._tool_part(call_id="c1")])
            await session.record("chat", "just text")
            r3 = await session.record("b", [self._tool_part(call_id="c2")])
            window = list(session._recent_tool_calls)

        assert r3.doom_loop_detected is False
        assert len(window) == 1

    # ── compaction_in_progress property ─────────────────────────────────────

    async def test_compaction_in_progress_property(self, tmp_path, monkeypatch):
        """compaction_in_progress returns False when no task is pending."""
        from mnesis import MnesisSession

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            assert session.compaction_in_progress is False

    # ── conversation_messages() ──────────────────────────────────────────────

    async def test_conversation_messages_excludes_summaries(self, tmp_path, monkeypatch):
        """conversation_messages() returns only non-summary messages."""
        from mnesis import MnesisSession

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        async with await MnesisSession.create(
            model="anthropic/claude-opus-4-6",
            db_path=str(tmp_path / "test.db"),
        ) as session:
            await session.send("Hello")
            conv = await session.conversation_messages()
            all_msgs = await session.messages()

        assert all(not m.is_summary for m in conv)
        assert len(conv) <= len(all_msgs)


class TestRetrySleepSemantics:
    """D2: the cancellable retry backoff keeps its semantics."""

    @staticmethod
    async def _session(tmp_path, base_delay: float):
        from mnesis import MnesisSession
        from mnesis.models.config import MnesisConfig, RetryConfig, SessionConfig

        cfg = MnesisConfig(
            session=SessionConfig(
                retry=RetryConfig(max_retries=2, base_delay=base_delay, jitter=False)
            )
        )
        return await MnesisSession.create(
            model="anthropic/claude-opus-4-6", config=cfg, db_path=str(tmp_path / "rs.db")
        )

    @staticmethod
    async def _sleep_task(session) -> asyncio.Task:
        for _ in range(500):
            if session._retry_sleep_task is not None:
                return session._retry_sleep_task
            await asyncio.sleep(0.01)
        raise AssertionError("retry sleep never started")

    @staticmethod
    def _flaky(fail_times: int):
        from litellm.exceptions import ServiceUnavailableError

        from mnesis.models.message import TokenUsage

        calls = {"n": 0}

        async def _stream(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= fail_times:
                raise ServiceUnavailableError(
                    message="down", llm_provider="anthropic", model="claude-opus-4-6"
                )
            return "recovered", TokenUsage(input=5, output=3, total=8), "stop"

        return _stream, calls

    async def test_backoff_elapses_then_call_is_retried(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        stream, calls = self._flaky(1)
        session = await self._session(tmp_path, base_delay=0.05)
        try:
            with patch.object(session, "_stream_response", new=AsyncMock(side_effect=stream)):
                result = await session.send("hi")
        finally:
            await session.close()
        assert result.text == "recovered"
        assert calls["n"] == 2
        assert session._retry_sleep_task is None

    async def test_cancelling_send_cancels_the_sleep(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        stream, _ = self._flaky(5)
        session = await self._session(tmp_path, base_delay=60.0)
        try:
            with patch.object(session, "_stream_response", new=AsyncMock(side_effect=stream)):
                task = asyncio.create_task(session.send("hi"))
                sleeper = await asyncio.wait_for(self._sleep_task(session), timeout=5)
                _ = task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    _ = await asyncio.wait_for(task, timeout=5)
            # Cancelling send() propagates (it is not swallowed into an error turn)
            # and stops the sleep too.
            assert task.cancelled()
            assert sleeper.cancelled()
        finally:
            await session.close()

    async def test_timeout_around_send_during_backoff_raises(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        stream, _ = self._flaky(5)
        session = await self._session(tmp_path, base_delay=60.0)
        try:
            with patch.object(session, "_stream_response", new=AsyncMock(side_effect=stream)):
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(0.3):
                        _ = await session.send("hi")
            assert session._retry_sleep_task is None
        finally:
            await session.close()


class TestCancelledSendLeavesNoEmptyAssistantTurn:
    async def _context_roles(self, session):
        ctx = await session.context_for_next_turn()
        return [(m["role"], (m["content"] or "").strip() != "") for m in ctx]

    async def test_backoff_cancel(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        stream, _ = TestRetrySleepSemantics._flaky(5)
        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=60.0)
        try:
            with patch.object(session, "_stream_response", new=AsyncMock(side_effect=stream)):
                with pytest.raises(TimeoutError):
                    async with asyncio.timeout(0.3):
                        _ = await session.send("hi")
            assert await self._context_roles(session) == [("user", True)]
        finally:
            await session.close()

    async def test_streaming_cancel(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")

        async def hang(*args, **kwargs):
            await asyncio.sleep(60)

        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        try:
            with patch.object(session, "_stream_response", new=AsyncMock(side_effect=hang)):
                task = asyncio.create_task(session.send("hi"))
                await asyncio.sleep(0.2)
                _ = task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    _ = await task
            assert await self._context_roles(session) == [("user", True)]
        finally:
            await session.close()


class TestCloseWaitsForInflightOperations:
    """close() lets in-flight send()/record() finish persisting before the store closes."""

    async def test_close_during_backoff_persists_the_error_turn(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        from mnesis.store.immutable import ImmutableStore

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        stream, _ = TestRetrySleepSemantics._flaky(5)
        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=60.0)
        with patch.object(session, "_stream_response", new=AsyncMock(side_effect=stream)):
            task = asyncio.create_task(session.send("hi"))
            _ = await asyncio.wait_for(TestRetrySleepSemantics._sleep_task(session), timeout=5)
            await session.close()
            result = await asyncio.wait_for(task, timeout=5)
        assert result.text == "[Error: retry cancelled]"
        # The error turn was persisted before the store closed.
        store = ImmutableStore(
            session._config.store.model_copy(update={"db_path": str(tmp_path / "rs.db")})
        )
        await store.initialize()
        try:
            texts = [
                m.parts[0].text
                for m in await store.get_messages_with_parts(session.id)
                if m.parts and hasattr(m.parts[0], "text")
            ]
        finally:
            await store.close()
        assert "[Error: retry cancelled]" in texts

    async def test_send_and_record_after_close_are_rejected(self, tmp_path):
        from mnesis import SessionClosedError

        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        await session.close()
        with pytest.raises(SessionClosedError):
            _ = await session.send("hi")
        with pytest.raises(SessionClosedError):
            _ = await session.record("q", "a")

    async def test_new_send_while_closing_is_rejected(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        from mnesis import SessionClosedError
        from mnesis.models.message import TokenUsage

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        release = asyncio.Event()
        entered = asyncio.Event()

        async def slow(*args, **kwargs):
            entered.set()
            await release.wait()
            return "done", TokenUsage(input=1, output=1, total=2), "stop"

        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        with patch.object(session, "_stream_response", new=AsyncMock(side_effect=slow)):
            first = asyncio.create_task(session.send("hi"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            closing = asyncio.create_task(session.close())
            await asyncio.sleep(0.05)
            assert not closing.done()  # waiting for the stream already on the wire
            with pytest.raises(SessionClosedError):
                _ = await session.send("late")
            release.set()
            assert (await asyncio.wait_for(first, timeout=5)).text == "done"
            await asyncio.wait_for(closing, timeout=5)

    async def test_double_close_is_fine(self, tmp_path):
        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        await session.close()
        await session.close()

    async def test_close_from_inside_a_send_does_not_deadlock(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        from mnesis.models.message import TokenUsage

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)

        async def closes_itself(*args, **kwargs):
            await session.close()  # runs in send()'s own task: must not wait for itself
            return "x", TokenUsage(input=1, output=1, total=2), "stop"

        with patch.object(session, "_stream_response", new=AsyncMock(side_effect=closes_itself)):
            with pytest.raises(Exception):  # noqa: B017 - the store is closed under it
                _ = await asyncio.wait_for(session.send("hi"), timeout=5)
        assert session._closed

    async def test_close_gives_up_waiting_on_a_hung_stream(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        monkeypatch.setattr("mnesis.session._CLOSE_INFLIGHT_TIMEOUT", 0.1)
        entered = asyncio.Event()

        async def hang(*args, **kwargs):
            entered.set()
            await asyncio.sleep(60)

        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        with patch.object(session, "_stream_response", new=AsyncMock(side_effect=hang)):
            task = asyncio.create_task(session.send("hi"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            await asyncio.wait_for(session.close(), timeout=5)  # bounded, not forever
            assert session._closed
            _ = task.cancel()
            _ = await asyncio.wait({task})


class TestClosedSessionErrors:
    async def test_every_public_operation_raises_session_closed_error(self, tmp_path):
        from mnesis import MnesisStoreError, SessionClosedError

        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        await session.close()
        assert issubclass(SessionClosedError, MnesisStoreError)

        async def drain_stream():
            async for _ in session.stream("hi"):
                pass

        for call in (
            lambda: session.send("hi"),
            lambda: session.record("q", "a"),
            drain_stream,
            session.context_for_next_turn,
            session.messages,
            session.conversation_messages,
            session.compact,
        ):
            with pytest.raises(SessionClosedError):
                _ = await call()
            # Existing ``except MnesisStoreError`` handlers keep catching it.
            with pytest.raises(MnesisStoreError):
                _ = await call()


class TestCloseOwnership:
    @staticmethod
    def _slow_stream(release: asyncio.Event, entered: asyncio.Event | None = None):
        from mnesis.models.message import TokenUsage

        async def slow(*args, **kwargs):
            if entered is not None:
                entered.set()
            await release.wait()
            return "done", TokenUsage(input=1, output=1, total=2), "stop"

        return slow

    async def test_close_inside_one_send_waits_for_the_other_sends(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        from mnesis.models.message import TokenUsage

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        release = asyncio.Event()
        slow_entered = asyncio.Event()
        calls = {"n": 0}

        async def stream(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:  # the "other" send: slow
                slow_entered.set()
                await release.wait()
            else:  # the closing send: closes the session from its own task
                await session.close()
            return "ok", TokenUsage(input=1, output=1, total=2), "stop"

        with patch.object(session, "_stream_response", new=AsyncMock(side_effect=stream)):
            other = asyncio.create_task(session.send("slow"))
            await asyncio.wait_for(slow_entered.wait(), timeout=5)
            closer = asyncio.create_task(session.send("closer"))
            await asyncio.sleep(0.1)
            # The closer's close() is waiting for the other send, not failing it.
            assert not session._closed
            release.set()
            assert (await asyncio.wait_for(other, timeout=5)).text == "ok"
            _ = await asyncio.wait({closer}, timeout=5)
        assert session._closed

    async def test_interrupted_second_close_does_not_reopen_the_session(
        self, tmp_path, monkeypatch
    ):
        from unittest.mock import AsyncMock, patch

        from mnesis import SessionClosedError

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        release = asyncio.Event()
        entered = asyncio.Event()
        slow = self._slow_stream(release, entered)
        with patch.object(session, "_stream_response", new=AsyncMock(side_effect=slow)):
            first_send = asyncio.create_task(session.send("long"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            close1 = asyncio.create_task(session.close())
            await asyncio.sleep(0.02)
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.05):
                    await session.close()  # second close, interrupted
            assert session._closing  # close #1 still owns the closing state
            with pytest.raises(SessionClosedError):
                _ = await session.send("sneaks in")
            release.set()
            _ = await asyncio.wait_for(first_send, timeout=5)
            await asyncio.wait_for(close1, timeout=5)
        assert session._closed


class TestConcurrentClose:
    async def test_two_concurrent_closes_on_an_idle_session(self, tmp_path):
        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        await asyncio.wait_for(asyncio.gather(session.close(), session.close()), timeout=5)
        assert session._closed

    async def test_two_concurrent_closes_with_a_send_in_flight(self, tmp_path, monkeypatch):
        from unittest.mock import AsyncMock, patch

        from mnesis.models.message import TokenUsage

        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        release = asyncio.Event()
        entered = asyncio.Event()

        async def slow(*args, **kwargs):
            entered.set()
            await release.wait()
            return "done", TokenUsage(input=1, output=1, total=2), "stop"

        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        with patch.object(session, "_stream_response", new=AsyncMock(side_effect=slow)):
            send = asyncio.create_task(session.send("hi"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            closes = asyncio.gather(session.close(), session.close())
            await asyncio.sleep(0.05)
            assert not closes.done()  # both wait for the stream on the wire
            release.set()
            await asyncio.wait_for(closes, timeout=5)
            assert (await asyncio.wait_for(send, timeout=5)).text == "done"
        assert session._closed

    async def test_mutual_close_inside_two_sends_completes(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        monkeypatch.setattr("mnesis.session._CLOSE_INFLIGHT_TIMEOUT", 2.0)
        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)

        def handler():
            fired = [False]

            async def on_part(part):
                if not fired[0]:
                    fired[0] = True
                    await session.close()

            return on_part

        results = await asyncio.wait_for(
            asyncio.gather(
                session.send("a", on_part=handler()),
                session.send("b", on_part=handler()),
                return_exceptions=True,
            ),
            timeout=10,
        )
        assert len(results) == 2
        assert session._closed

    async def test_later_abort_request_still_aborts_a_nonabort_close(self, tmp_path, monkeypatch):
        import mnesis.compaction.engine as engine_mod

        calls: list[float] = []
        started = asyncio.Event()

        async def slow_empty(**kwargs: object) -> str:
            calls.append(0.0)
            started.set()
            await asyncio.sleep(0.4)
            return ""

        monkeypatch.setattr(engine_mod, "_make_llm_call", lambda model, **kw: slow_empty)
        monkeypatch.setenv("MNESIS_MOCK_LLM", "0")
        session = await TestRetrySleepSemantics._session(tmp_path, base_delay=0.01)
        for i in range(4):
            _ = await session.record(f"question {i} " * 20, f"answer {i} " * 20)
        compaction = asyncio.create_task(session.compact())
        await asyncio.wait_for(started.wait(), timeout=5)
        first = asyncio.create_task(session.close())  # waits for the compaction
        await asyncio.sleep(0.02)
        second = asyncio.create_task(session.close(abort_compaction=True))
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
        result = await asyncio.wait_for(compaction, timeout=5)
        assert len(calls) == 1  # the run was aborted: no escalation to a further LLM call
        assert result.level_used == 0
        assert session._closed
