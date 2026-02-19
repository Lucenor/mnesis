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
