"""Mnesis Session — the primary public API entry point."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import structlog
from ulid import ULID

from mnesis.compaction.engine import CompactionEngine
from mnesis.context.builder import ContextBuilder
from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.models.config import MnesisConfig, ModelInfo, StoreConfig
from mnesis.models.message import (
    CompactionResult,
    Message,
    MessagePart,
    MessageWithParts,
    RecordResult,
    TextPart,
    TokenUsage,
    ToolPart,
    TurnResult,
)
from mnesis.store.immutable import ImmutableStore, RawMessagePart
from mnesis.store.pool import StorePool
from mnesis.store.summary_dag import SummaryDAGStore
from mnesis.tokens.estimator import TokenEstimator


def make_id(prefix: str) -> str:
    """
    Generate a ULID-based sortable identifier.

    Args:
        prefix: Short prefix for readability (e.g. ``"msg"``, ``"sess"``, ``"part"``).

    Returns:
        ID string in the format ``"{prefix}_{ulid}"``.
    """
    return f"{prefix}_{ULID()}"


class MnesisSession:
    """
    A single Mnesis session managing a conversation with an LLM.

    Handles the complete lifecycle: message persistence, context assembly,
    streaming, compaction, and cleanup.

    Usage::

        # Preferred: open() returns an async context manager directly
        async with MnesisSession.open(model="anthropic/claude-opus-4-6") as session:
            result = await session.send("Hello!")
            print(result.text)

        # Alternative: create() + async with (equivalent, for explicit lifecycle)
        async with await MnesisSession.create(model="anthropic/claude-opus-4-6") as session:
            result = await session.send("Hello!")

        # Manual lifecycle (no context manager)
        session = await MnesisSession.create(model="anthropic/claude-opus-4-6")
        result = await session.send("Hello!")
        await session.close()

    **BYO-LLM (bring-your-own LLM client)**

    If you manage LLM calls yourself (Anthropic SDK, OpenAI SDK, Gemini, etc.)
    and only want Mnesis for memory management, use :meth:`record` together
    with :meth:`context_for_next_turn`::

        session = await MnesisSession.create(model="anthropic/claude-opus-4-6")

        # Get the compaction-aware context window for your LLM call
        messages = await session.context_for_next_turn()
        response = your_client.chat(messages=messages, system=your_system_prompt)

        # Persist the completed turn so Mnesis tracks tokens and compaction
        await session.record(
            user_message="Explain quantum entanglement.",
            assistant_response=response.text,
            tokens=TokenUsage(input=response.usage.input, output=response.usage.output),
        )

    See ``docs/byo-llm.md`` and ``examples/06_byo_llm.py`` for a full walkthrough.
    """

    def __init__(
        self,
        session_id: str,
        model: str,
        model_info: ModelInfo,
        config: MnesisConfig,
        system_prompt: str,
        agent: str,
        store: ImmutableStore,
        dag_store: SummaryDAGStore,
        context_builder: ContextBuilder,
        compaction_engine: CompactionEngine,
        token_estimator: TokenEstimator,
        event_bus: EventBus,
    ) -> None:
        self._session_id = session_id
        self._model = model
        self._model_info = model_info
        self._config = config
        self._system_prompt = system_prompt
        self._agent = agent
        self._store = store
        self._dag_store = dag_store
        self._context_builder = context_builder
        self._compaction_engine = compaction_engine
        self._estimator = token_estimator
        self._event_bus = event_bus
        self._cumulative_tokens = TokenUsage()
        self._recent_tool_calls: list[tuple[str, str]] = []  # (tool_name, input_json)
        self._logger = structlog.get_logger("mnesis.session").bind(session_id=session_id)

    @classmethod
    async def create(
        cls,
        *,
        model: str,
        agent: str = "default",
        parent_id: str | None = None,
        config: MnesisConfig | None = None,
        system_prompt: str = "You are a helpful assistant.",
        db_path: str | None = None,
        pool: StorePool | None = None,
    ) -> MnesisSession:
        """
        Create a new Mnesis session.

        Args:
            model: LLM model string in litellm format (e.g. ``"anthropic/claude-opus-4-6"``).
            agent: Agent role name for multi-agent setups.
            parent_id: Parent session ID for sub-sessions (AgenticMap).
            config: Mnesis configuration. Defaults to ``MnesisConfig()``.
            system_prompt: System prompt for all turns in this session.
            db_path: Override database path (useful for testing). Raises ``ValueError``
                if both ``db_path`` and ``config.store.db_path`` are supplied.
            pool: Optional shared connection pool.  When provided, all sessions
                pointing at the same ``db_path`` share a single connection,
                avoiding SQLite write-lock contention between concurrent
                sub-agents.  The caller is responsible for calling
                ``pool.close_all()`` at shutdown.

        Returns:
            An initialized MnesisSession ready to receive messages.

        Raises:
            ValueError: If both ``db_path`` and ``config.store.db_path`` are supplied.
            aiosqlite.Error: If the database cannot be initialized.
        """
        cfg = config or MnesisConfig()
        if db_path is not None:
            _default_db_path = StoreConfig().db_path
            if config is not None and cfg.store.db_path != _default_db_path:
                raise ValueError(
                    "Specify db_path either via db_path= or config.store.db_path, not both."
                )
            cfg = cfg.model_copy(
                update={"store": cfg.store.model_copy(update={"db_path": db_path})}
            )

        store = ImmutableStore(cfg.store, pool=pool)
        await store.initialize()

        session_id = make_id("sess")
        model_info = ModelInfo.from_model_string(model)
        if cfg.model_overrides:
            model_info = model_info.model_copy(update=cfg.model_overrides)
        provider = model_info.provider_id

        await store.create_session(
            session_id,
            model_id=model,
            provider_id=provider,
            agent=agent,
            parent_id=parent_id,
            system_prompt=system_prompt,
        )

        dag_store = SummaryDAGStore(store)
        estimator = TokenEstimator()
        event_bus = EventBus()
        context_builder = ContextBuilder(store, dag_store, estimator)
        compaction_engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            cfg,
            id_generator=make_id,
            session_model=model,
        )

        session = cls(
            session_id=session_id,
            model=model,
            model_info=model_info,
            config=cfg,
            system_prompt=system_prompt,
            agent=agent,
            store=store,
            dag_store=dag_store,
            context_builder=context_builder,
            compaction_engine=compaction_engine,
            token_estimator=estimator,
            event_bus=event_bus,
        )

        event_bus.publish(MnesisEvent.SESSION_CREATED, {"session_id": session_id, "model": model})
        structlog.get_logger("mnesis.session").info(
            "session_created", session_id=session_id, model=model, agent=agent
        )
        return session

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        *,
        model: str,
        agent: str = "default",
        parent_id: str | None = None,
        config: MnesisConfig | None = None,
        system_prompt: str = "You are a helpful assistant.",
        db_path: str | None = None,
        pool: StorePool | None = None,
    ) -> AsyncGenerator[MnesisSession, None]:
        """
        Create a new session and use it as an async context manager.

        This is the preferred idiom for most callers — it avoids the
        ``await ... async with`` double-ceremony of :meth:`create`::

            async with MnesisSession.open(model="anthropic/claude-opus-4-6") as session:
                result = await session.send("Hello!")

        All parameters are identical to :meth:`create`. The session is
        automatically closed (pending compaction awaited, DB connection
        released) when the ``async with`` block exits, even on exception.

        Args:
            model: LLM model string in litellm format.
            agent: Agent role name for multi-agent setups.
            parent_id: Parent session ID for sub-sessions (AgenticMap).
            config: Mnesis configuration. Defaults to ``MnesisConfig()``.
            system_prompt: System prompt for all turns in this session.
            db_path: Override database path.
            pool: Optional shared connection pool (see :meth:`create`).

        Yields:
            An initialized MnesisSession.

        Note:
            To receive operator events (``LLMMap``, ``AgenticMap``), pass
            ``event_bus=session.event_bus`` when constructing operators.
        """
        session = await cls.create(
            model=model,
            agent=agent,
            parent_id=parent_id,
            config=config,
            system_prompt=system_prompt,
            db_path=db_path,
            pool=pool,
        )
        try:
            yield session
        finally:
            await session.close()

    @classmethod
    async def load(
        cls,
        session_id: str,
        config: MnesisConfig | None = None,
        db_path: str | None = None,
        pool: StorePool | None = None,
    ) -> MnesisSession:
        """
        Load an existing session from the store.

        Args:
            session_id: The session ID to load.
            config: Optional config override.
            db_path: Override database path.
            pool: Optional shared connection pool (see ``create()``).

        Returns:
            An MnesisSession wrapping the existing session.

        Raises:
            SessionNotFoundError: If the session does not exist.
        """
        cfg = config or MnesisConfig()
        if db_path is not None:
            _default_db_path = StoreConfig().db_path
            if config is not None and cfg.store.db_path != _default_db_path:
                raise ValueError(
                    "Specify db_path either via db_path= or config.store.db_path, not both."
                )
            cfg = cfg.model_copy(
                update={"store": cfg.store.model_copy(update={"db_path": db_path})}
            )

        store = ImmutableStore(cfg.store, pool=pool)
        await store.initialize()

        db_session = await store.get_session(session_id)
        if not db_session.model_id:
            raise ValueError(
                f"Session {session_id!r} has no stored model_id; pass model= explicitly"
            )
        model = db_session.model_id
        model_info = ModelInfo.from_model_string(model)
        if cfg.model_overrides:
            model_info = model_info.model_copy(update=cfg.model_overrides)

        dag_store = SummaryDAGStore(store)
        estimator = TokenEstimator()
        event_bus = EventBus()
        context_builder = ContextBuilder(store, dag_store, estimator)
        compaction_engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            cfg,
            id_generator=make_id,
            session_model=model,
        )

        return cls(
            session_id=session_id,
            model=model,
            model_info=model_info,
            config=cfg,
            system_prompt=db_session.system_prompt,
            agent=db_session.agent,
            store=store,
            dag_store=dag_store,
            context_builder=context_builder,
            compaction_engine=compaction_engine,
            token_estimator=estimator,
            event_bus=event_bus,
        )

    async def send(
        self,
        message: str | list[MessagePart],
        *,
        tools: list[Any] | None = None,
        on_part: Callable[[MessagePart], None | Awaitable[None]] | None = None,
        system_prompt: str | None = None,
    ) -> TurnResult:
        """
        Send a user message and receive a streaming assistant response.

        Args:
            message: User message text or list of MessagePart objects.
            tools: Optional list of tool definitions (litellm format dicts).
            on_part: Optional callback invoked for each streamed ``MessagePart``.
                During streaming, ``on_part`` receives :class:`TextPart` chunks
                only. Note: :class:`ToolPart` objects are never delivered via
                ``on_part`` during ``send()``; they are only available when
                passing pre-built parts to :meth:`record`.
            system_prompt: Override the session system prompt for this turn only.
                Note: not all providers support per-turn system prompt overrides —
                check your provider's documentation before relying on this.

        Returns:
            TurnResult with the assistant's text, token usage, and status.

        Raises:
            MnesisStoreError: If message persistence fails.
        """
        sys_prompt = system_prompt or self._system_prompt

        # Normalize message to parts
        if isinstance(message, str):
            user_parts: list[MessagePart] = [TextPart(text=message)]
        else:
            user_parts = list(message)

        # Persist user message
        user_msg_id = make_id("msg")
        user_msg = Message(
            id=user_msg_id,
            session_id=self._session_id,
            role="user",
            agent=self._agent,
            model_id=self._model,
        )
        await self._store.append_message(user_msg)

        for part in user_parts:
            part_id = make_id("part")
            raw = RawMessagePart(
                id=part_id,
                message_id=user_msg_id,
                session_id=self._session_id,
                part_type=part.type,
                content=json.dumps(part.model_dump()),
            )
            await self._store.append_part(raw)

        self._event_bus.publish(
            MnesisEvent.MESSAGE_CREATED, {"message_id": user_msg_id, "role": "user"}
        )

        # Hard threshold check: if the context is over the hard limit, ensure
        # compaction is triggered (if not already in flight) and then block
        # until it completes.  This prevents an over-limit context reaching the LLM.
        if self._compaction_engine.is_hard_overflow(self._cumulative_tokens, self._model_info):
            if not self._compaction_engine._pending_task:
                self._compaction_engine.check_and_trigger(
                    self._session_id,
                    self._cumulative_tokens,
                    self._model_info,
                )
            await self._compaction_engine.wait_for_pending()

        # Build context
        context = await self._context_builder.build(
            self._session_id,
            self._model_info,
            sys_prompt,
            self._config,
        )

        # Prepare LLM messages
        llm_messages = [{"role": m.role, "content": m.content} for m in context.messages]

        # Stream LLM response
        assistant_msg_id = make_id("msg")
        assistant_msg = Message(
            id=assistant_msg_id,
            session_id=self._session_id,
            role="assistant",
            agent=self._agent,
            model_id=self._model,
            provider_id=self._model_info.provider_id,
        )
        await self._store.append_message(assistant_msg)

        text_accumulator = ""
        final_tokens = TokenUsage()
        finish_reason = "stop"
        compaction_triggered = False
        compaction_result_obj: CompactionResult | None = None

        try:
            # Check for mock mode (for examples without API keys)
            import os

            if os.environ.get("MNESIS_MOCK_LLM") == "1":
                text_accumulator, final_tokens, finish_reason = await self._mock_response(
                    llm_messages, on_part, assistant_msg_id
                )
            else:
                text_accumulator, final_tokens, finish_reason = await self._stream_response(
                    llm_messages, sys_prompt, tools, on_part, assistant_msg_id
                )
        except Exception as exc:
            self._logger.error("llm_call_failed", error=str(exc))
            finish_reason = "error"
            text_accumulator = f"[Error: {exc}]"

        # Persist final text part
        text_part_id = make_id("part")
        text_raw = RawMessagePart(
            id=text_part_id,
            message_id=assistant_msg_id,
            session_id=self._session_id,
            part_type="text",
            content=json.dumps({"type": "text", "text": text_accumulator}),
            token_estimate=self._estimator.estimate(text_accumulator, self._model_info),
        )
        await self._store.append_part(text_raw)

        # Update message with token usage
        await self._store.update_message_tokens(assistant_msg_id, final_tokens, 0.0, finish_reason)

        # Accumulate session-level tokens
        self._cumulative_tokens = self._cumulative_tokens + final_tokens

        # Doom loop detection
        doom_loop = self._check_doom_loop()

        # Check for overflow and trigger compaction
        if self._compaction_engine.is_overflow(self._cumulative_tokens, self._model_info):
            triggered = self._compaction_engine.check_and_trigger(
                self._session_id, self._cumulative_tokens, self._model_info
            )
            compaction_triggered = triggered

        self._event_bus.publish(
            MnesisEvent.MESSAGE_CREATED, {"message_id": assistant_msg_id, "role": "assistant"}
        )

        return TurnResult(
            message_id=assistant_msg_id,
            text=text_accumulator,
            finish_reason=finish_reason,
            tokens=final_tokens,
            cost=0.0,
            compaction_triggered=compaction_triggered,
            compaction_result=compaction_result_obj,
            doom_loop_detected=doom_loop,
        )

    async def _stream_response(
        self,
        llm_messages: list[dict[str, Any]],
        system_prompt: str,
        tools: list[Any] | None,
        on_part: Callable[[MessagePart], None | Awaitable[None]] | None,
        assistant_msg_id: str,
    ) -> tuple[str, TokenUsage, str]:
        """Stream response from LLM provider via litellm."""
        import litellm

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system_prompt}, *llm_messages],
            "stream": True,
            "max_tokens": self._model_info.max_output_tokens,
        }
        if tools:
            call_kwargs["tools"] = tools

        text_accumulator = ""
        usage = TokenUsage()
        finish_reason = "stop"

        async for chunk in await litellm.acompletion(**call_kwargs):
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Text delta
            if delta.content:
                text_accumulator += delta.content
                part = TextPart(text=delta.content)
                if on_part is not None:
                    result = on_part(part)
                    if asyncio.iscoroutine(result):
                        await result

            # Finish reason
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            # Usage (usually in last chunk)
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    input=getattr(chunk.usage, "prompt_tokens", 0),
                    output=getattr(chunk.usage, "completion_tokens", 0),
                    total=getattr(chunk.usage, "total_tokens", 0),
                )

        return text_accumulator, usage, finish_reason

    async def _mock_response(
        self,
        llm_messages: list[dict[str, Any]],
        on_part: Callable[[MessagePart], None | Awaitable[None]] | None,
        assistant_msg_id: str,
    ) -> tuple[str, TokenUsage, str]:
        """Return a mock response for demonstration purposes (MNESIS_MOCK_LLM=1)."""
        last_user = next(
            (m["content"] for m in reversed(llm_messages) if m["role"] == "user"),
            "Hello",
        )
        mock_text = (
            f"[Mock LLM response to: {str(last_user)[:100]}]\n"
            "This is a simulated response for demonstration purposes. "
            "Set MNESIS_MOCK_LLM=0 and provide an API key to use a real LLM."
        )

        # Simulate streaming
        for word in mock_text.split():
            part = TextPart(text=word + " ")
            if on_part is not None:
                result = on_part(part)
                if asyncio.iscoroutine(result):
                    await result

        usage = TokenUsage(
            input=self._estimator.estimate(str(llm_messages)),
            output=self._estimator.estimate(mock_text),
        )
        return mock_text, usage, "stop"

    def _check_doom_loop(self) -> bool:
        """Detect consecutive identical tool calls."""
        threshold = self._config.session.doom_loop_threshold
        if len(self._recent_tool_calls) < threshold:
            return False
        last_n = self._recent_tool_calls[-threshold:]
        first = last_n[0]
        detected = all(t == first for t in last_n[1:])
        if detected:
            self._logger.warning("doom_loop_detected", tool=first[0])
            self._event_bus.publish(
                MnesisEvent.DOOM_LOOP_DETECTED,
                {"session_id": self._session_id, "tool": first[0]},
            )
        return detected

    async def record(
        self,
        user_message: str | list[MessagePart],
        assistant_response: str | list[MessagePart],
        *,
        tokens: TokenUsage | None = None,
        finish_reason: str = "stop",
    ) -> RecordResult:
        """
        Persist a completed user/assistant turn without making an LLM call.

        Use this when you manage LLM calls yourself (e.g. using the Anthropic,
        OpenAI, or Gemini SDKs directly) and only want mnesis to handle
        memory, context assembly, and compaction.

        Args:
            user_message: The user message text or parts to record.
            assistant_response: The assistant reply text or parts to record.
            tokens: Token usage for the turn. Estimated from text if omitted.
            finish_reason: The finish reason from your LLM response (e.g.
                ``"stop"``, ``"max_tokens"``). Defaults to ``"stop"``.

        Returns:
            RecordResult with the persisted message IDs and token usage.

        Example::

            import anthropic
            from mnesis import MnesisSession
            from mnesis.models.message import TokenUsage

            client = anthropic.Anthropic()
            session = await MnesisSession.create(model="anthropic/claude-opus-4-6")

            user_text = "Explain quantum entanglement."
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=1024,
                messages=[{"role": "user", "content": user_text}],
            )
            result = await session.record(
                user_message=user_text,
                assistant_response=response.content[0].text,
                tokens=TokenUsage(
                    input=response.usage.input_tokens,
                    output=response.usage.output_tokens,
                ),
            )
        """
        # Normalise inputs to part lists
        if isinstance(user_message, str):
            user_parts: list[MessagePart] = [TextPart(text=user_message)]
        else:
            user_parts = list(user_message)

        if isinstance(assistant_response, str):
            assistant_parts: list[MessagePart] = [TextPart(text=assistant_response)]
        else:
            assistant_parts = list(assistant_response)

        # Persist user message
        user_msg_id = make_id("msg")
        user_msg = Message(
            id=user_msg_id,
            session_id=self._session_id,
            role="user",
            agent=self._agent,
            model_id=self._model,
        )
        await self._store.append_message(user_msg)
        for part in user_parts:
            raw = RawMessagePart(
                id=make_id("part"),
                message_id=user_msg_id,
                session_id=self._session_id,
                part_type=part.type,
                content=json.dumps(part.model_dump()),
            )
            await self._store.append_part(raw)

        self._event_bus.publish(
            MnesisEvent.MESSAGE_CREATED, {"message_id": user_msg_id, "role": "user"}
        )

        # Persist assistant message
        assistant_msg_id = make_id("msg")
        assistant_msg = Message(
            id=assistant_msg_id,
            session_id=self._session_id,
            role="assistant",
            agent=self._agent,
            model_id=self._model,
            provider_id=self._model_info.provider_id,
        )
        await self._store.append_message(assistant_msg)

        assistant_output_tokens = 0
        for part in assistant_parts:
            token_estimate = 0
            tool_call_id: str | None = None
            tool_name: str | None = None
            tool_state: str | None = None
            if isinstance(part, TextPart):
                token_estimate = self._estimator.estimate(part.text, self._model_info)
            elif isinstance(part, ToolPart):
                tool_call_id = part.tool_call_id
                tool_name = part.tool_name
                tool_state = part.status.state
                tool_segments: list[str] = []
                if part.input:
                    tool_segments.append(
                        part.input if isinstance(part.input, str) else json.dumps(part.input)
                    )
                if part.output:
                    tool_segments.append(part.output)
                if part.error_message:
                    tool_segments.append(part.error_message)
                token_estimate = self._estimator.estimate(
                    "\n".join(tool_segments), self._model_info
                )
            else:
                # Other part types — repr-length heuristic, mirrors estimate_message()
                token_estimate = max(1, len(str(part)) // 4)
            assistant_output_tokens += token_estimate
            raw = RawMessagePart(
                id=make_id("part"),
                message_id=assistant_msg_id,
                session_id=self._session_id,
                part_type=part.type,
                content=json.dumps(part.model_dump()),
                token_estimate=token_estimate,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_state=tool_state,
            )
            await self._store.append_part(raw)

        # Resolve token usage — estimate from all parts if not provided
        if tokens is None:
            user_input_tokens = sum(
                self._estimator.estimate(p.text, self._model_info)
                if isinstance(p, TextPart)
                else max(1, len(str(p)) // 4)
                for p in user_parts
            )
            tokens = TokenUsage(
                input=user_input_tokens,
                output=assistant_output_tokens,
            )

        await self._store.update_message_tokens(assistant_msg_id, tokens, 0.0, finish_reason)

        self._cumulative_tokens = self._cumulative_tokens + tokens

        self._event_bus.publish(
            MnesisEvent.MESSAGE_CREATED,
            {"message_id": assistant_msg_id, "role": "assistant"},
        )

        # Check for overflow and trigger compaction
        compaction_triggered = False
        if self._compaction_engine.is_overflow(self._cumulative_tokens, self._model_info):
            compaction_triggered = self._compaction_engine.check_and_trigger(
                self._session_id, self._cumulative_tokens, self._model_info
            )

        self._logger.info(
            "turn_recorded",
            user_message_id=user_msg_id,
            assistant_message_id=assistant_msg_id,
        )

        return RecordResult(
            user_message_id=user_msg_id,
            assistant_message_id=assistant_msg_id,
            tokens=tokens,
            compaction_triggered=compaction_triggered,
        )

    async def messages(self) -> list[MessageWithParts]:
        """
        Return the full message history for this session.

        The returned list includes **all** stored rows:

        - Regular user and assistant turns.
        - Compaction summary messages (``m.is_summary == True``).
        - Messages whose parts contain :class:`~mnesis.models.message.CompactionMarkerPart`
          tombstones (tool outputs pruned by the compaction engine).

        To get only the conversational turns, use :meth:`conversation_messages`
        or filter manually with ``[m for m in msgs if not m.is_summary]``.

        Returns:
            List of MessageWithParts in chronological order.
        """
        return await self._store.get_messages_with_parts(self._session_id)

    async def conversation_messages(self) -> list[MessageWithParts]:
        """
        Return only the conversational turns, excluding compaction summaries.

        Convenience wrapper around :meth:`messages` that filters out summary
        messages produced by the compaction engine (``is_summary == True``).
        Tombstoned tool outputs within regular messages are still included —
        only top-level summary rows are removed.

        Returns:
            List of non-summary MessageWithParts in chronological order.
        """
        all_msgs = await self._store.get_messages_with_parts(self._session_id)
        return [m for m in all_msgs if not m.is_summary]

    async def context_for_next_turn(
        self,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the compaction-aware context window for the next LLM call.

        Intended for **BYO-LLM** callers who manage their own LLM client but
        want Mnesis to handle context assembly and compaction.  The returned
        list is in the format expected by most chat completion APIs
        (``[{"role": "user"|"assistant", "content": "..."}, ...]``).

        Summaries injected by the compaction engine are included as assistant
        messages at the correct position, so the context is always accurate
        even after one or more compaction cycles.

        After calling your LLM, persist the completed turn with
        :meth:`record` so Mnesis can track token usage and trigger compaction
        when needed.

        Args:
            system_prompt: Override the session system prompt for this turn.
                Defaults to the system prompt set at :meth:`create` / :meth:`load`.

        Returns:
            Ordered list of ``{"role": str, "content": str | list}`` dicts
            ready to pass to any chat completion API.

        Example::

            messages = await session.context_for_next_turn()
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": my_system_prompt}, *messages],
            )
            await session.record(
                user_message=my_user_text,
                assistant_response=response.choices[0].message.content,
            )
        """
        sys = self._system_prompt if system_prompt is None else system_prompt
        context = await self._context_builder.build(
            self._session_id, self._model_info, sys, self._config
        )
        return [{"role": m.role, "content": m.content} for m in context.messages]

    async def compact(self) -> CompactionResult:
        """
        Manually trigger synchronous compaction.

        Blocks until compaction completes. Useful for checkpointing before
        complex operations or for testing.

        Returns:
            CompactionResult describing the compaction outcome.
        """
        self._logger.info("manual_compaction_triggered", session_id=self._session_id)
        return await self._compaction_engine.run_compaction(self._session_id)

    async def close(self) -> None:
        """
        Clean up session resources.

        If background compaction is in progress, ``close()`` waits for it to
        finish before releasing the database connection. This guarantees that
        no in-flight write is interrupted and that the compaction summary is
        fully persisted before the DB handle is closed.

        Publishes :attr:`~mnesis.events.bus.MnesisEvent.SESSION_CLOSED` after
        cleanup.
        """
        await self._compaction_engine.wait_for_pending()
        self._event_bus.publish(MnesisEvent.SESSION_CLOSED, {"session_id": self._session_id})
        await self._store.close()
        self._logger.info("session_closed", session_id=self._session_id)

    async def __aenter__(self) -> MnesisSession:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    @property
    def id(self) -> str:
        """The session ID."""
        return self._session_id

    @property
    def model(self) -> str:
        """The model string for this session."""
        return self._model

    @property
    def token_usage(self) -> TokenUsage:
        """Cumulative token usage across all turns in this session."""
        return self._cumulative_tokens

    @property
    def compaction_in_progress(self) -> bool:
        """``True`` while a background compaction task is running."""
        task = self._compaction_engine._pending_task
        return task is not None and not task.done()

    @property
    def event_bus(self) -> EventBus:
        """The event bus for this session. Subscribe to monitor events."""
        return self._event_bus

    def subscribe(self, event: MnesisEvent, handler: Any) -> None:
        """
        Register an event handler on this session's event bus.

        Convenience wrapper for ``session.event_bus.subscribe()``.
        """
        self._event_bus.subscribe(event, handler)
