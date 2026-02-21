"""AgenticMap operator — parallel sub-agent sessions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Literal

import structlog
from jinja2 import Environment as _JinjaEnv
from pydantic import BaseModel, Field

from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.models.config import MnesisConfig, OperatorConfig
from mnesis.models.message import TokenUsage
from mnesis.operators.template_utils import require_item_variable
from mnesis.store.pool import StorePool


class AgentMapResult(BaseModel):
    """The result of running a single sub-agent through AgenticMap."""

    input: Any
    session_id: str
    output_text: str
    success: bool
    error: str | None = None
    error_kind: Literal["timeout", "validation", "llm_error", "schema_error"] | None = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    intermediate_outputs: list[str] = Field(default_factory=list)


class AgentMapBatch(BaseModel):
    """Aggregate result of a full AgenticMap.run_all() call."""

    successes: list[AgentMapResult]
    failures: list[AgentMapResult]
    total_attempts: int

    @property
    def total(self) -> int:
        return len(self.successes) + len(self.failures)


class AgenticMap:
    """
    Spawn independent sub-agent sessions per input item.

    Each item gets a full ``MnesisSession`` with multi-turn reasoning, tool access,
    and automatic context management. Sub-sessions are isolated — they do not
    inherit the parent's conversation history.

    The parent session pays only O(1) context cost per item: it sees only the
    final output text, regardless of how many turns the sub-session needed.

    Permission restrictions (always applied):
    - Sub-sessions cannot spawn further sub-agents.
    - ``read_only=True`` is not yet implemented; passing it raises ``NotImplementedError``.

    Example::

        agentic_map = AgenticMap(config.operators, model="anthropic/claude-opus-4-6")
        async for result in agentic_map.run(
            inputs=repositories,
            agent_prompt_template="Analyze this repository and report quality issues:\\n{{ item }}",
            read_only=False,
            max_turns=20,
        ):
            print(f"Repo: {result.input}\\nFindings: {result.output_text[:200]}")
    """

    def __init__(
        self,
        config: OperatorConfig,
        event_bus: EventBus | None = None,
        model: str | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._default_model = model
        self._logger = structlog.get_logger("mnesis.agentic_map")

    async def run(
        self,
        inputs: list[Any],
        agent_prompt_template: str,
        *,
        model: str | None = None,
        concurrency: int | None = None,
        read_only: bool = True,
        tools: list[Any] | None = None,
        max_turns: int = 20,
        continuation_message: str = "",
        # [Advanced] — implementation plumbing, not part of the primary API:
        parent_session_id: str | None = None,
        agent: str = "general",
        lcm_config: MnesisConfig | None = None,
        db_path: str | None = None,
        pool: StorePool | None = None,
    ) -> AsyncGenerator[AgentMapResult, None]:
        """
        Spawn sub-agent sessions in parallel, one per input item.

        This is an async generator — iterate with ``async for``, not ``await``.

        Args:
            inputs: List of items. Each is rendered into ``{{ item }}``.
            agent_prompt_template: Jinja2 template for the initial user message.
                Must reference the ``item`` variable.
            model: LLM model string for all sub-sessions. Falls back to the model
                set on ``__init__`` if not provided here.
            concurrency: Maximum concurrent sub-sessions.
            read_only: Reserved. ``True`` raises ``NotImplementedError`` until
                enforcement is implemented.
            tools: Optional tool definitions for sub-sessions.
            max_turns: Maximum turns per sub-session before stopping.
            continuation_message: Message sent as the user turn after turn 0.
                If an empty string (default), the sub-session runs only the
                initial turn and then stops, regardless of ``max_turns``.

            [Advanced] The following parameters are implementation plumbing and
            are not needed for typical use:

            parent_session_id: Optional parent session ID for lineage tracking.
            agent: Agent role for sub-sessions.
            lcm_config: Mnesis config for sub-sessions. Defaults to ``MnesisConfig()``.
            db_path: Override database path (useful for testing).
            pool: Shared ``StorePool`` so all sub-sessions use one connection.
                If omitted a fresh pool is created and closed automatically
                at the end of this call.

        Yields:
            AgentMapResult objects as sub-sessions complete (not in input order).

        Raises:
            NotImplementedError: If ``read_only=True`` is passed (not yet implemented).
            ValueError: If neither ``model`` nor the constructor ``model`` is set,
                or if ``agent_prompt_template`` does not reference ``item``.
        """
        if read_only:
            raise NotImplementedError(
                "read_only enforcement is not yet implemented. "
                "Pass read_only=False to proceed without write-tool filtering."
            )

        resolved_model = model or self._default_model
        if not resolved_model:
            raise ValueError("model must be provided either to run() or to AgenticMap.__init__()")

        # Validate template references 'item' via Jinja2 AST (not fragile regex).
        require_item_variable(agent_prompt_template)

        # Compile template once; each sub-agent task renders it with its own item.
        compiled_template = _JinjaEnv().from_string(agent_prompt_template)

        max_conc = concurrency or self._config.agentic_map_concurrency

        if self._event_bus:
            self._event_bus.publish(
                MnesisEvent.MAP_STARTED, {"total": len(inputs), "type": "agentic"}
            )

        # If no pool supplied, create a local one that we close at the end.
        # This ensures all parallel sub-sessions share a single connection to
        # the same db_path, avoiding SQLite write-lock contention.
        owned_pool = pool is None
        effective_pool = pool if pool is not None else StorePool()

        semaphore = asyncio.Semaphore(max_conc)
        tasks = [
            asyncio.create_task(
                self._run_sub_agent(
                    item=item,
                    compiled_template=compiled_template,
                    model=resolved_model,
                    parent_session_id=parent_session_id,
                    tools=tools,
                    max_turns=max_turns,
                    continuation_message=continuation_message,
                    agent=agent,
                    lcm_config=lcm_config,
                    db_path=db_path,
                    pool=effective_pool,
                    semaphore=semaphore,
                )
            )
            for item in inputs
        ]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            if self._event_bus:
                self._event_bus.publish(
                    MnesisEvent.MAP_ITEM_COMPLETED,
                    {"completed": completed, "total": len(inputs), "success": result.success},
                )
            yield result

        if self._event_bus:
            self._event_bus.publish(MnesisEvent.MAP_COMPLETED, {"total": len(inputs)})

        if owned_pool:
            await effective_pool.close_all()

    async def run_all(
        self,
        inputs: list[Any],
        agent_prompt_template: str,
        *,
        model: str | None = None,
        concurrency: int | None = None,
        read_only: bool = True,
        tools: list[Any] | None = None,
        max_turns: int = 20,
        continuation_message: str = "",
        parent_session_id: str | None = None,
        agent: str = "general",
        lcm_config: MnesisConfig | None = None,
        db_path: str | None = None,
        pool: StorePool | None = None,
    ) -> AgentMapBatch:
        """
        Spawn all sub-agents and return an aggregate ``AgentMapBatch``.

        Convenience alternative to ``run()`` for callers who do not need streaming.
        Collects all results before returning.

        Returns:
            AgentMapBatch with ``.successes``, ``.failures``, and ``.total_attempts``.
        """
        successes: list[AgentMapResult] = []
        failures: list[AgentMapResult] = []
        total_attempts = 0

        async for result in self.run(
            inputs=inputs,
            agent_prompt_template=agent_prompt_template,
            model=model,
            concurrency=concurrency,
            read_only=read_only,
            tools=tools,
            max_turns=max_turns,
            continuation_message=continuation_message,
            parent_session_id=parent_session_id,
            agent=agent,
            lcm_config=lcm_config,
            db_path=db_path,
            pool=pool,
        ):
            total_attempts += 1
            if result.success:
                successes.append(result)
            else:
                failures.append(result)

        return AgentMapBatch(
            successes=successes,
            failures=failures,
            total_attempts=total_attempts,
        )

    async def _run_sub_agent(
        self,
        *,
        item: Any,
        compiled_template: Any,
        model: str,
        parent_session_id: str | None,
        tools: list[Any] | None,
        max_turns: int,
        continuation_message: str,
        agent: str,
        lcm_config: MnesisConfig | None,
        db_path: str | None,
        pool: StorePool,
        semaphore: asyncio.Semaphore,
    ) -> AgentMapResult:
        """Run a single sub-agent session to completion."""
        async with semaphore:
            from mnesis.session import MnesisSession

            prompt = compiled_template.render(item=item)
            session: MnesisSession | None = None

            try:
                session = await MnesisSession.create(
                    model=model,
                    agent=agent,
                    parent_id=parent_session_id,
                    config=lcm_config,
                    db_path=db_path,
                    pool=pool,
                )

                output_text = ""
                intermediate_outputs: list[str] = []
                cumulative_usage = TokenUsage()

                for turn in range(max_turns):
                    if turn == 0:
                        user_message = prompt
                    elif continuation_message:
                        user_message = continuation_message
                    else:
                        # Empty continuation_message: no additional user turn.
                        # The agent has already stopped or we break on finish_reason.
                        break

                    result = await session.send(
                        user_message,
                        tools=tools,
                    )
                    output_text = result.text
                    intermediate_outputs.append(result.text)
                    cumulative_usage = cumulative_usage + result.tokens

                    # Stop conditions
                    if result.finish_reason in ("stop", "end_turn"):
                        break
                    if result.doom_loop_detected:
                        self._logger.warning(
                            "sub_agent_doom_loop",
                            session_id=session.id,
                            item=str(item)[:100],
                        )
                        break
                    if result.finish_reason == "max_tokens":
                        break

                return AgentMapResult(
                    input=item,
                    session_id=session.id,
                    output_text=output_text,
                    success=True,
                    token_usage=cumulative_usage,
                    intermediate_outputs=intermediate_outputs,
                )

            except Exception as exc:
                self._logger.error(
                    "sub_agent_failed",
                    item=str(item)[:100],
                    error=str(exc),
                )
                return AgentMapResult(
                    input=item,
                    session_id=session.id if session else "",
                    output_text="",
                    success=False,
                    error=str(exc),
                    error_kind="llm_error",
                )
            finally:
                if session is not None:
                    await session.close()
