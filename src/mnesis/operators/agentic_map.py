"""AgenticMap operator — parallel sub-agent sessions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import structlog
from jinja2 import Template
from pydantic import BaseModel

from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.models.config import MnesisConfig, OperatorConfig
from mnesis.models.message import TokenUsage
from mnesis.store.pool import StorePool


class AgentMapResult(BaseModel):
    """The result of running a single sub-agent through AgenticMap."""

    input: Any
    session_id: str
    output_text: str
    success: bool
    error: str | None = None
    token_usage: TokenUsage = TokenUsage()


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
    - When ``read_only=True``, write-oriented tools are blocked.

    Example::

        agentic_map = AgenticMap(config.operators)
        async for result in agentic_map.run(
            inputs=repositories,
            agent_prompt_template="Analyze this repository and report quality issues:\\n{{ item }}",
            model="anthropic/claude-opus-4-6",
            max_turns=20,
        ):
            print(f"Repo: {result.input}\\nFindings: {result.output_text[:200]}")
    """

    def __init__(
        self,
        config: OperatorConfig,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._logger = structlog.get_logger("mnesis.agentic_map")

    async def run(
        self,
        inputs: list[Any],
        agent_prompt_template: str,
        model: str,
        *,
        concurrency: int | None = None,
        read_only: bool = True,
        parent_session_id: str | None = None,
        tools: list[Any] | None = None,
        max_turns: int = 20,
        agent: str = "general",
        lcm_config: MnesisConfig | None = None,
        db_path: str | None = None,
        pool: StorePool | None = None,
    ) -> AsyncIterator[AgentMapResult]:
        """
        Spawn sub-agent sessions in parallel, one per input item.

        Args:
            inputs: List of items. Each is rendered into ``{{ item }}``.
            agent_prompt_template: Jinja2 template for the initial user message.
            model: LLM model string for all sub-sessions.
            concurrency: Maximum concurrent sub-sessions.
            read_only: Block write-oriented tools in sub-sessions.
            parent_session_id: Optional parent session ID for lineage tracking.
            tools: Optional tool definitions for sub-sessions.
            max_turns: Maximum turns per sub-session before stopping.
            agent: Agent role for sub-sessions.
            lcm_config: Mnesis config for sub-sessions. Defaults to MnesisConfig.default().
            db_path: Override database path (useful for testing).
            pool: Shared ``StorePool`` so all sub-sessions use one connection.
                If omitted a fresh pool is created and closed automatically
                at the end of this call.

        Yields:
            AgentMapResult objects as sub-sessions complete (not in input order).

        Raises:
            ValueError: If ``agent_prompt_template`` does not contain ``{{ item }}``.
        """
        if "{{ item }}" not in agent_prompt_template and "{{item}}" not in agent_prompt_template:
            raise ValueError("agent_prompt_template must contain {{ item }}")

        max_conc = concurrency or self._config.agentic_map_concurrency
        template = Template(agent_prompt_template)

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
                    template=template,
                    model=model,
                    parent_session_id=parent_session_id,
                    tools=tools,
                    max_turns=max_turns,
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
            self._event_bus.publish(
                MnesisEvent.MAP_COMPLETED, {"total": len(inputs)}
            )

        if owned_pool:
            await effective_pool.close_all()

    async def _run_sub_agent(
        self,
        *,
        item: Any,
        template: Template,
        model: str,
        parent_session_id: str | None,
        tools: list[Any] | None,
        max_turns: int,
        agent: str,
        lcm_config: MnesisConfig | None,
        db_path: str | None,
        pool: StorePool,
        semaphore: asyncio.Semaphore,
    ) -> AgentMapResult:
        """Run a single sub-agent session to completion."""
        async with semaphore:
            from mnesis.session import MnesisSession

            prompt = template.render(item=item)
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
                cumulative_usage = TokenUsage()

                for turn in range(max_turns):
                    result = await session.send(
                        prompt if turn == 0 else "Continue.",
                        tools=tools,
                    )
                    output_text = result.text
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
                )
            finally:
                if session is not None:
                    await session.close()
