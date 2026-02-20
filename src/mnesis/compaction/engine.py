"""Three-level compaction escalation engine."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from mnesis.compaction.levels import (
    SummaryCandidate,
    level1_summarise,
    level2_summarise,
    level3_deterministic,
)
from mnesis.compaction.pruner import ToolOutputPrunerAsync
from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.models.config import MnesisConfig, ModelInfo
from mnesis.models.message import CompactionResult, ContextBudget, TokenUsage
from mnesis.models.summary import SummaryNode
from mnesis.store.immutable import ImmutableStore
from mnesis.store.summary_dag import SummaryDAGStore
from mnesis.tokens.estimator import TokenEstimator


def _make_llm_call(model: str) -> Any:
    """Return an async function that calls an LLM for compaction."""

    async def _call(*, model: str = model, messages: list[dict[str, str]], max_tokens: int) -> str:
        import os

        if os.environ.get("MNESIS_MOCK_LLM") == "1":
            content = messages[0]["content"] if messages else ""
            conv_text = ""
            if "<conversation>" in content:
                conv_text = content.split("<conversation>")[1].split("</conversation>")[0].strip()
            lines = [ln.strip() for ln in conv_text.splitlines() if ln.strip()]
            bullets = "\n".join(f"- {ln[:120]}" for ln in lines[:8]) or "- (session in progress)"
            return (
                "## Goal\nComplete the described task.\n\n"
                "## Completed Work\n" + bullets + "\n\n"
                "## In Progress\nContinuing as directed.\n\n"
                "## Remaining Work\n- Follow up on outstanding items.\n"
            )

        import litellm

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""

    return _call


class CompactionEngine:
    """
    Orchestrates the three-level compaction escalation protocol.

    Guarantees:
    - ``run_compaction()`` never raises — errors are caught and escalation occurs.
    - The resulting summary always fits within the token budget.
    - Level 3 (deterministic) is the final fallback and always succeeds.
    - Atomic SQLite commit: partial failures leave no inconsistent state.
    - The EventBus receives ``COMPACTION_COMPLETED`` (or ``COMPACTION_FAILED``) after each run.

    Example::

        engine = CompactionEngine(store, dag_store, estimator, event_bus, config)
        if engine.is_overflow(tokens, model_info):
            engine.check_and_trigger(session_id, tokens, model_info)  # async, non-blocking
    """

    def __init__(
        self,
        store: ImmutableStore,
        dag_store: SummaryDAGStore,
        token_estimator: TokenEstimator,
        event_bus: EventBus,
        config: MnesisConfig,
        id_generator: Any = None,
    ) -> None:
        self._store = store
        self._dag_store = dag_store
        self._estimator = token_estimator
        self._event_bus = event_bus
        self._config = config
        self._pruner = ToolOutputPrunerAsync(store, token_estimator, config)
        self._id_gen = id_generator or _default_id_generator
        self._logger = structlog.get_logger("mnesis.compaction")
        self._pending_task: asyncio.Task[CompactionResult] | None = None

    def is_overflow(self, tokens: TokenUsage, model: ModelInfo) -> bool:
        """
        Return True if compaction should be triggered.

        Args:
            tokens: Current cumulative token usage for the session.
            model: Model metadata providing context limit.

        Returns:
            True if tokens exceed the usable budget threshold.
        """
        if not self._config.compaction.auto:
            return False
        if model.context_limit == 0:
            return False  # Unlimited context (local models)

        buffer = self._config.compaction.buffer
        reserved = model.max_output_tokens
        usable = model.context_limit - reserved - buffer
        return tokens.effective_total() >= usable

    def check_and_trigger(
        self,
        session_id: str,
        tokens: TokenUsage,
        model: ModelInfo,
        abort: asyncio.Event | None = None,
    ) -> bool:
        """
        Check for overflow and trigger async background compaction if needed.

        Non-blocking — schedules a background task and returns immediately.
        The task handle is stored in ``self._pending_task`` so callers can
        await or cancel it during shutdown.

        Args:
            session_id: The session to compact.
            tokens: Current token usage.
            model: Model info for overflow detection.
            abort: Optional event to signal early termination.

        Returns:
            True if compaction was triggered, False otherwise.
        """
        if not self.is_overflow(tokens, model):
            return False

        self._logger.info(
            "compaction_triggered",
            session_id=session_id,
            tokens=tokens.effective_total(),
        )
        self._event_bus.publish(
            MnesisEvent.COMPACTION_TRIGGERED,
            {"session_id": session_id, "tokens": tokens.effective_total()},
        )

        task = asyncio.create_task(self.run_compaction(session_id, abort=abort))
        self._pending_task = task
        return True

    async def wait_for_pending(self) -> None:
        """Await any in-flight background compaction task, then clear it."""
        task = self._pending_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._pending_task = None

    async def run_compaction(
        self,
        session_id: str,
        abort: asyncio.Event | None = None,
        model_override: str | None = None,
    ) -> CompactionResult:
        """
        Run the full three-level escalation. Never raises.

        Steps:
        1. Run tool output pruner (reduce input size first).
        2. Attempt Level 1 (structured LLM summarisation).
        3. If Level 1 fails → attempt Level 2 (aggressive compression).
        4. If Level 2 fails → run Level 3 (deterministic, guaranteed to fit).
        5. Commit atomically.
        6. Publish COMPACTION_COMPLETED event.

        Args:
            session_id: The session to compact.
            abort: Optional asyncio.Event — checked before each level attempt.
            model_override: Override compaction model (for testing).

        Returns:
            CompactionResult describing what happened.
        """
        start_ms = time.time() * 1000

        try:
            return await self._run_compaction_inner(
                session_id, abort=abort, model_override=model_override
            )
        except Exception as exc:
            elapsed = time.time() * 1000 - start_ms
            self._logger.error(
                "compaction_unexpected_error",
                session_id=session_id,
                error=str(exc),
                elapsed_ms=elapsed,
            )
            self._event_bus.publish(
                MnesisEvent.COMPACTION_FAILED,
                {"session_id": session_id, "error": str(exc)},
            )
            # Return a stub result indicating failure without crashing
            return CompactionResult(
                session_id=session_id,
                summary_message_id="",
                level_used=0,
                compacted_message_count=0,
                summary_token_count=0,
                tokens_before=0,
                tokens_after=0,
                elapsed_ms=elapsed,
            )

    async def _run_compaction_inner(
        self,
        session_id: str,
        abort: asyncio.Event | None,
        model_override: str | None,
    ) -> CompactionResult:
        start_ms = time.time() * 1000

        # Step 1: Run pruner first to reduce input size
        await self._pruner.prune(session_id)

        if abort and abort.is_set():
            raise asyncio.CancelledError("Compaction aborted")

        # Fetch messages for compaction
        messages_with_parts = await self._store.get_messages_with_parts(session_id)
        non_summary = [m for m in messages_with_parts if not m.is_summary]

        if not non_summary:
            return CompactionResult(
                session_id=session_id,
                summary_message_id="",
                level_used=0,
                compacted_message_count=0,
                summary_token_count=0,
                tokens_before=0,
                tokens_after=0,
                elapsed_ms=time.time() * 1000 - start_ms,
            )

        tokens_before = sum(self._estimator.estimate_message(m) for m in non_summary)

        # Determine compaction model
        compaction_model = (
            model_override
            or self._config.compaction.compaction_model
            or "anthropic/claude-haiku-3-5"
        )

        # Compute a generous budget for the summary
        budget = ContextBudget(
            model_context_limit=200_000,
            reserved_output_tokens=8_192,
            compaction_buffer=self._config.compaction.buffer,
        )

        llm_call = _make_llm_call(compaction_model)
        candidate: SummaryCandidate | None = None

        # Step 2: Level 1
        if abort and abort.is_set():
            raise asyncio.CancelledError("Compaction aborted")

        candidate = await level1_summarise(
            non_summary, compaction_model, budget, self._estimator, llm_call
        )

        # Step 3: Level 2 (if Level 1 failed and level2 is enabled)
        if candidate is None and self._config.compaction.level2_enabled:
            if abort and abort.is_set():
                raise asyncio.CancelledError("Compaction aborted")
            candidate = await level2_summarise(
                non_summary, compaction_model, budget, self._estimator, llm_call
            )

        # Step 4: Level 3 (deterministic fallback — always succeeds)
        if candidate is None:
            candidate = level3_deterministic(non_summary, budget, self._estimator)

        # Step 5: Atomic commit
        msg_id = self._id_gen("msg")
        summary_node = SummaryNode(
            id=msg_id,
            session_id=session_id,
            level=0,
            span_start_message_id=candidate.span_start_message_id,
            span_end_message_id=candidate.span_end_message_id,
            content=candidate.text,
            token_count=candidate.token_count,
            model_id=compaction_model,
            compaction_level=candidate.compaction_level,
        )
        await self._dag_store.insert_node(summary_node, id_generator=lambda: self._id_gen("part"))

        span_end_msg = next(
            (m for m in non_summary if m.id == candidate.span_end_message_id),
            non_summary[-1],
        )
        tokens_after = (
            tokens_before
            - sum(
                self._estimator.estimate_message(m)
                for m in non_summary[: non_summary.index(span_end_msg) + 1]
                if m.id != candidate.span_end_message_id
            )
            + candidate.token_count
        )

        elapsed_ms = time.time() * 1000 - start_ms
        result = CompactionResult(
            session_id=session_id,
            summary_message_id=msg_id,
            level_used=candidate.compaction_level,
            compacted_message_count=candidate.messages_covered,
            summary_token_count=candidate.token_count,
            tokens_before=tokens_before,
            tokens_after=max(0, tokens_after),
            elapsed_ms=elapsed_ms,
        )

        self._logger.info(
            "compaction_completed",
            session_id=session_id,
            level=candidate.compaction_level,
            messages_compacted=candidate.messages_covered,
            tokens_before=tokens_before,
            tokens_after=result.tokens_after,
            elapsed_ms=elapsed_ms,
        )

        self._event_bus.publish(MnesisEvent.COMPACTION_COMPLETED, result.model_dump())
        return result


def _default_id_generator(prefix: str) -> str:
    from mnesis.session import make_id

    return make_id(prefix)
