"""Compaction orchestration engine with condensation and multi-round loop.

This engine implements the full Mnesis compaction flow:

1. **Tool output pruning** — backward-scan and tombstone oversized tool outputs.
2. **Summarisation** — level 1 → level 2 → level 3 escalation for raw messages.
3. **Condensation** — if the context (live summaries included) is still at or
   above half the soft threshold after summarisation, condense the summary nodes
   (level 1 → 2 → 3).
4. **Multi-round loop** — repeat steps 2-3 up to ``max_compaction_rounds``
   times until either the context fits or no progress is made.

Soft/hard threshold distinction:

- **Soft** (``soft_threshold_fraction`` * usable, default 60 %) — triggers
  early background compaction so the next turn is likely already compact.
- **Hard** (100 % of usable) — blocks the *next* send until compaction
  finishes, preventing an over-limit context from reaching the LLM.

Both thresholds compare against the size of the **current context window**
(system prompt + live summaries + raw messages that the next LLM call would
carry), not lifetime billed tokens. The size shrinks as soon as compaction
swaps messages for a summary. The engine's own "does it fit" check uses the
same measure (see ``context_measure``) and condenses summaries until the
context is below half the soft threshold, which makes compactions rarer than
stopping at the trigger (not guaranteed when the system prompt and recent turns
alone exceed that target).

File IDs are propagated through every compaction round; see
:mod:`mnesis.compaction.file_ids` and :mod:`mnesis.compaction.levels`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from mnesis.compaction.levels import (
    CondensationCandidate,
    SummaryCandidate,
    condense_level1,
    condense_level2,
    condense_level3_deterministic,
    level1_summarise,
    level2_summarise,
    level3_deterministic,
)
from mnesis.compaction.pruner import ToolOutputPruner
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
            elif "<summaries>" in content:
                conv_text = content.split("<summaries>")[1].split("</summaries>")[0].strip()
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


# Condensation runs until the context is below this fraction of the soft
# threshold (see ``_run_compaction_inner``).
_CONDENSE_TARGET_FRACTION = 0.5


class CompactionEngine:
    """
    Orchestrates the full compaction protocol (summarise → condense → loop).

    Guarantees:
    - ``run_compaction()`` never raises — errors are caught and Level 3 runs.
    - The resulting summary always fits within the token budget.
    - Level 3 (deterministic) is the final fallback and always succeeds.
    - Atomic SQLite commit per round: partial failures leave no inconsistent state.
    - The EventBus receives ``COMPACTION_COMPLETED`` (or ``COMPACTION_FAILED``).

    Threshold semantics:

    - **Soft threshold** (``soft_threshold_fraction``, default 60 %): triggers
      early *background* compaction via ``check_and_trigger()``.  This keeps
      the context lean well before the hard limit.
    - **Hard threshold** (100 % of usable): checked inside ``session.send()``
      before the LLM call; if exceeded, the caller should ``await
      wait_for_pending()`` to block until compaction completes.

    Both thresholds take the size of the *current context window* in tokens
    (see :attr:`mnesis.context.builder.BuiltContext.context_tokens`), never
    lifetime usage. At most one compaction is in flight per engine:
    :meth:`check_and_trigger` is a no-op while a previous task is running, and
    :meth:`compact_exclusive` (manual compaction) waits for it first.

    Example::

        engine = CompactionEngine(store, dag_store, estimator, event_bus, config)
        if engine.is_soft_overflow(tokens, model_info):
            engine.check_and_trigger(session_id, tokens, model_info)  # non-blocking
        if engine.is_hard_overflow(tokens, model_info):
            await engine.wait_for_pending()  # block if must compact before LLM call
    """

    def __init__(
        self,
        store: ImmutableStore,
        dag_store: SummaryDAGStore,
        token_estimator: TokenEstimator,
        event_bus: EventBus,
        config: MnesisConfig,
        id_generator: Any = None,
        session_model: str | None = None,
        model_info: ModelInfo | None = None,
        context_measure: Callable[[str], Awaitable[int]] | None = None,
    ) -> None:
        self._store = store
        self._dag_store = dag_store
        self._estimator = token_estimator
        self._event_bus = event_bus
        self._config = config
        self._session_model = session_model
        # Session model limits (including ``model_overrides``). When given, the
        # multi-round loop stops below half the soft threshold that triggers
        # compaction; otherwise it uses a fixed 200K-window hard budget.
        self._model_info = model_info
        # Async callable ``(session_id) -> int`` returning the session's current context size (the
        # ``BuiltContext.context_tokens`` the trigger uses). When set, the
        # engine's stop condition and ``tokens_before``/``tokens_after`` use it
        # so they cannot diverge from the trigger; otherwise they fall back to
        # an estimator-based count of raw messages plus live summaries.
        self._context_measure = context_measure
        self._pruner = ToolOutputPruner(store, token_estimator, config)
        self._id_gen = id_generator or _default_id_generator
        self._logger = structlog.get_logger("mnesis.compaction")
        self._pending_task: asyncio.Task[CompactionResult] | None = None

    def set_context_measure(self, measure: Callable[[str], Awaitable[int]] | None) -> None:
        """Set the callable that returns the session's current context size."""
        self._context_measure = measure

    # ── Threshold helpers ───────────────────────────────────────────────────────

    def _usable_tokens(self, model: ModelInfo) -> int:
        """Return the hard-limit usable token count for *model*."""
        budget = self._config.compaction.compaction_output_budget
        return model.context_limit - model.max_output_tokens - budget

    @staticmethod
    def _as_count(tokens: TokenUsage | int) -> int:
        """Normalise a token measure to an int (``TokenUsage`` is summed)."""
        return tokens if isinstance(tokens, int) else tokens.effective_total()

    def is_soft_overflow(self, tokens: TokenUsage | int, model: ModelInfo) -> bool:
        """
        Return True if the soft threshold has been crossed.

        The soft threshold triggers early background compaction (non-blocking).

        Args:
            tokens: Size of the current context window in tokens (an ``int``,
                e.g. ``BuiltContext.context_tokens``). A ``TokenUsage`` is also
                accepted and compared by its ``effective_total()``.
            model: Model metadata providing context limit.

        Returns:
            True if tokens reach ``soft_threshold_fraction * usable``.
        """
        if not self._config.compaction.auto:
            return False
        if model.context_limit == 0:
            return False

        usable = self._usable_tokens(model)
        soft_limit = int(usable * self._config.compaction.soft_threshold_fraction)
        return self._as_count(tokens) >= soft_limit

    def is_hard_overflow(self, tokens: TokenUsage | int, model: ModelInfo) -> bool:
        """
        Return True if the hard threshold has been crossed.

        The hard threshold means the current context *must* be compacted before
        the next LLM call to avoid an over-limit request.

        Args:
            tokens: Size of the current context window in tokens (see
                :meth:`is_soft_overflow`).
            model: Model metadata providing context limit.

        Returns:
            True if tokens reach the full usable budget.
        """
        if not self._config.compaction.auto:
            return False
        if model.context_limit == 0:
            return False

        usable = self._usable_tokens(model)
        return self._as_count(tokens) >= usable

    def is_overflow(self, tokens: TokenUsage | int, model: ModelInfo) -> bool:
        """
        Return True if compaction should be triggered (soft threshold check).

        Preserved for backwards compatibility; delegates to
        :meth:`is_soft_overflow`.

        Args:
            tokens: Size of the current context window in tokens.
            model: Model metadata providing context limit.

        Returns:
            True if tokens exceed the soft threshold.
        """
        return self.is_soft_overflow(tokens, model)

    # ── Trigger / scheduling ────────────────────────────────────────────────────

    def check_and_trigger(
        self,
        session_id: str,
        tokens: TokenUsage | int,
        model: ModelInfo,
        abort: asyncio.Event | None = None,
    ) -> bool:
        """
        Check for overflow and trigger async background compaction if needed.

        Non-blocking — schedules a background task and returns immediately.
        The task handle is stored in ``self._pending_task`` so callers can
        await or cancel it during shutdown. At most one compaction is in
        flight: while a previous task is still running this is a no-op, so
        rapid overflowing turns cannot run concurrent compactions (duplicate
        summaries / conflicting context-item swaps) or orphan the tracked task.

        Args:
            session_id: The session to compact.
            tokens: Size of the current context window in tokens.
            model: Model info for overflow detection.
            abort: Optional event to signal early termination.

        Returns:
            True if a new compaction task was scheduled, False if the context
            is under the soft threshold or a compaction is already in flight.
        """
        if not self.is_overflow(tokens, model):
            return False
        if self._pending_task is not None and not self._pending_task.done():
            return False

        self._logger.info(
            "compaction_triggered",
            session_id=session_id,
            tokens=self._as_count(tokens),
        )
        self._event_bus.publish(
            MnesisEvent.COMPACTION_TRIGGERED,
            {"session_id": session_id, "tokens": self._as_count(tokens)},
        )

        task = asyncio.create_task(self.run_compaction(session_id, abort=abort))
        self._pending_task = task
        return True

    async def wait_for_pending(self) -> CompactionResult | None:
        """
        Await any in-flight background compaction task to completion, then clear it.

        Returns:
            The :class:`CompactionResult` of the pending task (whether it was
            still running or had already finished), or ``None`` if no task was
            pending or it was cancelled. :meth:`run_compaction` never raises:
            a failed run is returned as a stub result with ``level_used == 0``
            and an empty ``summary_message_id``, so callers should check
            ``level_used`` before treating the result as a real summary.
        """
        task = self._pending_task
        result: CompactionResult | None = None
        if task is not None and not task.done():
            try:
                result = await task
            except Exception as exc:
                self._logger.exception("background_compaction_failed", error=str(exc))
        elif task is not None and not task.cancelled() and task.exception() is None:
            result = task.result()
        # A newer run may have replaced the one awaited here; never drop its handle.
        if self._pending_task is task:
            self._pending_task = None
        return result

    @property
    def in_flight(self) -> bool:
        """``True`` while a compaction task (background or manual) is running."""
        task = self._pending_task
        return task is not None and not task.done()

    @property
    def has_pending(self) -> bool:
        """``True`` if a task handle is held, running or finished but not yet reaped."""
        return self._pending_task is not None

    async def compact_exclusive(self, session_id: str) -> CompactionResult:
        """
        Run a compaction as the tracked in-flight task, after any pending one.

        Used for manual compaction so that it never overlaps a background run
        (duplicate summaries / conflicting context-item swaps).
        """
        while self._pending_task is not None:
            _ = await self.wait_for_pending()
        task = asyncio.create_task(self.run_compaction(session_id))
        self._pending_task = task
        try:
            return await task
        finally:
            if self._pending_task is task:
                self._pending_task = None

    # ── Public compaction entry point ───────────────────────────────────────────

    async def run_compaction(
        self,
        session_id: str,
        abort: asyncio.Event | None = None,
        model_override: str | None = None,
    ) -> CompactionResult:
        """
        Run the full compaction protocol. Never raises.

        Steps (per round, up to ``max_compaction_rounds``):
        1. Run tool output pruner (reduce input size first).
        2. Summarise raw messages: level 1 → level 2 → level 3.
        3. Condense accumulated summary nodes if still over budget: lvl 1→2→3.
        4. If no progress was made, break early to avoid spinning.

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

    # ── Internal implementation ─────────────────────────────────────────────────

    async def _run_compaction_inner(
        self,
        session_id: str,
        abort: asyncio.Event | None,
        model_override: str | None,
    ) -> CompactionResult:
        start_ms = time.time() * 1000

        # Context size before this run, on the same basis as ``tokens_after``.
        measured_before = await self._measure_session(session_id)

        # Step 1: Run pruner first to reduce input size
        prune_result = await self._pruner.prune(session_id)
        if prune_result.pruned_count > 0:
            self._event_bus.publish(
                MnesisEvent.PRUNE_COMPLETED,
                {
                    "session_id": session_id,
                    "pruned_count": prune_result.pruned_count,
                    "pruned_tokens": prune_result.pruned_tokens,
                },
            )

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
                pruned_tool_outputs=prune_result.pruned_count,
                pruned_tokens=prune_result.pruned_tokens,
            )

        # Raw messages still in the context (the store also holds messages that
        # earlier compactions already replaced). Legacy databases without
        # ``context_items`` rows count every message.
        in_context_ids = {
            item_id
            for item_type, item_id in await self._store.get_context_items(session_id)
            if item_type != "summary"
        }
        in_context = [m for m in non_summary if m.id in in_context_ids] or non_summary
        raw_tokens = sum(self._estimator.estimate_message(m) for m in in_context)
        if measured_before is None:
            prior_nodes = await self._dag_store.get_active_nodes(session_id)
            measured_before = raw_tokens + sum(n.token_count for n in prior_nodes)

        # Determine compaction model: explicit override → config → session model
        compaction_model = (
            model_override or self._config.compaction.compaction_model or self._session_model
        )
        if not compaction_model:
            raise ValueError(
                "No compaction model available. Set compaction.compaction_model in "
                "MnesisConfig or pass a model when creating the session."
            )

        # Derive model context limit for input cap calculation.
        compaction_model_info = ModelInfo.from_model_string(compaction_model)

        # Compute a generous budget for the summary
        budget = ContextBudget(
            model_context_limit=200_000,
            reserved_output_tokens=8_192,
            compaction_buffer=self._config.compaction.compaction_output_budget,
        )

        llm_call = _make_llm_call(compaction_model)

        # ── Summarisation ────────────────────────────────────────────────────────
        if abort and abort.is_set():
            raise asyncio.CancelledError("Compaction aborted")

        compaction_prompt = self._config.compaction.compaction_prompt
        candidate = await self._run_summarisation(
            non_summary,
            compaction_model,
            budget,
            llm_call,
            compaction_prompt,
            compaction_model_info.context_limit,
            abort,
        )

        # Commit the leaf summary node.
        leaf_msg_id = self._id_gen("msg")
        leaf_node = SummaryNode(
            id=leaf_msg_id,
            session_id=session_id,
            level=0,
            kind="leaf",
            span_start_message_id=candidate.span_start_message_id,
            span_end_message_id=candidate.span_end_message_id,
            content=candidate.text,
            token_count=candidate.token_count,
            model_id=compaction_model,
            compaction_level=candidate.compaction_level,
        )
        await self._dag_store.insert_node(leaf_node, id_generator=lambda: self._id_gen("part"))

        # Atomic context swap: remove compacted messages, insert summary item.
        span_end_msg = next(
            (m for m in non_summary if m.id == candidate.span_end_message_id),
            non_summary[-1],
        )
        span_end_idx = non_summary.index(span_end_msg)
        span_start_msg = next(
            (m for m in non_summary if m.id == candidate.span_start_message_id),
            non_summary[0],
        )
        span_start_idx = non_summary.index(span_start_msg)
        compacted_ids = [m.id for m in non_summary[span_start_idx : span_end_idx + 1]]
        await self._store.swap_context_items(session_id, compacted_ids, leaf_msg_id)

        last_summary_msg_id = leaf_msg_id
        last_summary_level = candidate.compaction_level
        last_messages_covered = candidate.messages_covered
        last_summary_tokens = candidate.token_count

        # Raw messages that stay in context after the leaf swap. Older leaf
        # summaries stay in context until condensed, so the context after
        # compaction is the tail plus every live summary node (plus the system
        # prompt, when ``context_measure`` is set). The loop below stops once
        # that is below ``fit_limit``: a fraction of the soft threshold that
        # triggers compaction. Stopping exactly at the soft threshold would leave
        # the context hovering at the trigger and re-compact on nearly every
        # turn once summaries accumulate.
        compacted_set = set(compacted_ids)
        tail_tokens = sum(
            self._estimator.estimate_message(m) for m in in_context if m.id not in compacted_set
        )
        fit_limit = (
            int(
                self._usable_tokens(self._model_info)
                * self._config.compaction.soft_threshold_fraction
                * _CONDENSE_TARGET_FRACTION
            )
            if self._model_info is not None and self._model_info.context_limit > 0
            else budget.usable
        )
        tokens_after = await self._measure(session_id, tail_tokens)

        # ── Condensation + multi-round loop ──────────────────────────────────────
        if self._config.compaction.condensation_enabled:
            max_rounds = self._config.compaction.max_compaction_rounds
            for round_num in range(max_rounds):
                if abort and abort.is_set():
                    raise asyncio.CancelledError("Compaction aborted during condensation")

                # Fetch all live summary nodes (older leaves stay in context
                # until condensed, so they count towards the context size).
                # ``tokens_after`` is fresh here: measured after the leaf swap
                # (round 0) or after the previous round's condensation.
                active_nodes = await self._dag_store.get_active_nodes(session_id)
                if tokens_after < fit_limit:
                    break  # Under budget — done.

                if len(active_nodes) < 2:
                    break  # Nothing to condense — can't make further progress.

                tokens_before_condense = sum(n.token_count for n in active_nodes)

                cond = await self._run_condensation(
                    active_nodes,
                    compaction_model,
                    budget,
                    llm_call,
                    abort,
                )

                if cond.token_count >= tokens_before_condense:
                    # No progress — LLM generated as much as it consumed.
                    self._logger.info(
                        "condensation_no_progress",
                        round=round_num,
                        tokens_before=tokens_before_condense,
                        tokens_after=cond.token_count,
                    )
                    break

                # Determine span from the consumed nodes.
                span_start = active_nodes[0].span_start_message_id
                span_end = active_nodes[-1].span_end_message_id

                condensed_msg_id = self._id_gen("msg")
                condensed_node = SummaryNode(
                    id=condensed_msg_id,
                    session_id=session_id,
                    level=1,
                    kind="condensed",
                    span_start_message_id=span_start,
                    span_end_message_id=span_end,
                    content=cond.text,
                    token_count=cond.token_count,
                    model_id=compaction_model,
                    compaction_level=cond.compaction_level,
                    parent_node_ids=cond.parent_node_ids,
                )
                await self._dag_store.insert_node(
                    condensed_node, id_generator=lambda: self._id_gen("part")
                )
                # Atomically swap the superseded summary context_items for the
                # new condensed node.  parent_node_ids are the summary message
                # IDs that are being replaced.
                await self._store.swap_context_items(
                    session_id, cond.parent_node_ids, condensed_msg_id
                )
                # Mark consumed nodes as superseded so get_active_nodes()
                # excludes them in subsequent rounds.
                await self._dag_store.mark_superseded(cond.parent_node_ids)

                last_summary_msg_id = condensed_msg_id
                last_summary_level = cond.compaction_level
                last_summary_tokens = cond.token_count
                tokens_after = await self._measure(session_id, tail_tokens)

                self._logger.info(
                    "condensation_round_completed",
                    round=round_num,
                    level=cond.compaction_level,
                    tokens_after=tokens_after,
                )

        tokens_before = measured_before
        elapsed_ms = time.time() * 1000 - start_ms
        result = CompactionResult(
            session_id=session_id,
            summary_message_id=last_summary_msg_id,
            level_used=last_summary_level,
            compacted_message_count=last_messages_covered,
            summary_token_count=last_summary_tokens,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            elapsed_ms=elapsed_ms,
            pruned_tool_outputs=prune_result.pruned_count,
            pruned_tokens=prune_result.pruned_tokens,
        )

        self._logger.info(
            "compaction_completed",
            session_id=session_id,
            level=last_summary_level,
            messages_compacted=last_messages_covered,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            elapsed_ms=elapsed_ms,
        )

        self._event_bus.publish(MnesisEvent.COMPACTION_COMPLETED, result.model_dump())
        return result

    async def _measure_session(self, session_id: str) -> int | None:
        """Return the session's current context size, or ``None`` if unavailable.

        This is the builder's un-truncated context size (system prompt, live
        summaries and raw messages, session-model estimator): the number the
        soft/hard triggers compare.
        """
        if self._context_measure is None:
            return None
        try:
            return await self._context_measure(session_id)
        except Exception:
            self._logger.warning("context_measure_failed", session_id=session_id, exc_info=True)
            return None

    async def _measure(self, session_id: str, tail_tokens: int) -> int:
        """Return the context size after the leaf swap.

        Uses :meth:`_measure_session` when available; otherwise ``tail_tokens``
        plus the live summary nodes (estimator-based, no system prompt).
        """
        measured = await self._measure_session(session_id)
        if measured is not None:
            return measured
        nodes = await self._dag_store.get_active_nodes(session_id)
        return tail_tokens + sum(n.token_count for n in nodes)

    async def _run_summarisation(
        self,
        non_summary: list[Any],
        compaction_model: str,
        budget: ContextBudget,
        llm_call: Any,
        compaction_prompt: str | None,
        model_context_limit: int,
        abort: asyncio.Event | None,
    ) -> SummaryCandidate:
        """Run level 1 → 2 → 3 summarisation escalation and return a candidate."""
        candidate: SummaryCandidate | None = None

        # Level 1
        if abort and abort.is_set():
            raise asyncio.CancelledError("Compaction aborted")
        candidate = await level1_summarise(
            non_summary,
            compaction_model,
            budget,
            self._estimator,
            llm_call,
            compaction_prompt=compaction_prompt,
            model_context_limit=model_context_limit,
        )

        # Level 2 (if Level 1 failed and level2 is enabled)
        if candidate is None and self._config.compaction.level2_enabled:
            if abort and abort.is_set():
                raise asyncio.CancelledError("Compaction aborted")
            candidate = await level2_summarise(
                non_summary,
                compaction_model,
                budget,
                self._estimator,
                llm_call,
                compaction_prompt=compaction_prompt,
                model_context_limit=model_context_limit,
            )

        # Level 3 (deterministic fallback — always succeeds)
        if candidate is None:
            candidate = level3_deterministic(non_summary, budget, self._estimator)

        return candidate

    async def _run_condensation(
        self,
        nodes: list[SummaryNode],
        compaction_model: str,
        budget: ContextBudget,
        llm_call: Any,
        abort: asyncio.Event | None,
    ) -> CondensationCandidate:
        """Run level 1 → 2 → 3 condensation escalation and return a candidate."""
        cond: CondensationCandidate | None = None

        # Level 1
        if abort and abort.is_set():
            raise asyncio.CancelledError("Compaction aborted")
        cond = await condense_level1(nodes, compaction_model, budget, self._estimator, llm_call)

        # Level 2
        if cond is None and self._config.compaction.level2_enabled:
            if abort and abort.is_set():
                raise asyncio.CancelledError("Compaction aborted")
            cond = await condense_level2(nodes, compaction_model, budget, self._estimator, llm_call)

        # Level 3 (always succeeds)
        if cond is None:
            cond = condense_level3_deterministic(nodes, self._estimator)

        return cond


def _default_id_generator(prefix: str) -> str:
    from mnesis.session import make_id

    return make_id(prefix)
