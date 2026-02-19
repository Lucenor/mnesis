"""Tool output pruner — reclaims context space by tombstoning stale tool outputs."""

from __future__ import annotations

import time

import structlog

from mnesis.models.config import MnesisConfig
from mnesis.models.message import MessageWithParts, PruneResult, ToolPart
from mnesis.store.immutable import ImmutableStore
from mnesis.tokens.estimator import TokenEstimator

# Tools that should never be pruned
_PROTECTED_TOOLS: frozenset[str] = frozenset({"skill"})


class ToolOutputPruner:
    """
    Reclaims context space by tombstoning stale tool outputs.

    The protect window ensures the most recent tool outputs (within the last
    ``prune_protect_tokens`` worth of content) are never pruned. Only completed
    tool calls outside this window are candidates.

    No LLM call is required — this is entirely deterministic.

    Example::

        pruner = ToolOutputPruner(store, estimator, config)
        result = await pruner.prune("sess_01JXYZ...")
        print(f"Pruned {result.pruned_count} tool outputs ({result.pruned_tokens:,} tokens)")
    """

    def __init__(
        self,
        store: ImmutableStore,
        estimator: TokenEstimator,
        config: MnesisConfig,
    ) -> None:
        self._store = store
        self._estimator = estimator
        self._config = config
        self._logger = structlog.get_logger("mnesis.pruner")

    async def prune(self, session_id: str) -> PruneResult:
        """
        Run a prune pass for the given session.

        Algorithm:
        1. Fetch all messages (including summaries) in chronological order.
        2. Walk backward, tracking user turn count.
        3. Skip the most recent 2 user turns (protect recent content).
        4. Stop at any is_summary message (compaction boundary).
        5. For each completed, non-protected, non-already-pruned tool part:
           - Accumulate output tokens.
           - Once outside the protect window (>40K tokens), add to candidates.
        6. If total prunable volume < minimum threshold (20K): no-op.
        7. Apply tombstones in a batch.

        Args:
            session_id: The session to prune.

        Returns:
            PruneResult with counts of pruned parts and tokens.
        """
        if not self._config.compaction.prune:
            return PruneResult(pruned_count=0, pruned_tokens=0, candidates_scanned=0)

        protect_tokens = self._config.compaction.prune_protect_tokens
        minimum_tokens = self._config.compaction.prune_minimum_tokens

        messages = await self._store.get_messages_with_parts(session_id)
        if not messages:
            return PruneResult(pruned_count=0, pruned_tokens=0, candidates_scanned=0)

        candidates: list[str] = []  # part IDs to tombstone
        total_tool_tokens = 0       # cumulative tokens seen scanning backward
        pruned_volume = 0           # tokens in candidate parts
        candidates_scanned = 0
        user_turn_count = 0

        for msg in reversed(messages):
            # Count user turns as we scan backward
            if msg.role == "user":
                user_turn_count += 1

            # Protect the most recent 2 user turns
            if user_turn_count < 2:
                continue

            # Stop at a compaction boundary (summary message)
            if msg.is_summary:
                break

            # Process tool parts backward within the message
            for part in reversed(msg.parts):
                if not isinstance(part, ToolPart):
                    continue

                candidates_scanned += 1

                # Skip non-completed, protected, or already-pruned parts
                if part.status.state != "completed":
                    continue
                if part.tool_name in _PROTECTED_TOOLS:
                    continue
                if part.compacted_at is not None:
                    # Hit a prior compaction boundary within this message — stop
                    # scanning this message's tool parts but continue outer loop
                    break

                # Estimate output tokens
                output_tokens = self._estimator.estimate(part.output or "")
                total_tool_tokens += output_tokens

                # Only add to candidates if outside the protect window
                if total_tool_tokens > protect_tokens:
                    candidates.append(await self._get_part_id(msg, part))
                    pruned_volume += output_tokens

        if pruned_volume <= minimum_tokens:
            self._logger.debug(
                "prune_below_minimum",
                session_id=session_id,
                pruned_volume=pruned_volume,
                minimum=minimum_tokens,
            )
            return PruneResult(
                pruned_count=0,
                pruned_tokens=0,
                candidates_scanned=candidates_scanned,
            )

        # Apply tombstones
        now_ms = int(time.time() * 1000)
        for part_id in candidates:
            await self._store.update_part_status(part_id, compacted_at=now_ms)

        self._logger.info(
            "prune_completed",
            session_id=session_id,
            pruned_count=len(candidates),
            pruned_tokens=pruned_volume,
        )
        return PruneResult(
            pruned_count=len(candidates),
            pruned_tokens=pruned_volume,
            candidates_scanned=candidates_scanned,
        )

    async def _get_part_id(self, msg: MessageWithParts, part: ToolPart) -> str:
        """Look up the database ID for a specific tool part."""
        # The part ID is stored on the RawMessagePart but not on the typed ToolPart.
        # We need to find it by matching tool_call_id in the raw parts.
        raw_parts = await self._store.get_parts(msg.id)
        for raw in raw_parts:
            if raw.part_type == "tool" and raw.tool_call_id == part.tool_call_id:
                return raw.id
        # Fallback: shouldn't happen, but return a sentinel
        return ""

    def _get_part_id_sync(
        self, msg: MessageWithParts, part: ToolPart
    ) -> str:
        """Synchronous version — returns empty string, used as sentinel check."""
        return ""


class ToolOutputPrunerAsync(ToolOutputPruner):
    """Async version that properly resolves part IDs."""

    async def prune(self, session_id: str) -> PruneResult:
        """Run prune pass with async part ID resolution."""
        if not self._config.compaction.prune:
            return PruneResult(pruned_count=0, pruned_tokens=0, candidates_scanned=0)

        protect_tokens = self._config.compaction.prune_protect_tokens
        minimum_tokens = self._config.compaction.prune_minimum_tokens

        messages = await self._store.get_messages_with_parts(session_id)
        if not messages:
            return PruneResult(pruned_count=0, pruned_tokens=0, candidates_scanned=0)

        # Build a part_id lookup: (message_id, tool_call_id) -> part_id
        # Fetch all raw parts for all messages in one pass
        all_message_ids = [m.id for m in messages]
        part_id_map: dict[tuple[str, str], str] = {}
        for msg_id in all_message_ids:
            raw_parts = await self._store.get_parts(msg_id)
            for raw in raw_parts:
                if raw.tool_call_id:
                    part_id_map[(msg_id, raw.tool_call_id)] = raw.id

        candidates: list[str] = []
        total_tool_tokens = 0
        pruned_volume = 0
        candidates_scanned = 0
        user_turn_count = 0

        for msg in reversed(messages):
            if msg.role == "user":
                user_turn_count += 1

            if user_turn_count < 2:
                continue

            if msg.is_summary:
                break

            for part in reversed(msg.parts):
                if not isinstance(part, ToolPart):
                    continue

                candidates_scanned += 1

                if part.status.state != "completed":
                    continue
                if part.tool_name in _PROTECTED_TOOLS:
                    continue
                if part.compacted_at is not None:
                    break

                output_tokens = self._estimator.estimate(part.output or "")
                total_tool_tokens += output_tokens

                if total_tool_tokens > protect_tokens:
                    part_id = part_id_map.get((msg.id, part.tool_call_id), "")
                    if part_id:
                        candidates.append(part_id)
                        pruned_volume += output_tokens

        if pruned_volume <= minimum_tokens:
            return PruneResult(
                pruned_count=0,
                pruned_tokens=0,
                candidates_scanned=candidates_scanned,
            )

        now_ms = int(time.time() * 1000)
        for part_id in candidates:
            await self._store.update_part_status(part_id, compacted_at=now_ms)

        self._logger.info(
            "prune_completed",
            session_id=session_id,
            pruned_count=len(candidates),
            pruned_tokens=pruned_volume,
        )
        return PruneResult(
            pruned_count=len(candidates),
            pruned_tokens=pruned_volume,
            candidates_scanned=candidates_scanned,
        )
