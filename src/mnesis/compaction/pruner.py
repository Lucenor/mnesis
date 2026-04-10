"""Tool output pruner — reclaims context space by tombstoning stale tool outputs."""

from __future__ import annotations

import time

import structlog

from mnesis.models.config import MnesisConfig
from mnesis.models.message import PruneResult, ToolPart
from mnesis.store.immutable import ImmutableStore
from mnesis.tokens.estimator import TokenEstimator


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
        2. Pre-fetch all raw parts to build a (message_id, tool_call_id) → part_id map.
        3. Walk backward, tracking user turn count.
        4. Skip the most recent 2 user turns (protect recent content).
        5. Stop at any is_summary message (compaction boundary).
        6. For each completed, non-protected, non-already-pruned tool part:
           - Accumulate output tokens.
           - Once outside the protect window (>40K tokens), add to candidates.
        7. If total prunable volume < minimum threshold (20K): no-op.
        8. Apply tombstones in a single batch UPDATE.

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

        # Build a part_id lookup: (message_id, tool_call_id) -> part_id
        # Single IN query fetches all raw parts for all messages at once (no N+1).
        message_ids = [m.id for m in messages]
        part_id_map: dict[tuple[str, str], str] = {}
        for raw in await self._store.get_raw_parts_for_messages(message_ids):
            if raw.part_type == "tool" and raw.tool_call_id:
                part_id_map[(raw.message_id, raw.tool_call_id)] = raw.id

        candidates: list[str] = []  # part IDs to tombstone
        total_tool_tokens = 0  # cumulative tokens seen scanning backward
        pruned_volume = 0  # tokens in candidate parts
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
                if part.is_protected:
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
                    part_id = part_id_map.get((msg.id, part.tool_call_id), "")
                    if part_id:
                        candidates.append(part_id)
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

        # Apply tombstones in a single batch UPDATE
        now_ms = int(time.time() * 1000)
        await self._store.batch_set_compacted_at(candidates, now_ms)

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


# Backward-compatible alias
ToolOutputPrunerAsync = ToolOutputPruner
