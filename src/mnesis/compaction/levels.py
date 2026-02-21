"""Three-level compaction escalation functions.

This module provides:

* **Summarisation** — converts raw messages into a leaf summary node.
  Three escalation levels: level 1 (structured LLM), level 2 (aggressive
  LLM), level 3 (deterministic truncation — always succeeds).

* **Condensation** — merges one or more existing summary nodes into a single
  condensed node.  Three escalation levels mirror summarisation.

Both operations extract ``file_xxx`` identifiers from their inputs and append a
``[LCM File IDs: ...]`` footer to the output, preserving the "lossless"
guarantee across compaction rounds.

Summarisation input cap: messages passed to the LLM summarizer are capped at
``MAX_SUMMARISATION_INPUT_FRACTION`` (75 %) of the compaction model's context
window to prevent the compaction call itself from overflowing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from mnesis.compaction.file_ids import (
    append_file_ids_footer,
    collect_file_ids_from_nodes,
    extract_file_ids_from_messages,
)
from mnesis.models.message import ContextBudget, MessageWithParts, TextPart, ToolPart
from mnesis.models.summary import SummaryNode
from mnesis.tokens.estimator import TokenEstimator

logger = structlog.get_logger("mnesis.compaction.levels")

# 75 % of the compaction model's context window may be used for summarisation
# input — mirrors Volt's ``maxSummarizationInputTokens``.
MAX_SUMMARISATION_INPUT_FRACTION: float = 0.75

# Minimum number of messages that must be passed to the summariser even when
# the input cap would exclude them — mirrors Volt's ``MIN_MESSAGES_TO_SUMMARIZE``.
MIN_MESSAGES_TO_SUMMARISE: int = 3

# Maximum tokens for a level 3 condensation fallback (mirrors Volt's 512-token cap).
_CONDENSE_LEVEL3_MAX_TOKENS: int = 512

LEVEL1_PROMPT = """\
You are creating a detailed context summary to allow continuing this conversation.
Preserve all goals, instructions, constraints, file context, and tool results.
Be thorough — this summary will replace the original messages.

Format your response exactly as:

## Goal
(Describe the overall objective of this conversation)

## Key Instructions & Constraints
(List any rules, constraints, or guidelines that must be followed)

## Discoveries & Findings
(Notable things learned during the conversation)

## Completed Work
(What has been accomplished so far)

## In Progress
(What is currently being worked on)

## Remaining Work
(What still needs to be done)

## Relevant Files & Directories
(File paths, directory structures, important code locations)

## Other Important Context
(Anything else needed to continue effectively)
"""

LEVEL2_PROMPT = """\
Create a COMPRESSED continuation summary. Be extremely concise.
Drop intermediate reasoning, redundant details, and verbose explanations.
Preserve only: current goal, active constraints, key file locations, next step.

Format:
GOAL: <one sentence>
CONSTRAINTS: <comma-separated list>
FILES: <key paths>
NEXT: <immediate next action>
CONTEXT: <any other critical facts, max 3 sentences>
"""

CONDENSE_LEVEL1_PROMPT = """\
You are condensing multiple context summaries into one unified summary.
Each summary below represents a portion of the conversation history.
Merge them into a single coherent summary that preserves all critical information.

Format your response exactly as:

## Goal
(The overall objective carried across all summaries)

## Key Instructions & Constraints
(All rules, constraints, or guidelines from any summary)

## Discoveries & Findings
(All notable findings across all summaries)

## Completed Work
(Everything accomplished across all summaries)

## In Progress
(What is currently being worked on)

## Remaining Work
(What still needs to be done based on all summaries)

## Relevant Files & Directories
(All file paths and directories mentioned across summaries)

## Other Important Context
(Any critical context from any summary not captured above)
"""

CONDENSE_LEVEL2_PROMPT = """\
Compress these summaries into one very short summary.
Keep only: current goal, key constraints, critical file paths, immediate next step.

Format:
GOAL: <one sentence>
CONSTRAINTS: <comma-separated list>
FILES: <key paths>
NEXT: <immediate next action>
CONTEXT: <any other critical facts, max 2 sentences>
"""


@dataclass
class SummaryCandidate:
    """A candidate compaction summary before it is committed to the store."""

    text: str
    token_count: int
    span_start_message_id: str
    span_end_message_id: str
    compaction_level: int
    messages_covered: int


@dataclass
class CondensationCandidate:
    """A candidate condensation of one or more existing summary nodes."""

    text: str
    token_count: int
    parent_node_ids: list[str] = field(default_factory=list)
    compaction_level: int = 1
    """Condensation escalation level: 1 = normal, 2 = aggressive, 3 = deterministic."""


def _extract_text(msg: MessageWithParts, max_chars: int = 2000) -> str:
    """Extract readable text from a message, capped at max_chars per message."""
    parts: list[str] = []
    for part in msg.parts:
        if isinstance(part, TextPart):
            parts.append(part.text[:max_chars])
        elif isinstance(part, ToolPart):
            if part.compacted_at is None and part.output:
                parts.append(f"[Tool {part.tool_name}]: {part.output[:500]}")
    return "\n".join(parts)[:max_chars]


def _build_messages_text(messages: list[MessageWithParts]) -> str:
    """Format a list of messages as a readable transcript."""
    lines: list[str] = []
    for msg in messages:
        role_label = "USER" if msg.role == "user" else "ASSISTANT"
        text = _extract_text(msg)
        if text:
            lines.append(f"[{role_label}]:\n{text}")
    return "\n\n".join(lines)


def _messages_to_summarise(messages: list[MessageWithParts]) -> list[MessageWithParts]:
    """Return all messages except the most recent 2 user turns (protect recent work)."""
    # Find the index of the second-to-last user turn
    user_indices = [i for i, m in enumerate(messages) if m.role == "user"]
    if len(user_indices) <= 2:
        return []
    cutoff = user_indices[-2]
    return messages[:cutoff]


def _apply_input_cap(
    messages: list[MessageWithParts],
    estimator: TokenEstimator,
    model_context_limit: int,
) -> list[MessageWithParts]:
    """
    Trim *messages* so their total token count stays within the summarisation
    input cap (``MAX_SUMMARISATION_INPUT_FRACTION`` of *model_context_limit*).

    At least :data:`MIN_MESSAGES_TO_SUMMARISE` messages are always included
    even if they exceed the cap, mirroring Volt's guard.

    Args:
        messages: Messages to cap (already filtered by ``_messages_to_summarise``).
        estimator: Token estimator.
        model_context_limit: Full context limit of the compaction model.

    Returns:
        A (possibly shorter) list of messages to pass to the LLM.
    """
    if not messages:
        return messages

    max_input_tokens = int(model_context_limit * MAX_SUMMARISATION_INPUT_FRACTION)
    tokens_so_far = 0
    result: list[MessageWithParts] = []

    for msg in messages:
        msg_tokens = estimator.estimate_message(msg)
        if (
            tokens_so_far + msg_tokens > max_input_tokens
            and len(result) >= MIN_MESSAGES_TO_SUMMARISE
        ):
            logger.info(
                "summarisation_input_cap_applied",
                included=len(result),
                total=len(messages),
                cap_tokens=max_input_tokens,
            )
            break
        result.append(msg)
        tokens_so_far += msg_tokens

    return result


async def level1_summarise(
    messages: list[MessageWithParts],
    model: str,
    budget: ContextBudget,
    estimator: TokenEstimator,
    llm_call: Any,
    compaction_prompt: str | None = None,
    model_context_limit: int = 200_000,
) -> SummaryCandidate | None:
    """
    Attempt Level 1 (selective) summarisation via LLM.

    File IDs found in the input messages are automatically appended to the
    summary via a ``[LCM File IDs: ...]`` footer.

    Input messages are capped at 75 % of the compaction model's context window
    to prevent the summarisation call itself from overflowing.

    Args:
        messages: All non-summary messages in the session.
        model: Model string to use for compaction.
        budget: Token budget for validation.
        estimator: Token estimator for result validation.
        llm_call: Async callable ``(model, messages, max_tokens) -> str``.
        compaction_prompt: Custom system prompt override.
        model_context_limit: Context window of the compaction model.

    Returns:
        SummaryCandidate if successful and fits budget, or None to escalate.
    """
    to_summarise = _messages_to_summarise(messages)
    if not to_summarise:
        logger.debug("level1_skip_nothing_to_summarise")
        return None

    # Remember original span before the input cap narrows the list.
    original_to_summarise = to_summarise

    # Apply input token cap before passing to LLM.
    to_summarise = _apply_input_cap(to_summarise, estimator, model_context_limit)

    # Collect file IDs from the capped input.
    file_ids = extract_file_ids_from_messages(to_summarise)

    transcript = _build_messages_text(to_summarise)
    prompt = compaction_prompt if compaction_prompt is not None else LEVEL1_PROMPT
    prompt_messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\n<conversation>\n{transcript}\n</conversation>",
        }
    ]

    try:
        summary_text = await llm_call(
            model=model,
            messages=prompt_messages,
            max_tokens=budget.reserved_output_tokens,
        )
    except Exception as exc:
        logger.warning("level1_llm_failed", error=str(exc))
        return None

    # Propagate file IDs into the summary.
    summary_text = append_file_ids_footer(summary_text, file_ids)

    token_count = estimator.estimate(summary_text)
    if token_count > budget.usable:
        logger.info(
            "level1_summary_too_large",
            token_count=token_count,
            usable=budget.usable,
        )
        return None

    return SummaryCandidate(
        text=summary_text,
        token_count=token_count,
        span_start_message_id=original_to_summarise[0].id,
        span_end_message_id=original_to_summarise[-1].id,
        compaction_level=1,
        messages_covered=len(original_to_summarise),
    )


async def level2_summarise(
    messages: list[MessageWithParts],
    model: str,
    budget: ContextBudget,
    estimator: TokenEstimator,
    llm_call: Any,
    compaction_prompt: str | None = None,
    model_context_limit: int = 200_000,
) -> SummaryCandidate | None:
    """
    Attempt Level 2 (aggressive) summarisation via LLM.

    Uses a more compressed prompt format and drops reasoning details.
    File IDs found in the input messages are propagated to the summary.
    Input is capped at 75 % of the compaction model's context window.

    Args:
        messages: All non-summary messages in the session.
        model: Model string to use for compaction.
        budget: Token budget for validation.
        estimator: Token estimator for result validation.
        llm_call: Async callable.
        compaction_prompt: Custom system prompt override.
        model_context_limit: Context window of the compaction model.

    Returns:
        SummaryCandidate if successful and fits budget, or None to escalate.
    """
    to_summarise = _messages_to_summarise(messages)
    if not to_summarise:
        return None

    original_to_summarise = to_summarise

    # Apply input token cap.
    to_summarise = _apply_input_cap(to_summarise, estimator, model_context_limit)

    # Collect file IDs from input messages.
    file_ids = extract_file_ids_from_messages(to_summarise)

    # For level 2, cap transcript length more aggressively
    transcript_parts: list[str] = []
    for msg in to_summarise:
        text = _extract_text(msg, max_chars=500)
        if text:
            role = "U" if msg.role == "user" else "A"
            transcript_parts.append(f"[{role}]: {text}")
    transcript = "\n".join(transcript_parts)

    prompt = compaction_prompt if compaction_prompt is not None else LEVEL2_PROMPT
    prompt_messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\n<conversation>\n{transcript}\n</conversation>",
        }
    ]

    try:
        summary_text = await llm_call(
            model=model,
            messages=prompt_messages,
            max_tokens=min(budget.compaction_buffer, 4000),
        )
    except Exception as exc:
        logger.warning("level2_llm_failed", error=str(exc))
        return None

    # Propagate file IDs.
    summary_text = append_file_ids_footer(summary_text, file_ids)

    token_count = estimator.estimate(summary_text)
    if token_count > budget.usable:
        logger.info(
            "level2_summary_too_large",
            token_count=token_count,
            usable=budget.usable,
        )
        return None

    return SummaryCandidate(
        text=summary_text,
        token_count=token_count,
        span_start_message_id=original_to_summarise[0].id,
        span_end_message_id=original_to_summarise[-1].id,
        compaction_level=2,
        messages_covered=len(original_to_summarise),
    )


def level3_deterministic(
    messages: list[MessageWithParts],
    budget: ContextBudget,
    estimator: TokenEstimator,
) -> SummaryCandidate:
    """
    Level 3 deterministic fallback (no LLM required).

    Keeps the most recent messages that fit within 85% of the usable budget,
    prefixed with a truncation notice. This always produces a valid result.

    File IDs found in *all* input messages (not just kept ones) are preserved
    in a ``[LCM File IDs: ...]`` footer even when their surrounding context is
    truncated — this is the lossless guarantee.

    Args:
        messages: All non-summary messages in the session.
        budget: Token budget — result is guaranteed to fit within usable.
        estimator: Token estimator.

    Returns:
        SummaryCandidate that always fits within budget.usable.
    """
    # Collect file IDs from ALL messages before truncation — the whole point of
    # level 3 is that we never lose file pointers even when prose is discarded.
    all_file_ids = extract_file_ids_from_messages(messages)

    target = int(budget.usable * 0.85)
    header = "[CONTEXT TRUNCATED — DETERMINISTIC FALLBACK]\n\n## Kept Messages\n\n"
    header_tokens = estimator.estimate(header)

    kept: list[MessageWithParts] = []
    tokens_used = header_tokens
    for msg in reversed(messages):
        text = _extract_text(msg, max_chars=800)
        role = "USER" if msg.role == "user" else "ASSISTANT"
        line = f"[{role}]: {text[:500]}\n"
        line_tokens = estimator.estimate(line)
        if tokens_used + line_tokens > target:
            break
        kept.append(msg)
        tokens_used += line_tokens

    kept.reverse()

    lines = [header]
    for msg in kept:
        text = _extract_text(msg, max_chars=500)
        role = "USER" if msg.role == "user" else "ASSISTANT"
        lines.append(f"[{role}]: {text}\n")

    summary_text = "\n".join(lines)

    # Append file IDs footer — preserves all file references even for truncated content.
    summary_text = append_file_ids_footer(summary_text, all_file_ids)

    token_count = estimator.estimate(summary_text)

    # All messages if nothing was kept
    if not messages:
        span_start = ""
        span_end = ""
    else:
        span_start = messages[0].id
        span_end = messages[-1].id

    logger.info(
        "level3_deterministic_produced",
        kept_messages=len(kept),
        total_messages=len(messages),
        token_count=token_count,
    )

    return SummaryCandidate(
        text=summary_text,
        token_count=token_count,
        span_start_message_id=span_start,
        span_end_message_id=span_end,
        compaction_level=3,
        messages_covered=len(messages),
    )


# ── Condensation ───────────────────────────────────────────────────────────────


async def condense_level1(
    nodes: list[SummaryNode],
    model: str,
    budget: ContextBudget,
    estimator: TokenEstimator,
    llm_call: Any,
) -> CondensationCandidate | None:
    """
    Attempt Level 1 condensation: merge summary nodes via structured LLM prompt.

    File IDs from all parent nodes are collected and appended to the condensed
    output via a ``[LCM File IDs: ...]`` footer.

    Args:
        nodes: Summary nodes to condense (must be non-empty).
        model: LLM model string.
        budget: Token budget for the condensed result.
        estimator: Token estimator.
        llm_call: Async callable ``(model, messages, max_tokens) -> str``.

    Returns:
        CondensationCandidate if successful and fits budget, or None to escalate.
    """
    if not nodes:
        return None

    # Gather all file IDs from parent nodes (already embedded in their content).
    file_ids = collect_file_ids_from_nodes(nodes)

    summaries_text = "\n\n---\n\n".join(
        f"[Summary {i + 1}]:\n{node.content}" for i, node in enumerate(nodes)
    )
    prompt_messages = [
        {
            "role": "user",
            "content": (f"{CONDENSE_LEVEL1_PROMPT}\n\n<summaries>\n{summaries_text}\n</summaries>"),
        }
    ]

    try:
        condensed_text = await llm_call(
            model=model,
            messages=prompt_messages,
            max_tokens=budget.reserved_output_tokens,
        )
    except Exception as exc:
        logger.warning("condense_level1_llm_failed", error=str(exc))
        return None

    condensed_text = append_file_ids_footer(condensed_text, file_ids)

    token_count = estimator.estimate(condensed_text)
    if token_count > budget.usable:
        logger.info(
            "condense_level1_too_large",
            token_count=token_count,
            usable=budget.usable,
        )
        return None

    return CondensationCandidate(
        text=condensed_text,
        token_count=token_count,
        parent_node_ids=[n.id for n in nodes],
        compaction_level=1,
    )


async def condense_level2(
    nodes: list[SummaryNode],
    model: str,
    budget: ContextBudget,
    estimator: TokenEstimator,
    llm_call: Any,
) -> CondensationCandidate | None:
    """
    Attempt Level 2 condensation: aggressive merge via compressed prompt.

    Args:
        nodes: Summary nodes to condense.
        model: LLM model string.
        budget: Token budget.
        estimator: Token estimator.
        llm_call: Async callable.

    Returns:
        CondensationCandidate if successful and fits budget, or None to escalate.
    """
    if not nodes:
        return None

    file_ids = collect_file_ids_from_nodes(nodes)

    # Use a truncated excerpt from each summary for the aggressive prompt.
    summaries_text = "\n\n".join(
        f"[S{i + 1}]: {node.content[:800]}" for i, node in enumerate(nodes)
    )
    prompt_messages = [
        {
            "role": "user",
            "content": (f"{CONDENSE_LEVEL2_PROMPT}\n\n<summaries>\n{summaries_text}\n</summaries>"),
        }
    ]

    try:
        condensed_text = await llm_call(
            model=model,
            messages=prompt_messages,
            max_tokens=min(budget.compaction_buffer, 4000),
        )
    except Exception as exc:
        logger.warning("condense_level2_llm_failed", error=str(exc))
        return None

    condensed_text = append_file_ids_footer(condensed_text, file_ids)

    token_count = estimator.estimate(condensed_text)
    if token_count > budget.usable:
        logger.info(
            "condense_level2_too_large",
            token_count=token_count,
            usable=budget.usable,
        )
        return None

    return CondensationCandidate(
        text=condensed_text,
        token_count=token_count,
        parent_node_ids=[n.id for n in nodes],
        compaction_level=2,
    )


def condense_level3_deterministic(
    nodes: list[SummaryNode],
    estimator: TokenEstimator,
) -> CondensationCandidate:
    """
    Level 3 deterministic condensation fallback (no LLM required).

    Truncates the concatenated summary content to
    :data:`_CONDENSE_LEVEL3_MAX_TOKENS` tokens while always preserving all
    file IDs from every parent node.

    Args:
        nodes: Summary nodes to condense.
        estimator: Token estimator.

    Returns:
        CondensationCandidate that always succeeds.
    """
    # Always collect all file IDs — these must survive regardless of truncation.
    file_ids = collect_file_ids_from_nodes(nodes)

    parent_ids_str = ", ".join(n.id for n in nodes)
    ids_header = f"[Condensed from: {parent_ids_str}]\n\n"
    fallback_header = "[CONDENSED — DETERMINISTIC FALLBACK]\n"

    header = fallback_header + ids_header
    header_tokens = estimator.estimate(header)

    available = _CONDENSE_LEVEL3_MAX_TOKENS - header_tokens
    if available < 0:
        available = 0

    # Take as much content as fits from each node in order.
    content_parts: list[str] = []
    used = 0
    for node in nodes:
        chunk = node.content[:2000]  # per-node safety cap
        chunk_tokens = estimator.estimate(chunk)
        if used + chunk_tokens > available and content_parts:
            break
        content_parts.append(chunk)
        used += chunk_tokens

    combined = header + "\n\n---\n\n".join(content_parts)
    combined = append_file_ids_footer(combined, file_ids)

    token_count = estimator.estimate(combined)

    logger.info(
        "condense_level3_deterministic_produced",
        nodes_condensed=len(nodes),
        token_count=token_count,
    )

    return CondensationCandidate(
        text=combined,
        token_count=token_count,
        parent_node_ids=[n.id for n in nodes],
        compaction_level=3,
    )
