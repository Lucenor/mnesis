"""Three-level compaction escalation functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from mnesis.models.message import ContextBudget, MessageWithParts, TextPart, ToolPart
from mnesis.tokens.estimator import TokenEstimator

logger = structlog.get_logger("mnesis.compaction.levels")

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


@dataclass
class SummaryCandidate:
    """A candidate compaction summary before it is committed to the store."""

    text: str
    token_count: int
    span_start_message_id: str
    span_end_message_id: str
    compaction_level: int
    messages_covered: int


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


async def level1_summarise(
    messages: list[MessageWithParts],
    model: str,
    budget: ContextBudget,
    estimator: TokenEstimator,
    llm_call: Any,
) -> SummaryCandidate | None:
    """
    Attempt Level 1 (selective) summarisation via LLM.

    Args:
        messages: All non-summary messages in the session.
        model: Model string to use for compaction.
        budget: Token budget for validation.
        estimator: Token estimator for result validation.
        llm_call: Async callable ``(model, messages, max_tokens) -> str``.

    Returns:
        SummaryCandidate if successful and fits budget, or None to escalate.
    """
    to_summarise = _messages_to_summarise(messages)
    if not to_summarise:
        logger.debug("level1_skip_nothing_to_summarise")
        return None

    transcript = _build_messages_text(to_summarise)
    prompt_messages = [
        {
            "role": "user",
            "content": f"{LEVEL1_PROMPT}\n\n<conversation>\n{transcript}\n</conversation>",
        }
    ]

    try:
        summary_text = await llm_call(
            model=model,
            messages=prompt_messages,
            max_tokens=budget.compaction_buffer,
        )
    except Exception as exc:
        logger.warning("level1_llm_failed", error=str(exc))
        return None

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
        span_start_message_id=to_summarise[0].id,
        span_end_message_id=to_summarise[-1].id,
        compaction_level=1,
        messages_covered=len(to_summarise),
    )


async def level2_summarise(
    messages: list[MessageWithParts],
    model: str,
    budget: ContextBudget,
    estimator: TokenEstimator,
    llm_call: Any,
) -> SummaryCandidate | None:
    """
    Attempt Level 2 (aggressive) summarisation via LLM.

    Uses a more compressed prompt format and drops reasoning details.

    Args:
        messages: All non-summary messages in the session.
        model: Model string to use for compaction.
        budget: Token budget for validation.
        estimator: Token estimator for result validation.
        llm_call: Async callable.

    Returns:
        SummaryCandidate if successful and fits budget, or None to escalate.
    """
    to_summarise = _messages_to_summarise(messages)
    if not to_summarise:
        return None

    # For level 2, cap transcript length more aggressively
    transcript_parts: list[str] = []
    for msg in to_summarise:
        text = _extract_text(msg, max_chars=500)
        if text:
            role = "U" if msg.role == "user" else "A"
            transcript_parts.append(f"[{role}]: {text}")
    transcript = "\n".join(transcript_parts)

    prompt_messages = [
        {
            "role": "user",
            "content": f"{LEVEL2_PROMPT}\n\n<conversation>\n{transcript}\n</conversation>",
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
        span_start_message_id=to_summarise[0].id,
        span_end_message_id=to_summarise[-1].id,
        compaction_level=2,
        messages_covered=len(to_summarise),
    )


def level3_deterministic(
    messages: list[MessageWithParts],
    budget: ContextBudget,
    estimator: TokenEstimator,
) -> SummaryCandidate:
    """
    Level 3 deterministic fallback (no LLM required).

    Keeps the most recent messages that fit within 90% of the usable budget,
    prefixed with a truncation notice. This always produces a valid result.

    Args:
        messages: All non-summary messages in the session.
        budget: Token budget — result is guaranteed to fit within usable.
        estimator: Token estimator.

    Returns:
        SummaryCandidate that always fits within budget.usable.
    """
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
