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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import structlog

from mnesis.compaction.file_ids import (
    append_file_ids_footer,
    collect_file_id_paths_from_nodes,
    collect_file_ids_from_nodes,
    extract_file_id_paths_from_messages,
    extract_file_ids_from_messages,
    most_recent_file_ids,
    most_recent_file_ids_from_nodes,
    strip_file_ids_footer,
)
from mnesis.models.message import ContextBudget, MessageWithParts, TextPart, ToolPart
from mnesis.models.summary import SummaryNode
from mnesis.retry import RetriesExhaustedError
from mnesis.tokens.estimator import TokenEstimator

logger = structlog.get_logger("mnesis.compaction.levels")

# 75 % of the compaction model's context window may be used for summarisation
# input cap for summarisation.
MAX_SUMMARISATION_INPUT_FRACTION: float = 0.75

# Minimum number of messages that must be passed to the summariser even when
# the input cap would exclude them.
MIN_MESSAGES_TO_SUMMARISE: int = 3

# Character caps used when rendering messages/summaries as text.  Deliberately
# small: they bound prompt size (levels 1-2) and the size of the deterministic
# fallback (level 3), which must never depend on an LLM.
#
# Default per-message cap (level 1 transcripts).
_MESSAGE_TEXT_MAX_CHARS: int = 2000
# Cap on a single tool output included in a message's text.
_TOOL_OUTPUT_EXCERPT_CHARS: int = 500
# Per-message cap for the aggressive level-2 transcript.
_LEVEL2_MESSAGE_MAX_CHARS: int = 500
# Per-message cap for level-3 truncation; used for both sizing and rendering so
# the budget check measures exactly the text that is emitted.
_LEVEL3_MESSAGE_MAX_CHARS: int = 500
# Per-summary excerpt cap in the level-2 condensation prompt.
_CONDENSE_LEVEL2_NODE_MAX_CHARS: int = 800
# Per-node cap in level-3 condensation, so one oversized summary cannot crowd
# out the others before the token budget check runs.
_CONDENSE_LEVEL3_NODE_MAX_CHARS: int = 2000

# Used when the parent list would crowd the prose or displace file IDs.
_CONDENSE_LEVEL3_MINIMAL_HEADER = "[CONDENSED]\n"

# Maximum tokens of *prose* in a level 3 condensation fallback. The file-ID
# footer is sized separately (against the budget) so it never displaces IDs.
_CONDENSE_LEVEL3_MAX_TOKENS: int = 512

# Upper bound on the ``max_tokens`` requested from the compaction LLM for a
# level 1 summary or condensation. The summary must also fit ``budget.usable``,
# which is the tighter bound for small-window models.
_LEVEL1_MAX_OUTPUT_TOKENS: int = 8_192

LEVEL1_PROMPT = """\
You are creating a detailed context summary to allow continuing this conversation.
Preserve all goals, instructions, constraints, file context, and tool results.
Be thorough — this summary will replace the original messages.

Rules:
- Faithfulness: under In Progress, Remaining Work and next steps, list only items
  the user or assistant explicitly stated. Never invent or infer them. If none were
  explicitly stated, write exactly "None stated". Suggestions or recommendations the
  assistant made are not tasks unless the user accepted or requested them.
- Files: write every file that has an id together with it, as `path (file_<hex>)`,
  exactly as the conversation gives it. Each distinct path keeps its own entry;
  never merge two files under one id or offer guessed alternative paths.
- People: keep every named person with their role (owner, on-call, stakeholder).

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
Preserve only: current goal, active constraints, key file locations (with ids),
named people and their roles, next step.
Suggestions or recommendations the assistant made are not tasks unless the user
accepted or requested them.

Format:
GOAL: <one sentence>
CONSTRAINTS: <comma-separated list>
FILES: <key files, each as `path (file_<hex>)` when it has an id; never merge two files>
PEOPLE: <named people and roles (owners, on-call, stakeholders), one compact line, or "none">
NEXT: <next action ONLY if explicitly stated; never invent; else exactly "None stated">
CONTEXT: <any other critical facts, max 3 sentences>
"""

CONDENSE_LEVEL1_PROMPT = """\
You are condensing multiple context summaries into one unified summary.
Each summary below represents a portion of the conversation history.
Merge them into a single coherent summary that preserves all critical information.

Rules:
- Merge and deduplicate across the input summaries; do not copy them. State each
  fact once, in your own condensed wording.
- Omit any section that would be empty, except In Progress and Remaining Work,
  which say exactly "None stated" when nothing was explicitly stated.
- Faithfulness: under In Progress, Remaining Work and next steps, list only items
  the user or assistant explicitly stated. Never invent or infer them. If none were
  explicitly stated, write exactly "None stated". Suggestions or recommendations the
  assistant made are not tasks unless the user accepted or requested them.
- Files: write every file that has an id together with it, as `path (file_<hex>)`,
  exactly as the conversation gives it. Each distinct path keeps its own entry;
  never merge two files under one id or offer guessed alternative paths.
- People: keep every named person with their role (owner, on-call, stakeholder).

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
Keep only: current goal, key constraints, critical files (with ids), named people
and their roles, immediate next step.
Suggestions or recommendations the assistant made are not tasks unless the user
accepted or requested them.

Format:
GOAL: <one sentence>
CONSTRAINTS: <comma-separated list>
FILES: <key files, each as `path (file_<hex>)` when it has an id; never merge two files>
PEOPLE: <named people and roles (owners, on-call, stakeholders), one compact line, or "none">
NEXT: <only an explicitly stated next action; never invent; else exactly "None stated">
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


def _level1_max_tokens(budget: ContextBudget, model_max_output_tokens: int = 0) -> int:
    """``max_tokens`` for a level 1 LLM call.

    Bounded by the budget the result must fit and, when known (> 0), by the
    compaction model's own output limit.
    """
    limit = min(_LEVEL1_MAX_OUTPUT_TOKENS, budget.usable)
    if model_max_output_tokens > 0:
        limit = min(limit, model_max_output_tokens)
    return max(1, limit)


def _level2_max_tokens(budget: ContextBudget, model_max_output_tokens: int = 0) -> int:
    """``max_tokens`` for a level 2 LLM call (see :func:`_level1_max_tokens`)."""
    limit = min(budget.compaction_buffer, 4000)
    if model_max_output_tokens > 0:
        limit = min(limit, model_max_output_tokens)
    return max(1, limit)


# A summary/condensation should come out at about this fraction of its input...
_LENGTH_TARGET_INPUT_FRACTION: float = 0.5
# ...and at most this fraction of the ``max_tokens`` the call may use, so a
# verbose model still finishes before the output cap (``finish_reason="length"``
# discards the whole completion and forces an escalation).
_LENGTH_TARGET_CAP_FRACTION: float = 0.75


def _length_target(input_tokens: int, max_tokens: int) -> int:
    """Output size (tokens) to ask the model for: about half the input, under the cap."""
    return max(
        1,
        min(
            int(input_tokens * _LENGTH_TARGET_INPUT_FRACTION),
            int(max_tokens * _LENGTH_TARGET_CAP_FRACTION),
        ),
    )


def _with_length_target(prompt: str, target_tokens: int, max_tokens: int) -> str:
    """Append an explicit output-length instruction to *prompt*."""
    return (
        f"{prompt.rstrip()}\n\n"
        f"Length: aim for about {target_tokens} tokens in total (hard limit "
        f"{max_tokens} tokens; a longer answer is discarded). Compress wording, "
        f"not facts: keep every constraint, number, identifier and path."
    )


# Condensation sizing: a bullet is capped at about this many words (~35 tokens),
# and the summary has this many sections, so the per-section bullet cap follows
# from the token target.
_CONDENSE_BULLET_WORDS: int = 25
_CONDENSE_BULLET_TOKENS: int = 35
_CONDENSE_SECTIONS: int = 8


def _with_condense_limits(prompt: str, target_tokens: int, max_tokens: int) -> str:
    """Append structural size limits to a condensation prompt.

    Models follow "at most N bullets per section" far more reliably than a token
    target, so the target is turned into a per-section bullet cap.
    """
    bullets = max(1, target_tokens // (_CONDENSE_SECTIONS * _CONDENSE_BULLET_TOKENS))
    return (
        f"{prompt.rstrip()}\n\n"
        f"Limits: at most {bullets} bullets per section, each at most about "
        f"{_CONDENSE_BULLET_WORDS} words (hard limit {max_tokens} tokens; a longer "
        f"answer is discarded). Prefer merging bullets over dropping facts: keep "
        f"every constraint, number, identifier and path."
    )


def _extract_text(msg: MessageWithParts, max_chars: int = _MESSAGE_TEXT_MAX_CHARS) -> str:
    """Extract readable text from a message; the result is at most ``max_chars`` long."""
    parts: list[str] = []
    remaining = max_chars
    for part in msg.parts:
        if remaining <= 0:
            break
        if isinstance(part, TextPart):
            piece = part.text
        elif isinstance(part, ToolPart) and part.compacted_at is None and part.output:
            piece = f"[Tool {part.tool_name}]: {part.output[:_TOOL_OUTPUT_EXCERPT_CHARS]}"
        else:
            continue
        piece = piece[:remaining]
        parts.append(piece)
        remaining -= len(piece) + 1  # +1 for the "\n" separator
    return "\n".join(parts)


def _render_message(msg: MessageWithParts, max_chars: int = _MESSAGE_TEXT_MAX_CHARS) -> str:
    """One message as it appears in the transcript ("" when it has no text)."""
    text = _extract_text(msg, max_chars)
    if not text:
        return ""
    role_label = "USER" if msg.role == "user" else "ASSISTANT"
    return f"[{role_label}]:\n{text}"


def _build_messages_text(messages: list[MessageWithParts]) -> str:
    """Format a list of messages as a readable transcript."""
    return "\n\n".join(r for r in (_render_message(m) for m in messages) if r)


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
    reserved_tokens: int = 0,
    max_chars: int = _MESSAGE_TEXT_MAX_CHARS,
) -> list[MessageWithParts]:
    """
    Trim *messages* so their rendered transcript stays within the summarisation
    input cap: ``MAX_SUMMARISATION_INPUT_FRACTION`` of *model_context_limit*,
    and never more than the window left after *reserved_tokens* (the request's
    ``max_tokens`` plus its prompt).

    Messages are taken oldest-first, so the result is a contiguous prefix of
    *messages*; callers must record the span of the *result* (not of the
    input), leaving the rest raw for a later compaction. Oldest-first keeps the
    summary chronologically contiguous with earlier summaries and leaves the
    newest turns verbatim in context. At least :data:`MIN_MESSAGES_TO_SUMMARISE`
    messages are always included even if they exceed the cap.

    Args:
        messages: Messages to cap (already filtered by ``_messages_to_summarise``).
        estimator: Token estimator.
        model_context_limit: Full context limit of the compaction model.
        reserved_tokens: Tokens of the window the input must leave free.
        max_chars: Per-message text cap of the transcript that will be sent;
            each message is measured as rendered (after this truncation), not whole.

    Returns:
        A (possibly shorter) list of messages to pass to the LLM.
    """
    if not messages:
        return messages

    max_input_tokens = min(
        int(model_context_limit * MAX_SUMMARISATION_INPUT_FRACTION),
        model_context_limit - reserved_tokens,
    )
    tokens_so_far = 0
    result: list[MessageWithParts] = []

    for msg in messages:
        # Rendered text plus its "\n\n" separator; +1 covers per-message rounding so the
        # sum never undercounts the joined transcript.
        msg_tokens = estimator.estimate(_render_message(msg, max_chars) + "\n\n") + 1
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
    model_max_output_tokens: int = 0,
    compaction_estimator: TokenEstimator | None = None,
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
        model_max_output_tokens: Output limit of the compaction model (0 = unknown).
        compaction_estimator: Estimator in the compaction model's units, used to size
            the input against its window (defaults to *estimator*).

    Returns:
        SummaryCandidate if successful and fits budget, or None to escalate.
    """
    to_summarise = _messages_to_summarise(messages)
    if not to_summarise:
        logger.debug("level1_skip_nothing_to_summarise")
        return None

    # Apply input token cap before passing to LLM. The oldest prefix is kept, and
    # the span recorded below is exactly that prefix: messages past the cap stay
    # raw in context for the next compaction instead of being swapped out unsummarised.
    prompt = compaction_prompt if compaction_prompt is not None else LEVEL1_PROMPT
    max_tokens = _level1_max_tokens(budget, model_max_output_tokens)
    cap_estimator = compaction_estimator or estimator
    # The length-target line added below counts against the window too; size it
    # for the largest target (its digits never exceed those of ``max_tokens``).
    sized_prompt = (
        prompt
        if compaction_prompt is not None
        else _with_length_target(prompt, max_tokens, max_tokens)
    )
    to_summarise = _apply_input_cap(
        to_summarise,
        cap_estimator,
        model_context_limit,
        reserved_tokens=max_tokens + cap_estimator.estimate(sized_prompt),
    )

    # Collect file IDs from the capped input.
    file_ids = extract_file_ids_from_messages(to_summarise)

    file_paths = extract_file_id_paths_from_messages(to_summarise)

    transcript = _build_messages_text(to_summarise)
    input_token_count = estimator.estimate(transcript)
    if compaction_prompt is None:
        # An explicit length target lets a verbose model finish under the output
        # cap instead of being cut off (and discarded). A custom prompt is left as is.
        prompt = _with_length_target(
            # Target and ``max_tokens`` are both the compaction model's units.
            prompt,
            _length_target(cap_estimator.estimate(transcript), max_tokens),
            max_tokens,
        )
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
            max_tokens=max_tokens,
        )
    except RetriesExhaustedError:
        raise  # outage: the engine skips the remaining LLM levels
    except Exception as exc:
        logger.warning("level1_llm_failed", error=str(exc))
        return None

    if not (summary_text or "").strip():
        # An empty completion must never become the summary that replaces history.
        logger.warning("level1_empty_completion")
        return None

    # Propagate file IDs into the summary.
    summary_text = _append_bounded_footer(
        summary_text,
        file_ids,
        most_recent_file_ids(to_summarise),
        budget,
        estimator,
        paths=file_paths,
    )

    token_count = estimator.estimate(summary_text)
    if token_count > budget.usable:
        logger.info(
            "level1_summary_too_large",
            token_count=token_count,
            usable=budget.usable,
        )
        return None

    # Convergence check: if the summary is as large or larger than the input,
    # the LLM failed to compress — escalate to the next level.
    if token_count >= input_token_count:
        logger.info(
            "level1_no_convergence",
            summary_tokens=token_count,
            input_tokens=input_token_count,
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
    compaction_prompt: str | None = None,
    model_context_limit: int = 200_000,
    model_max_output_tokens: int = 0,
    compaction_estimator: TokenEstimator | None = None,
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
        model_max_output_tokens: Output limit of the compaction model (0 = unknown).
        compaction_estimator: Estimator in the compaction model's units, used to size
            the input against its window (defaults to *estimator*).

    Returns:
        SummaryCandidate if successful and fits budget, or None to escalate.
    """
    to_summarise = _messages_to_summarise(messages)
    if not to_summarise:
        return None

    # Apply input token cap (oldest prefix; the span below matches what is summarised).
    prompt = compaction_prompt if compaction_prompt is not None else LEVEL2_PROMPT
    max_tokens = _level2_max_tokens(budget, model_max_output_tokens)
    cap_estimator = compaction_estimator or estimator
    to_summarise = _apply_input_cap(
        to_summarise,
        cap_estimator,
        model_context_limit,
        reserved_tokens=max_tokens + cap_estimator.estimate(prompt),
        max_chars=_LEVEL2_MESSAGE_MAX_CHARS,
    )

    # Collect file IDs from input messages.
    file_ids = extract_file_ids_from_messages(to_summarise)

    # For level 2, cap transcript length more aggressively
    transcript_parts: list[str] = []
    for msg in to_summarise:
        text = _extract_text(msg, max_chars=_LEVEL2_MESSAGE_MAX_CHARS)
        if text:
            role = "U" if msg.role == "user" else "A"
            transcript_parts.append(f"[{role}]: {text}")
    transcript = "\n".join(transcript_parts)
    input_token_count = estimator.estimate(transcript)
    file_paths = extract_file_id_paths_from_messages(to_summarise)

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
            max_tokens=max_tokens,
        )
    except RetriesExhaustedError:
        raise  # outage: the engine skips the remaining LLM levels
    except Exception as exc:
        logger.warning("level2_llm_failed", error=str(exc))
        return None

    if not (summary_text or "").strip():
        # An empty completion must never become the summary that replaces history.
        logger.warning("level2_empty_completion")
        return None

    # Propagate file IDs.
    summary_text = _append_bounded_footer(
        summary_text,
        file_ids,
        most_recent_file_ids(to_summarise),
        budget,
        estimator,
        paths=file_paths,
    )

    token_count = estimator.estimate(summary_text)
    if token_count > budget.usable:
        logger.info(
            "level2_summary_too_large",
            token_count=token_count,
            usable=budget.usable,
        )
        return None

    # Convergence check: if the summary is as large or larger than the input,
    # the LLM failed to compress — escalate to the next level.
    if token_count >= input_token_count:
        logger.info(
            "level2_no_convergence",
            summary_tokens=token_count,
            input_tokens=input_token_count,
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


_LEVEL3_HEADER = "[CONTEXT TRUNCATED — DETERMINISTIC FALLBACK]\n\n## Kept Messages\n\n"
# Used when the full header exceeds the level-3 budget or would displace file IDs.
_LEVEL3_MINIMAL_HEADER = "[TRUNCATED]\n"
# Fraction of ``budget.usable`` that level-3 sizes prose and file IDs against.
# This is an inherited heuristic safety margin (it predates the final
# validation); the assembled text is additionally validated against 100% of
# ``budget.usable`` with the same estimator.
_LEVEL3_BUDGET_FRACTION: float = 0.85


def _fit_file_ids(
    file_ids: list[str],
    reserved_tokens: int,
    cap: int,
    estimator: TokenEstimator,
) -> list[str]:
    """Return the longest prefix of *file_ids* whose footer fits in ``cap - reserved_tokens``.

    Binary search over the prefix length: O(log n) estimator calls, so a very
    large ID set cannot make level 3 slow.  The empty prefix (no footer) is
    always acceptable, so the result may be empty when ``reserved_tokens``
    alone reaches ``cap``.
    """

    def cost(k: int) -> int:
        return estimator.estimate(append_file_ids_footer("", file_ids[:k])) if k else 0

    if reserved_tokens + cost(len(file_ids)) <= cap:
        return file_ids
    lo, hi = 0, len(file_ids)  # lo is always acceptable (empty footer); hi never fits
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if reserved_tokens + cost(mid) <= cap:
            lo = mid
        else:
            hi = mid
    return file_ids[:lo]


def _paths_that_fit(
    file_ids: list[str],
    paths: Mapping[str, str],
    reserved_tokens: int,
    cap: int,
    estimator: TokenEstimator,
) -> dict[str, str]:
    """Return the ``{id: path}`` pairs of *file_ids* if the paired footer fits *cap*.

    All-or-nothing: IDs outrank paths, so when the footer with its paths would
    not fit (``reserved_tokens`` already used), the bare IDs are written instead.
    """
    chosen = {fid: paths[fid] for fid in file_ids if fid in paths}
    if not chosen:
        return {}
    footer = append_file_ids_footer("", file_ids, chosen)
    return chosen if reserved_tokens + estimator.estimate(footer) <= cap else {}


def _with_paths_if_fit(
    text: str,
    file_ids: list[str],
    paths: Mapping[str, str],
    budget: ContextBudget,
    estimator: TokenEstimator,
) -> tuple[str, int]:
    """Upgrade *text*'s bare-ID footer to ``path (file_<hex>)`` pairs if they still fit.

    Priority is ids > prose > paths: the prose is already fixed, so paths only
    take what remains of ``budget.usable``. Returns the text and its token count.
    """
    chosen = _paths_that_fit(
        file_ids, paths, estimator.estimate(strip_file_ids_footer(text)), budget.usable, estimator
    )
    if chosen:
        paired = append_file_ids_footer(strip_file_ids_footer(text), file_ids, chosen)
        tokens = estimator.estimate(paired)
        if tokens <= budget.usable:
            return paired, tokens
    return text, estimator.estimate(text)


def _append_bounded_footer(
    text: str,
    file_ids: list[str],
    recent_first: list[str],
    budget: ContextBudget,
    estimator: TokenEstimator,
    paths: Mapping[str, str] | None = None,
) -> str:
    """Append the file-ID footer to an LLM summary.

    File IDs take priority over the model's prose (invariant: references are
    never lost). IDs are dropped only when the footer *alone* (plus the prose
    already present) cannot fit ``budget.usable``, which no escalation can
    avoid; the least recently referenced go first and the footer stays in
    first-occurrence order, as in level 3. When the footer fits alone but
    text + footer does not, the full footer is appended and the caller's budget
    check rejects the candidate, so the run escalates to the next level.

    ``paths`` pairs IDs with their files (``path (file_<hex>)``); they are
    written only when the paired footer still fits the budget with the prose.
    """
    if file_ids and estimator.estimate(append_file_ids_footer("", file_ids)) > budget.usable:
        # Fit the IDs against the budget alone, not what the prose leaves: the
        # caller's budget check then rejects this candidate and the run escalates
        # to level 3, which keeps as many IDs as the budget allows.
        fitted = _fit_file_ids(recent_first, 0, budget.usable, estimator)
        keep = set(fitted)
        logger.warning(
            "summary_file_ids_truncated",
            kept=len(fitted),
            dropped=len(file_ids) - len(fitted),
            budget_usable=budget.usable,
        )
        file_ids = [fid for fid in file_ids if fid in keep]
        paths = None
    if paths:
        chosen = _paths_that_fit(
            file_ids,
            paths,
            estimator.estimate(strip_file_ids_footer(text)),
            budget.usable,
            estimator,
        )
        return append_file_ids_footer(text, file_ids, chosen)
    return append_file_ids_footer(text, file_ids)


def _plan_header_and_file_ids(
    *,
    full_header: str,
    minimal_header: str,
    all_file_ids: list[str],
    recent_first: list[str],
    cap: int,
    usable: int,
    estimator: TokenEstimator,
    log_event: str,
) -> tuple[str, int, list[str]]:
    """Choose a level-3 header and the file IDs to keep so header + footer fit *cap*.

    File IDs take precedence over the decorative header: the full header is
    used only when it does not displace any ID. When even the minimal header
    plus the ID footer exceed *cap*, the least recently referenced IDs are
    dropped (the footer stays in first-occurrence order) and a warning is
    logged. If *usable* is smaller than the minimal header no output can fit,
    so all IDs are kept (lossless wins).

    Returns:
        ``(header, header_tokens, file_ids)``.
    """
    minimal_tokens = estimator.estimate(minimal_header)
    header, header_tokens = full_header, estimator.estimate(full_header)
    if header_tokens > cap:
        header, header_tokens = minimal_header, minimal_tokens

    file_ids = all_file_ids
    if usable >= minimal_tokens and all_file_ids:
        fitted = _fit_file_ids(recent_first, header_tokens, cap, estimator)
        if len(fitted) < len(recent_first) and header != minimal_header:
            # The decorative header is not worth displacing IDs: retry minimal.
            retry = _fit_file_ids(recent_first, minimal_tokens, cap, estimator)
            if len(retry) > len(fitted):
                header, header_tokens, fitted = minimal_header, minimal_tokens, retry
        if len(fitted) < len(all_file_ids):
            keep = set(fitted)
            file_ids = [fid for fid in all_file_ids if fid in keep]
    if len(file_ids) < len(all_file_ids):
        logger.warning(
            log_event,
            kept=len(file_ids),
            dropped=len(all_file_ids) - len(file_ids),
            budget_usable=usable,
        )
    return header, header_tokens, file_ids


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

    Precedence when the budget is tight: file IDs are reserved first, then
    prose fills what remains, so messages are dropped before any file ID is.
    Only if the header plus the ID footer cannot fit in 85% of
    ``budget.usable`` is the footer truncated, keeping the *most recently
    referenced* IDs (by last occurrence; the footer itself stays in
    first-occurrence order).  A warning with the dropped count is logged; the
    raw files stay addressable in the immutable store.  Truncating there is the
    only way to keep the "always fits the budget" guarantee that lets
    compaction make progress.  If ``budget.usable`` is smaller than the minimal
    header, no output can fit, so all IDs are kept (lossless wins).

    Args:
        messages: All non-summary messages in the session.
        budget: Token budget — result is guaranteed to fit within usable.
        estimator: Token estimator.

    Returns:
        SummaryCandidate that fits within budget.usable (barring a budget too
        small for even the minimal header), as measured by *estimator*.
    """
    # Collect file IDs from ALL messages before truncation — the whole point of
    # level 3 is that we never lose file pointers even when prose is discarded.
    all_file_ids = extract_file_ids_from_messages(messages)
    all_paths = extract_file_id_paths_from_messages(messages) if all_file_ids else {}

    cap = int(budget.usable * _LEVEL3_BUDGET_FRACTION)
    header, header_tokens, file_ids = _plan_header_and_file_ids(
        full_header=_LEVEL3_HEADER,
        minimal_header=_LEVEL3_MINIMAL_HEADER,
        all_file_ids=all_file_ids,
        recent_first=most_recent_file_ids(messages) if all_file_ids else [],
        cap=cap,
        usable=budget.usable,
        estimator=estimator,
        log_event="level3_file_ids_truncated",
    )
    # Sized with bare ids: ids > prose > paths, so paths never displace prose.
    footer_tokens = estimator.estimate(append_file_ids_footer("", file_ids)) if file_ids else 0
    target = cap - footer_tokens

    kept_rev: list[tuple[MessageWithParts, str]] = []
    tokens_used = header_tokens
    for msg in reversed(messages):
        text = _extract_text(msg, max_chars=_LEVEL3_MESSAGE_MAX_CHARS)
        role = "USER" if msg.role == "user" else "ASSISTANT"
        line = f"[{role}]: {text}\n"
        line_tokens = estimator.estimate(line)
        if tokens_used + line_tokens > target:
            break
        kept_rev.append((msg, line))
        tokens_used += line_tokens

    kept = [m for m, _ in reversed(kept_rev)]
    lines = [line for _, line in reversed(kept_rev)]

    def render(first: int) -> str:
        # Footer preserves file references even for truncated content.
        return append_file_ids_footer("\n".join([header, *lines[first:]]), file_ids)

    # The per-line sizing above ignores the "\n" joiners and footer separators;
    # validate the assembled text against the hard budget by shedding the
    # oldest kept messages.  Rendered size is non-increasing in the number shed,
    # so binary-search the smallest count that fits (O(n log n), not O(n^2)).
    summary_text = render(0)
    token_count = estimator.estimate(summary_text)
    if token_count > budget.usable and lines:
        lo, hi = 0, len(lines)  # lo does not fit; hi (all shed) is the floor
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if estimator.estimate(render(mid)) <= budget.usable:
                hi = mid
            else:
                lo = mid
        kept = kept[hi:]
        summary_text = render(hi)
        token_count = estimator.estimate(summary_text)
    if all_paths and file_ids:
        summary_text, token_count = _with_paths_if_fit(
            summary_text, file_ids, all_paths, budget, estimator
        )

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


# What surrounds the summaries in the request, after the prompt: the blank line, the
# tags and their newlines (see the ``f"{prompt}\n\n<summaries>\n...\n</summaries>"`` below).
_SUMMARIES_WRAPPER = "\n\n<summaries>\n\n</summaries>"
_CONDENSE_L1_SEPARATOR = "\n\n---\n\n"
_CONDENSE_L2_SEPARATOR = "\n\n"


def _fit_condensation_nodes(
    nodes: list[SummaryNode],
    texts: list[str],
    cap_estimator: TokenEstimator,
    model_context_limit: int,
    reserved_tokens: int,
    level: int,
    separator: str,
) -> list[SummaryNode]:
    """Oldest-first prefix of *nodes* whose rendered *texts* fit the compaction model.

    The input cap mirrors summarisation's (:func:`_apply_input_cap`):
    ``MAX_SUMMARISATION_INPUT_FRACTION`` of the model's window, and never more than
    what is left after *reserved_tokens* (the prompt plus the requested output).
    Tokens are counted with *cap_estimator* (the compaction model's units).

    Condensing the oldest nodes first matches the oldest-first drain and keeps the
    condensed node contiguous with the nodes that stay live. At least two nodes
    are needed for a merge to mean anything (one when only one exists); when even
    that does not fit, an empty list is returned so the level escalates (level 2
    sends bounded excerpts, level 3 needs no model).

    Returns:
        The nodes to condense, oldest first, or ``[]`` if too few fit.
    """
    if model_context_limit <= 0:
        return list(nodes)
    max_input = min(
        int(model_context_limit * MAX_SUMMARISATION_INPUT_FRACTION),
        model_context_limit - reserved_tokens,
    )
    # The joiners between summaries and the ``<summaries>`` wrapper are sent too.
    # Each piece is counted separately, so +1 per piece covers estimator rounding
    # (the sum must never undercount the assembled request).
    joiner = cap_estimator.estimate(separator)
    used = cap_estimator.estimate(_SUMMARIES_WRAPPER) + 1
    count = 0
    for text in texts:
        tokens = cap_estimator.estimate(text) + 1 + (joiner if count else 0)
        if used + tokens > max_input:
            break
        used += tokens
        count += 1
    if count < min(2, len(nodes)):
        logger.warning(
            "condensation_input_exceeds_window",
            condense_level=level,
            fitting=count,
            total=len(nodes),
            cap_tokens=max_input,
        )
        return []
    if count < len(nodes):
        logger.info(
            "condensation_input_cap_applied",
            condense_level=level,
            included=count,
            total=len(nodes),
            cap_tokens=max_input,
        )
    return nodes[:count]


async def condense_level1(
    nodes: list[SummaryNode],
    model: str,
    budget: ContextBudget,
    estimator: TokenEstimator,
    llm_call: Any,
    model_max_output_tokens: int = 0,
    model_context_limit: int = 200_000,
    compaction_estimator: TokenEstimator | None = None,
) -> CondensationCandidate | None:
    """
    Attempt Level 1 condensation: merge summary nodes via structured LLM prompt.

    The input is capped against the compaction model's window (see
    :func:`_fit_condensation_nodes`): when the nodes do not all fit, the oldest
    ones that do are condensed and the candidate's ``parent_node_ids`` name
    exactly those; the rest stay live for a later round. File IDs from the
    condensed nodes are collected and appended to the output via a
    ``[LCM File IDs: ...]`` footer.

    Args:
        nodes: Summary nodes to condense (must be non-empty).
        model: LLM model string.
        budget: Token budget for the condensed result.
        estimator: Token estimator.
        llm_call: Async callable ``(model, messages, max_tokens) -> str``.
        model_max_output_tokens: Output limit of the compaction model (0 = unknown).
        model_context_limit: Context window of the compaction model.
        compaction_estimator: Estimator in the compaction model's units, used to size
            the input against its window (defaults to *estimator*).

    Returns:
        CondensationCandidate if successful and fits budget, or None to escalate.
    """
    if not nodes:
        return None

    max_tokens = _level1_max_tokens(budget, model_max_output_tokens)
    cap_estimator = compaction_estimator or estimator
    # The length-target line counts against the window too (sized for its largest form).
    sized_prompt = _with_condense_limits(CONDENSE_LEVEL1_PROMPT, max_tokens, max_tokens)
    nodes = _fit_condensation_nodes(
        nodes,
        [f"[Summary {i + 1}]:\n{node.content}" for i, node in enumerate(nodes)],
        cap_estimator,
        model_context_limit,
        reserved_tokens=max_tokens + cap_estimator.estimate(sized_prompt),
        level=1,
        separator=_CONDENSE_L1_SEPARATOR,
    )
    if not nodes:
        return None

    # Gather file IDs from the nodes being condensed (already embedded in their content).
    file_ids = collect_file_ids_from_nodes(nodes)

    file_paths = collect_file_id_paths_from_nodes(nodes)

    summaries_text = _CONDENSE_L1_SEPARATOR.join(
        f"[Summary {i + 1}]:\n{node.content}" for i, node in enumerate(nodes)
    )
    # Condensation must shrink its input: derive per-section bullet caps from about
    # half of it (within the output cap) so the first level can succeed.
    # In the compaction model's units, like ``max_tokens`` (not the session's).
    input_tokens = cap_estimator.estimate(summaries_text)
    prompt = _with_condense_limits(
        CONDENSE_LEVEL1_PROMPT, _length_target(input_tokens, max_tokens), max_tokens
    )
    prompt_messages = [
        {
            "role": "user",
            "content": f"{prompt}\n\n<summaries>\n{summaries_text}\n</summaries>",
        }
    ]

    try:
        condensed_text = await llm_call(
            model=model,
            messages=prompt_messages,
            max_tokens=max_tokens,
        )
    except RetriesExhaustedError:
        raise  # outage: the engine skips the remaining LLM levels
    except Exception as exc:
        logger.warning("condense_level1_llm_failed", error=str(exc))
        return None

    if not (condensed_text or "").strip():
        # An empty completion must never become the summary that replaces history.
        logger.warning("condense_level1_empty_completion")
        return None

    condensed_text = _append_bounded_footer(
        condensed_text,
        file_ids,
        most_recent_file_ids_from_nodes(nodes),
        budget,
        estimator,
        paths=file_paths,
    )

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
    model_max_output_tokens: int = 0,
    model_context_limit: int = 200_000,
    compaction_estimator: TokenEstimator | None = None,
) -> CondensationCandidate | None:
    """
    Attempt Level 2 condensation: aggressive merge via compressed prompt.

    Each node contributes a bounded excerpt. The input is still capped against
    the compaction model's window (as in :func:`condense_level1`): when the
    excerpts do not all fit, the oldest nodes that do are condensed and the
    candidate's ``parent_node_ids`` name exactly those.

    Args:
        nodes: Summary nodes to condense.
        model: LLM model string.
        budget: Token budget.
        estimator: Token estimator.
        llm_call: Async callable.
        model_max_output_tokens: Output limit of the compaction model (0 = unknown).
        model_context_limit: Context window of the compaction model.
        compaction_estimator: Estimator in the compaction model's units, used to size
            the input against its window (defaults to *estimator*).

    Returns:
        CondensationCandidate if successful and fits budget, or None to escalate.
    """
    if not nodes:
        return None

    max_tokens = _level2_max_tokens(budget, model_max_output_tokens)
    cap_estimator = compaction_estimator or estimator
    # Use a truncated excerpt from each summary for the aggressive prompt.
    excerpts = [
        f"[S{i + 1}]: {node.content[:_CONDENSE_LEVEL2_NODE_MAX_CHARS]}"
        for i, node in enumerate(nodes)
    ]
    nodes = _fit_condensation_nodes(
        nodes,
        excerpts,
        cap_estimator,
        model_context_limit,
        reserved_tokens=max_tokens + cap_estimator.estimate(CONDENSE_LEVEL2_PROMPT),
        level=2,
        separator=_CONDENSE_L2_SEPARATOR,
    )
    if not nodes:
        return None

    file_ids = collect_file_ids_from_nodes(nodes)
    file_paths = collect_file_id_paths_from_nodes(nodes)

    summaries_text = _CONDENSE_L2_SEPARATOR.join(excerpts[: len(nodes)])
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
            max_tokens=max_tokens,
        )
    except RetriesExhaustedError:
        raise  # outage: the engine skips the remaining LLM levels
    except Exception as exc:
        logger.warning("condense_level2_llm_failed", error=str(exc))
        return None

    if not (condensed_text or "").strip():
        # An empty completion must never become the summary that replaces history.
        logger.warning("condense_level2_empty_completion")
        return None

    condensed_text = _append_bounded_footer(
        condensed_text,
        file_ids,
        most_recent_file_ids_from_nodes(nodes),
        budget,
        estimator,
        paths=file_paths,
    )

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


def _truncate_to_tokens(text: str, max_tokens: int, estimator: TokenEstimator) -> str:
    """Return the longest prefix of *text* that estimates to at most *max_tokens*.

    Binary search over the prefix length: O(log n) estimator calls.
    """
    if max_tokens <= 0:
        return ""
    if estimator.estimate(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)  # lo always fits (empty); hi never does
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if estimator.estimate(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid
    return text[:lo]


def condense_level3_deterministic(
    nodes: list[SummaryNode],
    estimator: TokenEstimator,
    budget: ContextBudget,
) -> CondensationCandidate:
    """
    Level 3 deterministic condensation fallback (no LLM required).

    Concatenates the parent summaries' prose (without their own file-ID
    footers) up to :data:`_CONDENSE_LEVEL3_MAX_TOKENS` tokens, then appends a
    single ``[LCM File IDs: ...]`` footer with the IDs from every parent node.

    Bounded like :func:`level3_deterministic`: the footer and header are sized
    against 85% of ``budget.usable`` with *estimator* and the assembled text
    is validated against ``budget.usable``.  File IDs take precedence over
    prose (prose gets what remains), and over the ``[Condensed from: ...]``
    parent list, which is replaced by a minimal header when it would crowd the
    prose.  Only when the footer cannot fit the budget are the least recently
    referenced IDs dropped (with a warning); the raw files stay addressable in
    the immutable store.

    Args:
        nodes: Summary nodes to condense.
        estimator: Token estimator.
        budget: Token budget the result must fit within.

    Returns:
        CondensationCandidate that always succeeds and fits ``budget.usable``
        (barring a budget too small for even the minimal header).
    """
    all_file_ids = collect_file_ids_from_nodes(nodes)
    all_paths = collect_file_id_paths_from_nodes(nodes) if all_file_ids else {}

    cap = int(budget.usable * _LEVEL3_BUDGET_FRACTION)
    prose_cap = min(_CONDENSE_LEVEL3_MAX_TOKENS, cap)

    parent_ids_str = ", ".join(n.id for n in nodes)
    full_header = f"[CONDENSED — DETERMINISTIC FALLBACK]\n[Condensed from: {parent_ids_str}]\n\n"
    if estimator.estimate(full_header) > prose_cap // 2:
        full_header = _CONDENSE_LEVEL3_MINIMAL_HEADER
    header, header_tokens, file_ids = _plan_header_and_file_ids(
        full_header=full_header,
        minimal_header=_CONDENSE_LEVEL3_MINIMAL_HEADER,
        all_file_ids=all_file_ids,
        recent_first=most_recent_file_ids_from_nodes(nodes) if all_file_ids else [],
        cap=cap,
        usable=budget.usable,
        estimator=estimator,
        log_event="condense_level3_file_ids_truncated",
    )
    # Sized with bare ids: ids > prose > paths, so paths never displace prose.
    footer_tokens = estimator.estimate(append_file_ids_footer("", file_ids)) if file_ids else 0
    available = prose_cap - header_tokens - footer_tokens

    # Each node's own footer is dropped from the prose: the authoritative one is
    # appended below. Take as much as fits from each node in order.
    chunks: list[str] = []
    used = 0
    for node in nodes:
        remaining = available - used
        if remaining <= 0:
            break
        chunk = strip_file_ids_footer(node.content)[:_CONDENSE_LEVEL3_NODE_MAX_CHARS]
        chunk_tokens = estimator.estimate(chunk)
        if chunk_tokens > remaining:
            chunk = _truncate_to_tokens(chunk, remaining, estimator)
            if chunk:
                chunks.append(chunk)
            break
        chunks.append(chunk)
        used += chunk_tokens

    def render(count: int) -> str:
        body = header + "\n\n---\n\n".join(chunks[:count])
        return append_file_ids_footer(body, file_ids)

    # The sizing above ignores the "---" joiners; validate the assembled text
    # against the hard budget by shedding trailing chunks (monotone, so
    # binary-search the largest count that fits).
    combined = render(len(chunks))
    token_count = estimator.estimate(combined)
    if token_count > budget.usable and chunks:
        lo, hi = 0, len(chunks)  # hi does not fit; lo (none kept) is the floor
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if estimator.estimate(render(mid)) <= budget.usable:
                lo = mid
            else:
                hi = mid
        combined = render(lo)
        token_count = estimator.estimate(combined)
    if all_paths and file_ids:
        combined, token_count = _with_paths_if_fit(combined, file_ids, all_paths, budget, estimator)

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
