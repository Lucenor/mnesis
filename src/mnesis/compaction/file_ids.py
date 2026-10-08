"""File ID extraction and propagation utilities.

Mnesis's "lossless" guarantee rests on preserving ``file_xxx`` identifiers
across every compaction round.  Even when prose context is discarded, the
pointer to the external file content is never lost.

This module provides:
- :func:`extract_file_ids` — pull all ``file_<hex>`` references from text.
- :func:`append_file_ids_footer` — attach a ``[LCM File IDs: ...]`` footer to
  a summary string when file IDs are present.
- :func:`collect_file_ids_from_nodes` — aggregate file IDs from a list of
  ``SummaryNode`` objects (for condensation input propagation).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from mnesis.models.message import MessageWithParts, TextPart, ToolPart
from mnesis.models.summary import SummaryNode

# Matches Mnesis file IDs: ``file_`` followed by 8-32 hex characters.
# The pattern is intentionally broad to catch variations across providers.
_FILE_ID_RE = re.compile(r"\bfile_[0-9a-fA-F]{8,32}\b")

# Footer template for the ``[LCM File IDs: ...]`` footer.
_FILE_IDS_FOOTER_TEMPLATE = "\n\n[LCM File IDs: {ids}]"

# A trailing ``[LCM File IDs: ...]`` footer (with any blank lines before it).
_EXISTING_FOOTER_RE = re.compile(r"\n*\[LCM File IDs:[^\]]*\]\s*$", re.MULTILINE)


def extract_file_ids(text: str) -> list[str]:
    """
    Extract all ``file_<hex>`` identifiers from *text*.

    Deduplicates and preserves first-occurrence order.

    Args:
        text: Raw text that may contain LCM file ID references.

    Returns:
        Ordered, deduplicated list of file ID strings (e.g.
        ``["file_a1b2c3d4", "file_deadbeef12345678"]``).
    """
    seen: set[str] = set()
    result: list[str] = []
    for match in _FILE_ID_RE.finditer(text):
        fid = match.group()
        if fid not in seen:
            seen.add(fid)
            result.append(fid)
    return result


def message_id_text(msg: MessageWithParts) -> str:
    """
    Return the full raw text of *msg* that may carry ``file_<hex>`` references.

    Unlike the display rendering used for summarisation prompts, nothing is
    truncated: text parts, tool inputs, tool outputs and tool errors are all
    included in full. Pruned tool outputs (``compacted_at`` set) are included
    too -- pruning only tombstones what the *context* shows; the original
    output stays in the append-only store, and an ID that appears only there
    must still reach the summary footer.

    Args:
        msg: Message to scan.

    Returns:
        The concatenated raw text, parts separated by newlines.
    """
    chunks: list[str] = []
    for part in msg.parts:
        if isinstance(part, TextPart):
            chunks.append(part.text)
        elif isinstance(part, ToolPart):
            if part.input:
                chunks.append(json.dumps(part.input, default=str))
            if part.output:
                chunks.append(part.output)
            if part.error_message:
                chunks.append(part.error_message)
    return "\n".join(chunks)


def extract_file_ids_from_messages(messages: list[MessageWithParts]) -> list[str]:
    """
    Extract all file IDs referenced across a list of messages.

    Scans the full raw content of every message (see :func:`message_id_text`),
    including pruned tool outputs, and deduplicates.

    Args:
        messages: Messages to scan for file ID references.

    Returns:
        Ordered, deduplicated list of file ID strings.
    """
    seen: set[str] = set()
    result: list[str] = []
    for msg in messages:
        for fid in extract_file_ids(message_id_text(msg)):
            if fid not in seen:
                seen.add(fid)
                result.append(fid)
    return result


def _recent_first(texts_newest_first: Iterable[str]) -> list[str]:
    """Deduplicated file IDs ordered by *last* occurrence, scanning newest text first."""
    seen: set[str] = set()
    result: list[str] = []
    for text in texts_newest_first:
        for fid in reversed(_FILE_ID_RE.findall(text)):
            if fid not in seen:
                seen.add(fid)
                result.append(fid)
    return result


def most_recent_file_ids(messages: list[MessageWithParts]) -> list[str]:
    """
    File IDs ordered most-recently-referenced first, deduplicated.

    "Recent" means the position of an ID's *last* occurrence in the full raw
    content, scanning messages newest to oldest and each message back to front.

    Args:
        messages: Messages in chronological order.
    """
    return _recent_first(message_id_text(msg) for msg in reversed(messages))


def most_recent_file_ids_from_nodes(nodes: list[SummaryNode]) -> list[str]:
    """Like :func:`most_recent_file_ids`, over summary nodes in chronological order."""
    return _recent_first(node.content for node in reversed(nodes))


def strip_file_ids_footer(text: str) -> str:
    """Remove a trailing ``[LCM File IDs: ...]`` footer from *text*, if present."""
    return _EXISTING_FOOTER_RE.sub("", text)


def collect_file_ids_from_nodes(nodes: list[SummaryNode]) -> list[str]:
    """
    Aggregate all file IDs already embedded in a list of summary nodes.

    Each node's content may contain a ``[LCM File IDs: ...]`` footer; this
    function extracts IDs from every node and returns a deduplicated union.

    Args:
        nodes: Summary nodes whose content is scanned for file IDs.

    Returns:
        Ordered, deduplicated list of file ID strings.
    """
    seen: set[str] = set()
    result: list[str] = []
    for node in nodes:
        for fid in extract_file_ids(node.content):
            if fid not in seen:
                seen.add(fid)
                result.append(fid)
    return result


def append_file_ids_footer(text: str, file_ids: list[str]) -> str:
    """
    Append a ``[LCM File IDs: ...]`` footer to *text* when *file_ids* is non-empty.

    If *text* already contains the footer this function is idempotent — it will
    not duplicate the block.  The footer is always placed at the end.

    Args:
        text: The summary text to annotate.
        file_ids: File IDs to include in the footer.

    Returns:
        Annotated text, or the original text unchanged if *file_ids* is empty.
    """
    if not file_ids:
        return text

    ids_str = ", ".join(file_ids)
    footer = _FILE_IDS_FOOTER_TEMPLATE.format(ids=ids_str)

    # Strip any existing footer before appending the authoritative one.
    return strip_file_ids_footer(text) + footer
