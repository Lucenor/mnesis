"""File ID extraction and propagation utilities.

Volt's "lossless" guarantee rests on preserving ``file_xxx`` identifiers
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

import re

from mnesis.models.message import MessageWithParts
from mnesis.models.summary import SummaryNode

# Matches Volt-style file IDs: ``file_`` followed by 8-32 hex characters.
# The pattern is intentionally broad to catch variations across providers.
_FILE_ID_RE = re.compile(r"\bfile_[0-9a-fA-F]{8,32}\b")

# Footer template — mirrors Volt's ``[LCM File IDs: ...]`` format.
_FILE_IDS_FOOTER_TEMPLATE = "\n\n[LCM File IDs: {ids}]"


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


def extract_file_ids_from_messages(messages: list[MessageWithParts]) -> list[str]:
    """
    Extract all file IDs referenced across a list of messages.

    Concatenates all text content from *messages* and deduplicates.

    Args:
        messages: Messages to scan for file ID references.

    Returns:
        Ordered, deduplicated list of file ID strings.
    """
    from mnesis.compaction.levels import _extract_text

    seen: set[str] = set()
    result: list[str] = []
    for msg in messages:
        text = _extract_text(msg, max_chars=100_000)
        for fid in extract_file_ids(text):
            if fid not in seen:
                seen.add(fid)
                result.append(fid)
    return result


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
    existing_footer_re = re.compile(r"\n*\[LCM File IDs:[^\]]*\]\s*$", re.MULTILINE)
    text = existing_footer_re.sub("", text)

    return text + footer
