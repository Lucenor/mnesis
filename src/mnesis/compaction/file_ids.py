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
- :func:`extract_file_id_paths` and friends — recover which path each ID
  belongs to, so the footer can record ``path (file_<hex>)`` pairs. Without
  them a summary can keep an ID and a path that are no longer associated.

Footer format::

    [LCM File IDs: services/billing/rates.yaml (file_3fa9c2d17b8e4a60), file_9b1e0c44a7d2f381]

A bare ``file_<hex>`` entry means the path was not known. The older footer
(bare IDs only) is still parsed everywhere; ID extraction only looks for
``file_<hex>`` tokens, so both forms yield the same IDs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping

from mnesis.models.message import MessageWithParts, TextPart, ToolPart
from mnesis.models.summary import SummaryNode

# Matches Mnesis file IDs: ``file_`` followed by 8-32 hex characters.
# The pattern is intentionally broad to catch variations across providers.
_FILE_ID_RE = re.compile(r"\bfile_[0-9a-fA-F]{8,32}\b")

# Footer template for the ``[LCM File IDs: ...]`` footer.
_FILE_IDS_FOOTER_TEMPLATE = "\n\n[LCM File IDs: {ids}]"

# A candidate path token: a run of path characters. Whether it is really a path
# is decided by :func:`_is_path` (and ``,`` ``(`` ``)`` ``]`` never occur in one,
# so a token can always be written into the footer grammar).
_PATH_RE = re.compile(r"(?<![\w/.~@-])[\w.~/@-]+")

# File extensions that make a bare name (no ``/``) a path. A bare ``config`` or
# ``json.loads`` is code or prose, not a file.
_PATH_EXTENSIONS = frozenset(
    "py pyi ts tsx js jsx mjs cjs json jsonl yaml yml toml ini cfg conf env md rst txt csv tsv "
    "sql sh bash zsh go rs java kt rb php c h cc cpp hpp cs swift scala html css scss xml lock "
    "log ipynb parquet db sqlite proto tf".split()
)

# What may separate a file ID from the path it names: an opening parenthesis, a
# colon or an equals sign, with at most two spaces around it, three characters in
# all (quotes and backticks around the path do not count). Covers the forms the
# prompts ask for -- ``path (file_x)`` -- and ``file_x: path``. Anything longer or
# with other words or punctuation (``,`` ``->`` ``is``...) is not an adjacency.
_PAIR_GAP_RE = re.compile(r" {0,2}[(:=] {0,2}")
_PAIR_GAP_MAX = 3

# The body of a ``[LCM File IDs: ...]`` footer.
_FOOTER_BODY_RE = re.compile(r"\[LCM File IDs:([^\]]*)\]")

# One footer entry: ``path (file_<hex>)`` or a bare ``file_<hex>``.
_FOOTER_ENTRY_RE = re.compile(
    r"(?:(?P<path>[^,()\[\]]+?)\s*\(\s*)?(?P<id>file_[0-9a-fA-F]{8,32})\b\)?"
)

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


def _clean_path(raw: str) -> str:
    """Normalise a candidate path: strip surrounding punctuation and trailing dots."""
    return raw.strip().strip("`'\"").rstrip(".")


def _is_path(token: str, line: str, end: int) -> bool:
    """Whether the token ending at ``line[end]`` is a file path rather than code/prose.

    A path has a ``/`` or a known file extension. Tokens touching ``@`` (e-mail),
    URLs, calls (``name.ext(``) and bare file IDs are rejected. When unsure, no.
    """
    token = token.rstrip(".")
    if not token or "@" in token or "://" in token or token.startswith("//"):
        return False
    if token.endswith("/") or _FILE_ID_RE.fullmatch(token):
        return False
    if line[end : end + 1] == "(":
        return False
    if "/" in token:
        return True
    name, dot, ext = token.rpartition(".")
    return bool(dot and name and ext.lower() in _PATH_EXTENSIONS)


def _heuristic_pairs(text: str) -> dict[str, str]:
    """Pair file IDs with the path written directly next to them.

    Only adjacent pairs count: the path, a gap of at most three separator
    characters (see ``_PAIR_GAP_RE``) and the ID, in either order. Each path
    names at most one ID. A wrong path is worse than none, so an ID with no
    adjacent path, or on a line where adjacency is ambiguous, stays unpaired.
    """
    pairs: dict[str, str] = {}
    for line in text.splitlines():
        id_matches = list(_FILE_ID_RE.finditer(line))
        if not id_matches:
            continue
        paths = [
            m
            for m in _PATH_RE.finditer(line)
            if _is_path(m.group(), line, m.end()) and not _FILE_ID_RE.fullmatch(m.group())
        ]
        used_paths: set[int] = set()
        for im in id_matches:
            chosen: int | None = None
            # Prefer the path before the ID (``path (file_x)``), then after it.
            for j, pm in enumerate(paths):
                if j in used_paths:
                    continue
                if pm.end() <= im.start() and _adjacent(line[pm.end() : im.start()]):
                    chosen = j
                    break
            if chosen is None:
                for j, pm in enumerate(paths):
                    if j in used_paths:
                        continue
                    if pm.start() >= im.end() and _adjacent(line[im.end() : pm.start()]):
                        chosen = j
                        break
            if chosen is not None:
                used_paths.add(chosen)
                pairs.setdefault(im.group(), _clean_path(paths[chosen].group()))
    return pairs


def _adjacent(gap: str) -> bool:
    """Whether *gap* is a short run of pairing separators (quotes/backticks ignored)."""
    gap = gap.replace("`", "").replace("'", "").replace('"', "")
    return len(gap) <= _PAIR_GAP_MAX and _PAIR_GAP_RE.fullmatch(gap) is not None


def footer_file_id_paths(text: str) -> dict[str, str]:
    """Return the ``{file_id: path}`` pairs recorded in *text*'s footer(s).

    Entries without a path (the older bare-ID footer) are skipped.
    """
    pairs: dict[str, str] = {}
    for body in _FOOTER_BODY_RE.finditer(text):
        for entry in _FOOTER_ENTRY_RE.finditer(body.group(1)):
            path = entry.group("path")
            fid = entry.group("id")
            if path and path.strip() and fid not in pairs:
                pairs[fid] = path.strip()
    return pairs


def extract_file_id_paths(text: str) -> dict[str, str]:
    """
    Map each file ID in *text* to its path, where one can be determined.

    A pair recorded in a ``[LCM File IDs: ...]`` footer wins; otherwise an ID is
    paired with the nearest path-like token on the same line (best effort: IDs
    with no path on their line stay unpaired). The first pairing found for an
    ID wins.

    Args:
        text: Text that may contain file IDs, paths and a footer.

    Returns:
        ``{file_id: path}`` for the IDs whose path is known.
    """
    pairs = footer_file_id_paths(text)
    for fid, path in _heuristic_pairs(strip_file_ids_footer(text)).items():
        pairs.setdefault(fid, path)
    return pairs


def extract_file_id_paths_from_messages(messages: list[MessageWithParts]) -> dict[str, str]:
    """Like :func:`extract_file_id_paths`, over the raw text of *messages* (first pairing wins)."""
    pairs: dict[str, str] = {}
    for msg in messages:
        for fid, path in extract_file_id_paths(message_id_text(msg)).items():
            pairs.setdefault(fid, path)
    return pairs


def collect_file_id_paths_from_nodes(nodes: list[SummaryNode]) -> dict[str, str]:
    """
    Aggregate ``{file_id: path}`` pairs from summary nodes (for condensation).

    Footer pairs (derived from the original messages when a leaf was written)
    take precedence over pairs inferred from a node's prose, which an LLM may
    have mangled. Within each tier the earliest node wins.
    """
    pairs: dict[str, str] = {}
    for node in nodes:
        for fid, path in footer_file_id_paths(node.content).items():
            pairs.setdefault(fid, path)
    for node in nodes:
        for fid, path in _heuristic_pairs(strip_file_ids_footer(node.content)).items():
            pairs.setdefault(fid, path)
    return pairs


def append_file_ids_footer(
    text: str, file_ids: list[str], paths: Mapping[str, str] | None = None
) -> str:
    """
    Append a ``[LCM File IDs: ...]`` footer to *text* when *file_ids* is non-empty.

    If *text* already contains the footer this function is idempotent — it will
    not duplicate the block.  The footer is always placed at the end.

    Args:
        text: The summary text to annotate.
        file_ids: File IDs to include in the footer.
        paths: Optional ``{file_id: path}``; an ID with a path is written as
            ``path (file_<hex>)``, others as the bare ID.

    Returns:
        Annotated text, or the original text unchanged if *file_ids* is empty.
    """
    if not file_ids:
        return text

    paths = paths or {}
    ids_str = ", ".join(f"{paths[fid]} ({fid})" if fid in paths else fid for fid in file_ids)
    footer = _FILE_IDS_FOOTER_TEMPLATE.format(ids=ids_str)

    # Strip any existing footer before appending the authoritative one.
    return strip_file_ids_footer(text) + footer
