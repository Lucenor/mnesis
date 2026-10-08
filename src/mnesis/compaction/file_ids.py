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
    "log ipynb parquet db sqlite proto tf jsonc markdown gz tgz zip tar bz2 xz".split()
)

# What may separate a file ID from the path it names: an opening parenthesis, a
# colon or an equals sign, with at most two spaces around it, or a spaced hyphen
# (``- README.md`` list items); three characters in all (quotes and backticks
# around the path do not count). Covers the forms the prompts ask for --
# ``path (file_x)`` -- and ``file_x: path``. Anything longer or
# with other words or punctuation (``,`` ``->`` ``is``...) is not an adjacency.
_PAIR_GAP_RE = re.compile(r"(?: {0,2}[(:=] {0,2}| - )")
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


# Names that look like files (``Node.js``) but are products/frameworks.
_NOT_FILES = frozenset(
    "node.js next.js vue.js react.js nuxt.js express.js angular.js d3.js three.js ember.js "
    "backbone.js nest.js chart.js socket.js deno.js bun.js".split()
)
# A path that starts like this is a path even without a file extension.
_PATH_PREFIXES = ("/", "./", "../", "~/")
# Well-known files with no extension.
_KNOWN_FILENAMES = frozenset(
    "Dockerfile Makefile Procfile Gemfile Rakefile Vagrantfile Pipfile Brewfile LICENSE README "
    "CHANGELOG CODEOWNERS .env .gitignore .gitattributes .dockerignore .editorconfig .npmrc "
    ".nvmrc .bashrc .zshrc .prettierrc .eslintrc".split()
)
# Lowercase only: ``.NET`` is a framework, ``.env`` a dotfile.
_DOTFILE_RE = re.compile(r"\.[a-z][a-z0-9._-]*")


def _has_known_extension(name: str) -> bool:
    stem, dot, ext = name.rpartition(".")
    return bool(dot and stem and ext.lower() in _PATH_EXTENSIONS)


def _is_path(token: str, line: str, end: int) -> bool:
    """Whether the token ending at ``line[end]`` is a file path rather than code/prose.

    A path is a token whose last segment has a known file extension
    (``src/a.py``, ``README.md``, ``app.tar.gz``), or one with an explicit path prefix
    (``/``, ``./``, ``../``, ``~/``) and a non-empty remainder. A bare ``/`` is
    not enough: ``N/A``, ``TCP/IP``, ``and/or``, ``input/output`` and ``3/4`` are
    prose, and a directory-looking ``a/b/c`` is too ambiguous to trust. Tokens
    touching ``@`` (e-mail), URLs, calls (``name.ext(``), framework names
    (``Node.js``), a purely numeric last segment and bare file IDs are rejected.
    Well-known extensionless names (``Dockerfile``, ``Makefile``, ``.env``...) and
    dotfiles inside a directory (``config/.env``) count. When unsure, no.
    """
    raw = token
    token = token.rstrip(".")
    if not token or "@" in token or "://" in token or token.startswith("//"):
        return False
    if token.endswith("/") or _FILE_ID_RE.fullmatch(token):
        return False
    if line[end : end + 1] == "(":
        return False
    segments = [seg for seg in token.split("/") if seg]
    # A purely numeric last segment is a ratio/date (``3/4``), not a file; numeric
    # directories (``logs/2024/01/app.log``) are fine.
    if not segments or segments[-1].isdigit():
        return False
    if token.lower() in _NOT_FILES:
        return False
    last = segments[-1]
    if _has_known_extension(last):
        return True
    if last in _KNOWN_FILENAMES:
        # A bare ``README`` / ``Makefile`` is prose unless it is the whole value
        # (``file_x (README first)`` is not a pairing). With a directory it is a path.
        if "/" in token:
            return True
        rest = line[end - (len(raw) - len(token)) :]
        return (
            rest == ""
            or rest[0] in ")],;`'\""
            or re.fullmatch(r"[.!?]?\s*", rest) is not None
            or re.match(r"\s*(?:[(:=]|-\s)\s*file_", rest) is not None  # ``README (file_x)``
        )
    if "/" in token and _DOTFILE_RE.fullmatch(last):
        return True  # ``config/.env``, ``~/.zshrc``
    return token.startswith(_PATH_PREFIXES) and any(c.isalpha() for c in token)


def _heuristic_pairs(text: str) -> dict[str, str]:
    """Pair file IDs with the path written directly next to them.

    Only unambiguous adjacent pairs count: the path, a gap of at most three
    separator characters (see ``_PAIR_GAP_RE``) and the ID, in either order. A
    pair is made only when the ID is adjacent to exactly one path *and* that
    path to exactly one ID; ``file_A: src/a.py (file_B)`` pairs neither. A wrong
    path is worse than none, so an ID with no adjacent path, or any ambiguity,
    stays unpaired.
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
        adjacent: list[tuple[int, int]] = []  # (id index, path index)
        for i, im in enumerate(id_matches):
            for j, pm in enumerate(paths):
                before = pm.end() <= im.start() and _adjacent(line[pm.end() : im.start()])
                after = pm.start() >= im.end() and _adjacent(line[im.end() : pm.start()])
                if before or after:
                    adjacent.append((i, j))
        for i, j in adjacent:
            if (
                sum(1 for a, _ in adjacent if a == i) == 1
                and sum(1 for _, b in adjacent if b == j) == 1
            ):
                pairs.setdefault(id_matches[i].group(), _clean_path(paths[j].group()))
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
    paired with a path written directly next to it (at most three separator
    characters between them, e.g. ``path (file_x)`` or ``file_x: path``), and
    only when that is unambiguous: an ID with no adjacent path, or sharing its
    path with another ID, stays unpaired. The first pairing found for an ID wins.

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
