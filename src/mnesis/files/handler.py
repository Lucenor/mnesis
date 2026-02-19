"""Content-addressed large file handler with exploration summaries."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import structlog

from mnesis.models.config import FileConfig
from mnesis.models.message import FileRefPart
from mnesis.models.summary import FileReference
from mnesis.store.immutable import ImmutableStore
from mnesis.tokens.estimator import TokenEstimator

FileType = Literal[
    "python",
    "typescript",
    "javascript",
    "rust",
    "go",
    "java",
    "json",
    "yaml",
    "toml",
    "markdown",
    "text",
    "csv",
    "tsv",
    "binary",
    "unknown",
]

_EXTENSION_MAP: dict[str, FileType] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".json": "json",
    ".jsonc": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".rst": "text",
    ".csv": "csv",
    ".tsv": "tsv",
}


@dataclass
class FileHandleResult:
    """Result of processing a file through the LargeFileHandler."""

    path: str
    inline_content: str | None = None
    """Present when the file fits within the inline threshold."""
    file_ref: FileRefPart | None = None
    """Present when the file exceeds the inline threshold."""

    @property
    def is_inline(self) -> bool:
        return self.inline_content is not None


class LargeFileHandler:
    """
    Token-gated file inclusion with content-addressed external storage.

    Files below ``FileConfig.inline_threshold`` are included verbatim.
    Files at or above the threshold are stored in the ``file_references``
    table and represented in context as ``FileRefPart`` objects.

    Content addressing ensures:
    - Identical file content is summarized only once (cached by SHA-256 hash).
    - Changed files get a new hash and trigger a fresh exploration summary.
    - Large file content is never stored directly in the ImmutableStore.

    Example::

        handler = LargeFileHandler(store, estimator, FileConfig())
        result = await handler.handle_file("/path/to/large.py")
        if result.is_inline:
            print(result.inline_content)
        else:
            print(result.file_ref.exploration_summary)
    """

    def __init__(
        self,
        store: ImmutableStore,
        estimator: TokenEstimator,
        config: FileConfig,
    ) -> None:
        self._store = store
        self._estimator = estimator
        self._config = config
        self._logger = structlog.get_logger("mnesis.files")

    async def handle_file(
        self,
        path: str,
        *,
        content: str | bytes | None = None,
    ) -> FileHandleResult:
        """
        Process a file for inclusion in context.

        Args:
            path: File path (used for display and type detection).
            content: Pre-read content. If None, reads from disk.

        Returns:
            FileHandleResult with either inline content or a FileRefPart.
        """
        # Read content if not provided
        if content is None:
            file_path = Path(path)
            if not file_path.exists():  # noqa: ASYNC240
                return FileHandleResult(
                    path=path,
                    inline_content=f"[File not found: {path}]",
                )
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")  # noqa: ASYNC240
            except Exception as exc:
                return FileHandleResult(
                    path=path,
                    inline_content=f"[Error reading {path}: {exc}]",
                )

        # Normalize to string
        if isinstance(content, bytes):
            content_str = content.decode("utf-8", errors="replace")
        else:
            content_str = content

        # Estimate tokens
        token_count = self._estimator.estimate(content_str)

        if token_count < self._config.inline_threshold:
            return FileHandleResult(path=path, inline_content=content_str)

        # Compute content hash for deduplication
        content_id = hashlib.sha256(content_str.encode("utf-8")).hexdigest()

        # Check cache
        existing = await self._store.get_file_reference(content_id)
        if existing is not None:
            self._logger.debug("file_cache_hit", path=path, content_id=content_id[:12])
            return FileHandleResult(
                path=path,
                file_ref=FileRefPart(
                    content_id=existing.content_id,
                    path=path,
                    file_type=existing.file_type,
                    token_count=existing.token_count,
                    exploration_summary=existing.exploration_summary,
                ),
            )

        # Generate exploration summary
        file_type = self._detect_file_type(path, content_str)
        summary = self._generate_exploration_summary(path, content_str, file_type)

        # Store reference
        ref = FileReference(
            content_id=content_id,
            path=path,
            file_type=file_type,
            token_count=token_count,
            exploration_summary=summary,
        )
        await self._store.store_file_reference(ref)

        self._logger.info(
            "file_reference_created",
            path=path,
            file_type=file_type,
            token_count=token_count,
            content_id=content_id[:12],
        )

        return FileHandleResult(
            path=path,
            file_ref=FileRefPart(
                content_id=content_id,
                path=path,
                file_type=file_type,
                token_count=token_count,
                exploration_summary=summary,
            ),
        )

    def _detect_file_type(self, path: str, content: str) -> FileType:
        """
        Detect file type from extension, falling back to content inspection.

        Args:
            path: File path for extension-based detection.
            content: File content for fallback detection.

        Returns:
            FileType string.
        """
        ext = Path(path).suffix.lower()
        if ext in _EXTENSION_MAP:
            return _EXTENSION_MAP[ext]

        # Binary detection heuristic
        null_count = content.count("\x00")
        if null_count > len(content) * 0.01:
            return "binary"

        # Try python-magic for unknown types
        try:
            import magic

            mime = magic.from_buffer(content[:2048].encode(), mime=True)
            if "python" in mime:
                return "python"
            if "javascript" in mime:
                return "javascript"
            if "json" in mime:
                return "json"
            if "text" in mime:
                return "text"
            if "application/octet-stream" in mime:
                return "binary"
        except Exception:
            pass

        return "unknown"

    def _generate_exploration_summary(
        self,
        path: str,
        content: str,
        file_type: FileType,
    ) -> str:
        """
        Generate a compact structural description of a file.

        Uses language-specific parsers for code files and heuristics for data files.

        Args:
            path: File path for display in the summary.
            content: File content to analyze.
            file_type: Detected file type.

        Returns:
            A multi-line summary string.
        """
        try:
            if file_type == "python":
                return self._summarise_python(content)
            elif file_type in ("json",):
                return self._summarise_json(content)
            elif file_type in ("yaml",):
                return self._summarise_yaml(content)
            elif file_type == "toml":
                return self._summarise_toml(content)
            elif file_type == "markdown":
                return self._summarise_markdown(content)
            elif file_type in ("csv", "tsv"):
                return self._summarise_csv(content, sep="\t" if file_type == "tsv" else ",")
            elif file_type in ("typescript", "javascript"):
                return self._summarise_ts_js(content)
            else:
                return self._summarise_generic(content)
        except Exception as exc:
            return f"[Summary generation failed: {exc}]\nFile type: {file_type}"

    @staticmethod
    def _summarise_python(content: str) -> str:
        """Use Python AST to extract top-level definitions."""
        lines: list[str] = []
        try:
            tree = ast.parse(content)
            classes: list[str] = []
            functions: list[str] = []
            imports: list[str] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]
                    classes.append(f"{node.name} ({len(methods)} methods)")
                elif isinstance(node, ast.FunctionDef) and not isinstance(
                    node, ast.AsyncFunctionDef
                ):
                    functions.append(node.name)
                elif isinstance(node, ast.AsyncFunctionDef):
                    functions.append(f"async {node.name}")
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.ImportFrom):
                        imports.append(f"{node.module}")
                    else:
                        for alias in node.names:
                            imports.append(alias.name)

            lines.append(
                f"Python module with {len(classes)} class(es), {len(functions)} function(s)."
            )
            if classes:
                lines.append(f"Classes: {', '.join(classes[:10])}")
            if functions:
                top_fns = [f for f in functions if not f.startswith("_")][:10]
                lines.append(f"Key functions: {', '.join(top_fns)}")
            if imports:
                lines.append(f"Imports: {', '.join(sorted(set(imports))[:15])}")
        except SyntaxError as exc:
            lines.append(f"Python file (parse error: {exc})")

        return "\n".join(lines)

    @staticmethod
    def _summarise_json(content: str) -> str:
        """Summarise JSON structure."""
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                keys = list(data.keys())[:20]
                arrays = [k for k, v in data.items() if isinstance(v, list)]
                return (
                    f"JSON object with {len(data)} keys.\n"
                    f"Top-level keys: {', '.join(keys)}"
                    + (f"\nArrays: {', '.join(arrays[:10])}" if arrays else "")
                )
            elif isinstance(data, list):
                sample_keys = (
                    list(data[0].keys())[:10] if data and isinstance(data[0], dict) else []
                )
                return (
                    f"JSON array with {len(data)} items."
                    + (f"\nItem keys: {', '.join(sample_keys)}" if sample_keys else "")
                )
            else:
                return f"JSON scalar: {type(data).__name__}"
        except json.JSONDecodeError as exc:
            return f"JSON file (parse error: {exc})"

    @staticmethod
    def _summarise_yaml(content: str) -> str:
        """Summarise YAML structure."""
        try:
            import yaml

            data = yaml.safe_load(content)
            if isinstance(data, dict):
                keys = list(data.keys())[:20]
                return (
                    f"YAML document with {len(data)} keys.\n"
                    f"Keys: {', '.join(str(k) for k in keys)}"
                )
            elif isinstance(data, list):
                return f"YAML sequence with {len(data)} items."
            else:
                return f"YAML scalar: {type(data).__name__}"
        except Exception:
            # Fallback: count keys heuristically
            key_lines = [
                ln for ln in content.split("\n") if ln and not ln.startswith(" ") and ":" in ln
            ]
            return f"YAML document. Top-level keys: {len(key_lines)}"

    @staticmethod
    def _summarise_toml(content: str) -> str:
        """Summarise TOML structure."""
        try:
            import tomllib  # Python 3.11+

            data = tomllib.loads(content)
            keys = list(data.keys())[:20]
            return f"TOML document with {len(data)} sections.\nSections: {', '.join(keys)}"
        except Exception:
            sections = [
                ln.strip("[]").strip() for ln in content.split("\n") if ln.startswith("[")
            ]
            return f"TOML document. Sections: {', '.join(sections[:15])}"

    @staticmethod
    def _summarise_markdown(content: str) -> str:
        """Extract headings from Markdown."""
        headings: list[str] = []
        for line in content.split("\n"):
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("#").strip()
                headings.append(f"{'  ' * (level - 1)}{text}")
        word_count = len(content.split())
        summary = f"Markdown document (~{word_count} words)."
        if headings:
            summary += "\nSections:\n" + "\n".join(headings[:20])
        return summary

    @staticmethod
    def _summarise_csv(content: str, sep: str = ",") -> str:
        """Summarise CSV/TSV structure."""
        lines = content.strip().split("\n")
        if not lines:
            return "Empty CSV file."
        header = lines[0].split(sep)
        row_count = len(lines) - 1
        return (
            f"Table with {row_count} rows, {len(header)} columns.\n"
            f"Columns: {', '.join(h.strip() for h in header[:20])}"
        )

    @staticmethod
    def _summarise_ts_js(content: str) -> str:
        """Summarise TypeScript/JavaScript via regex heuristics."""
        import re

        exports = re.findall(
            r"^export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)",
            content,
            re.MULTILINE,
        )
        classes = re.findall(r"^(?:export\s+)?class\s+(\w+)", content, re.MULTILINE)
        functions = re.findall(
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)", content, re.MULTILINE
        )
        imports = re.findall(
            r"^import\s+.*from\s+['\"]([^'\"]+)['\"]", content, re.MULTILINE
        )

        lines = ["TypeScript/JavaScript module."]
        if classes:
            lines.append(f"Classes: {', '.join(classes[:10])}")
        if functions:
            lines.append(f"Functions: {', '.join(functions[:10])}")
        if exports:
            lines.append(f"Exports: {', '.join(exports[:10])}")
        if imports:
            lines.append(f"Imports from: {', '.join(sorted(set(imports))[:10])}")
        return "\n".join(lines)

    @staticmethod
    def _summarise_generic(content: str) -> str:
        """Generic fallback: line count + first few lines."""
        lines = content.split("\n")
        preview = "\n".join(lines[:5])
        return f"Text file with {len(lines)} lines.\nPreview:\n{preview}"
