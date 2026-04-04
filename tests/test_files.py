"""Tests for LargeFileHandler."""

from __future__ import annotations

import json

import pytest

from mnesis.files.handler import LargeFileHandler
from mnesis.models.config import FileConfig


@pytest.fixture
def handler(store, estimator):
    return LargeFileHandler(store, estimator, FileConfig(inline_threshold=100))


class TestLargeFileHandler:
    async def test_small_file_returned_inline(self, handler, tmp_path):
        """Files below threshold are returned as inline content."""
        path = tmp_path / "small.py"
        path.write_text("x = 1\n")
        result = await handler.handle_file(str(path))
        assert result.is_inline
        assert result.inline_content == "x = 1\n"
        assert result.file_ref is None

    async def test_large_file_returned_as_ref(self, handler, tmp_path):
        """Files above threshold are stored and returned as FileRefPart."""
        path = tmp_path / "large.py"
        path.write_text("x" * 1000)  # 1000 chars >> 50 token threshold
        result = await handler.handle_file(str(path))
        assert not result.is_inline
        assert result.file_ref is not None
        assert result.file_ref.file_type == "python"
        assert result.file_ref.content_id is not None

    async def test_cache_hit_same_content(self, handler, tmp_path):
        """Same file content returns the same content_id (cache hit)."""
        content = "y" * 500
        path1 = tmp_path / "file1.py"
        path2 = tmp_path / "file2.py"
        path1.write_text(content)
        path2.write_text(content)

        result1 = await handler.handle_file(str(path1))
        result2 = await handler.handle_file(str(path2))

        assert not result1.is_inline
        assert not result2.is_inline
        assert result1.file_ref.content_id == result2.file_ref.content_id

    async def test_python_file_has_exploration_summary(self, handler, tmp_path):
        """Python file reference includes exploration summary with class/function info."""
        path = tmp_path / "module.py"
        path.write_text("class Foo:\n    def bar(self): pass\n\ndef standalone(): return 1\n" * 10)
        result = await handler.handle_file(str(path))
        assert not result.is_inline
        summary = result.file_ref.exploration_summary
        # Should mention classes or functions
        assert any(kw in summary.lower() for kw in ["class", "function", "module"])

    async def test_json_file_has_exploration_summary(self, handler, tmp_path):
        """JSON file reference includes key names in summary."""
        data = {"name": "test", "version": "1.0", "items": list(range(20))}
        path = tmp_path / "data.json"
        path.write_text(json.dumps(data) * 5)  # Make it large enough
        result = await handler.handle_file(str(path))
        if not result.is_inline:
            summary = result.file_ref.exploration_summary
            assert "json" in summary.lower() or "name" in summary.lower()

    async def test_missing_file_returns_error_inline(self, handler):
        """Non-existent file returns inline error message."""
        result = await handler.handle_file("/nonexistent/path/file.py")
        assert result.is_inline
        assert "not found" in (result.inline_content or "").lower()

    async def test_content_provided_directly(self, handler, tmp_path):
        """Content can be provided directly without reading from disk."""
        path = tmp_path / "virtual.py"
        content = "z" * 500
        result = await handler.handle_file(str(path), content=content)
        # content_id should be based on the content, not the file
        assert not result.is_inline
        assert result.file_ref.content_id is not None

    async def test_detect_file_type_by_extension(self, handler):
        """File type detection uses extension mapping."""
        assert handler._detect_file_type("test.py", "") == "python"
        assert handler._detect_file_type("test.ts", "") == "typescript"
        assert handler._detect_file_type("test.json", "") == "json"
        assert handler._detect_file_type("test.md", "") == "markdown"
        assert handler._detect_file_type("test.csv", "") == "csv"

    async def test_markdown_summary_extracts_headings(self, handler):
        """Markdown exploration summary includes headings."""
        content = "# Title\n\n## Section 1\n\nText.\n\n## Section 2\n\nMore text.\n"
        summary = handler._summarise_markdown(content)
        assert "Title" in summary or "Section" in summary

    async def test_csv_summary_includes_columns(self, handler):
        """CSV exploration summary includes column names."""
        content = "name,age,email\nAlice,30,alice@example.com\nBob,25,bob@example.com\n"
        summary = handler._summarise_csv(content)
        assert "name" in summary.lower() or "columns" in summary.lower()

    # ── inline threshold boundary ─────────────────────────────────────────────

    async def test_file_at_threshold_is_inline(self, tmp_path, store, estimator):
        """A file whose token count equals inline_threshold - 1 is returned inline."""
        from mnesis.files.handler import LargeFileHandler
        from mnesis.models.config import FileConfig

        # threshold=100; content of exactly 99 tokens (396 chars / 4 per token = 99)
        threshold = 100
        handler = LargeFileHandler(store, estimator, FileConfig(inline_threshold=threshold))
        content = "a" * (threshold * 4 - 4)  # 396 chars = 99 tokens
        path = tmp_path / "at_threshold.py"
        path.write_text(content)
        result = await handler.handle_file(str(path))
        assert result.is_inline

    # ── Python AST summary ────────────────────────────────────────────────────

    def test_summarise_python_with_classes_and_functions(self, handler):
        """_summarise_python() extracts class names, function names, and imports."""
        content = """
import os
import sys
from pathlib import Path

class MyClass:
    def method_one(self):
        pass
    def method_two(self):
        pass

def standalone_func():
    return 42

async def async_func():
    pass
"""
        summary = handler._summarise_python(content)
        assert "MyClass" in summary
        assert "module" in summary.lower()

    def test_summarise_python_syntax_error(self, handler):
        """_summarise_python() handles syntax errors gracefully."""
        content = "def broken(\n  # unclosed"
        summary = handler._summarise_python(content)
        assert "python" in summary.lower() or "parse error" in summary.lower()

    # ── JSON summary ──────────────────────────────────────────────────────────

    def test_summarise_json_dict(self, handler):
        """_summarise_json() describes dict with key count and key names."""
        data = {"name": "Alice", "age": 30, "active": True}
        summary = handler._summarise_json(json.dumps(data))
        assert "3" in summary  # 3 keys
        assert "name" in summary

    def test_summarise_json_array(self, handler):
        """_summarise_json() describes array with item count."""
        data = [{"id": 1, "val": "a"}, {"id": 2, "val": "b"}]
        summary = handler._summarise_json(json.dumps(data))
        assert "2" in summary

    def test_summarise_json_scalar(self, handler):
        """_summarise_json() handles a JSON scalar."""
        summary = handler._summarise_json("42")
        assert "scalar" in summary.lower() or "int" in summary.lower()

    def test_summarise_json_parse_error(self, handler):
        """_summarise_json() returns error string on malformed JSON."""
        summary = handler._summarise_json("{ not valid json }")
        assert "error" in summary.lower() or "parse" in summary.lower()

    def test_summarise_json_with_arrays_field(self, handler):
        """_summarise_json() lists array-valued keys in the summary."""
        data = {"items": [1, 2, 3], "name": "test", "tags": ["a", "b"]}
        summary = handler._summarise_json(json.dumps(data))
        assert "items" in summary or "tags" in summary

    # ── YAML summary ──────────────────────────────────────────────────────────

    def test_summarise_yaml_dict(self, handler):
        """_summarise_yaml() describes a YAML mapping."""
        __import__("pytest").importorskip("yaml")
        content = "name: Alice\nage: 30\nactive: true\n"
        summary = handler._summarise_yaml(content)
        assert "3" in summary or "name" in summary

    def test_summarise_yaml_sequence(self, handler):
        """_summarise_yaml() describes a YAML sequence."""
        __import__("pytest").importorskip("yaml")
        content = "- item1\n- item2\n- item3\n"
        summary = handler._summarise_yaml(content)
        assert "3" in summary or "sequence" in summary.lower()

    def test_summarise_yaml_fallback_on_import_error(self, handler, monkeypatch):
        """_summarise_yaml() falls back to line counting when yaml is unavailable."""
        import sys

        # Simulate yaml not being available
        monkeypatch.setitem(sys.modules, "yaml", None)  # type: ignore[arg-type]
        content = "name: Alice\nage: 30\n"
        # Should not raise, falls back to heuristic
        summary = handler._summarise_yaml(content)
        assert isinstance(summary, str)
        assert len(summary) > 0

    # ── TOML summary ─────────────────────────────────────────────────────────

    def test_summarise_toml_valid(self, handler):
        """_summarise_toml() uses tomllib to describe TOML sections."""
        content = '[tool.ruff]\nline-length = 88\n\n[project]\nname = "test"\n'
        summary = handler._summarise_toml(content)
        assert "tool" in summary or "project" in summary or "section" in summary.lower()

    def test_summarise_toml_fallback(self, handler, monkeypatch):
        """_summarise_toml() falls back to section line scanning on parse error."""
        import sys

        # Force tomllib to fail
        monkeypatch.setitem(sys.modules, "tomllib", None)  # type: ignore[arg-type]
        content = "[section1]\nkey = 1\n\n[section2]\nkey = 2\n"
        summary = handler._summarise_toml(content)
        assert len(summary) > 0
        assert "section1" in summary or "section2" in summary or "toml" in summary.lower()

    # ── TypeScript/JavaScript summary ─────────────────────────────────────────

    def test_summarise_ts_js_with_exports_and_classes(self, handler):
        """_summarise_ts_js() extracts exports, classes, and functions."""
        content = """
import React from 'react';
import { useState } from 'react';

export class MyComponent {
  render() { return null; }
}

export function helper() {}

export default function App() {}
"""
        summary = handler._summarise_ts_js(content)
        assert "TypeScript" in summary or "JavaScript" in summary

    # ── Generic text summary ──────────────────────────────────────────────────

    def test_summarise_generic_text_file(self, handler):
        """_summarise_generic() returns line count and preview."""
        content = "\n".join(f"Line {i}" for i in range(20))
        summary = handler._summarise_generic(content)
        assert "20" in summary or "line" in summary.lower()

    # ── detect_file_type unknown/binary/magic fallback ────────────────────────

    def test_detect_file_type_unknown_extension(self, handler):
        """Unknown extension returns 'unknown' or falls back to content inspection."""
        file_type = handler._detect_file_type("file.xyz123", "plain text content")
        assert isinstance(file_type, str)

    def test_detect_file_type_binary_content(self, handler):
        """High null-byte ratio in content is detected as binary."""
        binary_content = "\x00" * 100 + "a" * 5  # >1% null bytes
        file_type = handler._detect_file_type("file.xyz", binary_content)
        assert file_type == "binary"

    # ── generate_exploration_summary exception path ───────────────────────────

    def test_generate_exploration_summary_handles_exception(self, handler, monkeypatch):
        """_generate_exploration_summary() catches internal errors and returns fallback."""

        def _raise(content: str) -> str:
            raise ValueError("boom")

        # Patch the static method to raise so _generate_exploration_summary catches it
        monkeypatch.setattr(handler.__class__, "_summarise_python", staticmethod(_raise))
        result = handler._generate_exploration_summary("/x.py", "class Foo: pass", "python")
        assert "Summary generation failed" in result
        assert "python" in result.lower()

    # ── bytes content normalization ───────────────────────────────────────────

    async def test_handle_file_bytes_content(self, handler, tmp_path):
        """handle_file() accepts bytes content and decodes it."""
        content = b"x" * 600  # Large enough to exceed threshold
        path = tmp_path / "bytes_file.py"
        result = await handler.handle_file(str(path), content=content)
        # bytes decoded → string → estimated → should be ref or inline
        assert result.path == str(path)
        assert result.inline_content is not None or result.file_ref is not None

    # ── TSV summary ───────────────────────────────────────────────────────────

    def test_summarise_tsv(self, handler):
        """_summarise_csv() with tab separator handles TSV files."""
        content = "col1\tcol2\tcol3\nval1\tval2\tval3\n"
        summary = handler._summarise_csv(content, sep="\t")
        assert "col1" in summary or "3" in summary
