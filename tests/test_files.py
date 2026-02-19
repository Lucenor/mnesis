"""Tests for LargeFileHandler."""

from __future__ import annotations

import json
import os
import tempfile

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
        path.write_text(
            "class Foo:\n    def bar(self): pass\n\ndef standalone(): return 1\n" * 10
        )
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
