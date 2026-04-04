"""Tests for TokenEstimator — targeting 90%+ coverage of tokens/estimator.py."""

from __future__ import annotations

import hashlib

import pytest

from mnesis.models.message import (
    CompactionMarkerPart,
    FileRefPart,
    Message,
    MessageWithParts,
    TextPart,
    ToolPart,
    ToolStatus,
)
from mnesis.tokens.estimator import TokenEstimator


class TestTokenEstimatorHeuristic:
    """Tests for heuristic-only estimation paths."""

    def test_estimate_empty_string_returns_zero(self):
        """estimate('') == 0 regardless of model."""
        est = TokenEstimator(heuristic_only=True)
        assert est.estimate("") == 0

    def test_estimate_non_empty_heuristic_minimum_one(self):
        """Short strings return at least 1 token."""
        est = TokenEstimator(heuristic_only=True)
        assert est.estimate("x") >= 1

    def test_heuristic_four_chars_per_token(self):
        """Default heuristic: 4 chars per token, min 1."""
        est = TokenEstimator(heuristic_only=True)
        text = "a" * 40
        assert est.estimate(text) == 10  # 40 // 4

    def test_estimate_no_model_uses_heuristic(self):
        """estimate() with model=None falls back to heuristic."""
        est = TokenEstimator()
        text = "hello world test string here"
        # Should return a positive number
        assert est.estimate(text) > 0

    def test_heuristic_only_flag_bypasses_tiktoken(self, monkeypatch):
        """heuristic_only=True skips tiktoken even if a model is provided."""
        est = TokenEstimator(heuristic_only=True)

        from mnesis.models.config import ModelInfo

        model = ModelInfo.from_model_string("openai/gpt-4o")
        text = "hello world"
        result = est.estimate(text, model)
        # Heuristic: 11 chars // 4 = 2
        assert result == max(1, len(text) // 4)


class TestTokenEstimatorClaudeHeuristic:
    """Tests for the claude_heuristic encoding path (len // 3)."""

    def test_claude_heuristic_encoding(self):
        """Claude models use len//3 heuristic."""
        from mnesis.models.config import ModelInfo

        model = ModelInfo.from_model_string("anthropic/claude-opus-4-6")
        est = TokenEstimator()
        text = "a" * 30
        result = est.estimate(text, model)
        # claude_heuristic: 30 // 3 = 10
        assert result == 10

    def test_claude_heuristic_minimum_one(self):
        """claude_heuristic returns at least 1 for single char."""
        from mnesis.models.config import ModelInfo

        model = ModelInfo.from_model_string("anthropic/claude-opus-4-6")
        est = TokenEstimator()
        # 1 char // 3 == 0, so max(1, ...) = 1
        assert est.estimate("x", model) >= 1


class TestTokenEstimatorTiktoken:
    """Tests for tiktoken-based estimation paths."""

    def test_cl100k_base_encoding_uses_tiktoken(self):
        """cl100k_base encoding calls tiktoken for real token counts."""
        pytest.importorskip("tiktoken")

        from mnesis.models.config import ModelInfo

        model = ModelInfo.from_model_string("openai/gpt-4")
        est = TokenEstimator()
        text = "Hello, world!"
        result = est.estimate(text, model)
        # tiktoken gives exact count; just verify it's positive and reasonable
        assert result > 0
        assert result < len(text)  # tokens < chars for normal text

    def test_o200k_base_encoding_uses_tiktoken(self):
        """o200k_base encoding (gpt-4o) calls tiktoken."""
        pytest.importorskip("tiktoken")

        from mnesis.models.config import ModelInfo

        model = ModelInfo.from_model_string("openai/gpt-4o")
        est = TokenEstimator()
        text = "The quick brown fox jumps over the lazy dog."
        result = est.estimate(text, model)
        assert result > 0

    def test_tiktoken_encoder_cached_after_first_call(self):
        """Encoder object is cached in _encoder_cache after first use."""
        pytest.importorskip("tiktoken")

        from mnesis.models.config import ModelInfo

        model = ModelInfo.from_model_string("openai/gpt-4")
        est = TokenEstimator()
        est.estimate("first call", model)
        assert "cl100k_base" in est._encoder_cache

        # Second call reuses cached encoder — cache size does not grow
        est.estimate("second call", model)
        assert len(est._encoder_cache) == 1

    def test_tiktoken_exception_falls_back_to_heuristic(self, monkeypatch):
        """When tiktoken raises, estimate() falls back to heuristic silently."""
        from mnesis.models.config import ModelInfo

        model = ModelInfo.from_model_string("openai/gpt-4")
        est = TokenEstimator()

        # Patch _tiktoken_estimate to raise
        def _raise(text: str, enc: str) -> int:
            raise RuntimeError("tiktoken unavailable")

        monkeypatch.setattr(est, "_tiktoken_estimate", _raise)
        text = "hello world"
        result = est.estimate(text, model)
        # Falls back to _heuristic: len // 4
        assert result == max(1, len(text) // 4)

    def test_tiktoken_unavailable_falls_back_gracefully(self, monkeypatch):
        """estimate() uses heuristic when tiktoken import fails at call time."""
        from mnesis.models.config import ModelInfo

        model = ModelInfo.from_model_string("openai/gpt-4")
        est = TokenEstimator()

        def _raise_import(text: str, enc: str) -> int:
            raise ImportError("No module named 'tiktoken'")

        monkeypatch.setattr(est, "_tiktoken_estimate", _raise_import)
        text = "a" * 20
        result = est.estimate(text, model)
        assert result >= 1


class TestTokenEstimatorCached:
    """Tests for estimate_cached() — content-addressed caching."""

    def test_estimate_cached_returns_cached_value_on_second_call(self):
        """Second call with same cache_key returns cached result without re-computing."""
        est = TokenEstimator(heuristic_only=True)
        text = "hello world"
        key = "my_key"

        est.estimate_cached(text, key)
        # Modify internal cache to a different value to detect if it re-computes
        est._count_cache[key] = 9999
        second = est.estimate_cached(text, key)
        assert second == 9999  # returns cached, not recomputed

    def test_estimate_cached_stores_result(self):
        """estimate_cached() populates _count_cache for future lookups."""
        est = TokenEstimator(heuristic_only=True)
        text = "cache this string"
        key = hashlib.sha256(text.encode()).hexdigest()
        est.estimate_cached(text, key)
        assert key in est._count_cache

    def test_content_hash_is_stable(self):
        """content_hash() returns consistent SHA-256 for the same input."""
        h1 = TokenEstimator.content_hash("hello")
        h2 = TokenEstimator.content_hash("hello")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest is 64 chars

    def test_content_hash_differs_for_different_inputs(self):
        """content_hash() produces different digests for different inputs."""
        assert TokenEstimator.content_hash("a") != TokenEstimator.content_hash("b")


class TestEstimateMessage:
    """Tests for estimate_message() with all part types."""

    def _make_mwp(self, session_id: str, msg_id: str, parts: list) -> MessageWithParts:
        msg = Message(id=msg_id, session_id=session_id, role="assistant")
        return MessageWithParts(message=msg, parts=parts)

    def test_estimate_message_text_part(self):
        """estimate_message() counts tokens in a TextPart."""
        est = TokenEstimator(heuristic_only=True)
        text = "a" * 40  # 40 chars // 4 = 10 tokens
        mwp = self._make_mwp("sess_1", "msg_1", [TextPart(text=text)])
        result = est.estimate_message(mwp)
        # 4 overhead + 10 from text part
        assert result == 14

    def test_estimate_message_tool_part_active(self):
        """estimate_message() includes full tool content for non-pruned ToolPart."""
        est = TokenEstimator(heuristic_only=True)
        tool = ToolPart(
            tool_name="my_tool",
            tool_call_id="call_001",
            input={"key": "value"},
            output="output text here",
            status=ToolStatus(state="completed"),
        )
        mwp = self._make_mwp("sess_1", "msg_2", [tool])
        result = est.estimate_message(mwp)
        assert result > 4  # more than just overhead

    def test_estimate_message_tool_part_compacted(self):
        """estimate_message() uses compact tombstone string for pruned ToolPart."""
        import time as _time

        est = TokenEstimator(heuristic_only=True)
        now_ms = int(_time.time() * 1000)
        tool = ToolPart(
            tool_name="old_tool",
            tool_call_id="call_002",
            input={},
            output="x" * 10000,  # large output
            status=ToolStatus(state="completed", compacted_at=now_ms),
        )
        mwp = self._make_mwp("sess_1", "msg_3", [tool])
        # Compacted version should be much smaller than the full output
        result_compacted = est.estimate_message(mwp)

        tool_full = ToolPart(
            tool_name="old_tool",
            tool_call_id="call_002",
            input={},
            output="x" * 10000,
            status=ToolStatus(state="completed"),
        )
        mwp_full = self._make_mwp("sess_1", "msg_3b", [tool_full])
        result_full = est.estimate_message(mwp_full)

        assert result_compacted < result_full

    def test_estimate_message_file_ref_part(self):
        """estimate_message() handles FileRefPart via estimate_cached()."""
        est = TokenEstimator(heuristic_only=True)
        fref = FileRefPart(
            content_id="abc123def456",
            path="/some/file.py",
            file_type="python",
            token_count=500,
            exploration_summary="Python module with 3 classes.",
        )
        mwp = self._make_mwp("sess_1", "msg_4", [fref])
        result = est.estimate_message(mwp)
        assert result > 4  # overhead + file ref block tokens

    def test_estimate_message_file_ref_uses_cache(self):
        """FileRefPart tokens are cached on second estimate_message() call."""
        est = TokenEstimator(heuristic_only=True)
        fref = FileRefPart(
            content_id="abc123",
            path="/file.py",
            file_type="python",
            token_count=200,
            exploration_summary="Short summary.",
        )
        mwp = self._make_mwp("sess_1", "msg_5", [fref])
        first = est.estimate_message(mwp)
        # Cache should now be populated
        assert len(est._count_cache) > 0
        second = est.estimate_message(mwp)
        assert first == second

    def test_estimate_message_compaction_marker_part(self):
        """estimate_message() handles CompactionMarkerPart via else branch."""
        est = TokenEstimator(heuristic_only=True)
        marker = CompactionMarkerPart(
            summary_node_id="node_test_001",
            compacted_message_count=2,
            compacted_token_count=100,
        )
        mwp = self._make_mwp("sess_1", "msg_6", [marker])
        result = est.estimate_message(mwp)
        assert result >= 4  # at least overhead

    def test_estimate_message_tool_part_with_error_message(self):
        """estimate_message() includes error_message in token count for errored ToolPart."""
        est = TokenEstimator(heuristic_only=True)
        tool = ToolPart(
            tool_name="err_tool",
            tool_call_id="call_003",
            input={},
            error_message="Something went terribly wrong here",
            status=ToolStatus(state="error"),
        )
        mwp = self._make_mwp("sess_1", "msg_7", [tool])
        result = est.estimate_message(mwp)
        assert result > 4
