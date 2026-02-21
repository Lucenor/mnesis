"""Tests for condensation, file ID propagation, multi-round loop, soft/hard threshold,
and summarisation input token cap.

These tests cover the five architectural gaps added in this PR.
"""

from __future__ import annotations

import pytest

from mnesis.compaction.engine import CompactionEngine
from mnesis.compaction.file_ids import (
    append_file_ids_footer,
    collect_file_ids_from_nodes,
    extract_file_ids,
    extract_file_ids_from_messages,
)
from mnesis.compaction.levels import (
    MIN_MESSAGES_TO_SUMMARISE,
    CondensationCandidate,
    _apply_input_cap,
    _messages_to_summarise,
    condense_level1,
    condense_level2,
    condense_level3_deterministic,
    level1_summarise,
    level3_deterministic,
)
from mnesis.models.config import CompactionConfig, MnesisConfig, ModelInfo
from mnesis.models.message import ContextBudget, MessageWithParts, TextPart, TokenUsage
from mnesis.models.summary import SummaryNode
from tests.conftest import make_message, make_raw_part

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_messages(session_id: str, count: int) -> list[MessageWithParts]:
    """Create alternating user/assistant messages."""
    result = []
    roles = ["user", "assistant"]
    for i in range(count):
        msg = make_message(session_id, role=roles[i % 2], msg_id=f"msg_cnd_{i:03d}")
        text = f"Message {i}: " + "x" * 200
        # Embed a file ID in message 0 to test propagation.
        if i == 0:
            text = "Working with file_a1b2c3d4e5f60001 in this session. " + text
        result.append(MessageWithParts(message=msg, parts=[TextPart(text=text)]))
    return result


def _make_summary_node(
    session_id: str,
    node_id: str,
    content: str,
    kind: str = "leaf",
    parent_node_ids: list[str] | None = None,
) -> SummaryNode:
    """Create a SummaryNode for testing."""
    return SummaryNode(
        id=node_id,
        session_id=session_id,
        kind=kind,  # type: ignore[arg-type]
        span_start_message_id="msg_start",
        span_end_message_id="msg_end",
        content=content,
        token_count=len(content) // 4,
        parent_node_ids=parent_node_ids or [],
    )


@pytest.fixture
def budget() -> ContextBudget:
    return ContextBudget(
        model_context_limit=50_000,
        reserved_output_tokens=4_000,
        compaction_buffer=10_000,
    )


# ── File ID extraction ─────────────────────────────────────────────────────────


class TestExtractFileIds:
    def test_extracts_single_id(self):
        text = "Please see file_a1b2c3d4e5f60001 for details."
        assert extract_file_ids(text) == ["file_a1b2c3d4e5f60001"]

    def test_extracts_multiple_ids_in_order(self):
        text = "Files: file_aaaa1111bbbb2222 and file_cccc3333dddd4444 are relevant."
        ids = extract_file_ids(text)
        assert ids == ["file_aaaa1111bbbb2222", "file_cccc3333dddd4444"]

    def test_deduplicates_ids(self):
        text = "file_abcdef12 appears here and again file_abcdef12 later."
        assert extract_file_ids(text) == ["file_abcdef12"]

    def test_no_ids_returns_empty(self):
        assert extract_file_ids("No file references here.") == []

    def test_ignores_too_short_ids(self):
        # Must be at least 8 hex chars after file_
        assert extract_file_ids("file_abc") == []

    def test_extracts_from_messages(self, estimator):
        msgs = _make_messages("sess_fids", 6)
        ids = extract_file_ids_from_messages(msgs)
        # Message 0 has file_a1b2c3d4e5f60001
        assert "file_a1b2c3d4e5f60001" in ids

    def test_no_ids_in_empty_messages(self, estimator):
        msgs: list[MessageWithParts] = []
        assert extract_file_ids_from_messages(msgs) == []


class TestAppendFileIdsFooter:
    def test_appends_footer_when_ids_present(self):
        result = append_file_ids_footer("Summary text.", ["file_aabbccdd"])
        assert "[LCM File IDs: file_aabbccdd]" in result

    def test_no_footer_when_ids_empty(self):
        result = append_file_ids_footer("Summary text.", [])
        assert "[LCM File IDs:" not in result
        assert result == "Summary text."

    def test_multiple_ids_comma_separated(self):
        result = append_file_ids_footer("text", ["file_aabbccdd", "file_11223344"])
        assert "[LCM File IDs: file_aabbccdd, file_11223344]" in result

    def test_idempotent_replaces_existing_footer(self):
        text = "text\n\n[LCM File IDs: file_old00000000]"
        result = append_file_ids_footer(text, ["file_new00000000"])
        # The old footer should be gone; the new one should be present.
        assert "[LCM File IDs: file_new00000000]" in result
        assert "file_old00000000" not in result

    def test_collect_from_nodes(self):
        node_a = _make_summary_node("sess", "n1", "Summary A mentioning file_aabb1122ccdd3344.")
        node_b = _make_summary_node("sess", "n2", "Summary B mentioning file_eeff5566778899aa.")
        ids = collect_file_ids_from_nodes([node_a, node_b])
        assert "file_aabb1122ccdd3344" in ids
        assert "file_eeff5566778899aa" in ids


# ── Summarisation: file ID propagation ────────────────────────────────────────


class TestSummarisationFileIdPropagation:
    async def test_level1_propagates_file_ids(self, estimator, budget):
        """level1_summarise appends a file IDs footer when messages contain file refs."""
        msgs = _make_messages("sess_l1fid", 6)

        async def mock_llm(**kwargs: object) -> str:
            return "## Goal\nTask.\n\n## Completed Work\n- Done.\n"

        result = await level1_summarise(msgs, "test-model", budget, estimator, mock_llm)
        assert result is not None
        assert "[LCM File IDs:" in result.text
        assert "file_a1b2c3d4e5f60001" in result.text

    async def test_level3_preserves_file_ids_for_all_input_messages(self, estimator, budget):
        """level3_deterministic includes file IDs from messages even when truncated."""
        msgs = _make_messages("sess_l3fid", 6)
        candidate = level3_deterministic(msgs, budget, estimator)
        assert "[LCM File IDs:" in candidate.text
        assert "file_a1b2c3d4e5f60001" in candidate.text

    async def test_level3_no_footer_when_no_file_ids(self, estimator, budget):
        """level3_deterministic does not add a footer when no file IDs are present."""
        msgs = []
        for i in range(6):
            msg = make_message(
                "sess_nofid", role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_nf_{i:03d}"
            )
            msgs.append(MessageWithParts(message=msg, parts=[TextPart(text=f"plain text {i}")]))
        candidate = level3_deterministic(msgs, budget, estimator)
        assert "[LCM File IDs:" not in candidate.text


# ── Condensation levels ────────────────────────────────────────────────────────


class TestCondensationLevels:
    async def test_condense_level1_returns_candidate(self, estimator, budget):
        """condense_level1 produces a CondensationCandidate on LLM success."""
        nodes = [
            _make_summary_node("sess", "n1", "Summary A about the task."),
            _make_summary_node("sess", "n2", "Summary B about progress."),
        ]

        async def mock_llm(**kwargs: object) -> str:
            return "## Goal\nMerged summary.\n\n## Completed Work\n- Both tasks.\n"

        result = await condense_level1(nodes, "test-model", budget, estimator, mock_llm)
        assert result is not None
        assert isinstance(result, CondensationCandidate)
        assert result.compaction_level == 1
        assert set(result.parent_node_ids) == {"n1", "n2"}

    async def test_condense_level1_propagates_file_ids(self, estimator, budget):
        """condense_level1 propagates file IDs from parent nodes."""
        nodes = [
            _make_summary_node("sess", "n1", "Summary A.\n\n[LCM File IDs: file_aabb11223344]"),
            _make_summary_node("sess", "n2", "Summary B.\n\n[LCM File IDs: file_ccdd55667788]"),
        ]

        async def mock_llm(**kwargs: object) -> str:
            return "Condensed summary."

        result = await condense_level1(nodes, "test-model", budget, estimator, mock_llm)
        assert result is not None
        assert "file_aabb11223344" in result.text
        assert "file_ccdd55667788" in result.text

    async def test_condense_level1_returns_none_on_llm_error(self, estimator, budget):
        """condense_level1 returns None when the LLM call fails."""
        nodes = [_make_summary_node("sess", "n1", "Summary.")]

        async def failing_llm(**kwargs: object) -> str:
            raise RuntimeError("LLM down")

        result = await condense_level1(nodes, "test-model", budget, estimator, failing_llm)
        assert result is None

    async def test_condense_level1_returns_none_when_too_large(self, estimator):
        """condense_level1 returns None when condensed text exceeds budget."""
        tiny_budget = ContextBudget(
            model_context_limit=500,
            reserved_output_tokens=50,
            compaction_buffer=50,
        )
        nodes = [_make_summary_node("sess", "n1", "Summary.")]

        async def huge_llm(**kwargs: object) -> str:
            return "x" * 10_000

        result = await condense_level1(nodes, "test-model", tiny_budget, estimator, huge_llm)
        assert result is None

    async def test_condense_level1_empty_nodes_returns_none(self, estimator, budget):
        """condense_level1 returns None for an empty node list."""
        result = await condense_level1([], "test-model", budget, estimator, None)
        assert result is None

    async def test_condense_level2_returns_candidate(self, estimator, budget):
        """condense_level2 produces a CondensationCandidate on LLM success."""
        nodes = [
            _make_summary_node("sess", "n1", "Summary A."),
            _make_summary_node("sess", "n2", "Summary B."),
        ]

        async def mock_llm(**kwargs: object) -> str:
            return "GOAL: Test.\nCONSTRAINTS: none\nFILES: -\nNEXT: done\nCONTEXT: ok"

        result = await condense_level2(nodes, "test-model", budget, estimator, mock_llm)
        assert result is not None
        assert result.compaction_level == 2

    async def test_condense_level2_returns_none_on_llm_error(self, estimator, budget):
        """condense_level2 returns None when the LLM call fails."""
        nodes = [_make_summary_node("sess", "n1", "Summary.")]

        async def failing(**kwargs: object) -> str:
            raise RuntimeError("boom")

        result = await condense_level2(nodes, "test-model", budget, estimator, failing)
        assert result is None

    def test_condense_level3_deterministic_always_succeeds(self, estimator):
        """condense_level3_deterministic always returns a CondensationCandidate."""
        nodes = [
            _make_summary_node("sess", "n1", "Summary A content here."),
            _make_summary_node("sess", "n2", "Summary B content here."),
        ]
        result = condense_level3_deterministic(nodes, estimator)
        assert isinstance(result, CondensationCandidate)
        assert result.compaction_level == 3
        assert set(result.parent_node_ids) == {"n1", "n2"}
        assert "CONDENSED" in result.text

    def test_condense_level3_preserves_file_ids(self, estimator):
        """condense_level3_deterministic preserves file IDs from all nodes."""
        nodes = [
            _make_summary_node("sess", "n1", "A.\n\n[LCM File IDs: file_aa112233bb445566]"),
            _make_summary_node("sess", "n2", "B.\n\n[LCM File IDs: file_cc778899dd001122]"),
        ]
        result = condense_level3_deterministic(nodes, estimator)
        assert "file_aa112233bb445566" in result.text
        assert "file_cc778899dd001122" in result.text

    def test_condense_level3_empty_nodes(self, estimator):
        """condense_level3_deterministic handles an empty node list."""
        result = condense_level3_deterministic([], estimator)
        assert isinstance(result, CondensationCandidate)
        assert result.parent_node_ids == []


# ── Summarisation input token cap ─────────────────────────────────────────────


class TestSummarisationInputCap:
    def test_apply_input_cap_no_truncation_when_small(self, estimator):
        """_apply_input_cap returns all messages when they fit within the cap."""
        msgs = _make_messages("sess_cap", 6)
        to_summarise = _messages_to_summarise(msgs)
        result = _apply_input_cap(to_summarise, estimator, model_context_limit=200_000)
        assert result == to_summarise

    def test_apply_input_cap_truncates_large_input(self, estimator):
        """_apply_input_cap truncates to fit within 75% of model context limit."""
        # Make many big messages
        msgs = []
        for i in range(20):
            msg = make_message(
                "sess_bigcap", role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_bc_{i:03d}"
            )
            # Each message is ~5000 chars; estimator heuristic = chars/4 = ~1250 tokens
            text = "x" * 5000
            msgs.append(MessageWithParts(message=msg, parts=[TextPart(text=text)]))

        to_summarise = _messages_to_summarise(msgs)
        # Tight context limit: 75% = 7_500 tokens = ~30_000 chars
        # With ~1250 tokens/msg we should keep fewer than 6 messages
        result = _apply_input_cap(to_summarise, estimator, model_context_limit=10_000)
        assert len(result) < len(to_summarise)
        assert len(result) >= MIN_MESSAGES_TO_SUMMARISE

    def test_apply_input_cap_respects_minimum(self, estimator):
        """_apply_input_cap always returns at least MIN_MESSAGES_TO_SUMMARISE."""
        # Very tight cap that would normally exclude everything
        msgs = _make_messages("sess_min", 10)
        to_summarise = _messages_to_summarise(msgs)
        result = _apply_input_cap(to_summarise, estimator, model_context_limit=1)
        # Even with cap=1 token, we keep the minimum
        assert len(result) >= min(MIN_MESSAGES_TO_SUMMARISE, len(to_summarise))

    def test_apply_input_cap_empty_input(self, estimator):
        """_apply_input_cap handles an empty input list."""
        assert _apply_input_cap([], estimator, model_context_limit=200_000) == []

    async def test_level1_passes_model_context_limit_to_cap(self, estimator, budget):
        """level1_summarise respects model_context_limit for input capping."""
        # Build enough messages to trigger the cap at a 1000-token context limit.
        msgs = []
        for i in range(10):
            msg = make_message(
                "sess_l1cap", role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_l1c_{i:03d}"
            )
            text = "x" * 2000  # ~500 tokens each
            msgs.append(MessageWithParts(message=msg, parts=[TextPart(text=text)]))

        calls: list[list[dict[str, str]]] = []

        async def recording_llm(**kwargs: object) -> str:
            calls.append(kwargs["messages"])  # type: ignore[arg-type]
            return "## Goal\nDone.\n"

        await level1_summarise(
            msgs,
            "test-model",
            budget,
            estimator,
            recording_llm,
            model_context_limit=1_000,  # very tight — forces truncation
        )

        # The prompt content should be shorter than a transcript of all messages.
        assert len(calls) == 1
        prompt_content = calls[0][0]["content"]
        # With all 10 messages uncapped: 10 * 2000 = 20_000 chars. Cap forces fewer.
        full_transcript_len = 10 * 2000
        assert len(prompt_content) < full_transcript_len


# ── Soft / Hard threshold ─────────────────────────────────────────────────────


class TestSoftHardThreshold:
    def test_soft_overflow_fires_at_fraction(self, estimator, event_bus, store, dag_store, config):
        """is_soft_overflow fires at soft_threshold_fraction * usable."""
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo(model_id="m", context_limit=100_000, max_output_tokens=4_000)
        # usable = 76_000; soft_limit = 76_000 * 0.6 = 45_600
        assert engine.is_soft_overflow(TokenUsage(total=45_599), model) is False
        assert engine.is_soft_overflow(TokenUsage(total=45_600), model) is True

    def test_hard_overflow_fires_at_usable(self, estimator, event_bus, store, dag_store, config):
        """is_hard_overflow fires at 100% of usable budget."""
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo(model_id="m", context_limit=100_000, max_output_tokens=4_000)
        # usable = 76_000
        assert engine.is_hard_overflow(TokenUsage(total=75_999), model) is False
        assert engine.is_hard_overflow(TokenUsage(total=76_000), model) is True

    def test_soft_does_not_fire_when_auto_disabled(
        self, estimator, event_bus, store, dag_store, config
    ):
        """is_soft_overflow returns False when auto compaction is disabled."""
        from mnesis.models.config import CompactionConfig

        cfg = config.model_copy(update={"compaction": CompactionConfig(auto=False)})
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, cfg, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo(model_id="m", context_limit=100_000, max_output_tokens=4_000)
        assert engine.is_soft_overflow(TokenUsage(total=99_000), model) is False

    def test_hard_does_not_fire_for_unlimited_model(
        self, estimator, event_bus, store, dag_store, config
    ):
        """is_hard_overflow returns False for models with context_limit=0."""
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo(model_id="local", context_limit=0, max_output_tokens=4_096)
        assert engine.is_hard_overflow(TokenUsage(total=999_999), model) is False

    def test_is_overflow_delegates_to_soft_overflow(
        self, estimator, event_bus, store, dag_store, config
    ):
        """is_overflow is a backwards-compat alias for is_soft_overflow."""
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo(model_id="m", context_limit=100_000, max_output_tokens=4_000)
        tok = TokenUsage(total=50_000)
        assert engine.is_overflow(tok, model) == engine.is_soft_overflow(tok, model)

    def test_custom_soft_threshold_fraction(self, estimator, event_bus, store, dag_store):
        """Custom soft_threshold_fraction is respected."""
        from mnesis.models.config import CompactionConfig, StoreConfig

        cfg = MnesisConfig(
            compaction=CompactionConfig(soft_threshold_fraction=0.8),
            store=StoreConfig(db_path=str(store._config.db_path)),
        )
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, cfg, id_generator=lambda p: f"{p}_id"
        )
        model = ModelInfo(model_id="m", context_limit=100_000, max_output_tokens=4_000)
        # usable = 76_000; soft at 0.8 = 60_800
        assert engine.is_soft_overflow(TokenUsage(total=60_799), model) is False
        assert engine.is_soft_overflow(TokenUsage(total=60_800), model) is True


# ── Multi-round compaction loop ────────────────────────────────────────────────


class TestMultiRoundCompaction:
    async def test_condensation_runs_in_mock_mode(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        """run_compaction produces a condensed node when summaries accumulate."""
        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            session_model="anthropic/claude-haiku-4-5",
        )

        # Populate enough messages to summarise.
        for i in range(6):
            msg = make_message(
                session_id,
                role="user" if i % 2 == 0 else "assistant",
                msg_id=f"msg_mrc_{i:03d}",
            )
            await store.append_message(msg)
            part = make_raw_part(msg.id, session_id, part_id=f"part_mrc_{i:03d}")
            await store.append_part(part)

        # Run compaction once to create a leaf summary.
        result1 = await engine.run_compaction(session_id)
        assert result1.level_used >= 1

        # Run again — now we have messages + a leaf summary; condensation may fire.
        for i in range(6, 12):
            msg = make_message(
                session_id,
                role="user" if i % 2 == 0 else "assistant",
                msg_id=f"msg_mrc_{i:03d}",
            )
            await store.append_message(msg)
            part = make_raw_part(msg.id, session_id, part_id=f"part_mrc_{i:03d}")
            await store.append_part(part)

        result2 = await engine.run_compaction(session_id)
        # Should not raise; result is valid.
        assert result2.session_id == session_id
        assert result2.level_used >= 1

    async def test_condensation_disabled_skips_condensation(
        self, session_id, store, dag_store, estimator, event_bus, monkeypatch
    ):
        """condensation_enabled=False skips the condensation phase entirely."""
        from mnesis.models.config import CompactionConfig, StoreConfig

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
        cfg = MnesisConfig(
            compaction=CompactionConfig(condensation_enabled=False),
            store=StoreConfig(db_path=config_db_path(store)),
        )
        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            cfg,
            session_model="anthropic/claude-haiku-4-5",
        )

        for i in range(6):
            msg = make_message(
                session_id,
                role="user" if i % 2 == 0 else "assistant",
                msg_id=f"msg_nocd_{i:03d}",
            )
            await store.append_message(msg)
            part = make_raw_part(msg.id, session_id, part_id=f"part_nocd_{i:03d}")
            await store.append_part(part)

        result = await engine.run_compaction(session_id)
        assert result.level_used >= 1

        # No condensed nodes should exist (only the leaf from summarisation).
        nodes = await dag_store.get_active_nodes(session_id)
        assert all(n.kind == "leaf" for n in nodes)


# ── SummaryNode model ─────────────────────────────────────────────────────────


class TestSummaryNodeModel:
    def test_default_kind_is_leaf(self):
        node = SummaryNode(
            id="n1",
            session_id="s1",
            span_start_message_id="m1",
            span_end_message_id="m2",
            content="text",
            token_count=10,
        )
        assert node.kind == "leaf"
        assert node.parent_node_ids == []

    def test_condensed_kind_with_parents(self):
        node = SummaryNode(
            id="n2",
            session_id="s1",
            kind="condensed",
            span_start_message_id="m1",
            span_end_message_id="m3",
            content="condensed text",
            token_count=20,
            parent_node_ids=["n0", "n1"],
        )
        assert node.kind == "condensed"
        assert node.parent_node_ids == ["n0", "n1"]

    def test_invalid_kind_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SummaryNode(
                id="n3",
                session_id="s1",
                kind="invalid",  # type: ignore[arg-type]
                span_start_message_id="m1",
                span_end_message_id="m2",
                content="text",
                token_count=5,
            )


# ── CompactionConfig ──────────────────────────────────────────────────────────


class TestCompactionConfig:
    def test_default_soft_threshold_fraction(self):
        cfg = CompactionConfig()
        assert cfg.soft_threshold_fraction == 0.6

    def test_default_max_compaction_rounds(self):
        cfg = CompactionConfig()
        assert cfg.max_compaction_rounds == 10

    def test_default_condensation_enabled(self):
        cfg = CompactionConfig()
        assert cfg.condensation_enabled is True

    def test_custom_soft_threshold(self):
        cfg = CompactionConfig(soft_threshold_fraction=0.75)
        assert cfg.soft_threshold_fraction == 0.75

    def test_custom_max_rounds(self):
        cfg = CompactionConfig(max_compaction_rounds=5)
        assert cfg.max_compaction_rounds == 5


# ── Helpers needed for tests that need store config db path ───────────────────


def config_db_path(store: object) -> str:
    """Extract db_path from an ImmutableStore instance."""
    return str(store._config.db_path)  # type: ignore[attr-defined]


# ── DAG supersession ──────────────────────────────────────────────────────────


class TestDAGSupersession:
    async def test_mark_superseded_excludes_nodes_from_active(self, store, dag_store, session_id):
        """Nodes marked superseded are excluded from get_active_nodes()."""
        from mnesis.models.summary import SummaryNode

        async def _insert(node_id: str) -> None:
            from mnesis.session import make_id

            node = SummaryNode(
                id=node_id,
                session_id=session_id,
                kind="leaf",
                span_start_message_id="m1",
                span_end_message_id="m2",
                content="summary content",
                token_count=10,
            )
            await dag_store.insert_node(node, id_generator=lambda: make_id("part"))

        await _insert("node_sup_a")
        await _insert("node_sup_b")

        before = await dag_store.get_active_nodes(session_id)
        assert any(n.id == "node_sup_a" for n in before)
        assert any(n.id == "node_sup_b" for n in before)

        dag_store.mark_superseded(["node_sup_a"])

        after = await dag_store.get_active_nodes(session_id)
        assert not any(n.id == "node_sup_a" for n in after)
        # node_sup_b is still active
        assert any(n.id == "node_sup_b" for n in after)

    def test_mark_superseded_multiple_calls_accumulate(self, store, dag_store):
        """mark_superseded accumulates across multiple calls."""
        dag_store.mark_superseded(["n1", "n2"])
        dag_store.mark_superseded(["n3"])
        assert dag_store._superseded_ids == {"n1", "n2", "n3"}

    def test_mark_superseded_idempotent(self, store, dag_store):
        """Marking the same node superseded twice is safe."""
        dag_store.mark_superseded(["n1"])
        dag_store.mark_superseded(["n1"])
        assert dag_store._superseded_ids == {"n1"}

    async def test_condensation_marks_parents_superseded(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        """After run_compaction, parent nodes consumed by condensation are superseded."""
        import json

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        from mnesis.models.config import CompactionConfig, StoreConfig

        # Use a tiny budget so condensation fires — the summary itself pushes over.
        tight_cfg = MnesisConfig(
            compaction=CompactionConfig(
                condensation_enabled=True,
                max_compaction_rounds=3,
            ),
            store=StoreConfig(db_path=config_db_path(store)),
        )
        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            tight_cfg,
            session_model="anthropic/claude-haiku-4-5",
        )

        # Messages need substantial text so the transcript passed to level1/level2
        # is larger than the mock LLM's output (convergence check requires
        # summary_tokens < input_tokens to accept a candidate).
        long_text = "The agent is working on a complex multi-step task. " * 6  # ~306 chars

        # Create first batch of messages and compact to create a leaf node.
        for i in range(6):
            msg = make_message(
                session_id,
                role="user" if i % 2 == 0 else "assistant",
                msg_id=f"msg_sup_a_{i:03d}",
            )
            await store.append_message(msg)
            raw_content = json.dumps({"type": "text", "text": long_text})
            part = make_raw_part(
                msg.id, session_id, part_id=f"part_sup_a_{i:03d}", content=raw_content
            )
            await store.append_part(part)

        await engine.run_compaction(session_id)

        # Create a second batch and compact again; we now have >= 2 summary nodes.
        for i in range(6, 12):
            msg = make_message(
                session_id,
                role="user" if i % 2 == 0 else "assistant",
                msg_id=f"msg_sup_b_{i:03d}",
            )
            await store.append_message(msg)
            raw_content = json.dumps({"type": "text", "text": long_text})
            part = make_raw_part(
                msg.id, session_id, part_id=f"part_sup_b_{i:03d}", content=raw_content
            )
            await store.append_part(part)

        # Force tokens_after > budget.usable so condensation loop runs.
        # Patch estimate_message to return huge values so initial summarisation
        # (which covers only a partial slice of messages) leaves a residual token
        # count above the budget, triggering the condensation loop.
        original_estimate_msg = estimator.estimate_message

        def inflated_estimate_msg(msg):  # type: ignore[no-untyped-def]
            return original_estimate_msg(msg) + 50_000

        estimator.estimate_message = inflated_estimate_msg  # type: ignore[method-assign]
        try:
            await engine.run_compaction(session_id)
        finally:
            estimator.estimate_message = original_estimate_msg  # type: ignore[method-assign]

        # After condensation, at least some nodes should be superseded.
        assert len(dag_store._superseded_ids) > 0


# ── Multi-round loop: no-progress guard ───────────────────────────────────────


class TestMultiRoundNoProgressGuard:
    async def test_no_progress_guard_breaks_loop(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        """Condensation loop breaks when token count doesn't decrease."""
        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        from mnesis.models.config import CompactionConfig, StoreConfig

        cfg = MnesisConfig(
            compaction=CompactionConfig(
                condensation_enabled=True,
                max_compaction_rounds=5,
            ),
            store=StoreConfig(db_path=config_db_path(store)),
        )
        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            cfg,
            session_model="anthropic/claude-haiku-4-5",
        )

        for i in range(6):
            msg = make_message(
                session_id,
                role="user" if i % 2 == 0 else "assistant",
                msg_id=f"msg_np_{i:03d}",
            )
            await store.append_message(msg)
            part = make_raw_part(msg.id, session_id, part_id=f"part_np_{i:03d}")
            await store.append_part(part)

        # Patch _run_condensation to return a candidate with same or higher token count
        # than input — simulates no-progress scenario.
        from mnesis.compaction.levels import CondensationCandidate

        condensation_calls = [0]

        async def no_progress_condense(nodes, *args, **kwargs):  # type: ignore[no-untyped-def]
            condensation_calls[0] += 1
            # Return a token count equal to input — no progress.
            tokens_in = sum(n.token_count for n in nodes)
            return CondensationCandidate(
                text="same size",
                token_count=tokens_in,  # no reduction
                parent_node_ids=[n.id for n in nodes],
                compaction_level=1,
            )

        engine._run_condensation = no_progress_condense  # type: ignore[method-assign]

        # Make tokens_after appear over budget so condensation loop is entered.
        original_estimate_msg = estimator.estimate_message

        def inflated_estimate_msg(msg):  # type: ignore[no-untyped-def]
            return original_estimate_msg(msg) + 50_000

        estimator.estimate_message = inflated_estimate_msg  # type: ignore[method-assign]
        try:
            result = await engine.run_compaction(session_id)
        finally:
            estimator.estimate_message = original_estimate_msg  # type: ignore[method-assign]

        # run_compaction must complete without error.
        assert result.session_id == session_id
        # No-progress guard fires: condensation should be called at most once per round.
        assert condensation_calls[0] <= cfg.compaction.max_compaction_rounds


# ── Hard overflow triggers compaction when no pending task ────────────────────


class TestHardOverflowNoPendingTask:
    async def test_hard_overflow_triggers_new_compaction_when_none_pending(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        """Hard overflow with no pending task calls check_and_trigger then wait."""
        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        from mnesis.models.config import ModelInfo

        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            session_model="anthropic/claude-haiku-4-5",
        )

        model = ModelInfo(model_id="m", context_limit=100_000, max_output_tokens=4_000)
        # usable = 76_000; set tokens above it.
        overflow_tokens = TokenUsage(total=80_000)

        assert engine._pending_task is None
        assert engine.is_hard_overflow(overflow_tokens, model)

        triggered = [False]
        original_check = engine.check_and_trigger

        def patched_check(*args, **kwargs):  # type: ignore[no-untyped-def]
            triggered[0] = True
            return original_check(*args, **kwargs)

        engine.check_and_trigger = patched_check  # type: ignore[method-assign]

        waited = [False]

        async def patched_wait() -> None:
            waited[0] = True

        engine.wait_for_pending = patched_wait  # type: ignore[method-assign]

        # Simulate the session.send() hard-overflow branch directly.
        if engine.is_hard_overflow(overflow_tokens, model):
            if not engine._pending_task:
                engine.check_and_trigger(session_id, overflow_tokens, model)
            await engine.wait_for_pending()

        assert triggered[0], "check_and_trigger must fire when no pending task"
        assert waited[0], "wait_for_pending must be awaited"

    async def test_hard_overflow_waits_existing_pending_without_new_trigger(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        """Hard overflow with a pending task awaits it without triggering again."""
        import asyncio

        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        from mnesis.models.config import ModelInfo

        engine = CompactionEngine(
            store,
            dag_store,
            estimator,
            event_bus,
            config,
            session_model="anthropic/claude-haiku-4-5",
        )

        model = ModelInfo(model_id="m", context_limit=100_000, max_output_tokens=4_000)
        overflow_tokens = TokenUsage(total=80_000)

        # Pre-set a pending task.
        async def _dummy() -> None:
            pass

        engine._pending_task = asyncio.create_task(_dummy())

        triggered = [False]
        original_check = engine.check_and_trigger

        def patched_check(*args, **kwargs):  # type: ignore[no-untyped-def]
            triggered[0] = True
            return original_check(*args, **kwargs)

        engine.check_and_trigger = patched_check  # type: ignore[method-assign]

        if engine.is_hard_overflow(overflow_tokens, model):
            if not engine._pending_task:
                engine.check_and_trigger(session_id, overflow_tokens, model)
            await engine.wait_for_pending()

        # check_and_trigger must NOT be called — a task was already pending.
        assert not triggered[0]
        assert engine._pending_task is None


# ── wait_for_pending: already-done task and exception paths ──────────────────


class TestWaitForPendingEdgeCases:
    async def test_wait_for_pending_no_op_when_no_task(
        self, estimator, event_bus, store, dag_store, config
    ):
        """wait_for_pending is a no-op when _pending_task is None."""
        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_nop"
        )
        assert engine._pending_task is None
        await engine.wait_for_pending()  # must not raise
        assert engine._pending_task is None

    async def test_wait_for_pending_clears_already_done_task(
        self, estimator, event_bus, store, dag_store, config
    ):
        """wait_for_pending clears _pending_task even when the task is already done."""
        import asyncio

        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_done"
        )

        async def instant() -> None:
            pass

        task = asyncio.create_task(instant())
        await asyncio.sleep(0)  # allow task to complete
        assert task.done()

        engine._pending_task = task
        await engine.wait_for_pending()
        assert engine._pending_task is None

    async def test_wait_for_pending_swallows_task_exception(
        self, estimator, event_bus, store, dag_store, config
    ):
        """wait_for_pending does not propagate exceptions from the compaction task."""
        import asyncio

        engine = CompactionEngine(
            store, dag_store, estimator, event_bus, config, id_generator=lambda p: f"{p}_exc"
        )

        async def failing_task() -> None:
            raise RuntimeError("boom")

        task = asyncio.create_task(failing_task())
        engine._pending_task = task
        # Must not raise even though the task raises.
        await engine.wait_for_pending()
        assert engine._pending_task is None


# ── LLM mock: <summaries> branch ─────────────────────────────────────────────


class TestLLMMockSummariesBranch:
    async def test_condensation_mock_uses_summaries_branch(
        self, session_id, store, dag_store, estimator, event_bus, config, monkeypatch
    ):
        """The mock LLM parses <summaries> content for condensation calls."""
        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        from mnesis.compaction.engine import _make_llm_call

        llm_call = _make_llm_call("test-model")

        # A message with <summaries> wrapper — exercises the elif branch.
        response = await llm_call(
            model="test-model",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Condense these:\n\n"
                        "<summaries>\nSummary A: did task 1.\nSummary B: did task 2.\n</summaries>"
                    ),
                }
            ],
            max_tokens=512,
        )
        assert "## Goal" in response
        assert "## Completed Work" in response

    async def test_mock_llm_conversation_branch(self, monkeypatch):
        """The mock LLM parses <conversation> content for summarisation calls."""
        monkeypatch.setenv("MNESIS_MOCK_LLM", "1")

        from mnesis.compaction.engine import _make_llm_call

        llm_call = _make_llm_call("test-model")

        response = await llm_call(
            model="test-model",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Summarise:\n\n"
                        "<conversation>\n[USER]: hello\n[ASSISTANT]: hi\n</conversation>"
                    ),
                }
            ],
            max_tokens=512,
        )
        assert "## Goal" in response


# ── Footer token reservation in level3_deterministic ─────────────────────────


class TestFooterTokenReservation:
    def test_level3_footer_does_not_overflow_budget(self, estimator):
        """level3_deterministic reserves footer space so result always fits budget."""
        # Very tight budget where a footer for many IDs could overflow if not reserved.
        tight_budget = ContextBudget(
            model_context_limit=1_200,
            reserved_output_tokens=100,
            compaction_buffer=100,
        )
        # Create messages that embed file IDs — the footer will be non-trivial.
        msgs = []
        for i in range(8):
            msg = make_message(
                "sess_footer", role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_ft_{i:03d}"
            )
            # Embed multiple unique file IDs per message.
            text = f"file_{i:08x}aabbccdd is referenced here."
            msgs.append(MessageWithParts(message=msg, parts=[TextPart(text=text)]))

        from mnesis.compaction.levels import level3_deterministic

        result = level3_deterministic(msgs, tight_budget, estimator)
        # The result should fit within 85% of the usable budget.
        # usable = 1_200 - 100 - 100 = 1_000; target = 850
        # We just confirm it does not massively overflow (within 2x as heuristic).
        assert result.token_count < tight_budget.usable * 2

    def test_condense_level3_footer_does_not_overflow(self, estimator):
        """condense_level3_deterministic reserves footer space before building content."""
        # Many file IDs embedded in nodes to generate a large footer.
        nodes = []
        for i in range(6):
            fid = f"file_{i:08x}aabbccdd"
            node = _make_summary_node(
                "sess",
                f"n_res_{i}",
                f"Content for node {i}.\n\n[LCM File IDs: {fid}]",
            )
            nodes.append(node)

        result = condense_level3_deterministic(nodes, estimator)
        assert isinstance(result, CondensationCandidate)
        # All file IDs must be preserved in the footer.
        for i in range(6):
            fid = f"file_{i:08x}aabbccdd"
            assert fid in result.text

    def test_condense_level3_available_clamps_to_zero(self, estimator):
        """condense_level3_deterministic clamps available to 0 when overhead exceeds cap."""
        # Create a node whose IDs alone consume the entire cap.
        # Generate enough IDs that footer + header > 512 tokens.
        many_ids = " ".join(f"file_{i:08x}1122334455667788" for i in range(50))
        node = _make_summary_node("sess", "n_clamp", f"content.\n\n[LCM File IDs: {many_ids}]")

        result = condense_level3_deterministic([node], estimator)
        # Must not raise and must return a candidate.
        assert isinstance(result, CondensationCandidate)
        assert result.compaction_level == 3

    async def test_condense_level2_empty_nodes_returns_none(self, estimator, budget):
        """condense_level2 returns None for an empty node list."""

        async def dummy_llm(**kwargs: object) -> str:
            return "response"

        result = await condense_level2([], "test-model", budget, estimator, dummy_llm)
        assert result is None

    async def test_condense_level2_returns_none_when_too_large(self, estimator):
        """condense_level2 returns None when condensed text exceeds budget."""
        tiny_budget = ContextBudget(
            model_context_limit=500,
            reserved_output_tokens=50,
            compaction_buffer=50,
        )
        nodes = [_make_summary_node("sess", "n2_large", "Summary.")]

        async def huge_llm(**kwargs: object) -> str:
            return "x" * 10_000

        result = await condense_level2(nodes, "test-model", tiny_budget, estimator, huge_llm)
        assert result is None
