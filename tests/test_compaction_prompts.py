"""Prompt contents and output-length targets for summarisation/condensation (A1, A3, A4)."""

from __future__ import annotations

import re
from typing import Any

import pytest

from mnesis.compaction.levels import (
    CONDENSE_LEVEL1_PROMPT,
    CONDENSE_LEVEL2_PROMPT,
    LEVEL1_PROMPT,
    LEVEL2_PROMPT,
    _length_target,
    _with_condense_limits,
    _with_length_target,
    condense_level1,
    condense_level2,
    level1_summarise,
    level2_summarise,
)
from mnesis.models.message import ContextBudget, Message, MessageWithParts, TextPart
from mnesis.models.summary import SummaryNode
from mnesis.tokens.estimator import TokenEstimator

ALL_PROMPTS = [LEVEL1_PROMPT, LEVEL2_PROMPT, CONDENSE_LEVEL1_PROMPT, CONDENSE_LEVEL2_PROMPT]


@pytest.fixture
def estimator() -> TokenEstimator:
    e = TokenEstimator()
    e._force_heuristic = True
    return e


@pytest.fixture
def budget() -> ContextBudget:
    return ContextBudget(
        model_context_limit=50_000, reserved_output_tokens=4_000, compaction_buffer=10_000
    )


def _msgs(n: int = 10) -> list[MessageWithParts]:
    out = []
    for i in range(n):
        msg = Message(id=f"msg_{i:03d}", session_id="s", role="user" if i % 2 == 0 else "assistant")
        out.append(
            MessageWithParts(message=msg, parts=[TextPart(text=f"turn {i} " + "word " * 80)])
        )
    return out


def _nodes(tokens_each: int = 600) -> list[SummaryNode]:
    return [
        SummaryNode(
            id=f"n{i}",
            session_id="s",
            kind="leaf",
            span_start_message_id="a",
            span_end_message_id="b",
            content=f"summary {i} " + "fact " * max(50, tokens_each * 4 // 5),
            token_count=tokens_each,
        )
        for i in range(3)
    ]


class Spy:
    """llm_call stub recording the prompt and max_tokens it receives."""

    def __init__(self, reply: str = "## Goal\nok\n") -> None:
        self.reply = reply
        self.prompt = ""
        self.max_tokens = 0

    async def __call__(self, **kwargs: Any) -> str:
        self.prompt = kwargs["messages"][0]["content"]
        self.max_tokens = kwargs["max_tokens"]
        return self.reply


class TestPromptWording:
    @pytest.mark.parametrize("prompt", ALL_PROMPTS)
    def test_never_invent_next_steps(self, prompt):
        """A4: every prompt forbids inventing next steps."""
        assert "never invent" in prompt.lower()
        assert "explicitly" in prompt
        assert 'exactly "None stated"' in prompt

    @pytest.mark.parametrize("prompt", [LEVEL1_PROMPT, CONDENSE_LEVEL1_PROMPT])
    def test_level1_lists_only_explicit_items(self, prompt):
        assert "list only items" in prompt
        assert "the user or assistant explicitly stated" in prompt

    def test_condense_level1_merges_and_omits_empty_sections(self):
        assert "Merge and deduplicate" in CONDENSE_LEVEL1_PROMPT
        assert "do not copy" in CONDENSE_LEVEL1_PROMPT
        assert "Omit any section that would be empty" in CONDENSE_LEVEL1_PROMPT

    @pytest.mark.parametrize("prompt", ALL_PROMPTS)
    def test_files_written_as_path_with_id(self, prompt):
        """A2: every prompt asks for ``path (file_<hex>)`` pairs."""
        assert "path (file_<hex>)" in prompt

    @pytest.mark.parametrize("prompt", ALL_PROMPTS)
    def test_named_people_and_roles_kept(self, prompt):
        """A3: named people and roles survive (compact PEOPLE line at level 2)."""
        assert "on-call" in prompt
        assert "people" in prompt.lower()

    def test_level2_prompts_have_a_people_line(self):
        assert re.search(r"^PEOPLE:", LEVEL2_PROMPT, re.MULTILINE)
        assert re.search(r"^PEOPLE:", CONDENSE_LEVEL2_PROMPT, re.MULTILINE)

    def test_distinct_files_not_merged(self):
        for prompt in (LEVEL1_PROMPT, CONDENSE_LEVEL1_PROMPT):
            assert "never merge" in prompt

    async def test_prompts_reach_the_model(self, estimator, budget):
        spy = Spy()
        _ = await level2_summarise(_msgs(), "m", budget, estimator, spy)
        assert "PEOPLE:" in spy.prompt and "never invent" in spy.prompt
        spy = Spy()
        _ = await condense_level2(_nodes(), "m", budget, estimator, spy)
        assert "PEOPLE:" in spy.prompt and "never invent" in spy.prompt


class TestLengthTarget:
    def test_half_of_input_under_cap(self):
        assert _length_target(2_000, 8_192) == 1_000
        # Capped at 75% of max_tokens so a verbose model still finishes in time.
        assert _length_target(100_000, 2_048) == 1_536
        assert _length_target(1, 2_048) == 1  # never zero

    def test_prompt_states_target_and_hard_limit(self):
        text = _with_length_target("BASE", 500, 2_048)
        assert text.startswith("BASE")
        assert "about 500 tokens" in text and "2048" in text

    def test_condense_limits_derive_bullet_cap_from_target(self):
        text = _with_condense_limits("BASE", 1_162, 2_048)
        assert text.startswith("BASE")
        assert "at most 4 bullets per section" in text
        assert "about 25 words" in text and "hard limit 2048" in text
        assert "tokens in total" not in text  # no token target to ignore
        assert "at most 1 bullets" in _with_condense_limits("BASE", 10, 2_048)

    @staticmethod
    def _expected_bullets(estimator, nodes, max_tokens):
        text = "\n\n---\n\n".join(f"[Summary {i + 1}]:\n{n.content}" for i, n in enumerate(nodes))
        return max(1, _length_target(estimator.estimate(text), max_tokens) // (8 * 35))

    async def test_condense_level1_asks_for_half_within_output_cap(self, estimator, budget):
        """A1: bullet caps derive from the text actually sent, in compaction units."""
        nodes = _nodes(tokens_each=700)  # ~2,100 input tokens: half is under the cap
        spy = Spy()
        cand = await condense_level1(
            nodes, "m", budget, estimator, spy, model_max_output_tokens=2048
        )
        assert cand is not None
        assert spy.max_tokens == 2048
        bullets = self._expected_bullets(estimator, nodes, 2048)
        assert 2 <= bullets < 5  # derived from half the input, below the cap's 5
        assert f"at most {bullets} bullets per section" in spy.prompt
        assert "tokens in total" not in spy.prompt
        assert "hard limit 2048" in spy.prompt
        assert CONDENSE_LEVEL1_PROMPT.splitlines()[0] in spy.prompt

    async def test_condense_target_never_exceeds_cap(self, estimator, budget):
        nodes = _nodes(tokens_each=50_000)
        spy = Spy()
        _ = await condense_level1(
            nodes,
            "m",
            budget,
            estimator,
            spy,
            model_max_output_tokens=2048,
            model_context_limit=1_000_000,
        )
        assert "at most 5 bullets per section" in spy.prompt  # 1536 // 280

    async def test_length_target_uses_compaction_estimator_units(self, estimator, budget):
        """The input size behind the target is measured in the compaction model's units."""
        dense = TokenEstimator()
        dense._force_heuristic = True
        dense.estimate = lambda text, model=None: len(text) // 2 or 1  # type: ignore[method-assign]
        nodes = _nodes(tokens_each=300)
        sparse_spy, dense_spy = Spy(), Spy()
        _ = await condense_level1(
            nodes, "m", budget, estimator, sparse_spy, model_max_output_tokens=4096
        )
        _ = await condense_level1(
            nodes,
            "m",
            budget,
            estimator,
            dense_spy,
            model_max_output_tokens=4096,
            compaction_estimator=dense,
        )
        n_sparse = int(re.search(r"at most (\d+) bullets", sparse_spy.prompt).group(1))
        n_dense = int(re.search(r"at most (\d+) bullets", dense_spy.prompt).group(1))
        assert n_dense > n_sparse

    async def test_summarise_level1_default_prompt_gets_target(self, estimator, budget):
        spy = Spy()
        cand = await level1_summarise(_msgs(), "m", budget, estimator, spy)
        assert cand is not None
        assert "Length: aim for about" in spy.prompt
        assert f"hard limit {spy.max_tokens} tokens" in spy.prompt

    async def test_summarise_level1_custom_prompt_left_untouched(self, estimator, budget):
        spy = Spy()
        _ = await level1_summarise(
            _msgs(), "m", budget, estimator, spy, compaction_prompt="CUSTOM PROMPT"
        )
        assert spy.prompt.startswith("CUSTOM PROMPT\n\n<conversation>")
        assert "Length:" not in spy.prompt

    async def test_target_line_counts_against_small_compaction_window(self, estimator, budget):
        """The extra line is reserved when capping input to the model's window."""
        window, out = 8_192, 2_048
        spy = Spy()
        _ = await level1_summarise(
            _msgs(60),
            "m",
            budget,
            estimator,
            spy,
            model_context_limit=window,
            model_max_output_tokens=out,
        )
        assert estimator.estimate(spy.prompt) + spy.max_tokens <= window
