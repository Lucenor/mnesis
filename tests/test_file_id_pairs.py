"""``path (file_<hex>)`` pairing in file-ID footers (A2) across compaction levels."""

from __future__ import annotations

import pytest

from mnesis.compaction.file_ids import (
    append_file_ids_footer,
    collect_file_id_paths_from_nodes,
    collect_file_ids_from_nodes,
    extract_file_id_paths,
    extract_file_id_paths_from_messages,
    extract_file_ids,
    footer_file_id_paths,
    strip_file_ids_footer,
)
from mnesis.compaction.levels import (
    condense_level1,
    condense_level2,
    condense_level3_deterministic,
    level1_summarise,
    level2_summarise,
    level3_deterministic,
)
from mnesis.models.message import ContextBudget, Message, MessageWithParts, TextPart
from mnesis.models.summary import SummaryNode
from mnesis.tokens.estimator import TokenEstimator

RATES = "file_3fa9c2d17b8e4a60"
MIGRATE = "file_9b1e0c44a7d2f381"
LONE = "file_deadbeef01234567"
RATES_PATH = "services/billing/config/rates.yaml"
MIGRATE_PATH = "scripts/migrate_ledger.py"


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


def _msgs() -> list[MessageWithParts]:
    texts = [
        f"The rates config lives at {RATES_PATH} ({RATES}).",
        "Acknowledged.",
        f"Migration is {MIGRATE_PATH}: {MIGRATE}. Also see {LONE}.",
        "Noted.",
        "Third question " + "x" * 50,
        "Answer " + "y" * 50,
        "Fourth question",
        "Answer four",
    ]
    out = []
    for i, text in enumerate(texts):
        msg = Message(id=f"msg_{i:03d}", session_id="s", role="user" if i % 2 == 0 else "assistant")
        out.append(MessageWithParts(message=msg, parts=[TextPart(text=text)]))
    return out


def _node(node_id: str, content: str) -> SummaryNode:
    return SummaryNode(
        id=node_id,
        session_id="s",
        kind="leaf",
        span_start_message_id="a",
        span_end_message_id="b",
        content=content,
        token_count=max(1, len(content) // 4),
    )


class TestFooterFormat:
    def test_pairs_render_and_round_trip(self):
        text = append_file_ids_footer(
            "prose", [RATES, MIGRATE, LONE], {RATES: RATES_PATH, MIGRATE: MIGRATE_PATH}
        )
        assert text.endswith(
            f"[LCM File IDs: {RATES_PATH} ({RATES}), {MIGRATE_PATH} ({MIGRATE}), {LONE}]"
        )
        assert footer_file_id_paths(text) == {RATES: RATES_PATH, MIGRATE: MIGRATE_PATH}
        assert extract_file_ids(text) == [RATES, MIGRATE, LONE]  # unpaired id kept
        assert strip_file_ids_footer(text) == "prose"
        # Idempotent: re-appending replaces the footer instead of duplicating it.
        again = append_file_ids_footer(text, [RATES, MIGRATE, LONE], {RATES: RATES_PATH})
        assert again.count("[LCM File IDs:") == 1

    def test_old_bare_format_still_parses(self):
        old = f"summary\n\n[LCM File IDs: {RATES}, {MIGRATE}]"
        assert extract_file_ids(old) == [RATES, MIGRATE]
        assert footer_file_id_paths(old) == {}
        assert collect_file_ids_from_nodes([_node("n", old)]) == [RATES, MIGRATE]
        assert strip_file_ids_footer(old) == "summary"
        # Bare ids with no path anywhere stay bare when re-footered.
        assert append_file_ids_footer("x", [RATES, MIGRATE], {}).endswith(
            f"[LCM File IDs: {RATES}, {MIGRATE}]"
        )

    def test_no_ids_no_footer(self):
        assert append_file_ids_footer("x", [], {RATES: RATES_PATH}) == "x"

    def test_mixed_old_and_new_entries(self):
        text = f"s\n\n[LCM File IDs: {RATES}, {MIGRATE_PATH} ({MIGRATE})]"
        assert footer_file_id_paths(text) == {MIGRATE: MIGRATE_PATH}


class TestPathExtraction:
    @pytest.mark.parametrize(
        ("line", "path"),
        [
            (f"{RATES_PATH} ({RATES})", RATES_PATH),
            (f"`{RATES_PATH}` ({RATES})", RATES_PATH),
            (f"{RATES_PATH}: {RATES}", RATES_PATH),
            (f"{RATES}: {RATES_PATH}", RATES_PATH),
            (f"{RATES} ({RATES_PATH})", RATES_PATH),
            (f"{RATES}={RATES_PATH}", RATES_PATH),
            (f"- {RATES_PATH}  ({RATES}) -- rates", RATES_PATH),
            (f"rates.yaml ({RATES})", "rates.yaml"),
        ],
    )
    def test_adjacent_forms_pair(self, line, path):
        assert extract_file_id_paths(line) == {RATES: path}

    def test_two_ids_two_paths_in_the_prompt_form(self):
        line = f"{RATES_PATH} ({RATES}), {MIGRATE_PATH} ({MIGRATE})"
        assert extract_file_id_paths(line) == {RATES: RATES_PATH, MIGRATE: MIGRATE_PATH}

    def test_path_containing_file_underscore_pairs(self):
        assert extract_file_id_paths(f"src/file_utils.py ({RATES})") == {RATES: "src/file_utils.py"}

    @pytest.mark.parametrize(
        "line",
        [
            f"Loaded {RATES} via json.loads then saved",
            f"Result of {RATES} at self.config",
            f"Priya sent {RATES} from priya.raman@corp.com",
            f"{RATES} sent to priya.raman@corp.com",
            f"see https://example.com/docs/page.html for {RATES}",
            f"{RATES} -> src/a.py",
            f"{RATES} belongs to README.md, e.g. the docs",
            f"call load.py({RATES})",
            f"the id {RATES} and v1.2.3 build",
            f"e.g. {RATES} has no path",
            '{"ids": ["' + RATES + '", "' + MIGRATE + '"], "paths": ["a/x.py", "b/y.py"]}',
            f"{RATES} {MIGRATE} -> src/a.py src/b.py",
            f"{RATES} is NOT config.yaml (that one is {MIGRATE})",
            f"{RATES_PATH} (or other.yaml) -- file id: {RATES}",
            f"{RATES_PATH} content id {RATES}",
            f"{RATES}: N/A",
            f"{RATES} (and/or the backup)",
            f"{RATES} (TCP/IP dump)",
            f"{RATES} (3/4 done)",
            f"{RATES} (input/output)",
            f"{RATES} (Node.js build)",
            f"{RATES}: Next.js",
            f"{RATES}: Vue.js app",
            f"{RATES}: src/a",  # directory-looking, no extension, no path prefix
            f"{RATES}: {MIGRATE}: n/a",
        ],
    )
    def test_no_confident_wrong_pairs(self, line):
        assert extract_file_id_paths(line) == {}

    def test_one_path_between_two_ids_pairs_neither(self):
        line = f"{RATES}: src/a.py ({MIGRATE})"
        assert extract_file_id_paths(line) == {}
        assert extract_file_id_paths(f"{RATES} ({MIGRATE_PATH}) {MIGRATE}") == {RATES: MIGRATE_PATH}

    @pytest.mark.parametrize(
        "path", ["/etc/hosts", "./run.sh", "../lib/x", "~/notes/todo", "App.tsx"]
    )
    def test_prefixed_and_extension_paths_pair(self, path):
        assert extract_file_id_paths(f"{path} ({RATES})") == {RATES: path}

    @pytest.mark.parametrize(
        "path",
        [
            "logs/2024/01/app.log",
            "data/2023/report.csv",
            "docker/Dockerfile",
            "Makefile",
            "src/Procfile",
            "LICENSE",
            "README",
            "config/.env",
            ".env",
            ".gitignore",
            "~/.zshrc",
            "dist/app.tar.gz",
            "types/index.d.ts",
            "settings/tsconfig.jsonc",
            "docs/guide.markdown",
            "src/node.js",
        ],
    )
    def test_realistic_paths_pair(self, path):
        assert extract_file_id_paths(f"{path} ({RATES})") == {RATES: path}

    @pytest.mark.parametrize("token", ["3/4", "2024/01", "1/2", "10/20/30"])
    def test_numeric_last_segment_is_not_a_path(self, token):
        assert extract_file_id_paths(f"{RATES} ({token} done)") == {}
        assert extract_file_id_paths(f"{token} ({RATES})") == {}

    def test_each_path_names_one_id(self):
        pairs = extract_file_id_paths(f"{RATES} {MIGRATE}: scripts/m.py")
        assert pairs == {MIGRATE: "scripts/m.py"}

    def test_pairs_with_adjacent_path_only(self):
        text = (
            f"- {RATES_PATH} (or other.yaml) -- file id: {RATES}\n"
            f"`{MIGRATE_PATH}` ({MIGRATE})\n"
            f"e.g. {LONE} has no path\n"
        )
        assert extract_file_id_paths(text) == {MIGRATE: MIGRATE_PATH}

    def test_from_messages(self):
        assert extract_file_id_paths_from_messages(_msgs()) == {
            RATES: RATES_PATH,
            MIGRATE: MIGRATE_PATH,
        }

    def test_footer_pairs_outrank_prose_for_nodes(self):
        node = _node(
            "n1",
            f"- other_name.yaml ({RATES})\n\n[LCM File IDs: {RATES_PATH} ({RATES})]",
        )
        assert collect_file_id_paths_from_nodes([node]) == {RATES: RATES_PATH}

    def test_earliest_node_wins(self):
        a = _node("a", f"[LCM File IDs: first.py ({RATES})]")
        b = _node("b", f"[LCM File IDs: second.py ({RATES})]")
        assert collect_file_id_paths_from_nodes([a, b]) == {RATES: "first.py"}


class TestLevelsKeepPairs:
    async def test_level1_and_level2_summaries(self, estimator, budget):
        async def llm(**kwargs: object) -> str:
            return "## Goal\nProse that names no ids.\n"

        for fn in (level1_summarise, level2_summarise):
            cand = await fn(_msgs(), "m", budget, estimator, llm)
            assert cand is not None
            assert footer_file_id_paths(cand.text) == {RATES: RATES_PATH, MIGRATE: MIGRATE_PATH}
            assert LONE in extract_file_ids(cand.text)

    def test_level3_summary(self, estimator, budget):
        cand = level3_deterministic(_msgs(), budget, estimator)
        assert footer_file_id_paths(cand.text) == {RATES: RATES_PATH, MIGRATE: MIGRATE_PATH}

    async def test_condensed_l1_l2_l3_keep_pairing_from_leaf_footers(self, estimator, budget):
        leaf1 = _node("n1", f"## Goal\nstuff\n\n[LCM File IDs: {RATES_PATH} ({RATES}), {LONE}]")
        leaf2 = _node("n2", f"## Goal\nmore\n\n[LCM File IDs: {MIGRATE_PATH} ({MIGRATE})]")
        nodes = [leaf1, leaf2]

        async def llm(**kwargs: object) -> str:
            return "GOAL: merged. FILES: (no ids written)"

        expected = {RATES: RATES_PATH, MIGRATE: MIGRATE_PATH}
        for cond in (
            await condense_level1(nodes, "m", budget, estimator, llm),
            await condense_level2(nodes, "m", budget, estimator, llm),
            condense_level3_deterministic(nodes, estimator, budget),
        ):
            assert cond is not None
            assert footer_file_id_paths(cond.text) == expected
            assert set(extract_file_ids(cond.text)) == {RATES, MIGRATE, LONE}

    async def test_pairing_survives_two_generations(self, estimator, budget):
        """leaf footer -> condensed footer -> condensed-of-condensed footer."""

        async def llm(**kwargs: object) -> str:
            return "GOAL: g"

        leaf = _node("n1", f"s\n\n[LCM File IDs: {RATES_PATH} ({RATES})]")
        other = _node("n2", f"t\n\n[LCM File IDs: {MIGRATE_PATH} ({MIGRATE})]")
        gen1 = await condense_level2([leaf, other], "m", budget, estimator, llm)
        assert gen1 is not None
        node1 = _node("c1", gen1.text)
        gen2 = await condense_level2([node1, _node("n3", "plain")], "m", budget, estimator, llm)
        assert gen2 is not None
        assert footer_file_id_paths(gen2.text) == {RATES: RATES_PATH, MIGRATE: MIGRATE_PATH}


class TestPairsAreSecondaryToIds:
    def test_paths_dropped_when_paired_footer_does_not_fit(self, estimator):
        """Ids outrank paths: a tight budget keeps every id, bare."""
        ids = [f"file_{i:016x}" for i in range(40)]
        long_path = "d/" * 40
        msgs = []
        for i in range(8):
            text = " ".join(
                f"{long_path}x{j}.py ({fid})" for j, fid in enumerate(ids[i * 5 : i * 5 + 5])
            )
            msg = Message(id=f"m{i}", session_id="s", role="user" if i % 2 == 0 else "assistant")
            msgs.append(MessageWithParts(message=msg, parts=[TextPart(text=text)]))
        # Budget sized so bare ids fit but their paired form does not.
        bare = estimator.estimate(append_file_ids_footer("", ids))
        tiny = ContextBudget(
            model_context_limit=int(bare / 0.85) + 60,
            reserved_output_tokens=0,
            compaction_buffer=0,
        )
        assert tiny.usable < estimator.estimate(
            append_file_ids_footer("", ids, extract_file_id_paths_from_messages(msgs))
        )
        cand = level3_deterministic(msgs, tiny, estimator)
        assert set(extract_file_ids(cand.text)) == set(ids)
        assert footer_file_id_paths(cand.text) == {}
        assert cand.token_count <= tiny.usable


class TestPathsRankBelowProse:
    """ids > prose > paths: paths never shrink the prose the deterministic levels keep."""

    @staticmethod
    def _msgs() -> list[MessageWithParts]:
        out = []
        for i in range(12):
            text = f"turn {i} " + "word " * 30 + f" see src/mod_{i}.py (file_{i:016x})"
            msg = Message(id=f"m{i}", session_id="s", role="user" if i % 2 == 0 else "assistant")
            out.append(MessageWithParts(message=msg, parts=[TextPart(text=text)]))
        return out

    def test_level3_prose_is_the_same_with_and_without_paths(self, estimator, monkeypatch):
        msgs = self._msgs()
        ids = extract_file_id_paths_from_messages(msgs)
        assert len(ids) == 12
        bare_footer = estimator.estimate(append_file_ids_footer("", list(ids)))
        # Room for prose plus bare ids, but not for the paired footer on top of that.
        budget = ContextBudget(
            model_context_limit=int((bare_footer + 330) / 0.85),
            reserved_output_tokens=0,
            compaction_buffer=0,
        )
        with_paths = level3_deterministic(msgs, budget, estimator)
        monkeypatch.setattr(
            "mnesis.compaction.levels.extract_file_id_paths_from_messages", lambda m: {}
        )
        without = level3_deterministic(msgs, budget, estimator)
        assert strip_file_ids_footer(with_paths.text) == strip_file_ids_footer(without.text)
        assert with_paths.token_count <= budget.usable


def test_paths_that_fit_ignores_ids_without_a_path(estimator):
    from mnesis.compaction.levels import _paths_that_fit

    assert _paths_that_fit([RATES], {MIGRATE: MIGRATE_PATH}, 0, 10_000, estimator) == {}
