#!/usr/bin/env python3
"""
locomo.py — LOCOMO long-context memory benchmark for mnesis
===========================================================

Measures whether mnesis context compaction preserves conversational memory by
evaluating QA accuracy before and after compacting long conversation histories.

Two conditions are compared for each LOCOMO conversation:

  baseline   Full conversation history injected and kept intact; no compaction.
  mnesis     Full history injected via session.record(), then compact() called
             before QA questions are asked.

The F1 difference between conditions answers: *does compaction preserve enough
information to answer questions accurately?*

Quick start (no API key — token/compaction stats only):
    uv run python benchmarks/locomo.py --generate-baseline --metrics-only
    uv run python benchmarks/locomo.py --metrics-only

The first command generates baseline data (run once per model). The second
runs the mnesis condition and compares against those files.

Full QA evaluation (requires an LLM API key):
    uv run python benchmarks/locomo.py --generate-baseline \\
        --model anthropic/claude-haiku-4-5 --conversations 1 --questions-per 20
    ANTHROPIC_API_KEY=sk-... uv run python benchmarks/locomo.py \\
        --model anthropic/claude-haiku-4-5 \\
        --conversations 1 \\
        --questions-per 20

Regenerate charts from an existing results file (no API key needed):
    uv run python benchmarks/locomo.py --replot --model anthropic/claude-haiku-4-5 \\
        --conversations 1 --questions-per 20
    uv run python benchmarks/locomo.py --replot \\
        --results-file benchmarks/results/locomo_anthropic-claude-haiku-4-5_c1_q20_all.json

See benchmarks/README.md for full documentation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
except ImportError:
    print(
        "ERROR: matplotlib, numpy, and tqdm are required.\n"
        "Install:  pip install matplotlib numpy tqdm\n"
        "  or:     uv add --dev matplotlib numpy tqdm"
    )
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ── constants ─────────────────────────────────────────────────────────────────

LOCOMO_DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
LOCOMO_DATA_FILENAME = "locomo10.json"

# Integer categories as used in locomo10.json
CATEGORY_NAMES: dict[int, str] = {
    1: "Single-Hop",
    2: "Multi-Hop",
    3: "Temporal",
    4: "Open-Domain",
    5: "Adversarial",
}

# Short labels for compact axes (bar charts with limited x-axis width)
SHORT_CAT: dict[int, str] = {
    1: "Single",
    2: "Multi",
    3: "Temporal",
    4: "Open",
    5: "Adversarial",
}

# Human F1 baseline from LOCOMO paper (Table 1)
HUMAN_F1 = 0.879

# ── compaction prompt ──────────────────────────────────────────────────────────

LOCOMO_COMPACTION_PROMPT = """\
You are creating a detailed memory summary of a long personal conversation.
Your goal is to preserve every factual detail so that questions about this
conversation can be answered accurately later.

Focus especially on:
- Specific dates and times when events occurred (preserve exact dates verbatim)
- Each person's identity, background, relationships, and characteristics
- Activities, hobbies, habits, and interests of each person
- Life events: what happened, when it happened, and outcomes
- Plans and upcoming events, with dates where mentioned
- Numbers, durations, locations, and named entities (people, places, organisations)

Format your response as:

## Participants
(For each person: full name, key identity facts, background, relationships)

## Timeline of Events
(Chronological list — one event per line in the form: DATE — PERSON — EVENT)

## Personal Facts
(Per-person subsections: hobbies, interests, recurring topics, opinions)

## Plans & Upcoming Events
(What each person has planned or mentioned wanting to do, with dates if given)

## Key Details
(Any other specific facts, numbers, durations, locations, or names that might
be relevant to answer questions about this conversation)
"""

BASELINE_DIR = Path(__file__).parent / "baseline"

COLORS = {
    "baseline": "#4878cf",  # blue
    "mnesis": "#e87d3e",  # orange
    "human": "#6aaa5a",  # green
    "reduction": "#d62728",  # red
}

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── run key ───────────────────────────────────────────────────────────────────


def _run_key(model: str, conversations: int, questions_per: int, category: int | None) -> str:
    """Produce a filesystem-safe identifier for a benchmark run configuration.

    The full model string (including provider prefix) is slugified to avoid
    collisions between providers that share a model suffix, e.g.
    ``openai/gpt-4o`` vs ``azure/gpt-4o`` produce distinct keys.
    """
    model_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", model)
    cat = f"cat{category}" if category is not None else "all"
    return f"{model_slug}_c{conversations}_q{questions_per}_{cat}"


# ── data loading ──────────────────────────────────────────────────────────────


def ensure_data(data_path: Path) -> None:
    """Download locomo10.json if not already present."""
    if data_path.exists():
        return
    data_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading LOCOMO dataset → {data_path}")
    try:
        urllib.request.urlretrieve(LOCOMO_DATA_URL, data_path)
    except Exception as exc:
        print(f"ERROR: Could not download LOCOMO data: {exc}")
        print(f"Download manually:\n  {LOCOMO_DATA_URL}")
        sys.exit(1)
    print("Download complete.\n")


def load_conversations(data_path: Path) -> list[dict[str, Any]]:
    """
    Load locomo10.json and return a flat list of conversation dicts.

    Each returned dict has the shape::

        {
            "qa": [...],          # list of QA pairs
            "conversation": {     # session turns
                "speaker_a": "Name",
                "speaker_b": "Name",
                "session_1": [...turns...],
                "session_1_date_time": "...",
                ...
            }
        }
    """
    try:
        with Path(data_path).open() as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}\n"
            "Pass --data /path/to/locomo10.json or let the benchmark download it."
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse dataset file {data_path}: {exc}") from exc
    # Top-level may be a dict keyed by conversation ID or a list
    if isinstance(raw, list):
        return raw
    return list(raw.values())


def extract_turns(convo: dict[str, Any]) -> list[tuple[str, str]]:
    """
    Return all (speaker_name, text) pairs from a conversation in session order.

    Sessions are numbered starting at 1 (``session_1``, ``session_2``, …).
    """
    conv = convo.get("conversation", convo)  # handle nested or flat format
    turns: list[tuple[str, str]] = []
    idx = 1
    while (key := f"session_{idx}") in conv:
        for turn in conv[key]:
            turns.append((turn.get("speaker", ""), turn.get("text", "")))
        idx += 1
    return turns


def extract_qa(convo: dict[str, Any]) -> list[dict[str, Any]]:
    """Return QA pairs with category normalised to int (0 if unrecognised)."""
    pairs = []
    for item in convo.get("qa", []):
        cat = item.get("category", 0)
        try:
            cat = int(cat)
        except (ValueError, TypeError):
            cat = 0
        pairs.append({**item, "category": cat, "answer": str(item.get("answer", ""))})
    return pairs


def speaker_names(convo: dict[str, Any]) -> tuple[str, str]:
    conv = convo.get("conversation", convo)
    return conv.get("speaker_a", "Person A"), conv.get("speaker_b", "Person B")


# ── F1 metric ─────────────────────────────────────────────────────────────────


def _tokenise(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", " ", str(text).lower()).split()


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level partial-match F1 (SQuAD-style)."""
    pred = _tokenise(prediction)
    truth = _tokenise(ground_truth)
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    common = sum((Counter(pred) & Counter(truth)).values())
    if common == 0:
        return 0.0
    p = common / len(pred)
    r = common / len(truth)
    return 2 * p * r / (p + r)


# ── benchmark conditions ───────────────────────────────────────────────────────


async def run_condition(
    convo: dict[str, Any],
    qa_pairs: list[dict[str, Any]],
    *,
    model: str,
    compaction_model: str,
    compact: bool,
    db_path: str,
    max_questions: int,
    metrics_only: bool,
) -> dict[str, Any]:
    """
    Inject a LOCOMO conversation into a mnesis session, optionally compact,
    then answer QA questions.

    Args:
        convo:             LOCOMO conversation dict (with ``qa`` and ``conversation``).
        qa_pairs:          QA pairs to evaluate (pre-filtered by category if needed).
        model:             LiteLLM model string for QA inference.
        compaction_model:  LiteLLM model string for compaction summarisation.
                           Defaults to the session model so no second API key is needed.
        compact:           If True, call ``session.compact()`` after injection.
        db_path:           SQLite path (unique per run to avoid state bleed).
        max_questions:     Cap on QA pairs evaluated.
        metrics_only:      Skip QA inference; record token and compaction stats only.

    Returns:
        Dict with per-question results, token counts, and compaction details.
    """
    from mnesis import MnesisConfig, MnesisSession
    from mnesis.models.config import CompactionConfig

    config = MnesisConfig(
        compaction=CompactionConfig(
            auto=False,
            compaction_model=compaction_model,
            compaction_prompt=LOCOMO_COMPACTION_PROMPT,
        )
    )
    # Build turns with session date headers injected.
    # Session dates are required for multi-hop temporal questions (cat=2) whose
    # answers ("7 May 2023") are derived from the session timestamp combined with
    # relative time cues in the text ("yesterday", "last Sunday"). Without the
    # session date in context these questions are structurally unanswerable.
    conv = convo.get("conversation", convo)
    turns: list[tuple[str, str]] = []
    session_idx = 1
    while (skey := f"session_{session_idx}") in conv:
        date_str = conv.get(f"{skey}_date_time", "")
        for j, turn in enumerate(conv[skey]):
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            if j == 0 and date_str:
                text = f"[Session {session_idx} — {date_str}]\n{text}"
            turns.append((speaker, text))
        session_idx += 1

    snapshots = []
    async with MnesisSession.open(
        model=model,
        system_prompt=(
            "You are a helpful assistant with access to a long conversation history. "
            "Answer questions about that conversation as accurately and concisely as possible."
        ),
        config=config,
        db_path=db_path,
    ) as session:
        # ── inject conversation turns ──────────────────────────────────────
        for i in tqdm(
            range(0, len(turns), 2),
            desc="  injecting",
            unit="turn",
            leave=False,
            dynamic_ncols=True,
        ):
            user_text = f"{turns[i][0]}: {turns[i][1]}"
            asst_text = (
                f"{turns[i + 1][0]}: {turns[i + 1][1]}" if i + 1 < len(turns) else "(no reply)"
            )
            await session.record(
                user_message=user_text,
                assistant_response=asst_text,
            )

        compact_result = None

        if compact:
            compact_result = await session.compact()

        # Use compaction result for window-level token counts when available;
        # fall back to cumulative session usage for the baseline condition.
        if compact_result is not None:
            tokens_before = compact_result.tokens_before
            tokens_after = compact_result.tokens_after
        else:
            tokens_before = session.token_usage.effective_total()
            tokens_after = tokens_before

        # ── QA evaluation ──────────────────────────────────────────────────
        qa_results: list[dict[str, Any]] = []
        if not metrics_only:
            for qa in tqdm(
                qa_pairs[:max_questions],
                desc="  answering",
                unit="q",
                leave=False,
                dynamic_ncols=True,
            ):
                try:
                    turn = await session.send(
                        "Based on the conversation above, answer this question "
                        f"as concisely as possible:\n{qa['question']}"
                    )
                    prediction = turn.text.strip()
                except Exception as exc:
                    print(f"    [warn] QA inference failed: {exc}", file=sys.stderr)
                    prediction = ""
                qa_results.append(
                    {
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "prediction": prediction,
                        "category": qa["category"],
                        "f1": token_f1(prediction, qa["answer"]),
                    }
                )

        # Collect per-turn snapshots before closing the session.
        snapshots = session.history()

    reduction_pct = (1 - tokens_after / tokens_before) * 100 if tokens_before > 0 else 0.0
    return {
        "results": qa_results,
        "tokens_before_compaction": tokens_before,
        "tokens_after_compaction": tokens_after,
        "token_reduction_pct": reduction_pct,
        "compact_result": compact_result.model_dump() if compact_result else None,
        "turns_injected": (len(turns) + 1) // 2,
        "total_qa": len(qa_results),
        "snapshot_metrics": [
            {
                "turn_index": s.turn_index,
                "context_tokens": s.context_tokens.model_dump(),
                "compaction_triggered": s.compaction_triggered,
                "compact_result": s.compact_result.model_dump() if s.compact_result else None,
            }
            for s in snapshots
        ],
    }


# ── aggregation ───────────────────────────────────────────────────────────────


def aggregate_f1(results: list[dict[str, Any]]) -> dict[int, float]:
    """Mean F1 per question category."""
    by_cat: dict[int, list[float]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["f1"])
    return {cat: sum(v) / len(v) for cat, v in by_cat.items() if v}


def overall_f1(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return sum(r["f1"] for r in results) / len(results)


# ── plotting ──────────────────────────────────────────────────────────────────


def _annotate_bars(ax: Any, bars: Any, fmt: str = "{:.1%}") -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_f1_by_category(
    baseline_by_cat: dict[int, float],
    mnesis_by_cat: dict[int, float],
    output_path: Path,
) -> None:
    """
    Grouped bar chart: token-level F1 per question category, baseline vs mnesis.

    Δ annotations above each bar pair show the compaction quality signal.
    Absolute F1 values are reference only; the delta is the meaningful metric.
    """
    cats = sorted(CATEGORY_NAMES)
    labels = [CATEGORY_NAMES[c] for c in cats]
    b_vals = [baseline_by_cat.get(c, 0.0) for c in cats]
    m_vals = [mnesis_by_cat.get(c, 0.0) for c in cats]

    x = np.arange(len(cats))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_b = ax.bar(
        x - w / 2,
        b_vals,
        w,
        label="Baseline (no compaction)",
        color=COLORS["baseline"],
        alpha=0.85,
    )
    bars_m = ax.bar(
        x + w / 2,
        m_vals,
        w,
        label="Mnesis (compacted)",
        color=COLORS["mnesis"],
        alpha=0.85,
    )
    _annotate_bars(ax, bars_b)
    _annotate_bars(ax, bars_m)

    # Annotate each category with the delta — this is the primary signal
    for i, (bv, mv) in enumerate(zip(b_vals, m_vals, strict=False)):
        delta = mv - bv
        color = "#2ca02c" if delta >= -0.02 else "#d62728"
        top = min(max(bv, mv) + 0.06, 1.12)
        ax.text(
            x[i],
            top,
            f"Δ {delta:+.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=color,
        )

    ax.set_xlabel("Question Category")
    ax.set_ylabel("Token-level F1  (absolute values are reference only — see Δ)")
    ax.set_title("LOCOMO: F1 by Category — Baseline vs. Mnesis  (Δ = compaction quality signal)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.2)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_token_usage(
    token_data: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Grouped bar chart: effective token count before and after compaction,
    one pair of bars per evaluated conversation.

    Red percentage labels show the token reduction achieved by compaction.
    """
    n = len(token_data)
    labels = [f"Conv {i + 1}" for i in range(n)]
    before = [d["baseline"]["tokens_before_compaction"] for d in token_data]
    after = [d["mnesis"]["tokens_after_compaction"] for d in token_data]
    reductions = [d["mnesis"]["token_reduction_pct"] for d in token_data]

    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n * 2), 5))
    ax.bar(
        x - w / 2,
        before,
        w,
        label="Before compaction (baseline)",
        color=COLORS["baseline"],
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        after,
        w,
        label="After compaction (mnesis)",
        color=COLORS["mnesis"],
        alpha=0.85,
    )

    for i, (a, r) in enumerate(zip(after, reductions, strict=False)):
        ax.annotate(
            f"-{r:.0f}%",
            xy=(x[i] + w / 2, a),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=COLORS["reduction"],
            fontweight="bold",
        )

    ax.set_xlabel("Conversation")
    ax.set_ylabel("Effective Token Count")
    ax.set_title("LOCOMO: Token Usage Before and After Compaction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_summary(
    b_overall: float,
    m_overall: float,
    baseline_by_cat: dict[int, float],
    mnesis_by_cat: dict[int, float],
    token_data: list[dict[str, Any]],
    compact_stats: dict[str, Any],
    output_path: Path,
    metrics_only: bool,
) -> None:
    """
    2x2 dashboard combining all benchmark metrics into a single figure.

    Layout (primary metrics first):
    - Top-left:  F1 delta by category (compaction quality signal, N/A in metrics-only)
    - Top-right: Token usage before / after compaction (always shown)
    - Bottom-left:  Absolute F1 baseline vs mnesis (reference, N/A in metrics-only)
    - Bottom-right: Compaction summary table
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    na_kw = {"ha": "center", "va": "center", "fontsize": 12, "color": "gray"}

    # ── top-left: F1 delta by category (PRIMARY quality signal) ──────
    ax0 = fig.add_subplot(gs[0, 0])
    if metrics_only:
        ax0.text(
            0.5,
            0.5,
            "QA not run\n(omit --metrics-only\nto see F1 delta)",
            transform=ax0.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#aaaaaa",
            fontstyle="italic",
        )
        ax0.set_title("F1 Delta by Category")
        ax0.axis("off")
    else:
        cats = sorted(CATEGORY_NAMES)
        deltas = [mnesis_by_cat.get(c, 0.0) - baseline_by_cat.get(c, 0.0) for c in cats]
        bar_colors = ["#2ca02c" if d >= -0.02 else "#d62728" for d in deltas]
        x = np.arange(len(cats))
        bars = ax0.bar(x, deltas, color=bar_colors, alpha=0.85, width=0.5)
        offset = 0.012
        for bar, d in zip(bars, deltas, strict=False):
            label_y = d + offset if d >= 0 else d - offset
            va = "bottom" if d >= 0 else "top"
            ax0.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{d:+.2f}",
                ha="center",
                va=va,
                fontsize=9,
                fontweight="bold",
            )
        ax0.axhline(0, color="black", linewidth=0.8, linestyle="-", zorder=0)
        y_pad = 0.08
        ax0.set_ylim(
            min(min(deltas) - y_pad, -y_pad),
            max(max(deltas) + y_pad * 2, y_pad),
        )
        ax0.set_xticks(x)
        ax0.set_xticklabels([SHORT_CAT[c] for c in cats], fontsize=8, rotation=15, ha="right")
        ax0.set_ylabel("F1 delta (Δ)  — closer to 0 is better")
        overall_delta = m_overall - b_overall
        ax0.set_title(f"F1 Delta by Category  (overall Δ {overall_delta:+.3f})")
        ax0.grid(axis="y", alpha=0.3)

    # ── top-right: Token usage (PRIMARY compaction metric, always shown) ──
    ax1 = fig.add_subplot(gs[0, 1])
    if token_data:
        n = len(token_data)
        x = np.arange(n)
        w = 0.35
        before = [d["baseline"]["tokens_before_compaction"] for d in token_data]
        after = [d["mnesis"]["tokens_after_compaction"] for d in token_data]
        reductions = [d["mnesis"]["token_reduction_pct"] for d in token_data]
        ax1.bar(x - w / 2, before, w, color=COLORS["baseline"], alpha=0.85, label="Before")
        ax1.bar(x + w / 2, after, w, color=COLORS["mnesis"], alpha=0.85, label="After")
        for i, (a, r) in enumerate(zip(after, reductions, strict=False)):
            ax1.annotate(
                f"-{r:.0f}%",
                xy=(x[i] + w / 2, a),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=COLORS["reduction"],
                fontweight="bold",
            )
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Conv {i + 1}" for i in range(n)], fontsize=8)
        ax1.set_ylabel("Token Count")
        ax1.legend(fontsize=8)
        ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Tokens Before / After Compaction  (primary metric)")

    # ── bottom-left: Absolute F1 reference ───────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    if metrics_only:
        ax2.text(0.5, 0.5, "N/A\n(run without --metrics-only)", transform=ax2.transAxes, **na_kw)
        ax2.axis("off")
    else:
        cats = sorted(CATEGORY_NAMES)
        x = np.arange(len(cats))
        w = 0.35
        ax2.bar(
            x - w / 2,
            [baseline_by_cat.get(c, 0) for c in cats],
            w,
            color=COLORS["baseline"],
            alpha=0.85,
            label="Baseline",
        )
        ax2.bar(
            x + w / 2,
            [mnesis_by_cat.get(c, 0) for c in cats],
            w,
            color=COLORS["mnesis"],
            alpha=0.85,
            label="Mnesis",
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels([SHORT_CAT[c] for c in cats], fontsize=8, rotation=15, ha="right")
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Token-level F1")
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.3)
    ax2.set_title("Absolute F1 by Category  (reference only — low values are expected)")

    # ── bottom-right: Compaction summary table ────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    rows: list[list[str]] = []
    if not metrics_only:
        delta_overall = m_overall - b_overall
        rows += [
            ["★ F1 delta (overall)", f"{delta_overall:+.3f}"],
            ["★ Avg token reduction", f"{compact_stats.get('avg_reduction_pct', 0):.1f}%"],
            ["", ""],
        ]
    rows += [
        ["Conversations evaluated", str(compact_stats.get("conversations", "—"))],
        ["Avg turns injected", f"{compact_stats.get('avg_turns', 0):.0f}"],
        ["Avg compaction level", f"{compact_stats.get('avg_level', 0):.1f}"],
        ["Avg messages compacted", f"{compact_stats.get('avg_msgs_compacted', 0):.0f}"],
    ]
    if not metrics_only:
        rows += [
            ["", ""],
            ["Baseline overall F1", f"{b_overall:.3f}"],
            ["Mnesis overall F1", f"{m_overall:.3f}"],
        ]
    tbl = ax3.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    ax3.set_title("Compaction Summary  (★ = primary metrics)", pad=20)

    fig.suptitle(
        "LOCOMO Benchmark — Mnesis Context Compaction",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for suptitle
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_sawtooth(
    snapshot_data: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Line plot of total context-window tokens per turn for baseline and mnesis.

    Shows the sawtooth pattern of mnesis (tokens rise → compaction → drop → rise)
    versus the monotonically growing baseline. One subplot per evaluated
    conversation, arranged in a grid.

    Args:
        snapshot_data: The ``"snapshot_metrics"`` dict from the results JSON,
            with keys ``"baseline"`` and ``"mnesis"``. Each value is a list
            of per-conversation snapshot lists (one list of snapshot dicts
            per conversation). ``None`` entries are skipped.
        output_path:   Destination PNG path.
    """
    baseline_list: list[Any] = snapshot_data.get("baseline") or []
    mnesis_list: list[Any] = snapshot_data.get("mnesis") or []

    n_convos = max(len(baseline_list), len(mnesis_list))
    if n_convos == 0:
        return

    # Grid layout: up to 3 columns
    ncols = min(n_convos, 3)
    nrows = math.ceil(n_convos / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(6, ncols * 5), max(4, nrows * 3.5)),
        squeeze=False,
    )

    for conv_idx in range(n_convos):
        row, col = divmod(conv_idx, ncols)
        ax = axes[row][col]

        bl_snaps: list[dict[str, Any]] = (
            baseline_list[conv_idx] if conv_idx < len(baseline_list) else None
        ) or []
        mn_snaps: list[dict[str, Any]] = (
            mnesis_list[conv_idx] if conv_idx < len(mnesis_list) else None
        ) or []

        if bl_snaps:
            bl_turns = [s["turn_index"] for s in bl_snaps]
            bl_totals = [s["context_tokens"]["total"] for s in bl_snaps]
            ax.plot(
                bl_turns,
                bl_totals,
                color=COLORS["baseline"],
                linewidth=1.5,
                label="Baseline",
            )

        if mn_snaps:
            mn_turns = [s["turn_index"] for s in mn_snaps]
            mn_totals = [s["context_tokens"]["total"] for s in mn_snaps]
            ax.plot(
                mn_turns,
                mn_totals,
                color=COLORS["mnesis"],
                linewidth=1.5,
                label="Mnesis",
            )

            # Mark compaction events recorded in snapshot data (compaction_triggered=True).
            # Label only the first line per subplot to avoid duplicate legend entries.
            compact_turns = [s["turn_index"] for s in mn_snaps if s.get("compaction_triggered")]
            ct_label_added = False
            for ct in compact_turns:
                ax.axvline(
                    ct,
                    color=COLORS["reduction"],
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.7,
                    label="Compaction (triggered)" if not ct_label_added else None,
                )
                ct_label_added = True

            # When compaction_triggered is not set in all snapshots, fall back to
            # valley detection: mark turns where total context drops >20%.
            # Label only the first valley to avoid duplicate legend entries.
            valleys = [
                mn_turns[i]
                for i in range(1, len(mn_totals))
                if mn_totals[i] < mn_totals[i - 1] * 0.8  # >20% drop signals compaction
            ]
            valley_label_added = False
            for v in valleys:
                ax.axvline(
                    v,
                    color=COLORS["reduction"],
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.9,
                    label="Compaction (valley)" if not valley_label_added else None,
                )
                valley_label_added = True

        ax.set_title(f"Conv {conv_idx + 1}", fontsize=9)
        ax.set_xlabel("Turn", fontsize=8)
        ax.set_ylabel("Context tokens", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)
        if conv_idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Hide unused subplots
    for extra_idx in range(n_convos, nrows * ncols):
        row, col = divmod(extra_idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "LOCOMO: Context-Window Tokens per Turn  (sawtooth = compaction events)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── replot ────────────────────────────────────────────────────────────────────


def replot(results_path: Path, output_dir: Path) -> None:
    """
    Regenerate all PNG charts from an existing keyed results JSON.

    No LLM calls are made; the function only reads the JSON and writes PNGs.
    Output PNG filenames are derived from the results file stem so they share
    the same run key (e.g. ``locomo_{key}.json`` → ``locomo_f1_{key}.png``,
    ``locomo_tokens_{key}.png``, ``locomo_summary_{key}.png``).

    Args:
        results_path: Path to the ``locomo_{key}.json`` file produced by a
            previous benchmark run.
        output_dir:   Directory where the regenerated PNGs will be written.
            Typically the same directory that contains ``results_path``.

    Raises:
        SystemExit: If ``results_path`` does not exist or cannot be parsed.
    """
    if not results_path.exists():
        print(
            f"ERROR: Results file not found: {results_path}\n"
            "Run the benchmark first, or pass --output-dir to point at an existing"
            " results directory."
        )
        sys.exit(1)

    try:
        with results_path.open() as f:
            data: dict[str, Any] = json.load(f)
    except OSError as exc:
        print(f"ERROR: Could not read {results_path}: {exc}")
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: Could not parse {results_path}: {exc}")
        sys.exit(1)

    metrics_only: bool = data.get("metrics_only", False)

    # Needed to decide whether to render the F1-by-category chart (not available
    # in metrics-only runs where no QA inference was performed).
    has_qa_results: bool = bool(data.get("baseline", {}).get("results"))

    # Reconstruct per-category F1 dicts (keys were serialised as strings)
    baseline_by_cat: dict[int, float] = {
        int(k): v for k, v in data.get("baseline", {}).get("by_category", {}).items()
    }
    mnesis_by_cat: dict[int, float] = {
        int(k): v for k, v in data.get("mnesis", {}).get("by_category", {}).items()
    }

    b_overall: float = data.get("baseline", {}).get("overall_f1", 0.0)
    m_overall: float = data.get("mnesis", {}).get("overall_f1", 0.0)

    # Reconstruct the nested token_data shape expected by plot_token_usage /
    # plot_summary from the flat keys written by main().
    raw_token_data: list[dict[str, Any]] = data.get("token_data", [])
    token_data: list[dict[str, Any]] = [
        {
            "baseline": {
                "tokens_before_compaction": entry["baseline_tokens_before"],
                "tokens_after_compaction": entry["baseline_tokens_before"],  # no compaction
                "token_reduction_pct": 0.0,
            },
            "mnesis": {
                "tokens_before_compaction": entry["baseline_tokens_before"],
                "tokens_after_compaction": entry["mnesis_tokens_after"],
                "token_reduction_pct": entry["token_reduction_pct"],
            },
        }
        for entry in raw_token_data
    ]

    compact_stats: dict[str, Any] = data.get("compact_stats", {})

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: Could not create output directory '{output_dir}': {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Replotting from: {results_path}")
    print(f"Output dir     : {output_dir}\n")
    print("Generating plots…")

    # Derive a key from the results filename so PNGs share the same prefix.
    # Given "locomo_{key}.json", key = stem[len("locomo_"):].
    stem = results_path.stem
    key_suffix = stem[len("locomo_") :] if stem.startswith("locomo_") else stem

    if not metrics_only and has_qa_results:
        plot_f1_by_category(
            baseline_by_cat,
            mnesis_by_cat,
            output_dir / f"locomo_f1_{key_suffix}.png",
        )
    plot_token_usage(token_data, output_dir / f"locomo_tokens_{key_suffix}.png")
    plot_summary(
        b_overall,
        m_overall,
        baseline_by_cat,
        mnesis_by_cat,
        token_data,
        compact_stats,
        output_dir / f"locomo_summary_{key_suffix}.png",
        metrics_only=metrics_only,
    )

    snapshot_data: dict[str, Any] | None = data.get("snapshot_metrics")
    if snapshot_data is not None:
        plot_sawtooth(snapshot_data, output_dir / f"locomo_sawtooth_{key_suffix}.png")
    else:
        print("  (No snapshot_metrics in results file — skipping sawtooth plot.)")

    print("\nDone.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _silence_library_logs() -> None:
    """Suppress INFO/DEBUG output from noisy libraries during benchmark runs.

    Configures both stdlib ``logging`` and ``structlog`` so that mnesis,
    litellm, aiosqlite, and httpx only emit WARNING-level messages or above.
    Must be called before any library modules are first imported so that
    structlog's wrapper class is set before loggers are bound and cached.
    """
    import logging

    import structlog

    # Stdlib-backed loggers (litellm, httpx, httpcore, aiosqlite).
    logging.basicConfig(level=logging.WARNING)
    for name in ("litellm", "LiteLLM", "httpx", "httpcore", "aiosqlite", "mnesis"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # structlog uses its own PrintLogger pipeline by default — reconfigure its
    # wrapper class so all structlog loggers (used by mnesis internals) are
    # clamped to WARNING and above.
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOCOMO benchmark: measures mnesis compaction quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Dry-run (no API key): generate baseline, then run mnesis metrics
  uv run python benchmarks/locomo.py --generate-baseline --metrics-only
  uv run python benchmarks/locomo.py --metrics-only

  # Quick QA evaluation: 1 conversation, 20 questions (generate baseline first)
  uv run python benchmarks/locomo.py --generate-baseline \\
      --model anthropic/claude-haiku-4-5 --conversations 1 --questions-per 20
  ANTHROPIC_API_KEY=sk-... uv run python benchmarks/locomo.py \\
      --model anthropic/claude-haiku-4-5 --conversations 1 --questions-per 20

  # Full evaluation: all 10 conversations (generate baseline first)
  uv run python benchmarks/locomo.py --generate-baseline \\
      --model anthropic/claude-haiku-4-5 --conversations 10 --questions-per 100
  ANTHROPIC_API_KEY=sk-... uv run python benchmarks/locomo.py \\
      --model anthropic/claude-haiku-4-5 --conversations 10 --questions-per 100

  # Only temporal reasoning questions
  uv run python benchmarks/locomo.py --category 3 --metrics-only

  # Replot from a keyed results file (no API key needed)
  uv run python benchmarks/locomo.py --replot \\
      --model anthropic/claude-haiku-4-5 --conversations 1 --questions-per 20
  uv run python benchmarks/locomo.py --replot \\
      --results-file benchmarks/results/locomo_anthropic-claude-haiku-4-5_c1_q20_all.json
""",
    )
    p.add_argument(
        "--model",
        default="anthropic/claude-haiku-4-5",
        help="LiteLLM model string (default: anthropic/claude-haiku-4-5)",
    )
    p.add_argument(
        "--conversations",
        type=int,
        default=1,
        metavar="N",
        choices=range(1, 11),
        help="Conversations to evaluate, 1-10 (default: 1)",
    )
    p.add_argument(
        "--questions-per",
        type=int,
        default=20,
        metavar="N",
        dest="questions_per",
        help="Max QA pairs per conversation (default: 20)",
    )
    p.add_argument(
        "--category",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Restrict to one question category (1=Single-Hop … 5=Adversarial)",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to locomo10.json (auto-downloaded if absent)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Output directory for PNGs and JSON (default: benchmarks/results/)",
    )
    p.add_argument(
        "--compaction-model",
        default=None,
        metavar="MODEL",
        dest="compaction_model",
        help=(
            "LiteLLM model string for compaction summarisation. "
            "Defaults to the same model as --model so only one API key is needed. "
            "Override to use a cheaper model for compaction "
            "(e.g. gemini/gemini-2.0-flash-lite)."
        ),
    )
    p.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip QA inference; report token/compaction stats only (no API key needed)",
    )
    p.add_argument(
        "--replot",
        action="store_true",
        help=(
            "Regenerate PNG charts from an existing results JSON without re-running the "
            "benchmark. No API key required. "
            "If --results-file is given, use that path directly. "
            "Otherwise the keyed filename is derived from --model, --conversations, "
            "--questions-per, and --category and looked up in --output-dir."
        ),
    )
    p.add_argument(
        "--results-file",
        type=Path,
        default=None,
        metavar="PATH",
        dest="results_file",
        help=(
            "Path to a specific results JSON to use with --replot. "
            "When set, --model / --conversations / --questions-per / --category are "
            "ignored for file lookup (though --output-dir still controls where PNGs land)."
        ),
    )
    p.add_argument(
        "--generate-baseline",
        action="store_true",
        dest="generate_baseline",
        help=(
            "Run only the baseline condition (no compaction) and save per-conversation "
            "results to benchmarks/baseline/. No API key needed when combined with "
            "--metrics-only. Use this before running the main benchmark."
        ),
    )
    p.add_argument(
        "--no-snapshot-metrics",
        action="store_true",
        dest="no_snapshot_metrics",
        help=(
            "Omit per-turn context snapshot data from the results JSON "
            "(smaller file, faster serialisation)."
        ),
    )
    return p.parse_args()


# ── baseline generation ───────────────────────────────────────────────────────


async def generate_baseline(
    args: argparse.Namespace,
    conversations: list[dict[str, Any]],
) -> None:
    """
    Run the baseline condition for each conversation and save results to disk.

    Each conversation's result dict is written as a pretty-printed JSON file
    under ``benchmarks/baseline/`` using a key derived from the run configuration.
    After all conversations are processed, a summary is printed showing how many
    files were written and the command to run the main benchmark.

    Args:
        args:          Parsed CLI arguments (model, questions_per, category, metrics_only).
        conversations: List of LOCOMO conversation dicts to evaluate.
    """
    _silence_library_logs()

    key = _run_key(args.model, len(conversations), args.questions_per, args.category)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    cat_label = CATEGORY_NAMES.get(args.category, "all") if args.category else "all"
    mode_label = "metrics-only" if args.metrics_only else "full QA"
    n = len(conversations)
    print(
        f"LOCOMO Baseline  |  model: {args.model}  |  {n} conv{'s' if n != 1 else ''}  "
        f"|  {args.questions_per} q/conv  |  {mode_label}  |  category: {cat_label}"
    )
    print()

    files_written: list[Path] = []

    for conv_idx, convo in tqdm(
        enumerate(conversations),
        total=len(conversations),
        desc="Conversations",
        unit="conv",
        dynamic_ncols=True,
    ):
        spk_a, spk_b = speaker_names(convo)

        qa_pairs = extract_qa(convo)
        if args.category is not None:
            qa_pairs = [q for q in qa_pairs if q["category"] == args.category]
        turns = extract_turns(convo)

        db_path = str(BASELINE_DIR / f"_tmp_{key}_conv{conv_idx}.db")
        t0 = time.monotonic()
        result = await run_condition(
            convo,
            qa_pairs,
            model=args.model,
            compaction_model=args.model,  # compact=False; model unused but required by signature
            compact=False,
            db_path=db_path,
            max_questions=args.questions_per,
            metrics_only=args.metrics_only,
        )
        elapsed = time.monotonic() - t0

        # Clean up the temporary session database and any SQLite WAL/SHM sidecars;
        # best-effort, do not abort on failure.
        for path_str in (db_path, f"{db_path}-wal", f"{db_path}-shm"):
            try:
                Path(path_str).unlink(missing_ok=True)
            except OSError as exc:
                print(f"  [warn] Could not remove temp DB file {path_str}: {exc}", file=sys.stderr)

        if args.no_snapshot_metrics:
            result.pop("snapshot_metrics", None)

        out_path = BASELINE_DIR / f"{key}_conv{conv_idx}.json"
        with out_path.open("w") as f:
            json.dump(result, f, indent=2)
        files_written.append(out_path)

        n_turns = result.get("turns_injected", len(turns) // 2)
        tqdm.write(
            f"[{conv_idx + 1}/{len(conversations)}] {spk_a} & {spk_b}"
            f"   {n_turns} turns   ({elapsed:.1f}s)"
        )

    print(f"\nBaseline data complete: {len(files_written)} file(s) written to {BASELINE_DIR}")
    print("\nRun the mnesis benchmark against this baseline:")
    cat_flag = f" --category {args.category}" if args.category else ""
    print(
        f"  ANTHROPIC_API_KEY=sk-... uv run python benchmarks/locomo.py"
        f" --model {args.model}"
        f" --conversations {len(conversations)}"
        f" --questions-per {args.questions_per}"
        f"{cat_flag}"
    )


# ── main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    _silence_library_logs()
    args = parse_args()

    if args.replot:
        if args.results_file:
            results_path = args.results_file
        else:
            # Derive the expected results filename from CLI args.
            key = _run_key(args.model, args.conversations, args.questions_per, args.category)
            candidate = args.output_dir / f"locomo_{key}.json"
            if candidate.is_file():
                results_path = candidate
            else:
                # Fallback: try to locate a matching results file that may have been
                # created with fewer conversations than requested (e.g. custom data).
                # Derive the glob pattern from the computed key, wildcarding only the
                # conversation count so we stay consistent with `_run_key()`.
                pattern_key = key.replace(f"_c{args.conversations}_", "_c*_")
                pattern = f"locomo_{pattern_key}.json"
                matches = sorted(args.output_dir.glob(pattern))
                if len(matches) == 1:
                    results_path = matches[0]
                elif len(matches) == 0:
                    raise SystemExit(
                        f"Could not find LOCOMO results file. Expected {candidate} "
                        f"or a file matching pattern '{pattern}'. Consider passing "
                        "--results-file explicitly."
                    )
                else:
                    raise SystemExit(
                        "Multiple LOCOMO results files match the requested model, "
                        "questions-per, and category. Please disambiguate by passing "
                        "--results-file explicitly."
                    )
        replot(results_path, args.output_dir)
        return

    data_path = args.data or (Path(__file__).parent / "data" / LOCOMO_DATA_FILENAME)

    ensure_data(data_path)
    conversations = load_conversations(data_path)[: args.conversations]

    compaction_model = args.compaction_model or args.model

    # In metrics-only mode, enable mock LLM for compaction if no API key is
    # detected so compaction reaches level 1 (LLM summary) instead of falling
    # back to level 3 (deterministic), giving a realistic token-reduction figure.
    if args.metrics_only and not os.environ.get("MNESIS_MOCK_LLM"):
        _has_key = any(
            os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
        )
        if not _has_key:
            os.environ["MNESIS_MOCK_LLM"] = "1"
            print("(No API key found — using mock LLM for compaction in metrics-only mode.)\n")

    # ── generate-baseline mode ─────────────────────────────────────────
    if args.generate_baseline:
        await generate_baseline(args, conversations)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cat_label = CATEGORY_NAMES.get(args.category, "all") if args.category else "all"
    mode_label = "metrics-only" if args.metrics_only else "full QA"
    n_convs = len(conversations)
    conv_label = f"{n_convs} conv{'s' if n_convs != 1 else ''}"
    print(
        f"LOCOMO Benchmark  |  model: {args.model}  |  {conv_label}"
        f"  |  {args.questions_per} q/conv  |  {mode_label}  |  category: {cat_label}"
    )
    if compaction_model != args.model:
        print(f"  compaction model : {compaction_model}")
    print()

    if os.environ.get("MNESIS_MOCK_LLM") == "1" and not args.metrics_only:
        print(
            "WARNING: MNESIS_MOCK_LLM=1 is set. QA answers will be mock text and\n"
            "F1 scores will be ~0. Use --metrics-only for a meaningful dry-run,\n"
            "or unset MNESIS_MOCK_LLM and provide a real API key.\n"
        )

    # ── load baseline data from pre-generated files ────────────────────
    key = _run_key(args.model, len(conversations), args.questions_per, args.category)
    baseline_paths = [BASELINE_DIR / f"{key}_conv{i}.json" for i in range(len(conversations))]
    missing = [p for p in baseline_paths if not p.exists()]
    if missing:
        print("ERROR: Baseline data files are missing. Generate them first with:")
        cat_flag = f" --category {args.category}" if args.category else ""
        print(
            f"  uv run python benchmarks/locomo.py --generate-baseline --metrics-only"
            f" --model {args.model}"
            f" --conversations {len(conversations)}"
            f" --questions-per {args.questions_per}"
            f"{cat_flag}"
        )
        print("\nMissing files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    loaded_baselines: list[dict[str, Any]] = []
    for p in baseline_paths:
        with p.open() as f:
            loaded_baselines.append(json.load(f))

    # Warn if full QA run uses baselines generated in metrics-only mode (no QA
    # results), which would silently produce all-zero F1 scores.
    if not args.metrics_only:
        metrics_only_baselines = [
            p
            for p, bl in zip(baseline_paths, loaded_baselines, strict=True)
            if not bl.get("results")
        ]
        if metrics_only_baselines:
            print(
                "WARNING: The following baseline files contain no QA results (were generated\n"
                "  with --metrics-only). F1 scores will be meaningless for this run.\n"
                "  Regenerate baseline without --metrics-only to include QA results:\n"
                f"  uv run python benchmarks/locomo.py --generate-baseline"
                f" --model {args.model}"
                f" --conversations {len(conversations)}"
                f" --questions-per {args.questions_per}"
                + (f" --category {args.category}" if args.category else "")
            )
            for p in metrics_only_baselines:
                print(f"    {p}")
            print()

    all_baseline: list[dict[str, Any]] = []
    all_mnesis: list[dict[str, Any]] = []
    all_mnesis_raw: list[dict[str, Any]] = []  # full per-conversation result dicts
    token_data: list[dict[str, Any]] = []
    compact_levels: list[float] = []
    compact_msgs: list[float] = []

    for idx, convo in tqdm(
        enumerate(conversations),
        total=len(conversations),
        desc="Conversations",
        unit="conv",
        dynamic_ncols=True,
    ):
        spk_a, spk_b = speaker_names(convo)

        qa_pairs = extract_qa(convo)
        if args.category is not None:
            qa_pairs = [q for q in qa_pairs if q["category"] == args.category]

        # ── baseline condition (loaded from pre-generated file) ────────
        baseline = loaded_baselines[idx]

        db_mnes = str(args.output_dir / f"_conv{idx}_mnesis.db")

        # ── mnesis condition ───────────────────────────────────────────
        t0 = time.monotonic()
        mnesis = await run_condition(
            convo,
            qa_pairs,
            model=args.model,
            compaction_model=compaction_model,
            compact=True,
            db_path=db_mnes,
            max_questions=args.questions_per,
            metrics_only=args.metrics_only,
        )
        elapsed = time.monotonic() - t0

        all_baseline.extend(baseline["results"])
        all_mnesis.extend(mnesis["results"])
        all_mnesis_raw.append(mnesis)
        token_data.append({"baseline": baseline, "mnesis": mnesis})

        cr = mnesis.get("compact_result") or {}
        compact_levels.append(cr.get("level_used", 0))
        compact_msgs.append(cr.get("compacted_message_count", 0))

        pct = mnesis["token_reduction_pct"]
        level = cr.get("level_used", 0)
        n_turns = mnesis.get("turns_injected", 0)

        # Build a concise one-liner printed after each conversation completes.
        # Fields included depend on run mode so the line never shows placeholder dashes.
        summary_parts = [
            f"[{idx + 1}/{len(conversations)}] {spk_a} & {spk_b}",
            f"  {n_turns} turns",
            f"  Level {level}",
            f"  -{pct:.1f}%",
        ]
        if not args.metrics_only and baseline["results"] and mnesis["results"]:
            b_f1 = overall_f1(baseline["results"])
            m_f1 = overall_f1(mnesis["results"])
            delta = m_f1 - b_f1
            summary_parts.append(f"  F1D={delta:+.3f}")
        summary_parts.append(f"  ({elapsed:.1f}s)")
        tqdm.write("".join(summary_parts))

        # Clean up temporary session database
        try:
            Path(db_mnes).unlink(missing_ok=True)
        except OSError:
            pass  # Cleanup is best-effort; ignore failures for temp files

    # ── aggregate ──────────────────────────────────────────────────────
    baseline_by_cat = aggregate_f1(all_baseline)
    mnesis_by_cat = aggregate_f1(all_mnesis)
    b_overall = overall_f1(all_baseline)
    m_overall = overall_f1(all_mnesis)

    avg_turns = (
        sum(d["mnesis"]["turns_injected"] for d in token_data) / len(token_data)
        if token_data
        else 0
    )
    avg_red = (
        sum(d["mnesis"]["token_reduction_pct"] for d in token_data) / len(token_data)
        if token_data
        else 0
    )
    compact_stats = {
        "conversations": len(conversations),
        "avg_turns": avg_turns,
        "avg_reduction_pct": avg_red,
        "avg_level": sum(compact_levels) / len(compact_levels) if compact_levels else 0,
        "avg_msgs_compacted": sum(compact_msgs) / len(compact_msgs) if compact_msgs else 0,
    }

    # ── console summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOCOMO Results Summary")
    print("=" * 60)

    # Primary metrics — always shown
    print("\n── Compaction Quality ─────────────────────────────────────")
    reduction = compact_stats["avg_reduction_pct"]
    level = compact_stats["avg_level"]
    msgs = compact_stats["avg_msgs_compacted"]
    print(f"  Avg token reduction : {reduction:.1f}%  (higher = more budget recovered)")
    print(f"  Avg compaction level: {level:.1f}  (1=LLM structured, 3=deterministic)")
    print(f"  Avg messages compacted: {msgs:.0f}")

    if not args.metrics_only and all_baseline:
        delta_overall = m_overall - b_overall
        print(f"\n  Overall F1 delta (Δ): {delta_overall:+.3f}  (0.00 = no information lost)")
        print(f"  Baseline F1 : {b_overall:.3f}  |  Mnesis F1: {m_overall:.3f}")
        print()
        print("  Note: absolute F1 values are reference only — even with session dates")
        print("  injected, models may answer temporal questions with relative phrasing")
        print("  ('yesterday' vs '7 May 2023'), scoring 0 even when conceptually correct.")
        print("  The delta (Δ) is the meaningful metric.")

        # F1 breakdown by category
        print("\n── F1 by Category (reference) ─────────────────────────────")
        print(f"  {'Category':<16} {'Baseline':>10} {'Mnesis':>10} {'Δ':>8}")
        print("  " + "-" * 46)
        for cat in sorted(CATEGORY_NAMES):
            bv = baseline_by_cat.get(cat, float("nan"))
            mv = mnesis_by_cat.get(cat, float("nan"))
            d = mv - bv if not math.isnan(bv) and not math.isnan(mv) else float("nan")
            bvs = f"{bv:.3f}" if not math.isnan(bv) else "—"
            mvs = f"{mv:.3f}" if not math.isnan(mv) else "—"
            ds = f"{d:+.3f}" if not math.isnan(d) else "—"
            print(f"  {CATEGORY_NAMES[cat]:<16} {bvs:>10} {mvs:>10} {ds:>8}")
        print("  " + "-" * 46)
        print(f"  {'Overall':<16} {b_overall:>10.3f} {m_overall:>10.3f} {delta_overall:>+8.3f}")
        print(f"  {'Human baseline':<16} {HUMAN_F1:>10.3f}")

    # ── build snapshot_metrics payload ─────────────────────────────────
    snapshot_metrics: dict[str, Any] | None
    if args.no_snapshot_metrics:
        snapshot_metrics = None
    else:
        # Check whether any baseline file has snapshot data; if not all do,
        # the sawtooth chart will have gaps — still include what we have.
        bl_snapshots: list[Any] = [bl.get("snapshot_metrics") for bl in loaded_baselines]
        mn_snapshots: list[Any] = [r.get("snapshot_metrics") for r in all_mnesis_raw]
        has_any_snapshots = any(s is not None for s in bl_snapshots + mn_snapshots)
        snapshot_metrics = (
            {"baseline": bl_snapshots, "mnesis": mn_snapshots} if has_any_snapshots else None
        )

    # ── save raw results ───────────────────────────────────────────────
    key = _run_key(args.model, len(conversations), args.questions_per, args.category)
    out_json = args.output_dir / f"locomo_{key}.json"
    with out_json.open("w") as f:
        json.dump(
            {
                "model": args.model,
                "conversations_evaluated": len(conversations),
                "metrics_only": args.metrics_only,
                "category_filter": args.category,
                "baseline": {
                    "overall_f1": b_overall,
                    "by_category": {str(k): v for k, v in baseline_by_cat.items()},
                    "results": all_baseline,
                },
                "mnesis": {
                    "overall_f1": m_overall,
                    "by_category": {str(k): v for k, v in mnesis_by_cat.items()},
                    "results": all_mnesis,
                },
                "token_data": [
                    {
                        "baseline_tokens_before": d["baseline"]["tokens_before_compaction"],
                        "mnesis_tokens_after": d["mnesis"]["tokens_after_compaction"],
                        "token_reduction_pct": d["mnesis"]["token_reduction_pct"],
                        "turns_injected": d["mnesis"]["turns_injected"],
                        "compact_result": d["mnesis"]["compact_result"],
                    }
                    for d in token_data
                ],
                "compact_stats": compact_stats,
                "snapshot_metrics": snapshot_metrics,
            },
            f,
            indent=2,
        )
    print(f"\n  Saved: {out_json}")

    # ── generate plots ─────────────────────────────────────────────────
    print("\nGenerating plots…")
    if not args.metrics_only and all_baseline:
        plot_f1_by_category(
            baseline_by_cat,
            mnesis_by_cat,
            args.output_dir / f"locomo_f1_{key}.png",
        )
    plot_token_usage(token_data, args.output_dir / f"locomo_tokens_{key}.png")
    plot_summary(
        b_overall,
        m_overall,
        baseline_by_cat,
        mnesis_by_cat,
        token_data,
        compact_stats,
        args.output_dir / f"locomo_summary_{key}.png",
        metrics_only=args.metrics_only,
    )
    if snapshot_metrics is not None:
        plot_sawtooth(snapshot_metrics, args.output_dir / f"locomo_sawtooth_{key}.png")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
