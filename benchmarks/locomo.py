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
    uv run python benchmarks/locomo.py --metrics-only

Full QA evaluation (requires an LLM API key):
    ANTHROPIC_API_KEY=sk-... uv run python benchmarks/locomo.py \\
        --model anthropic/claude-haiku-4-5 \\
        --conversations 1 \\
        --questions-per 20

See benchmarks/README.md for full documentation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
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
except ImportError:
    print(
        "ERROR: matplotlib and numpy are required.\n"
        "Install:  pip install matplotlib numpy\n"
        "  or:     uv add --dev matplotlib numpy"
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
    import re

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
    turns = extract_turns(convo)

    async with await MnesisSession.create(
        model=model,
        system_prompt=(
            "You are a helpful assistant with access to a long conversation history. "
            "Answer questions about that conversation as accurately and concisely as possible."
        ),
        config=config,
        db_path=db_path,
    ) as session:
        # ── inject conversation turns ──────────────────────────────────────
        for i in range(0, len(turns), 2):
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
            for qa in qa_pairs[:max_questions]:
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

    reduction_pct = (1 - tokens_after / tokens_before) * 100 if tokens_before > 0 else 0.0
    return {
        "results": qa_results,
        "tokens_before_compaction": tokens_before,
        "tokens_after_compaction": tokens_after,
        "token_reduction_pct": reduction_pct,
        "compact_result": compact_result.model_dump() if compact_result else None,
        "turns_injected": (len(turns) + 1) // 2,
        "total_qa": len(qa_results),
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
        top = max(bv, mv) + 0.06
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
        ax0.text(0.5, 0.5, "N/A\n(run without --metrics-only)", transform=ax0.transAxes, **na_kw)
        ax0.axis("off")
    else:
        cats = sorted(CATEGORY_NAMES)
        deltas = [mnesis_by_cat.get(c, 0.0) - baseline_by_cat.get(c, 0.0) for c in cats]
        bar_colors = ["#2ca02c" if d >= -0.02 else "#d62728" for d in deltas]
        x = np.arange(len(cats))
        bars = ax0.bar(x, deltas, color=bar_colors, alpha=0.85, width=0.5)
        for bar, d in zip(bars, deltas, strict=False):
            ax0.text(
                bar.get_x() + bar.get_width() / 2,
                d + (0.005 if d >= 0 else -0.015),
                f"{d:+.2f}",
                ha="center",
                va="bottom" if d >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )
        ax0.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax0.set_xticks(x)
        ax0.set_xticklabels([CATEGORY_NAMES[c][:6] for c in cats], fontsize=8)
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
        ax2.set_xticklabels([CATEGORY_NAMES[c][:6] for c in cats], fontsize=8)
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
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOCOMO benchmark: measures mnesis compaction quality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Dry-run (no API key): token/compaction stats only
  uv run python benchmarks/locomo.py --metrics-only

  # Quick QA evaluation: 1 conversation, 20 questions
  ANTHROPIC_API_KEY=sk-... uv run python benchmarks/locomo.py \\
      --model anthropic/claude-haiku-4-5 --conversations 1 --questions-per 20

  # Full evaluation: all 10 conversations
  ANTHROPIC_API_KEY=sk-... uv run python benchmarks/locomo.py \\
      --model anthropic/claude-haiku-4-5 --conversations 10 --questions-per 100

  # Only temporal reasoning questions
  uv run python benchmarks/locomo.py --category 3 --metrics-only
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
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────


async def main() -> None:
    args = parse_args()

    data_path = args.data or (Path(__file__).parent / "data" / LOCOMO_DATA_FILENAME)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ensure_data(data_path)
    conversations = load_conversations(data_path)[: args.conversations]

    compaction_model = args.compaction_model or args.model

    print("LOCOMO Benchmark")
    print(f"  Model            : {args.model}")
    print(f"  Compaction model : {compaction_model}")
    print(f"  Conversations    : {len(conversations)}")
    print(f"  Max questions    : {args.questions_per}")
    cat_label = CATEGORY_NAMES.get(args.category, "all") if args.category else "all"
    print(f"  Category         : {cat_label}")
    print(f"  Mode             : {'metrics-only' if args.metrics_only else 'full QA'}")
    print(f"  Output           : {args.output_dir}\n")

    if os.environ.get("MNESIS_MOCK_LLM") == "1" and not args.metrics_only:
        print(
            "WARNING: MNESIS_MOCK_LLM=1 is set. QA answers will be mock text and\n"
            "F1 scores will be ~0. Use --metrics-only for a meaningful dry-run,\n"
            "or unset MNESIS_MOCK_LLM and provide a real API key.\n"
        )

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

    all_baseline: list[dict[str, Any]] = []
    all_mnesis: list[dict[str, Any]] = []
    token_data: list[dict[str, Any]] = []
    compact_levels: list[float] = []
    compact_msgs: list[float] = []

    for idx, convo in enumerate(conversations):
        spk_a, spk_b = speaker_names(convo)
        print(f"Conversation {idx + 1}/{len(conversations)}: {spk_a} & {spk_b}")

        qa_pairs = extract_qa(convo)
        if args.category is not None:
            qa_pairs = [q for q in qa_pairs if q["category"] == args.category]
        turns = extract_turns(convo)
        print(f"  Turns: {len(turns)}  QA pairs available: {len(qa_pairs)}")

        db_base = str(args.output_dir / f"_conv{idx}_baseline.db")
        db_mnes = str(args.output_dir / f"_conv{idx}_mnesis.db")

        # ── baseline condition ─────────────────────────────────────────
        print("  [baseline] injecting history…", end=" ", flush=True)
        t0 = time.monotonic()
        baseline = await run_condition(
            convo,
            qa_pairs,
            model=args.model,
            compaction_model=compaction_model,
            compact=False,
            db_path=db_base,
            max_questions=args.questions_per,
            metrics_only=args.metrics_only,
        )
        print(f"done ({time.monotonic() - t0:.1f}s)")

        # ── mnesis condition ───────────────────────────────────────────
        print("  [mnesis  ] injecting + compacting…", end=" ", flush=True)
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
        print(f"done ({time.monotonic() - t0:.1f}s)")

        all_baseline.extend(baseline["results"])
        all_mnesis.extend(mnesis["results"])
        token_data.append({"baseline": baseline, "mnesis": mnesis})

        cr = mnesis.get("compact_result") or {}
        compact_levels.append(cr.get("level_used", 0))
        compact_msgs.append(cr.get("compacted_message_count", 0))

        pct = mnesis["token_reduction_pct"]
        print(f"  Token reduction : {pct:.1f}%")
        if not args.metrics_only and baseline["results"]:
            b_f1 = overall_f1(baseline["results"])
            m_f1 = overall_f1(mnesis["results"])
            delta = m_f1 - b_f1
            print(f"  F1 Δ : {delta:+.3f}  (baseline {b_f1:.3f} → mnesis {m_f1:.3f})")

        # Clean up temporary session databases
        for db in (db_base, db_mnes):
            try:
                Path(db).unlink(missing_ok=True)
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
        print("  Note: absolute F1 values are reference only — LOCOMO questions")
        print("  use exact dates/numbers that models answer with relative phrasing")
        print("  ('yesterday' vs '7 May 2023'), scoring 0 even when correct.")
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

    # ── save raw results ───────────────────────────────────────────────
    out_json = args.output_dir / "locomo_results.json"
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
            args.output_dir / "locomo_f1_by_category.png",
        )
    plot_token_usage(token_data, args.output_dir / "locomo_token_usage.png")
    plot_summary(
        b_overall,
        m_overall,
        baseline_by_cat,
        mnesis_by_cat,
        token_data,
        compact_stats,
        args.output_dir / "locomo_summary.png",
        metrics_only=args.metrics_only,
    )

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
