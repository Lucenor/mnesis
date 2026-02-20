# Mnesis Benchmarks

## LOCOMO — Long-Context Conversational Memory

[LOCOMO](https://arxiv.org/abs/2402.17753) is an academic benchmark that tests whether a
model can recall facts from very long multi-session conversations (~300 turns, ~9 K tokens
each, across up to 35 sessions per conversation).

This benchmark evaluates the quality of **mnesis context compaction**: after injecting a
full LOCOMO conversation and running `compact()`, can the model still answer questions about
that conversation accurately?

---

### What it measures

Two conditions are compared for each conversation:

| Condition | Description |
|-----------|-------------|
| **Baseline** | Full conversation history kept in context; no compaction |
| **Mnesis** | Full history injected via `session.record()`, then `session.compact()` called before QA |

The primary metric is **token-level F1** (SQuAD-style partial match), broken down by
question category. The difference in F1 between conditions shows the quality cost (or gain)
of compaction.

Token usage before and after compaction is always measured, even without an LLM API key.

---

### Question categories

| # | Category | What it tests |
|---|----------|---------------|
| 1 | Single-Hop | Fact retrievable from a single session |
| 2 | Multi-Hop | Requires synthesising information across sessions |
| 3 | Temporal | Sequence or ordering of events |
| 4 | Open-Domain | Conversation facts combined with world knowledge |
| 5 | Adversarial | Designed to elicit incorrect answers |

Human F1 baseline from the paper: **87.9%**

---

### Prerequisites

**Python packages** (not included in mnesis's core dependencies):

```bash
pip install matplotlib numpy
# or
uv add --dev matplotlib numpy
```

**LLM API key** — required for QA evaluation, not for `--metrics-only`:

```bash
export ANTHROPIC_API_KEY=sk-...
# Other providers work too (OpenAI, Google, etc.) via litellm
```

---

### Usage

**Dry-run — no API key needed, token/compaction stats only:**

```bash
uv run python benchmarks/locomo.py --metrics-only
```

**Quick evaluation — 1 conversation, 20 questions (~$0.10–$0.50 with claude-haiku-4-5):**

```bash
ANTHROPIC_API_KEY=sk-... uv run python benchmarks/locomo.py \
    --model anthropic/claude-haiku-4-5 \
    --conversations 1 \
    --questions-per 20
```

**Full evaluation — all 10 conversations (~$2–$10 with claude-haiku-4-5):**

```bash
ANTHROPIC_API_KEY=sk-... uv run python benchmarks/locomo.py \
    --model anthropic/claude-haiku-4-5 \
    --conversations 10 \
    --questions-per 100
```

**Restrict to one question category:**

```bash
# Only temporal reasoning questions
ANTHROPIC_API_KEY=sk-... uv run python benchmarks/locomo.py \
    --category 3 \
    --conversations 3
```

**Use a local copy of the dataset (skips download):**

```bash
uv run python benchmarks/locomo.py \
    --data /path/to/locomo10.json \
    --metrics-only
```

---

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `anthropic/claude-haiku-4-5` | LiteLLM model string |
| `--conversations` | `1` | Conversations to evaluate (1–10) |
| `--questions-per` | `20` | Max QA pairs per conversation |
| `--category` | all | Restrict to one category (1–5) |
| `--data` | auto-downloaded | Path to `locomo10.json` |
| `--output-dir` | `benchmarks/results/` | Output directory for PNGs and JSON |
| `--metrics-only` | off | Skip QA inference; token/compaction stats only |

---

### Outputs

All files are written to `benchmarks/results/` (or `--output-dir`).

| File | When produced | Description |
|------|--------------|-------------|
| `locomo_f1_by_category.png` | Full QA mode only | Grouped bar chart: F1 per question category, baseline vs mnesis. Human baseline shown as a dashed line. |
| `locomo_token_usage.png` | Always | Token count before and after compaction per conversation, with percentage reduction labels. |
| `locomo_summary.png` | Always | 2×2 dashboard: overall F1, category F1, token usage, and a compaction summary table. F1 panels show N/A in `--metrics-only` mode. |
| `locomo_results.json` | Always | Raw per-question results, per-category F1, token statistics, and compaction metadata. |

The dataset (`locomo10.json`) is downloaded once to `benchmarks/data/` and reused on
subsequent runs.

---

### Interpreting results

| Observation | Likely meaning |
|-------------|----------------|
| Small F1 gap between baseline and mnesis | Compaction is lossless for this workload |
| Large F1 drop after compaction | Key facts are being over-aggressively summarised; consider tuning `prune_protect_tokens` |
| Token reduction > 50% | Compaction is effective at recovering context budget |
| Compaction level 1 | Mnesis used LLM-based structured summarisation (highest quality) |
| Compaction level 3 | Fell back to deterministic truncation — check your API key and model string |
| High F1 on Single-Hop, low on Temporal | Normal pattern; temporal reasoning is the hardest category for all models |

---

### Cost estimate

Each conversation is evaluated **twice** (baseline + mnesis), so token costs double compared
to a single-pass evaluation. Conversation injection (~150 `record()` calls per conversation)
adds negligible cost because `record()` does not call the LLM. Only QA questions use the LLM.

| Configuration | Approx cost |
|--------------|-------------|
| `--metrics-only` | Free |
| 1 conv, 20 questions, claude-haiku-4-5 | ~$0.10–$0.50 |
| 10 conv, 100 questions, claude-haiku-4-5 | ~$2–$10 |
| 10 conv, 100 questions, claude-sonnet-4-6 | ~$15–$50 |

Costs are rough estimates and depend on conversation length and answer verbosity.

---

### Dataset

The LOCOMO dataset is downloaded automatically from the
[official repository](https://github.com/snap-research/locomo).
It contains 10 curated long-term conversations averaging 300 turns and 9 K tokens each,
with human-verified QA annotations across 5 reasoning categories.

Citation:

```bibtex
@inproceedings{maharana2024evaluating,
  title={Evaluating Very Long-Term Conversational Memory of LLM Agents},
  author={Maharana, Adyasha and Lee, Dong-Ho and Tulyakov, Sergey and
          Bansal, Mohit and Barbieri, Francesco and Fang, Yuwei},
  booktitle={ACL},
  year={2024}
}
```
