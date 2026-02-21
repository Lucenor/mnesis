<p align="center">
  <img src="https://raw.githubusercontent.com/Lucenor/mnesis/main/docs/images/logo_icon.png" alt="mnesis logo" width="120"><br><br>
  <img src="https://raw.githubusercontent.com/Lucenor/mnesis/main/docs/images/logo_wordmark.png" alt="mnesis" width="320"><br><br>
  <em>Lossless Context Management for long-horizon LLM agents</em>
  <br><br>
  <a href="https://pypi.org/project/mnesis/"><img src="https://img.shields.io/pypi/v/mnesis?color=5c6bc0&labelColor=1a1a2e" alt="PyPI"></a>
  <a href="https://pypi.org/project/mnesis/"><img src="https://img.shields.io/pypi/pyversions/mnesis?color=5c6bc0&labelColor=1a1a2e" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-5c6bc0?labelColor=1a1a2e" alt="License"></a>
  <a href="https://github.com/Lucenor/mnesis/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Lucenor/mnesis/ci.yml?color=5c6bc0&labelColor=1a1a2e" alt="CI"></a>
  <a href="https://codecov.io/github/Lucenor/mnesis"><img src="https://img.shields.io/codecov/c/github/Lucenor/mnesis?color=5c6bc0&labelColor=1a1a2e" alt="Coverage"></a>
  <a href="https://mnesis.lucenor.tech"><img src="https://img.shields.io/badge/docs-mnesis.lucenor.tech-5c6bc0?labelColor=1a1a2e" alt="Docs"></a>
  <a href="https://github.com/Lucenor/mnesis/attestations"><img src="https://img.shields.io/badge/provenance-attested-5c6bc0?labelColor=1a1a2e&logo=githubactions&logoColor=white" alt="Attestation"></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/Lucenor/mnesis"><img src="https://api.scorecard.dev/projects/github.com/Lucenor/mnesis/badge" alt="OpenSSF Scorecard"></a>
</p>

---

LLMs suffer from **context rot**: accuracy degrades 30–40% before hitting nominal token limits — not because the model runs out of space, but because reasoning quality collapses as the window fills with stale content.

The standard fix — telling the model to "summarize itself" — is unreliable. The model may silently drop constraints, forget file paths, or produce a summary that is itself too large.

**mnesis** solves this by making the *engine* — not the model — responsible for memory. It is a Python implementation of the [LCM: Lossless Context Management](https://github.com/Lucenor/mnesis/blob/main/docs/LCM.pdf) architecture.

---

## Benchmarks

Evaluated on [OOLONG](https://github.com/abertsch72/oolong), a long-context reasoning and aggregation benchmark. Both LCM-managed and Claude Code agents are built on Claude Opus 4.6; the gap comes entirely from context architecture.

> The charts below compare LCM-managed context against Claude Code and unmanaged Opus 4.6 across context lengths from 8K to 1M tokens.

**Score improvement over raw Opus 4.6 at each context length:**

![OOLONG benchmark — score improvement over raw Opus 4.6](https://raw.githubusercontent.com/Lucenor/mnesis/main/docs/images/benchmark_improvement.png)

**Absolute scores vs raw Opus 4.6 baseline:**

![OOLONG benchmark — absolute scores](https://raw.githubusercontent.com/Lucenor/mnesis/main/docs/images/benchmark_scores.png)

Raw Opus 4.6 uses no context management — scores collapse past 32K tokens.

---

## How it works

Traditional agentic frameworks ("RLM" — Recursive Language Models) ask the model to manage its own context via tool calls. LCM moves that responsibility to a deterministic engine layer:

![RLM vs LCM approach](https://raw.githubusercontent.com/Lucenor/mnesis/main/docs/images/rlm_vs_lcm.png)

The engine handles memory deterministically so the model can focus entirely on the task.

---

## Key properties

| | RLM (e.g. raw Claude Code) | mnesis |
|---|---|---|
| Context trigger | Model judgment | Token threshold |
| Summarization failure | Silent data loss | Three-level fallback — never fails |
| Tool output growth | Unbounded | Backward-scan pruner |
| Large files | Inline (eats budget) | Content-addressed references |
| Parallel workloads | Sequential or ad-hoc | `LLMMap` / `AgenticMap` operators |
| History | Ephemeral | Append-only SQLite log |

---

## Quick Start

```bash
uv add mnesis
```

```python
import asyncio
from mnesis import MnesisSession

async def main():
    async with await MnesisSession.create(
        model="anthropic/claude-opus-4-6",
        system_prompt="You are a helpful assistant.",
    ) as session:
        result = await session.send("Explain the GIL in Python.")
        print(result.text)

asyncio.run(main())
```

No API key needed to try it — set `MNESIS_MOCK_LLM=1` and run any of the [examples](#examples).

### Provider support

mnesis works with any LLM provider via [litellm](https://docs.litellm.ai/). Pass the model string and set the corresponding API key:

| Provider | Model string | API key env var |
|---|---|---|
| Anthropic | `"anthropic/claude-opus-4-6"` | `ANTHROPIC_API_KEY` |
| OpenAI | `"openai/gpt-4o"` | `OPENAI_API_KEY` |
| Google Gemini | `"gemini/gemini-1.5-pro"` | `GEMINI_API_KEY` |
| OpenRouter | `"openrouter/meta-llama/llama-3.1-70b-instruct"` | `OPENROUTER_API_KEY` |

See the [Provider Configuration guide](https://mnesis.lucenor.tech/providers/) for the full provider configuration guide.

### BYO-LLM — use your own SDK

If you already use the Anthropic, OpenAI, or another SDK directly, use `session.record()` to let mnesis handle memory and compaction without routing calls through litellm:

```python
import anthropic
from mnesis import MnesisSession, TokenUsage

client = anthropic.Anthropic()
session = await MnesisSession.create(model="anthropic/claude-opus-4-6")

user_text = "Explain quantum entanglement."
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": user_text}],
)

await session.record(
    user_message=user_text,
    assistant_response=response.content[0].text,
    tokens=TokenUsage(
        input=response.usage.input_tokens,
        output=response.usage.output_tokens,
    ),
)
```

---

## Core Concepts

### Immutable Store + Active Context
Every message and tool result is appended to an SQLite log and never modified. Each turn, the engine assembles a *curated view* of the log that fits the model's token budget — the Active Context.

### Three-Level Compaction
When token usage crosses the threshold, the `CompactionEngine` escalates automatically:

1. **Level 1** — Structured LLM summary: Goal / Discoveries / Accomplished / Remaining
2. **Level 2** — Aggressive compression: drop reasoning, maximum conciseness
3. **Level 3** — Deterministic truncation: no LLM, always fits, never fails

Compaction runs asynchronously and never blocks a turn.

### Tool Output Pruning
The `ToolOutputPruner` scans backward through history and tombstones completed tool outputs that fall outside a configurable protect window (default 40K tokens). Tombstoned outputs are replaced by a compact marker in the context — the data is still in the immutable store.

### Large File References
Files exceeding the inline threshold (default 10K tokens) are stored externally as `FileRefPart` objects with structural exploration summaries — AST outlines for Python, schema keys for JSON/YAML, headings for Markdown. The file is never re-read unless the model explicitly requests it.

### Parallel Operators
- **`LLMMap`** — stateless parallel LLM calls over a list of inputs with Pydantic schema validation and per-item retry. O(1) context cost to the caller.
- **`AgenticMap`** — independent sub-agent sessions per input item, each with full multi-turn reasoning. The parent session sees only the final output.

---

## Configuration

```python
from mnesis import MnesisConfig, CompactionConfig, FileConfig, SessionConfig

config = MnesisConfig(
    compaction=CompactionConfig(
        auto=True,
        compaction_output_budget=20_000,  # tokens reserved for compaction output
        prune=True,
        prune_protect_tokens=40_000,      # never prune within last 40K tokens
        prune_minimum_tokens=20_000,      # skip pruning if volume is too small
        level2_enabled=True,
    ),
    file=FileConfig(
        inline_threshold=10_000,          # files > 10K tokens → FileRefPart
    ),
    session=SessionConfig(
        doom_loop_threshold=3,            # consecutive identical tool calls before warning
    ),
)
```

| Parameter | Default | Description |
|---|---|---|
| `compaction.auto` | `True` | Auto-trigger on overflow |
| `compaction.compaction_output_budget` | `20,000` | Tokens reserved for summary output |
| `compaction.prune_protect_tokens` | `40,000` | Recent tokens never pruned |
| `compaction.prune_minimum_tokens` | `20,000` | Minimum prunable volume |
| `file.inline_threshold` | `10,000` | Inline limit in tokens |
| `operators.llm_map_concurrency` | `16` | `LLMMap` parallel calls |
| `operators.agentic_map_concurrency` | `4` | `AgenticMap` parallel sessions |
| `session.doom_loop_threshold` | `3` | Identical tool call threshold |

---

## Examples

All examples run without an API key via `MNESIS_MOCK_LLM=1`:

```bash
MNESIS_MOCK_LLM=1 uv run python examples/01_basic_session.py
MNESIS_MOCK_LLM=1 uv run python examples/05_parallel_processing.py
```

| File | Demonstrates |
|---|---|
| `examples/01_basic_session.py` | `create()`, `send()`, context manager, compaction monitoring |
| `examples/02_long_running_agent.py` | EventBus subscriptions, streaming callbacks, manual `compact()` |
| `examples/03_tool_use.py` | Tool lifecycle, `ToolPart` streaming states, tombstone inspection |
| `examples/04_large_files.py` | `LargeFileHandler`, `FileRefPart`, cache hits, exploration summaries |
| `examples/05_parallel_processing.py` | `LLMMap` with Pydantic schema, `AgenticMap` sub-sessions |
| `examples/06_byo_llm.py` | `record()` — BYO-LLM, inject turns from your own SDK |

---

## Documentation

Full documentation is available at **[mnesis.lucenor.tech](https://mnesis.lucenor.tech)**, including:

- [Getting Started](https://mnesis.lucenor.tech/getting-started/)
- [Provider Configuration](https://mnesis.lucenor.tech/providers/)
- [BYO-LLM](https://mnesis.lucenor.tech/byo-llm/)
- [Concepts](https://mnesis.lucenor.tech/concepts/)
- [API Reference](https://mnesis.lucenor.tech/api/)

---

## Architecture

```
MnesisSession
├── ImmutableStore         (SQLite append-only log)
├── SummaryDAGStore        (logical DAG over is_summary messages)
├── ContextBuilder         (assembles LLM message list each turn)
├── CompactionEngine       (three-level escalation + atomic commit)
│   ├── ToolOutputPruner   (backward-scanning tombstone pruner)
│   └── levels.py          (level 1 / 2 / 3 functions)
├── LargeFileHandler       (content-addressed file references)
├── TokenEstimator         (tiktoken + heuristic fallback)
└── EventBus               (in-process pub/sub)

Operators (independent of session)
├── LLMMap                 (parallel stateless LLM calls)
└── AgenticMap             (parallel sub-agent sessions)
```

---

## Contributing

Contributions are welcome — bug reports, feature requests, and pull requests alike.

See [CONTRIBUTING.md](https://github.com/Lucenor/mnesis/blob/main/CONTRIBUTING.md) for the full guide. The short version:

```bash
# Clone and install with dev dependencies
git clone https://github.com/Lucenor/mnesis.git
cd mnesis
uv sync --group dev

# Lint + format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/mnesis

# Test (with coverage)
uv run pytest

# Run examples without API keys
MNESIS_MOCK_LLM=1 uv run python examples/01_basic_session.py
MNESIS_MOCK_LLM=1 uv run python examples/05_parallel_processing.py

# Build docs locally
uv sync --group docs
uv run mkdocs serve
```

Open an issue before starting large changes — it avoids duplicated effort.

---

## License

Apache 2.0 — see [LICENSE](https://github.com/Lucenor/mnesis/blob/main/LICENSE) and [NOTICE](https://github.com/Lucenor/mnesis/blob/main/NOTICE).
