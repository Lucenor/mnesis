# mnesis — Lossless Context Management

A Python library that solves context window degradation in long-running LLM agents by moving memory management out of the model layer and into a deterministic engine.

## Why mnesis?

LLMs suffer from **context rot**: accuracy degrades 30-40% before hitting nominal token limits, not because the model runs out of space, but because reasoning quality degrades as the window fills with irrelevant content.

The standard approach — telling the model to "summarize itself" — is unreliable. The model may forget critical constraints, silently drop file paths, or produce a summary that is itself too large.

`mnesis` solves this by making the engine — not the model — responsible for memory:

- **Immutable store**: Every message is preserved forever in an append-only SQLite log.
- **Deterministic triggers**: Token thresholds (not model judgment) decide when to compact.
- **Three-level escalation**: Structured LLM summary → aggressive LLM summary → deterministic fallback. Compaction never fails.
- **Tool output pruning**: Stale tool outputs are tombstoned automatically, not summarized.
- **Large file references**: Files exceeding a token threshold are stored externally with exploration summaries.

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

## Core Concepts

### Dual-State Memory Architecture
The **Immutable Store** is an append-only SQLite log containing every message and part ever produced. The **Active Context** is the ephemeral window assembled on each turn — a curated view of the store that fits within the model's token budget.

### Automatic Compaction
When token usage crosses the usable threshold, the `CompactionEngine` runs a three-level escalation:
1. **Level 1**: Structured LLM summarization (Goal / Discoveries / Accomplished / Remaining)
2. **Level 2**: Aggressive compression (drop reasoning, max conciseness)
3. **Level 3**: Deterministic truncation (no LLM — always fits)

Compaction runs asynchronously and never blocks a turn.

### Tool Output Pruning
The `ToolOutputPruner` scans backward through the history, tombstoning completed tool outputs that fall outside a protect window (default 40K tokens from the end). Tombstoned outputs are replaced by a compact marker in the context.

### Large File References
Files exceeding the inline threshold (default 10K tokens) are stored externally and represented as `FileRefPart` objects. The `LargeFileHandler` generates structural exploration summaries using AST parsing (Python), schema inspection (JSON/YAML/TOML), or heading extraction (Markdown).

### Parallel Operators
- **`LLMMap`**: Stateless parallel LLM calls over a list of items. O(1) context cost to the caller.
- **`AgenticMap`**: Independent sub-agent sessions per item, each with full multi-turn reasoning. Parent sees only the final output.

## Configuration

```python
from mnesis import MnesisConfig, CompactionConfig, FileConfig

config = MnesisConfig(
    compaction=CompactionConfig(
        auto=True,
        buffer=20_000,                      # Tokens reserved for compaction output
        prune=True,
        prune_protect_tokens=40_000,        # Never prune within last 40K tokens
        prune_minimum_tokens=20_000,        # Min volume before pruning is worthwhile
        compaction_model=None,              # None = use session model
        level2_enabled=True,
    ),
    file=FileConfig(
        inline_threshold=10_000,            # Files > 10K tokens → FileRefPart
    ),
    doom_loop_threshold=3,                  # Consecutive identical tool calls before warning
)
```

| Parameter | Default | Description |
|---|---|---|
| `compaction.auto` | `True` | Auto-trigger on overflow |
| `compaction.buffer` | `20,000` | Tokens reserved for summary output |
| `compaction.prune_protect_tokens` | `40,000` | Recent tokens never pruned |
| `compaction.prune_minimum_tokens` | `20,000` | Minimum prunable volume |
| `file.inline_threshold` | `10,000` | Inline limit in tokens |
| `operators.llm_map_concurrency` | `16` | LLMMap parallel calls |
| `operators.agentic_map_concurrency` | `4` | AgenticMap parallel sessions |
| `doom_loop_threshold` | `3` | Identical tool call threshold |

## Examples

| File | Demonstrates |
|---|---|
| `examples/01_basic_session.py` | `create()`, `send()`, context manager, compaction monitoring |
| `examples/02_long_running_agent.py` | EventBus subscriptions, streaming, manual `compact()` |
| `examples/03_tool_use.py` | Tool lifecycle, `ToolPart` streaming, tombstone inspection |
| `examples/04_large_files.py` | `LargeFileHandler`, `FileRefPart`, cache hits, exploration summaries |
| `examples/05_parallel_processing.py` | `LLMMap` with Pydantic schema, `AgenticMap` sub-sessions |

All examples work without an API key:

```bash
MNESIS_MOCK_LLM=1 uv run python examples/01_basic_session.py
```

## API Reference

See [docs/api.md](docs/api.md) for the full API reference.

## Architecture

```
MnesisSession
├── ImmutableStore         (SQLite append-only log)
├── SummaryDAGStore        (logical DAG over is_summary messages)
├── ContextBuilder         (assembles LLM message list each turn)
├── CompactionEngine       (three-level escalation + atomic commit)
│   ├── ToolOutputPruner   (backward-scanning tombstone pruner)
│   └── levels.py          (level1 / level2 / level3 functions)
├── LargeFileHandler       (content-addressed file references)
├── TokenEstimator         (tiktoken + heuristic fallback)
└── EventBus               (in-process pub/sub)

Operators (independent of session)
├── LLMMap                 (parallel stateless LLM calls)
└── AgenticMap             (parallel sub-agent sessions)
```

## Contributing

Contributions are welcome — bug reports, feature requests, and pull requests alike.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. The short version:

```bash
# Clone and install with dev dependencies
git clone https://github.com/Lucenor/mnesis.git
cd mnesis
uv sync --dev

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
```

Open an issue before starting large changes — it avoids duplicated effort.

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
