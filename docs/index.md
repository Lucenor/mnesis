# mnesis

**Lossless Context Management for long-horizon LLM agents.**

LLMs suffer from **context rot**: accuracy degrades 30–40% before hitting nominal token limits — not because the model runs out of space, but because reasoning quality collapses as the window fills with stale content.

The standard fix — telling the model to "summarize itself" — is unreliable. The model may silently drop constraints, forget file paths, or produce a summary that is itself too large.

**mnesis** solves this by making the *engine* — not the model — responsible for memory. It is a Python implementation of the [LCM: Lossless Context Management](https://github.com/Lucenor/mnesis/blob/main/docs/LCM.pdf) architecture.

---

## Benchmarks

Evaluated on [OOLONG](https://github.com/abertsch72/oolong), a long-context reasoning and aggregation benchmark across context lengths from 8K to 1M tokens.

![OOLONG benchmark — score improvement over raw Opus 4.6](images/benchmark_improvement.png)

![OOLONG benchmark — absolute scores](images/benchmark_scores.png)

Raw Opus 4.6 uses no context management — scores collapse past 32K tokens.

---

## Key properties

|  | RLM (e.g. raw Claude Code) | mnesis |
|---|---|---|
| Context trigger | Model judgment | Token threshold |
| Summarization failure | Silent data loss | Three-level fallback — never fails |
| Tool output growth | Unbounded | Backward-scan pruner |
| Large files | Inline (eats budget) | Content-addressed references |
| Parallel workloads | Sequential or ad-hoc | `LLMMap` / `AgenticMap` operators |
| History | Ephemeral | Append-only SQLite log |

---

## Quick install

```bash
uv add mnesis
# or
pip install mnesis
```

Jump to [Getting Started](getting-started.md) for a full walkthrough.
