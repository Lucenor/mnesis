# Configuration

All configuration is done through `MnesisConfig`, which groups settings into sub-configs. Every field has a sensible default — you only need to override what you want to change.

```python
from mnesis import MnesisSession, MnesisConfig, CompactionConfig, FileConfig, StoreConfig, OperatorConfig

config = MnesisConfig(
    compaction=CompactionConfig(...),
    file=FileConfig(...),
    store=StoreConfig(...),
    operators=OperatorConfig(...),
)

session = await MnesisSession.create(model="openai/gpt-4o", config=config)
```

---

## CompactionConfig

Controls when and how context compaction fires.

| Field | Default | Description |
|---|---|---|
| `auto` | `True` | Auto-trigger compaction on overflow |
| `compaction_output_budget` | `20_000` | Tokens reserved as headroom for compaction summary output |
| `prune` | `True` | Run tool output pruning before compaction |
| `prune_protect_tokens` | `40_000` | Token window from the end of history that is never pruned |
| `prune_minimum_tokens` | `20_000` | Minimum prunable volume required before pruning fires |
| `compaction_model` | `None` | Model for summarisation. `None` = use session model |
| `level2_enabled` | `True` | Attempt Level 2 compression before falling back to Level 3 |
| `compaction_prompt` | `None` | Custom prompt string for Level 1/2 LLM summarisation. `None` = use the built-in agentic prompt |

### Tuning for large models

For models with 1M+ token contexts (e.g. Gemini 1.5 Pro), raise the budget and protect window:

```python
CompactionConfig(
    compaction_output_budget=100_000,
    prune_protect_tokens=200_000,
    prune_minimum_tokens=50_000,
)
```

### Custom compaction prompt

```python
CompactionConfig(
    compaction_prompt="Summarise this conversation focusing on technical decisions only.",
)
```

---

## FileConfig

Controls how large files are handled.

| Field | Default | Description |
|---|---|---|
| `inline_threshold` | `10_000` | Files estimated above this token count are stored as `FileRefPart` objects |
| `storage_dir` | `~/.mnesis/files/` | Directory for external file storage. Defaults to `~/.mnesis/files/` |
| `exploration_summary_model` | `None` | Model for LLM-based file summaries. `None` = deterministic only |

---

## StoreConfig

Controls the SQLite persistence layer.

| Field | Default | Description |
|---|---|---|
| `db_path` | `~/.mnesis/sessions.db` | Path to the SQLite database file (`~` is expanded at runtime) |
| `wal_mode` | `True` | Use WAL journal mode for better concurrent read performance |
| `connection_timeout` | `30.0` | Seconds to wait for the database connection |

---

## OperatorConfig

Controls `LLMMap` and `AgenticMap` parallelism.

| Field | Default | Description |
|---|---|---|
| `llm_map_concurrency` | `16` | Maximum concurrent LLM calls in `LLMMap.run()` |
| `agentic_map_concurrency` | `4` | Maximum concurrent sub-agent sessions in `AgenticMap.run()` |
| `max_retries` | `3` | Per-item retry attempts on validation or transient errors |

---

## SessionConfig

Controls session-level behaviour.

| Field | Default | Description |
|---|---|---|
| `doom_loop_threshold` | `3` | Consecutive identical tool calls before `DOOM_LOOP_DETECTED` fires |
