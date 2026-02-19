# Configuration

All configuration is done through `MnesisConfig`, which groups settings into sub-configs. Every field has a sensible default — you only need to override what you want to change.

```python
from mnesis import MnesisSession, MnesisConfig, CompactionConfig, FileConfig
from mnesis.models.config import StoreConfig, OperatorConfig

config = MnesisConfig(
    compaction=CompactionConfig(...),
    file=FileConfig(...),
    store=StoreConfig(...),
    operators=OperatorConfig(...),
    doom_loop_threshold=3,
)

session = await MnesisSession.create(model="openai/gpt-4o", config=config)
```

---

## CompactionConfig

Controls when and how context compaction fires.

| Field | Default | Description |
|---|---|---|
| `auto` | `True` | Auto-trigger compaction on overflow |
| `buffer` | `20_000` | Tokens reserved as headroom for compaction summary output |
| `prune` | `True` | Run tool output pruning before compaction |
| `prune_protect_tokens` | `40_000` | Token window from the end of history that is never pruned |
| `prune_minimum_tokens` | `20_000` | Minimum prunable volume required before pruning fires |
| `compaction_model` | `None` | Model for summarisation. `None` = use session model |
| `level2_enabled` | `True` | Attempt Level 2 compression before falling back to Level 3 |
| `plugin_hook` | `None` | Dotted import path to a callable returning a custom compaction prompt |

### Tuning for large models

For models with 1M+ token contexts (e.g. Gemini 1.5 Pro), raise the buffer and protect window:

```python
CompactionConfig(
    buffer=100_000,
    prune_protect_tokens=200_000,
    prune_minimum_tokens=50_000,
)
```

### Custom compaction prompt

```python
# myapp/hooks.py
def custom_prompt(messages: list, config: dict) -> str:
    return "Summarise this conversation focusing on technical decisions only."

# Configuration
CompactionConfig(plugin_hook="myapp.hooks.custom_prompt")
```

---

## FileConfig

Controls how large files are handled.

| Field | Default | Description |
|---|---|---|
| `inline_threshold` | `10_000` | Files estimated above this token count are stored as `FileRefPart` objects |
| `storage_dir` | `None` | Directory for external file storage. Defaults to `~/.mnesis/files/` |
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

## Top-level MnesisConfig

| Field | Default | Description |
|---|---|---|
| `doom_loop_threshold` | `3` | Consecutive identical tool calls before `DOOM_LOOP_DETECTED` fires |
