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
| `soft_threshold_fraction` | `0.6` | Fraction of usable context at which background compaction triggers (before hard threshold). Advanced. |
| `max_compaction_rounds` | `10` | Cap on summarise+condense cycles in multi-round loop. Advanced. |
| `condensation_enabled` | `True` | Whether to attempt condensation of accumulated summary nodes. Advanced. |

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
| `exploration_summary_model` | `None` | Reserved for future LLM-based structural summaries (AST, key lists, headings). Currently ignored — structural exploration summaries are generated deterministically only. |

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
| `retry` | `RetryConfig()` | Automatic retry configuration for transient LLM errors in `send()` |

---

## RetryConfig

Controls automatic retry of transient LLM errors inside `send()`. Retry is **opt-in**: the default `max_retries=0` preserves the pre-0.3.0 behaviour of failing immediately.

| Field | Default | Description |
|---|---|---|
| `max_retries` | `0` | Maximum retry attempts. `0` disables retry entirely |
| `base_delay` | `1.0` | Base delay in seconds for exponential backoff |
| `max_delay` | `60.0` | Maximum delay cap in seconds |
| `jitter` | `True` | Add random jitter (sampled from `[0, base_delay)`) to spread retries |

**Backoff formula:** `min(base_delay × 2^(attempt-1) + jitter, max_delay)`

where `attempt` is the 1-based retry number matching `LlmRetryPayload.attempt` (first retry = 1, so the first backoff = `base_delay × 2^0 = base_delay`).

### Retryable errors

Only transient provider-side errors are retried:

| litellm exception | HTTP status | Scenario |
|---|---|---|
| `RateLimitError` | 429 | Provider rate limit |
| `InternalServerError` | 500 | Provider server error |
| `ServiceUnavailableError` | 503 | Provider temporarily down |
| `Timeout` | — | Request timed out |
| `APIConnectionError` | — | Network connection failed |

All other exceptions (including `AuthenticationError`, `ContextWindowExceededError`, `BadRequestError`) fail immediately without retry — they indicate caller mistakes that retrying will not fix.

### litellm num_retries interaction

Mnesis explicitly passes `num_retries=0` to `litellm.acompletion()` to disable litellm's own built-in retry mechanism. All retry logic is owned by Mnesis via `RetryConfig`, so there is no double-retrying. Do not set `num_retries` in `call_kwargs` passed to litellm alongside Mnesis.

### Difference from OperatorConfig.max_retries

`OperatorConfig.max_retries` handles per-item validation errors inside `LLMMap` and `AgenticMap` operators — a different failure domain (schema validation, JSON parse failures). `RetryConfig` handles transient LLM *transport* errors in the `send()` call path. Both can be set independently.

### AgenticMap sub-sessions

Each `AgenticMap` sub-session has its own independent `RetryConfig`. The effective maximum LLM calls per sub-agent is `(max_retries + 1) × max_turns`. Plan capacity accordingly.

### Example

```python
from mnesis import MnesisSession, MnesisConfig
from mnesis.models.config import SessionConfig, RetryConfig

config = MnesisConfig(
    session=SessionConfig(
        retry=RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            jitter=True,
        )
    )
)

async with MnesisSession.open(model="openai/gpt-4o", config=config) as session:
    # send() will now automatically retry up to 3 times on rate limits,
    # server errors, timeouts, and connection failures.
    result = await session.send("Hello!")
```

### Monitoring retries via the event bus

Each retry attempt publishes an `LLM_RETRY` event before the backoff sleep:

```python
from mnesis.events.bus import MnesisEvent
from mnesis.events.payloads import LlmRetryPayload

def on_retry(event: MnesisEvent, payload: LlmRetryPayload) -> None:
    print(
        f"Retry {payload['attempt']}/{payload['max_retries']}: "
        f"{payload['error_type']} — sleeping {payload['delay_seconds']:.1f}s"
    )

session.event_bus.subscribe(MnesisEvent.LLM_RETRY, on_retry)  # type: ignore[arg-type]
```

---

## model_overrides

`MnesisConfig.model_overrides` lets you correct or override the context and
output token limits that Mnesis auto-detects from the model string. This is
useful when you are using a fine-tuned model, a custom deployment, or a model
that litellm does not yet know about.

**Supported keys:**

| Key | Type | Description |
|---|---|---|
| `context_limit` | `int` | Total input + output token limit for the model |
| `max_output_tokens` | `int` | Maximum tokens the model can generate per response |

### Example — custom or fine-tuned model

```python
from mnesis import MnesisSession, MnesisConfig

config = MnesisConfig(
    model_overrides={
        "context_limit": 128_000,
        "max_output_tokens": 16_384,
    }
)

async with MnesisSession.open(model="openai/my-finetuned-gpt4o", config=config) as session:
    result = await session.send("Hello!")
    print(result.text)
```

### Example — correcting an underestimated limit

If Mnesis picks a conservative default for a newly released model, override it
without touching any other configuration:

```python
config = MnesisConfig(
    model_overrides={"context_limit": 1_000_000},  # Gemini 1.5 Pro 1M variant
)
```

The overrides are applied after `ModelInfo.from_model_string()` resolves the
base limits, so only the keys you specify are changed — the rest (encoding,
provider, etc.) are inferred normally. Both `context_limit` and
`max_output_tokens` affect how Mnesis sizes the compaction budget, so incorrect
values can cause over-limit contexts to reach the provider or leave headroom
unused. Always set them to match the model's true limits.

`model_overrides` applies to the session model only. If you configure a
separate `compaction.compaction_model`, overrides are not applied to it — the
compaction model's limits are always auto-detected from the model string.

!!! note
    `model_overrides` only affects how Mnesis allocates the context budget and
    compaction thresholds — it does not change how litellm routes the request.
    You still need to configure litellm (API base, headers, etc.) separately
    for non-standard endpoints. See [LLM Providers](providers.md) for the full
    list of supported providers and model string formats.
