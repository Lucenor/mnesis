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
| `compaction_output_budget` | `20_000` | Tokens reserved as headroom for compaction summary output. Range 1,000-100,000. Usable context is `context_limit - max_output_tokens - compaction_output_budget`, and summaries (including the Level 3 fallback) are sized against it, so it must be well below your model's window. Session creation/loading raises `ValueError` when the budget leaves no usable context (`context_limit - max_output_tokens <= compaction_output_budget`, e.g. a small `model_overrides` window) and logs a `compaction_output_budget_exceeds_usable_window` warning when the budget is at least as large as the usable window. |
| `prune` | `True` | Run tool output pruning before compaction |
| `prune_protect_tokens` | `40_000` | Token window from the end of history that is never pruned |
| `prune_minimum_tokens` | `20_000` | Minimum prunable volume required before pruning fires |
| `compaction_model` | `None` | Model for summarisation. `None` = use session model |
| `level2_enabled` | `True` | Attempt Level 2 compression before falling back to Level 3 |
| `compaction_prompt` | `None` | Custom prompt string for Level 1/2 LLM summarisation. `None` = use the built-in agentic prompt |
| `soft_threshold_fraction` | `0.6` | Fraction of usable context at which background compaction triggers (before hard threshold). Measured against the size of the current context window, not lifetime token usage. Also sets condensation's stop target: after summarising, summaries are condensed until the context is below half this threshold (`soft_threshold_fraction * 0.5` of usable; the 0.5 is not configurable). Advanced. |
| `max_compaction_rounds` | `10` | Upper bound on condensation rounds per run, and on summarisation passes when the summariser's input cap (75% of the compaction model's window) forces several passes. Each condensation round merges all live summary nodes into one, so a run condenses at most once in practice. Advanced. |
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

### Small windows: when session creation raises

`MnesisSession.create()` and `load()` raise `ValueError` whenever
`context_limit - max_output_tokens <= compaction_output_budget`, whether the
budget was set explicitly or left at the default (20,000). **This is a breaking
change for `model_overrides` users with small windows.** Those configs were
already unusable before the check: no history fit next to the reserved headroom,
so `context_for_next_turn()` returned a single message after four turns and
compaction ran and blocked every turn. Mnesis now fails fast instead.

Overrides that raise with the default budget (`max_output_tokens` is inherited
from the model string unless you set it):

| Model | `model_overrides` | `max_output_tokens` | Usable (`limit - out - 20,000`) |
|---|---|---|---|
| `ollama/llama3` | `context_limit=8_192` | inherited | negative |
| `ollama/llama3` | `context_limit=8_192, max_output_tokens=2_048` | 2,048 | -13,856 |
| `ollama/llama3` | `context_limit=16_384` | inherited | negative |
| `gpt-4o` | `context_limit=32_768` | 16,384 (inherited) | -3,616 |
| `anthropic/claude-opus-4-6` | `context_limit=50_000` | 32,000 (inherited) | -2,000 |
| `o3-mini` | `context_limit=120_000` | 100,000 (inherited) | 0 |

Fix either side of the inequality: lower `compaction_output_budget` (it only
needs to cover the summary the compaction model writes, so a few thousand
tokens suit a small window), or set `max_output_tokens` in `model_overrides`
to the reply length you actually need instead of inheriting the model's maximum:

```python
MnesisConfig(
    model_overrides={"context_limit": 32_768, "max_output_tokens": 4_096},
    compaction=CompactionConfig(compaction_output_budget=4_000),
)
```

A run that blocks `send()` at the hard limit stops once the context is under
the soft threshold (leaving the rest for the background run; a bounded extra run
covers a per-turn `system_prompt` that is larger than the session prompt); a background run can make up to `2 * max_compaction_rounds` sequential
summariser calls in the worst case (20 with the default), typically 1 to 2.

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

`send()` always passes `num_retries=0` to `litellm.acompletion()`. For compaction calls: with `RetryConfig.max_retries > 0`, Mnesis retries compaction calls and disables LiteLLM retries (`num_retries=0`) to avoid double-retrying; with `max_retries == 0` (default), compaction uses LiteLLM/provider default retries. Do not set `num_retries` in `call_kwargs` passed to litellm alongside Mnesis.

With `max_retries > 0`, compaction calls (summarisation and condensation, Levels 1 and 2) use the same attempts, backoff and error classification as `send()`: a transient 429/5xx is retried at the same level before escalating, instead of dropping straight to a lossier level. If a call outlasts its retries on a retryable error (an outage), the rest of that compaction run skips its remaining LLM levels and falls to the deterministic level (or skips condensation), so a run sleeps through at most `max_retries` backoffs of up to `max_delay` each, not one sequence per level. Non-retryable failures (auth errors, empty or truncated completions) still escalate level by level. Because a hard-limit `send()` waits for compaction, it can wait through that time, and so can `session.close()`, which waits for an in-flight compaction including its retry backoffs. To bound the wait, keep `max_retries` and `max_delay` small, or cancel the awaiting task (external cancellation propagates). The `abort` event is an engine-level parameter for callers driving `run_compaction()` or `check_and_trigger()` directly; a session does not set it.

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

async with MnesisSession.open(model="openai/acme-support-ft-v1", config=config) as session:
    result = await session.send("Hello!")
    print(result.text)
```

### Example — correcting an underestimated limit

If Mnesis falls back to conservative defaults for a model it does not recognise,
override just the field that is wrong without touching any other configuration:

```python
config = MnesisConfig(
    model_overrides={"max_output_tokens": 32_768},  # provider raised the output limit
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
