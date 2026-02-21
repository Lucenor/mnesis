# Operators

Operators are standalone parallel-processing primitives that work independently
of any particular `MnesisSession`. They live in `src/mnesis/operators/` and are
importable directly from the top-level `mnesis` package.

Two operators are provided:

| Operator | Use case |
|---|---|
| `LLMMap` | Stateless single-turn extraction/transformation over a list |
| `AgenticMap` | Multi-turn sub-agent reasoning over a list |

---

## LLMMap

**Source:** `src/mnesis/operators/llm_map.py`

`LLMMap` is a stateless parallel LLM processor. Each input item receives one
independent LLM call. There is no session state, no `ImmutableStore`, and no
compaction — everything is purely in memory. It is the right tool when:

- You need to extract structured data from many items in parallel.
- Each item can be processed in a single turn with no back-and-forth.
- You want typed output validated against a Pydantic model or JSON Schema.

### Full example with Pydantic schema

```python
import asyncio
from pydantic import BaseModel
from mnesis import LLMMap, MnesisConfig, OperatorConfig

class DocumentMetadata(BaseModel):
    title: str
    language: str
    summary: str
    keywords: list[str]

async def main():
    documents = [
        "The mitochondria is the powerhouse of the cell...",
        "Le chat est sur le tapis. Il regarde par la fenêtre...",
        "In Q3 2024 revenue grew 14% year-over-year to $2.4B...",
    ]

    llm_map = LLMMap(
        config=OperatorConfig(llm_map_concurrency=8, max_retries=2),
        model="anthropic/claude-haiku-3-5",
    )

    async for result in llm_map.run(
        inputs=documents,
        prompt_template=(
            "Extract metadata from this document and return JSON.\n\n"
            "Document:\n{{ item }}"
        ),
        output_schema=DocumentMetadata,
        temperature=0.0,
        timeout_secs=30.0,
    ):
        if result.success:
            meta: DocumentMetadata = result.output
            print(f"Title: {meta.title}, Language: {meta.language}")
            print(f"Keywords: {', '.join(meta.keywords)}")
        else:
            print(f"Failed ({result.error_kind}): {result.error}")
            print(f"Attempted {result.attempts} time(s)")
```

### Collecting all results at once

If you do not need streaming, use `run_all()` which returns a `MapBatch`:

```python
batch = await llm_map.run_all(
    inputs=documents,
    prompt_template="Translate to Spanish: {{ item }}",
    output_schema={"type": "object", "properties": {"translation": {"type": "string"}}, "required": ["translation"]},
    model="openai/gpt-4o-mini",
)

print(f"{len(batch.successes)} succeeded, {len(batch.failures)} failed")
print(f"Total attempts across all items: {batch.total_attempts}")

for result in batch.successes:
    print(result.output)
```

### Template requirements

The `prompt_template` argument is a Jinja2 template string. It **must**
reference `{{ item }}` — validation uses the Jinja2 AST parser, so complex
expressions like `{{ item['key'] }}`, `{{ item | upper }}`, and
`{% for x in item %}` are all accepted. If `item` is absent, `ValueError` is
raised before any tasks are spawned.

```python
# Valid: references item via attribute access
template = "Summarize this article:\n\nTitle: {{ item['title'] }}\nBody: {{ item['body'] }}"

# Valid: uses item in a filter
template = "Classify the sentiment of: {{ item | truncate(500) }}"

# Invalid: raises ValueError immediately
template = "Summarize this: {{ document }}"
```

### Output schema

`output_schema` accepts either a Pydantic `BaseModel` subclass or a plain JSON
Schema dict:

**Pydantic (recommended):**
```python
class Result(BaseModel):
    score: int
    label: str

async for r in llm_map.run(inputs=..., output_schema=Result, ...):
    result: Result = r.output  # fully typed
```

**JSON Schema dict** (requires `pip install jsonschema`):
```python
schema = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "label": {"type": "string"},
    },
    "required": ["score", "label"],
}

async for r in llm_map.run(inputs=..., output_schema=schema, ...):
    r.output  # dict validated against schema
```

The LLM response is parsed by stripping code fences (` ```json ... ``` `)
before JSON parsing. If parsing or validation fails, the item is retried.

---

## AgenticMap

**Source:** `src/mnesis/operators/agentic_map.py`

`AgenticMap` spawns one full `MnesisSession` per input item and runs multi-turn
reasoning inside each. Sub-sessions are isolated at the session level: they do
not share conversation history or compaction state with each other or with any
parent session, even though they typically persist into the same underlying
store (the same SQLite database via a shared `StorePool`) under different
`session_id`s.

Use `AgenticMap` instead of `LLMMap` when:

- Each item requires back-and-forth reasoning over multiple turns.
- The item needs tool access to gather information or take actions.
- You want each item's reasoning history to be independently compacted and
  stored for auditing.

The parent's context cost is O(1) per item: only `result.output_text` (the
final turn's text) is surfaced to the caller, regardless of how many turns
the sub-session ran.

### Basic example

```python
import asyncio
from mnesis import AgenticMap, OperatorConfig

async def main():
    repositories = [
        "https://github.com/example/repo-a",
        "https://github.com/example/repo-b",
        "https://github.com/example/repo-c",
    ]

    agentic_map = AgenticMap(
        config=OperatorConfig(agentic_map_concurrency=3),
        model="anthropic/claude-opus-4-6",
    )

    async for result in agentic_map.run(
        inputs=repositories,
        agent_prompt_template=(
            "Analyze this repository URL and identify potential security risks.\n"
            "Repository: {{ item }}\n\n"
            "Provide a structured risk assessment."
        ),
        read_only=False,
        max_turns=5,
    ):
        if result.success:
            print(f"Repository: {result.input}")
            print(f"Session ID: {result.session_id}")
            print(f"Tokens used: {result.token_usage.effective_total():,}")
            print(f"Assessment:\n{result.output_text[:500]}")
        else:
            print(f"Failed: {result.error}")
```

### Multi-turn reasoning with a continuation message

By default (`continuation_message=""`), `AgenticMap` sends only the initial
prompt and stops. To enable multi-turn reasoning, provide a `continuation_message`
that will be sent as the user turn after each assistant response:

```python
async for result in agentic_map.run(
    inputs=tasks,
    agent_prompt_template="Complete this task: {{ item }}",
    read_only=False,
    max_turns=20,
    continuation_message="Continue. Report your progress and next step.",
):
    print(result.output_text)
    # result.intermediate_outputs contains all turn texts in order
    for i, turn in enumerate(result.intermediate_outputs):
        print(f"Turn {i}: {turn[:100]}")
```

### Sub-session stop conditions

Each sub-session stops when any of the following occurs (checked in order after
each turn):

1. `finish_reason` is `"stop"` or `"end_turn"` — natural completion.
2. `doom_loop_detected` is `True` — consecutive identical tool calls detected.
3. `finish_reason` is `"max_tokens"` — output token limit hit.
4. `max_turns` turns have been executed.
5. `continuation_message` is empty and turn > 0 — single-turn mode.

### Collecting all results at once

```python
batch = await agentic_map.run_all(
    inputs=items,
    agent_prompt_template="Investigate: {{ item }}",
    read_only=False,
    max_turns=10,
)

print(f"{len(batch.successes)} succeeded, {len(batch.failures)} failed")
for result in batch.successes:
    print(f"{result.input}: {result.output_text[:200]}")
```

### LLMMap vs. AgenticMap — decision guide

| Factor | Use LLMMap | Use AgenticMap |
|---|---|---|
| Turns per item | 1 | 2+ |
| Tool calls | No | Yes |
| History stored | No | Yes, per sub-session |
| Compaction | No | Yes, per sub-session |
| Concurrency default | 16 | 4 |
| Token overhead | Minimal | Full session overhead |
| Output type | Structured (Pydantic/JSON) | Free-form text |

---

## Concurrency

Both operators use `asyncio.Semaphore` to cap parallel tasks. The defaults are
conservative for AgenticMap (LLM API rate limits and SQLite write contention
are the limiting factors with many sub-sessions).

### Configure via `OperatorConfig`

```python
from mnesis import MnesisConfig, OperatorConfig

config = MnesisConfig(
    operators=OperatorConfig(
        llm_map_concurrency=32,    # default: 16, max: 128
        agentic_map_concurrency=8, # default: 4, max: 32
        max_retries=5,             # default: 3, for LLMMap per-item retries
    )
)
```

### Override per call

`concurrency` can be overridden on each `run()` / `run_all()` call:

```python
async for r in llm_map.run(
    inputs=items,
    prompt_template="...",
    output_schema=MySchema,
    concurrency=4,   # override for this call only
):
    ...
```

### Completion order

Both operators use `asyncio.as_completed()` internally, so results stream back
as they finish — not in input order. Use `result.input` to correlate with the
original item:

```python
results_by_input = {}
async for result in llm_map.run(inputs=items, ...):
    results_by_input[id(result.input)] = result
```

---

## Retries

`LLMMap` retries each item on two classes of failure:

- **Validation/schema failures** — the LLM returned output that could not be
  parsed as valid JSON or did not match the schema. On retry, a
  `retry_guidance` message is appended to the prompt (default:
  `"Your previous response was not valid JSON. Return only a JSON object matching the required schema."`).
  Override this to avoid leaking schema details:

    ```python
    async for r in llm_map.run(
        inputs=items,
        prompt_template="...",
        output_schema=MySchema,
        retry_guidance="Please return only valid JSON.",
    ):
        ...
    ```

- **Transient errors** — `litellm` exceptions (network, rate limit, etc.).
  Retried with exponential backoff: `min(0.5 * 2^(attempt-1), 8.0)` seconds.
  This backoff is applied only to non-timeout exceptions.

- **Timeout failures** — `TimeoutError` consumes one attempt and counts toward
  `max_retries`, but does **not** apply the exponential backoff. Timeout retries
  proceed immediately on the next attempt.

`AgenticMap` does not implement per-item retries. Only failures that propagate
as unhandled exceptions out of the sub-session loop are captured in
`AgentMapResult.error` and `AgentMapResult.error_kind`. LLM provider errors
that `MnesisSession.send()` catches internally and converts to a `TurnResult`
with `finish_reason="error"` are treated as completed turns — in those cases
`AgentMapResult.success` may still be `True` and the error details appear in
the final turn's `output_text`.

### `MapResult` fields

| Field | Type | Description |
|---|---|---|
| `input` | `Any` | The original input item |
| `output` | `Any \| None` | Parsed output (Pydantic model or dict), or `None` on failure |
| `success` | `bool` | True when output is valid |
| `error` | `str \| None` | Error message on failure |
| `error_kind` | `"timeout" \| "validation" \| "llm_error" \| "schema_error" \| None` | Error category |
| `attempts` | `int` | How many attempts were made (1 = first try succeeded) |

---

## Operator Events

Both operators can publish `MAP_STARTED`, `MAP_ITEM_COMPLETED`, and
`MAP_COMPLETED` events to an `EventBus` if one is provided. By default,
`event_bus=None` and no events are emitted — the operator runs normally
but produces no observable events unless you explicitly supply a bus.

To receive operator events on a session's event bus, inject it at construction
time:

```python
from mnesis import MnesisSession, LLMMap, AgenticMap, MnesisConfig, MnesisEvent, OperatorConfig

config = MnesisConfig(
    operators=OperatorConfig(llm_map_concurrency=8),
)

async with MnesisSession.open(model="anthropic/claude-haiku-3-5", config=config) as session:

    def on_map_progress(event: MnesisEvent, payload: dict):
        if event == MnesisEvent.MAP_ITEM_COMPLETED:
            print(f"  Completed {payload['completed']}/{payload['total']} "
                  f"(success={payload['success']})")
        elif event == MnesisEvent.MAP_STARTED:
            print(f"Map started: {payload['total']} items")
        elif event == MnesisEvent.MAP_COMPLETED:
            print(f"Map done: {payload['total']} items")

    session.subscribe(MnesisEvent.MAP_STARTED, on_map_progress)
    session.subscribe(MnesisEvent.MAP_ITEM_COMPLETED, on_map_progress)
    session.subscribe(MnesisEvent.MAP_COMPLETED, on_map_progress)

    llm_map = LLMMap(
        config=config.operators,
        event_bus=session.event_bus,  # inject the session's bus
        model="anthropic/claude-haiku-3-5",
    )

    async for result in llm_map.run(inputs=[...], prompt_template="...", output_schema=...):
        pass
```

The same pattern works for `AgenticMap`:

```python
agentic_map = AgenticMap(
    config=config.operators,
    event_bus=session.event_bus,
    model="anthropic/claude-opus-4-6",
)
```

### Event payloads

| Event | Payload keys |
|---|---|
| `MAP_STARTED` | `total: int`, `model: str` (LLMMap) or `type: "agentic"` (AgenticMap) |
| `MAP_ITEM_COMPLETED` | `completed: int`, `total: int`, `success: bool` |
| `MAP_COMPLETED` | `total: int`, `completed: int` (LLMMap) or `total: int` (AgenticMap) |

---

## `MapBatch` and `AgentMapBatch`

`run_all()` on each operator returns a batch result object that collects all
items before returning.

### `MapBatch` (from `LLMMap.run_all()`)

```python
from mnesis import MapResult

batch: MapBatch = await llm_map.run_all(inputs=..., ...)

batch.successes     # list[MapResult] — items that produced valid output
batch.failures      # list[MapResult] — items that exhausted retries
batch.total         # int — total item count (successes + failures)
batch.total_attempts  # int — sum of result.attempts across all items
```

### `AgentMapBatch` (from `AgenticMap.run_all()`)

```python
from mnesis import AgentMapResult

batch: AgentMapBatch = await agentic_map.run_all(inputs=..., ...)

batch.successes     # list[AgentMapResult]
batch.failures      # list[AgentMapResult]
batch.total         # int
batch.total_attempts  # int (always 1 per item for AgenticMap)
```

### `AgentMapResult` fields

| Field | Type | Description |
|---|---|---|
| `input` | `Any` | The original input item |
| `session_id` | `str` | The sub-session ID (for store lookup or debugging) |
| `output_text` | `str` | Final assistant text from the last turn |
| `success` | `bool` | True when the session completed without exception |
| `error` | `str \| None` | Exception message on failure |
| `error_kind` | `str \| None` | Always `"llm_error"` for sub-agent failures |
| `token_usage` | `TokenUsage` | Cumulative token usage across all sub-session turns |
| `intermediate_outputs` | `list[str]` | Text from each turn in order |

---

## Permission Restrictions

`AgenticMap` operates under the following constraints on sub-sessions, derived
directly from the source:

- Further sub-agents are **not supported**. `AgenticMap` is designed for a
  single level of nesting; the `parent_id` chain assumes this. Passing another
  `AgenticMap` as a tool is not prevented at runtime, but the behavior is
  undefined and not tested.

- `read_only=True` **raises `NotImplementedError`** (not yet implemented).
  Always pass `read_only=False` to proceed. Write-tool filtering is planned
  for a future release.

```python
# This raises NotImplementedError:
await agentic_map.run_all(inputs=..., read_only=True, ...)

# Correct:
await agentic_map.run_all(inputs=..., read_only=False, ...)
```
