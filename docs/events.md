# Events

## Overview

Every significant operation in Mnesis publishes an event to an in-process `EventBus`. The bus enables external monitoring — compaction progress, message creation, doom loop detection, operator fan-out — without polling or modifying core logic.

Each `MnesisSession` owns its own `EventBus` instance. Operators (`LLMMap`, `AgenticMap`) do not create a bus by default: they only emit `MAP_*` events when an `EventBus` is explicitly passed (for example, by injecting the session bus at construction time to unify all events on one subscriber).

**Key properties of the bus:**

- Sync handlers are called inline within `publish()`. For a given event, handlers registered with `subscribe()` run first (in their registration order), followed by handlers registered with `subscribe_all()` (also in their registration order).
- Async handlers are scheduled as background tasks (`asyncio.create_task`), non-blocking; they follow the same per-event-then-global ordering as sync handlers.
- Sync handler exceptions are logged and swallowed — they never propagate to the publisher. Async handler exceptions follow normal `asyncio` task semantics and are not caught or logged by the `EventBus`.
- `unsubscribe()` is a silent no-op if the handler is not registered.

---

## Subscribing to Events

```python
import asyncio
from mnesis import MnesisSession, MnesisConfig
from mnesis.events.bus import MnesisEvent
from mnesis.events.payloads import CompactionCompletedPayload, MessageCreatedPayload

async def main() -> None:
    session = await MnesisSession.create(model="openai/gpt-4o")

    # Sync handler — called inline, no await needed
    def on_message(event: MnesisEvent, payload: MessageCreatedPayload) -> None:
        print(f"[{event}] role={payload['role']} id={payload['message_id']}")

    # Async handler — scheduled as a background task
    async def on_compaction(event: MnesisEvent, payload: CompactionCompletedPayload) -> None:
        print(
            f"[{event}] level={payload['level_used']} "
            f"tokens {payload['tokens_before']} → {payload['tokens_after']}"
        )

    session.event_bus.subscribe(MnesisEvent.MESSAGE_CREATED, on_message)  # type: ignore[arg-type]
    session.event_bus.subscribe(MnesisEvent.COMPACTION_COMPLETED, on_compaction)  # type: ignore[arg-type]

    async with session:
        result = await session.send("Hello, world!")
        print(result.text)

    # Clean up when done
    session.event_bus.unsubscribe(MnesisEvent.MESSAGE_CREATED, on_message)
```

To subscribe to every event type with a single handler, use `subscribe_all`:

```python
def log_all(event: MnesisEvent, payload: dict) -> None:  # type: ignore[type-arg]
    print(f"EVENT {event}: {payload}")

session.event_bus.subscribe_all(log_all)
```

---

## Event Reference

### Currently Published Events

| Event | Value | When it fires | Payload class |
|---|---|---|---|
| `SESSION_CREATED` | `session.created` | Session is initialised | `SessionCreatedPayload` |
| `SESSION_CLOSED` | `session.closed` | Session context manager exits | `SessionClosedPayload` |
| `MESSAGE_CREATED` | `message.created` | A user or assistant message is persisted | `MessageCreatedPayload` |
| `COMPACTION_TRIGGERED` | `compaction.triggered` | Token usage crosses the soft or hard threshold | `CompactionTriggeredPayload` |
| `COMPACTION_COMPLETED` | `compaction.completed` | A compaction pass finishes (success or partial) | `CompactionCompletedPayload` |
| `COMPACTION_FAILED` | `compaction.failed` | Compaction raises an unhandled exception | `CompactionFailedPayload` |
| `DOOM_LOOP_DETECTED` | `doom_loop.detected` | The same tool call repeats past the threshold | `DoomLoopDetectedPayload` |
| `MAP_STARTED` | `map.started` | An operator begins processing its item list | `MapStartedPayload` |
| `MAP_ITEM_COMPLETED` | `map.item_completed` | One item finishes (success or failure) | `MapItemCompletedPayload` |
| `MAP_COMPLETED` | `map.completed` | All items have been processed | `MapCompletedPayload` |

### Reserved (Defined but Not Yet Published)

| Event | Value | Notes |
|---|---|---|
| `SESSION_UPDATED` | `session.updated` | Reserved for future use |
| `SESSION_DELETED` | `session.deleted` | Reserved for future use |
| `MESSAGE_UPDATED` | `message.updated` | Reserved for future use |
| `PART_CREATED` | `part.created` | Reserved for streaming granularity |
| `PART_UPDATED` | `part.updated` | Reserved for streaming granularity |
| `PRUNE_COMPLETED` | `prune.completed` | Reserved for future pruning events |

---

## Payload Schemas

Each event carries a typed `TypedDict` payload. Import the payload classes from `mnesis.events.payloads` for IDE auto-complete and static type checking.

### `SessionCreatedPayload`

Fired by: `MnesisEvent.SESSION_CREATED`

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | The newly created session's ID |
| `model` | `str` | The LLM model string the session was created with |

### `SessionClosedPayload`

Fired by: `MnesisEvent.SESSION_CLOSED`

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | The session that was closed |

### `MessageCreatedPayload`

Fired by: `MnesisEvent.MESSAGE_CREATED`

| Field | Type | Description |
|---|---|---|
| `message_id` | `str` | The newly persisted message's ID |
| `role` | `str` | `"user"` or `"assistant"` |

### `CompactionTriggeredPayload`

Fired by: `MnesisEvent.COMPACTION_TRIGGERED`

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | The session whose compaction was triggered |
| `tokens` | `int` | Current cumulative token count that crossed the threshold |

### `CompactionCompletedPayload`

Fired by: `MnesisEvent.COMPACTION_COMPLETED`

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | The session that was compacted |
| `summary_message_id` | `str` | ID of the newly created summary message, or `""` on failure |
| `level_used` | `int` | Escalation level used: 1 = selective LLM, 2 = aggressive LLM, 3 = deterministic |
| `compacted_message_count` | `int` | Number of messages folded into the summary |
| `summary_token_count` | `int` | Token count of the resulting summary |
| `tokens_before` | `int` | Cumulative tokens before compaction |
| `tokens_after` | `int` | Cumulative tokens after compaction |
| `elapsed_ms` | `float` | Wall-clock time the compaction pass took |
| `pruned_tool_outputs` | `int` | Number of tool outputs tombstoned during this run |
| `pruned_tokens` | `int` | Tokens reclaimed by pruning tool outputs |

### `CompactionFailedPayload`

Fired by: `MnesisEvent.COMPACTION_FAILED`

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | The session whose compaction failed |
| `error` | `str` | Human-readable error description |

### `DoomLoopDetectedPayload`

Fired by: `MnesisEvent.DOOM_LOOP_DETECTED`

| Field | Type | Description |
|---|---|---|
| `session_id` | `str` | The session where the doom loop was detected |
| `tool` | `str` | Name of the tool call that was repeated past the threshold |

---

## Operator Events

`MAP_STARTED`, `MAP_ITEM_COMPLETED`, and `MAP_COMPLETED` are emitted by operators only when they are constructed with an `event_bus=...` argument, and the events are published on that provided bus. To receive them via the session bus, pass `session.event_bus` when constructing the operator:

```python
from mnesis.operators import LLMMap, AgenticMap
from mnesis.events.payloads import MapStartedPayload, MapItemCompletedPayload, MapCompletedPayload

# Inject the session bus so operator events appear on it
llm_map = LLMMap(config.operators, event_bus=session.event_bus)
agentic_map = AgenticMap(config.operators, event_bus=session.event_bus)
```

### `MapStartedPayload`

Fired by: `MnesisEvent.MAP_STARTED`

| Field | Type | Present for | Description |
|---|---|---|---|
| `total` | `int` | All operators | Total number of items to process |
| `model` | `str` (optional) | `LLMMap` only | LLM model string |
| `type` | `str` (optional) | `AgenticMap` only | Operator type string (`"agentic"`) |

### `MapItemCompletedPayload`

Fired by: `MnesisEvent.MAP_ITEM_COMPLETED` (once per item)

| Field | Type | Description |
|---|---|---|
| `completed` | `int` | Number of items completed so far (including this one) |
| `total` | `int` | Total number of items |
| `success` | `bool` | Whether this item succeeded |

### `MapCompletedPayload`

Fired by: `MnesisEvent.MAP_COMPLETED`

| Field | Type | Present for | Description |
|---|---|---|---|
| `total` | `int` | All operators | Total number of items processed |
| `completed` | `int` (optional) | `LLMMap` only | Number of items that completed successfully |

!!! note "AgenticMap omits `completed`"
    `AgenticMap` publishes only `total` in its `MAP_COMPLETED` payload. `LLMMap` includes both `total` and `completed`.

---

## Complete Example

The following snippet wires up multiple event handlers, runs a session, and then uses an operator — all events on a single bus:

```python
import asyncio
from pydantic import BaseModel

from mnesis import MnesisConfig, MnesisSession
from mnesis.events.bus import MnesisEvent
from mnesis.events.payloads import (
    CompactionCompletedPayload,
    DoomLoopDetectedPayload,
    MapCompletedPayload,
    MapItemCompletedPayload,
    MessageCreatedPayload,
    SessionCreatedPayload,
)
from mnesis.operators import LLMMap


class Sentiment(BaseModel):
    label: str  # "positive" | "negative" | "neutral"
    confidence: float


async def main() -> None:
    config = MnesisConfig()
    session = await MnesisSession.create(model="openai/gpt-4o", config=config)

    # ── event handlers ────────────────────────────────────────────────────────

    def on_session_created(event: MnesisEvent, payload: SessionCreatedPayload) -> None:
        print(f"Session started: {payload['session_id']} using {payload['model']}")

    def on_message(event: MnesisEvent, payload: MessageCreatedPayload) -> None:
        print(f"  Message persisted: role={payload['role']} id={payload['message_id']}")

    def on_compaction(event: MnesisEvent, payload: CompactionCompletedPayload) -> None:
        print(
            f"  Compacted at level {payload['level_used']}: "
            f"{payload['tokens_before']} → {payload['tokens_after']} tokens "
            f"({payload['elapsed_ms']:.0f} ms)"
        )

    def on_doom_loop(event: MnesisEvent, payload: DoomLoopDetectedPayload) -> None:
        print(f"  WARNING: doom loop on tool '{payload['tool']}'")

    def on_map_item(event: MnesisEvent, payload: MapItemCompletedPayload) -> None:
        status = "OK" if payload["success"] else "FAIL"
        print(f"  Map item {payload['completed']}/{payload['total']}: {status}")

    def on_map_done(event: MnesisEvent, payload: MapCompletedPayload) -> None:
        print(f"  Map complete: {payload['total']} items processed")

    # ── register handlers ─────────────────────────────────────────────────────

    session.event_bus.subscribe(MnesisEvent.SESSION_CREATED, on_session_created)  # type: ignore[arg-type]
    session.event_bus.subscribe(MnesisEvent.MESSAGE_CREATED, on_message)  # type: ignore[arg-type]
    session.event_bus.subscribe(MnesisEvent.COMPACTION_COMPLETED, on_compaction)  # type: ignore[arg-type]
    session.event_bus.subscribe(MnesisEvent.DOOM_LOOP_DETECTED, on_doom_loop)  # type: ignore[arg-type]
    session.event_bus.subscribe(MnesisEvent.MAP_ITEM_COMPLETED, on_map_item)  # type: ignore[arg-type]
    session.event_bus.subscribe(MnesisEvent.MAP_COMPLETED, on_map_done)  # type: ignore[arg-type]

    # ── run session ───────────────────────────────────────────────────────────

    async with session:
        result = await session.send("What is the capital of France?")
        print(f"Response: {result.text}")

        # Operator events flow to the session bus because we injected it
        llm_map = LLMMap(config.operators, event_bus=session.event_bus)
        reviews = ["Great product!", "Terrible quality.", "It is okay."]
        batch = await llm_map.run_all(
            inputs=reviews,
            prompt_template="Classify the sentiment of: {{ item }}",
            output_schema=Sentiment,
            model="openai/gpt-4o",
        )
        for result in batch.successes:
            sentiment = result.output
            if isinstance(sentiment, Sentiment):
                print(f"  {result.input!r} → {sentiment.label} ({sentiment.confidence:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
```

!!! tip "Async handlers and task lifetime"
    Async handlers are scheduled via `asyncio.create_task` and are not awaited by the `EventBus`. If your program exits immediately after the last `await`, pending handler tasks may be cancelled before they complete. In long-running services this is rarely an issue; in short scripts, keep the event loop alive long enough for handlers to run, or have your handlers manage any long-running work they spawn if you need deterministic completion.
