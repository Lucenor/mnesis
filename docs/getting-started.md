# Installation & Quick Start

## Installation

```bash
uv add mnesis
# or
pip install mnesis
```

Requires Python 3.12+. No external database or service needed — mnesis uses a local SQLite file.

## Your first session

```python
import asyncio
from mnesis import MnesisSession

async def main():
    async with MnesisSession.open(
        model="anthropic/claude-opus-4-6",
        system_prompt="You are a helpful assistant.",
    ) as session:
        result = await session.send("Explain the GIL in Python.")
        print(result.text)
        print(f"Tokens used: {result.tokens.effective_total()}")

asyncio.run(main())
```

`MnesisSession.open()` is the recommended entry point — it creates a session and acts as an async context manager in a single call, automatically closing the session (and awaiting any pending background compaction) when the block exits.

Set `ANTHROPIC_API_KEY` in your environment before running. See [Providers](providers.md) for other LLM providers.

### Manual lifecycle (advanced)

If you need to control the session lifetime explicitly — for example, to share a session across multiple coroutines, or to close it conditionally — use `create()` instead:

```python
# create() returns the session directly; use it as an async context manager
# to get automatic cleanup, or call close() manually.
async with await MnesisSession.create(
    model="anthropic/claude-opus-4-6",
    system_prompt="You are a helpful assistant.",
) as session:
    result = await session.send("Explain the GIL in Python.")
    print(result.text)

# Or without a context manager — you are responsible for calling close():
session = await MnesisSession.create(model="anthropic/claude-opus-4-6")
result = await session.send("Explain the GIL in Python.")
await session.close()
```

## Try it without an API key

Every example ships with a mock LLM mode:

```bash
MNESIS_MOCK_LLM=1 uv run python examples/01_basic_session.py
```

## Multi-turn conversations

Sessions automatically persist every message turn, assemble the context window, and trigger compaction when needed — you don't need to manage any of that manually:

```python
async with MnesisSession.open(model="anthropic/claude-opus-4-6") as session:
    await session.send("My name is Alice.")
    await session.send("What's my name?")   # model still knows: Alice
    await session.send("Write a poem about context management.")
```

## Resuming a session

Sessions are persisted to SQLite by default at `~/.mnesis/sessions.db`. Load a previous session by ID:

```python
# First run — save the session ID
session = await MnesisSession.create(model="anthropic/claude-opus-4-6")
session_id = session.id
await session.send("Remember: the secret code is 42.")
await session.close()

# Later — resume it
session = await MnesisSession.load(session_id)
result = await session.send("What was the secret code?")
print(result.text)  # "The secret code is 42."
await session.close()
```

## Monitoring compaction

`TurnResult.compaction_triggered` tells you when compaction fired:

```python
result = await session.send("...")
if result.compaction_triggered:
    print("Compaction ran in the background — context is fresh again.")
```

Or subscribe to events for richer observability:

```python
from mnesis.events.bus import MnesisEvent

session.subscribe(MnesisEvent.COMPACTION_COMPLETED, lambda e, p: print("Compacted!", p))
```

## Streaming

`session.stream()` is an async generator that yields text chunks as they arrive, making it ideal for chat UIs and progressive rendering.  It is a thin wrapper around `send()` — all persistence, compaction, and token tracking work exactly as they do with `send()`.

```python
async with MnesisSession.open(model="anthropic/claude-opus-4-6") as session:
    async for event in session.stream("Explain the GIL in Python."):
        if event.type == "text_delta":
            print(event.text, end="", flush=True)
        elif event.type == "turn_complete":
            print()  # newline after streaming ends
            print(f"Tokens: {event.result.tokens.effective_total()}")
```

Two event types are yielded:

- `TextDelta` — one or more text chunks with a `text` attribute.
- `TurnComplete` — the final event, carrying the full `TurnResult` (token counts, finish reason, compaction status).

You can also match on the `type` discriminator string (`"text_delta"`, `"turn_complete"`) instead of using `isinstance`.

### Abandonment safety

If you `break` out of the `async for` loop or an exception interrupts iteration, the underlying `send()` task still completes in the background.  The turn is fully persisted, token counters are updated, and compaction is triggered if the threshold is crossed.  The `TurnComplete` event may not be consumed by the caller, but the `TURN_COMPLETED` event is still published on the event bus.

```python
# Safe to break early — the turn is still persisted
async for event in session.stream("Long response..."):
    if event.type == "text_delta" and "stop" in event.text:
        break  # stop reading, but send() finishes in background
```

## Next steps

- [Providers](providers.md) — configure OpenAI, Gemini, OpenRouter, etc.
- [BYO-LLM](byo-llm.md) — use your own SDK and let mnesis handle memory only
- [Concepts](concepts.md) — how compaction, pruning, and file references work
- [Configuration](configuration.md) — tune compaction thresholds, storage paths, concurrency
- [Examples](examples.md) — runnable example scripts
