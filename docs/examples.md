# Examples

All examples run without an API key using `MNESIS_MOCK_LLM=1`:

```bash
MNESIS_MOCK_LLM=1 uv run python examples/01_basic_session.py
MNESIS_MOCK_LLM=1 uv run python examples/05_parallel_processing.py
```

To use a real LLM, set the appropriate API key and omit the flag:

```bash
ANTHROPIC_API_KEY=sk-ant-... uv run python examples/01_basic_session.py
```

---

## 01 — Basic Session

**[`examples/01_basic_session.py`](https://github.com/Lucenor/mnesis/blob/main/examples/01_basic_session.py)**

Demonstrates the core `send()` loop:

- Creating a session with `MnesisSession.create()`
- Sending messages and reading `TurnResult`
- Monitoring `compaction_triggered`
- Inspecting cumulative token usage
- Async context manager lifecycle

---

## 02 — Long-Running Agent

**[`examples/02_long_running_agent.py`](https://github.com/Lucenor/mnesis/blob/main/examples/02_long_running_agent.py)**

- Subscribing to `EventBus` events
- Streaming callbacks via `on_part`
- Manually triggering `session.compact()`
- Inspecting compaction results

---

## 03 — Tool Use

**[`examples/03_tool_use.py`](https://github.com/Lucenor/mnesis/blob/main/examples/03_tool_use.py)**

- Passing tool definitions to `send()`
- Handling `ToolPart` streaming states (`pending → running → completed`)
- Inspecting tombstoned tool outputs after pruning

---

## 04 — Large Files

**[`examples/04_large_files.py`](https://github.com/Lucenor/mnesis/blob/main/examples/04_large_files.py)**

- Using `LargeFileHandler` to ingest files
- `FileRefPart` and content-addressed storage
- Cache hits on repeated file access
- Exploration summaries (AST outline, schema keys, etc.)

---

## 05 — Parallel Processing

**[`examples/05_parallel_processing.py`](https://github.com/Lucenor/mnesis/blob/main/examples/05_parallel_processing.py)**

- `LLMMap` with a Pydantic output schema
- `AgenticMap` with independent sub-agent sessions
- Concurrency control and per-item retry

---

## 06 — BYO-LLM

**[`examples/06_byo_llm.py`](https://github.com/Lucenor/mnesis/blob/main/examples/06_byo_llm.py)**

- Using `session.record()` to inject turns from your own SDK
- Building the LLM message list from `session.messages()`
- Passing explicit `TokenUsage` for accurate compaction budgeting

This example includes a canned-response stub so it runs without any API key. See [BYO-LLM](byo-llm.md) for a full explanation.
