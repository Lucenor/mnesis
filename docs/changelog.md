# Changelog

## 0.1.0 — Initial release

- `MnesisSession` with `create()`, `load()`, `send()`, `record()`, `compact()`, `close()`
- Three-level compaction engine (selective LLM → aggressive LLM → deterministic fallback)
- Tool output pruner with configurable protect window
- Content-addressed large file handler with structural exploration summaries
- `LLMMap` — stateless parallel LLM calls with Pydantic schema validation
- `AgenticMap` — parallel sub-agent sessions with full multi-turn reasoning
- Append-only SQLite persistence via `ImmutableStore`
- In-process `EventBus` with typed `MnesisEvent` enum
- `session.record()` for BYO-LLM turn injection without litellm
- Support for all litellm providers (Anthropic, OpenAI, Gemini, OpenRouter, Azure, etc.)
- Python 3.12, 3.13, 3.14 support
