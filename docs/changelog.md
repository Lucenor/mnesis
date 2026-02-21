# Changelog

## [Unreleased]

### Added

**Session ergonomics**

- `MnesisSession.open()` — async context manager factory as the preferred entry point.
  Eliminates the `await ... async with` double-ceremony of `create()`; the session is
  automatically closed (pending compaction awaited, DB released) on exit, even on exception:

  ```python
  async with MnesisSession.open(model="anthropic/claude-opus-4-6") as session:
      result = await session.send("Hello!")
  ```

- `MnesisSession.context_for_next_turn(system_prompt=None)` — returns the
  compaction-aware context window as a list of `{"role", "content"}` dicts ready to pass
  to any chat completion API. Intended for BYO-LLM callers who manage their own LLM client
  but want Mnesis for memory management. Pair with `session.record()` to persist the turn.

- `MnesisSession.conversation_messages()` — convenience wrapper around `messages()` that
  filters out compaction summary rows (`is_summary=True`), returning only the actual
  user/assistant turns.

- `MnesisSession.compaction_in_progress` property — `True` while a background compaction
  task is actively running (correctly goes `False` once the task completes).

- `FinishReason` string enum (`STOP`, `MAX_TOKENS`, `TOOL_CALLS`, `LENGTH`, `ERROR`) — now
  exported from `mnesis` and `mnesis.models`. `TurnResult.finish_reason` is typed as
  `FinishReason | str`, so existing `== "stop"` comparisons continue to work without change.

**Operator improvements**

- `LLMMap` and `AgenticMap` now accept a `model` keyword argument in their constructors
  as a default model. Per-call `model=` overrides the constructor default; a `ValueError`
  is raised at call time only if neither is set.

- `LLMMap.run_all()` / `AgenticMap.run_all()` — non-streaming counterparts to `run()` that
  collect all results and return a `MapBatch` / `AgentMapBatch` with `.successes`,
  `.failures`, and `.total_attempts`. Use these when you don't need streaming.

- `MapResult.error_kind` field — categorises failures as `"timeout"`, `"validation"`,
  `"llm_error"`, or `"schema_error"` so callers can branch on failure mode without parsing
  error strings.

- `AgentMapResult.intermediate_outputs` — list of per-turn text outputs from the sub-agent,
  allowing inspection of multi-turn reasoning traces.

- `LLMMap.run()` accepts `retry_guidance: str` parameter — controls the message appended
  to the retry prompt on schema/validation errors. Defaults to a generic JSON correction
  hint; prevents internal error details from leaking into the LLM context.

- `AgenticMap.run()` accepts `continuation_message: str` parameter — text injected as an
  additional user turn between sub-agent turns. Empty string (default) means the loop exits
  after turn 0 finishes normally.

- Jinja2 template validation now uses a proper AST check instead of regex, so expressions
  like `{{ item['key'] }}`, `{{ item | upper }}`, and any valid Jinja2 syntax are accepted.

**Event system**

- Typed `TypedDict` payload classes for every published event, importable from
  `mnesis.events`: `SessionCreatedPayload`, `SessionClosedPayload`,
  `MessageCreatedPayload`, `CompactionTriggeredPayload`, `CompactionCompletedPayload`,
  `CompactionFailedPayload`, `DoomLoopDetectedPayload`, `MapStartedPayload`,
  `MapItemCompletedPayload`, `MapCompletedPayload`. Use these for static type checking of
  event handler payloads.

- `EventBus.unsubscribe_all(handler)` — removes a handler from all per-event registrations
  and from the global handler list in one call, mirroring `subscribe_all()` for symmetric
  teardown. Silent no-op for any slot where the handler is not registered.

**File handling**

- `LargeFileHandler.from_session(session)` — classmethod factory that constructs a handler
  from any `MnesisSession`-like object, avoiding a circular import. Simpler than
  constructing `LargeFileHandler` directly with individual config/store arguments.

- `FileHandleResult` is now a Pydantic `BaseModel` with a validator that enforces exactly
  one of `inline_content` or `file_ref` is set. The `is_inline` property allows clean
  branching without inspecting both fields.

**Token estimator**

- `TokenEstimator(heuristic_only=True)` — public constructor keyword that forces the
  character-based heuristic and skips tiktoken entirely. Replaces the private
  `_force_heuristic` attribute backdoor. Useful in tests and environments where tiktoken
  is not installed.

**Configuration**

- `SessionConfig` sub-object on `MnesisConfig` — new home for session-level tunables,
  starting with `doom_loop_threshold`. Access as `config.session.doom_loop_threshold`.

- `MnesisConfig.model_overrides` dict — override auto-detected model limits without
  constructing `ModelInfo` directly:

  ```python
  MnesisConfig(model_overrides={"context_limit": 128_000, "max_output_tokens": 16_384})
  ```

- `StoreConfig.db_path` and `FileConfig.storage_dir` now accept `str | Path` values; `~`
  is expanded and the path is resolved at construction time.

- `MnesisStoreError` and `SessionNotFoundError` are now exported from `mnesis` directly.

### Changed

- **`mnesis.__all__` reduced from 36 to 26 entries.** Removed internal types that callers
  never construct: `ContextBudget`, `SummaryNode`, `CompactionMarkerPart`, `Message`,
  `FileReference`, `LargeFileHandler`, `FileHandleResult`, `PruneResult`, `ModelInfo`,
  `make_id`, and several part types not part of the public contract. All removed names
  remain importable from their source modules for existing code that references them.

- **`CompactionConfig.buffer` renamed to `compaction_output_budget`** (breaking rename).
  Update any config construction that used `buffer=`:

  ```python
  # Before
  CompactionConfig(buffer=20_000)
  # After
  CompactionConfig(compaction_output_budget=20_000)
  ```

- **`MnesisConfig.doom_loop_threshold` moved** to `config.session.doom_loop_threshold`.
  The top-level field has been removed.

- `CompactionResult` now exposes `pruned_tool_outputs` (count) and `pruned_tokens` fields
  populated from the pruner. `PruneResult` is no longer in `__all__`; use these fields instead.

- `MessageWithParts` gains `model_id` and `tokens` properties, removing the need to import
  the internal `Message` storage type to access these values.

- `MnesisEvent` enum and `EventBus` docstrings now document which events are published
  versus reserved, and clarify that `MAP_*` events fire on the operator's own `EventBus`
  (inject the session bus via `LLMMap(config.operators, event_bus=session.event_bus)` to
  receive them on the session bus).

- `LLMMap.run()` and `AgenticMap.run()` return type annotation corrected from
  `AsyncIterator` to `AsyncGenerator` (the accurate annotation for async generators).

- `LLMMap.output_schema` type narrowed from `dict[str, Any] | type` to
  `dict[str, Any] | type[BaseModel]`. Non-`BaseModel` types now raise `TypeError` eagerly
  at `run()` start. Dict schemas raise `ImportError` with install instructions if
  `jsonschema` is not installed (previously the error surfaced only after tasks launched).

- `MnesisConfig.default()` classmethod removed. Use `MnesisConfig()` directly — it
  produces an identical result and is the idiomatic Pydantic pattern.

### Fixed

- `AgenticMap.run(read_only=True)` previously accepted the parameter silently without
  enforcing it. It now raises `NotImplementedError` immediately so callers are not misled
  into thinking tool isolation is active. Pass `read_only=False` to proceed.

- `MnesisSession.create()` and `load()` now raise `ValueError` when both the `db_path`
  keyword argument and a non-default `config.store.db_path` are supplied, preventing silent
  path shadowing.

- `MnesisSession.send()` system prompt override now uses an explicit `None` check instead
  of a falsy test, so passing an empty string intentionally suppresses the session prompt
  rather than silently falling back to it.

- `session.messages()` docstring now correctly documents that summary messages
  (`is_summary=True`) and `CompactionMarkerPart` tombstones are included in the returned
  list.

---

## 0.1.1 — 2026-02-20

- fix: make all examples functional in MNESIS_MOCK_LLM=1 mode (#26)
- Fix missing comma in SECURITY.md disclosure policy (#24)
- fix: accept {{ item['key'] }} and {{ item.attr }} in operator templates (#25)
- chore: add project URLs for PyPI sidebar (#23)
- chore: upgrade codeql-action to v4 (#21)

## 0.1.0 — 2026-02-20

- chore: pre-release fixes for 0.1.0 (#20)
- fix: correct OpenSSF Scorecard badge to scorecard.dev domain (#19)
- ci: SBOM attestation, dependency review, and OpenSSF Scorecard (#18)
- ci: add build provenance attestation, add badges to README (#17)
- chore(deps): bump actions/upload-pages-artifact from 3.0.1 to 4.0.0 (#16)
- chore(deps): bump astral-sh/setup-uv from 4.2.0 to 7.3.0 (#15)
- chore(deps): bump actions/create-github-app-token from 1.12.0 to 2.2.1 (#13)
- chore(deps): bump actions/checkout from 4.3.1 to 6.0.2 (#14)
- ci: automated publish workflow (#12)
- docs: add mkdocs-material site with auto-generated API reference (#11)
- feat: add session.record() for BYO-LLM turn injection (#10)
- chore: drop unused anthropic direct dep, document provider configuration (#9)
- chore: untrack uv.lock and drop --frozen from CI (#8)
- ci: add Python 3.14 to test matrix (#7)
- Potential fix for code scanning alert no. 3: Workflow does not contain permissions (#5)
- docs: add SECURITY.md with vulnerability reporting policy (#4)
- fix: correct license identifier to Apache-2.0 (#2)
- ci: add CI workflow and PyPI publish workflow
- Set package-ecosystem to 'uv' in dependabot.yml
- docs: add logo and derived icon/wordmark assets
- docs: add logo icon and wordmark to README header
- docs: remove copyright line from CONTRIBUTING
- docs: fix OOLONG link, reference LCM paper, fix benchmark attribution
- docs: clean up benchmarks section
- docs: tighten whitespace on all benchmark figures
- docs: add benchmark figures, rewrite README with images and comparison table
- docs: remove copyright line from README license section
- docs: update README license, add CONTRIBUTING guide
- docs: add README and API reference
- docs: add example scripts
- test: add full test suite (76 tests, 79% coverage)
- feat: add MnesisSession and package public API
- feat: add LLMMap and AgenticMap operators
- feat: add large file handler
- feat: add three-level compaction engine
- feat: add context builder
- feat: add SQLite persistence layer
- feat: add token estimator and event bus
- feat: add core data models
- chore: add pyproject.toml and uv.lock
- chore: extend .gitignore and add NOTICE
- Initial commit

All releases are tagged in [GitHub Releases](https://github.com/Lucenor/mnesis/releases).
