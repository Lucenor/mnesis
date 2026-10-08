# Changelog

## 0.4.0 — 2026-10-08

Compaction thresholds now measure the current context window instead of lifetime token usage, so long sessions no longer compact on nearly every turn, and several paths that could drop history without summarizing it are closed. Shutdown, store, event-bus and operator fixes cover concurrency and cancellation. Some changes are breaking; read the first section before upgrading. The database schema is unchanged.

### Breaking changes

**Session creation rejects a compaction budget that leaves no usable context** (#115).
`MnesisSession.create()`, `open()` and `load()` raise `ValueError` when `context_limit - max_output_tokens <= compaction_output_budget` for the resolved model, `model_overrides` included, even with the default budget (20,000). Affected: small-window overrides such as `{"context_limit": 32_768}` on `gpt-4o` (16,384 max output inherited), which had no usable context in 0.3.0. Action: lower `compaction_output_budget`, set `max_output_tokens` in `model_overrides` (it is also the `max_tokens` that `send()` requests, so it caps reply length), or both.

```python
from mnesis import CompactionConfig, MnesisConfig

MnesisConfig(
    model_overrides={"context_limit": 32_768, "max_output_tokens": 4_096},
    compaction=CompactionConfig(compaction_output_budget=4_000),
)
```

**`compaction_output_budget` maximum lowered from 200,000 to 100,000** (#115).
Larger values fail validation with `pydantic.ValidationError`. Action: use 100,000 or less.

**Operations on a closed session raise `SessionClosedError`** (#122).
After `close()`, `send()`, `record()`, `stream()`, `messages()`, `conversation_messages()`, `context_for_next_turn()` and `compact()` raise `SessionClosedError`, a `MnesisStoreError` subclass; the first three also raise it while `close()` runs. In 0.3.0 all but `compact()` already raised `MnesisStoreError` here, and `compact()` returned a `level_used == 0` result. Action: do not call `compact()` after `close()`, or catch `SessionClosedError` (exported from `mnesis`).

**Compaction token fields report the current context size** (#114).
`COMPACTION_TRIGGERED.tokens` and `tokens_before`/`tokens_after` (`CompactionResult`, `COMPACTION_COMPLETED`) measure the active context: system prompt, live summaries, raw messages in context and the latest `send()`'s tool schemas. In 0.3.0 `tokens` was lifetime usage and `tokens_before` the whole raw history. Affected: dashboards, alerts and logs on these values. Action: read lifetime usage from `session.token_usage`.

**`CompactionFailedPayload` has a required `aborted` key** (#121).
Code that constructs this `TypedDict`, such as test doubles, no longer type-checks without it. Action: add `"aborted": False`.

**`CompactionEngine.run_compaction(abort=...)` returns instead of raising** (#121).
With the `abort` event set, the run returns the failure result (`level_used == 0`) and publishes `COMPACTION_FAILED` with `aborted=True`; 0.3.0 raised `asyncio.CancelledError`. `Task.cancel()` still raises. Affected: direct `CompactionEngine` users. Action: instead of catching `CancelledError`, check `abort.is_set()` after the call, or `aborted` on the `COMPACTION_FAILED` payload; `level_used == 0` alone also covers failed and no-op runs.

**`ImmutableStore.list_sessions(parent_id=...)` excludes soft-deleted children** (#113).
`active_only` (default `True`) was ignored when `parent_id` was given. Action: pass `active_only=False` to include them.

**`ImmutableStore.get_messages()` rejects an unknown `since_message_id`** (#121).
`get_messages()` and `get_messages_with_parts()` raise `MessageNotFoundError` when `since_message_id` is not a message of the session; 0.3.0 returned the whole history for an unknown ID and used a foreign ID's timestamp. Action: pass an ID from the same session, or catch `mnesis.store.MessageNotFoundError`.

### Behavior changes

- Soft and hard thresholds, in `send()` and `record()`, compare the current context size, including tool schemas from `send(tools=...)`, not lifetime usage (#114, #115). `session.token_usage` stays lifetime usage, unused by them; `load()` rebuilds it from stored turns (0.3.0 started at zero) (#112).
- Compaction condenses until the context is below half the soft threshold (#114). A run that `send()` blocks on at the hard limit stops under the soft threshold, leaving the rest to later background runs (#115).
- Summaries, including the level 3 fallback, are sized against the session model's window (`model_overrides` included) instead of a fixed 200K and counted with its tokenizer, so `token_count` values change (#115).
- Compaction summarizes only raw messages still in context. At the summarizer's input cap (75% of the compaction model's window), a run makes several passes, one leaf summary each; `max_compaction_rounds` now bounds these passes as well as condensation rounds. `compacted_message_count` totals the run; `summary_token_count` sums the leaves, or is the condensed node's size if the run condensed (#115).
- A repeated `compact()` with nothing new returns a no-op result (`level_used == 0`) with `COMPACTION_COMPLETED` instead of a duplicate summary (#115).
- Background compaction pauses after a run that ends at or above the soft threshold with nothing left to summarize or condense, until the next user turn; the hard-limit path and `compact()` never pause (#115).
- With `SessionConfig.retry.max_retries > 0` (default 0), compaction LLM calls get the `send()` retry policy, with LiteLLM retries off, and publish `LLM_RETRY` with `source="compaction"`; once a call exhausts its retries, the run falls to level 3. A hard-limit `send()` and `close()` can wait through these backoffs (#121, #122).
- An empty or truncated (`finish_reason == "length"`) completion fails its level and the run escalates (#121). After two consecutive truncated level 1 condensations, the session starts later condensations at level 2 if `level2_enabled` (the count is in memory only) (#122).
- New default prompts: the level 1 prompts set a length target; all ask for `path (file_<hex>)` pairs, named people with roles (`PEOPLE:` at level 2), next steps only when stated (else "None stated") and no assistant suggestions as tasks. Compliance depends on the compaction model. A custom `compaction_prompt` is sent unchanged (#122).
- The `[LCM File IDs: ...]` footer can hold `path (file_<hex>)` entries; old footers parse as before, and 0.3.0 reads the IDs in the new format (#122).
- Condensation follows context order and, if the summaries exceed the compaction model's window, condenses the oldest subset that fits (#122).
- `close()` rejects new calls, cancels retry backoff, waits up to 30 seconds for in-flight `send()` and `record()` calls (then closes anyway), drains compaction and closes the store. It is idempotent, publishes `SESSION_CLOSED` once (0.3.0: every call) and shares one close among concurrent callers. 0.3.0 could close the store under an in-flight call, and concurrent calls raced (#122).
- Canceling `send()` during a retry backoff (`Task.cancel()`, `asyncio.timeout()`) raises `CancelledError` or `TimeoutError`; 0.3.0 swallowed it and returned an error turn. A backoff canceled by `close()` still ends in an error turn (#122).
- Canceling a caller that waits on compaction no longer cancels the compaction; `close()` drains it (#122).
- `compact()` waits for an in-flight background compaction, and `compaction_in_progress` covers manual runs (#114).
- Empty assistant messages, such as those a canceled `send()` leaves, are left out of the assembled context and `context_for_next_turn()` (#122).
- Any awaitable an `on_part` callback returns (Future, Task, `__await__` object) is awaited; 0.3.0 awaited only coroutines and ignored a failing Future. If `on_part` or its awaitable raises, `send()` returns an error turn (`finish_reason == "error"`), as for a raising coroutine in 0.3.0 (#112, #121).
- `LLMMap` passes `num_retries=0` to LiteLLM when `OperatorConfig.max_retries > 0` (default 3), so LiteLLM and the provider client no longer retry on top of Mnesis; depending on the provider, a failing item makes fewer attempts (#122).
- `LLMMap` releases its concurrency slot during retry backoff, so under rate limiting new items start while failed ones wait; lower `llm_map_concurrency` or `max_retries` to throttle (#123).
- `ModelInfo.from_model_string()` recognizes OpenAI o-series models by the model name after any provider prefix (`o` plus digits, then `-` or the end, as in `o3-mini` or `openrouter/openai/o3-mini`) and gives them the 200K window, 100K max output and `o200k_base`. `o4` and later models, such as `o4-mini`, get these limits instead of the 128K/4K default; names that only contain `o1` or `o3`, such as `ollama/qwen2:o1`, now get the default (#97, #124).
- `ImmutableStore.update_part_status(output=..., error_message=...)` needs SQLite JSON1 (built in since 3.38) (#121).

### Fixes

#### Compaction and history integrity

- Turns past the summarizer's input cap left the context without being summarized (#115).
- Each compaction summarized again from the first message, leaving overlapping live summaries (#115).
- After waiting for an in-flight compaction, `send()` skipped the hard-limit re-check, so the oldest messages could be dropped without notice (#114).
- Two compactions could run at once (#114).
- Level 3 output could exceed the usable budget when it carried many file IDs (#113, #115).
- With the summarizer down, level 3 could replace the protected last two user turns, including the one being answered (#115).
- File-ID extraction stopped at fixed cut-offs (100K characters per message, 500 of tool output) and skipped tool inputs, errors and pruned outputs (#115).
- Condensation input was not bounded by the compaction model's window (#122).
- A condensation committed its node insert, context swap and parent supersession separately, so a failure between them could leave the parents live but out of the context (#122).
- `TurnResult.compaction_result` was always `None`; it is set when `send()` waited on a hard-limit compaction (#112).
- Doom-loop detection never fired. `record()` tracks tool calls, sets the new `RecordResult.doom_loop_detected` and publishes `DOOM_LOOP_DETECTED`; a turn without tool calls, any `send()` turn included, resets it. `TurnResult.doom_loop_detected` stays `False` (#112).

#### Store and lifecycle

- Concurrent `append_part()` calls on one message could get the same `part_index` (#113).
- Concurrent `update_part_status()` calls could lose each other's `output` or `error_message` (#121).
- Same-millisecond messages were not kept in insertion order by `get_messages()`, `get_last_summary_message()` and the summary-node queries, and `since_message_id` skipped same-millisecond messages after the boundary (#121, #122).
- Multi-statement writes on a `StorePool` connection could interleave with other writers or stay half-applied after an error or cancellation; writes are now locked transactions that roll back on any exception (#122).
- `MnesisSession.create()`, `load()` and `ImmutableStore.initialize()` left a connection open when they failed partway (#115, #122).
- `SummaryDAGStore.get_node_by_id()` loaded the whole session per lookup (#122).
- File-reference lookups on a shared connection could read another session's uncommitted insert (#123).
- Persisting a `ToolPart` whose `input` held a `datetime`, `UUID`, `Decimal` or `Path` raised `TypeError`; such values are stored as ISO-8601 strings or `str()` (#115).

#### Events

- `EventBus.publish()` raised `TypeError` when a synchronous handler raised, since 0.1.0; handler errors are logged and swallowed, as documented (#115).
- Async handler tasks could be garbage-collected mid-run, and their exceptions went unlogged (#112).
- Handlers returning a Future or other non-coroutine awaitable were not awaited (#121).

#### Operators

- Closing an `LLMMap` or `AgenticMap` generator early left its remaining items running; they are canceled and awaited, and `AgenticMap` aborts compaction in their sub-sessions. `MAP_COMPLETED` is still published only on normal completion (#123).
- `AgenticMap` did not close its own `StorePool` when closed early (#123).

#### Documentation

- The BYO-LLM guide built prompts from `session.messages()`, which includes compacted turns; it uses `context_for_next_turn()`, which can start with an assistant summary (#113, #115).
- The events guide called `MAP_COMPLETED.completed` a success count (it includes failures) and listed the published `PRUNE_COMPLETED` as reserved (#113, #123).
- The configuration guide adds sizing advice for `soft_threshold_fraction` and lists small-window configurations that raise (#115, #122).

### New

- `MnesisSession.close(*, abort_compaction=False)`: with `True`, an in-flight compaction ends its retry backoff, stops at its next check and publishes `COMPACTION_FAILED` with `aborted=True`; a request already sent is not interrupted (#122).
- `CompactionConfig.condense_skip_level1` (default `False`) starts condensation at level 2, for compaction models that overrun the level 1 output limit (#122).
- `LLM_RETRY` payload keys `source` (`"send"` or `"compaction"`) and, for compaction, `stage` (`"summarisation"` or `"condensation"`) and `compaction_level` (1 or 2). Treat a missing `source` as `"send"` (#122).
- `BuiltContext.context_tokens` and `full_context_tokens`, the un-truncated context size the thresholds compare (#114), and `TokenEstimator.for_model()` (#115).
- `CompactionEngine.wait_for_pending()` returns the awaited `CompactionResult`; the engine and stores gain methods for direct use, such as `compact_exclusive()` and `ImmutableStore.sum_token_usage()` (see the API reference) (#97, #112, #114, #115, #122).

## 0.3.0 — 2026-04-04

- chore(deps): bump codecov/codecov-action from 5.5.3 to 6.0.0 (#88)
- chore(deps): bump astral-sh/setup-uv from 7.6.0 to 8.0.0 (#87)
- docs: document read_only=True behavior and model_overrides (Stream 5) (#89)
- test: improve coverage for pruner, session, store, files, and estimator (#93)
- feat: complete unfinished API surfaces (Stream 3 of 0.3.0) (#91)
- feat: async streaming iterator API (session.stream()) (#90)
- feat: retry & resilience in send() (Mnesis 0.3.0 Stream 1) (#92)
- chore(deps): bump anchore/sbom-action from 0.23.1 to 0.24.0 (#86)
- chore(deps): bump github/codeql-action from 4.32.6 to 4.34.1 (#85)
- chore(deps): bump actions/deploy-pages from 4.0.5 to 5.0.0 (#84)
- chore(deps): bump codecov/codecov-action from 5.5.2 to 5.5.3 (#83)
- chore(deps): bump actions/attest-sbom from 4.0.0 to 4.1.0 (#81)
- chore(deps): bump astral-sh/setup-uv from 7.3.1 to 7.6.0 (#80)
- chore(deps): bump actions/create-github-app-token from 2.2.1 to 3.0.0 (#82)
- chore(deps): bump anchore/sbom-action from 0.23.0 to 0.23.1 (#78)
- chore(deps): bump github/codeql-action from 4.32.4 to 4.32.5 (#77)
- chore(deps): bump actions/dependency-review-action from 4.8.3 to 4.9.0 (#76)
- chore(deps): bump astral-sh/setup-uv from 7.3.0 to 7.3.1 (#75)
- chore(deps): bump anchore/sbom-action from 0.22.2 to 0.23.0 (#74)
- chore(deps): bump actions/dependency-review-action from 4.6.0 to 4.8.3 (#70)
- chore(deps): bump actions/attest-build-provenance from 2.4.0 to 4.0.0 (#72)
- chore(deps): bump actions/upload-artifact from 4.6.2 to 6.0.0 (#73)
- chore(deps): bump actions/attest-sbom from 2.4.0 to 4.0.0 (#71)
- chore(deps): bump github/codeql-action from 4.32.3 to 4.32.4 (#69)

## 0.2.0 — 2026-02-25

- chore: source __version__ from importlib.metadata (#68)
- fix: clean up benchmark terminal output with tqdm progress bars (#67)
- feat: persist per-turn snapshot metrics in benchmark results (#66)
- feat: add session.history() for per-turn context snapshots (#65)
- fix: inject session date headers into LOCOMO conversation turns (#64)
- feat: key benchmark output filenames by run configuration (#61)
- fix: improve locomo chart labels, y-axis padding, and layout (#62)
- feat: add --generate-baseline command to LOCOMO benchmark (#63)
- feat: add --replot flag to regenerate benchmark charts without re-running (#60)
- fix: update locomo benchmark to use MnesisSession.open() (#59)
- fix: update examples to use open(), context_for_next_turn(), and correct imports (#58)
- docs: replace Mermaid with D2 + mkdocs-panzoom, remove all Cloudflare workarounds (#57)
- docs: pre-render Mermaid diagrams to inline SVG at build time (#56)
- fix: add data-cfasync=false to bypass Cloudflare Rocket Loader on all critical scripts (#55)
- docs: remove manually written [Unreleased] changelog section (#54)
- fix: inline panzoom via template override to bypass Cloudflare Rocket Loader (#53)
- fix: bundle panzoom inline to bypass Cloudflare Rocket Loader type mangling (#52)
- fix: use ES module import for panzoom to bypass Cloudflare Rocket Loader (#51)
- feat: switch to mkdocs-mermaid2 plugin with panzoom zoom/pan support (#50)
- docs: fix state diagram syntax and Mermaid refresh race condition (#49)
- docs: fix Mermaid syntax errors in architecture.md diagrams (#48)
- docs: add architecture.md deep-dive and operators.md guide (#47)
- docs: add events.md page and improve concepts/configuration coverage (#46)
- docs: add Wave 2-5 changelog entries (#45)
- docs: update open() pattern in getting-started, README, and contributing guide (#44)
- docs: critical fixes and one-line doc corrections (#43)
- docs: Wave 5 documentation and polish (L-4, L-5, L-6, L-7, L-8, L-10, L-13, L-15) (#42)
- feat: Wave 3 session ergonomics (M-1, M-7, M-8, L-9) (#41)
- feat: Wave 3 events and files ergonomics (M-13, M-14, M-17) (#40)
- feat: Wave 3 operators ergonomics (H-6, H-7, M-10, M-11, M-12, L-11, L-12) (#39)
- feat: Wave 2 __all__ surface reduction (H-2, H-3, H-4, H-5, M-15, M-16, L-1, L-14) (#38)
- fix: Wave 1 session correctness (C-1, C-3, H-1, M-9) + Wave 4 config cleanup (#37)
- fix: C-2 enforce read_only on AgenticMap; H-8 guard jsonschema import in LLMMap (#36)
- test: add convergence escalation unit tests for level1/level2 summarisation (#35)
- feat: context_items table for O(1) context assembly (#34)
- feat: persist summary DAG to SQLite (kind, parent_node_ids, superseded) (#33)
- feat: add convergence-based escalation to level1/level2 summarisation (#32)
- fix: address PR #30 review comments — DAG supersession, token accounting, coverage (#31)
- feat: add condensation, file ID propagation, multi-round loop, soft/hard threshold, input cap (#30)
- docs: beautify mkdocs site with Material theme enhancements (#29)
- feat: add LOCOMO benchmark for evaluating compaction quality (#27)

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
