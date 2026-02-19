# Concepts

## The problem: context rot

LLMs don't degrade linearly as the context window fills. Accuracy drops sharply once the window exceeds roughly 32K tokens — not because the model runs out of space, but because stale, redundant content dilutes the signal. mnesis calls this **context rot**.

The naive fix — asking the model to summarize itself — fails silently. The model may drop constraints, forget file paths, or produce a summary that is itself too large. mnesis moves that responsibility out of the model and into a deterministic engine.

## Immutable Store + Active Context

Every message, tool call, and tool result is appended to an **append-only SQLite log**. Nothing is ever deleted or modified. This gives a complete audit trail of everything that happened in a session.

On each turn, the **ContextBuilder** assembles a curated *Active Context* — a slice of the log that fits the model's token budget. Older messages may be replaced by compaction summaries; pruned tool outputs are replaced by compact markers. The model always sees a coherent, budget-fitting view.

## Three-Level Compaction

When cumulative token usage crosses the threshold, the `CompactionEngine` escalates through three levels automatically:

| Level | Strategy | Failure mode |
|---|---|---|
| **1** | Structured LLM summary: Goal / Discoveries / Accomplished / Remaining | LLM error or output too large → escalate |
| **2** | Aggressive compression: drop reasoning, maximum conciseness | LLM error or still too large → escalate |
| **3** | Deterministic truncation | Never fails — always fits |

Level 3 is the unconditional safety net. Compaction runs **asynchronously** and never blocks a turn.

## Tool Output Pruning

Tool outputs tend to dominate context usage in agentic sessions. The `ToolOutputPruner` scans backward through history and **tombstones** completed tool outputs that fall outside a configurable protect window (default: last 40K tokens).

Tombstoned outputs are replaced with a compact `[tool: name — output pruned]` marker in the Active Context. The full output is still in the immutable store and can be retrieved at any time.

## Large File References

Files exceeding the inline threshold (default: 10K tokens) are stored externally by the `LargeFileHandler` as content-addressed `FileRefPart` objects. The ContextBuilder renders them as structured `[FILE: path]` blocks with:

- Detected MIME type / language
- Structural exploration summary (AST outline for Python, schema keys for JSON/YAML, headings for Markdown)
- Token count

The file is never re-read or re-inlined unless the model explicitly requests it — keeping large files from consuming the context budget on every turn.

## Parallel Operators

### LLMMap

Stateless parallel LLM calls over a list of inputs. Each item gets an independent LLM call with schema-validated output and per-item retry. O(1) context cost to the parent session — results flow back as a batch.

### AgenticMap

Each input item gets a full independent `MnesisSession` with multi-turn reasoning. Sub-sessions run in parallel up to a configurable concurrency limit. The parent session sees only the final output text from each sub-session.

## EventBus

Every significant operation publishes an event to the session's `EventBus`. Subscribe to monitor compaction, track message creation, detect doom loops, and more — without polling or modifying core logic.

```python
from mnesis.events.bus import MnesisEvent

session.subscribe(MnesisEvent.COMPACTION_COMPLETED, lambda e, p: print("Compacted:", p))
session.subscribe(MnesisEvent.DOOM_LOOP_DETECTED, lambda e, p: print("Doom loop!", p))
```

## Doom Loop Detection

If the model makes the same tool call (same name and input) more than `doom_loop_threshold` (default: 3) times consecutively, mnesis raises a `DOOM_LOOP_DETECTED` event and sets `TurnResult.doom_loop_detected = True`. The session continues — the caller decides how to handle it.
