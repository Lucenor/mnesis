# Mnesis API Reference

## MnesisSession

The primary entry point for creating and managing LLM sessions.

### `MnesisSession.create()`

```python
@classmethod
async def create(
    cls,
    *,
    model: str,
    agent: str = "default",
    parent_id: str | None = None,
    config: MnesisConfig | None = None,
    system_prompt: str = "You are a helpful assistant.",
    db_path: str | None = None,
) -> MnesisSession
```

Create a new session.

**Args:**
- `model`: LLM model string in litellm format (e.g. `"anthropic/claude-opus-4-6"`, `"openai/gpt-4o"`)
- `agent`: Agent role name for multi-agent setups
- `parent_id`: Parent session ID for sub-sessions (used by AgenticMap)
- `config`: Mnesis configuration. Defaults to `MnesisConfig.default()`
- `system_prompt`: System prompt applied to all turns
- `db_path`: Override database path. Defaults to `~/.mnesis/sessions.db`

**Returns:** Initialized `MnesisSession`

---

### `MnesisSession.load()`

```python
@classmethod
async def load(
    cls,
    session_id: str,
    config: MnesisConfig | None = None,
    db_path: str | None = None,
) -> MnesisSession
```

Load an existing session from the database.

**Raises:** `SessionNotFoundError` if the session does not exist.

---

### `session.send()`

```python
async def send(
    self,
    message: str | list[MessagePart],
    *,
    tools: list[Any] | None = None,
    on_part: Callable[[MessagePart], None | Awaitable[None]] | None = None,
    system_prompt: str | None = None,
) -> TurnResult
```

Send a user message and receive an assistant response.

**Args:**
- `message`: Text string or list of `MessagePart` objects
- `tools`: Tool definitions in litellm/OpenAI format
- `on_part`: Streaming callback invoked for each `MessagePart` as it arrives
- `system_prompt`: Override the session system prompt for this turn

**Returns:** `TurnResult` with `text`, `tokens`, `finish_reason`, `compaction_triggered`, `doom_loop_detected`

---

### `session.record()`

```python
async def record(
    self,
    user_message: str | list[MessagePart],
    assistant_response: str | list[MessagePart],
    *,
    tokens: TokenUsage | None = None,
    finish_reason: str = "stop",
) -> RecordResult
```

Persist a completed user/assistant turn **without making an LLM call**.

Use this when you manage LLM calls yourself (using the Anthropic, OpenAI, Gemini, or any other SDK directly) and only want mnesis to handle memory, context assembly, and compaction.

**Args:**
- `user_message`: User message text or list of `MessagePart` objects
- `assistant_response`: Assistant reply text or list of `MessagePart` objects
- `tokens`: Token usage for the turn. Estimated from text length if omitted
- `finish_reason`: Finish reason from your LLM response. Defaults to `"stop"`

**Returns:** `RecordResult`

**Example — using the Anthropic SDK directly:**

```python
import anthropic
from mnesis import MnesisSession, TokenUsage

client = anthropic.Anthropic()
session = await MnesisSession.create(model="anthropic/claude-opus-4-6")

user_text = "Explain quantum entanglement."
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": user_text}],
)

await session.record(
    user_message=user_text,
    assistant_response=response.content[0].text,
    tokens=TokenUsage(
        input=response.usage.input_tokens,
        output=response.usage.output_tokens,
    ),
)
```

**Example — using the OpenAI SDK directly:**

```python
from openai import AsyncOpenAI
from mnesis import MnesisSession, TokenUsage

client = AsyncOpenAI()
session = await MnesisSession.create(model="openai/gpt-4o")

user_text = "What is the capital of France?"
response = await client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": user_text}],
)

await session.record(
    user_message=user_text,
    assistant_response=response.choices[0].message.content,
    tokens=TokenUsage(
        input=response.usage.prompt_tokens,
        output=response.usage.completion_tokens,
    ),
)
```

Compaction is triggered automatically after `record()` if cumulative token usage exceeds the model's context budget, exactly as with `send()`.

### `RecordResult`

```python
class RecordResult(BaseModel):
    user_message_id: str       # Persisted user message ID
    assistant_message_id: str  # Persisted assistant message ID
    tokens: TokenUsage         # Token usage for the turn (provided or estimated)
    compaction_triggered: bool # True if compaction was scheduled after recording
```

---

### `session.messages()`

```python
async def messages(self) -> list[MessageWithParts]
```

Return the full message history including compaction summary messages.

---

### `session.compact()`

```python
async def compact(self) -> CompactionResult
```

Manually trigger synchronous compaction. Blocks until complete.

---

### `session.close()`

```python
async def close(self) -> None
```

Close the session and release database resources. Also accessible as `async with` context manager.

---

### Properties

| Property | Type | Description |
|---|---|---|
| `session.id` | `str` | ULID-based session identifier |
| `session.model` | `str` | Model string for this session |
| `session.token_usage` | `TokenUsage` | Cumulative token usage across all turns |
| `session.event_bus` | `EventBus` | Subscribe to session events |

---

## MnesisConfig

```python
class MnesisConfig(BaseModel):
    compaction: CompactionConfig
    file: FileConfig
    store: StoreConfig
    operators: OperatorConfig
    doom_loop_threshold: int  # default: 3
```

### CompactionConfig

| Field | Default | Description |
|---|---|---|
| `auto` | `True` | Auto-trigger compaction on overflow |
| `buffer` | `20_000` | Tokens reserved for compaction summary output |
| `prune` | `True` | Enable tool output pruning |
| `prune_protect_tokens` | `40_000` | Token window never pruned |
| `prune_minimum_tokens` | `20_000` | Min volume before pruning fires |
| `compaction_model` | `None` | Override model for summarization |
| `level2_enabled` | `True` | Enable Level 2 before Level 3 fallback |
| `plugin_hook` | `None` | Dotted path to custom prompt callable |

### FileConfig

| Field | Default | Description |
|---|---|---|
| `inline_threshold` | `10_000` | Max tokens for inline file inclusion |
| `storage_dir` | `None` | External file storage directory |
| `exploration_summary_model` | `None` | Model for LLM-based file summaries |

### StoreConfig

| Field | Default | Description |
|---|---|---|
| `db_path` | `~/.mnesis/sessions.db` | SQLite database path |
| `wal_mode` | `True` | Use WAL journal mode |

### OperatorConfig

| Field | Default | Description |
|---|---|---|
| `llm_map_concurrency` | `16` | Max concurrent LLMMap calls |
| `agentic_map_concurrency` | `4` | Max concurrent sub-agent sessions |
| `max_retries` | `3` | Retries per item on failure |

---

## LLMMap

```python
class LLMMap:
    def __init__(self, config: OperatorConfig, ...) -> None: ...

    async def run(
        self,
        inputs: list[Any],
        prompt_template: str,      # Jinja2 with {{ item }}
        output_schema: dict | type,  # JSON Schema dict or Pydantic model
        model: str,
        *,
        concurrency: int | None = None,
        max_retries: int | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> AsyncIterator[MapResult]
```

Stateless parallel LLM processing. Yields `MapResult` objects as they complete.

### MapResult

```python
class MapResult(BaseModel):
    input: Any           # Echo of the original input item
    output: Any | None   # Parsed and validated output
    success: bool
    error: str | None    # Set on failure
    attempts: int        # Number of attempts made
```

---

## AgenticMap

```python
class AgenticMap:
    def __init__(self, config: OperatorConfig, ...) -> None: ...

    async def run(
        self,
        inputs: list[Any],
        agent_prompt_template: str,  # Jinja2 with {{ item }}
        model: str,
        *,
        concurrency: int | None = None,
        read_only: bool = True,
        parent_session_id: str | None = None,
        tools: list[Any] | None = None,
        max_turns: int = 20,
        agent: str = "general",
        lcm_config: MnesisConfig | None = None,
    ) -> AsyncIterator[AgentMapResult]
```

Parallel sub-agent sessions. Each item gets a full `MnesisSession` with multi-turn reasoning.

### AgentMapResult

```python
class AgentMapResult(BaseModel):
    input: Any
    session_id: str      # Sub-session ID for debugging/resumption
    output_text: str     # Final assistant response text
    success: bool
    error: str | None
    token_usage: TokenUsage
```

---

## EventBus

```python
bus = session.event_bus

# Subscribe to a specific event
bus.subscribe(MnesisEvent.COMPACTION_COMPLETED, handler)

# Subscribe to all events
bus.subscribe_all(handler)

# Unsubscribe
bus.unsubscribe(MnesisEvent.COMPACTION_COMPLETED, handler)
```

Handler signature: `(event: MnesisEvent, payload: dict[str, Any]) -> None | Awaitable[None]`

### MnesisEvent Values

| Event | When |
|---|---|
| `SESSION_CREATED` | Session created |
| `SESSION_CLOSED` | Session closed |
| `MESSAGE_CREATED` | User or assistant message stored |
| `PART_CREATED` | A new message part stored |
| `PART_UPDATED` | Part status updated (pruning, tool completion) |
| `COMPACTION_TRIGGERED` | Overflow detected, compaction scheduled |
| `COMPACTION_COMPLETED` | Compaction succeeded |
| `COMPACTION_FAILED` | All levels failed (rare) |
| `PRUNE_COMPLETED` | Tool output pruning completed |
| `DOOM_LOOP_DETECTED` | Consecutive identical tool calls exceeded threshold |
| `MAP_STARTED` | LLMMap/AgenticMap started |
| `MAP_ITEM_COMPLETED` | One item finished |
| `MAP_COMPLETED` | All items finished |

---

## Data Models

### MessagePart

Discriminated union — the `type` field selects the concrete class.

| Type | Class | Key Fields |
|---|---|---|
| `"text"` | `TextPart` | `text: str` |
| `"reasoning"` | `ReasoningPart` | `text: str` |
| `"tool"` | `ToolPart` | `tool_name`, `tool_call_id`, `input`, `output`, `status` |
| `"compaction"` | `CompactionMarkerPart` | `summary_node_id`, `level`, `compacted_message_count` |
| `"step_start"` | `StepStartPart` | `step_index` |
| `"step_finish"` | `StepFinishPart` | `step_index`, `tokens_used` |
| `"patch"` | `PatchPart` | `unified_diff`, `files_changed` |
| `"file_ref"` | `FileRefPart` | `content_id`, `path`, `file_type`, `token_count`, `exploration_summary` |

### TokenUsage

```python
class TokenUsage(BaseModel):
    input: int
    output: int
    cache_read: int
    cache_write: int
    total: int

    def effective_total(self) -> int: ...  # Computed if total == 0
    def __add__(self, other: TokenUsage) -> TokenUsage: ...
```

### TurnResult

```python
class TurnResult(BaseModel):
    message_id: str
    text: str
    finish_reason: str       # "stop" | "max_tokens" | "tool_use" | "error"
    tokens: TokenUsage
    cost: float
    compaction_triggered: bool
    compaction_result: CompactionResult | None
    doom_loop_detected: bool
```

### CompactionResult

```python
class CompactionResult(BaseModel):
    session_id: str
    summary_message_id: str
    level_used: int              # 1, 2, or 3
    compacted_message_count: int
    summary_token_count: int
    tokens_before: int
    tokens_after: int
    elapsed_ms: float
```

---

## Exceptions

| Exception | Module | Raised When |
|---|---|---|
| `MnesisStoreError` | `mnesis.store.immutable` | Base class for all store errors |
| `SessionNotFoundError` | `mnesis.store.immutable` | Session ID does not exist |
| `MessageNotFoundError` | `mnesis.store.immutable` | Message ID does not exist |
| `PartNotFoundError` | `mnesis.store.immutable` | Part ID does not exist |
| `DuplicateIDError` | `mnesis.store.immutable` | Primary key collision |
| `ImmutableFieldError` | `mnesis.store.immutable` | Attempt to modify immutable field |

---

## LargeFileHandler

```python
class LargeFileHandler:
    def __init__(
        self,
        store: ImmutableStore,
        estimator: TokenEstimator,
        config: FileConfig,
    ) -> None: ...

    async def handle_file(
        self,
        path: str,
        *,
        content: str | bytes | None = None,
    ) -> FileHandleResult
```

### FileHandleResult

```python
@dataclass
class FileHandleResult:
    path: str
    inline_content: str | None   # Set when file fits within threshold
    file_ref: FileRefPart | None # Set when file exceeds threshold

    @property
    def is_inline(self) -> bool: ...
```

---

## ModelInfo

Auto-detected from the model string. Can be constructed manually to override defaults.

```python
class ModelInfo(BaseModel):
    model_id: str
    provider_id: str
    context_limit: int      # Total context window in tokens
    max_output_tokens: int  # Maximum output per response
    encoding: str           # tiktoken encoding name or "claude_heuristic"

    @classmethod
    def from_model_string(cls, model: str) -> ModelInfo: ...
```

---

## Advanced: Plugin Hooks

Override the compaction prompt by setting `CompactionConfig.plugin_hook` to a dotted import path:

```python
# myapp/hooks.py
def custom_compaction_prompt(messages: list, config: dict) -> str:
    return "Custom system prompt for compaction..."

# Configuration
config = MnesisConfig(
    compaction=CompactionConfig(
        plugin_hook="myapp.hooks.custom_compaction_prompt"
    )
)
```

## Advanced: Custom Token Estimator

```python
from mnesis.tokens.estimator import TokenEstimator
from mnesis.models.config import ModelInfo

class MyEstimator(TokenEstimator):
    def estimate(self, text: str, model: ModelInfo | None = None) -> int:
        # Use your own tokenizer
        return len(text.split())  # Simple word count
```

Pass to session components via the internal wiring (requires creating components manually).
