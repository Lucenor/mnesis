"""Core message and part data models for Mnesis."""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# ── Part Models ────────────────────────────────────────────────────────────────


class TextPart(BaseModel):
    """A plain text segment of a message."""

    type: Literal["text"] = "text"
    text: str


class ReasoningPart(BaseModel):
    """Chain-of-thought reasoning text (e.g. Claude extended thinking, o1 reasoning)."""

    type: Literal["reasoning"] = "reasoning"
    text: str


class ToolStatus(BaseModel):
    """Mutable lifecycle status of a tool call. Updated in-place via update_part_status()."""

    state: Literal["pending", "running", "completed", "error"] = "pending"
    compacted_at: int | None = Field(
        default=None,
        description="Unix millisecond timestamp when this output was pruned. None = not pruned.",
    )
    started_at: int | None = None
    completed_at: int | None = None


class ToolPart(BaseModel):
    """A tool call and its result within an assistant message."""

    type: Literal["tool"] = "tool"
    tool_name: str
    tool_call_id: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: str | None = None
    error_message: str | None = None
    status: ToolStatus = Field(default_factory=ToolStatus)

    @property
    def compacted_at(self) -> int | None:
        """Convenience accessor for the pruning tombstone timestamp."""
        return self.status.compacted_at

    @property
    def is_protected(self) -> bool:
        """Protected tools are never pruned by ToolOutputPruner."""
        _PROTECTED: frozenset[str] = frozenset({"skill"})
        return self.tool_name in _PROTECTED


class CompactionMarkerPart(BaseModel):
    """
    Metadata marker embedded in a compaction summary message.

    Records which messages were compacted and which escalation level succeeded.
    """

    type: Literal["compaction"] = "compaction"
    summary_node_id: str
    compacted_message_count: int
    compacted_token_count: int
    level: int = 1


class StepStartPart(BaseModel):
    """Marker for the start of an agentic step within a turn."""

    type: Literal["step_start"] = "step_start"
    step_index: int = 0
    snapshot_ref: str | None = None


class StepFinishPart(BaseModel):
    """Marker for the end of an agentic step, including token usage."""

    type: Literal["step_finish"] = "step_finish"
    step_index: int = 0
    snapshot_ref: str | None = None
    tokens_used: TokenUsage | None = None


class PatchPart(BaseModel):
    """A git-format unified diff representing file changes made during a turn."""

    type: Literal["patch"] = "patch"
    unified_diff: str
    files_changed: list[str] = Field(default_factory=list)
    additions: int = 0
    deletions: int = 0


class FileRefPart(BaseModel):
    """
    A content-addressed reference to a large file stored outside the context window.

    When the LargeFileHandler intercepts a file that exceeds the inline threshold,
    it stores it externally and replaces the content with this reference object.
    The ContextBuilder renders this as a structured ``[FILE: ...]`` block.
    """

    type: Literal["file_ref"] = "file_ref"
    content_id: str
    """SHA-256 hex digest of the file content. Used as cache key."""
    path: str
    """Original file path as provided by the caller."""
    file_type: str
    """Detected MIME type or language identifier (e.g. 'python', 'application/json')."""
    token_count: int
    """Estimated token count of the full file content."""
    exploration_summary: str
    """Deterministic structural description of the file (classes, functions, keys, etc.)."""


# Discriminated union — the ``type`` field is the discriminator key.
MessagePart = Annotated[
    TextPart
    | ReasoningPart
    | ToolPart
    | CompactionMarkerPart
    | StepStartPart
    | StepFinishPart
    | PatchPart
    | FileRefPart,
    Field(discriminator="type"),
]


# ── Token Usage ────────────────────────────────────────────────────────────────


class TokenUsage(BaseModel):
    """Token counts for a single LLM response or cumulative session usage."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total: int = 0

    def effective_total(self) -> int:
        """Return total, computing from parts when the explicit total is zero."""
        if self.total:
            return self.total
        return self.input + self.output + self.cache_read + self.cache_write

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            total=self.effective_total() + other.effective_total(),
        )


# ── Error Types ────────────────────────────────────────────────────────────────


class MessageError(BaseModel):
    """Structured error attached to an assistant message when the turn fails."""

    code: Literal[
        "output_length",
        "aborted",
        "structured_output",
        "auth",
        "api_error",
        "context_overflow",
        "unknown",
    ]
    message: str
    retriable: bool = False
    provider_error_code: str | None = None


# ── Message Models ─────────────────────────────────────────────────────────────


class Message(BaseModel):
    """
    A single message stored in the ImmutableStore.

    Messages are append-only — the only mutable fields are token usage and
    finish_reason (updated after streaming completes), and part status fields
    (managed via update_part_status).
    """

    id: str
    """ULID-based sortable ID, e.g. ``msg_01JXYZ6K3MNPQR4STUVWXYZ01``."""
    session_id: str
    role: Literal["user", "assistant"]
    created_at: int = Field(default_factory=lambda: int(time.time() * 1000))
    """Unix millisecond timestamp."""
    parent_id: str | None = None
    agent: str = "default"
    model_id: str = ""
    provider_id: str = ""
    # Assistant-only fields
    is_summary: bool = False
    """True if this message is a compaction summary produced by the CompactionEngine."""
    tokens: TokenUsage | None = None
    cost: float = 0.0
    finish_reason: str | None = None
    error: MessageError | None = None
    mode: str | None = None
    """'compaction' for summary messages, None for regular turns."""


class MessageWithParts(BaseModel):
    """A message together with its ordered list of typed parts."""

    message: Message
    parts: list[MessagePart] = Field(default_factory=list)

    @property
    def id(self) -> str:
        return self.message.id

    @property
    def role(self) -> str:
        return self.message.role

    @property
    def is_summary(self) -> bool:
        return self.message.is_summary

    @property
    def model_id(self) -> str:
        """The model that produced this message (empty string for user messages)."""
        return self.message.model_id

    @property
    def tokens(self) -> TokenUsage | None:
        """Token usage for this message (None for user messages or before streaming completes)."""
        return self.message.tokens

    def text_content(self) -> str:
        """Concatenate text from all TextPart objects in this message."""
        return "".join(part.text for part in self.parts if isinstance(part, TextPart))


# ── Context Budget ─────────────────────────────────────────────────────────────


class ContextBudget(BaseModel):
    """Token budget for assembling the context window on a single turn."""

    model_context_limit: int
    reserved_output_tokens: int
    compaction_buffer: int = 20_000

    @property
    def usable(self) -> int:
        """Maximum tokens available for conversation history."""
        return self.model_context_limit - self.reserved_output_tokens - self.compaction_buffer

    def fits(self, token_count: int) -> bool:
        """Return True if token_count fits within the usable budget."""
        return token_count <= self.usable

    def remaining(self, used: int) -> int:
        """Return remaining tokens given how many have been used."""
        return self.usable - used


# ── Result Types ───────────────────────────────────────────────────────────────


class TurnResult(BaseModel):
    """
    The result of a single ``MnesisSession.send()`` call.

    Contains the assistant's text response, token usage, finish reason, and
    indicators for compaction and doom loop detection.

    **Important — ``finish_reason == "error"``**: When ``finish_reason`` is
    ``"error"``, the LLM call failed and ``text`` contains an error description,
    *not* an LLM response. Do not treat the text as a model reply in this case.
    Always check ``finish_reason`` before processing ``text``.
    """

    message_id: str
    text: str
    finish_reason: Literal["stop", "max_tokens", "tool_calls", "length", "error"] | str
    """
    Why the LLM stopped generating. Known values:

    - ``"stop"`` — natural end of response.
    - ``"max_tokens"`` / ``"length"`` — output token limit reached.
    - ``"tool_calls"`` — the model requested tool execution.
    - ``"error"`` — **the LLM call failed**; ``text`` contains the error
      description, not a model response. Do not pass this text to the model
      as if it were a real reply.

    Providers may return additional values; the ``| str`` allows for these.
    """
    tokens: TokenUsage
    cost: float
    compaction_triggered: bool = False
    compaction_result: CompactionResult | None = None
    doom_loop_detected: bool = False


class CompactionResult(BaseModel):
    """
    The result of a compaction run.

    Records which escalation level succeeded, how many messages were compressed,
    the token savings achieved, and how many tool outputs were pruned.
    """

    session_id: str
    summary_message_id: str
    level_used: int
    """1 = selective LLM, 2 = aggressive LLM, 3 = deterministic fallback."""
    compacted_message_count: int
    summary_token_count: int
    tokens_before: int
    tokens_after: int
    elapsed_ms: float
    pruned_tool_outputs: int = 0
    """Number of tool output parts tombstoned by the ToolOutputPruner during this run."""
    pruned_tokens: int = 0
    """Tokens reclaimed by pruning tool outputs during this run."""


class PruneResult(BaseModel):
    """The result of a ToolOutputPruner pass."""

    pruned_count: int
    pruned_tokens: int
    candidates_scanned: int


class RecordResult(BaseModel):
    """
    The result of a ``MnesisSession.record()`` call.

    Records the IDs of the persisted user and assistant messages so callers
    can reference them later (e.g. for event subscriptions or debugging).
    """

    user_message_id: str
    assistant_message_id: str
    tokens: TokenUsage
    compaction_triggered: bool = False
