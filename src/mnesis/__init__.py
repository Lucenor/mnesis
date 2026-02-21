"""
Mnesis — Lossless Context Management for LLM agentic tasks.

Primary entry point::

    from mnesis import MnesisSession, MnesisConfig

    async with await MnesisSession.create(model="anthropic/claude-opus-4-6") as session:
        result = await session.send("Hello!")
        print(result.text)
"""

from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.models import (
    CompactionConfig,
    CompactionResult,
    FileConfig,
    FileRefPart,
    FinishReason,
    MessagePart,
    MessageWithParts,
    MnesisConfig,
    OperatorConfig,
    RecordResult,
    SessionConfig,
    StoreConfig,
    TextPart,
    TokenUsage,
    ToolPart,
    TurnResult,
)
from mnesis.operators.agentic_map import AgenticMap, AgentMapBatch, AgentMapResult
from mnesis.operators.llm_map import LLMMap, MapBatch, MapResult
from mnesis.session import MnesisSession
from mnesis.store.immutable import MnesisStoreError, SessionNotFoundError
from mnesis.tokens.estimator import TokenEstimator

__version__ = "0.1.1"

__all__ = [
    "AgentMapBatch",
    "AgentMapResult",
    "AgenticMap",
    "CompactionConfig",
    "CompactionResult",
    "EventBus",
    "FileConfig",
    "FileRefPart",
    "FinishReason",
    "LLMMap",
    "MapBatch",
    "MapResult",
    "MessagePart",
    "MessageWithParts",
    "MnesisConfig",
    "MnesisEvent",
    "MnesisSession",
    "MnesisStoreError",
    "OperatorConfig",
    "RecordResult",
    "SessionConfig",
    "SessionNotFoundError",
    "StoreConfig",
    "TextPart",
    "TokenEstimator",
    "TokenUsage",
    "ToolPart",
    "TurnResult",
]
