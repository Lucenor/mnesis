"""
Mnesis — Lossless Context Management for LLM agentic tasks.

Primary entry point::

    from mnesis import MnesisSession, MnesisConfig

    async with await MnesisSession.create(model="anthropic/claude-opus-4-6") as session:
        result = await session.send("Hello!")
        print(result.text)
"""

from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.files.handler import FileHandleResult, LargeFileHandler
from mnesis.models import (
    CompactionConfig,
    CompactionResult,
    ContextBudget,
    FileConfig,
    FileReference,
    FileRefPart,
    Message,
    MessagePart,
    MessageWithParts,
    MnesisConfig,
    ModelInfo,
    OperatorConfig,
    PruneResult,
    RecordResult,
    StoreConfig,
    SummaryNode,
    TextPart,
    TokenUsage,
    ToolPart,
    TurnResult,
)
from mnesis.operators.agentic_map import AgenticMap, AgentMapResult
from mnesis.operators.llm_map import LLMMap, MapResult
from mnesis.session import MnesisSession, make_id
from mnesis.tokens.estimator import TokenEstimator

__version__ = "0.1.0"

__all__ = [
    "AgentMapResult",
    "AgenticMap",
    "CompactionConfig",
    "CompactionResult",
    "ContextBudget",
    "EventBus",
    "FileConfig",
    "FileHandleResult",
    "FileRefPart",
    "FileReference",
    "LLMMap",
    "LargeFileHandler",
    "MapResult",
    "Message",
    "MessagePart",
    "MessageWithParts",
    "MnesisConfig",
    "MnesisEvent",
    "MnesisSession",
    "ModelInfo",
    "OperatorConfig",
    "PruneResult",
    "RecordResult",
    "StoreConfig",
    "SummaryNode",
    "TextPart",
    "TokenEstimator",
    "TokenUsage",
    "ToolPart",
    "TurnResult",
    "make_id",
]
