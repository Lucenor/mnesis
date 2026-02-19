"""
Mnesis — Lossless Context Management for LLM agentic tasks.

Primary entry point::

    from mnesis import MnesisSession, MnesisConfig

    async with await MnesisSession.create(model="anthropic/claude-opus-4-6") as session:
        result = await session.send("Hello!")
        print(result.text)
"""

from mnesis.session import MnesisSession, make_id
from mnesis.models import (
    MnesisConfig,
    CompactionConfig,
    FileConfig,
    StoreConfig,
    OperatorConfig,
    ModelInfo,
    TextPart,
    ToolPart,
    FileRefPart,
    MessagePart,
    Message,
    MessageWithParts,
    TokenUsage,
    TurnResult,
    CompactionResult,
    PruneResult,
    ContextBudget,
    SummaryNode,
    FileReference,
)
from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.operators.llm_map import LLMMap, MapResult
from mnesis.operators.agentic_map import AgenticMap, AgentMapResult
from mnesis.files.handler import LargeFileHandler, FileHandleResult
from mnesis.tokens.estimator import TokenEstimator

__version__ = "0.1.0"

__all__ = [
    # Core
    "MnesisSession",
    "make_id",
    # Config
    "MnesisConfig",
    "CompactionConfig",
    "FileConfig",
    "StoreConfig",
    "OperatorConfig",
    "ModelInfo",
    # Models
    "TextPart",
    "ToolPart",
    "FileRefPart",
    "MessagePart",
    "Message",
    "MessageWithParts",
    "TokenUsage",
    "TurnResult",
    "CompactionResult",
    "PruneResult",
    "ContextBudget",
    "SummaryNode",
    "FileReference",
    # Events
    "EventBus",
    "MnesisEvent",
    # Operators
    "LLMMap",
    "MapResult",
    "AgenticMap",
    "AgentMapResult",
    # Files
    "LargeFileHandler",
    "FileHandleResult",
    # Tokens
    "TokenEstimator",
]
