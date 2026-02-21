"""Configuration models for Mnesis sessions and components."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CompactionConfig(BaseModel):
    """
    Configuration for the compaction engine.

    Everyday knobs::

        CompactionConfig(
            auto=True,
            compaction_output_budget=20_000,
            compaction_model="anthropic/claude-haiku-4-5-20251001",
        )

    # Advanced tuning

    The following fields control internal compaction behaviour and should only
    be changed with a thorough understanding of the compaction loop:

    - ``soft_threshold_fraction`` — when to start early background compaction
    - ``max_compaction_rounds`` — cap on summarise+condense cycles
    - ``condensation_enabled`` — whether to merge accumulated summary nodes
    """

    auto: bool = True
    """Whether to automatically trigger compaction when overflow is detected."""

    compaction_output_budget: int = Field(
        default=20_000,
        ge=1_000,
        le=200_000,
        description="Tokens reserved as compaction headroom (output budget for summary).",
    )

    prune: bool = True
    """Whether to run tool output pruning before compaction."""

    prune_protect_tokens: int = Field(
        default=40_000,
        ge=0,
        description="Token window from the end of history that is never pruned.",
    )

    prune_minimum_tokens: int = Field(
        default=20_000,
        ge=0,
        description="Minimum prunable volume required to actually apply pruning.",
    )

    compaction_model: str | None = Field(
        default=None,
        description="Model to use for compaction summarisation. None = use session model.",
    )

    level2_enabled: bool = True
    """Whether to attempt Level 2 (aggressive) compaction before falling back to Level 3."""

    compaction_prompt: str | None = Field(
        default=None,
        description="Custom system prompt for Level 1/2 LLM summarisation. "
        "None = use the built-in agentic prompt.",
    )

    soft_threshold_fraction: float = Field(
        default=0.6,
        ge=0.1,
        le=0.95,
        description=(
            "[Advanced] Fraction of the usable context window at which background compaction is "
            "triggered early (soft threshold). Must be less than 1.0 so compaction "
            "starts before the hard limit is reached."
        ),
    )

    max_compaction_rounds: int = Field(
        default=10,
        ge=1,
        le=50,
        description=(
            "[Advanced] Maximum number of summarise + condense cycles to run when the context "
            "is still over the hard threshold after an initial compaction pass."
        ),
    )

    condensation_enabled: bool = Field(
        default=True,
        description=(
            "[Advanced] Whether to attempt condensation of accumulated summary nodes after"
            " summarisation."
        ),
    )

    @model_validator(mode="after")
    def validate_prune_thresholds(self) -> CompactionConfig:
        if self.prune_minimum_tokens >= self.prune_protect_tokens:
            raise ValueError("prune_minimum_tokens must be strictly less than prune_protect_tokens")
        return self


class FileConfig(BaseModel):
    """Configuration for large file handling."""

    inline_threshold: int = Field(
        default=10_000,
        ge=100,
        description=(
            "Files estimated to exceed this many tokens are stored externally "
            "and represented as FileRefPart objects rather than inlined."
        ),
    )

    storage_dir: str = Field(
        default_factory=lambda: str(Path.home() / ".mnesis" / "files"),
        description="Directory for external file content storage. Defaults to ~/.mnesis/files/.",
    )

    exploration_summary_model: str | None = Field(
        default=None,
        description="Model to use for LLM-based file summarisation. None = deterministic only.",
    )

    @field_validator("storage_dir", mode="before")
    @classmethod
    def expand_storage_dir(cls, v: Any) -> str:
        return str(Path(str(v)).expanduser().resolve())


class StoreConfig(BaseModel):
    """Configuration for the SQLite persistence layer."""

    db_path: str = Field(
        default_factory=lambda: str(Path.home() / ".mnesis" / "sessions.db"),
        description="Path to the SQLite database file. ~ is expanded at construction time.",
    )

    wal_mode: bool = True
    """Use WAL journal mode for better concurrent read performance."""

    connection_timeout: float = 30.0
    """Seconds to wait for the database connection before raising."""

    @field_validator("db_path", mode="before")
    @classmethod
    def expand_db_path(cls, v: Any) -> str:
        return str(Path(str(v)).expanduser().resolve())


class OperatorConfig(BaseModel):
    """Configuration for LLMMap and AgenticMap operators."""

    llm_map_concurrency: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Maximum concurrent LLM calls in LLMMap.run().",
    )

    agentic_map_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum concurrent sub-agent sessions in AgenticMap.run().",
    )

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum per-item retry attempts on validation or transient errors.",
    )


class SessionConfig(BaseModel):
    """Configuration for session-level behaviour."""

    doom_loop_threshold: int = Field(
        default=3,
        ge=2,
        le=10,
        description=(
            "Number of consecutive identical tool calls that triggers doom loop detection."
        ),
    )


class MnesisConfig(BaseModel):
    """
    Top-level configuration for an Mnesis session.

    All sub-configs have sensible defaults and can be overridden individually.

    Example::

        config = MnesisConfig(
            compaction=CompactionConfig(
                compaction_output_budget=30_000,
                compaction_model="anthropic/claude-haiku-4-5-20251001",
            ),
            file=FileConfig(inline_threshold=5_000),
        )

    To override model context/output limits without constructing ``ModelInfo``
    directly, use ``model_overrides``::

        config = MnesisConfig(
            model_overrides={"context_limit": 128_000, "max_output_tokens": 16_384},
        )
    """

    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    file: FileConfig = Field(default_factory=FileConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    operators: OperatorConfig = Field(default_factory=OperatorConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)

    model_overrides: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Override auto-detected model limits. Supported keys: "
            "``context_limit`` (int), ``max_output_tokens`` (int)."
        ),
    )


class ModelInfo(BaseModel):
    """Resolved model metadata used for budget calculations."""

    model_id: str
    provider_id: str = ""
    context_limit: int = Field(
        default=200_000,
        description="Total input + output token limit for this model.",
    )
    max_output_tokens: int = Field(
        default=8_192,
        description="Maximum output tokens for a single response.",
    )
    encoding: Literal["cl100k_base", "o200k_base", "claude_heuristic", "unknown"] = "cl100k_base"

    @classmethod
    def from_model_string(cls, model: str) -> ModelInfo:
        """
        Create a ModelInfo by heuristically parsing a model string.

        Supports litellm-style strings like ``anthropic/claude-opus-4-6``,
        ``gpt-4o``, ``openai/gpt-4-turbo``, etc.
        """
        lower = model.lower()
        provider = ""
        model_name = lower

        if "/" in lower:
            provider, model_name = lower.split("/", 1)

        # Context limits and output limits by model family
        if "claude-opus" in model_name or "claude-3-opus" in model_name:
            return cls(
                model_id=model,
                provider_id=provider or "anthropic",
                context_limit=200_000,
                max_output_tokens=32_000,
                encoding="claude_heuristic",
            )
        if "claude" in model_name:
            return cls(
                model_id=model,
                provider_id=provider or "anthropic",
                context_limit=200_000,
                max_output_tokens=8_192,
                encoding="claude_heuristic",
            )
        if "o1" in model_name or "o3" in model_name:
            return cls(
                model_id=model,
                provider_id=provider or "openai",
                context_limit=200_000,
                max_output_tokens=100_000,
                encoding="o200k_base",
            )
        # gpt-4o-mini must be checked before gpt-4o to avoid prefix match
        if "gpt-4o-mini" in model_name:
            return cls(
                model_id=model,
                provider_id=provider or "openai",
                context_limit=128_000,
                max_output_tokens=16_384,
                encoding="o200k_base",
            )
        if "gpt-4o" in model_name:
            return cls(
                model_id=model,
                provider_id=provider or "openai",
                context_limit=128_000,
                max_output_tokens=16_384,
                encoding="o200k_base",
            )
        if "gpt-4" in model_name or "gpt-3" in model_name:
            return cls(
                model_id=model,
                provider_id=provider or "openai",
                context_limit=128_000,
                max_output_tokens=4_096,
                encoding="cl100k_base",
            )
        if "gemini" in model_name:
            return cls(
                model_id=model,
                provider_id=provider or "google",
                context_limit=1_000_000,
                max_output_tokens=8_192,
                encoding="cl100k_base",
            )
        # Safe default for unknown models
        return cls(
            model_id=model,
            provider_id=provider,
            context_limit=128_000,
            max_output_tokens=4_096,
            encoding="cl100k_base",
        )
