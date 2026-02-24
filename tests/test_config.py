"""Tests for Wave 4 config changes."""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

import pytest

import mnesis
from mnesis import MnesisSession
from mnesis.models.config import (
    CompactionConfig,
    FileConfig,
    MnesisConfig,
    ModelInfo,
    SessionConfig,
    StoreConfig,
)


class TestCompactionOutputBudgetRename:
    """M-5: buffer → compaction_output_budget."""

    def test_default_value(self) -> None:
        cfg = CompactionConfig()
        assert cfg.compaction_output_budget == 20_000

    def test_custom_value(self) -> None:
        cfg = CompactionConfig(compaction_output_budget=30_000)
        assert cfg.compaction_output_budget == 30_000

    def test_old_field_name_absent(self) -> None:
        cfg = CompactionConfig()
        assert not hasattr(cfg, "buffer")

    def test_bounds_enforced(self) -> None:
        with pytest.raises(ValueError):
            CompactionConfig(compaction_output_budget=500)  # below ge=1_000
        with pytest.raises(ValueError):
            CompactionConfig(compaction_output_budget=300_000)  # above le=200_000


class TestSessionConfig:
    """M-3: SessionConfig sub-object with doom_loop_threshold."""

    def test_default_on_mnesis_config(self) -> None:
        cfg = MnesisConfig()
        assert isinstance(cfg.session, SessionConfig)
        assert cfg.session.doom_loop_threshold == 3

    def test_custom_threshold(self) -> None:
        cfg = MnesisConfig(session=SessionConfig(doom_loop_threshold=5))
        assert cfg.session.doom_loop_threshold == 5

    def test_doom_loop_not_top_level(self) -> None:
        cfg = MnesisConfig()
        assert not hasattr(cfg, "doom_loop_threshold")

    def test_bounds(self) -> None:
        with pytest.raises(ValueError):
            SessionConfig(doom_loop_threshold=1)  # below ge=2
        with pytest.raises(ValueError):
            SessionConfig(doom_loop_threshold=11)  # above le=10


class TestAdvancedLabels:
    """M-4: Advanced fields labeled in CompactionConfig."""

    def test_soft_threshold_description_has_advanced_prefix(self) -> None:
        info = CompactionConfig.model_fields["soft_threshold_fraction"]
        assert info.description is not None
        assert "[Advanced]" in info.description

    def test_max_compaction_rounds_description_has_advanced_prefix(self) -> None:
        info = CompactionConfig.model_fields["max_compaction_rounds"]
        assert info.description is not None
        assert "[Advanced]" in info.description

    def test_condensation_enabled_description_has_advanced_prefix(self) -> None:
        info = CompactionConfig.model_fields["condensation_enabled"]
        assert info.description is not None
        assert "[Advanced]" in info.description


class TestStoreConfigDbPathExpansion:
    """M-6 / L-3: db_path accepts str | Path, expands ~ eagerly."""

    def test_default_is_expanded(self) -> None:
        cfg = StoreConfig()
        assert "~" not in cfg.db_path
        assert cfg.db_path.startswith("/")

    def test_tilde_expanded(self) -> None:
        cfg = StoreConfig(db_path="~/.mnesis/test.db")
        assert "~" not in cfg.db_path
        assert cfg.db_path.startswith("/")

    def test_path_object_accepted(self, tmp_path: Path) -> None:
        cfg = StoreConfig(db_path=tmp_path / "test.db")
        assert isinstance(cfg.db_path, str)
        assert str(tmp_path) in cfg.db_path

    def test_relative_path_resolved(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        cfg = StoreConfig(db_path="relative/test.db")
        assert cfg.db_path.startswith("/")


class TestFileConfigStorageDirExpansion:
    """L-3: storage_dir accepts str | Path, expands ~ eagerly."""

    def test_default_is_expanded(self) -> None:
        cfg = FileConfig()
        assert "~" not in cfg.storage_dir
        assert cfg.storage_dir.startswith("/")

    def test_tilde_expanded(self) -> None:
        cfg = FileConfig(storage_dir="~/.mnesis/files")
        assert "~" not in cfg.storage_dir

    def test_path_object_accepted(self, tmp_path: Path) -> None:
        cfg = FileConfig(storage_dir=tmp_path / "files")
        assert isinstance(cfg.storage_dir, str)
        assert str(tmp_path) in cfg.storage_dir


class TestModelOverrides:
    """M-2: model_overrides on MnesisConfig."""

    def test_no_overrides_by_default(self) -> None:
        cfg = MnesisConfig()
        assert cfg.model_overrides is None

    def test_overrides_accepted(self) -> None:
        cfg = MnesisConfig(model_overrides={"context_limit": 64_000, "max_output_tokens": 8_192})
        assert cfg.model_overrides == {"context_limit": 64_000, "max_output_tokens": 8_192}


class TestDefaultClassmethodRemoved:
    """L-2: MnesisConfig.default() should be removed."""

    def test_default_classmethod_absent(self) -> None:
        assert not hasattr(MnesisConfig, "default")

    def test_plain_constructor_gives_same_result(self) -> None:
        cfg = MnesisConfig()
        assert isinstance(cfg, MnesisConfig)
        assert cfg.compaction.compaction_output_budget == 20_000


class TestModelInfoGptFoAdded:
    """M-2: gpt-4o and gpt-4o-mini added to heuristic table."""

    def test_gpt4o_has_correct_limits(self) -> None:
        info = ModelInfo.from_model_string("gpt-4o")
        assert info.context_limit == 128_000
        assert info.max_output_tokens == 16_384

    def test_gpt4o_mini_has_correct_limits(self) -> None:
        info = ModelInfo.from_model_string("gpt-4o-mini")
        assert info.context_limit == 128_000
        assert info.max_output_tokens == 16_384

    def test_openai_prefixed_gpt4o(self) -> None:
        info = ModelInfo.from_model_string("openai/gpt-4o")
        assert info.provider_id == "openai"
        assert info.max_output_tokens == 16_384

    def test_gpt4o_mini_takes_priority_over_gpt4o(self) -> None:
        # gpt-4o-mini must not fall into the gpt-4o branch
        mini = ModelInfo.from_model_string("gpt-4o-mini")
        full = ModelInfo.from_model_string("gpt-4o")
        assert mini.max_output_tokens == full.max_output_tokens == 16_384


class TestDualDbPathRaisesValueError:
    """M-6: supplying both db_path and config.store.db_path raises ValueError."""

    async def test_create_raises_on_dual_db_path(self, tmp_path: Path) -> None:
        cfg = MnesisConfig(store=StoreConfig(db_path=str(tmp_path / "a.db")))
        with pytest.raises(ValueError, match="db_path"):
            await MnesisSession.create(
                model="anthropic/claude-opus-4-6",
                config=cfg,
                db_path=str(tmp_path / "b.db"),
            )

    async def test_load_raises_on_dual_db_path(self, tmp_path: Path) -> None:
        cfg = MnesisConfig(store=StoreConfig(db_path=str(tmp_path / "a.db")))
        with pytest.raises(ValueError, match="db_path"):
            await MnesisSession.load(
                session_id="dummy-id",
                config=cfg,
                db_path=str(tmp_path / "b.db"),
            )


class TestVersion:
    """__version__ is sourced from installed package metadata."""

    def test_version_matches_package_metadata(self) -> None:
        assert mnesis.__version__ == version("mnesis")

    def test_version_is_nonempty_string(self) -> None:
        assert isinstance(mnesis.__version__, str)
        assert mnesis.__version__ != ""

    def test_version_has_semver_shape(self) -> None:
        parts = mnesis.__version__.split(".")
        assert len(parts) >= 2, "Expected at least MAJOR.MINOR"
        assert all(re.match(r"^\d", p) for p in parts), "Each segment must start with a digit"
