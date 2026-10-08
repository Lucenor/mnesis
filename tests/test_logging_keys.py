"""D1: no structlog call may pass a key that structlog's own processors overwrite."""

from __future__ import annotations

import ast
from pathlib import Path

import structlog.testing

import mnesis
from mnesis.compaction.engine import CompactionEngine
from mnesis.models.config import MnesisConfig, StoreConfig
from tests.conftest import make_message, make_raw_part

# ``level`` is rewritten by ``add_log_level``, ``event`` is the message itself,
# ``timestamp``/``logger``/``logger_name`` by the timestamper and logger-name processors.
RESERVED = {"level", "event", "timestamp", "logger", "logger_name"}
LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "bind", "new"}


def test_no_reserved_structlog_keys_passed_to_loggers():
    offenders: list[str] = []
    root = Path(mnesis.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in LOG_METHODS:
                continue
            target = ast.unparse(node.func.value).lower()
            if "log" not in target:
                continue
            offenders.extend(
                f"{path.relative_to(root)}:{node.lineno} {kw.arg}="
                for kw in node.keywords
                if kw.arg in RESERVED
            )
    assert offenders == []


async def test_compaction_logs_carry_the_compaction_level(
    session_id, store, dag_store, estimator, event_bus, tmp_path, monkeypatch
):
    monkeypatch.setenv("MNESIS_MOCK_LLM", "1")
    for i in range(8):
        msg = make_message(
            session_id, role="user" if i % 2 == 0 else "assistant", msg_id=f"msg_lk_{i}"
        )
        await store.append_message(msg)
        await store.append_part(make_raw_part(msg.id, session_id, part_id=f"part_lk_{i}"))
    engine = CompactionEngine(
        store,
        dag_store,
        estimator,
        event_bus,
        MnesisConfig(store=StoreConfig(db_path=str(tmp_path / "lk.db"))),
        session_model="anthropic/claude-haiku-4-5",
    )
    with structlog.testing.capture_logs() as logs:
        result = await engine.run_compaction(session_id)

    assert result.level_used > 0
    by_event = {entry["event"]: entry for entry in logs}
    assert by_event["compaction_completed"]["compaction_level"] == result.level_used
    assert by_event["summary_node_inserted"]["compaction_level"] == result.level_used
    assert all("level" not in entry for entry in logs)
