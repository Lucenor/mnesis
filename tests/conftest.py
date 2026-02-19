"""Shared fixtures for Mnesis tests."""

from __future__ import annotations

import time
from typing import Any

import pytest
import pytest_asyncio

from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.models.config import MnesisConfig, StoreConfig
from mnesis.models.message import Message
from mnesis.store.immutable import ImmutableStore, RawMessagePart
from mnesis.store.pool import StorePool
from mnesis.store.summary_dag import SummaryDAGStore
from mnesis.tokens.estimator import TokenEstimator


@pytest.fixture
def config(tmp_path):
    """MnesisConfig with a temp database path."""
    return MnesisConfig(store=StoreConfig(db_path=str(tmp_path / "test.db")))


@pytest_asyncio.fixture
async def pool(config):
    """StorePool for the test database. Closed after each test."""
    p = StorePool()
    yield p
    await p.close_all()


@pytest_asyncio.fixture
async def store(config, pool):
    """Initialized ImmutableStore backed by a temp SQLite database (pool-managed)."""
    s = ImmutableStore(config.store, pool=pool)
    await s.initialize()
    yield s
    await s.close()  # no-op for pool-managed conn; pool fixture closes the connection


@pytest_asyncio.fixture
async def dag_store(store):
    """SummaryDAGStore backed by the test store."""
    return SummaryDAGStore(store)


@pytest.fixture
def estimator():
    """TokenEstimator using heuristic only (no tiktoken required in tests)."""
    e = TokenEstimator()
    e._force_heuristic = True
    return e


@pytest.fixture
def event_bus():
    """EventBus with a .collected list for asserting events."""
    bus = EventBus()
    collected: list[tuple[MnesisEvent, dict[str, Any]]] = []

    def _collect(event: MnesisEvent, payload: dict[str, Any]) -> None:
        collected.append((event, payload))

    bus.subscribe_all(_collect)
    bus.collected = collected  # type: ignore[attr-defined]
    return bus


@pytest_asyncio.fixture
async def session_id(store):
    """A pre-created session ID in the store."""
    sid = "sess_TEST01"
    await store.create_session(sid, model_id="anthropic/claude-opus-4-6", agent="test")
    return sid


def make_message(
    session_id: str,
    role: str = "user",
    msg_id: str | None = None,
    is_summary: bool = False,
) -> Message:
    """Helper to create a test Message."""
    return Message(
        id=msg_id or f"msg_{int(time.time() * 1000)}_{role[:1]}",
        session_id=session_id,
        role=role,
        is_summary=is_summary,
        mode="compaction" if is_summary else None,
    )


def make_raw_part(
    message_id: str,
    session_id: str,
    part_type: str = "text",
    content: str | None = None,
    part_id: str | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    tool_state: str | None = None,
) -> RawMessagePart:
    """Helper to create a test RawMessagePart."""
    import json

    if content is None:
        if part_type == "text":
            content = json.dumps({"type": "text", "text": "Hello world"})
        elif part_type == "tool":
            content = json.dumps({
                "type": "tool",
                "tool_name": tool_name or "test_tool",
                "tool_call_id": tool_call_id or "call_001",
                "input": {},
                "output": "tool output here",
                "status": {"state": tool_state or "completed"},
            })
        else:
            content = json.dumps({"type": part_type})

    return RawMessagePart(
        id=part_id or f"part_{int(time.time() * 1000)}",
        message_id=message_id,
        session_id=session_id,
        part_type=part_type,
        content=content,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        tool_state=tool_state,
    )
