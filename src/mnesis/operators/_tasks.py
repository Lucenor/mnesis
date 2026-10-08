"""Shared task-lifecycle helpers for the parallel operators."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable


async def cancel_and_drain(tasks: Iterable[asyncio.Task[object]]) -> None:
    """Cancel every unfinished task and wait for all of them to settle.

    Used when an operator's async generator is closed early (consumer ``break``,
    exception, or cancellation) so no request keeps running unobserved and no
    "Task exception was never retrieved" warning is emitted. Exceptions from
    the tasks are collected and discarded (``return_exceptions=True``).
    """
    pending = list(tasks)
    for task in pending:
        if not task.done():
            _ = task.cancel()
    if pending:
        _ = await asyncio.gather(*pending, return_exceptions=True)
