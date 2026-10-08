"""Shared retry policy for transient LLM errors.

Used by ``MnesisSession.send()`` and by the compaction engine so that
:class:`~mnesis.models.config.RetryConfig` governs every LLM call Mnesis makes.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

import structlog

from mnesis.models.config import RetryConfig


class RetriesExhaustedError(Exception):
    """A retryable LLM error persisted through every configured Mnesis retry.

    ``__cause__`` is the last underlying error. Raised only when Mnesis owns
    retries (``max_retries > 0``); it signals an outage, so callers should stop
    sending further LLM calls rather than start another retry sequence.
    """


def is_retryable(exc: BaseException) -> bool:
    """Return True if ``exc`` is a transient LLM error worth retrying.

    Retryable errors are transient provider-side problems (rate limits,
    server errors, timeouts, connection issues).  Non-retryable errors
    indicate caller mistakes (bad credentials, bad request, context
    window exceeded) that will not resolve on retry.

    Any exception type not in either known list is treated as
    non-retryable (fail-fast) to avoid hiding unexpected errors.
    """
    try:
        from litellm.exceptions import (
            APIConnectionError,
            InternalServerError,
            RateLimitError,
            ServiceUnavailableError,
            Timeout,
        )
    except ImportError:
        return False

    return isinstance(
        exc,
        (
            RateLimitError,
            InternalServerError,
            ServiceUnavailableError,
            Timeout,
            APIConnectionError,
        ),
    )


def backoff_delay(cfg: RetryConfig, attempt: int) -> float:
    """Exponential backoff (with optional jitter) before retry ``attempt + 1``."""
    return min(
        cfg.base_delay * (2.0**attempt)
        + (random.uniform(0, cfg.base_delay) if cfg.jitter else 0.0),
        cfg.max_delay,
    )


async def _sleep_unless_aborted(delay: float, abort: asyncio.Event | None) -> None:
    """Sleep ``delay`` seconds; raise ``CancelledError`` as soon as ``abort`` is set."""
    if abort is None:
        await asyncio.sleep(delay)
        return
    sleeper = asyncio.ensure_future(asyncio.sleep(delay))
    waiter = asyncio.ensure_future(abort.wait())
    try:
        _ = await asyncio.wait({sleeper, waiter}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        sleeper.cancel()
        waiter.cancel()
    if abort.is_set():
        raise asyncio.CancelledError("Compaction aborted during retry backoff")


async def call_with_retry[T](
    call: Callable[[], Awaitable[T]],
    cfg: RetryConfig,
    *,
    abort: asyncio.Event | None = None,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> T:
    """Await ``call()``, retrying transient errors per ``cfg``.

    Non-retryable errors propagate unchanged. A retryable error that outlasts
    ``cfg.max_retries`` (> 0) raises :class:`RetriesExhaustedError`. If ``abort``
    is set during backoff the wait ends immediately with ``asyncio.CancelledError``; external task
    cancellation also propagates.
    """
    log = logger or structlog.get_logger("mnesis.retry")
    attempt = 0
    while True:
        try:
            return await call()
        except Exception as exc:
            if not is_retryable(exc):
                raise
            if attempt >= cfg.max_retries:
                if cfg.max_retries == 0:
                    raise  # Mnesis retries disabled: LiteLLM's own retries already ran
                raise RetriesExhaustedError(str(exc)) from exc
            delay = backoff_delay(cfg, attempt)
            log.warning(
                "llm_call_retrying",
                attempt=attempt + 1,
                max_retries=cfg.max_retries,
                delay=delay,
                error=str(exc),
            )
            await _sleep_unless_aborted(delay, abort)
            attempt += 1
