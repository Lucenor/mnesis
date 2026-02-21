"""LLMMap operator — stateless parallel LLM processing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any, Literal

import structlog
from pydantic import BaseModel

from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.models.config import OperatorConfig
from mnesis.tokens.estimator import TokenEstimator

_DEFAULT_RETRY_GUIDANCE = (
    "Your previous response was not valid JSON. "
    "Return only a JSON object matching the required schema."
)


class MapResult(BaseModel):
    """The result of processing a single item through LLMMap."""

    input: Any
    output: Any | None = None
    success: bool
    error: str | None = None
    error_kind: Literal["timeout", "validation", "llm_error", "schema_error"] | None = None
    attempts: int = 1


class MapBatch(BaseModel):
    """Aggregate result of a full LLMMap.run_all() call."""

    successes: list[MapResult]
    failures: list[MapResult]
    total_attempts: int

    @property
    def total(self) -> int:
        return len(self.successes) + len(self.failures)


class LLMMap:
    """
    Stateless parallel LLM processing over a list of items.

    Each item is processed in an independent, single-turn LLM call with no
    session state. The parent session context is not consumed or affected.
    Results stream back via ``async for`` as they complete.

    Example::

        class ExtractedData(BaseModel):
            title: str
            summary: str
            keywords: list[str]

        llm_map = LLMMap(config.operators, model="anthropic/claude-haiku-3-5")
        async for result in llm_map.run(
            inputs=documents,
            prompt_template="Extract metadata from this document:\\n\\n{{ item }}",
            output_schema=ExtractedData,
        ):
            if result.success:
                print(result.output)
    """

    def __init__(
        self,
        config: OperatorConfig,
        estimator: TokenEstimator | None = None,
        event_bus: EventBus | None = None,
        model: str | None = None,
    ) -> None:
        self._config = config
        self._estimator = estimator or TokenEstimator()
        self._event_bus = event_bus
        self._default_model = model
        self._logger = structlog.get_logger("mnesis.llm_map")

    async def run(
        self,
        inputs: list[Any],
        prompt_template: str,
        output_schema: dict[str, Any] | type[BaseModel],
        model: str | None = None,
        *,
        concurrency: int | None = None,
        max_retries: int | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_secs: float = 60.0,
        retry_guidance: str = _DEFAULT_RETRY_GUIDANCE,
    ) -> AsyncGenerator[MapResult, None]:
        """
        Process inputs in parallel with the given prompt template.

        This is an async generator — iterate with ``async for``, not ``await``.

        Args:
            inputs: List of items to process. Each is rendered into ``{{ item }}``.
            prompt_template: Jinja2 template string. Must reference ``item``.
            output_schema: Expected output shape. Pass a Pydantic ``BaseModel``
                subclass or a JSON Schema ``dict``. When a ``dict`` is passed,
                ``jsonschema`` must be installed (``pip install jsonschema``).
                Responses are validated and retried on failure.
            model: LLM model string in litellm format. Falls back to the model
                set on ``__init__`` if not provided here.
            concurrency: Override config concurrency limit.
            max_retries: Override config max retries per item.
            system_prompt: Optional system prompt for all calls.
            temperature: Sampling temperature. Use 0 for deterministic extraction.
            timeout_secs: Per-item timeout in seconds.
            retry_guidance: Message appended to the prompt on retry after a
                validation failure. Defaults to a generic JSON correction hint.
                Override to avoid leaking schema details into the LLM context.

        Yields:
            MapResult objects as they complete (not in input order).
            Use ``result.input`` to correlate with the original input.

        Raises:
            ValueError: If neither ``model`` nor the constructor ``model`` is set,
                or if ``prompt_template`` does not reference ``item``.
            TypeError: If ``output_schema`` is a type that is not a ``BaseModel`` subclass.
            ImportError: If ``output_schema`` is a dict and ``jsonschema`` is not installed.
        """
        resolved_model = model or self._default_model
        if not resolved_model:
            raise ValueError("model must be provided either to run() or to LLMMap.__init__()")

        # Validate output_schema eagerly before spawning tasks.
        if isinstance(output_schema, type):
            if not issubclass(output_schema, BaseModel):
                raise TypeError(
                    f"output_schema must be a BaseModel subclass, got {output_schema!r}"
                )
        elif isinstance(output_schema, dict):
            # dict path — jsonschema is an optional dependency; fail fast with a helpful message.
            try:
                import jsonschema as _jsonschema  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "jsonschema is required when passing output_schema as a dict. "
                    "Install it with: pip install jsonschema"
                ) from exc
        else:
            raise TypeError(
                "output_schema must be a BaseModel subclass or a dict JSON Schema, "
                f"got {output_schema!r}"
            )

        # Validate template references 'item' via Jinja2 AST (not fragile regex).
        _require_item_variable(prompt_template)

        max_conc = concurrency or self._config.llm_map_concurrency
        retries = max_retries if max_retries is not None else self._config.max_retries

        # Resolve schema to JSON Schema dict
        if isinstance(output_schema, type) and issubclass(output_schema, BaseModel):
            schema_dict = output_schema.model_json_schema()
            pydantic_model: type[BaseModel] | None = output_schema
        else:
            schema_dict = output_schema
            pydantic_model = None

        semaphore = asyncio.Semaphore(max_conc)

        if self._event_bus:
            self._event_bus.publish(
                MnesisEvent.MAP_STARTED, {"total": len(inputs), "model": resolved_model}
            )

        tasks = [
            asyncio.create_task(
                self._process_item(
                    item=item,
                    template_str=prompt_template,
                    schema=schema_dict,
                    pydantic_model=pydantic_model,
                    model=resolved_model,
                    semaphore=semaphore,
                    max_retries=retries,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    timeout=timeout_secs,
                    retry_guidance=retry_guidance,
                )
            )
            for item in inputs
        ]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1
            if self._event_bus:
                self._event_bus.publish(
                    MnesisEvent.MAP_ITEM_COMPLETED,
                    {"completed": completed, "total": len(inputs), "success": result.success},
                )
            yield result

        if self._event_bus:
            self._event_bus.publish(
                MnesisEvent.MAP_COMPLETED, {"total": len(inputs), "completed": completed}
            )

    async def run_all(
        self,
        inputs: list[Any],
        prompt_template: str,
        output_schema: dict[str, Any] | type[BaseModel],
        model: str | None = None,
        *,
        concurrency: int | None = None,
        max_retries: int | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_secs: float = 60.0,
        retry_guidance: str = _DEFAULT_RETRY_GUIDANCE,
    ) -> MapBatch:
        """
        Process all inputs and return an aggregate ``MapBatch``.

        Convenience alternative to ``run()`` for callers who do not need streaming.
        Collects all results before returning.

        Returns:
            MapBatch with ``.successes``, ``.failures``, and ``.total_attempts``.
        """
        successes: list[MapResult] = []
        failures: list[MapResult] = []
        total_attempts = 0

        async for result in self.run(
            inputs=inputs,
            prompt_template=prompt_template,
            output_schema=output_schema,
            model=model,
            concurrency=concurrency,
            max_retries=max_retries,
            system_prompt=system_prompt,
            temperature=temperature,
            timeout_secs=timeout_secs,
            retry_guidance=retry_guidance,
        ):
            total_attempts += result.attempts
            if result.success:
                successes.append(result)
            else:
                failures.append(result)

        return MapBatch(
            successes=successes,
            failures=failures,
            total_attempts=total_attempts,
        )

    async def _process_item(
        self,
        *,
        item: Any,
        template_str: str,
        schema: dict[str, Any],
        pydantic_model: type[BaseModel] | None,
        model: str,
        semaphore: asyncio.Semaphore,
        max_retries: int,
        system_prompt: str | None,
        temperature: float,
        timeout: float,  # noqa: ASYNC109
        retry_guidance: str,
    ) -> MapResult:
        """Process a single item with retry logic."""
        import os

        from jinja2 import Environment as _Env

        env = _Env()
        prompt = env.from_string(template_str).render(item=item)

        if os.environ.get("MNESIS_MOCK_LLM") == "1":
            async with semaphore:
                return MapResult(input=item, output={"_mock": True}, success=True, attempts=1)

        last_error = ""
        last_error_kind: Literal["timeout", "validation", "llm_error", "schema_error"] | None = None

        for attempt in range(1, max_retries + 2):
            async with semaphore:
                try:
                    full_prompt = prompt + (f"\n\n{retry_guidance}" if last_error else "")
                    response_text = await asyncio.wait_for(
                        self._call_llm(
                            model=model,
                            prompt=full_prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                        ),
                        timeout=timeout,
                    )

                    # Parse and validate output
                    parsed, parse_error_kind = self._parse_response(
                        response_text, schema, pydantic_model
                    )
                    if parse_error_kind is not None:
                        last_error = "parse/validation failed"
                        last_error_kind = parse_error_kind
                        self._logger.warning(
                            "llm_map_validation_failed",
                            attempt=attempt,
                            error_kind=parse_error_kind,
                        )
                        continue

                    return MapResult(
                        input=item,
                        output=parsed,
                        success=True,
                        attempts=attempt,
                    )

                except TimeoutError:
                    last_error = f"Timeout after {timeout}s"
                    last_error_kind = "timeout"
                    self._logger.warning("llm_map_timeout", attempt=attempt)
                except Exception as exc:
                    last_error = str(exc)
                    last_error_kind = "llm_error"
                    self._logger.warning("llm_map_error", attempt=attempt, error=last_error)
                    # Exponential backoff for transient errors
                    if attempt <= max_retries:
                        await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))

        return MapResult(
            input=item,
            output=None,
            success=False,
            error=last_error,
            error_kind=last_error_kind,
            attempts=max_retries + 1,
        )

    async def _call_llm(
        self,
        *,
        model: str,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
    ) -> str:
        """Make a single LLM call and return the text response."""
        import os

        if os.environ.get("MNESIS_MOCK_LLM") == "1":
            return json.dumps({"mock": True, "prompt_preview": prompt[:100]})

        import litellm

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_response(
        text: str,
        schema: dict[str, Any],
        pydantic_model: type[BaseModel] | None,
    ) -> tuple[Any, Literal["validation", "schema_error"] | None]:
        """
        Parse and validate the LLM response against the expected schema.

        Returns ``(parsed_value, error_kind)`` where ``error_kind`` is ``None`` on success.
        """
        import re

        json_match = re.search(r"```(?:json)?\n?(.*?)```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None, "validation"

        if pydantic_model is not None:
            try:
                return pydantic_model.model_validate(data), None
            except Exception:
                return None, "validation"

        # Validate against JSON Schema
        import jsonschema

        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError:
            return None, "schema_error"

        return data, None


def _require_item_variable(template_str: str) -> None:
    """
    Raise ValueError if the Jinja2 template does not reference the ``item`` variable.

    Uses Jinja2 AST parsing instead of regex for correctness with complex expressions
    like ``{{ item['key'] }}`` and ``{{ item | upper }}``.
    """
    from jinja2 import Environment, meta

    env = Environment()
    ast = env.parse(template_str)
    variables = meta.find_undeclared_variables(ast)
    if "item" not in variables:
        raise ValueError(
            "prompt_template must reference {{ item }} — "
            "the template does not use the 'item' variable"
        )
