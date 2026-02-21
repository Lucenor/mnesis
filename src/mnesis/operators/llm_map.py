"""LLMMap operator — stateless parallel LLM processing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import structlog
from jinja2 import Template
from pydantic import BaseModel

from mnesis.events.bus import EventBus, MnesisEvent
from mnesis.models.config import OperatorConfig
from mnesis.tokens.estimator import TokenEstimator


class MapResult(BaseModel):
    """The result of processing a single item through LLMMap."""

    input: Any
    output: Any | None = None
    success: bool
    error: str | None = None
    attempts: int = 1


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

        llm_map = LLMMap(config.operators)
        async for result in llm_map.run(
            inputs=documents,
            prompt_template="Extract metadata from this document:\\n\\n{{ item }}",
            output_schema=ExtractedData,
            model="anthropic/claude-haiku-3-5",
        ):
            if result.success:
                print(result.output)
    """

    def __init__(
        self,
        config: OperatorConfig,
        estimator: TokenEstimator | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._config = config
        self._estimator = estimator or TokenEstimator()
        self._event_bus = event_bus
        self._logger = structlog.get_logger("mnesis.llm_map")

    async def run(
        self,
        inputs: list[Any],
        prompt_template: str,
        output_schema: dict[str, Any] | type[BaseModel],
        model: str,
        *,
        concurrency: int | None = None,
        max_retries: int | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        timeout_secs: float = 60.0,
    ) -> AsyncIterator[MapResult]:
        """
        Process inputs in parallel with the given prompt template.

        Args:
            inputs: List of items to process. Each is rendered into ``{{ item }}``.
            prompt_template: Jinja2 template string. Must contain ``{{ item }}``.
            output_schema: Expected output shape. Pass a Pydantic ``BaseModel``
                subclass or a JSON Schema ``dict``. When a ``dict`` is passed,
                ``jsonschema`` must be installed (``pip install jsonschema``).
                Responses are validated and retried on failure.
            model: LLM model string in litellm format.
            concurrency: Override config concurrency limit.
            max_retries: Override config max retries per item.
            system_prompt: Optional system prompt for all calls.
            temperature: Sampling temperature. Use 0 for deterministic extraction.
            timeout_secs: Per-item timeout in seconds.

        Yields:
            MapResult objects as they complete (not in input order).
            Use ``result.input`` to correlate with the original input.

        Raises:
            TypeError: If ``output_schema`` is a type that is not a ``BaseModel`` subclass.
            ImportError: If ``output_schema`` is a dict and ``jsonschema`` is not installed.
            ValueError: If ``prompt_template`` does not contain ``{{ item }}``.
        """
        import re

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

        if not re.search(r"\{\{[^}]*\bitem\b[^}]*\}\}", prompt_template):
            raise ValueError("prompt_template must contain {{ item }}")

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
        template = Template(prompt_template)

        if self._event_bus:
            self._event_bus.publish(MnesisEvent.MAP_STARTED, {"total": len(inputs), "model": model})

        tasks = [
            asyncio.create_task(
                self._process_item(
                    item=item,
                    template=template,
                    schema=schema_dict,
                    pydantic_model=pydantic_model,
                    model=model,
                    semaphore=semaphore,
                    max_retries=retries,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    timeout=timeout_secs,
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

    async def _process_item(
        self,
        *,
        item: Any,
        template: Template,
        schema: dict[str, Any],
        pydantic_model: type[BaseModel] | None,
        model: str,
        semaphore: asyncio.Semaphore,
        max_retries: int,
        system_prompt: str | None,
        temperature: float,
        timeout: float,  # noqa: ASYNC109
    ) -> MapResult:
        """Process a single item with retry logic."""
        import os

        prompt = template.render(item=item)

        if os.environ.get("MNESIS_MOCK_LLM") == "1":
            async with semaphore:
                return MapResult(input=item, output={"_mock": True}, success=True, attempts=1)

        last_error = ""

        for attempt in range(1, max_retries + 2):
            async with semaphore:
                try:
                    response_text = await asyncio.wait_for(
                        self._call_llm(
                            model=model,
                            prompt=prompt
                            + (f"\n\nPrevious error: {last_error}" if last_error else ""),
                            system_prompt=system_prompt,
                            temperature=temperature,
                        ),
                        timeout=timeout,
                    )

                    # Parse and validate output
                    parsed = self._parse_response(response_text, schema, pydantic_model)
                    return MapResult(
                        input=item,
                        output=parsed,
                        success=True,
                        attempts=attempt,
                    )

                except TimeoutError:
                    last_error = f"Timeout after {timeout}s"
                    self._logger.warning("llm_map_timeout", attempt=attempt)
                except (json.JSONDecodeError, ValueError) as exc:
                    last_error = str(exc)
                    self._logger.warning(
                        "llm_map_validation_failed", attempt=attempt, error=last_error
                    )
                except Exception as exc:
                    last_error = str(exc)
                    self._logger.warning("llm_map_error", attempt=attempt, error=last_error)
                    # Exponential backoff for transient errors
                    if attempt <= max_retries:
                        await asyncio.sleep(min(0.5 * (2 ** (attempt - 1)), 8.0))

        return MapResult(
            input=item,
            output=None,
            success=False,
            error=last_error,
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
    ) -> Any:
        """Parse and validate the LLM response against the expected schema."""
        # Extract JSON from response (handle markdown code fences)
        import re

        json_match = re.search(r"```(?:json)?\n?(.*?)```", text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        data = json.loads(text)

        if pydantic_model is not None:
            return pydantic_model.model_validate(data)

        # Validate against JSON Schema
        import jsonschema

        jsonschema.validate(data, schema)
        return data
