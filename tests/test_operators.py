"""Tests for LLMMap and AgenticMap operators."""

from __future__ import annotations

import asyncio
import json

import pytest
from jinja2 import Environment as _JinjaEnv
from pydantic import BaseModel

from mnesis.events.bus import EventBus
from mnesis.models.config import OperatorConfig
from mnesis.operators.llm_map import LLMMap, MapBatch


@pytest.fixture(autouse=True)
def mock_llm_env(monkeypatch):
    """Enable mock LLM mode for all operator tests."""
    monkeypatch.setenv("MNESIS_MOCK_LLM", "1")


@pytest.fixture
def op_config():
    return OperatorConfig(llm_map_concurrency=4, agentic_map_concurrency=2, max_retries=2)


class OutputSchema(BaseModel):
    summary: str
    keywords: list[str]


class TestLLMMap:
    async def test_processes_all_items(self, op_config):
        """LLMMap yields a result for every input item."""
        llm_map = LLMMap(op_config)
        inputs = [f"Document {i}" for i in range(5)]
        results = []
        async for result in llm_map.run(
            inputs=inputs,
            prompt_template="Summarize: {{ item }}",
            output_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
            model="anthropic/claude-haiku-3-5",
        ):
            results.append(result)

        assert len(results) == 5

    async def test_invalid_template_raises(self, op_config):
        """Missing item variable in template raises ValueError."""
        llm_map = LLMMap(op_config)
        with pytest.raises(ValueError, match="item"):
            async for _ in llm_map.run(
                inputs=["test"],
                prompt_template="No item placeholder here",
                output_schema={},
                model="test-model",
            ):
                pass

    async def test_invalid_jinja2_syntax_raises_value_error(self, op_config):
        """Invalid Jinja2 syntax in the template raises ValueError (not TemplateSyntaxError)."""
        llm_map = LLMMap(op_config)
        with pytest.raises(ValueError, match="Invalid Jinja2 template syntax"):
            async for _ in llm_map.run(
                inputs=["test"],
                prompt_template="{{ item | ",  # unclosed filter — invalid syntax
                output_schema={},
                model="test-model",
            ):
                pass

    async def test_template_with_filter_accepted(self, op_config):
        """Jinja2 templates with filters ({{ item | upper }}) are accepted."""
        llm_map = LLMMap(op_config)
        results = []
        async for result in llm_map.run(
            inputs=["hello"],
            prompt_template="Process: {{ item | upper }}",
            output_schema=OutputSchema,
            model="test-model",
        ):
            results.append(result)
        assert len(results) == 1

    async def test_template_with_subscript_accepted(self, op_config):
        """Jinja2 templates with subscripts ({{ item['key'] }}) are accepted."""
        llm_map = LLMMap(op_config)
        results = []
        async for result in llm_map.run(
            inputs=[{"key": "value"}],
            prompt_template="Process: {{ item['key'] }}",
            output_schema=OutputSchema,
            model="test-model",
        ):
            results.append(result)
        assert len(results) == 1

    async def test_result_has_input_reference(self, op_config):
        """MapResult.input echoes back the original input."""
        llm_map = LLMMap(op_config)
        inputs = ["apple", "banana", "cherry"]
        result_inputs = []
        async for result in llm_map.run(
            inputs=inputs,
            prompt_template="Process: {{ item }}",
            output_schema={"type": "object"},
            model="anthropic/claude-haiku-3-5",
        ):
            result_inputs.append(result.input)

        assert set(result_inputs) == set(inputs)

    async def test_output_schema_non_basemodel_type_raises(self, op_config):
        """Passing a non-BaseModel type as output_schema raises TypeError."""
        llm_map = LLMMap(op_config)
        with pytest.raises(TypeError, match="BaseModel subclass"):
            async for _ in llm_map.run(
                inputs=["test"],
                prompt_template="Process: {{ item }}",
                output_schema=dict,  # type: ignore[arg-type]
                model="test-model",
            ):
                pass

    async def test_output_schema_unsupported_value_raises(self, op_config):
        """Passing a non-type, non-dict value (e.g. a list) raises TypeError."""
        llm_map = LLMMap(op_config)
        with pytest.raises(TypeError, match="BaseModel subclass or a dict JSON Schema"):
            async for _ in llm_map.run(
                inputs=["test"],
                prompt_template="Process: {{ item }}",
                output_schema=["not", "a", "schema"],  # type: ignore[arg-type]
                model="test-model",
            ):
                pass

    async def test_output_schema_pydantic_model_accepted(self, op_config):
        """A valid BaseModel subclass is accepted without error."""
        llm_map = LLMMap(op_config)
        results = []
        async for result in llm_map.run(
            inputs=["test"],
            prompt_template="Process: {{ item }}",
            output_schema=OutputSchema,
            model="test-model",
        ):
            results.append(result)
        assert len(results) == 1

    async def test_output_schema_dict_missing_jsonschema_raises(self, op_config, monkeypatch):
        """Passing a dict schema raises ImportError when jsonschema is not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "jsonschema":
                raise ImportError("No module named 'jsonschema'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        llm_map = LLMMap(op_config)
        with pytest.raises(ImportError, match="pip install jsonschema"):
            async for _ in llm_map.run(
                inputs=["test"],
                prompt_template="Process: {{ item }}",
                output_schema={"type": "object"},
                model="test-model",
            ):
                pass

    async def test_model_default_on_init(self, op_config):
        """Model set on __init__ is used when run() does not specify one."""
        llm_map = LLMMap(op_config, model="test-model")
        results = []
        async for result in llm_map.run(
            inputs=["x"],
            prompt_template="Process: {{ item }}",
            output_schema=OutputSchema,
            # model not passed here
        ):
            results.append(result)
        assert len(results) == 1

    async def test_model_required_if_not_on_init(self, op_config):
        """ValueError raised if model is absent from both __init__ and run()."""
        llm_map = LLMMap(op_config)
        with pytest.raises(ValueError, match="model"):
            async for _ in llm_map.run(
                inputs=["x"],
                prompt_template="Process: {{ item }}",
                output_schema=OutputSchema,
            ):
                pass

    async def test_run_model_overrides_init_default(self, op_config):
        """model passed to run() takes precedence over the __init__ default."""
        llm_map = LLMMap(op_config, model="init-model")
        results = []
        async for result in llm_map.run(
            inputs=["x"],
            prompt_template="Process: {{ item }}",
            output_schema=OutputSchema,
            model="run-model",
        ):
            results.append(result)
        assert len(results) == 1

    async def test_map_result_has_error_kind_none_on_success(self, op_config):
        """Successful results have error_kind=None."""
        llm_map = LLMMap(op_config)
        results = []
        async for result in llm_map.run(
            inputs=["x"],
            prompt_template="Process: {{ item }}",
            output_schema=OutputSchema,
            model="test-model",
        ):
            results.append(result)
        assert results[0].success is True
        assert results[0].error_kind is None

    async def test_run_all_returns_map_batch(self, op_config):
        """run_all() returns a MapBatch with successes/failures."""
        llm_map = LLMMap(op_config)
        batch = await llm_map.run_all(
            inputs=["a", "b", "c"],
            prompt_template="Process: {{ item }}",
            output_schema=OutputSchema,
            model="test-model",
        )
        assert isinstance(batch, MapBatch)
        assert batch.total == 3
        assert len(batch.successes) == 3
        assert len(batch.failures) == 0
        assert batch.total_attempts == 3

    async def test_concurrency_limit_respected(self, op_config, monkeypatch):
        """At most N concurrent calls are made at once."""
        # Mock mode short-circuits before _call_llm; disable it so the
        # instrumented call (and therefore the semaphore) is exercised.
        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)
        active: list[int] = []
        max_concurrent: list[int] = [0]

        async def tracked_call(self, **kwargs):
            active.append(1)
            max_concurrent[0] = max(max_concurrent[0], len(active))
            await asyncio.sleep(0.05)
            active.pop()
            return '{"result": "ok"}'

        llm_map = LLMMap(op_config)
        llm_map._call_llm = lambda **kw: tracked_call(llm_map, **kw)  # type: ignore

        inputs = [f"item_{i}" for i in range(8)]
        results = []
        async for result in llm_map.run(
            inputs=inputs,
            prompt_template="Process {{ item }}",
            output_schema={"type": "object"},
            model="test-model",
        ):
            results.append(result)

        assert len(results) == 8
        # The semaphore is honored (never exceeded) and, with delayed calls
        # over 8 items, the configured limit is actually reached.
        assert max_concurrent[0] == op_config.llm_map_concurrency

    def test_parse_response_json_decode_error_returns_validation(self, op_config):
        """_parse_response returns validation error kind for invalid JSON."""
        llm_map = LLMMap(op_config)
        result, error_kind = llm_map._parse_response("not json at all", {}, None)
        assert result is None
        assert error_kind == "validation"

    def test_parse_response_pydantic_validation_failure(self, op_config):
        """_parse_response returns validation error kind when pydantic validation fails."""
        llm_map = LLMMap(op_config)
        # Valid JSON but missing required fields for OutputSchema
        result, error_kind = llm_map._parse_response('{"wrong_field": 123}', {}, OutputSchema)
        assert result is None
        assert error_kind == "validation"

    def test_parse_response_pydantic_success(self, op_config):
        """_parse_response returns parsed model and None error for valid data."""
        llm_map = LLMMap(op_config)
        data = {"summary": "hello", "keywords": ["a", "b"]}
        result, error_kind = llm_map._parse_response(json.dumps(data), {}, OutputSchema)
        assert error_kind is None
        assert isinstance(result, OutputSchema)
        assert result.summary == "hello"

    def test_parse_response_json_schema_validation_failure(self, op_config):
        """_parse_response returns schema_error when jsonschema validation fails."""
        llm_map = LLMMap(op_config)
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        result, error_kind = llm_map._parse_response('{"other": "value"}', schema, None)
        assert result is None
        assert error_kind == "schema_error"

    def test_parse_response_json_schema_success(self, op_config):
        """_parse_response returns data and None error for valid JSON schema match."""
        llm_map = LLMMap(op_config)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result, error_kind = llm_map._parse_response('{"name": "test"}', schema, None)
        assert error_kind is None
        assert result == {"name": "test"}

    def test_parse_response_strips_markdown_fences(self, op_config):
        """_parse_response extracts JSON from markdown code fences."""
        llm_map = LLMMap(op_config)
        text = 'Here is the result:\n```json\n{"name": "test"}\n```'
        result, error_kind = llm_map._parse_response(text, {"type": "object"}, None)
        assert error_kind is None
        assert result == {"name": "test"}

    async def test_event_bus_published_on_run(self, op_config):
        """Event bus receives MAP_STARTED, MAP_ITEM_COMPLETED, and MAP_COMPLETED events."""
        from mnesis.events.bus import MnesisEvent

        bus = EventBus()
        events: list[MnesisEvent] = []

        # Use a synchronous handler — async handlers are fire-and-forget background tasks
        # so they may not have run yet when we check at the end of the test.
        def collector(event: MnesisEvent, data: object) -> None:
            events.append(event)

        bus.subscribe(MnesisEvent.MAP_STARTED, collector)
        bus.subscribe(MnesisEvent.MAP_ITEM_COMPLETED, collector)
        bus.subscribe(MnesisEvent.MAP_COMPLETED, collector)

        llm_map = LLMMap(op_config, event_bus=bus)
        async for _ in llm_map.run(
            inputs=["x"],
            prompt_template="Process: {{ item }}",
            output_schema=OutputSchema,
            model="test-model",
        ):
            pass

        assert MnesisEvent.MAP_STARTED in events
        assert MnesisEvent.MAP_ITEM_COMPLETED in events
        assert MnesisEvent.MAP_COMPLETED in events

    async def test_process_item_non_mock_success(self, op_config, monkeypatch):
        """_process_item uses _call_llm when MNESIS_MOCK_LLM is not set."""
        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)

        llm_map = LLMMap(op_config)
        valid_response = json.dumps({"summary": "ok", "keywords": ["a"]})

        calls: list[dict] = []

        async def fake_call_llm(**kwargs):
            calls.append(kwargs)
            return valid_response

        import asyncio as _asyncio

        semaphore = _asyncio.Semaphore(1)
        _tmpl = _JinjaEnv().from_string("Process: {{ item }}")
        # Replace _call_llm before the only invocation: no real LiteLLM request.
        llm_map._call_llm = fake_call_llm  # type: ignore
        result = await llm_map._process_item(
            item="test",
            compiled_template=_tmpl,
            schema=OutputSchema.model_json_schema(),
            pydantic_model=OutputSchema,
            model="test-model",
            semaphore=semaphore,
            max_retries=0,
            system_prompt=None,
            temperature=0.0,
            timeout=30.0,
            retry_guidance="retry",
        )
        assert result.success is True
        assert len(calls) == 1

    async def test_process_item_timeout_sets_error_kind(self, op_config, monkeypatch):
        """_process_item sets error_kind='timeout' on TimeoutError."""
        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)

        async def slow_call(**kwargs):
            await asyncio.sleep(100)
            return "{}"

        import asyncio as _asyncio

        semaphore = _asyncio.Semaphore(1)
        llm_map = LLMMap(op_config)
        llm_map._call_llm = slow_call  # type: ignore

        result = await llm_map._process_item(
            item="test",
            compiled_template=_JinjaEnv().from_string("Process: {{ item }}"),
            schema={},
            pydantic_model=None,
            model="test-model",
            semaphore=semaphore,
            max_retries=0,
            system_prompt=None,
            temperature=0.0,
            timeout=0.001,
            retry_guidance="retry",
        )
        assert result.success is False
        assert result.error_kind == "timeout"

    async def test_process_item_llm_error_sets_error_kind(self, op_config, monkeypatch):
        """_process_item sets error_kind='llm_error' on unexpected exception."""
        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)

        async def error_call(**kwargs):
            raise RuntimeError("LLM unavailable")

        import asyncio as _asyncio

        semaphore = _asyncio.Semaphore(1)
        llm_map = LLMMap(op_config)
        llm_map._call_llm = error_call  # type: ignore

        result = await llm_map._process_item(
            item="test",
            compiled_template=_JinjaEnv().from_string("Process: {{ item }}"),
            schema={},
            pydantic_model=None,
            model="test-model",
            semaphore=semaphore,
            max_retries=0,
            system_prompt=None,
            temperature=0.0,
            timeout=30.0,
            retry_guidance="retry",
        )
        assert result.success is False
        assert result.error_kind == "llm_error"

    async def test_process_item_validation_retry(self, op_config, monkeypatch):
        """_process_item retries on parse failure and includes retry_guidance in prompt."""
        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)

        call_count = 0

        async def alternating_call(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "not valid json"
            return json.dumps({"summary": "ok", "keywords": []})

        import asyncio as _asyncio

        semaphore = _asyncio.Semaphore(1)
        llm_map = LLMMap(op_config)
        llm_map._call_llm = alternating_call  # type: ignore

        result = await llm_map._process_item(
            item="test",
            compiled_template=_JinjaEnv().from_string("Process: {{ item }}"),
            schema=OutputSchema.model_json_schema(),
            pydantic_model=OutputSchema,
            model="test-model",
            semaphore=semaphore,
            max_retries=2,
            system_prompt=None,
            temperature=0.0,
            timeout=30.0,
            retry_guidance="Please fix the JSON.",
        )
        assert result.success is True
        assert call_count == 2

    async def test_call_llm_mock_mode_returns_json(self, op_config):
        """_call_llm returns mock JSON when MNESIS_MOCK_LLM=1 (env already set by fixture)."""
        llm_map = LLMMap(op_config)
        result = await llm_map._call_llm(
            model="test-model",
            prompt="hello",
            system_prompt=None,
            temperature=0.0,
        )
        data = json.loads(result)
        assert data["mock"] is True

    async def test_call_llm_mock_mode_with_system_prompt(self, op_config):
        """_call_llm with system_prompt in mock mode still returns mock JSON."""
        llm_map = LLMMap(op_config)
        result = await llm_map._call_llm(
            model="test-model",
            prompt="hello",
            system_prompt="You are helpful.",
            temperature=0.0,
        )
        # In mock mode, system_prompt does not affect output
        data = json.loads(result)
        assert data["mock"] is True

    async def test_process_item_llm_error_retries_with_backoff(self, op_config, monkeypatch):
        """_process_item retries after llm_error and exponential backoff is applied."""
        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)

        call_count = 0

        async def fail_then_succeed(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return json.dumps({"summary": "recovered", "keywords": []})

        import asyncio as _asyncio

        # Patch sleep to avoid actually waiting
        async def fast_sleep(_: float) -> None:
            pass

        monkeypatch.setattr(_asyncio, "sleep", fast_sleep)

        semaphore = _asyncio.Semaphore(1)
        llm_map = LLMMap(op_config)
        llm_map._call_llm = fail_then_succeed  # type: ignore

        result = await llm_map._process_item(
            item="test",
            compiled_template=_JinjaEnv().from_string("Process: {{ item }}"),
            schema=OutputSchema.model_json_schema(),
            pydantic_model=OutputSchema,
            model="test-model",
            semaphore=semaphore,
            max_retries=1,
            system_prompt=None,
            temperature=0.0,
            timeout=30.0,
            retry_guidance="retry",
        )
        assert result.success is True
        assert call_count == 2

    async def test_run_all_counts_failures(self, op_config, monkeypatch):
        """run_all() populates failures list when items fail."""
        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)

        async def always_fail(**kwargs):
            raise RuntimeError("fail")

        llm_map = LLMMap(op_config)
        llm_map._call_llm = always_fail  # type: ignore

        batch = await llm_map.run_all(
            inputs=["a", "b"],
            prompt_template="Process: {{ item }}",
            output_schema=OutputSchema,
            model="test-model",
            max_retries=0,
        )
        assert len(batch.failures) == 2
        assert len(batch.successes) == 0

    def test_parse_response_schema_error_returns_schema_error(self, op_config):
        """_parse_response returns schema_error for an invalid schema (SchemaError)."""
        llm_map = LLMMap(op_config)
        # An invalid JSON Schema (unknown type value) causes jsonschema.SchemaError.
        bad_schema = {"type": "not-a-valid-type"}
        result, error_kind = llm_map._parse_response('{"key": "value"}', bad_schema, None)
        assert result is None
        assert error_kind == "schema_error"

    async def test_retry_guidance_not_appended_on_timeout(self, op_config, monkeypatch):
        """retry_guidance is NOT appended to the prompt on a timeout failure."""
        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)

        received_prompts: list[str] = []

        async def slow_then_succeed(**kwargs):
            received_prompts.append(kwargs.get("prompt", ""))
            if len(received_prompts) == 1:
                await asyncio.sleep(100)
            return json.dumps({"summary": "ok", "keywords": []})

        import asyncio as _asyncio

        semaphore = _asyncio.Semaphore(1)
        llm_map = LLMMap(op_config)
        llm_map._call_llm = slow_then_succeed  # type: ignore

        result = await llm_map._process_item(
            item="test",
            compiled_template=_JinjaEnv().from_string("Process: {{ item }}"),
            schema=OutputSchema.model_json_schema(),
            pydantic_model=OutputSchema,
            model="test-model",
            semaphore=semaphore,
            max_retries=1,
            system_prompt=None,
            temperature=0.0,
            timeout=0.001,
            retry_guidance="DO NOT ADD THIS",
        )
        # On retry after timeout, retry_guidance must NOT appear in the prompt.
        assert result.success is True
        assert all("DO NOT ADD THIS" not in p for p in received_prompts)


class TestAgenticMap:
    async def test_creates_sub_sessions(self, tmp_path, op_config):
        """AgenticMap creates a separate session for each item."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        inputs = ["task_A", "task_B"]
        results = []
        async for result in agentic_map.run(
            inputs=inputs,
            agent_prompt_template="Complete this task: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "agentic_test.db"),
            max_turns=2,
        ):
            results.append(result)

        assert len(results) == 2
        session_ids = [r.session_id for r in results]
        # Each item should get a different session
        assert len(set(session_ids)) == 2

    async def test_invalid_template_raises(self, op_config):
        """Missing item variable raises ValueError."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        with pytest.raises(ValueError, match="item"):
            async for _ in agentic_map.run(
                inputs=["test"],
                agent_prompt_template="No item placeholder",
                model="test-model",
                read_only=False,
            ):
                pass

    async def test_template_with_filter_accepted(self, tmp_path, op_config):
        """Jinja2 templates with filters ({{ item | upper }}) are accepted."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["hello"],
            agent_prompt_template="Greet: {{ item | upper }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "filter_test.db"),
            max_turns=1,
        ):
            results.append(result)
        assert len(results) == 1

    async def test_results_have_input_reference(self, tmp_path, op_config):
        """AgentMapResult.input echoes back the original item."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        inputs = ["alpha", "beta"]
        result_inputs = []
        async for result in agentic_map.run(
            inputs=inputs,
            agent_prompt_template="Process: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "agentic_inputs.db"),
            max_turns=1,
        ):
            result_inputs.append(result.input)

        assert set(result_inputs) == set(inputs)

    async def test_sub_sessions_have_output_text(self, tmp_path, op_config):
        """Completed sub-sessions return non-empty output_text."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["hello world"],
            agent_prompt_template="Greet: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "agentic_output.db"),
            max_turns=1,
        ):
            results.append(result)

        assert len(results) == 1
        assert results[0].success is True
        assert len(results[0].output_text) > 0

    async def test_intermediate_outputs_captured(self, tmp_path, op_config):
        """intermediate_outputs contains one entry per completed turn."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["item1"],
            agent_prompt_template="Do: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "intermediate.db"),
            max_turns=1,
        ):
            results.append(result)

        assert len(results) == 1
        # With mock LLM and max_turns=1, exactly one turn ran.
        assert isinstance(results[0].intermediate_outputs, list)
        assert len(results[0].intermediate_outputs) >= 1

    async def test_continuation_message_empty_stops_after_turn0(self, tmp_path, op_config):
        """continuation_message='' (default) causes only one turn to run."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["x"],
            agent_prompt_template="Do: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            continuation_message="",
            db_path=str(tmp_path / "cont_empty.db"),
            max_turns=5,
        ):
            results.append(result)

        assert len(results) == 1
        # Only turn 0 ran — intermediate_outputs has exactly one entry.
        assert len(results[0].intermediate_outputs) == 1

    async def test_read_only_true_passes_tools_none(self, tmp_path, op_config, monkeypatch):
        """read_only=True passes tools=None to every sub-session send() call."""
        from mnesis.models.message import TokenUsage, TurnResult
        from mnesis.operators.agentic_map import AgenticMap

        tools_seen: list[object] = []

        async def fake_send(user_message, *, tools=None):
            tools_seen.append(tools)
            return TurnResult(
                message_id="msg_1",
                text="read-only response",
                finish_reason="stop",
                tokens=TokenUsage(),
                cost=0.0,
            )

        import mnesis.session as session_module

        original_create = session_module.MnesisSession.create

        async def patched_create(*args, **kwargs):
            sess = await original_create(*args, **kwargs)
            sess.send = fake_send  # type: ignore
            return sess

        monkeypatch.setattr(session_module.MnesisSession, "create", patched_create)

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["x", "y"],
            agent_prompt_template="Analyze: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=True,  # default
            db_path=str(tmp_path / "read_only_true.db"),
            max_turns=1,
        ):
            results.append(result)

        assert len(results) == 2
        # Every send() call must have received tools=None
        assert all(t is None for t in tools_seen), f"Expected all None, got {tools_seen}"

    async def test_read_only_true_strips_explicit_tools_with_warning(
        self, tmp_path, op_config, monkeypatch
    ):
        """read_only=True strips tools=[...] and logs a warning."""
        from mnesis.models.message import TokenUsage, TurnResult
        from mnesis.operators.agentic_map import AgenticMap

        tools_seen: list[object] = []

        async def fake_send(user_message, *, tools=None):
            tools_seen.append(tools)
            return TurnResult(
                message_id="msg_1",
                text="response",
                finish_reason="stop",
                tokens=TokenUsage(),
                cost=0.0,
            )

        import mnesis.session as session_module

        original_create = session_module.MnesisSession.create

        async def patched_create(*args, **kwargs):
            sess = await original_create(*args, **kwargs)
            sess.send = fake_send  # type: ignore
            return sess

        monkeypatch.setattr(session_module.MnesisSession, "create", patched_create)

        agentic_map = AgenticMap(op_config)

        import structlog.testing

        with structlog.testing.capture_logs() as cap_logs:
            results = []
            async for result in agentic_map.run(
                inputs=["x"],
                agent_prompt_template="Analyze: {{ item }}",
                model="anthropic/claude-opus-4-6",
                read_only=True,
                tools=[{"type": "function", "function": {"name": "search"}}],  # should be stripped
                db_path=str(tmp_path / "read_only_strip.db"),
                max_turns=1,
            ):
                results.append(result)

        # tools must have been stripped — send() sees None
        assert all(t is None for t in tools_seen)
        # Warning event must have been logged
        assert any(
            log.get("log_level") == "warning" and "read_only_tools_stripped" in log.get("event", "")
            for log in cap_logs
        ), f"Expected warning not found in: {cap_logs}"

    async def test_read_only_false_does_not_raise_not_implemented(self, tmp_path, op_config):
        """read_only=False is accepted — no NotImplementedError is raised."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["item"],
            agent_prompt_template="Process: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "read_only_false.db"),
            max_turns=1,
        ):
            results.append(result)

        assert len(results) == 1

    async def test_model_default_on_init(self, tmp_path, op_config):
        """Model set on __init__ is used when run() does not specify one."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config, model="anthropic/claude-opus-4-6")
        results = []
        async for result in agentic_map.run(
            inputs=["x"],
            agent_prompt_template="Process: {{ item }}",
            # model not passed here
            read_only=False,
            db_path=str(tmp_path / "model_default.db"),
            max_turns=1,
        ):
            results.append(result)
        assert len(results) == 1

    async def test_model_required_if_not_on_init(self, op_config):
        """ValueError raised if model is absent from both __init__ and run()."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        with pytest.raises(ValueError, match="model"):
            async for _ in agentic_map.run(
                inputs=["x"],
                agent_prompt_template="Process: {{ item }}",
                read_only=False,
            ):
                pass

    async def test_agent_map_result_has_error_kind(self, tmp_path, op_config):
        """Successful AgentMapResult has error_kind=None."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["x"],
            agent_prompt_template="Process: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "error_kind.db"),
            max_turns=1,
        ):
            results.append(result)

        assert results[0].success is True
        assert results[0].error_kind is None

    async def test_run_all_returns_agent_map_batch(self, tmp_path, op_config):
        """run_all() returns an AgentMapBatch with successes/failures."""
        from mnesis.operators.agentic_map import AgenticMap, AgentMapBatch

        agentic_map = AgenticMap(op_config)
        batch = await agentic_map.run_all(
            inputs=["a", "b"],
            agent_prompt_template="Process: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "run_all.db"),
            max_turns=1,
        )
        assert isinstance(batch, AgentMapBatch)
        assert batch.total == 2
        assert batch.total_attempts == 2

    async def test_sub_agent_exception_yields_failure_result(
        self, tmp_path, op_config, monkeypatch
    ):
        """When MnesisSession.create raises, _run_sub_agent returns a failure AgentMapResult."""
        import mnesis.session as session_module
        from mnesis.operators.agentic_map import AgenticMap

        async def failing_create(*args, **kwargs):
            raise RuntimeError("DB unavailable")

        monkeypatch.setattr(session_module.MnesisSession, "create", failing_create)

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["x"],
            agent_prompt_template="Do: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "exc_test.db"),
            max_turns=1,
        ):
            results.append(result)

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error_kind == "llm_error"
        assert "DB unavailable" in (results[0].error or "")

    async def test_run_all_failure_counts(self, tmp_path, op_config, monkeypatch):
        """run_all() populates failures when sub-agents fail."""
        import mnesis.session as session_module
        from mnesis.operators.agentic_map import AgenticMap, AgentMapBatch

        async def failing_create(*args, **kwargs):
            raise RuntimeError("fail")

        monkeypatch.setattr(session_module.MnesisSession, "create", failing_create)

        agentic_map = AgenticMap(op_config)
        batch = await agentic_map.run_all(
            inputs=["a", "b"],
            agent_prompt_template="Do: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "fail_all.db"),
            max_turns=1,
        )
        assert isinstance(batch, AgentMapBatch)
        assert len(batch.failures) == 2
        assert len(batch.successes) == 0

    async def test_continuation_message_empty_breaks_after_non_stop_turn0(
        self, tmp_path, op_config, monkeypatch
    ):
        """With continuation_message='' and non-stop finish_reason, loop breaks at turn 1."""
        from mnesis.models.message import TokenUsage, TurnResult
        from mnesis.operators.agentic_map import AgenticMap

        async def fake_send(user_message, *, tools=None):
            # Return tool_calls so the stop condition on line 320 is NOT hit,
            # then continuation_message="" triggers the else: break on line 309.
            return TurnResult(
                message_id="msg_1",
                text="used a tool",
                finish_reason="tool_calls",
                tokens=TokenUsage(),
                cost=0.0,
            )

        import mnesis.session as session_module

        original_create = session_module.MnesisSession.create

        async def patched_create(*args, **kwargs):
            sess = await original_create(*args, **kwargs)
            sess.send = fake_send  # type: ignore
            return sess

        monkeypatch.setattr(session_module.MnesisSession, "create", patched_create)

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["x"],
            agent_prompt_template="Do: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            continuation_message="",  # empty — triggers else: break on turn 1
            db_path=str(tmp_path / "empty_cont_non_stop.db"),
            max_turns=5,
        ):
            results.append(result)

        assert len(results) == 1
        assert results[0].success is True
        assert len(results[0].intermediate_outputs) == 1  # only turn 0 ran

    async def test_continuation_message_non_empty_runs_multiple_turns(
        self, tmp_path, op_config, monkeypatch
    ):
        """Non-empty continuation_message causes additional turns (beyond turn 0)."""
        from mnesis.models.message import TokenUsage, TurnResult
        from mnesis.operators.agentic_map import AgenticMap

        call_count = 0

        async def fake_send(user_message, *, tools=None):
            nonlocal call_count
            call_count += 1
            # Return non-stop finish_reason on first call so loop continues
            fr = "tool_calls" if call_count == 1 else "stop"
            return TurnResult(
                message_id=f"msg_{call_count}",
                text=f"turn {call_count}",
                finish_reason=fr,
                tokens=TokenUsage(),
                cost=0.0,
            )

        import mnesis.session as session_module

        original_create = session_module.MnesisSession.create

        async def patched_create(*args, **kwargs):
            sess = await original_create(*args, **kwargs)
            sess.send = fake_send  # type: ignore
            return sess

        monkeypatch.setattr(session_module.MnesisSession, "create", patched_create)

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["x"],
            agent_prompt_template="Do: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            continuation_message="Continue please.",
            db_path=str(tmp_path / "multi_turn.db"),
            max_turns=3,
        ):
            results.append(result)

        assert len(results) == 1
        assert results[0].success is True
        assert call_count == 2  # turn 0 + turn 1 (with continuation)

    async def test_max_tokens_finish_reason_stops_agent(self, tmp_path, op_config, monkeypatch):
        """finish_reason='max_tokens' causes the sub-agent to stop the loop."""
        from mnesis.models.message import TokenUsage, TurnResult
        from mnesis.operators.agentic_map import AgenticMap

        async def fake_send(user_message, *, tools=None):
            return TurnResult(
                message_id="msg_1",
                text="truncated response",
                finish_reason="max_tokens",
                tokens=TokenUsage(),
                cost=0.0,
            )

        import mnesis.session as session_module

        original_create = session_module.MnesisSession.create

        async def patched_create(*args, **kwargs):
            sess = await original_create(*args, **kwargs)
            sess.send = fake_send  # type: ignore
            return sess

        monkeypatch.setattr(session_module.MnesisSession, "create", patched_create)

        agentic_map = AgenticMap(op_config)
        results = []
        async for result in agentic_map.run(
            inputs=["x"],
            agent_prompt_template="Do: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            continuation_message="Continue.",
            db_path=str(tmp_path / "max_tokens.db"),
            max_turns=5,
        ):
            results.append(result)

        assert len(results) == 1
        assert results[0].success is True
        assert len(results[0].intermediate_outputs) == 1

    async def test_event_bus_published_on_agentic_run(self, tmp_path, op_config):
        """AgenticMap publishes MAP_STARTED, MAP_ITEM_COMPLETED, MAP_COMPLETED events."""
        from mnesis.events.bus import MnesisEvent
        from mnesis.operators.agentic_map import AgenticMap

        bus = EventBus()
        events: list[MnesisEvent] = []

        # Use a synchronous handler — async handlers are fire-and-forget background tasks.
        def collector(event: MnesisEvent, data: object) -> None:
            events.append(event)

        bus.subscribe(MnesisEvent.MAP_STARTED, collector)
        bus.subscribe(MnesisEvent.MAP_ITEM_COMPLETED, collector)
        bus.subscribe(MnesisEvent.MAP_COMPLETED, collector)

        agentic_map = AgenticMap(op_config, event_bus=bus)
        async for _ in agentic_map.run(
            inputs=["x"],
            agent_prompt_template="Do: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "bus_test.db"),
            max_turns=1,
        ):
            pass

        assert MnesisEvent.MAP_STARTED in events
        assert MnesisEvent.MAP_ITEM_COMPLETED in events
        assert MnesisEvent.MAP_COMPLETED in events


def _other_tasks() -> set[asyncio.Task]:
    return {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}


class TestOperatorLifecycle:
    """Early close cancels remaining work; backoff does not hold a concurrency slot."""

    async def test_llm_map_early_close_cancels_remaining(self, op_config, monkeypatch):
        import gc

        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)
        started: list[str] = []
        cancelled: list[str] = []
        never = asyncio.Event()
        errors: list[dict] = []
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx))

        async def call(**kwargs):
            item = kwargs["prompt"].removeprefix("Process ")
            started.append(item)
            if item == "item_0":
                return json.dumps({"ok": True})
            try:
                _ = await never.wait()
            except asyncio.CancelledError:
                cancelled.append(item)
                raise
            return "{}"

        try:
            llm_map = LLMMap(op_config)
            llm_map._call_llm = call  # type: ignore
            gen = llm_map.run(
                inputs=[f"item_{i}" for i in range(5)],
                prompt_template="Process {{ item }}",
                output_schema={"type": "object"},
                model="test-model",
                concurrency=5,
            )
            first = await anext(gen)
            assert first.input == "item_0"
            item_tasks = _other_tasks()
            assert len(item_tasks) == 4  # the 4 unfinished items
            await gen.aclose()

            # Every unfinished item task was cancelled and awaited.
            assert all(t.done() for t in item_tasks)
            assert sum(t.cancelled() for t in item_tasks) == 4
            assert sorted(cancelled) == [f"item_{i}" for i in range(1, 5)]
            assert not _other_tasks()
            calls_before = len(started)
            await asyncio.sleep(0.05)
            assert len(started) == calls_before  # no further LLM calls
            # "Task exception was never retrieved" only fires on GC: force it.
            del item_tasks
            gc.collect()
            await asyncio.sleep(0)
            assert errors == []
        finally:
            loop.set_exception_handler(None)

    async def test_llm_map_backoff_does_not_hold_semaphore(self, op_config, monkeypatch):
        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)
        sleeping = asyncio.Event()
        release = asyncio.Event()
        real_sleep = asyncio.sleep

        async def gated_sleep(delay: float) -> None:
            if delay >= 0.5:  # LLMMap backoff (min 0.5s); leave other sleeps alone
                sleeping.set()
                _ = await release.wait()
            else:
                await real_sleep(delay)

        monkeypatch.setattr(asyncio, "sleep", gated_sleep)
        attempts: dict[str, int] = {}

        async def call(**kwargs):
            item = kwargs["prompt"].removeprefix("Process ")
            attempts[item] = attempts.get(item, 0) + 1
            if item == "bad" and attempts[item] == 1:
                raise RuntimeError("transient")
            return json.dumps({"ok": True})

        llm_map = LLMMap(op_config)
        llm_map._call_llm = call  # type: ignore
        gen = llm_map.run(
            inputs=["bad", "good"],
            prompt_template="Process {{ item }}",
            output_schema={"type": "object"},
            model="test-model",
            concurrency=1,
        )
        # "bad" fails and enters backoff; with concurrency=1, "good" must still
        # complete while "bad" sleeps because the slot was released.
        first = await asyncio.wait_for(anext(gen), timeout=5)
        assert sleeping.is_set()
        assert first.input == "good"
        assert first.success is True
        release.set()
        second = await asyncio.wait_for(anext(gen), timeout=5)
        assert second.input == "bad"
        assert second.success is True
        assert second.attempts == 2
        await gen.aclose()

    @staticmethod
    def _patch_slow_send(monkeypatch, *, pending_compaction: bool = False):
        """Make non-``fast`` sub-agents hang in send(); returns observation state."""
        from mnesis.session import MnesisSession

        original_send = MnesisSession.send
        state = {
            "cancelled": [],
            "slow_started": 0,
            "both_started": asyncio.Event(),
            "compaction_aborted": [],
        }
        never = asyncio.Event()

        async def send(self, message, **kwargs):
            if message.endswith("fast"):
                return await original_send(self, message, **kwargs)
            if pending_compaction:
                # A background compaction in flight that only ends when aborted.
                async def fake_compaction():
                    _ = await self._compaction_abort.wait()
                    state["compaction_aborted"].append(message)

                self._compaction_engine._pending_task = asyncio.create_task(fake_compaction())
            state["slow_started"] += 1
            if state["slow_started"] == 2:
                state["both_started"].set()
            try:
                _ = await never.wait()
            except asyncio.CancelledError:
                state["cancelled"].append(message)
                raise
            raise AssertionError("slow sub-agent should only exit by cancellation")

        monkeypatch.setattr(MnesisSession, "send", send)
        return state

    @staticmethod
    def _agentic_gen(tmp_path, op_config, name):
        from mnesis.operators.agentic_map import AgenticMap

        return AgenticMap(op_config).run(
            inputs=["fast", "slow_a", "slow_b"],
            agent_prompt_template="Task {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / f"{name}.db"),
            concurrency=3,
            max_turns=1,
        )

    async def test_agentic_map_early_close_cancels_remaining(
        self, tmp_path, op_config, monkeypatch
    ):
        state = self._patch_slow_send(monkeypatch)
        gen = self._agentic_gen(tmp_path, op_config, "early_close")
        first = await anext(gen)
        assert first.input == "fast"
        # Both slow sub-agents must be inside send() before the close.
        await asyncio.wait_for(state["both_started"].wait(), timeout=5)
        await gen.aclose()

        assert sorted(state["cancelled"]) == ["Task slow_a", "Task slow_b"]
        assert not _other_tasks()

    async def test_agentic_map_early_close_aborts_pending_compaction(
        self, tmp_path, op_config, monkeypatch
    ):
        """Closing cancelled sub-sessions aborts, rather than awaits, their compaction."""
        state = self._patch_slow_send(monkeypatch, pending_compaction=True)
        gen = self._agentic_gen(tmp_path, op_config, "abort_compaction")
        _ = await anext(gen)
        await asyncio.wait_for(state["both_started"].wait(), timeout=5)
        # Without abort_compaction=True the fake compaction never ends: this hangs.
        await asyncio.wait_for(gen.aclose(), timeout=5)

        assert sorted(state["compaction_aborted"]) == ["Task slow_a", "Task slow_b"]
        assert not _other_tasks()

    async def test_agentic_map_closes_owned_pool_when_drain_interrupted(
        self, tmp_path, op_config, monkeypatch
    ):
        """A second cancel during the drain must not skip closing the owned pool."""
        import mnesis.operators.agentic_map as am_mod
        from mnesis.store.pool import StorePool

        pools: list[StorePool] = []

        class SpyPool(StorePool):
            def __init__(self) -> None:
                super().__init__()
                pools.append(self)

        async def stuck_drain(_tasks):
            await asyncio.sleep(30)

        monkeypatch.setattr(am_mod, "StorePool", SpyPool)
        state = self._patch_slow_send(monkeypatch)
        gen = self._agentic_gen(tmp_path, op_config, "double_cancel")

        async def consume() -> None:
            async for _ in gen:
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(state["both_started"].wait(), timeout=5)
        assert len(pools) == 1 and len(pools[0]._connections) == 1
        real_drain = am_mod.cancel_and_drain
        monkeypatch.setattr(am_mod, "cancel_and_drain", stuck_drain)
        _ = task.cancel()  # first cancel: enters the (stuck) drain in the finally
        await asyncio.sleep(0.05)
        _ = task.cancel()  # second cancel: interrupts the drain
        _ = await asyncio.wait({task}, timeout=5)
        assert task.cancelled()
        assert pools[0]._connections == {}
        # Clean up the still-running sub-agent tasks left by the stuck drain.
        await real_drain(_other_tasks())


class TestLiteLLMRetryInteraction:
    """C4: LiteLLM's own retries are off exactly when LLMMap retries the item itself."""

    @staticmethod
    def _patch_litellm(monkeypatch) -> list[dict]:
        import litellm

        monkeypatch.delenv("MNESIS_MOCK_LLM", raising=False)
        seen: list[dict] = []

        class _Msg:
            content = json.dumps({"summary": "ok", "keywords": ["a"]})

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = (_Choice(),)

        async def fake_acompletion(**kwargs):
            seen.append(kwargs)
            return _Resp()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        return seen

    @staticmethod
    async def _run_one(max_retries: int) -> None:
        llm_map = LLMMap(OperatorConfig(max_retries=max_retries))
        _tmpl = _JinjaEnv().from_string("Process: {{ item }}")
        result = await llm_map._process_item(
            item="x",
            compiled_template=_tmpl,
            schema=OutputSchema.model_json_schema(),
            pydantic_model=OutputSchema,
            model="test-model",
            semaphore=asyncio.Semaphore(1),
            max_retries=max_retries,
            system_prompt=None,
            temperature=0.0,
            timeout=30.0,
            retry_guidance="retry",
        )
        assert result.success is True

    async def test_litellm_retries_disabled_when_mnesis_retries(self, monkeypatch):
        seen = self._patch_litellm(monkeypatch)
        await self._run_one(max_retries=3)
        assert seen[0]["num_retries"] == 0

    async def test_litellm_defaults_kept_when_mnesis_retries_off(self, monkeypatch):
        seen = self._patch_litellm(monkeypatch)
        await self._run_one(max_retries=0)
        assert "num_retries" not in seen[0]
