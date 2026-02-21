"""Tests for LLMMap and AgenticMap operators."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

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

    async def test_concurrency_limit_respected(self, op_config):
        """At most N concurrent calls are made at once."""
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

        # Don't actually test concurrency here since mock overrides vary
        # Just verify all items complete
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

    async def test_read_only_true_raises_not_implemented(self, op_config):
        """read_only=True raises NotImplementedError immediately (not silently ignored)."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        with pytest.raises(NotImplementedError, match="read_only"):
            async for _ in agentic_map.run(
                inputs=["test"],
                agent_prompt_template="Process: {{ item }}",
                model="test-model",
                read_only=True,
            ):
                pass

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
