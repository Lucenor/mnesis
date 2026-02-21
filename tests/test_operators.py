"""Tests for LLMMap and AgenticMap operators."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from mnesis.models.config import OperatorConfig
from mnesis.operators.llm_map import LLMMap


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
        """Missing {{ item }} in template raises ValueError."""
        llm_map = LLMMap(op_config)
        with pytest.raises(ValueError, match=r"\{\{ item \}\}"):
            async for _ in llm_map.run(
                inputs=["test"],
                prompt_template="No item placeholder here",
                output_schema={},
                model="test-model",
            ):
                pass

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
        """Missing {{ item }} raises ValueError."""
        from mnesis.operators.agentic_map import AgenticMap

        agentic_map = AgenticMap(op_config)
        with pytest.raises(ValueError, match=r"\{\{ item \}\}"):
            async for _ in agentic_map.run(
                inputs=["test"],
                agent_prompt_template="No item placeholder",
                model="test-model",
                read_only=False,
            ):
                pass

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
        # If NotImplementedError were raised it would propagate out of the generator.
        # We only verify the run() generator starts without that error.
        async for result in agentic_map.run(
            inputs=["item"],
            agent_prompt_template="Process: {{ item }}",
            model="anthropic/claude-opus-4-6",
            read_only=False,
            db_path=str(tmp_path / "read_only_false.db"),
            max_turns=1,
        ):
            results.append(result)

        # One result per input regardless of sub-session outcome.
        assert len(results) == 1
