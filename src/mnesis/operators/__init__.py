"""Mnesis parallel operator primitives."""

from mnesis.operators.agentic_map import AgenticMap, AgentMapResult
from mnesis.operators.llm_map import LLMMap, MapResult

__all__ = ["AgentMapResult", "AgenticMap", "LLMMap", "MapResult"]
