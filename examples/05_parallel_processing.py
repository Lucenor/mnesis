"""
Example 05: Parallel Processing with LLMMap and AgenticMap
===========================================================

Demonstrates both parallel operator primitives:
- LLMMap: stateless parallel extraction over many items
- AgenticMap: multi-turn sub-agents for deep per-item analysis
- Streaming results via async for as they complete
- Handling partial failures gracefully
- Token efficiency: O(1) parent context cost

Run without an API key:
    MNESIS_MOCK_LLM=1 uv run python examples/05_parallel_processing.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Sample documents to process
DOCUMENTS = [
    {
        "id": i,
        "title": f"Research Paper {i}",
        "abstract": f"This paper investigates topic {i}. " * 10 + f"Conclusion {i}.",
    }
    for i in range(10)
]

# Simulated repository descriptions for AgenticMap
REPOSITORIES = [
    {"name": "data-pipeline", "description": "ETL pipeline for processing CSV files"},
    {"name": "api-gateway", "description": "REST API gateway with rate limiting"},
    {"name": "auth-service", "description": "JWT-based authentication microservice"},
]


async def demo_llm_map() -> None:
    """Demonstrate LLMMap for parallel stateless extraction."""
    from mnesis import LLMMap
    from mnesis.models.config import OperatorConfig
    from pydantic import BaseModel

    print("=" * 50)
    print("LLMMap: Parallel Document Metadata Extraction")
    print("=" * 50)

    class DocumentMetadata(BaseModel):
        topic: str
        key_findings: list[str]
        relevance_score: int

    op_config = OperatorConfig(llm_map_concurrency=5, max_retries=2)
    llm_map = LLMMap(op_config)

    start = time.time()
    successful = 0
    failed = 0

    print(f"\nProcessing {len(DOCUMENTS)} documents in parallel (concurrency=5)...\n")

    async for result in llm_map.run(
        inputs=DOCUMENTS,
        prompt_template=(
            "Extract metadata from this document:\n\n"
            "Title: {{ item['title'] }}\n"
            "Abstract: {{ item['abstract'] }}\n\n"
            "Return JSON with: topic (string), key_findings (list), relevance_score (0-10)"
        ),
        output_schema=DocumentMetadata,
        model="anthropic/claude-haiku-3-5",
        temperature=0.0,
    ):
        elapsed = time.time() - start
        doc_id = result.input["id"] if isinstance(result.input, dict) else result.input
        if result.success:
            successful += 1
            print(f"  [+] Doc {doc_id} ({elapsed:.1f}s, attempt {result.attempts})")
        else:
            failed += 1
            print(f"  [!] Doc {doc_id} failed: {result.error}")

    total_time = time.time() - start
    print(f"\nCompleted in {total_time:.2f}s")
    print(f"Results: {successful} succeeded, {failed} failed")
    print(f"Throughput: {len(DOCUMENTS) / total_time:.1f} docs/sec")


async def demo_agentic_map(tmp_path: str) -> None:
    """Demonstrate AgenticMap for parallel multi-turn analysis."""
    from mnesis import AgenticMap
    from mnesis.models.config import OperatorConfig

    print("\n" + "=" * 50)
    print("AgenticMap: Parallel Repository Analysis")
    print("=" * 50)

    op_config = OperatorConfig(agentic_map_concurrency=2, max_retries=1)
    agentic_map = AgenticMap(op_config)

    start = time.time()
    results_collected = []

    print(f"\nAnalyzing {len(REPOSITORIES)} repositories with independent sub-agents...\n")
    print("(Each sub-agent runs its own multi-turn session — parent context is O(1))\n")

    async for result in agentic_map.run(
        inputs=REPOSITORIES,
        agent_prompt_template=(
            "You are a code quality analyst.\n"
            "Analyze this repository:\n\n"
            "Name: {{ item['name'] }}\n"
            "Description: {{ item['description'] }}\n\n"
            "Provide: 1) Risk assessment, 2) Key concerns, 3) Recommendations"
        ),
        model="anthropic/claude-opus-4-6",
        max_turns=2,
        db_path=os.path.join(tmp_path, "agentic_map.db"),
    ):
        results_collected.append(result)
        repo_name = result.input["name"] if isinstance(result.input, dict) else result.input
        elapsed = time.time() - start
        status = "SUCCESS" if result.success else "FAILED"
        print(f"  [{status}] {repo_name} (sub-session: {result.session_id[:20]}...)")
        print(f"           Output preview: {result.output_text[:120]}...")
        print()

    total_time = time.time() - start
    print(f"\nAll sub-agents completed in {total_time:.2f}s")
    print(f"Sub-sessions created: {len(results_collected)}")
    total_tokens = sum(r.token_usage.effective_total() for r in results_collected)
    print(f"Total sub-agent tokens: {total_tokens:,}")
    print(f"Parent context cost: O(1) — parent only sees final outputs")


async def main() -> None:
    import tempfile

    print("=== Mnesis Parallel Processing Example ===\n")

    # Run LLMMap demo
    await demo_llm_map()

    # Run AgenticMap demo
    with tempfile.TemporaryDirectory() as tmp:
        await demo_agentic_map(tmp)

    print("\n=== Done ===")


if __name__ == "__main__":
    asyncio.run(main())
