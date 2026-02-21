"""
Example 01: Basic Session
=========================

Demonstrates the simplest end-to-end usage of MnesisSession:
- Creating a session with create()
- Sending messages in a loop
- Monitoring compaction via TurnResult.compaction_triggered
- Using session as an async context manager
- Inspecting token usage

Run without an API key:
    MNESIS_MOCK_LLM=1 uv run python examples/01_basic_session.py

Run with a real LLM (set your API key first):
    ANTHROPIC_API_KEY=sk-... uv run python examples/01_basic_session.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main() -> None:
    from mnesis import CompactionConfig, MnesisConfig, MnesisSession

    print("=== Mnesis Basic Session Example ===\n")

    # Configure with a smaller compaction buffer for demo purposes
    config = MnesisConfig(
        compaction=CompactionConfig(
            buffer=10_000,  # Trigger compaction sooner
            auto=True,
        )
    )

    # Create session — using async context manager for clean cleanup
    async with await MnesisSession.create(
        model="anthropic/claude-opus-4-6",
        system_prompt="You are a helpful coding assistant. Be concise.",
        config=config,
        db_path="/tmp/mnesis_example_01.db",
    ) as session:
        print(f"Session created: {session.id}")
        print(f"Model: {session.model}\n")

        # Simulate a multi-turn conversation
        questions = [
            "What is Python's GIL?",
            "How does asyncio work at a high level?",
            "What's the difference between async/await and threading?",
            "When should I use asyncio vs multiprocessing?",
            "Can you show a simple asyncio example?",
        ]

        for i, question in enumerate(questions, 1):
            print(f"Turn {i}: {question}")
            result = await session.send(question)
            print(f"  Response ({result.tokens.effective_total()} tokens): {result.text[:120]}...")
            print(f"  Finish reason: {result.finish_reason}")

            if result.compaction_triggered:
                print("  *** Compaction triggered in background! ***")

            print()

        # Show cumulative stats
        print(f"Total session tokens: {session.token_usage.effective_total():,}")

        # Show message history
        messages = await session.messages()
        print(f"Messages in history: {len(messages)}")
        summary_count = sum(1 for m in messages if m.message.is_summary)
        if summary_count:
            print(f"  (including {summary_count} compaction summary)")

    print("\nSession closed cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
