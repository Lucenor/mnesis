"""
Example 07: Streaming Iterator API
===================================

Demonstrates ``MnesisSession.stream()`` — an async generator that yields
``TextDelta`` events carrying the response text and a final ``TurnComplete``
event with the full ``TurnResult``.

Note: ``stream()`` is a thin wrapper around ``send()``.  Because ``send()``
buffers ``on_part`` callbacks per attempt and forwards them only after the
attempt succeeds, ``TextDelta`` events are delivered as a batch after the
LLM call completes, not as live token-by-token output during generation.

Run without an API key:
    MNESIS_MOCK_LLM=1 uv run python examples/07_streaming.py

Run with a real LLM (set your API key first):
    ANTHROPIC_API_KEY=sk-... uv run python examples/07_streaming.py

**Abandonment safety**

If you ``break`` out of the ``async for`` loop, the underlying ``send()``
task still completes in the background.  The turn is fully persisted, token
counters are updated, and compaction is triggered if needed.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main() -> None:
    from mnesis import MnesisSession, TextDelta, TurnComplete

    print("=== Mnesis Streaming Iterator Example ===\n")

    async with MnesisSession.open(
        model="anthropic/claude-opus-4-6",
        system_prompt="You are a concise assistant.",
        db_path="/tmp/mnesis_example_07.db",
    ) as session:
        print(f"Session: {session.id}")
        print(f"Model:   {session.model}\n")

        questions = [
            "What is Python's asyncio event loop in one sentence?",
            "Name three benefits of async programming.",
        ]

        for i, question in enumerate(questions, 1):
            print(f"--- Turn {i} ---")
            print(f"User: {question}")
            print("Assistant: ", end="", flush=True)

            token_count = 0
            compaction_triggered = False

            async for event in session.stream(question):
                if isinstance(event, TextDelta):
                    print(event.text, end="", flush=True)
                elif isinstance(event, TurnComplete):
                    token_count = event.result.tokens.effective_total()
                    compaction_triggered = event.result.compaction_triggered

            print()  # newline after streamed output
            print(f"[{token_count} tokens | compaction={'yes' if compaction_triggered else 'no'}]")
            print()

        print(f"Total session tokens: {session.token_usage.effective_total():,}")

    print("Session closed cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
