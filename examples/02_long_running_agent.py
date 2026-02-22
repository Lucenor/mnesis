"""
Example 02: Long-Running Agent
==============================

Demonstrates advanced session management for long tasks:
- Custom MnesisConfig with compaction model override
- Real-time streaming via the on_part callback
- Subscribing to EventBus for compaction notifications
- Inspecting session.messages() including summary messages
- Manual session.compact() as a checkpoint

Run without an API key:
    MNESIS_MOCK_LLM=1 uv run python examples/02_long_running_agent.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main() -> None:
    from mnesis import CompactionConfig, MnesisConfig, MnesisEvent, MnesisSession
    from mnesis.models.message import TextPart

    print("=== Mnesis Long-Running Agent Example ===\n")

    # Subscribe to compaction events
    compaction_log: list[dict[str, Any]] = []

    def on_compaction_completed(event: MnesisEvent, payload: dict[str, Any]) -> None:
        compaction_log.append(payload)
        print(
            f"\n  [EVENT] Compaction completed! Level {payload.get('level_used', '?')}, "
            f"compacted {payload.get('compacted_message_count', '?')} messages."
        )

    # Configure compaction to trigger early (for demo)
    config = MnesisConfig(
        compaction=CompactionConfig(
            compaction_output_budget=5_000,
            auto=True,
        )
    )

    # Streaming callback — print text as it arrives
    def on_part(part: Any) -> None:
        if isinstance(part, TextPart):
            print(part.text, end="", flush=True)

    async with MnesisSession.open(
        model="anthropic/claude-opus-4-6",
        system_prompt=(
            "You are a senior software engineer helping with a large refactoring task. "
            "Be detailed and methodical."
        ),
        config=config,
        db_path="/tmp/mnesis_example_02.db",
    ) as session:
        # Subscribe to compaction events
        session.subscribe(MnesisEvent.COMPACTION_COMPLETED, on_compaction_completed)
        session.subscribe(
            MnesisEvent.DOOM_LOOP_DETECTED, lambda e, p: print("\n  [WARNING] Doom loop detected!")
        )

        print(f"Session: {session.id}\n")

        # Simulate a long coding task
        tasks = [
            "We're refactoring a monolithic Python codebase. "
            + "Start by outlining the main challenges.",
            "What's the best strategy for breaking up a large models.py file?",
            "How should we handle circular imports during the refactor?",
            "Describe the testing strategy for the refactored modules.",
            "What tooling should we use to verify the refactor is complete?",
            "How do we handle database migrations alongside the refactor?",
            "Summarize the full refactoring plan we've discussed.",
        ]

        for i, task in enumerate(tasks, 1):
            print(f"\n--- Turn {i} ---")
            print(f"User: {task}")
            print("Assistant: ", end="")

            result = await session.send(task, on_part=on_part)
            print()  # Newline after streaming

            if result.doom_loop_detected:
                print("  [WARNING] Doom loop — stopping early.")
                break

        # Manual checkpoint compaction
        print("\n--- Manual Checkpoint Compaction ---")
        compact_result = await session.compact()
        print(
            f"Compacted {compact_result.compacted_message_count} messages "
            f"using Level {compact_result.level_used}"
        )
        print(
            f"Tokens before: {compact_result.tokens_before:,}, "
            f"after: {compact_result.tokens_after:,}"
        )

        # Inspect history
        messages = await session.messages()
        regular = [m for m in messages if not m.message.is_summary]
        summaries = [m for m in messages if m.message.is_summary]
        print(f"\nHistory: {len(regular)} regular messages, {len(summaries)} summaries")
        print(f"Total session tokens used: {session.token_usage.effective_total():,}")
        print(f"Compaction events received: {len(compaction_log)}")


if __name__ == "__main__":
    asyncio.run(main())
