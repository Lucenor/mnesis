"""
Example 03: Tool Use
====================

Demonstrates tool integration with Mnesis:
- Defining tools in litellm format
- How ToolPart appears in message history
- How pruned tool outputs appear as tombstones
- Custom on_part callback for tool visualization

Run without an API key:
    MNESIS_MOCK_LLM=1 uv run python examples/03_tool_use.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Define example tools in litellm/OpenAI format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


async def main() -> None:
    from mnesis import MnesisSession, MnesisConfig, CompactionConfig
    from mnesis.models.message import TextPart, ToolPart

    print("=== Mnesis Tool Use Example ===\n")

    config = MnesisConfig(
        compaction=CompactionConfig(
            prune=True,
            prune_protect_tokens=500,  # Very small for demo
            prune_minimum_tokens=100,
        )
    )

    received_parts: list = []

    def on_part(part):
        received_parts.append(part)
        if isinstance(part, TextPart):
            print(part.text, end="", flush=True)
        elif isinstance(part, ToolPart):
            state = part.status.state
            if state == "pending":
                print(f"\n  [TOOL] Calling {part.tool_name}({json.dumps(part.input)})")
            elif state == "completed":
                output_preview = (part.output or "")[:100]
                print(f"  [TOOL] {part.tool_name} completed: {output_preview}")
            elif state == "error":
                print(f"  [TOOL] {part.tool_name} error: {part.error_message}")

    async with await MnesisSession.create(
        model="anthropic/claude-opus-4-6",
        system_prompt="You are a file system explorer. Use tools to explore the codebase.",
        config=config,
        db_path="/tmp/mnesis_example_03.db",
    ) as session:
        print(f"Session: {session.id}\n")

        # Send a message that would trigger tool use
        print("User: List the files in /tmp and read one of them.")
        print("Assistant: ", end="")
        result = await session.send(
            "List the files in /tmp and read one of them.",
            tools=TOOLS,
            on_part=on_part,
        )
        print()

        print(f"\nFinish reason: {result.finish_reason}")
        print(f"Tokens used: {result.tokens.effective_total():,}")

        # Inspect message history for tool parts
        print("\n--- Message History Analysis ---")
        messages = await session.messages()
        for mwp in messages:
            role = mwp.role.upper()
            tool_count = sum(1 for p in mwp.parts if hasattr(p, "tool_name"))
            pruned = sum(1 for p in mwp.parts if hasattr(p, "compacted_at") and p.compacted_at)
            if tool_count > 0:
                print(f"  [{role}]: {tool_count} tool call(s), {pruned} pruned")
            else:
                print(f"  [{role}]: text message")

        # Show total parts received during streaming
        tool_parts = [p for p in received_parts if isinstance(p, ToolPart)]
        text_parts = [p for p in received_parts if isinstance(p, TextPart)]
        print(f"\nStreaming summary:")
        print(f"  TextPart callbacks: {len(text_parts)}")
        print(f"  ToolPart callbacks: {len(tool_parts)}")


if __name__ == "__main__":
    asyncio.run(main())
