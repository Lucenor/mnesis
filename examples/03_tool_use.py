"""
Example 03: Tool Use
====================

Demonstrates tool integration with Mnesis using the record() pattern:
- Constructing ToolPart objects to record tool calls and results
- How ToolPart (pending / completed / error states) appears in message history
- How pruned tool outputs appear as tombstones after compaction
- The on_part callback pattern for real-time tool visualization

In production, your LLM issues tool calls; mnesis handles memory and pruning.
Use session.record() to inject completed turns (user message + assistant parts
including ToolPart objects) into the session store.

Run without an API key:
    MNESIS_MOCK_LLM=1 uv run python examples/03_tool_use.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# Tool definitions (passed to LLM in production)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
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


def show_tool_call(tool_name: str, args: dict, output: str | None, error: str | None) -> None:
    """Visualize a tool call as it would appear via on_part streaming callback."""
    print(f"  [TOOL call  ] {tool_name}({json.dumps(args)})")
    if error:
        print(f"  [TOOL error ] {error}")
    else:
        print(f"  [TOOL result] {(output or '')[:100]}")


async def main() -> None:
    from mnesis import MnesisSession, MnesisConfig, CompactionConfig
    from mnesis.models.message import TextPart, ToolPart, ToolStatus

    print("=== Mnesis Tool Use Example ===\n")
    print("Tool calls are recorded via session.record() with ToolPart objects.")
    print("In production your LLM issues the calls; mnesis tracks + prunes them.\n")

    # Low prune thresholds so the demo shows tombstoning within a short session.
    # Production defaults are 40K / 20K — large enough to keep tool outputs
    # intact for the entire recent working set.
    config = MnesisConfig(
        compaction=CompactionConfig(
            prune=True,
            prune_protect_tokens=50,   # Demo: only protect the nearest ~50 output tokens
            prune_minimum_tokens=10,   # Demo: prune even small volumes
        )
    )

    async with await MnesisSession.create(
        model="anthropic/claude-opus-4-6",
        system_prompt="You are a file system explorer. Use tools to investigate the codebase.",
        config=config,
        db_path="/tmp/mnesis_example_03.db",
    ) as session:
        print(f"Session: {session.id}\n")

        # ── Turn 1: list /tmp ─────────────────────────────────────────────────
        print("Turn 1: List /tmp")
        t1 = ToolPart(
            tool_name="list_directory",
            tool_call_id="call_001",
            input={"path": "/tmp"},
            output=(
                "mnesis_example_01.db  mnesis_example_02.db  mnesis_example_03.db  "
                "mnesis_example_04.db  sample_data.csv  config.json  app.log"
            ),
            status=ToolStatus(state="completed"),
        )
        show_tool_call(t1.tool_name, t1.input, t1.output, None)
        await session.record(
            user_message="List all files in /tmp.",
            assistant_response=[
                t1,
                TextPart(text="Found 7 files in /tmp including database files, a CSV, JSON config, and a log."),
            ],
        )

        # ── Turn 2: read the log ──────────────────────────────────────────────
        print("\nTurn 2: Read app.log")
        t2 = ToolPart(
            tool_name="read_file",
            tool_call_id="call_002",
            input={"path": "/tmp/app.log"},
            output=(
                "2026-02-20 10:00:01 INFO  Service started on port 8080\n"
                "2026-02-20 10:01:15 WARN  High memory usage: 87%\n"
                "2026-02-20 10:02:30 ERROR Database connection timeout after 30s\n"
                "2026-02-20 10:03:00 INFO  Reconnected to database successfully\n"
                "2026-02-20 10:05:45 INFO  Processed 1,204 requests"
            ),
            status=ToolStatus(state="completed"),
        )
        show_tool_call(t2.tool_name, t2.input, t2.output, None)
        await session.record(
            user_message="Read /tmp/app.log — I want to see if there are any errors.",
            assistant_response=[
                t2,
                TextPart(text="Found one ERROR: database connection timeout at 10:02:30, recovered at 10:03:00."),
            ],
        )

        # ── Turn 3: read the config ───────────────────────────────────────────
        print("\nTurn 3: Read config.json")
        t3 = ToolPart(
            tool_name="read_file",
            tool_call_id="call_003",
            input={"path": "/tmp/config.json"},
            output=json.dumps({
                "service": "api-gateway",
                "version": "2.1.0",
                "database": {"host": "db.internal", "port": 5432, "pool_size": 10},
                "cache": {"backend": "redis", "ttl": 3600},
                "log_level": "INFO",
            }, indent=2),
            status=ToolStatus(state="completed"),
        )
        show_tool_call(t3.tool_name, t3.input, t3.output, None)
        await session.record(
            user_message="Read /tmp/config.json — check the database pool size.",
            assistant_response=[
                t3,
                TextPart(text="The database pool size is 10. Service is api-gateway v2.1.0."),
            ],
        )

        # ── Turn 4: list a sub-directory ──────────────────────────────────────
        print("\nTurn 4: List /tmp/archive/")
        t4_err = ToolPart(
            tool_name="list_directory",
            tool_call_id="call_004",
            input={"path": "/tmp/archive/"},
            error_message="PermissionError: /tmp/archive/ — access denied",
            status=ToolStatus(state="error"),
        )
        show_tool_call(t4_err.tool_name, t4_err.input, None, t4_err.error_message)
        await session.record(
            user_message="List the /tmp/archive/ directory.",
            assistant_response=[
                t4_err,
                TextPart(text="Cannot access /tmp/archive/ — permission denied."),
            ],
        )

        # ── Turn 5: write a summary report ────────────────────────────────────
        print("\nTurn 5: Write summary report")
        summary_content = (
            "# Investigation Report\n"
            "Date: 2026-02-20\n\n"
            "## Findings\n"
            "- Service api-gateway v2.1.0 running on port 8080\n"
            "- One DB timeout error recovered within 30s\n"
            "- DB pool_size=10 (may need tuning given 87% memory warning)\n"
            "- /tmp/archive/ inaccessible — check permissions\n\n"
            "## Recommendation\n"
            "Increase DB pool size to 20 and investigate memory usage."
        )
        t5 = ToolPart(
            tool_name="write_file",
            tool_call_id="call_005",
            input={"path": "/tmp/investigation_report.md", "content": summary_content},
            output="Written 312 bytes to /tmp/investigation_report.md",
            status=ToolStatus(state="completed"),
        )
        show_tool_call(t5.tool_name, t5.input, t5.output, None)
        await session.record(
            user_message="Write a summary report of everything you found.",
            assistant_response=[
                t5,
                TextPart(text="Report written to /tmp/investigation_report.md."),
            ],
        )

        # ── Inspect message history before compaction ─────────────────────────
        print("\n" + "=" * 55)
        print("Message History Before Compaction")
        print("=" * 55)
        messages = await session.messages()
        total_tool_parts = 0
        for mwp in messages:
            role = mwp.role.upper()
            tool_parts = [p for p in mwp.parts if isinstance(p, ToolPart)]
            text_parts = [p for p in mwp.parts if isinstance(p, TextPart)]
            if tool_parts:
                total_tool_parts += len(tool_parts)
                for tp in tool_parts:
                    status_str = tp.status.state
                    if tp.error_message:
                        detail = f"error: {tp.error_message[:50]}"
                    else:
                        detail = f"output: {(tp.output or '')[:50]}..."
                    print(f"  [{role}] ToolPart({tp.tool_name}, {status_str}) — {detail}")
            if text_parts:
                for txp in text_parts:
                    print(f"  [{role}] TextPart: {txp.text[:70]}...")

        print(f"\n{len(messages)} messages, {total_tool_parts} tool parts")
        print(f"Cumulative tokens: {session.token_usage.effective_total():,}")

        # ── Trigger compaction (runs pruner + LLM summarisation) ──────────────
        print("\n" + "=" * 55)
        print("Compaction + Tool Output Pruning")
        print("=" * 55)
        compact_result = await session.compact()
        print(f"Level used      : {compact_result.level_used}")
        print(f"Messages compacted: {compact_result.compacted_message_count}")
        print(f"Tokens before   : {compact_result.tokens_before:,}")
        print(f"Tokens after    : {compact_result.tokens_after:,}")

        # ── Inspect after compaction — show tombstones ────────────────────────
        print("\n" + "=" * 55)
        print("Message History After Compaction")
        print("=" * 55)
        messages_after = await session.messages()
        tombstoned = 0
        active_tools = 0
        for mwp in messages_after:
            role = mwp.role.upper()
            for part in mwp.parts:
                if isinstance(part, ToolPart):
                    if part.compacted_at is not None:
                        tombstoned += 1
                        print(
                            f"  [{role}] TOMBSTONE  {part.tool_name} — "
                            "output pruned, data preserved in store"
                        )
                    else:
                        active_tools += 1
                        print(f"  [{role}] ACTIVE     {part.tool_name} — output intact in context")

        summaries = [m for m in messages_after if m.message.is_summary]
        if summaries:
            summary_text = next(
                (p.text for p in summaries[-1].parts if isinstance(p, TextPart)), ""
            )
            print(f"\n  Compaction summary (level {compact_result.level_used}):")
            print(f"  {summary_text[:300].replace(chr(10), chr(10) + '  ')}")

        print(f"\n{tombstoned} tool output(s) tombstoned, {active_tools} still active in context")
        if tombstoned == 0:
            print(
                "  (session too short to cross prune_protect_tokens threshold — "
                "tombstoning is automatic in longer sessions)"
            )
        else:
            print(
                "  Tombstoned outputs are replaced by a compact marker in the active context.\n"
                "  The original data remains intact in the SQLite store."
            )


if __name__ == "__main__":
    asyncio.run(main())
