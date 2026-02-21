"""
Example 06: BYO-LLM with session.record()
==========================================

Demonstrates using mnesis as a pure memory/compaction layer while you
manage LLM calls yourself with any SDK.

Instead of session.send() — which routes calls through litellm — use
session.record() to persist a completed turn after you've made the LLM
call yourself. mnesis handles storage, context assembly, and compaction
exactly as it would with send().

This example simulates an external LLM by returning canned responses so
it runs without any API key. Swap the `call_my_llm()` function with your
real SDK call (Anthropic, OpenAI, Gemini, etc.) to use it for real.

Run:
    uv run python examples/06_byo_llm.py

To use with the real Anthropic SDK, replace call_my_llm() with:
    import anthropic
    client = anthropic.Anthropic()

    def call_my_llm(messages, system):
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return (
            response.content[0].text,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Stub: replace this with your real SDK call
# ---------------------------------------------------------------------------

_CANNED_RESPONSES = [
    "Photosynthesis is the process by which plants convert sunlight, water, "
    + "and CO2 into glucose and oxygen.",
    "The light-dependent reactions occur in the thylakoid membranes and produce ATP and NADPH.",
    "The Calvin cycle (light-independent reactions) takes place in the stroma "
    + "and fixes CO2 into sugar.",
    "Chlorophyll a and b are the primary pigments; "
    + "they absorb red and blue light most efficiently.",
    "Without photosynthesis, virtually all life on Earth would cease - "
    + "it is the base of most food chains.",
]


def call_my_llm(
    messages: list[dict],
    system: str,
    turn_index: int,
) -> tuple[str, int, int]:
    """
    Stub that returns a canned response simulating an external LLM call.

    Returns:
        (response_text, input_tokens, output_tokens)
    """
    response = _CANNED_RESPONSES[turn_index % len(_CANNED_RESPONSES)]
    # Rough token estimates for the stub
    input_tokens = sum(len(m["content"].split()) * 4 // 3 for m in messages)
    output_tokens = len(response.split()) * 4 // 3
    return response, input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    from mnesis import MnesisSession, TokenUsage

    print("=== Mnesis BYO-LLM Example (session.record()) ===\n")
    print("Using stub LLM — swap call_my_llm() with your real SDK.\n")

    system_prompt = "You are a helpful science tutor. Be concise and accurate."

    async with await MnesisSession.create(
        model="anthropic/claude-opus-4-6",
        system_prompt=system_prompt,
        db_path="/tmp/mnesis_example_06.db",
    ) as session:
        print(f"Session: {session.id}\n")

        questions = [
            "What is photosynthesis?",
            "Where do the light-dependent reactions occur?",
            "What is the Calvin cycle?",
            "Which pigments are involved?",
            "Why is photosynthesis important for life on Earth?",
        ]

        for i, question in enumerate(questions):
            print(f"Turn {i + 1}: {question}")

            # 1. Build the message list from mnesis context
            #    (normally you'd pass this to your SDK so the model sees history)
            context = await session.messages()
            llm_messages = [{"role": m.role, "content": m.text_content()} for m in context]
            llm_messages.append({"role": "user", "content": question})

            # 2. Call your own LLM
            response_text, input_tok, output_tok = call_my_llm(llm_messages, system_prompt, i)

            # 3. Record the turn — mnesis persists both sides and handles compaction
            result = await session.record(
                user_message=question,
                assistant_response=response_text,
                tokens=TokenUsage(input=input_tok, output=output_tok),
            )

            print(f"  Response: {response_text[:100]}")
            print(f"  Tokens  : {result.tokens.input} in / {result.tokens.output} out")
            if result.compaction_triggered:
                print("  *** Compaction triggered! ***")
            print()

        # Inspect persisted history
        messages = await session.messages()
        print(f"Total turns recorded : {len(messages) // 2}")
        print(f"Cumulative tokens    : {session.token_usage.effective_total():,}")
        print(f"Messages in store    : {len(messages)}")

    print("\nSession closed. History is persisted in /tmp/mnesis_example_06.db")


if __name__ == "__main__":
    asyncio.run(main())
