# BYO-LLM — Use Your Own SDK

If you already use the Anthropic, OpenAI, Gemini, or any other SDK for your LLM calls, you can use mnesis purely as a memory and compaction layer — without routing any calls through litellm.

Use `session.record()` instead of `session.send()`. It persists a completed user/assistant turn to the store, accumulates token usage, and triggers compaction automatically — but makes no LLM call itself.

## How it works

```
Your code                           mnesis
──────────────────────────────      ──────────────────────────────
1. Build messages from context  ←── session.messages()
2. Call your LLM SDK
3. Record the result            ──► session.record(user, assistant)
                                     └─ persists both messages
                                     └─ updates token counters
                                     └─ triggers compaction if needed
```

## Examples

=== "Anthropic SDK"

    ```python
    import anthropic
    from mnesis import MnesisSession, TokenUsage

    client = anthropic.Anthropic()
    session = await MnesisSession.create(model="anthropic/claude-opus-4-6")

    user_text = "Explain quantum entanglement."

    # Build context from previous turns
    history = await session.messages()
    messages = [{"role": m.role, "content": m.text_content()} for m in history]
    messages.append({"role": "user", "content": user_text})

    # Call your SDK
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=messages,
    )

    # Record the turn — mnesis handles the rest
    await session.record(
        user_message=user_text,
        assistant_response=response.content[0].text,
        tokens=TokenUsage(
            input=response.usage.input_tokens,
            output=response.usage.output_tokens,
        ),
    )
    ```

=== "OpenAI SDK"

    ```python
    from openai import AsyncOpenAI
    from mnesis import MnesisSession, TokenUsage

    client = AsyncOpenAI()
    session = await MnesisSession.create(model="openai/gpt-4o")

    user_text = "What is the capital of France?"

    history = await session.messages()
    messages = [{"role": m.role, "content": m.text_content()} for m in history]
    messages.append({"role": "user", "content": user_text})

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    await session.record(
        user_message=user_text,
        assistant_response=response.choices[0].message.content,
        tokens=TokenUsage(
            input=response.usage.prompt_tokens,
            output=response.usage.completion_tokens,
        ),
    )
    ```

## Token usage is optional

If you omit `tokens`, mnesis estimates them from the text length using its built-in `TokenEstimator`. The estimate is a heuristic — pass real usage from your SDK response for accurate compaction budgeting:

```python
# Without token usage — mnesis estimates
await session.record(
    user_message="Hello",
    assistant_response="Hi there!",
)

# With token usage — accurate compaction budgeting
await session.record(
    user_message="Hello",
    assistant_response="Hi there!",
    tokens=TokenUsage(input=5, output=3),
)
```

## Compaction still works automatically

`record()` checks for overflow after every turn, exactly as `send()` does. If cumulative tokens exceed the model's context budget minus the compaction buffer, compaction is triggered in the background — no extra configuration required.

## RecordResult

`record()` returns a `RecordResult`:

```python
result = await session.record(...)

result.user_message_id      # ID of the persisted user message
result.assistant_message_id # ID of the persisted assistant message
result.tokens               # TokenUsage (provided or estimated)
result.compaction_triggered # True if compaction was scheduled
```

## Runnable example

See [`examples/06_byo_llm.py`](https://github.com/Lucenor/mnesis/blob/main/examples/06_byo_llm.py) for a complete, self-contained example that runs without any API key.
