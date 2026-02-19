# LLM Providers

mnesis uses [litellm](https://docs.litellm.ai/) internally, which means it works with any provider litellm supports — Anthropic, OpenAI, Google Gemini, OpenRouter, Azure, and more. You never interact with litellm directly; just pass a model string and set the corresponding API key.

## Model string format

```
"<provider>/<model-name>"
```

| Provider | Example model string | API key env var |
|---|---|---|
| Anthropic | `"anthropic/claude-opus-4-6"` | `ANTHROPIC_API_KEY` |
| OpenAI | `"openai/gpt-4o"` | `OPENAI_API_KEY` |
| Google Gemini | `"gemini/gemini-1.5-pro"` | `GEMINI_API_KEY` |
| OpenRouter | `"openrouter/anthropic/claude-opus-4-6"` | `OPENROUTER_API_KEY` |
| Azure OpenAI | `"azure/<your-deployment>"` | `AZURE_API_KEY` + `AZURE_API_BASE` |

The provider prefix is optional for well-known model names (e.g. `"gpt-4o"` resolves to OpenAI automatically), but the full form is recommended for clarity.

## Examples

=== "Anthropic"

    ```python
    import os
    from mnesis import MnesisSession

    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
    session = await MnesisSession.create(model="anthropic/claude-opus-4-6")
    ```

=== "OpenAI"

    ```python
    import os
    from mnesis import MnesisSession

    os.environ["OPENAI_API_KEY"] = "sk-..."
    session = await MnesisSession.create(model="openai/gpt-4o")
    ```

=== "Google Gemini"

    ```python
    import os
    from mnesis import MnesisSession

    os.environ["GEMINI_API_KEY"] = "AIza..."
    session = await MnesisSession.create(model="gemini/gemini-1.5-pro")
    ```

=== "OpenRouter"

    ```python
    import os
    from mnesis import MnesisSession

    os.environ["OPENROUTER_API_KEY"] = "sk-or-..."
    session = await MnesisSession.create(
        model="openrouter/meta-llama/llama-3.1-70b-instruct"
    )
    ```

## Using a cheaper model for compaction

Compaction (context summarisation) defaults to the session model. You can point it at a cheaper, faster model to save cost:

```python
from mnesis import MnesisSession, MnesisConfig, CompactionConfig

config = MnesisConfig(
    compaction=CompactionConfig(
        compaction_model="openai/gpt-4o-mini",
    )
)
session = await MnesisSession.create(model="openai/gpt-4o", config=config)
```

## Passing extra litellm parameters

For provider-specific options (custom `api_base`, extra headers for OpenRouter, etc.), configure litellm globally before creating a session:

```python
import litellm

litellm.api_base = "https://my-proxy.example.com"
litellm.headers = {"X-Custom-Header": "value"}
```

See the [litellm provider docs](https://docs.litellm.ai/docs/providers) for the full list of supported providers and their options.

## Prefer your own SDK?

If you want to use the Anthropic, OpenAI, or another SDK directly for LLM calls, see [BYO-LLM](byo-llm.md) — mnesis can act as a pure memory/compaction layer without routing calls through litellm at all.
