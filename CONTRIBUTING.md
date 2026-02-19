# Contributing to mnesis

Thanks for your interest in contributing. This document covers how to set up a dev environment, the conventions used in the codebase, and the process for getting changes merged.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Project Layout](#project-layout)
- [Submitting Changes](#submitting-changes)
- [Reporting Bugs](#reporting-bugs)
- [License](#license)

---

## Getting Started

1. Fork the repo and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/mnesis.git
   cd mnesis
   ```

2. Install [uv](https://docs.astral.sh/uv/) if you don't have it:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Install the project with dev dependencies:
   ```bash
   uv sync --dev
   ```

That's it — no other system dependencies required for development.

---

## Development Setup

The project uses:

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **[ruff](https://docs.astral.sh/ruff/)** for linting and formatting
- **[mypy](https://mypy.readthedocs.io/)** (strict mode) for type checking
- **[pytest](https://docs.pytest.org/)** + **pytest-asyncio** for tests
- **SQLite** (via aiosqlite) — no external database needed

All tests run in mock LLM mode by default (`MNESIS_MOCK_LLM=1`), so no API keys are required for development.

---

## Running Tests

```bash
# Full test suite with coverage report
uv run pytest

# A specific test file
uv run pytest tests/test_store.py -v

# A specific test
uv run pytest tests/test_session.py::TestMnesisSession::test_load_restores_session -v

# Skip coverage (faster)
uv run pytest --no-cov
```

Coverage threshold is configured in `pyproject.toml`. New code should be covered by tests.

To run the examples without an API key:

```bash
MNESIS_MOCK_LLM=1 uv run python examples/01_basic_session.py
MNESIS_MOCK_LLM=1 uv run python examples/05_parallel_processing.py
```

---

## Code Style

Run these before every commit:

```bash
uv run ruff check src/ tests/          # lint
uv run ruff format src/ tests/         # auto-format
uv run mypy src/mnesis                 # type check
```

**Key conventions:**

- All public methods and classes require Google-style docstrings (`Args:`, `Returns:`, `Raises:`).
- Use `async`/`await` throughout — no blocking I/O on the main thread.
- Pydantic v2 for all data models. Use `model_validator` and `field_validator`, not `__init__` overrides.
- Prefer `pathlib.Path` over `os.path` string manipulation.
- New store operations must go through `ImmutableStore` — do not add direct SQL elsewhere.
- Events should be published through `EventBus`, not via direct callbacks between components.

---

## Project Layout

```
src/mnesis/
├── session.py              # MnesisSession — public entry point
├── models/                 # Pydantic data models (config, message, summary)
├── store/
│   ├── schema.sql          # SQLite schema (append-only, WAL mode)
│   ├── immutable.py        # ImmutableStore — all DB reads/writes
│   ├── summary_dag.py      # Logical DAG adapter over ImmutableStore
│   └── pool.py             # StorePool — shared connection registry
├── context/builder.py      # Context window assembly
├── compaction/
│   ├── engine.py           # Three-level escalation orchestrator
│   ├── pruner.py           # Tool output backward-scanner
│   └── levels.py           # Level 1 / 2 / 3 summarization
├── files/handler.py        # Content-addressed large file handler
├── tokens/estimator.py     # Token counting (tiktoken + heuristic fallback)
├── events/bus.py           # In-process pub/sub EventBus
└── operators/
    ├── llm_map.py          # Stateless parallel LLM calls
    └── agentic_map.py      # Parallel sub-agent sessions
```

Tests mirror the source structure under `tests/`.

---

## Submitting Changes

1. **Open an issue first** for non-trivial changes. This avoids duplicate work and lets us align on the approach before you invest time in implementation.

2. Create a branch:
   ```bash
   git checkout -b your-feature-or-fix
   ```

3. Make your changes. Keep commits focused — one logical change per commit.

4. Ensure lint, types, and tests all pass:
   ```bash
   uv run ruff check src/ tests/
   uv run ruff format src/ tests/
   uv run mypy src/mnesis
   uv run pytest
   ```

5. Push and open a PR against `main`. Describe what the change does and why, reference any related issues.

PRs that add features without tests, or that drop coverage below threshold, will be asked to add coverage before merging.

---

## Reporting Bugs

Open a [GitHub issue](https://github.com/Lucenor/mnesis/issues) with:

- Python version and OS
- mnesis version (`uv run python -c "import mnesis; print(mnesis.__version__)"`)
- Minimal reproduction script (anonymize any sensitive data)
- What you expected vs. what happened

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).

Copyright 2026 ByteWorks LLC-FZ. See [NOTICE](NOTICE) for details.
