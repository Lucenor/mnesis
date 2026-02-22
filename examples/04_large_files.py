"""
Example 04: Large File Handling
================================

Demonstrates content-addressed file references:
- Configuring FileConfig.inline_threshold
- Using LargeFileHandler directly
- How FileRefPart differs from inline content
- The exploration summary structure for Python and JSON files
- Cache hits when the same file is processed twice

Run without an API key:
    MNESIS_MOCK_LLM=1 uv run python examples/04_large_files.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Sample large Python file content
LARGE_PYTHON_FILE = (
    '''"""
A sample large Python module for demonstration.
"""
import asyncio
import json
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class DataProcessor:
    """Processes raw data with configurable pipelines."""
    batch_size: int = 100
    max_retries: int = 3

    async def process(self, data: list[Any]) -> list[Any]:
        """Process a batch of data items."""
        results = []
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            processed = await self._process_batch(batch)
            results.extend(processed)
        return results

    async def _process_batch(self, batch: list[Any]) -> list[Any]:
        """Process a single batch."""
        await asyncio.sleep(0)  # Yield to event loop
        return [self._transform(item) for item in batch]

    def _transform(self, item: Any) -> Any:
        """Apply transformation to a single item."""
        return item


class ValidationPipeline:
    """Validates data against schemas."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema

    def validate(self, data: Any) -> bool:
        """Validate data against the schema."""
        return isinstance(data, dict)


class DatabaseAdapter:
    """Manages database connections and queries."""

    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string
        self._pool: Optional[Any] = None

    async def connect(self) -> None:
        """Establish database connection pool."""
        pass

    async def disconnect(self) -> None:
        """Close all connections in the pool."""
        pass

    async def execute(self, query: str, params: list[Any] | None = None) -> list[dict]:
        """Execute a query and return results."""
        return []


async def run_pipeline(data: list[Any], config: dict[str, Any]) -> dict[str, Any]:
    """Run the full data processing pipeline."""
    processor = DataProcessor(
        batch_size=config.get("batch_size", 100),
        max_retries=config.get("max_retries", 3),
    )
    results = await processor.process(data)
    return {
        "processed": len(results),
        "results": results[:10],
    }
'''
    * 5
)  # Repeat to make it large


LARGE_JSON_FILE = json.dumps(
    {
        "configuration": {
            "database": {"host": "localhost", "port": 5432, "name": "app_db"},
            "cache": {"backend": "redis", "ttl": 3600},
            "auth": {"provider": "jwt", "secret_key": "...", "expiry": 86400},
            "features": {f"feature_{i}": {"enabled": True, "config": {}} for i in range(50)},
            "api": {
                "rate_limits": {"default": 100, "premium": 1000},
                "endpoints": {f"/api/v{i}": {"methods": ["GET", "POST"]} for i in range(20)},
            },
        },
        "data": [{"id": i, "name": f"item_{i}", "value": i * 1.5} for i in range(200)],
    },
    indent=2,
)


async def main() -> None:
    from mnesis.files import LargeFileHandler
    from mnesis.models.config import FileConfig, StoreConfig
    from mnesis.store.immutable import ImmutableStore
    from mnesis.tokens.estimator import TokenEstimator

    print("=== Mnesis Large File Handling Example ===\n")

    # Setup store and handler
    db_path = "/tmp/mnesis_example_04.db"
    store_config = StoreConfig(db_path=db_path)
    store = ImmutableStore(store_config)
    await store.initialize()

    estimator = TokenEstimator(heuristic_only=True)  # Use heuristic for demo

    file_config = FileConfig(inline_threshold=500)  # Very low for demo
    handler = LargeFileHandler(store, estimator, file_config)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write test files
        py_path = Path(tmpdir) / "data_pipeline.py"
        json_path = Path(tmpdir) / "config.json"

        py_path.write_text(LARGE_PYTHON_FILE)
        json_path.write_text(LARGE_JSON_FILE)

        print(f"Python file: {len(LARGE_PYTHON_FILE):,} chars")
        print(f"JSON file: {len(LARGE_JSON_FILE):,} chars\n")

        # Process the Python file
        print("--- Processing Python file ---")
        py_result = await handler.handle_file(py_path)
        if py_result.is_inline:
            print(f"Inline content ({len(py_result.inline_content or ''):,} chars)")
        else:
            ref = py_result.file_ref
            print("File reference created!")
            print(f"  Content-ID: {ref.content_id[:20]}...")
            print(f"  File type: {ref.file_type}")
            print(f"  Estimated tokens: {ref.token_count:,}")
            indented = ref.exploration_summary.replace("\n", "\n    ")
            print(f"  Exploration summary:\n    {indented}")

        # Process the JSON file
        print("\n--- Processing JSON file ---")
        json_result = await handler.handle_file(json_path)
        if not json_result.is_inline:
            ref = json_result.file_ref
            print("File reference created!")
            print(f"  Content-ID: {ref.content_id[:20]}...")
            indented = ref.exploration_summary.replace("\n", "\n    ")
            print(f"  Exploration summary:\n    {indented}")

        # Demonstrate cache hit
        print("\n--- Cache Hit Demonstration ---")
        py_result2 = await handler.handle_file(py_path)
        assert not py_result2.is_inline
        assert py_result2.file_ref.content_id == py_result.file_ref.content_id
        print(f"Same content_id returned (cache hit): {py_result2.file_ref.content_id[:20]}...")
        print("No re-summarization occurred!")

        # Show how it renders in context
        if not py_result.is_inline:
            from mnesis.context.builder import ContextBuilder

            rendered = ContextBuilder._render_file_ref(py_result.file_ref)
            print("\n--- Context Window Rendering ---")
            print(rendered[:400])
            print("...")

    await store.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
