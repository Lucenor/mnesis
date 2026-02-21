"""Generate API reference pages from source modules automatically.

This script is run standalone (e.g. ``python docs/gen_ref_pages.py``) from
the project root before running ``zensical build``.  It walks the
``src/mnesis`` package tree and writes one Markdown stub per module under
``docs/api/``.  The stubs contain only an ``:::`` mkdocstrings directive so
that Zensical + mkdocstrings can render the full API documentation.

Unlike the previous version, this script does **not** depend on
``mkdocs_gen_files`` or ``mkdocs_literate_nav``, which are MkDocs-only plugins
that Zensical does not execute.  The generated files are committed to the
repository so that ``zensical build`` can find them without any plugin hooks.

Usage::

    python docs/gen_ref_pages.py          # from project root
    uv run python docs/gen_ref_pages.py   # via uv

The script is idempotent: re-running it overwrites existing stubs with the
same content, leaving the working tree clean if nothing changed.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src"
DOCS_API = REPO_ROOT / "docs" / "api"


def main() -> None:
    """Walk src/mnesis and write one .md stub per public module."""

    # Collect pages: (full_doc_path, mkdocstrings_identifier, edit_path)
    pages: list[tuple[Path, str, Path]] = []

    for path in sorted(SRC.rglob("*.py")):
        module_path = path.relative_to(SRC).with_suffix("")
        doc_path = path.relative_to(SRC / "mnesis").with_suffix(".md")
        full_doc_path = DOCS_API / doc_path

        parts = tuple(module_path.parts)

        # Skip __main__ and private helpers (but not __init__)
        if parts[-1] in ("__main__",) or any(
            p.startswith("_") and p != "__init__" for p in parts
        ):
            continue

        if parts[-1] == "__init__":
            parts = parts[:-1]
            full_doc_path = full_doc_path.with_name("index.md")

        if not parts:
            continue

        ident = ".".join(parts)
        edit_path = path.relative_to(REPO_ROOT)
        pages.append((full_doc_path, ident, edit_path))

    # Write stubs
    for full_doc_path, ident, edit_path in pages:
        full_doc_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            f"# `{ident}`\n\n"
            f"::: {ident}\n"
        )
        full_doc_path.write_text(content, encoding="utf-8")
        print(f"  wrote {full_doc_path.relative_to(REPO_ROOT)}")

    print(f"\nGenerated {len(pages)} API reference pages under docs/api/")


if __name__ == "__main__":
    main()
