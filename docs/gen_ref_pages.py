"""Generate API reference pages from source modules automatically.

This script is executed by mkdocs-gen-files at build time. It walks the
src/mnesis package tree and creates one Markdown page per module under
docs/api/, then writes a SUMMARY.md for mkdocs-literate-nav to consume.

Nothing in this file is committed to the docs/ directory — everything is
generated in memory during the mkdocs build.
"""

from pathlib import Path

import mkdocs_gen_files

src = Path(__file__).parent.parent / "src"
nav = mkdocs_gen_files.Nav()

for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src / "mnesis").with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    # Skip __main__ and private helpers
    if parts[-1] in ("__main__",) or any(p.startswith("_") and p != "__init__" for p in parts):
        continue

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    if not parts:
        continue

    # Strip the leading "mnesis" package segment so nav sections are clean
    nav_parts = parts[1:] if parts[0] == "mnesis" else parts
    if not nav_parts:
        continue

    # SUMMARY.md lives inside api/, so paths must be relative to api/
    nav[nav_parts] = str(doc_path)

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# `{ident}`\n\n")
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(src.parent))

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
