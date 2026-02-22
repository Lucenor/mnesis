"""MkDocs hook: pre-render Mermaid diagrams to inline SVG at build time using mmdc."""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("mkdocs.hooks.mermaid_render")

_MERMAID_BLOCK = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)


def on_page_markdown(
    markdown: str,
    *,
    page: Any,
    config: Any,
    files: Any,
    **kwargs: object,
) -> str:
    """Replace fenced mermaid blocks with inline SVG rendered by mmdc."""

    def render(m: re.Match[str]) -> str:
        source = m.group(1).strip()
        svg = _render_svg(source, config)
        if svg is None:
            log.warning(
                "mmdc failed for a diagram on %s; leaving block unchanged",
                page.file.src_path,
            )
            return m.group(0)
        return f'\n<div class="mermaid-diagram">\n{svg}\n</div>\n'

    return _MERMAID_BLOCK.sub(render, markdown)


def _render_svg(source: str, config: Any) -> str | None:
    """Render mermaid source to inline SVG string, with caching."""
    cache_key = hashlib.sha256(source.encode()).hexdigest()[:16]
    cache_dir = Path(config["docs_dir"]).parent / ".mermaid-cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.svg"

    if cache_file.exists():
        return _clean_svg(cache_file.read_text(encoding="utf-8"))

    with tempfile.NamedTemporaryFile(suffix=".mmd", mode="w", delete=False, encoding="utf-8") as fh:
        fh.write(source)
        input_path = Path(fh.name)

    output_path = input_path.with_suffix(".svg")
    try:
        result = subprocess.run(
            [
                "mmdc",
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "-b",
                "transparent",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        log.error(
            "mmdc not found — install @mermaid-js/mermaid-cli"
            " (npm install -g @mermaid-js/mermaid-cli)"
        )
        input_path.unlink(missing_ok=True)
        return None
    finally:
        input_path.unlink(missing_ok=True)

    if result.returncode != 0 or not output_path.exists():
        log.error("mmdc error: %s", result.stderr.strip())
        output_path.unlink(missing_ok=True)
        return None

    raw = output_path.read_text(encoding="utf-8")
    output_path.unlink(missing_ok=True)

    cleaned = _clean_svg(raw)
    cache_file.write_text(cleaned, encoding="utf-8")
    return cleaned


def _clean_svg(svg: str) -> str:
    """Strip XML declaration and DOCTYPE from SVG string."""
    svg = re.sub(r"<\?xml[^?]*\?>", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]*>", "", svg)
    return svg.strip()
