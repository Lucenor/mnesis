"""MkDocs hook: add data-cfasync='false' to all script tags so Cloudflare
Rocket Loader loads them natively without deferral or type mangling."""

from __future__ import annotations

import re


def on_post_page(output: str, **kwargs: object) -> str:
    """Add data-cfasync='false' to every <script> tag.

    Rocket Loader intercepts ALL <script> tags and replaces their type attribute
    with a random token (e.g. type="48b6d53c566b4bbed776367b-module"), breaking
    any script relying on type="module" or on native execution order.
    data-cfasync="false" is the canonical per-script opt-out.
    """

    def add_cfasync(m: re.Match[str]) -> str:
        tag = m.group(0)
        if "data-cfasync" in tag.lower():
            return tag  # already patched
        open_tag = m.group(1)
        return tag.replace(open_tag, open_tag + ' data-cfasync="false"', 1)

    return re.sub(r"(<script)([^>]*>)", add_cfasync, output, flags=re.IGNORECASE)
