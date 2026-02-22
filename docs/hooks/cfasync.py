"""MkDocs hook: add data-cfasync='false' to Mermaid and critical script tags
so Cloudflare Rocket Loader skips them and executes them natively."""

from __future__ import annotations

import re


def on_post_page(output: str, **kwargs: object) -> str:
    """Add data-cfasync='false' to <script> tags that Rocket Loader must not defer.

    Rocket Loader intercepts ALL <script> tags and replaces their type attribute
    with a random token (e.g. type="48b6d53c566b4bbed776367b-module"), which
    causes any script relying on type="module" (such as the mermaid2 plugin's
    ESM import) to fail with a syntax error when the browser tries to execute
    an import statement in a non-module context.

    The only reliable opt-out without Cloudflare dashboard access is the
    data-cfasync="false" attribute: Rocket Loader unconditionally skips any
    script element that carries it.

    This hook runs after all plugins (hooks are appended to the plugin list)
    so it sees the fully-assembled HTML including the mermaid2 module script.
    """

    def add_cfasync(m: re.Match[str]) -> str:
        tag = m.group(0)
        tag_lower = tag.lower()
        if "data-cfasync" in tag_lower:
            return tag  # already patched
        if 'type="module"' in tag_lower or "mermaid" in tag_lower:
            # Preserve original case of the opening tag name while inserting the attribute.
            open_tag = m.group(1)
            return tag.replace(open_tag, open_tag + ' data-cfasync="false"', 1)
        return tag

    return re.sub(r"(<script)([^>]*>)", add_cfasync, output, flags=re.IGNORECASE)
