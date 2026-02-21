"""Shared Jinja2 template utilities for operator modules."""

from __future__ import annotations


def require_item_variable(template_str: str) -> None:
    """
    Raise ValueError if the Jinja2 template does not reference the ``item`` variable.

    Uses Jinja2 AST parsing instead of regex for correctness with complex expressions
    like ``{{ item['key'] }}`` and ``{{ item | upper }}``. Invalid template syntax
    is also reported as ``ValueError``.
    """
    from jinja2 import Environment, TemplateSyntaxError, meta

    env = Environment()
    try:
        ast = env.parse(template_str)
    except TemplateSyntaxError as exc:
        raise ValueError(f"Invalid Jinja2 template syntax: {exc}") from exc
    variables = meta.find_undeclared_variables(ast)
    if "item" not in variables:
        raise ValueError(
            "prompt_template must reference {{ item }} — "
            "the template does not use the 'item' variable"
        )
