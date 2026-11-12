"""Record OpenAI Agents SDK tool calls in Sovara traces.

Call ``instrument()`` once at startup, after importing ``agents``. Function tools are recorded as
tool steps; agent tools open a child trace. Model calls are recorded by ``sovara.auto_instrument()``
through the HTTP client.
"""

from __future__ import annotations

__all__ = ["instrument"]


def instrument() -> None:
    """Record OpenAI Agents function and agent tools inside Sovara traces; idempotent."""
    from sovara_openai_agents._patch import openai_agents_patch

    openai_agents_patch()
