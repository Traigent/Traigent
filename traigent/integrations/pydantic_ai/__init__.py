"""PydanticAI integration for Traigent.

Provides a handler that wraps PydanticAI agent calls to capture metrics
(tokens, cost, latency) and a plugin for parameter mapping and validation.

Quick Start::

    from traigent.integrations.pydantic_ai import PydanticAIHandler

    handler = PydanticAIHandler(agent, metric_prefix="pydantic_ai_")
    result = await handler.run("What is AI?")
    metrics = handler.get_measures_dict()
"""

from __future__ import annotations

from typing import Any

# Check PydanticAI availability
try:
    import pydantic_ai  # noqa: F401

    PYDANTICAI_AVAILABLE = True
except ImportError:
    PYDANTICAI_AVAILABLE = False

if PYDANTICAI_AVAILABLE:
    from traigent.integrations.pydantic_ai.handler import (
        PydanticAIHandler,
        create_pydantic_ai_handler,
    )
    from traigent.integrations.pydantic_ai.plugin import PydanticAIPlugin
else:

    def _pydantic_ai_unavailable(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            "PydanticAI integration is unavailable. "
            "Install with: pip install 'pydantic-ai>=1,<2'"
        )

    PydanticAIHandler = _pydantic_ai_unavailable  # type: ignore[assignment, misc]
    PydanticAIPlugin = _pydantic_ai_unavailable  # type: ignore[assignment, misc]
    create_pydantic_ai_handler = _pydantic_ai_unavailable

__all__ = [
    "PYDANTICAI_AVAILABLE",
    "PydanticAIHandler",
    "PydanticAIPlugin",
    "create_pydantic_ai_handler",
]
