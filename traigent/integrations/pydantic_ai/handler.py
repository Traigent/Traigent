"""PydanticAI handler for Traigent instrumentation.

This module provides a handler that wraps PydanticAI agent calls to
capture metrics (tokens, cost, latency) for Traigent optimization.

Usage::

    from traigent.integrations.pydantic_ai import PydanticAIHandler

    handler = PydanticAIHandler(agent, metric_prefix="pydantic_ai_")

    # Instead of: result = await agent.run("prompt")
    result = await handler.run("prompt")

    # Get captured metrics
    metrics = handler.get_measures_dict()
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

from traigent.integrations.pydantic_ai._types import (
    AgentRunMetrics,
    PydanticAIHandlerMetrics,
)

logger = logging.getLogger(__name__)

# Check PydanticAI availability
try:
    import pydantic_ai  # noqa: F401

    PYDANTICAI_AVAILABLE = True
except ImportError:
    PYDANTICAI_AVAILABLE = False


class PydanticAIHandler:
    """Wraps PydanticAI agent calls to capture metrics for Traigent.

    Thread-safe: each handler uses a ``threading.Lock`` for metric
    aggregation, so multiple threads can safely call ``run_sync()``
    concurrently.

    Args:
        agent: A ``pydantic_ai.Agent`` instance.
        metric_prefix: Prefix for metric keys in ``get_measures_dict()``.
        traigent_config: Optional dict of Traigent config params to inject
            into ``model_settings`` on every call.
    """

    def __init__(
        self,
        agent: Any,
        *,
        metric_prefix: str = "pydantic_ai_",
        traigent_config: dict[str, Any] | None = None,
    ) -> None:
        if not PYDANTICAI_AVAILABLE:
            raise ImportError(
                "PydanticAI is required for PydanticAIHandler. "
                "Install with: pip install 'pydantic-ai>=1,<2'"
            )

        self._agent = agent
        self._metric_prefix = metric_prefix
        self._traigent_config = traigent_config or {}
        self._lock = threading.Lock()
        self._runs: list[AgentRunMetrics] = []
        self._model_name = self._extract_model_name(agent)
        from traigent.utils.env_config import is_strict_cost_accounting

        # Latch strict mode at handler creation for consistent per-run behavior.
        self._strict_cost_accounting = is_strict_cost_accounting()

    # -----------------------------------------------------------------
    # Public run wrappers
    # -----------------------------------------------------------------

    async def run(
        self,
        prompt: Any,
        *,
        model_settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Wrap ``agent.run()`` — inject params into model_settings, capture metrics."""
        merged = self._merge_model_settings(model_settings)
        start = time.time()
        try:
            result = await self._agent.run(prompt, model_settings=merged, **kwargs)
        except Exception:
            self._record_error(start)
            raise
        self._record_usage(result, start)
        return result

    def run_sync(
        self,
        prompt: Any,
        *,
        model_settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Wrap ``agent.run_sync()`` — sync variant."""
        merged = self._merge_model_settings(model_settings)
        start = time.time()
        try:
            result = self._agent.run_sync(prompt, model_settings=merged, **kwargs)
        except Exception:
            self._record_error(start)
            raise
        self._record_usage(result, start)
        return result

    @asynccontextmanager
    async def run_stream(
        self,
        prompt: Any,
        *,
        model_settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Wrap ``agent.run_stream()`` — usage extracted after stream consumed."""
        merged = self._merge_model_settings(model_settings)
        start = time.time()
        error_occurred = False
        async with self._agent.run_stream(
            prompt, model_settings=merged, **kwargs
        ) as stream_result:
            try:
                yield stream_result
            except BaseException:
                error_occurred = True
                self._record_error(start)
                raise
        if not error_occurred:
            self._record_usage(stream_result, start)

    @contextmanager
    def run_stream_sync(
        self,
        prompt: Any,
        *,
        model_settings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Wrap ``agent.run_stream_sync()`` — sync streaming variant."""
        merged = self._merge_model_settings(model_settings)
        start = time.time()
        error_occurred = False
        with self._agent.run_stream_sync(
            prompt, model_settings=merged, **kwargs
        ) as stream_result:
            try:
                yield stream_result
            except BaseException:
                error_occurred = True
                self._record_error(start)
                raise
        if not error_occurred:
            self._record_usage(stream_result, start)

    # -----------------------------------------------------------------
    # Metrics access
    # -----------------------------------------------------------------

    def get_metrics(self) -> PydanticAIHandlerMetrics:
        """Return aggregated metrics for all runs."""
        with self._lock:
            return PydanticAIHandlerMetrics(runs=list(self._runs))

    def get_measures_dict(self) -> dict[str, float | int]:
        """Get metrics in MeasuresDict-compatible format."""
        return self.get_metrics().to_measures_dict(prefix=self._metric_prefix)

    def reset(self) -> None:
        """Reset handler state for reuse."""
        with self._lock:
            self._runs.clear()

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _merge_model_settings(
        self, user_settings: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Merge Traigent config into model_settings.

        User-provided values take precedence over Traigent config values.
        Params are injected INTO model_settings, not as top-level kwargs.
        """
        base: dict[str, Any] = {}

        # Add Traigent config params
        param_keys = {"temperature", "max_tokens", "top_p"}
        for key in param_keys:
            if key in self._traigent_config:
                base[key] = self._traigent_config[key]

        # User values override
        if user_settings:
            base.update(user_settings)

        return base if base else {}

    def _record_usage(self, result: Any, start_time: float) -> None:
        """Extract usage from result and record metrics."""
        end_time = time.time()
        try:
            usage = result.usage()
            metrics = AgentRunMetrics(
                model=self._model_name,
                start_time=start_time,
                end_time=end_time,
                request_count=getattr(usage, "requests", 0) or 0,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                total_tokens=getattr(usage, "total_tokens", 0) or 0,
                cost=self._estimate_cost(
                    getattr(usage, "input_tokens", 0) or 0,
                    getattr(usage, "output_tokens", 0) or 0,
                ),
            )
        except Exception:
            logger.warning(
                "Failed to extract usage from PydanticAI result", exc_info=True
            )
            metrics = AgentRunMetrics(
                model=self._model_name,
                start_time=start_time,
                end_time=end_time,
            )

        with self._lock:
            self._runs.append(metrics)

    def _record_error(self, start_time: float) -> None:
        """Record a failed run with zero usage."""
        metrics = AgentRunMetrics(
            model=self._model_name,
            start_time=start_time,
            end_time=time.time(),
        )
        with self._lock:
            self._runs.append(metrics)

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost using the canonical cost_from_tokens path."""
        if input_tokens == 0 and output_tokens == 0:
            return 0.0
        strict_cost_accounting = self._strict_cost_accounting
        try:
            from traigent.utils.cost_calculator import cost_from_tokens

            input_cost, output_cost = cost_from_tokens(
                input_tokens,
                output_tokens,
                self._model_name or "unknown",
                strict=strict_cost_accounting,
            )
            return float(input_cost + output_cost)
        except Exception:
            if strict_cost_accounting:
                raise
            logger.debug(
                "Cost estimation failed for model %s", self._model_name, exc_info=True
            )
            return 0.0

    @staticmethod
    def _extract_model_name(agent: Any) -> str:
        """Extract model name from a PydanticAI Agent.

        Handles both string format (``"openai:gpt-4o"``) and Model objects.
        """
        model = getattr(agent, "model", None)
        if model is None:
            return "unknown"

        # String format: "openai:gpt-4o" → "gpt-4o"
        if isinstance(model, str):
            if ":" in model:
                return model.split(":", 1)[1]
            return model

        # Model object: try .model_name or .model_id
        for attr in ("model_name", "model_id", "name"):
            name = getattr(model, attr, None)
            if isinstance(name, str) and name:
                return name

        # Last resort: str(model)
        model_str = str(model)
        if ":" in model_str:
            return model_str.split(":", 1)[1]
        return model_str or "unknown"


def create_pydantic_ai_handler(
    agent: Any,
    *,
    metric_prefix: str = "pydantic_ai_",
    traigent_config: dict[str, Any] | None = None,
) -> PydanticAIHandler:
    """Factory function to create a PydanticAIHandler.

    Args:
        agent: A ``pydantic_ai.Agent`` instance.
        metric_prefix: Prefix for metric keys (default: ``"pydantic_ai_"``).
        traigent_config: Optional dict of Traigent config params to inject.

    Returns:
        Configured PydanticAIHandler instance.
    """
    return PydanticAIHandler(
        agent,
        metric_prefix=metric_prefix,
        traigent_config=traigent_config,
    )
