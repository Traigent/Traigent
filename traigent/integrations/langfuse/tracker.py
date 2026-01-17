"""High-level Langfuse tracker for Traigent integration.

Provides a unified interface matching MLflow/W&B tracker patterns.
The tracker wraps the LangfuseClient and provides a callback for
orchestrator integration.

Usage:
    from traigent.integrations.langfuse import create_langfuse_tracker

    # Create tracker with trace_id resolver
    tracker = create_langfuse_tracker(
        trace_id_resolver=lambda trial: trial.metadata.get("langfuse_trace_id")
    )

    # Use callback in optimization
    @optimize(..., callbacks=[tracker.get_callback()])
    def my_workflow(...):
        ...

    # Or access client directly
    metrics = tracker.client.get_trace_metrics("trace_123")
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from traigent.integrations.langfuse.callback import (
    LangfuseOptimizationCallback,
    TraceIdResolver,
)
from traigent.integrations.langfuse.client import LangfuseClient
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.core.types import TrialResult
    from traigent.utils.callbacks import OptimizationCallback

logger = get_logger(__name__)


class LangfuseTracker:
    """High-level tracker for Langfuse integration with Traigent.

    Provides a unified interface matching MLflow/W&B tracker patterns.
    Wraps the low-level LangfuseClient and provides a callback for
    automatic metric enrichment during optimization.

    Args:
        client: LangfuseClient instance (dependency injected)
        trace_id_resolver: Function to map trial → trace_id
        metric_prefix: Prefix for Langfuse metrics (default: "langfuse_")
        include_per_agent: Include per-agent cost breakdown (default: True)

    Example:
        >>> tracker = create_langfuse_tracker(
        ...     trace_id_resolver=lambda t: t.metadata.get("trace_id")
        ... )
        >>> callback = tracker.get_callback()
    """

    def __init__(
        self,
        client: LangfuseClient,
        trace_id_resolver: TraceIdResolver | Callable[[TrialResult], str | None],
        *,
        metric_prefix: str = "langfuse_",
        include_per_agent: bool = True,
    ) -> None:
        self._client = client
        self._trace_id_resolver = trace_id_resolver
        self._metric_prefix = metric_prefix
        self._include_per_agent = include_per_agent
        self._callback: LangfuseOptimizationCallback | None = None

    @property
    def client(self) -> LangfuseClient:
        """Access underlying Langfuse client for direct API calls."""
        return self._client

    def get_callback(self) -> OptimizationCallback:
        """Return callback for orchestrator registration.

        Creates callback lazily on first call. Subsequent calls return
        the same callback instance.

        Returns:
            LangfuseOptimizationCallback configured with tracker settings
        """
        if self._callback is None:
            self._callback = LangfuseOptimizationCallback(
                client=self._client,
                trace_id_resolver=self._trace_id_resolver,
                metric_prefix=self._metric_prefix,
                include_per_agent=self._include_per_agent,
            )
        return self._callback


def create_langfuse_tracker(
    trace_id_resolver: TraceIdResolver | Callable[[TrialResult], str | None],
    *,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    metric_prefix: str = "langfuse_",
    include_per_agent: bool = True,
) -> LangfuseTracker:
    """Factory function to create Langfuse tracker.

    This is the recommended way to create a Langfuse tracker for Traigent.
    Credentials can be provided directly or via environment variables:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST (optional, defaults to cloud.langfuse.com)

    Args:
        trace_id_resolver: Function to map trial → Langfuse trace_id.
            Called for each trial to get the trace_id for metric lookup.
            Should return None if no trace_id is available.
        public_key: Langfuse public key (or LANGFUSE_PUBLIC_KEY env var)
        secret_key: Langfuse secret key (or LANGFUSE_SECRET_KEY env var)
        host: Langfuse host URL (or LANGFUSE_HOST env var)
        metric_prefix: Prefix for metrics in MeasuresDict (default: "langfuse_")
        include_per_agent: Include per-agent breakdown (default: True)

    Returns:
        Configured LangfuseTracker instance

    Example:
        >>> # Using environment variables for credentials
        >>> tracker = create_langfuse_tracker(
        ...     trace_id_resolver=lambda t: t.metadata.get("langfuse_trace_id")
        ... )

        >>> # Using explicit credentials
        >>> tracker = create_langfuse_tracker(
        ...     trace_id_resolver=lambda t: t.metadata.get("trace_id"),
        ...     public_key="pk-lf-xxx",
        ...     secret_key="sk-lf-xxx",
        ...     host="https://us.cloud.langfuse.com",
        ... )

        >>> # Get callback for optimization
        >>> callback = tracker.get_callback()
    """
    client = LangfuseClient(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )
    logger.info(
        f"Created Langfuse tracker: prefix='{metric_prefix}', "
        f"per_agent={include_per_agent}, host={client.host}"
    )
    return LangfuseTracker(
        client=client,
        trace_id_resolver=trace_id_resolver,
        metric_prefix=metric_prefix,
        include_per_agent=include_per_agent,
    )
