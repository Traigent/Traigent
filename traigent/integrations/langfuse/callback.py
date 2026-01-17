"""Langfuse optimization callback for automatic trace metric extraction.

This callback integrates with Traigent's optimization loop to automatically
fetch metrics from Langfuse traces and enrich trial results.

Usage:
    from traigent.integrations.langfuse import create_langfuse_tracker

    tracker = create_langfuse_tracker(
        trace_id_resolver=lambda trial: trial.metadata.get("langfuse_trace_id")
    )

    @optimize(..., callbacks=[tracker.get_callback()])
    def my_workflow(...):
        ...
"""

# Traceability: CONC-Layer-Integration FUNC-INTEGRATIONS REQ-INT-008

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from traigent.utils.callbacks import OptimizationCallback
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.core.types import OptimizationResult, TrialResult
    from traigent.integrations.langfuse.client import LangfuseClient
    from traigent.utils.callbacks import ProgressInfo


logger = get_logger(__name__)


@runtime_checkable
class TraceIdResolver(Protocol):
    """Protocol for resolving trial → trace_id mapping.

    Users provide a function that extracts the Langfuse trace_id from a trial.
    This allows flexibility in how trace IDs are stored/accessed.

    Example implementations:
        # From trial metadata
        lambda trial: trial.metadata.get("langfuse_trace_id")

        # From trial config
        lambda trial: trial.config.get("trace_id")

        # Custom logic
        def resolver(trial):
            return f"trace_{trial.trial_id}"
    """

    def __call__(self, trial: TrialResult) -> str | None:
        """Return Langfuse trace_id for a trial, or None if not available."""
        ...


class LangfuseOptimizationCallback(OptimizationCallback):
    """Callback that enriches trial metrics with Langfuse trace data.

    Follows dependency injection pattern per CLAUDE.md:
    - Client injected, not instantiated
    - Thread-safe for parallel trial execution (stateless per-trial ops)
    - Errors logged but don't block optimization

    Args:
        client: Injected LangfuseClient instance
        trace_id_resolver: Function to map trial → trace_id
        metric_prefix: Prefix for Langfuse metrics (default: "langfuse_")
        include_per_agent: Include per-agent cost breakdown (default: True)

    Example:
        >>> client = LangfuseClient(public_key="pk", secret_key="sk")
        >>> callback = LangfuseOptimizationCallback(
        ...     client=client,
        ...     trace_id_resolver=lambda t: t.metadata.get("trace_id"),
        ... )
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

    def on_optimization_start(
        self, config_space: dict[str, Any], objectives: list[str], algorithm: str
    ) -> None:
        """Called when optimization starts. Log configuration."""
        logger.info(
            f"Langfuse callback initialized: prefix='{self._metric_prefix}', "
            f"per_agent={self._include_per_agent}"
        )

    def on_trial_start(self, trial_number: int, config: dict[str, Any]) -> None:
        """Called when a trial starts. No-op for Langfuse."""
        pass

    def on_trial_complete(self, trial: TrialResult, progress: ProgressInfo) -> None:
        """Enrich trial metrics with Langfuse trace data.

        Fetches metrics from Langfuse using the trace_id from the resolver.
        Merges Langfuse metrics into trial.metrics using |= operator.
        """
        trace_id = self._trace_id_resolver(trial)
        if not trace_id:
            logger.debug(
                f"No trace_id for trial {trial.trial_id}, skipping Langfuse enrichment"
            )
            return

        try:
            # Use sync method (callback is called synchronously by orchestrator)
            metrics = self._client.get_trace_metrics(trace_id)
            if metrics:
                measures = metrics.to_measures_dict(
                    prefix=self._metric_prefix,
                    include_per_agent=self._include_per_agent,
                )
                # Merge into trial metrics (dict |= operation)
                if trial.metrics is None:
                    trial.metrics = {}
                trial.metrics |= measures
                logger.debug(
                    f"Enriched trial {trial.trial_id} with {len(measures)} Langfuse metrics"
                )
            else:
                logger.debug(f"No Langfuse metrics found for trace {trace_id}")
        except Exception as e:
            # Log warning but don't block optimization
            logger.warning(
                f"Failed to fetch Langfuse metrics for trace {trace_id}: {e}"
            )

    def on_optimization_complete(self, result: OptimizationResult) -> None:
        """Called when optimization completes. Log summary."""
        logger.info(
            f"Langfuse callback: optimization complete with {len(result.trials)} trials"
        )
