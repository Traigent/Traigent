"""JavaScript runtime evaluator for Node.js trial execution.

This module provides JSEvaluator, which enables running optimization trials
in a Node.js subprocess via the JSBridge.

Usage:
    evaluator = JSEvaluator(
        js_module="./dist/my-trial.js",
        js_function="runTrial",
    )
    result = await evaluator.evaluate(func, config, dataset)
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from traigent.bridges.js_bridge import (
    JSBridge,
    JSBridgeConfig,
    JSBridgeError,
    JSTrialTimeoutError,
)
from traigent.bridges.process_pool import JSProcessPool
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.core.sample_budget import SampleBudgetLease

logger = get_logger(__name__)


@dataclass
class JSEvaluatorConfig:
    """Configuration for JSEvaluator.

    Attributes:
        js_module: Path to the JS module containing the trial function.
        js_function: Name of the exported function to call.
        js_timeout: Timeout for trial execution in seconds.
        experiment_run_id: ID of the experiment run for trial tracking.
    """

    js_module: str
    js_function: str = "runTrial"
    js_timeout: float = 300.0
    experiment_run_id: str | None = None


class JSEvaluator(BaseEvaluator):
    """Evaluator that executes trials in a Node.js subprocess.

    This evaluator delegates trial execution to a Node.js process via the
    JSBridge. It is used when execution.runtime="node" is specified in
    the @traigent.optimize decorator.

    Supports two execution modes:
    - Sequential (default): Single JSBridge, one trial at a time
    - Parallel: Uses a JSProcessPool for concurrent trial execution

    The JS trial function receives:
    - trial_id: Unique trial identifier
    - trial_number: Sequential trial number
    - experiment_run_id: ID of the experiment run
    - config: Configuration parameters to test
    - dataset_subset: Indices and total count for dataset sampling

    The JS trial function must return:
    - metrics: Dictionary of metric values
    - duration: Optional duration in seconds
    - metadata: Optional metadata dictionary
    """

    def __init__(
        self,
        js_module: str,
        js_function: str = "runTrial",
        js_timeout: float = 300.0,
        experiment_run_id: str | None = None,
        process_pool: JSProcessPool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize JSEvaluator.

        Args:
            js_module: Path to the JS module containing the trial function.
            js_function: Name of the exported function to call.
            js_timeout: Timeout for trial execution in seconds.
            experiment_run_id: ID of the experiment run for trial tracking.
            process_pool: Optional process pool for parallel execution.
                If provided, trials will be executed via the pool instead of
                a single bridge, enabling concurrent execution.
            **kwargs: Additional arguments passed to BaseEvaluator.
        """
        super().__init__(**kwargs)
        self._js_config = JSEvaluatorConfig(
            js_module=js_module,
            js_function=js_function,
            js_timeout=js_timeout,
            experiment_run_id=experiment_run_id,
        )
        self._bridge: JSBridge | None = None
        self._process_pool = process_pool
        self._trial_counter = 0

    async def _ensure_bridge(self) -> JSBridge:
        """Ensure the JS bridge is started and return it."""
        if self._bridge is None or not self._bridge.is_running:
            bridge_config = JSBridgeConfig(
                module_path=self._js_config.js_module,
                function_name=self._js_config.js_function,
                trial_timeout_seconds=self._js_config.js_timeout,
            )
            self._bridge = JSBridge(bridge_config)
            await self._bridge.start()
        return self._bridge

    async def close(self) -> None:
        """Close the JS bridge and release resources.

        Note: If using a process pool, the pool is NOT closed here since it
        may be shared across multiple evaluators. The pool should be closed
        by the orchestrator.
        """
        if self._bridge is not None:
            await self._bridge.stop()
            self._bridge = None
        # Note: Don't close process_pool here - it may be shared

    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: SampleBudgetLease | None = None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate configuration by running trial in Node.js.

        The Python function `func` is NOT called directly. Instead, the
        configuration is sent to the Node.js process which executes the
        corresponding JS trial function.

        Args:
            func: Python function (not used, but required by interface).
            config: Configuration parameters to test.
            dataset: Evaluation dataset.
            sample_lease: Optional sample budget lease.
            progress_callback: Optional progress callback.

        Returns:
            EvaluationResult with metrics from the JS trial.
        """
        # Generate trial ID
        self._trial_counter += 1
        trial_number = self._trial_counter
        trial_id = f"js_trial_{uuid.uuid4().hex[:8]}"

        # Build dataset subset info
        dataset_size = len(dataset)
        if sample_lease is not None:
            # Respect sample budget
            available = min(sample_lease.remaining, dataset_size)
            indices = list(range(available))
        else:
            indices = list(range(dataset_size))

        # Build trial config for JS
        trial_config = {
            "trial_id": trial_id,
            "trial_number": trial_number,
            "experiment_run_id": self._js_config.experiment_run_id or "unknown",
            "config": config,
            "dataset_subset": {
                "indices": indices,
                "total": dataset_size,
            },
        }

        logger.info(
            "Running JS trial %s (trial #%d) with config: %s",
            trial_id,
            trial_number,
            config,
        )

        try:
            # Choose execution path: pool or single bridge
            if self._process_pool is not None:
                # Parallel mode: Use pool worker
                result = await self._process_pool.run_trial(trial_config)
            else:
                # Sequential mode: Use single bridge (lazy start)
                bridge = await self._ensure_bridge()
                result = await bridge.run_trial(trial_config)
        except JSTrialTimeoutError as e:
            logger.error("JS trial %s timed out: %s", trial_id, e)
            return EvaluationResult(
                config=config,
                example_results=[],
                aggregated_metrics={},
                total_examples=len(indices),
                successful_examples=0,
                duration=0.0,
                errors=[str(e)],
            )
        except JSBridgeError as e:
            logger.error("JS bridge error for trial %s: %s", trial_id, e)
            return EvaluationResult(
                config=config,
                example_results=[],
                aggregated_metrics={},
                total_examples=len(indices),
                successful_examples=0,
                duration=0.0,
                errors=[str(e)],
            )

        # Convert JS result to EvaluationResult
        if result.status == "completed":
            # Mark samples consumed if we have a lease
            consumed = len(indices)
            if sample_lease is not None:
                sample_lease.consume(consumed)

            # Convert metrics to float for aggregated_metrics
            aggregated_metrics: dict[str, float] = {}
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    aggregated_metrics[key] = float(value)

            return EvaluationResult(
                config=config,
                example_results=[],  # JS trials don't return individual outputs
                aggregated_metrics=aggregated_metrics,
                total_examples=consumed,
                successful_examples=consumed,
                duration=result.duration,
                summary_stats=result.metadata,
                examples_consumed=consumed,
            )
        else:
            error_msg = (
                result.error_message or f"Trial failed with status: {result.status}"
            )
            return EvaluationResult(
                config=config,
                example_results=[],
                aggregated_metrics={},
                total_examples=len(indices),
                successful_examples=0,
                duration=result.duration,
                errors=[error_msg],
                summary_stats={
                    "error_code": result.error_code,
                    "retryable": result.retryable,
                },
            )
