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
    from traigent.bridges.js_bridge import JSTrialResult
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

    def _build_trial_config(
        self,
        trial_id: str,
        trial_number: int,
        config: dict[str, Any],
        dataset: Dataset,
        indices: list[int],
    ) -> dict[str, Any]:
        """Build the trial configuration dict for JS execution."""
        dataset_metadata = getattr(dataset, "metadata", {}) or {}
        trial_config: dict[str, Any] = {
            "trial_id": trial_id,
            "trial_number": trial_number,
            "experiment_run_id": self._js_config.experiment_run_id or "unknown",
            "config": config,
            "dataset_subset": {"indices": indices, "total": len(dataset)},
        }

        # Add dataset info if available
        dataset_path = dataset_metadata.get("source_path")
        dataset_hash = dataset_metadata.get("dataset_hash")
        dataset_name = getattr(dataset, "name", None)
        if dataset_path or dataset_hash or dataset_name:
            trial_config["dataset_info"] = {
                "path": dataset_path,
                "hash": dataset_hash,
                "name": dataset_name,
            }
        return trial_config

    def _build_error_result(
        self, config: dict[str, Any], indices: list[int], error: str
    ) -> EvaluationResult:
        """Build an EvaluationResult for error cases."""
        return EvaluationResult(
            config=config,
            example_results=[],
            aggregated_metrics={},
            total_examples=len(indices),
            successful_examples=0,
            duration=0.0,
            errors=[error],
        )

    def _convert_js_result(
        self,
        result: JSTrialResult,
        config: dict[str, Any],
        indices: list[int],
        sample_lease: SampleBudgetLease | None,
    ) -> EvaluationResult:
        """Convert JSTrialResult to EvaluationResult."""
        if result.status == "completed":
            consumed = len(indices)
            if sample_lease is not None:
                sample_lease.try_take(consumed)

            aggregated_metrics = {
                k: float(v)
                for k, v in result.metrics.items()
                if isinstance(v, (int, float))
            }
            return EvaluationResult(
                config=config,
                example_results=[],
                aggregated_metrics=aggregated_metrics,
                total_examples=consumed,
                successful_examples=consumed,
                duration=result.duration,
                summary_stats=result.metadata,
                examples_consumed=consumed,
            )

        error_msg = result.error_message or f"Trial failed with status: {result.status}"
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
        self._trial_counter += 1
        trial_number = self._trial_counter
        trial_id = f"js_trial_{uuid.uuid4().hex[:8]}"

        # Build dataset subset based on sample budget
        dataset_size = len(dataset)
        if sample_lease is not None:
            available = min(int(sample_lease.remaining()), dataset_size)
            indices = list(range(available))
        else:
            indices = list(range(dataset_size))

        trial_config = self._build_trial_config(
            trial_id, trial_number, config, dataset, indices
        )

        logger.info(
            "Running JS trial %s (trial #%d) with config: %s",
            trial_id,
            trial_number,
            config,
        )

        try:
            if self._process_pool is not None:
                result = await self._process_pool.run_trial(trial_config)
            else:
                bridge = await self._ensure_bridge()
                result = await bridge.run_trial(trial_config)
        except (JSTrialTimeoutError, JSBridgeError) as e:
            logger.error("JS trial %s failed: %s", trial_id, e)
            return self._build_error_result(config, indices, str(e))

        return self._convert_js_result(result, config, indices, sample_lease)
