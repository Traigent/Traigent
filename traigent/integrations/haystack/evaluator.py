"""Haystack pipeline evaluator for Traigent optimization.

This module provides HaystackEvaluator, which extends the core BaseEvaluator
to enable Haystack pipeline optimization using the existing Traigent
orchestration infrastructure.

Example usage:
    from traigent.integrations.haystack import (
        HaystackEvaluator,
        EvaluationDataset,
    )
    from traigent.optimizers import GridSearchOptimizer
    from traigent.core.orchestrator import OptimizationOrchestrator

    # Create evaluator
    evaluator = HaystackEvaluator(
        pipeline=my_pipeline,
        haystack_dataset=EvaluationDataset.from_dicts([...]),
        metrics=["accuracy"],
    )

    # Use with existing optimizers
    orchestrator = OptimizationOrchestrator(
        optimizer=GridSearchOptimizer(config),
        evaluator=evaluator,
    )
    result = await orchestrator.run_optimization()
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult
from traigent.integrations.haystack.cost_tracking import (
    HaystackCostTracker,
    get_cost_metrics,
)
from traigent.integrations.haystack.evaluation import EvaluationDataset
from traigent.integrations.haystack.execution import (
    ExampleResult,
    execute_with_config,
)
from traigent.integrations.haystack.latency_tracking import (
    compute_latency_stats,
    extract_latencies_from_results,
    get_latency_metrics,
)
from traigent.integrations.haystack.metric_constraints import (
    MetricConstraint,
    check_constraints,
)
from traigent.utils.exceptions import EvaluationError
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.core.sample_budget import SampleBudgetLease

logger = get_logger(__name__)


class HaystackEvaluator(BaseEvaluator):
    """Evaluator for Haystack pipelines using Traigent optimization.

    HaystackEvaluator bridges Haystack pipelines with the Traigent
    optimization infrastructure. It implements the BaseEvaluator
    interface, allowing it to work with:
    - OptimizationOrchestrator
    - GridSearchOptimizer, RandomSearchOptimizer, OptunaOptimizer
    - TrialLifecycle, SampleBudgetManager, CostEnforcer

    The evaluator wraps a Haystack pipeline and runs it against an
    evaluation dataset, collecting outputs and computing metrics.

    Attributes:
        pipeline: The Haystack Pipeline instance to optimize.
        haystack_dataset: The evaluation dataset in Haystack format.
        output_key: Key to extract from pipeline output dict (default: None = use full output).
        track_costs: Whether to track token usage and compute costs.
        track_latency: Whether to compute latency percentiles (p50, p95, p99).
        constraints: List of MetricConstraint for post-evaluation validation.

    Example:
        >>> from haystack import Pipeline
        >>> from traigent.integrations.haystack import HaystackEvaluator, EvaluationDataset

        >>> pipeline = Pipeline()
        >>> # ... add components ...

        >>> dataset = EvaluationDataset.from_dicts([
        ...     {"input": {"query": "What is AI?"}, "expected": "Artificial Intelligence"},
        ... ])

        >>> evaluator = HaystackEvaluator(
        ...     pipeline=pipeline,
        ...     haystack_dataset=dataset,
        ...     metrics=["accuracy"],
        ... )

        >>> # Evaluate with a specific config
        >>> result = await evaluator.evaluate(
        ...     func=pipeline.run,  # The function to evaluate
        ...     config={"generator.temperature": 0.8},
        ...     dataset=dataset.to_core_dataset(),
        ... )
    """

    def __init__(
        self,
        pipeline: Any,
        haystack_dataset: EvaluationDataset,
        metrics: list[str] | None = None,
        timeout: float = 60.0,
        max_workers: int = 1,
        output_key: str | None = None,
        metric_functions: dict[str, Callable[..., Any]] | None = None,
        track_costs: bool = True,
        track_latency: bool = True,
        default_model: str | None = None,
        constraints: list[MetricConstraint] | None = None,
        early_stop_on_violation: bool = False,
        violation_threshold: float = 0.5,
        min_examples_before_stop: int = 3,
        **kwargs: Any,
    ) -> None:
        """Initialize the Haystack evaluator.

        Args:
            pipeline: Haystack Pipeline instance to optimize.
            haystack_dataset: Evaluation dataset in Haystack format.
            metrics: List of metric names to compute (e.g., ["accuracy"]).
            timeout: Timeout for individual evaluations in seconds.
            max_workers: Maximum concurrent evaluations (default 1 for pipelines).
            output_key: Key to extract from pipeline output dict.
                If None, uses the full output dict for comparison.
                E.g., "llm.replies" to get output["llm"]["replies"].
            metric_functions: Custom metric functions to register.
            track_costs: Whether to track token usage and compute costs.
                Defaults to True. Costs are extracted from pipeline outputs.
            track_latency: Whether to compute latency percentiles (p50, p95, p99).
                Defaults to True. Latencies are computed from execution times.
            default_model: Default model name for cost calculation if not
                found in pipeline outputs.
            constraints: List of MetricConstraint for post-evaluation validation.
                Constraints are checked after metrics are computed.
            early_stop_on_violation: If True, stop evaluation early when
                constraint violations exceed threshold. Defaults to False.
            violation_threshold: Fraction of examples that must violate
                constraints before early stopping (0.0-1.0). Defaults to 0.5.
            min_examples_before_stop: Minimum examples to evaluate before
                early stopping is allowed. Defaults to 3.
            **kwargs: Additional configuration passed to BaseEvaluator.
        """
        # Use metric function names as metrics if provided
        if metrics is None and metric_functions:
            metrics = list(metric_functions.keys())

        super().__init__(metrics, timeout, max_workers, **kwargs)

        self.pipeline = pipeline
        self.haystack_dataset = haystack_dataset
        self.output_key = output_key
        self._metric_functions = metric_functions or {}
        self.track_costs = track_costs
        self.track_latency = track_latency
        self._cost_tracker = HaystackCostTracker(model=default_model)
        self.constraints = constraints or []

        # Early stopping configuration
        self.early_stop_on_violation = early_stop_on_violation
        self.violation_threshold = violation_threshold
        self.min_examples_before_stop = min_examples_before_stop

        # Register custom metric functions
        for name, func in self._metric_functions.items():
            self.register_metric(name, func)

    def _create_early_stop_callback(
        self,
    ) -> Callable[[list[ExampleResult]], bool] | None:
        """Create an early stop callback if early stopping is enabled.

        Returns:
            Callback function or None if early stopping is disabled.
        """
        if not self.early_stop_on_violation or not self.constraints:
            return None

        # Find latency constraints for per-example checking
        latency_constraints = [
            c for c in self.constraints if "latency" in c.metric_name.lower()
        ]

        if not latency_constraints:
            return None

        def count_violations(results: list[ExampleResult]) -> int:
            """Count examples that violate latency constraints."""
            violations = 0
            for result in results:
                if self._example_violates_latency(result, latency_constraints):
                    violations += 1
            return violations

        def early_stop_callback(results: list[ExampleResult]) -> bool:
            """Check if we should stop early based on latency violations."""
            if len(results) < self.min_examples_before_stop:
                return False

            violations = count_violations(results)
            violation_fraction = violations / len(results)

            if violation_fraction > self.violation_threshold:
                logger.info(
                    f"Early stopping: {violations}/{len(results)} examples "
                    f"({violation_fraction:.1%}) exceeded latency constraints"
                )
                return True
            return False

        return early_stop_callback

    def _example_violates_latency(
        self,
        result: ExampleResult,
        latency_constraints: list[MetricConstraint],
    ) -> bool:
        """Check if a single example violates any latency constraint."""
        if not result.success:
            return True

        latency_ms = result.execution_time * 1000
        for constraint in latency_constraints:
            if not constraint.check({constraint.metric_name: latency_ms}):
                return True
        return False

    def _resolve_dataset(self, dataset: Dataset) -> EvaluationDataset:
        """Resolve which dataset to use for evaluation.

        If a non-empty core Dataset is provided, convert it to EvaluationDataset.
        Otherwise, fall back to the instance's haystack_dataset.

        Args:
            dataset: Core Dataset passed to evaluate().

        Returns:
            EvaluationDataset to use for evaluation.
        """
        # If dataset is provided and has examples, convert and use it
        if dataset is not None and hasattr(dataset, "examples") and dataset.examples:
            # Convert core Dataset to EvaluationDataset
            return EvaluationDataset.from_core_dataset(dataset)
        # Fall back to instance dataset
        return self.haystack_dataset

    async def evaluate(
        self,
        func: Callable[..., Any],
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: SampleBudgetLease | None = None,
        progress_callback: Callable[[int, dict[str, Any]], Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate the pipeline with given configuration.

        This method implements the BaseEvaluator interface, allowing
        HaystackEvaluator to work with the existing optimization
        infrastructure (OptimizationOrchestrator, TrialLifecycle, etc.).

        Args:
            func: The function to evaluate (typically pipeline.run).
                For Haystack pipelines, this is ignored - we use self.pipeline.
            config: Configuration parameters as qualified names.
                E.g., {"generator.temperature": 0.8, "retriever.top_k": 10}.
            dataset: Core Dataset to evaluate on. If provided and non-empty,
                this overrides the instance's haystack_dataset. The dataset
                should be convertible to EvaluationDataset format.
            sample_lease: Optional sample budget lease for controlling
                per-trial sample consumption.
            progress_callback: Optional callback invoked after each example
                with (example_index, payload) for progress tracking.

        Returns:
            EvaluationResult with metrics, outputs, and execution details.

        Raises:
            EvaluationError: If evaluation fails.
        """
        self.validate_config(config)

        # Use provided dataset if non-empty, otherwise fall back to instance dataset
        working_dataset = self._resolve_dataset(dataset)

        if not working_dataset.examples:
            raise EvaluationError("Haystack dataset cannot be empty")

        logger.info(
            f"Starting Haystack pipeline evaluation with "
            f"{len(working_dataset)} examples, config: {config}"
        )

        start_time = time.time()

        # Determine how many examples to run based on sample_lease
        max_examples = len(working_dataset.examples)
        if sample_lease is not None:
            available = sample_lease.remaining()  # Note: remaining() is a method
            max_examples = min(max_examples, int(available))
            logger.debug(f"Sample budget limits evaluation to {max_examples} examples")

        # Create a subset dataset if limited by budget
        if max_examples < len(working_dataset.examples):
            limited_examples = working_dataset.examples[:max_examples]
            limited_dataset = EvaluationDataset(examples=limited_examples)
        else:
            limited_dataset = working_dataset

        # Create early stop callback if enabled
        early_stop_callback = self._create_early_stop_callback()

        # Execute pipeline with configuration using our Haystack execution module
        run_result = execute_with_config(
            pipeline=self.pipeline,
            config=config,
            dataset=limited_dataset,
            copy_pipeline=True,  # Don't mutate original pipeline
            abort_on_error=False,  # Continue on individual example failures
            early_stop_callback=early_stop_callback,
        )

        # Check for catastrophic errors (e.g., config errors)
        if run_result.error is not None:
            raise EvaluationError(f"Pipeline execution failed: {run_result.error}")

        # Consume samples from lease using try_take() API
        consumed = len(run_result.example_results)
        budget_exhausted = False
        if sample_lease is not None:
            # Use try_take() to properly consume from the budget
            # If budget is exhausted mid-run, this will return False
            if consumed > 0:
                sample_lease.try_take(consumed)
            budget_exhausted = sample_lease.remaining() == 0

        # Extract outputs and errors for metrics computation
        outputs: list[Any] = []
        errors: list[str | None] = []
        expected_outputs: list[Any] = []

        for i, example_result in enumerate(run_result.example_results):
            if example_result.success:
                output = self._extract_output(example_result.output)
                outputs.append(output)
                errors.append(None)
            else:
                outputs.append(None)
                errors.append(example_result.error)

            # Get expected output
            if i < len(limited_dataset.examples):
                expected_outputs.append(limited_dataset.examples[i].expected)
            else:
                expected_outputs.append(None)

        # Invoke progress callbacks for each example
        if progress_callback:
            for i, example_result in enumerate(run_result.example_results):
                payload = {
                    "success": example_result.success,
                    "error": example_result.error,
                    "output": outputs[i] if i < len(outputs) else None,
                    "execution_time": example_result.execution_time,
                }
                progress_callback(i, payload)

        # Compute metrics using the parent class infrastructure
        metrics = self.compute_metrics(
            outputs=outputs,
            expected_outputs=expected_outputs,
            errors=errors,
            config=config,
        )

        # Track costs if enabled
        if self.track_costs:
            raw_outputs = [er.output for er in run_result.example_results if er.success]
            cost_result = self._cost_tracker.extract_and_calculate(raw_outputs)
            cost_metrics = get_cost_metrics(cost_result)
            metrics.update(cost_metrics)

            if cost_result.total_cost > 0:
                logger.debug(
                    f"Cost tracking: ${cost_result.total_cost:.6f} "
                    f"({cost_result.tokens.total_tokens} tokens)"
                )

        # Track latency if enabled
        if self.track_latency:
            latencies = extract_latencies_from_results(
                run_result.example_results,
                include_failed=True,  # Include failed examples in latency stats
            )
            latency_stats = compute_latency_stats(latencies)
            latency_metrics = get_latency_metrics(latency_stats)
            metrics.update(latency_metrics)

            if latency_stats.count > 0:
                logger.debug(
                    f"Latency tracking: p50={latency_stats.p50_ms:.1f}ms, "
                    f"p95={latency_stats.p95_ms:.1f}ms, "
                    f"p99={latency_stats.p99_ms:.1f}ms"
                )

        # Check constraints if defined
        constraints_satisfied = True
        constraint_violations: list[str] = []
        if self.constraints:
            constraint_result = check_constraints(self.constraints, metrics)
            constraints_satisfied = constraint_result.all_satisfied
            constraint_violations = constraint_result.violation_messages

            # Add constraint info to metrics
            metrics["constraints_satisfied"] = constraints_satisfied
            metrics["constraints_checked"] = constraint_result.total_count
            metrics["constraints_passed"] = constraint_result.satisfied_count

            if not constraints_satisfied:
                logger.info(f"Constraint violations: {constraint_violations}")

        # Track early stopping
        stopped_early = run_result.stopped_early
        if stopped_early:
            metrics["stopped_early"] = True
            logger.info(
                f"Evaluation stopped early after {len(run_result.example_results)} "
                f"of {len(limited_dataset)} examples"
            )

        duration = time.time() - start_time

        logger.info(
            f"Haystack evaluation completed: "
            f"{run_result.success_count}/{len(run_result)} succeeded, "
            f"metrics={metrics}, duration={duration:.2f}s"
        )

        # Count successful examples
        successful = sum(1 for e in errors if e is None)

        return EvaluationResult(
            config=config,
            aggregated_metrics=metrics,
            total_examples=consumed,
            successful_examples=successful,
            duration=duration,
            sample_budget_exhausted=budget_exhausted,
            examples_consumed=consumed,
            # Legacy fields for backward compatibility
            metrics=metrics,
            outputs=outputs,
            errors=errors,
        )

    def _extract_output(self, output: Any) -> Any:
        """Extract the relevant value from pipeline output.

        Haystack pipelines return dict outputs like:
        {"llm": {"replies": ["answer"]}, "retriever": {"documents": [...]}}

        This method extracts the specified output_key path.

        Args:
            output: Raw pipeline output dict.

        Returns:
            Extracted value or full output if no key specified.
        """
        if output is None:
            return None

        if self.output_key is None:
            return output

        # Handle nested keys like "llm.replies"
        current = output
        for key in self.output_key.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                logger.warning(
                    f"Output key '{self.output_key}' not found in output, "
                    f"returning None"
                )
                return None

        return current

    def create_pipeline_wrapper(self) -> Callable[..., Any]:
        """Create a wrapper function for the pipeline.

        This is useful when integrating with systems that expect a
        callable function rather than a pipeline object.

        Returns:
            A callable that wraps pipeline.run() with config application.
        """

        def wrapper(**kwargs: Any) -> Any:
            return self.pipeline.run(**kwargs)

        return wrapper

    def get_core_dataset(self) -> Dataset:
        """Get the dataset in core format for orchestrator compatibility.

        This is a convenience method equivalent to
        self.haystack_dataset.to_core_dataset().

        Returns:
            Core Dataset instance.
        """
        return self.haystack_dataset.to_core_dataset()
