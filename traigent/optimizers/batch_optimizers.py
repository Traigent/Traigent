"""Batch-optimized optimization strategies for TraiGent SDK.

This module provides optimization algorithms specifically designed for batch processing,
including parallel optimization, multi-objective batch optimization, and distributed strategies.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from traigent.api.types import TrialResult
from traigent.evaluators.base import BaseEvaluator, Dataset
from traigent.invokers.base import BaseInvoker
from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.optimizers.results import OptimizationResult, Trial
from traigent.utils.batch_processing import AdaptiveBatchSizer
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BatchOptimizationConfig:
    """Configuration for batch optimization strategies."""

    max_parallel_trials: int = 4
    batch_size: int = 10
    adaptive_batching: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    distributed_workers: int = 1
    enable_checkpointing: bool = True
    memory_limit_mb: float = 1000.0


class ParallelBatchOptimizer(BaseOptimizer):
    """Parallel batch optimizer that runs multiple optimization trials concurrently.

    This optimizer can run multiple configuration trials in parallel, with each trial
    processing its dataset in batches. Useful for CPU-bound optimization tasks.
    """

    def __init__(
        self,
        base_optimizer: BaseOptimizer,
        batch_config: BatchOptimizationConfig,
        objectives: list[str] | None = None,
        context=None,
    ) -> None:
        super().__init__(
            config_space=base_optimizer.config_space,
            objectives=objectives or base_optimizer.objectives,
            context=context,
        )
        self.base_optimizer = base_optimizer
        self.batch_config = batch_config
        self.adaptive_sizer = AdaptiveBatchSizer(
            initial_batch_size=batch_config.batch_size,
            max_batch_size=batch_config.batch_size * 2,
        )

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration to evaluate."""
        return self.base_optimizer.suggest_next_trial(history)

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop."""
        return self.base_optimizer.should_stop(history)

    async def optimize(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: BaseEvaluator,
        max_trials: int = 100,
    ) -> OptimizationResult:
        """Run parallel batch optimization."""
        logger.info(
            f"Starting parallel batch optimization with {self.batch_config.max_parallel_trials} workers"
        )

        start_time = time.time()
        completed_trials: list[Trial] = []
        best_config = None
        best_score = float("-inf")

        # Early stopping tracking
        trials_without_improvement = 0

        # Create configuration candidates
        remaining_configs = list(self.base_optimizer.generate_candidates(max_trials))

        with ThreadPoolExecutor(
            max_workers=self.batch_config.max_parallel_trials
        ) as executor:
            while remaining_configs and len(completed_trials) < max_trials:
                # Submit parallel trials
                current_batch_size = min(
                    self.batch_config.max_parallel_trials,
                    len(remaining_configs),
                    max_trials - len(completed_trials),
                )

                trial_configs = remaining_configs[:current_batch_size]
                remaining_configs = remaining_configs[current_batch_size:]

                # Submit trials to thread pool
                future_to_config = {
                    executor.submit(
                        self._run_single_trial_sync,
                        config,
                        func,
                        dataset,
                        invoker,
                        evaluator,
                    ): config
                    for config in trial_configs
                }

                # Collect results as they complete
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        trial = future.result()
                        completed_trials.append(trial)

                        # Update best result
                        if trial.score > best_score:
                            best_score = trial.score
                            best_config = config
                            trials_without_improvement = 0
                            logger.info(
                                f"New best score: {best_score:.4f} with config: {config}"
                            )
                        else:
                            trials_without_improvement += 1

                        # Early stopping check
                        if (
                            self.batch_config.early_stopping_patience > 0
                            and trials_without_improvement
                            >= self.batch_config.early_stopping_patience
                        ):
                            logger.info(
                                f"Early stopping after {trials_without_improvement} trials without improvement"
                            )
                            break

                    except Exception as e:
                        logger.error(f"Trial failed for config {config}: {e}")
                        # Create failed trial
                        failed_trial = Trial(
                            configuration=config,
                            score=float("-inf"),
                            duration=0.0,
                            metadata={"error": str(e), "failed": True},
                        )
                        completed_trials.append(failed_trial)

                # Break if early stopping triggered
                if (
                    self.batch_config.early_stopping_patience > 0
                    and trials_without_improvement
                    >= self.batch_config.early_stopping_patience
                ):
                    break

        total_duration = time.time() - start_time

        return OptimizationResult(
            best_config=best_config or {},
            best_score=best_score,
            trials=completed_trials,
            duration=total_duration,
            convergence_info={
                "parallel_workers": self.batch_config.max_parallel_trials,
                "early_stopped": trials_without_improvement
                >= self.batch_config.early_stopping_patience,
                "total_trials": len(completed_trials),
            },
        )

    def _run_single_trial_sync(
        self,
        config: dict[str, Any],
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: BaseEvaluator,
    ) -> Trial:
        """Run a single optimization trial synchronously."""
        return asyncio.run(
            self._run_single_trial(config, func, dataset, invoker, evaluator)
        )

    async def _run_single_trial(
        self,
        config: dict[str, Any],
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: BaseEvaluator,
    ) -> Trial:
        """Run a single optimization trial with batch processing."""
        trial_start = time.time()

        try:
            # Process dataset in batches
            all_invocation_results = []
            batch_size = self.adaptive_sizer.get_next_batch_size(len(dataset.examples))

            for i in range(0, len(dataset.examples), batch_size):
                batch_examples = dataset.examples[i : i + batch_size]
                batch_inputs = [ex.input_data for ex in batch_examples]

                # Batch invocation
                if hasattr(invoker, "invoke_batch"):
                    batch_results = await invoker.invoke_batch(
                        func, config, batch_inputs
                    )
                else:
                    # Sequential fallback
                    batch_results = []
                    for input_data in batch_inputs:
                        result = await invoker.invoke(func, config, input_data)
                        batch_results.append(result)

                all_invocation_results.extend(batch_results)

            # Evaluate all results
            expected_outputs = [ex.expected_output for ex in dataset.examples]
            evaluation_result = await evaluator.evaluate(
                all_invocation_results, expected_outputs, dataset  # type: ignore[arg-type]
            )

            # Calculate composite score
            score = self._calculate_composite_score(evaluation_result.metrics or {})

            # Update adaptive batch sizing
            successful_count = evaluation_result.successful_invocations
            error_rate = 1.0 - (
                successful_count / max(1, evaluation_result.total_invocations)
            )

            trial_duration = time.time() - trial_start
            throughput = (
                len(dataset.examples) / trial_duration if trial_duration > 0 else 0
            )

            self.adaptive_sizer.update_performance(
                batch_size=batch_size,
                throughput=throughput,
                memory_usage_mb=0.0,  # Would measure actual memory in production
                error_rate=error_rate,
            )

            return Trial(
                configuration=config,
                score=score,
                duration=trial_duration,
                metadata={
                    "evaluation_result": evaluation_result,
                    "batch_size": batch_size,
                    "throughput": throughput,
                    "error_rate": error_rate,
                },
            )

        except Exception as e:
            logger.error(f"Trial failed for config {config}: {e}")
            return Trial(
                configuration=config,
                score=float("-inf"),
                duration=time.time() - trial_start,
                metadata={"error": str(e), "failed": True},
            )

    def _calculate_composite_score(self, metrics: dict[str, float]) -> float:
        """Calculate composite score from multiple metrics using weighted scalarization."""
        if not metrics:
            return 0.0

        # Use scalarize_objectives for weighted scoring
        from traigent.utils.multi_objective import scalarize_objectives

        return scalarize_objectives(metrics, self.objective_weights)


class MultiObjectiveBatchOptimizer(BaseOptimizer):
    """Multi-objective batch optimizer with Pareto frontier exploration.

    Optimizes multiple objectives simultaneously while processing datasets in batches.
    Maintains a Pareto frontier of non-dominated solutions.
    """

    def __init__(
        self,
        configuration_space: dict[str, Any],
        objectives: list[str],
        batch_config: BatchOptimizationConfig,
        pareto_frontier_size: int = 50,
        context=None,
        objective_weights: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config_space=configuration_space,
            objectives=objectives,
            context=context,
            objective_weights=objective_weights,
            **kwargs,
        )
        self.batch_config = batch_config
        self.pareto_frontier_size = pareto_frontier_size
        self.pareto_frontier: list[Trial] = []
        self._current_trial_count = 0

        # Define objective directions (True = maximize, False = minimize)
        # Cost should be minimized, accuracy should be maximized
        self.objective_directions: dict[str, Any] = {}
        for obj in objectives:
            if (
                "cost" in obj.lower()
                or "latency" in obj.lower()
                or "error" in obj.lower()
            ):
                self.objective_directions[obj] = False  # minimize
            else:
                self.objective_directions[obj] = True  # maximize

        # Use random search for multi-objective exploration
        self._base_optimizer = RandomSearchOptimizer(configuration_space, objectives)

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration to evaluate."""
        return self._base_optimizer.suggest_next_trial(history)

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop."""
        # Stop when we've reached a reasonable number of trials for multi-objective
        return len(history) >= 100

    async def optimize(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: BaseEvaluator,
        max_trials: int = 100,
    ) -> OptimizationResult:
        """Run multi-objective batch optimization."""
        logger.info(
            f"Starting multi-objective batch optimization for {len(self.objectives)} objectives"
        )

        start_time = time.time()
        all_trials = []

        # Use random search for multi-objective exploration
        base_optimizer = RandomSearchOptimizer(self.config_space, self.objectives)

        for trial_idx in range(max_trials):
            # Generate candidate configuration
            candidates = list(base_optimizer.generate_candidates(1))
            if not candidates:
                break

            config = candidates[0]

            # Run trial with batch processing
            trial = await self._run_batch_trial(
                config, func, dataset, invoker, evaluator, trial_idx
            )
            all_trials.append(trial)

            # Update Pareto frontier
            self._update_pareto_frontier(trial)

            # Log progress
            if (trial_idx + 1) % 10 == 0:
                logger.info(
                    f"Completed {trial_idx + 1}/{max_trials} trials, "
                    f"Pareto frontier size: {len(self.pareto_frontier)}"
                )

        # Select best configuration from Pareto frontier
        best_trial = self._select_best_from_pareto()

        total_duration = time.time() - start_time

        return OptimizationResult(
            best_config=best_trial.configuration if best_trial else {},
            best_score=best_trial.score if best_trial else float("-inf"),
            trials=all_trials,
            duration=total_duration,
            convergence_info={
                "pareto_frontier": [
                    {
                        "config": trial.configuration,
                        "scores": trial.metadata.get("objective_scores", {}),
                        "composite_score": trial.score,
                    }
                    for trial in self.pareto_frontier
                ],
                "pareto_frontier_size": len(self.pareto_frontier),
                "objectives": self.objectives,
            },
        )

    async def _run_batch_trial(
        self,
        config: dict[str, Any],
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: BaseEvaluator,
        trial_idx: int,
    ) -> Trial:
        """Run a single trial with batch processing and multi-objective evaluation."""
        trial_start = time.time()

        try:
            # Adaptive batch processing
            batch_size = self.batch_config.batch_size
            all_invocation_results = []

            # Process in batches
            for i in range(0, len(dataset.examples), batch_size):
                batch_examples = dataset.examples[i : i + batch_size]
                batch_inputs = [ex.input_data for ex in batch_examples]

                # Batch invocation
                if hasattr(invoker, "invoke_batch"):
                    batch_results = await invoker.invoke_batch(
                        func, config, batch_inputs
                    )
                else:
                    batch_results = []
                    for input_data in batch_inputs:
                        result = await invoker.invoke(func, config, input_data)
                        batch_results.append(result)

                all_invocation_results.extend(batch_results)

            # Multi-objective evaluation
            expected_outputs = [ex.expected_output for ex in dataset.examples]
            evaluation_result = await evaluator.evaluate(
                all_invocation_results, expected_outputs, dataset  # type: ignore[arg-type]
            )

            # Extract objective scores
            objective_scores = {}
            metrics = evaluation_result.metrics or {}
            for objective in self.objectives:
                objective_scores[objective] = metrics.get(objective, 0.0)

            # Calculate composite score using weighted averaging
            # Import the scalarize_objectives function for proper weighted scoring
            from traigent.utils.multi_objective import scalarize_objectives

            # Get weights from config or use equal weights by default
            objective_weights = getattr(self, "objective_weights", {})
            if not objective_weights:
                # Equal weights for all objectives if not specified
                objective_weights = dict.fromkeys(self.objectives, 1.0)

            # Use the scalarize_objectives function for proper weighted averaging
            composite_score = scalarize_objectives(objective_scores, objective_weights)

            trial_duration = time.time() - trial_start

            return Trial(
                configuration=config,
                score=composite_score,
                duration=trial_duration,
                metadata={
                    "evaluation_result": evaluation_result,
                    "objective_scores": objective_scores,
                    "trial_index": trial_idx,
                    "batch_processing": True,
                },
            )

        except Exception as e:
            logger.error(f"Multi-objective trial {trial_idx} failed: {e}")
            return Trial(
                configuration=config,
                score=float("-inf"),
                duration=time.time() - trial_start,
                metadata={"error": str(e), "failed": True, "trial_index": trial_idx},
            )

    def _update_pareto_frontier(self, new_trial: Trial) -> None:
        """Update Pareto frontier with new trial."""
        if (
            new_trial.score == float("-inf")
            or "objective_scores" not in new_trial.metadata
        ):
            return

        new_scores = new_trial.metadata["objective_scores"]

        # Check if new trial is dominated by existing solutions
        is_dominated = False
        dominated_trials = []

        for existing_trial in self.pareto_frontier:
            existing_scores = existing_trial.metadata.get("objective_scores", {})

            # Check dominance relationships
            new_dominates_existing = self._dominates(new_scores, existing_scores)
            existing_dominates_new = self._dominates(existing_scores, new_scores)

            if existing_dominates_new:
                is_dominated = True
                break
            elif new_dominates_existing:
                dominated_trials.append(existing_trial)

        # Add new trial to frontier if not dominated
        if not is_dominated:
            # Remove dominated trials
            for dominated in dominated_trials:
                self.pareto_frontier.remove(dominated)

            # Add new trial
            self.pareto_frontier.append(new_trial)

            # Maintain frontier size limit
            if len(self.pareto_frontier) > self.pareto_frontier_size:
                # Remove trial with lowest composite score
                self.pareto_frontier.sort(key=lambda t: t.score, reverse=True)
                self.pareto_frontier = self.pareto_frontier[: self.pareto_frontier_size]

    def _dominates(self, scores1: dict[str, float], scores2: dict[str, float]) -> bool:
        """Check if scores1 dominates scores2 (all objectives better or equal, at least one strictly better)."""
        if not scores1 or not scores2:
            return False

        all_better_or_equal = True
        at_least_one_better = False

        for objective in self.objectives:
            score1 = scores1.get(objective, 0.0)
            score2 = scores2.get(objective, 0.0)

            # Check if we should maximize (True) or minimize (False) this objective
            should_maximize = self.objective_directions.get(objective, True)

            if should_maximize:
                # Higher is better
                if score1 < score2:
                    all_better_or_equal = False
                    break
                elif score1 > score2:
                    at_least_one_better = True
            else:
                # Lower is better (for cost, latency, etc.)
                if score1 > score2:
                    all_better_or_equal = False
                    break
                elif score1 < score2:
                    at_least_one_better = True

        return all_better_or_equal and at_least_one_better

    def _select_best_from_pareto(self) -> Trial | None:
        """Select best configuration from Pareto frontier."""
        if not self.pareto_frontier:
            return None

        # For now, select the one with highest composite score
        return max(self.pareto_frontier, key=lambda t: t.score)


class AdaptiveBatchOptimizer(BaseOptimizer):
    """Adaptive batch optimizer that adjusts batch sizes based on performance."""

    def __init__(
        self,
        base_optimizer: BaseOptimizer,
        batch_config: BatchOptimizationConfig,
        context=None,
    ) -> None:
        super().__init__(
            config_space=base_optimizer.config_space,
            objectives=base_optimizer.objectives,
            context=context,
        )
        self.base_optimizer = base_optimizer
        self.batch_config = batch_config
        self.adaptive_sizer = AdaptiveBatchSizer(
            initial_batch_size=batch_config.batch_size,
            target_memory_mb=batch_config.memory_limit_mb,
        )
        self.performance_history: list[Any] = []

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration to evaluate."""
        return self.base_optimizer.suggest_next_trial(history)

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop."""
        return self.base_optimizer.should_stop(history)

    async def optimize(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: BaseEvaluator,
        max_trials: int = 100,
    ) -> OptimizationResult:
        """Run adaptive batch optimization."""
        logger.info("Starting adaptive batch optimization")

        start_time = time.time()
        trials = []
        best_config = None
        best_score = float("-inf")

        for trial_idx in range(max_trials):
            # Generate configuration
            candidates = list(self.base_optimizer.generate_candidates(1))
            if not candidates:
                break

            config = candidates[0]

            # Run trial with adaptive batching
            trial = await self._run_adaptive_trial(
                config, func, dataset, invoker, evaluator, trial_idx
            )
            trials.append(trial)

            # Update best result
            if trial.score > best_score:
                best_score = trial.score
                best_config = config
                logger.info(f"Trial {trial_idx}: New best score {best_score:.4f}")

            # Update performance tracking
            self._update_performance_history(trial)

        total_duration = time.time() - start_time

        return OptimizationResult(
            best_config=best_config or {},
            best_score=best_score,
            trials=trials,
            duration=total_duration,
            convergence_info={
                "adaptive_batching": True,
                "final_batch_size": self.adaptive_sizer.current_batch_size,
                "performance_history": self.performance_history[
                    -10:
                ],  # Last 10 entries
                "total_trials": len(trials),
            },
        )

    async def _run_adaptive_trial(
        self,
        config: dict[str, Any],
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: BaseEvaluator,
        trial_idx: int,
    ) -> Trial:
        """Run trial with adaptive batch sizing."""
        trial_start = time.time()

        try:
            # Get adaptive batch size
            total_examples = len(dataset.examples)
            current_batch_size = self.adaptive_sizer.get_next_batch_size(total_examples)

            logger.debug(f"Trial {trial_idx}: Using batch size {current_batch_size}")

            # Process with current batch size
            all_invocation_results = []
            batch_durations = []

            for i in range(0, total_examples, current_batch_size):
                batch_start = time.time()

                batch_examples = dataset.examples[i : i + current_batch_size]
                batch_inputs = [ex.input_data for ex in batch_examples]

                # Batch invocation
                if hasattr(invoker, "invoke_batch"):
                    batch_results = await invoker.invoke_batch(
                        func, config, batch_inputs
                    )
                else:
                    batch_results = []
                    for input_data in batch_inputs:
                        result = await invoker.invoke(func, config, input_data)
                        batch_results.append(result)

                all_invocation_results.extend(batch_results)
                batch_durations.append(time.time() - batch_start)

            # Evaluate results
            expected_outputs = [ex.expected_output for ex in dataset.examples]
            evaluation_result = await evaluator.evaluate(
                all_invocation_results, expected_outputs, dataset  # type: ignore[arg-type]
            )

            # Calculate metrics for adaptive sizing
            trial_duration = time.time() - trial_start
            throughput = total_examples / trial_duration if trial_duration > 0 else 0
            error_rate = 1.0 - (
                evaluation_result.successful_invocations
                / max(1, evaluation_result.total_invocations)
            )
            avg_batch_duration = (
                sum(batch_durations) / len(batch_durations) if batch_durations else 0
            )

            # Update adaptive sizer
            self.adaptive_sizer.update_performance(
                batch_size=current_batch_size,
                throughput=throughput,
                memory_usage_mb=0.0,  # Would measure actual memory in production
                error_rate=error_rate,
            )

            # Calculate score
            score = self._calculate_composite_score(evaluation_result.metrics or {})

            return Trial(
                configuration=config,
                score=score,
                duration=trial_duration,
                metadata={
                    "evaluation_result": evaluation_result,
                    "batch_size": current_batch_size,
                    "throughput": throughput,
                    "error_rate": error_rate,
                    "avg_batch_duration": avg_batch_duration,
                    "trial_index": trial_idx,
                },
            )

        except Exception as e:
            logger.error(f"Adaptive trial {trial_idx} failed: {e}")
            return Trial(
                configuration=config,
                score=float("-inf"),
                duration=time.time() - trial_start,
                metadata={"error": str(e), "failed": True, "trial_index": trial_idx},
            )

    def _update_performance_history(self, trial: Trial) -> None:
        """Update performance history for monitoring."""
        if trial.score != float("-inf"):
            self.performance_history.append(
                {
                    "trial_index": trial.metadata.get("trial_index", 0),
                    "score": trial.score,
                    "batch_size": trial.metadata.get("batch_size", 0),
                    "throughput": trial.metadata.get("throughput", 0),
                    "error_rate": trial.metadata.get("error_rate", 0),
                    "duration": trial.duration,
                }
            )

            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

    def _calculate_composite_score(self, metrics: dict[str, float]) -> float:
        """Calculate composite score from multiple metrics using weighted scalarization."""
        if not metrics:
            return 0.0

        # Use scalarize_objectives for weighted scoring
        from traigent.utils.multi_objective import scalarize_objectives

        return scalarize_objectives(metrics, self.objective_weights)
