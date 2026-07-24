"""Batch-optimized optimization strategies for Traigent SDK.

This module provides optimization algorithms specifically designed for batch processing,
including parallel optimization, multi-objective batch optimization, and distributed strategies.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Performance CONC-Quality-Reliability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import asyncio
import math
import numbers
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol, cast

from traigent.api.types import TrialResult
from traigent.evaluators.base import Dataset, EvaluationResult
from traigent.invokers.base import BaseInvoker, InvocationResult
from traigent.optimizers.base import BaseOptimizer
from traigent.optimizers.random import RandomSearchOptimizer
from traigent.optimizers.results import OptimizationResult, Trial
from traigent.utils.batch_processing import AdaptiveBatchSizer
from traigent.utils.logging import get_logger
from traigent.utils.objectives import is_minimization_objective

logger = get_logger(__name__)


class TrialBatchEvaluator(Protocol):
    """Protocol for evaluators that score precomputed invocation batches."""

    async def evaluate(
        self,
        invocation_results: list[InvocationResult],
        expected_outputs: list[Any],
        dataset: Dataset,
    ) -> EvaluationResult: ...


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

    def __post_init__(self) -> None:
        # Mirrors OptimizationStrategy's boundary (remote_services.py): now that
        # the batch loop actually consumes early_stopping_min_delta, an
        # unvalidated NaN makes every improvement comparison False (premature
        # stop on improving runs) and a negative delta makes flat scores count
        # as improvements (never stops). bool is a Real, so reject it explicitly.
        delta = self.early_stopping_min_delta
        if (
            isinstance(delta, bool)
            or not isinstance(delta, numbers.Real)
            or not math.isfinite(float(delta))
            or delta < 0
        ):
            raise ValueError(
                f"early_stopping_min_delta must be a finite non-negative number, "
                f"got {delta!r}"
            )


class ParallelBatchOptimizer(BaseOptimizer):
    """Parallel batch optimizer that runs multiple optimization trials concurrently.

    This optimizer can run multiple configuration trials in parallel, with each trial
    processing its dataset in batches. Useful for CPU-bound optimization tasks.
    """

    def __init__(
        self,
        config_space: dict[str, Any] | BaseOptimizer | None = None,
        objectives: list[str] | BatchOptimizationConfig | None = None,
        base_optimizer: BaseOptimizer | None = None,
        batch_config: BatchOptimizationConfig | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> None:
        # Legacy positional form: ParallelBatchOptimizer(base_optimizer, batch_config).
        # Detect by type to keep the registry-standard
        # (config_space, objectives, **kwargs) path working.
        if isinstance(config_space, BaseOptimizer):
            if base_optimizer is None:
                base_optimizer = config_space
            config_space = None
        if isinstance(objectives, BatchOptimizationConfig):
            if batch_config is None:
                batch_config = objectives
            objectives = None

        if base_optimizer is None:
            if config_space is None:
                raise ValueError(
                    "ParallelBatchOptimizer requires either a base_optimizer "
                    "or a config_space."
                )
            base_optimizer = RandomSearchOptimizer(config_space, objectives or [])
        if batch_config is None:
            batch_config = BatchOptimizationConfig()

        resolved_objectives = (
            objectives if objectives is not None else base_optimizer.objectives
        )
        super().__init__(
            config_space=base_optimizer.config_space,
            objectives=resolved_objectives,
            context=context,
            **kwargs,
        )
        self.base_optimizer = base_optimizer
        self.batch_config = batch_config
        self.adaptive_sizer = AdaptiveBatchSizer(
            initial_batch_size=batch_config.batch_size,
            max_batch_size=batch_config.batch_size * 2,
        )

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration to evaluate."""
        return cast(dict[str, Any], self.base_optimizer.suggest_next_trial(history))

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop."""
        return bool(self.base_optimizer.should_stop(history))

    async def optimize(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: TrialBatchEvaluator,
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

                        # Patience counter resets only on a MATERIAL improvement
                        # (> min_delta over the previous best); sub-delta creep
                        # must not defer early stopping. Previously min_delta was
                        # declared in BatchConfig but never read, so any epsilon
                        # gain reset the counter.
                        if (
                            trial.score
                            > best_score + self.batch_config.early_stopping_min_delta
                        ):
                            trials_without_improvement = 0
                        else:
                            trials_without_improvement += 1

                        # Best-result tracking stays exact: even a sub-delta gain
                        # is still the best config seen so far.
                        if trial.score > best_score:
                            best_score = trial.score
                            best_config = config
                            logger.info(
                                f"New best score: {best_score:.4f} with config: {config}"
                            )

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
        evaluator: TrialBatchEvaluator,
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
        evaluator: TrialBatchEvaluator,
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
                all_invocation_results,
                expected_outputs,
                dataset,  # type: ignore[arg-type]
            )

            # Calculate composite score
            score = self._calculate_composite_score(evaluation_result.metrics or {})

            # Update adaptive batch sizing
            successful_count = evaluation_result.successful_examples
            error_rate = 1.0 - (
                successful_count / max(1, evaluation_result.total_examples)
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

        # Use scalarize_objectives for weighted scoring, honoring objective
        # orientation so minimize objectives (cost/latency/error) lower the
        # composite instead of raising it (#1466).
        from traigent.utils.multi_objective import scalarize_objectives

        return float(
            scalarize_objectives(
                metrics,
                self.objective_weights,
                minimize_objectives=self._minimize_objectives,
            )
        )


class MultiObjectiveBatchOptimizer(BaseOptimizer):
    """Multi-objective batch optimizer with Pareto frontier exploration.

    Optimizes multiple objectives simultaneously while processing datasets in batches.
    Maintains a Pareto frontier of non-dominated solutions.
    """

    def __init__(
        self,
        configuration_space: dict[str, Any] | None = None,
        objectives: list[str] | None = None,
        batch_config: BatchOptimizationConfig | None = None,
        pareto_frontier_size: int = 50,
        context=None,
        objective_weights: dict[str, float] | None = None,
        *,
        config_space: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Support both `configuration_space=` (legacy) and `config_space=`
        # (registry-standard) keyword aliases.
        if configuration_space is None:
            configuration_space = config_space
        if configuration_space is None:
            raise ValueError(
                "MultiObjectiveBatchOptimizer requires a configuration_space."
            )
        if batch_config is None:
            batch_config = BatchOptimizationConfig()
        super().__init__(
            config_space=configuration_space,
            objectives=objectives or [],
            context=context,
            objective_weights=objective_weights,
            **kwargs,
        )
        self.batch_config = batch_config
        self.pareto_frontier_size = pareto_frontier_size
        self.pareto_frontier: list[Trial] = []
        self._current_trial_count = 0

        # Define objective directions (True = maximize, False = minimize).
        # Derive from the canonical orientation helper (#1466) so Pareto
        # dominance in ``_dominates`` uses the same orientation as the
        # composite/scalarized score (cost/latency/error/loss/... → minimize).
        resolved_objectives = objectives or []
        self.objective_directions: dict[str, Any] = {
            obj: not is_minimization_objective(obj) for obj in resolved_objectives
        }

        # Use random search for multi-objective exploration. Thread the caller's
        # ``max_trials`` (default 100, matching the historical hard-coded cap) so
        # the delegated ``should_stop`` honours the configured budget and
        # config-space exhaustion instead of a magic constant (#1916).
        #
        # Normalise a direct-constructor ``max_trials=None`` to the historical
        # 100-trial cap. ``RandomSearchOptimizer.should_stop`` evaluates
        # ``self._trial_count >= self.max_trials``, which raises
        # ``TypeError: '>=' not supported between 'int' and 'NoneType'`` on a
        # ``None`` budget; 100 matches the legacy ``len(history) >= 100`` cap.
        requested_max_trials = kwargs.get("max_trials", 100)
        if requested_max_trials is None:
            requested_max_trials = 100
        self._max_trials = requested_max_trials
        self._base_optimizer = RandomSearchOptimizer(
            configuration_space,
            resolved_objectives,
            max_trials=requested_max_trials,
        )

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        """Suggest next configuration to evaluate."""
        return cast(dict[str, Any], self._base_optimizer.suggest_next_trial(history))

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop.

        Combines two budgets so neither the resumed-history nor the fresh-run
        path regresses (#1916 follow-up):

        * Legacy resumed-history cap - stop once ``len(history)`` reaches the
          configured budget. ``optimize`` drives trials through a *local*
          RandomSearchOptimizer, so the injected ``_base_optimizer`` trial
          counter stays at zero on a resumed run; without this check a
          pre-populated history would never trigger a stop (the historical
          behaviour was ``len(history) >= 100``).
        * Delegate budget / exhaustion - otherwise defer to the injected base
          optimizer so its configured ``max_trials`` and discrete config-space
          exhaustion are honoured instead of a hard-coded 100-trial cap.
        """
        if len(history) >= self._max_trials:
            return True
        return bool(self._base_optimizer.should_stop(history))

    async def optimize(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: TrialBatchEvaluator,
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
        evaluator: TrialBatchEvaluator,
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
                all_invocation_results,
                expected_outputs,
                dataset,  # type: ignore[arg-type]
            )

            # Extract objective scores and scalarize, rejecting the trial
            # outright when a metric is missing or non-finite (#1944 hardening).
            composed = self._compose_trial_scores(evaluation_result.metrics or {})
            trial_duration = time.time() - trial_start

            if composed is None:
                metrics = evaluation_result.metrics or {}
                missing = [o for o in self.objectives if o not in metrics]
                logger.warning(
                    "Multi-objective trial %s rejected: metric-incomplete or "
                    "non-finite objectives (missing=%s) — a partial trial must "
                    "never enter the Pareto frontier.",
                    trial_idx,
                    missing,
                )
                return Trial(
                    configuration=config,
                    score=float("-inf"),
                    duration=trial_duration,
                    metadata={
                        "evaluation_result": evaluation_result,
                        "failed": True,
                        "metric_incomplete": missing,
                        "trial_index": trial_idx,
                        "batch_processing": True,
                    },
                )

            objective_scores, composite_score = composed
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

    def _compose_trial_scores(
        self, metrics: dict[str, Any]
    ) -> tuple[dict[str, float], float] | None:
        """Extract objective scores and the scalarized composite for one trial.

        Returns ``None`` — reject the trial — when any declared objective is
        missing from ``metrics``, has a non-finite value, or the composite
        itself is non-finite. Rejection must happen HERE, before the trial is
        built: the previous orientation-worst sentinel (±inf) approach (#1944)
        leaked through scalarization when an objective carried an *allowed*
        zero weight — ``0.0 * ±inf = nan`` — and the downstream
        ``score == -inf`` gate cannot see NaN (every NaN comparison is False),
        so the metric-incomplete trial entered the frontier and, inserted
        first, could even be SELECTED by ``max()`` (NaN keeps first place).
        """
        for objective in self.objectives:
            value = metrics.get(objective)
            if value is None or not math.isfinite(value):
                return None

        objective_scores = {
            objective: float(metrics[objective]) for objective in self.objectives
        }

        from traigent.utils.multi_objective import scalarize_objectives

        objective_weights = getattr(self, "objective_weights", None) or {}
        if not objective_weights:
            objective_weights = dict.fromkeys(self.objectives, 1.0)

        # Weighted, orientation-aware scalarization (#1466): minimize
        # objectives lower the composite, consistent with ``_dominates`` and
        # the ``max(score)`` pick in ``_select_best_from_pareto``.
        composite_score = scalarize_objectives(
            objective_scores,
            objective_weights,
            minimize_objectives=self._minimize_objectives,
        )
        if not math.isfinite(composite_score):
            return None
        return objective_scores, composite_score

    def _update_pareto_frontier(self, new_trial: Trial) -> None:
        """Update Pareto frontier with new trial."""
        # Fail-closed admission (#1944 hardening): gate on ``isfinite``, not a
        # ``== -inf`` sentinel comparison — NaN compares False to everything,
        # so a NaN-scored trial sailed through the old gate. Also reject any
        # non-finite *objective* value so ±inf/NaN can never sit on the
        # frontier regardless of how the trial was constructed.
        if (
            not math.isfinite(new_trial.score)
            or "objective_scores" not in new_trial.metadata
        ):
            return

        new_scores = new_trial.metadata["objective_scores"]
        if any(v is None or not math.isfinite(v) for v in new_scores.values()):
            return
        # Completeness is enforced HERE at the choke point, not only in
        # _compose_trial_scores: direct callers could otherwise insert a
        # finite-but-INCOMPLETE mapping (e.g. {"accuracy": 0.91} with cost
        # declared), which is non-comparable under _dominates and would sit on
        # the frontier as an unmeasured, falsely-attractive point.
        if any(objective not in new_scores for objective in self.objectives):
            return

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

            # Maintain frontier size limit by pruning the *least diverse*
            # points, not the lowest composite score (#1942). Every member is
            # mutually non-dominated, so trimming by descending ``t.score`` would
            # silently evict Pareto-optimal trade-off extremes (e.g. the
            # cost-minimal corner) and collapse the frontier toward the
            # highest-scalar corner. Crowding distance (NSGA-II) assigns the
            # boundary points infinite distance, so the extremes are retained.
            while len(self.pareto_frontier) > self.pareto_frontier_size:
                distances = self._crowding_distances(self.pareto_frontier)
                most_crowded = min(
                    range(len(self.pareto_frontier)),
                    key=lambda i: distances[i],
                )
                self.pareto_frontier.pop(most_crowded)

    def _crowding_distances(self, trials: list[Trial]) -> list[float]:
        """Compute NSGA-II crowding distance for each trial on the frontier.

        Boundary points (min/max on any objective) receive ``inf`` so the
        trade-off extremes are never pruned; interior points accumulate the
        normalized gap to their neighbours per objective. Higher = more
        isolated = more valuable to keep for diversity.
        """
        n = len(trials)
        if n <= 2:
            # With two or fewer points every point is a boundary extreme.
            return [float("inf")] * n

        distances = [0.0] * n
        for objective in self.objectives:
            values = [
                t.metadata.get("objective_scores", {}).get(objective, 0.0)
                for t in trials
            ]
            order = sorted(range(n), key=lambda i: values[i])

            # Boundary points are the diversity extremes — always keep them.
            distances[order[0]] = float("inf")
            distances[order[-1]] = float("inf")

            span = values[order[-1]] - values[order[0]]
            if span <= 0 or not math.isfinite(span):
                # Degenerate/undefined spread on this objective — skip it rather
                # than inject NaN/inf into interior distances.
                continue

            for k in range(1, n - 1):
                idx = order[k]
                if math.isinf(distances[idx]):
                    continue
                gap = values[order[k + 1]] - values[order[k - 1]]
                distances[idx] += gap / span

        return distances

    def _dominates(self, scores1: dict[str, float], scores2: dict[str, float]) -> bool:
        """Check if scores1 dominates scores2 (all objectives better or equal, at least one strictly better)."""
        if not scores1 or not scores2:
            return False

        all_better_or_equal = True
        at_least_one_better = False

        for objective in self.objectives:
            # Check if we should maximize (True) or minimize (False) this objective
            should_maximize = self.objective_directions.get(objective, True)

            # Default a missing objective to the orientation-*worst* sentinel,
            # never 0.0 (#1944) — for a minimize objective 0.0 is the best value,
            # which would let a metric-incomplete trial falsely dominate.
            worst = float("-inf") if should_maximize else float("inf")
            score1 = scores1.get(objective, worst)
            score2 = scores2.get(objective, worst)

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
        config_space: dict[str, Any] | BaseOptimizer | None = None,
        objectives: list[str] | BatchOptimizationConfig | None = None,
        base_optimizer: BaseOptimizer | None = None,
        batch_config: BatchOptimizationConfig | None = None,
        context: Any = None,
        **kwargs: Any,
    ) -> None:
        # Legacy positional form: AdaptiveBatchOptimizer(base_optimizer, batch_config).
        # Detect by type to keep the registry-standard
        # (config_space, objectives, **kwargs) path working.
        if isinstance(config_space, BaseOptimizer):
            if base_optimizer is None:
                base_optimizer = config_space
            config_space = None
        if isinstance(objectives, BatchOptimizationConfig):
            if batch_config is None:
                batch_config = objectives
            objectives = None

        if base_optimizer is None:
            if config_space is None:
                raise ValueError(
                    "AdaptiveBatchOptimizer requires either a base_optimizer "
                    "or a config_space."
                )
            base_optimizer = RandomSearchOptimizer(config_space, objectives or [])
        if batch_config is None:
            batch_config = BatchOptimizationConfig()

        resolved_objectives = (
            objectives if objectives is not None else base_optimizer.objectives
        )
        super().__init__(
            config_space=base_optimizer.config_space,
            objectives=resolved_objectives,
            context=context,
            **kwargs,
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
        return cast(dict[str, Any], self.base_optimizer.suggest_next_trial(history))

    def should_stop(self, history: list[TrialResult]) -> bool:
        """Determine if optimization should stop."""
        return bool(self.base_optimizer.should_stop(history))

    async def optimize(
        self,
        func: Callable[..., Any],
        dataset: Dataset,
        invoker: BaseInvoker,
        evaluator: TrialBatchEvaluator,
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
        evaluator: TrialBatchEvaluator,
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
                all_invocation_results,
                expected_outputs,
                dataset,  # type: ignore[arg-type]
            )

            # Calculate metrics for adaptive sizing
            trial_duration = time.time() - trial_start
            throughput = total_examples / trial_duration if trial_duration > 0 else 0
            error_rate = 1.0 - (
                evaluation_result.successful_examples
                / max(1, evaluation_result.total_examples)
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

        # Use scalarize_objectives for weighted scoring, honoring objective
        # orientation so minimize objectives (cost/latency/error) lower the
        # composite instead of raising it (#1466).
        from traigent.utils.multi_objective import scalarize_objectives

        return float(
            scalarize_objectives(
                metrics,
                self.objective_weights,
                minimize_objectives=self._minimize_objectives,
            )
        )
