"""Progress tracking and pruning for Optuna trial optimization."""

# Traceability: CONC-Layer-Core CONC-Quality-Performance FUNC-OPT-ALGORITHMS FUNC-ORCH-LIFECYCLE REQ-OPT-ALG-004 REQ-ORCH-003 SYNC-OptimizationFlow

from typing import TYPE_CHECKING, Any, cast

from traigent.evaluators.base import Dataset
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.exceptions import TrialPrunedError
from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from traigent.tvl.models import BandTarget

logger = get_logger(__name__)


class PruningProgressTracker:
    """Tracks evaluation progress and triggers pruning for Optuna trials.

    This tracker receives progress callbacks during trial evaluation and:
    1. Updates state (evaluated count, accuracy, cost)
    2. Collects objective values (with cost projection for stronger signals)
    3. Reports intermediate values to Optuna
    4. Triggers pruning when optimizer decides trial should stop
    5. For banded objectives, triggers pruning when values are far outside band
    """

    # Multiplier for band width to determine pruning threshold
    # Values more than BAND_PRUNE_THRESHOLD * band_width from band center are pruned
    BAND_PRUNE_THRESHOLD = 2.0

    def __init__(
        self,
        optimizer: BaseOptimizer,
        dataset: Dataset,
        trial_id: str,
        optuna_trial_id: int,
        *,
        band_target: "BandTarget | None" = None,
    ) -> None:
        """Initialize pruning progress tracker.

        Args:
            optimizer: Optimizer instance (for should_prune checks)
            dataset: Evaluation dataset
            trial_id: Trial identifier for logging
            optuna_trial_id: Optuna trial identifier
            band_target: Optional band target for banded objective pruning
        """
        self.optimizer = optimizer
        self.dataset = dataset
        self.trial_id = trial_id
        self.optuna_trial_id = optuna_trial_id
        self.band_target = band_target

        total_examples = len(dataset.examples) if hasattr(dataset, "examples") else 0

        self.state: dict[str, Any] = {
            "evaluated": 0,
            "correct_sum": 0.0,
            "total_cost": 0.0,
            "total_examples": total_examples,
        }

    def callback(self, step_index: int, payload: dict[str, Any]) -> None:
        """Progress callback that updates state and checks for pruning.

        Args:
            step_index: Current evaluation step index
            payload: Evaluation payload containing metrics and output

        Raises:
            TrialPrunedError: If trial should be pruned
        """
        self._update_state(step_index, payload)

        values, optimistic_values = self._collect_objective_values(payload)
        if not values:
            return

        report_value_list: list[float] = (
            optimistic_values if optimistic_values else values
        )
        report_value: list[float] | float = report_value_list
        if len(report_value_list) == 1:
            report_value = report_value_list[0]

        # Optuna expects monotonically increasing step indices. In concurrent
        # evaluation modes, examples may complete out-of-order, so we report
        # progress using the number of completed callbacks instead of the raw
        # dataset index.
        report_step = int(self.state.get("evaluated", 0)) - 1

        self._log_progress(report_step, report_value)

        should_prune = self.optimizer.report_intermediate_value(  # type: ignore[attr-defined]
            self.optuna_trial_id,
            step=report_step,
            value=report_value,
        )

        # Check band-based pruning if we have a band target
        band_prune = self._should_prune_for_band(report_value)
        if band_prune:
            should_prune = True

        if should_prune:
            prune_reason = "band-based" if band_prune else "optuna"
            logger.info("   → Pruning decision: prune (%s)", prune_reason)
        else:
            logger.debug("   → Pruning decision: keep")

        if should_prune:
            raise TrialPrunedError(step=report_step)

    def _update_state(self, step_index: int, payload: dict[str, Any]) -> None:
        """Update internal state with evaluation results."""
        self.state["evaluated"] += 1

        metrics_payload = payload.get("metrics") or {}

        accuracy_value = self._extract_accuracy(step_index, payload, metrics_payload)
        if accuracy_value is not None:
            self.state["correct_sum"] += float(accuracy_value)

        cost_value = self._extract_cost(payload, metrics_payload)
        if cost_value is not None:
            self.state["total_cost"] += cost_value

    def _collect_objective_values(
        self, payload: dict[str, Any]
    ) -> tuple[list[float], list[float]]:
        """Collect objective values from payload with cost projection.

        For cost objectives, projects total cost based on current per-example rate
        to give Optuna a stronger signal about expected final cost.

        Args:
            payload: Evaluation payload

        Returns:
            Tuple of (values, optimistic_values)
        """
        metrics_payload = payload.get("metrics") or {}
        values: list[float] = []
        optimistic_values: list[float] = []

        for objective in getattr(self.optimizer, "objectives", []):
            value = metrics_payload.get(objective)
            if value is None:
                value = self._fallback_objective_value(objective)

            if value is None:
                continue

            try:
                float_value = float(value)
            except (TypeError, ValueError):
                logger.debug(
                    "Unable to convert value %s for objective %s in trial %s",
                    value,
                    objective,
                    self.trial_id,
                )
                continue

            # For cost objectives, project total cost based on current per-example rate
            # This gives Optuna a stronger signal about expected final cost
            if (
                objective in {"cost", "total_cost"}
                and self.state.get("evaluated", 0) > 0
            ):
                # Calculate projected total cost based on current average
                evaluated = self.state.get("evaluated", 0)
                total_examples = self.state.get("total_examples", 0)
                if total_examples > 0 and evaluated > 0:
                    # Use actual accumulated cost for better accuracy
                    current_total = self.state.get("total_cost", 0.0)
                    avg_per_example = current_total / evaluated
                    projected_total = avg_per_example * total_examples
                    # Use projected total as the reported value for stronger pruning signal
                    float_value = projected_total
                    logger.debug(
                        "Projecting cost for trial %s: current_total=%.6f, avg=%.6f, projected=%.6f",
                        self.trial_id,
                        current_total,
                        avg_per_example,
                        projected_total,
                    )

            values.append(float_value)

            if objective in {"accuracy", "success_rate"}:
                optimistic = self._optimistic_accuracy()
                optimistic_values.append(optimistic)
            else:
                optimistic_values.append(float_value)

        return values, optimistic_values

    def _extract_accuracy(
        self,
        step_index: int,
        payload: dict[str, Any],
        metrics_payload: dict[str, Any],
    ) -> float | None:
        """Extract accuracy value from payload.

        Accuracy should be based on correctness (output matches expected), not just
        successful execution. Only fall back to exact match comparison if accuracy
        metric is not provided.

        Args:
            step_index: Index of the example being evaluated
            payload: Evaluation payload containing output
            metrics_payload: Metrics dictionary from evaluator

        Returns:
            Accuracy value (1.0 or 0.0) if determinable, None otherwise
        """
        # First, check if evaluator already computed accuracy
        accuracy_value = metrics_payload.get("accuracy")
        if accuracy_value is not None:
            return float(accuracy_value)

        # Second, try exact match comparison if expected_output available
        expected_output = self._get_expected_output(step_index)
        if expected_output is not None:
            return 1.0 if payload.get("output") == expected_output else 0.0

        # If no accuracy metric and no expected_output, we cannot determine accuracy
        # Do NOT fall back to success field - that only indicates execution without error
        return None

    def _extract_cost(
        self,
        payload: dict[str, Any],
        metrics_payload: dict[str, Any],
    ) -> float | None:
        """Extract cost value from payload."""
        cost_value = None
        for cost_key in ("total_cost", "cost", "example_cost"):
            metric_value = metrics_payload.get(cost_key)
            if metric_value is not None:
                cost_value = metric_value
                break

        if cost_value is None:
            cost_value = payload.get("total_cost") or payload.get("cost")

        if cost_value is None:
            return None

        try:
            return float(cost_value)
        except (TypeError, ValueError):
            logger.debug(
                "Unable to accumulate cost metric %s for trial %s",
                cost_value,
                self.trial_id,
            )
        return None

    def _fallback_objective_value(self, objective: str) -> float | None:
        """Calculate fallback value for objective from state."""
        if objective in {"accuracy", "success_rate"}:
            evaluated = cast(int, self.state.get("evaluated", 0))
            if evaluated:
                correct_sum = cast(float, self.state.get("correct_sum", 0.0))
                return correct_sum / evaluated
            return None

        if objective in {"cost", "latency", "error"}:
            return cast(float | None, self.state.get("total_cost"))

        return None

    def _should_prune_for_band(self, value: float | list[float]) -> bool:
        """Check if value is far enough outside the band to trigger pruning.

        For banded objectives, we prune trials that produce values far outside
        the target band. This helps focus the search on configurations that
        can realistically achieve the band target.

        Args:
            value: Current objective value(s) from the trial

        Returns:
            True if the trial should be pruned based on band deviation
        """
        if self.band_target is None:
            return False

        # Extract single value for band check
        if isinstance(value, list):
            if not value:
                return False
            check_value = value[0]  # Use first objective value
        else:
            check_value = value

        # Only prune after a few evaluations to allow for noisy starts
        evaluated = cast(int, self.state.get("evaluated", 0))
        if evaluated < 3:
            return False

        # Calculate band width and deviation threshold
        band = self.band_target
        if band.low is not None and band.high is not None:
            band_width = band.high - band.low
            band_center = (band.low + band.high) / 2.0
        elif band.center is not None and band.tol is not None:
            band_width = 2 * band.tol
            band_center = band.center
        else:
            return False

        # Compute distance from band center
        distance_from_center = abs(check_value - band_center)

        # Prune if value is more than BAND_PRUNE_THRESHOLD * band_width from center
        prune_threshold = self.BAND_PRUNE_THRESHOLD * band_width
        if distance_from_center > prune_threshold:
            logger.debug(
                "Band pruning: value=%.4f, center=%.4f, distance=%.4f, threshold=%.4f",
                check_value,
                band_center,
                distance_from_center,
                prune_threshold,
            )
            return True

        return False

    def _optimistic_accuracy(self) -> float:
        """Calculate optimistic accuracy assuming remaining examples are correct."""
        total_examples = max(cast(int, self.state.get("total_examples", 0)), 1)
        evaluated = cast(int, self.state.get("evaluated", 0))
        remaining = max(total_examples - evaluated, 0)
        correct_sum_val = cast(float, self.state.get("correct_sum", 0.0))
        return (correct_sum_val + remaining) / total_examples

    def _log_progress(self, step_index: int, report_value: float | list[float]) -> None:
        """Log pruning progress."""
        logger.info(
            "📊 Trial %s Step %d: evaluated=%d/%d, value=%s",
            self.optuna_trial_id,
            step_index,
            self.state.get("evaluated", 0),
            self.state.get("total_examples", 0),
            report_value,
        )

    def _get_expected_output(self, step_index: int) -> Any:
        """Get expected output for comparison."""
        try:
            if hasattr(self.dataset, "examples"):
                examples = self.dataset.examples
                if 0 <= step_index < len(examples):
                    return examples[step_index].expected_output
        except Exception as e:
            logger.debug(f"Could not get expected output for step {step_index}: {e}")
            return None
        return None
