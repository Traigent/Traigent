"""Collection utilities for mandatory optimization metrics."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from traigent.api.types import TrialResult, TrialStatus
from traigent.core.utils import extract_examples_attempted
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MandatoryMetricsTotals:
    """Aggregated totals for mandatory metrics."""

    total_cost: float = 0.0
    total_tokens: int = 0
    total_duration: float = 0.0
    total_examples_attempted: int = 0

    def as_metrics_dict(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.total_cost > 0:
            metrics["total_cost"] = self.total_cost
        if self.total_tokens > 0:
            metrics["total_tokens"] = float(self.total_tokens)
        if self.total_duration > 0:
            metrics["total_duration"] = self.total_duration
        if self.total_examples_attempted > 0:
            metrics["examples_attempted_total"] = float(self.total_examples_attempted)
        return metrics


class MandatoryMetricsCollector:
    """Collector that accumulates mandatory metrics across trials."""

    def __init__(self) -> None:
        self._totals = MandatoryMetricsTotals()

    def accumulate(self, trial: TrialResult) -> None:
        if trial.status != TrialStatus.COMPLETED:
            return

        try:
            self._totals.total_duration += float(trial.duration or 0.0)
        except (TypeError, ValueError):
            logger.debug("Unable to parse duration for trial %s", trial.trial_id)

        self._totals.total_examples_attempted += self._extract_examples_attempted(trial)

        cost_increment, token_increment = self._extract_cost_and_tokens(trial)
        self._totals.total_cost += cost_increment
        self._totals.total_tokens += token_increment

    def totals(self) -> MandatoryMetricsTotals:
        return self._totals

    def _extract_examples_attempted(self, trial: TrialResult) -> int:
        return (
            extract_examples_attempted(trial, default=0, check_example_results=False)
            or 0
        )

    def _extract_cost_and_tokens(self, trial: TrialResult) -> tuple[float, int]:
        metrics = trial.metrics or {}
        cost_increment = self._safe_float(metrics.get("total_cost"))
        token_increment = self._safe_tokens(metrics.get("total_tokens"), trial)

        eval_result = (trial.metadata or {}).get("evaluation_result")
        if eval_result is not None:
            aggregated_metrics = getattr(eval_result, "aggregated_metrics", None) or {}
            if cost_increment <= 0:
                cost_increment = self._from_aggregated_cost(
                    aggregated_metrics, eval_result, trial
                )
            if token_increment <= 0:
                token_increment = self._from_aggregated_tokens(
                    aggregated_metrics, eval_result, trial
                )

        return cost_increment, token_increment

    def _from_aggregated_cost(
        self, aggregated: dict[str, Any], eval_result: Any, trial: TrialResult
    ) -> float:
        cost_info = aggregated.get("total_cost", {})
        if not isinstance(cost_info, dict):
            return 0.0
        trial_cost = cost_info.get("mean", 0.0)
        if trial_cost is None or trial_cost <= 0:
            return 0.0
        try:
            total_examples = int(getattr(eval_result, "total_examples", 0))
            return float(trial_cost) * total_examples
        except (TypeError, ValueError):
            logger.debug(
                "Unable to accumulate aggregated cost for trial %s", trial.trial_id
            )
            return 0.0

    def _from_aggregated_tokens(
        self, aggregated: dict[str, Any], eval_result: Any, trial: TrialResult
    ) -> int:
        token_info = aggregated.get("total_tokens", {})
        if not isinstance(token_info, dict):
            return 0
        trial_tokens = token_info.get("mean", 0.0)
        if trial_tokens is None or trial_tokens <= 0:
            return 0
        try:
            total_examples = int(getattr(eval_result, "total_examples", 0))
            return int(float(trial_tokens) * total_examples)
        except (TypeError, ValueError):
            logger.debug(
                "Unable to accumulate aggregated tokens for trial %s", trial.trial_id
            )
            return 0

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            if value is None:
                return 0.0
            value = float(value)
            return value if value > 0 else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_tokens(value: Any, trial: TrialResult) -> int:
        try:
            if value is None:
                return 0
            float_val = float(value)
            return int(float_val) if float_val > 0 else 0
        except (TypeError, ValueError):
            logger.debug("Unable to parse trial-level tokens for %s", trial.trial_id)
            return 0


__all__ = ["MandatoryMetricsCollector", "MandatoryMetricsTotals"]
