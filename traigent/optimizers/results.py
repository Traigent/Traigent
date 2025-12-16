"""Optimization result types for batch optimizers."""

# Traceability: CONC-Layer-Data CONC-Quality-Maintainability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.api.types import StopReason


@dataclass
class Trial:
    """Single optimization trial result."""

    configuration: dict[str, Any]
    score: float
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Check if trial was successful."""
        return self.score != float("-inf") and not self.metadata.get("failed", False)


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    best_config: dict[str, Any]
    best_score: float
    trials: list[Trial]
    duration: float
    convergence_info: dict[str, Any] = field(default_factory=dict)
    stop_reason: StopReason | None = None

    @property
    def total_trials(self) -> int:
        """Get total number of trials."""
        return len(self.trials)

    @property
    def successful_trials(self) -> int:
        """Get number of successful trials."""
        return sum(1 for trial in self.trials if trial.is_successful)

    @property
    def success_rate(self) -> float:
        """Get trial success rate."""
        if not self.trials:
            return 0.0
        return self.successful_trials / len(self.trials)
