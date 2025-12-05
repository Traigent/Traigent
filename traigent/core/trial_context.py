"""Trial execution context."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE REQ-ORCH-003 SYNC-OptimizationFlow

from dataclasses import dataclass
from typing import Any, Callable

from traigent.evaluators.base import Dataset


@dataclass
class TrialRunContext:
    """Context for individual trial execution.

    Bundles trial execution parameters to reduce method parameter counts
    and provide clear ownership of trial state.

    Attributes:
        func: Function to evaluate
        config: Configuration to test
        dataset: Evaluation dataset
        trial_number: Trial number for tracking
        session_id: Backend session identifier (None if backend disabled)
        optuna_trial_id: Optuna trial identifier for pruning (optional)
    """

    func: Callable[..., Any]
    config: dict[str, Any]
    dataset: Dataset
    trial_number: int
    session_id: str | None
    optuna_trial_id: int | None = None
