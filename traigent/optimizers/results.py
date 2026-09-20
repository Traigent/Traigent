"""Optimization result types for batch optimizers.

Historically this module defined ``Trial`` and ``OptimizationResult`` classes
that happened to share their names with the unrelated, much richer public
types in ``traigent.api.types`` (same name, structurally divergent shape —
e.g. ``successful_trials`` returned a ``list[TrialResult]`` on the public
type but an ``int`` count here). That same-name collision is issue #1393,
Smell 1. The batch-optimizer types are renamed to ``BatchTrial`` /
``BatchResult`` to remove the collision; ``Trial`` / ``OptimizationResult``
remain as deprecated aliases below, needed because
``traigent/utils/persistence.py`` allowlists the dotted names
``traigent.optimizers.results.OptimizationResult`` / ``.Trial`` for
unpickling legacy trial artifacts (``RestrictedUnpickler.find_class`` does a
plain ``getattr(module, name)``), so removing the names outright would break
restoring old pickles. New code should import ``BatchTrial`` / ``BatchResult``.
"""

# Traceability: CONC-Layer-Data CONC-Quality-Maintainability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from traigent.api.types import StopReason


@dataclass
class BatchTrial:
    """Single batch-optimizer trial result."""

    configuration: dict[str, Any]
    score: float
    duration: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        """Check if trial was successful."""
        return self.score != float("-inf") and not self.metadata.get("failed", False)


@dataclass
class BatchResult:
    """Result of a standalone batch-optimizer run.

    Not to be confused with the public ``traigent.api.types.OptimizationResult``
    (issue #1393): this minimal shape is produced only by the batch
    optimizers' own standalone ``.optimize()`` and is not reachable from the
    public ``@traigent.optimize`` / ``OptimizedFunction.optimize()`` path,
    which always returns the rich public type.
    """

    best_config: dict[str, Any]
    best_score: float
    trials: list[BatchTrial]
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


# Deprecated aliases kept only so `traigent.optimizers.results.OptimizationResult`
# / `.Trial` keep resolving for the pickle allowlist in
# `traigent/utils/persistence.py` (see module docstring above). Do not use in
# new code; import `BatchResult` / `BatchTrial` instead.
OptimizationResult = BatchResult
Trial = BatchTrial
