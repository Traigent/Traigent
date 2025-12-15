"""Promotion gate for epsilon-Pareto dominance evaluation.

This module implements the promotion policy logic from TVL 0.9,
including epsilon-Pareto dominance testing, chance constraint
evaluation, and multi-objective comparison.

Concept: CONC-TVLSpec
Implements: FUNC-TVLSPEC
Sync: SYNC-OptimizationFlow
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .models import BandTarget, PromotionPolicy
from .objectives import tost_equivalence_test
from .statistics import (
    benjamini_hochberg_adjust,
    clopper_pearson_lower_bound,
    paired_comparison_test,
)


@dataclass(slots=True)
class ObjectiveResult:
    """Result of comparing a single objective.

    Attributes:
        name: Name of the objective.
        direction: Optimization direction or "band".
        candidate_better: Whether candidate is better (considering epsilon).
        p_value: P-value of the comparison test.
        effect_size: Estimated effect size.
        candidate_mean: Mean value for candidate.
        incumbent_mean: Mean value for incumbent.
        epsilon: Epsilon tolerance applied.
    """

    name: str
    direction: Literal["maximize", "minimize", "band"]
    candidate_better: bool
    p_value: float
    effect_size: float
    candidate_mean: float
    incumbent_mean: float
    epsilon: float


@dataclass(slots=True)
class ChanceConstraintResult:
    """Result of evaluating a chance constraint.

    Attributes:
        name: Name of the constrained metric.
        satisfied: Whether the constraint is satisfied.
        observed_rate: Observed success rate.
        lower_bound: Lower confidence bound for the rate.
        threshold: Required threshold.
        confidence: Required confidence level.
    """

    name: str
    satisfied: bool
    observed_rate: float
    lower_bound: float
    threshold: float
    confidence: float


@dataclass(slots=True)
class PromotionDecision:
    """Result of the promotion gate evaluation.

    Attributes:
        decision: The promotion decision ("promote", "reject", "no_decision").
        reason: Human-readable explanation of the decision.
        objective_results: Per-objective comparison results.
        chance_results: Per-constraint evaluation results.
        adjusted_p_values: P-values after BH adjustment (if applied).
        dominance_satisfied: Whether epsilon-Pareto dominance holds.
    """

    decision: Literal["promote", "reject", "no_decision"]
    reason: str
    objective_results: list[ObjectiveResult] = field(default_factory=list)
    chance_results: list[ChanceConstraintResult] = field(default_factory=list)
    adjusted_p_values: dict[str, float] = field(default_factory=dict)
    dominance_satisfied: bool = False


@dataclass
class ObjectiveSpec:
    """Specification for an objective in the promotion gate.

    Attributes:
        name: Name of the objective.
        direction: Optimization direction ("maximize", "minimize", or "band").
        band: Band target for banded objectives (None for standard).
        band_alpha: Significance level for TOST (banded objectives only).
    """

    name: str
    direction: Literal["maximize", "minimize", "band"]
    band: BandTarget | None = None
    band_alpha: float = 0.05


class PromotionGate:
    """Promotion gate implementing epsilon-Pareto dominance.

    The promotion gate evaluates whether a candidate configuration should
    be promoted over the incumbent based on statistical testing with
    configurable error rates and effect sizes.

    Example:
        ```python
        policy = PromotionPolicy(
            dominance="epsilon_pareto",
            alpha=0.05,
            min_effect={"accuracy": 0.01, "latency": 5.0},
            adjust="BH",
        )
        objectives = [
            ObjectiveSpec("accuracy", "maximize"),
            ObjectiveSpec("latency", "minimize"),
        ]
        gate = PromotionGate(policy, objectives)

        decision = gate.evaluate(
            incumbent_metrics={"accuracy": [0.85, 0.87], "latency": [100, 95]},
            candidate_metrics={"accuracy": [0.90, 0.88], "latency": [90, 92]},
        )
        print(f"Decision: {decision.decision} - {decision.reason}")
        ```
    """

    def __init__(
        self,
        policy: PromotionPolicy,
        objectives: list[ObjectiveSpec],
    ) -> None:
        """Initialize the promotion gate.

        Args:
            policy: Promotion policy configuration.
            objectives: List of objective specifications.
        """
        self.policy = policy
        self.objectives = {obj.name: obj for obj in objectives}
        self._objective_list = objectives

    def evaluate(
        self,
        incumbent_metrics: dict[str, Sequence[float]],
        candidate_metrics: dict[str, Sequence[float]],
        constraint_data: dict[str, tuple[int, int]] | None = None,
    ) -> PromotionDecision:
        """Evaluate candidate for promotion against incumbent.

        Args:
            incumbent_metrics: Metric samples for incumbent.
                Keys are objective names, values are sequences of samples.
            candidate_metrics: Metric samples for candidate.
                Keys are objective names, values are sequences of samples.
            constraint_data: Optional data for chance constraints.
                Keys are constraint names, values are (successes, trials).

        Returns:
            PromotionDecision with the evaluation result.
        """
        # First, check chance constraints
        chance_results = self._evaluate_chance_constraints(constraint_data or {})
        constraints_satisfied = all(r.satisfied for r in chance_results)

        if not constraints_satisfied:
            failed_names = [r.name for r in chance_results if not r.satisfied]
            return PromotionDecision(
                decision="reject",
                reason=f"Chance constraints not satisfied: {', '.join(failed_names)}",
                chance_results=chance_results,
                dominance_satisfied=False,
            )

        # Evaluate each objective
        objective_results = []
        raw_p_values: dict[str, float] = {}

        for obj in self._objective_list:
            if obj.name not in incumbent_metrics or obj.name not in candidate_metrics:
                continue

            inc_samples = incumbent_metrics[obj.name]
            cand_samples = candidate_metrics[obj.name]

            result = self._compare_objective(obj, inc_samples, cand_samples)
            objective_results.append(result)
            raw_p_values[obj.name] = result.p_value

        if not objective_results:
            return PromotionDecision(
                decision="no_decision",
                reason="No objectives to compare",
                objective_results=[],
                chance_results=chance_results,
                dominance_satisfied=False,
            )

        # Apply multiple testing adjustment if configured
        if self.policy.adjust == "BH" and len(raw_p_values) > 1:
            p_list = list(raw_p_values.values())
            adjusted_list = benjamini_hochberg_adjust(p_list)
            adjusted_p_values = dict(
                zip(raw_p_values.keys(), adjusted_list, strict=True)
            )
        else:
            adjusted_p_values = raw_p_values.copy()

        # Check epsilon-Pareto dominance
        # Candidate dominates if it's significantly better (p < alpha) on at least one
        # objective and not significantly worse on any.
        alpha = self.policy.alpha

        # Update results with adjusted p-values and check dominance
        any_better = False
        any_worse = False

        for result in objective_results:
            adj_p = adjusted_p_values.get(result.name, result.p_value)

            if result.candidate_better:
                if adj_p < alpha:
                    any_better = True
            else:
                # Check if incumbent is significantly better
                # For this, we'd need to test the reverse hypothesis
                # Simplified: if candidate is not better and effect size shows
                # incumbent is better by more than epsilon, consider it worse
                if result.effect_size < -result.epsilon:
                    any_worse = True

        dominance_satisfied = any_better and not any_worse

        # Make decision
        if dominance_satisfied:
            return PromotionDecision(
                decision="promote",
                reason="Candidate satisfies epsilon-Pareto dominance",
                objective_results=objective_results,
                chance_results=chance_results,
                adjusted_p_values=adjusted_p_values,
                dominance_satisfied=True,
            )
        elif any_worse:
            return PromotionDecision(
                decision="reject",
                reason="Candidate is dominated by incumbent on some objectives",
                objective_results=objective_results,
                chance_results=chance_results,
                adjusted_p_values=adjusted_p_values,
                dominance_satisfied=False,
            )
        else:
            return PromotionDecision(
                decision="no_decision",
                reason="Insufficient evidence for dominance",
                objective_results=objective_results,
                chance_results=chance_results,
                adjusted_p_values=adjusted_p_values,
                dominance_satisfied=False,
            )

    def _compare_objective(
        self,
        obj: ObjectiveSpec,
        incumbent_samples: Sequence[float],
        candidate_samples: Sequence[float],
    ) -> ObjectiveResult:
        """Compare a single objective between incumbent and candidate.

        Args:
            obj: Objective specification.
            incumbent_samples: Samples for incumbent.
            candidate_samples: Samples for candidate.

        Returns:
            ObjectiveResult with comparison outcome.
        """
        epsilon = self.policy.get_epsilon(obj.name, 0.0)

        # Compute means
        inc_mean = sum(incumbent_samples) / len(incumbent_samples)
        cand_mean = sum(candidate_samples) / len(candidate_samples)

        if obj.direction == "band" and obj.band is not None:
            # For banded objectives, use TOST
            result = self._compare_banded(
                obj, incumbent_samples, candidate_samples, epsilon
            )
            return result

        # For standard objectives, use paired comparison test
        # Use policy.alpha for rejection decision, not hardcoded threshold
        if obj.direction == "maximize":
            # Test if candidate > incumbent + epsilon
            comparison = paired_comparison_test(
                list(candidate_samples),
                list(incumbent_samples),
                epsilon,
                "greater",
            )
            # Apply policy alpha for rejection decision
            candidate_better = comparison.p_value < self.policy.alpha
            effect_size = cand_mean - inc_mean
        else:
            # minimize: Test if candidate < incumbent - epsilon
            comparison = paired_comparison_test(
                list(candidate_samples),
                list(incumbent_samples),
                epsilon,
                "less",
            )
            # Apply policy alpha for rejection decision
            candidate_better = comparison.p_value < self.policy.alpha
            effect_size = inc_mean - cand_mean  # Positive if candidate is better

        return ObjectiveResult(
            name=obj.name,
            direction=obj.direction,
            candidate_better=candidate_better,
            p_value=comparison.p_value,
            effect_size=effect_size,
            candidate_mean=cand_mean,
            incumbent_mean=inc_mean,
            epsilon=epsilon,
        )

    def _compare_banded(
        self,
        obj: ObjectiveSpec,
        incumbent_samples: Sequence[float],
        candidate_samples: Sequence[float],
        epsilon: float,
    ) -> ObjectiveResult:
        """Compare banded objectives using TOST.

        For banded objectives, the goal is to be within the target band.
        The candidate is better if it passes TOST and incumbent doesn't,
        or if both pass but candidate is closer to band center.

        Args:
            obj: Objective specification (must have band).
            incumbent_samples: Samples for incumbent.
            candidate_samples: Samples for candidate.
            epsilon: Epsilon tolerance (applied to band width).

        Returns:
            ObjectiveResult for the banded comparison.
        """
        assert obj.band is not None

        # Perform TOST on both
        inc_tost = tost_equivalence_test(incumbent_samples, obj.band, obj.band_alpha)
        cand_tost = tost_equivalence_test(candidate_samples, obj.band, obj.band_alpha)

        inc_mean = inc_tost.sample_mean
        cand_mean = cand_tost.sample_mean

        # Determine which is better
        if cand_tost.is_equivalent and not inc_tost.is_equivalent:
            candidate_better = True
            # Use the max of the two TOST p-values as our p-value
            p_value = max(cand_tost.p_lower, cand_tost.p_upper)
        elif not cand_tost.is_equivalent and inc_tost.is_equivalent:
            candidate_better = False
            p_value = 1.0  # Candidate fails TOST
        else:
            # Both equivalent or both not - compare by deviation from center
            if obj.band.low is not None and obj.band.high is not None:
                center = (obj.band.low + obj.band.high) / 2
                cand_dist = abs(cand_mean - center)
                inc_dist = abs(inc_mean - center)

                candidate_better = cand_dist < inc_dist - epsilon
                # Approximate p-value based on relative distance
                p_value = 0.5 * (1 + (cand_dist - inc_dist) / max(inc_dist, 1e-10))
                p_value = max(0.0, min(1.0, p_value))
            else:
                candidate_better = False
                p_value = 0.5

        # Effect size is how much closer candidate is to band center
        if obj.band.low is not None and obj.band.high is not None:
            center = (obj.band.low + obj.band.high) / 2
            effect_size = abs(inc_mean - center) - abs(cand_mean - center)
        else:
            effect_size = 0.0

        return ObjectiveResult(
            name=obj.name,
            direction="band",
            candidate_better=candidate_better,
            p_value=p_value,
            effect_size=effect_size,
            candidate_mean=cand_mean,
            incumbent_mean=inc_mean,
            epsilon=epsilon,
        )

    def _evaluate_chance_constraints(
        self,
        constraint_data: dict[str, tuple[int, int]],
    ) -> list[ChanceConstraintResult]:
        """Evaluate chance constraints.

        Args:
            constraint_data: Dict mapping constraint name to (successes, trials).

        Returns:
            List of ChanceConstraintResult.
        """
        results = []

        for constraint in self.policy.chance_constraints:
            if constraint.name not in constraint_data:
                # Missing data - treat as not satisfied
                results.append(
                    ChanceConstraintResult(
                        name=constraint.name,
                        satisfied=False,
                        observed_rate=0.0,
                        lower_bound=0.0,
                        threshold=constraint.threshold,
                        confidence=constraint.confidence,
                    )
                )
                continue

            successes, trials = constraint_data[constraint.name]

            if trials == 0:
                results.append(
                    ChanceConstraintResult(
                        name=constraint.name,
                        satisfied=False,
                        observed_rate=0.0,
                        lower_bound=0.0,
                        threshold=constraint.threshold,
                        confidence=constraint.confidence,
                    )
                )
                continue

            observed_rate = successes / trials
            lower_bound = clopper_pearson_lower_bound(
                successes, trials, constraint.confidence
            )

            # Constraint is satisfied if lower bound >= threshold
            satisfied = lower_bound >= constraint.threshold

            results.append(
                ChanceConstraintResult(
                    name=constraint.name,
                    satisfied=satisfied,
                    observed_rate=observed_rate,
                    lower_bound=lower_bound,
                    threshold=constraint.threshold,
                    confidence=constraint.confidence,
                )
            )

        return results

    @classmethod
    def from_spec_artifact(
        cls,
        artifact: Any,  # TVLSpecArtifact - avoid circular import
    ) -> PromotionGate | None:
        """Create a PromotionGate from a TVLSpecArtifact.

        Args:
            artifact: The loaded TVL spec artifact.

        Returns:
            PromotionGate if promotion policy is defined, None otherwise.
        """
        if artifact.promotion_policy is None:
            return None

        # Build objective specs from the artifact
        objectives: list[ObjectiveSpec] = []

        for obj_data in artifact.objectives:
            name = obj_data.get("name", "")
            direction = obj_data.get("direction", "maximize")

            band = None
            band_alpha = 0.05

            if "band" in obj_data:
                band_data = obj_data["band"]
                target = band_data.get("target")
                if target:
                    if isinstance(target, list) and len(target) == 2:
                        band = BandTarget(low=target[0], high=target[1])
                    elif isinstance(target, dict):
                        band = BandTarget.from_dict(target)
                band_alpha = float(band_data.get("alpha", 0.05))
                direction = "band"

            objectives.append(
                ObjectiveSpec(
                    name=name,
                    direction=direction,
                    band=band,
                    band_alpha=band_alpha,
                )
            )

        return cls(artifact.promotion_policy, objectives)
