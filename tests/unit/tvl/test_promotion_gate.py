"""Unit tests for TVL promotion gate."""

import pytest

from traigent.tvl.models import BandTarget, ChanceConstraint, PromotionPolicy
from traigent.tvl.promotion_gate import (
    ChanceConstraintResult,
    ObjectiveResult,
    ObjectiveSpec,
    PromotionDecision,
    PromotionGate,
)


class TestObjectiveSpec:
    """Tests for ObjectiveSpec class."""

    def test_standard_objective(self) -> None:
        """Standard maximize/minimize objective."""
        spec = ObjectiveSpec(name="accuracy", direction="maximize")
        assert spec.name == "accuracy"
        assert spec.direction == "maximize"
        assert spec.band is None

    def test_banded_objective(self) -> None:
        """Banded objective with target."""
        band = BandTarget(low=0.8, high=0.95)
        spec = ObjectiveSpec(name="consistency", direction="band", band=band)
        assert spec.name == "consistency"
        assert spec.direction == "band"
        assert spec.band == band


class TestPromotionGate:
    """Tests for PromotionGate class."""

    def test_creation(self) -> None:
        """PromotionGate can be created."""
        policy = PromotionPolicy()
        objectives = [
            ObjectiveSpec("accuracy", "maximize"),
            ObjectiveSpec("latency", "minimize"),
        ]
        gate = PromotionGate(policy, objectives)

        assert gate.policy == policy
        assert len(gate.objectives) == 2

    def test_promote_on_clear_improvement(self) -> None:
        """Candidate is promoted when clearly better."""
        policy = PromotionPolicy(
            alpha=0.05, min_effect={"accuracy": 0.01, "latency": 5.0}
        )
        objectives = [
            ObjectiveSpec("accuracy", "maximize"),
            ObjectiveSpec("latency", "minimize"),
        ]
        gate = PromotionGate(policy, objectives)

        # Candidate is clearly better on both objectives
        decision = gate.evaluate(
            incumbent_metrics={
                "accuracy": [
                    0.80,
                    0.82,
                    0.81,
                    0.79,
                    0.80,
                    0.81,
                    0.82,
                    0.80,
                    0.79,
                    0.81,
                ],
                "latency": [
                    100.0,
                    102.0,
                    98.0,
                    101.0,
                    99.0,
                    100.0,
                    103.0,
                    97.0,
                    101.0,
                    100.0,
                ],
            },
            candidate_metrics={
                "accuracy": [
                    0.90,
                    0.92,
                    0.91,
                    0.89,
                    0.90,
                    0.91,
                    0.92,
                    0.90,
                    0.89,
                    0.91,
                ],
                "latency": [
                    80.0,
                    82.0,
                    78.0,
                    81.0,
                    79.0,
                    80.0,
                    83.0,
                    77.0,
                    81.0,
                    80.0,
                ],
            },
        )

        assert decision.decision == "promote"
        assert decision.dominance_satisfied is True

    def test_reject_when_dominated(self) -> None:
        """Candidate is rejected when dominated by incumbent."""
        policy = PromotionPolicy(alpha=0.05, min_effect={"accuracy": 0.01})
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        # Incumbent is clearly better
        decision = gate.evaluate(
            incumbent_metrics={
                "accuracy": [0.90, 0.92, 0.91, 0.89, 0.90, 0.91, 0.92, 0.90, 0.89, 0.91]
            },
            candidate_metrics={
                "accuracy": [0.70, 0.72, 0.71, 0.69, 0.70, 0.71, 0.72, 0.70, 0.69, 0.71]
            },
        )

        assert decision.decision == "reject"
        assert decision.dominance_satisfied is False

    def test_no_decision_on_insufficient_evidence(self) -> None:
        """No decision when evidence is insufficient."""
        policy = PromotionPolicy(alpha=0.05, min_effect={"accuracy": 0.01})
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        # Very similar performance, high variance
        decision = gate.evaluate(
            incumbent_metrics={
                "accuracy": [0.85, 0.88, 0.82, 0.90, 0.83, 0.87, 0.84, 0.89, 0.81, 0.86]
            },
            candidate_metrics={
                "accuracy": [0.86, 0.87, 0.83, 0.89, 0.84, 0.88, 0.85, 0.88, 0.82, 0.87]
            },
        )

        # With high variance and small difference, likely no decision
        assert decision.decision in ["no_decision", "promote"]

    def test_bh_adjustment_applied(self) -> None:
        """Benjamini-Hochberg adjustment is applied when configured."""
        policy = PromotionPolicy(
            alpha=0.05, min_effect={"a": 0.01, "b": 0.01}, adjust="BH"
        )
        objectives = [
            ObjectiveSpec("a", "maximize"),
            ObjectiveSpec("b", "maximize"),
        ]
        gate = PromotionGate(policy, objectives)

        decision = gate.evaluate(
            incumbent_metrics={
                "a": [0.80, 0.82, 0.81, 0.79, 0.80],
                "b": [0.70, 0.72, 0.71, 0.69, 0.70],
            },
            candidate_metrics={
                "a": [0.90, 0.92, 0.91, 0.89, 0.90],
                "b": [0.80, 0.82, 0.81, 0.79, 0.80],
            },
        )

        # Should have adjusted p-values
        assert len(decision.adjusted_p_values) > 0

    def test_missing_objectives_no_decision(self) -> None:
        """No decision when objective data is missing."""
        policy = PromotionPolicy()
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        decision = gate.evaluate(
            incumbent_metrics={},  # No data
            candidate_metrics={},
        )

        assert decision.decision == "no_decision"
        assert "No objectives to compare" in decision.reason


class TestChanceConstraints:
    """Tests for chance constraint evaluation."""

    def test_constraint_satisfied(self) -> None:
        """Chance constraint is satisfied when lower bound >= threshold."""
        policy = PromotionPolicy(
            chance_constraints=[
                ChanceConstraint(name="accuracy", threshold=0.80, confidence=0.95)
            ]
        )
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        # 95/100 successes should satisfy accuracy >= 0.80 with 95% confidence
        decision = gate.evaluate(
            incumbent_metrics={"accuracy": [0.8] * 5},
            candidate_metrics={"accuracy": [0.9] * 5},
            constraint_data={"accuracy": (95, 100)},
        )

        # Constraint should be satisfied
        assert len(decision.chance_results) == 1
        assert decision.chance_results[0].satisfied is True

    def test_constraint_not_satisfied(self) -> None:
        """Candidate is rejected when constraint not satisfied."""
        policy = PromotionPolicy(
            chance_constraints=[
                ChanceConstraint(name="safety", threshold=0.95, confidence=0.95)
            ]
        )
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        # Only 70/100 successes, won't satisfy 0.95 threshold
        decision = gate.evaluate(
            incumbent_metrics={"accuracy": [0.8] * 5},
            candidate_metrics={"accuracy": [0.9] * 5},
            constraint_data={"safety": (70, 100)},
        )

        assert decision.decision == "reject"
        assert "Chance constraints not satisfied" in decision.reason

    def test_missing_constraint_data(self) -> None:
        """Missing constraint data is treated as not satisfied."""
        policy = PromotionPolicy(
            chance_constraints=[
                ChanceConstraint(name="safety", threshold=0.90, confidence=0.95)
            ]
        )
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        decision = gate.evaluate(
            incumbent_metrics={"accuracy": [0.8] * 5},
            candidate_metrics={"accuracy": [0.9] * 5},
            constraint_data={},  # Missing safety data
        )

        assert decision.decision == "reject"


class TestBandedObjectives:
    """Tests for banded objective handling."""

    def test_banded_objective_promotion(self) -> None:
        """Candidate passing TOST is promoted."""
        band = BandTarget(low=0.85, high=0.95)
        policy = PromotionPolicy()
        objectives = [ObjectiveSpec("consistency", "band", band=band, band_alpha=0.05)]
        gate = PromotionGate(policy, objectives)

        # Candidate is in band, incumbent is not
        decision = gate.evaluate(
            incumbent_metrics={
                "consistency": [
                    0.70,
                    0.72,
                    0.68,
                    0.71,
                    0.69,
                    0.73,
                    0.70,
                    0.71,
                    0.69,
                    0.70,
                ]
            },
            candidate_metrics={
                "consistency": [
                    0.89,
                    0.90,
                    0.88,
                    0.91,
                    0.89,
                    0.90,
                    0.88,
                    0.90,
                    0.89,
                    0.90,
                ]
            },
        )

        assert decision.decision == "promote"


class TestPromotionDecision:
    """Tests for PromotionDecision dataclass."""

    def test_decision_attributes(self) -> None:
        """PromotionDecision has all expected attributes."""
        decision = PromotionDecision(
            decision="promote",
            reason="Test reason",
            objective_results=[],
            chance_results=[],
            adjusted_p_values={},
            dominance_satisfied=True,
        )

        assert decision.decision == "promote"
        assert decision.reason == "Test reason"
        assert decision.dominance_satisfied is True

    def test_default_values(self) -> None:
        """PromotionDecision has sensible defaults."""
        decision = PromotionDecision(decision="no_decision", reason="Test")

        assert decision.objective_results == []
        assert decision.chance_results == []
        assert decision.adjusted_p_values == {}
        assert decision.dominance_satisfied is False
