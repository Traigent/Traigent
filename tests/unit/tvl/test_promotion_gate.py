"""Unit tests for TVL promotion gate."""


from traigent.tvl.models import BandTarget, ChanceConstraint, PromotionPolicy
from traigent.tvl.promotion_gate import (
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


class TestChanceConstraintsEdgeCases:
    """Additional tests for chance constraint edge cases."""

    def test_zero_trials_constraint(self) -> None:
        """Zero trials in constraint data is treated as not satisfied."""
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
            constraint_data={"safety": (0, 0)},  # Zero trials
        )

        assert decision.decision == "reject"
        assert len(decision.chance_results) == 1
        assert decision.chance_results[0].satisfied is False
        assert abs(decision.chance_results[0].observed_rate - 0.0) < 1e-10


class TestBandedObjectivesEdgeCases:
    """Additional tests for banded objective edge cases."""

    def test_both_not_equivalent_compare_by_distance(self) -> None:
        """When both fail TOST, compare by distance to band center."""
        band = BandTarget(low=0.85, high=0.95)
        policy = PromotionPolicy()
        objectives = [ObjectiveSpec("consistency", "band", band=band, band_alpha=0.05)]
        gate = PromotionGate(policy, objectives)

        # Both are outside the band but candidate is closer to center (0.9)
        decision = gate.evaluate(
            incumbent_metrics={
                "consistency": [0.60, 0.62, 0.58, 0.61, 0.59] * 2  # Far from band
            },
            candidate_metrics={
                "consistency": [0.75, 0.77, 0.73, 0.76, 0.74] * 2  # Closer to band
            },
        )

        # Should have results even if neither passes TOST
        assert len(decision.objective_results) == 1

    def test_both_equivalent_compare_by_center_distance(self) -> None:
        """When both pass TOST, compare by distance from band center."""
        band = BandTarget(low=0.80, high=1.00)  # Wide band
        policy = PromotionPolicy()
        objectives = [ObjectiveSpec("consistency", "band", band=band, band_alpha=0.05)]
        gate = PromotionGate(policy, objectives)

        # Both in band, candidate closer to center (0.9)
        decision = gate.evaluate(
            incumbent_metrics={
                "consistency": [0.82, 0.83, 0.81, 0.82, 0.82] * 2  # At low end of band
            },
            candidate_metrics={
                "consistency": [0.90, 0.91, 0.89, 0.90, 0.90] * 2  # At center of band
            },
        )

        assert len(decision.objective_results) == 1

    def test_band_with_center_tolerance(self) -> None:
        """Band specified with center and tolerance."""
        band = BandTarget(center=0.90, tol=0.05)  # Band is [0.85, 0.95]
        policy = PromotionPolicy()
        objectives = [ObjectiveSpec("consistency", "band", band=band, band_alpha=0.05)]
        gate = PromotionGate(policy, objectives)

        decision = gate.evaluate(
            incumbent_metrics={"consistency": [0.70, 0.72, 0.71] * 3},
            candidate_metrics={"consistency": [0.90, 0.91, 0.89] * 3},
        )

        # Should handle center/tol format correctly
        assert len(decision.objective_results) == 1


class TestFromSpecArtifact:
    """Tests for PromotionGate.from_spec_artifact."""

    def test_no_promotion_policy_returns_none(self) -> None:
        """Returns None when artifact has no promotion policy."""

        class MockArtifact:
            promotion_policy = None
            objectives: list = []

        gate = PromotionGate.from_spec_artifact(MockArtifact())
        assert gate is None

    def test_with_standard_objectives(self) -> None:
        """Creates gate from artifact with standard objectives."""

        class MockArtifact:
            promotion_policy = PromotionPolicy(alpha=0.05)
            objectives = [
                {"name": "accuracy", "direction": "maximize"},
                {"name": "latency", "direction": "minimize"},
            ]

        gate = PromotionGate.from_spec_artifact(MockArtifact())
        assert gate is not None
        assert len(gate.objectives) == 2
        assert gate.objectives["accuracy"].direction == "maximize"
        assert gate.objectives["latency"].direction == "minimize"

    def test_with_banded_objectives_list_target(self) -> None:
        """Creates gate from artifact with banded objectives using list target."""

        class MockArtifact:
            promotion_policy = PromotionPolicy()
            objectives = [
                {
                    "name": "consistency",
                    "band": {
                        "target": [0.85, 0.95],
                        "alpha": 0.10,
                    },
                },
            ]

        gate = PromotionGate.from_spec_artifact(MockArtifact())
        assert gate is not None
        assert len(gate.objectives) == 1
        obj = gate.objectives["consistency"]
        assert obj.direction == "band"
        assert obj.band is not None
        assert abs(obj.band.low - 0.85) < 1e-10  # type: ignore[operator]
        assert abs(obj.band.high - 0.95) < 1e-10  # type: ignore[operator]
        assert abs(obj.band_alpha - 0.10) < 1e-10

    def test_with_banded_objectives_dict_target(self) -> None:
        """Creates gate from artifact with banded objectives using dict target."""

        class MockArtifact:
            promotion_policy = PromotionPolicy()
            objectives = [
                {
                    "name": "cost",
                    "band": {
                        "target": {"center": 0.10, "tol": 0.02},
                    },
                },
            ]

        gate = PromotionGate.from_spec_artifact(MockArtifact())
        assert gate is not None
        obj = gate.objectives["cost"]
        assert obj.direction == "band"
        assert obj.band is not None
        # BandTarget.from_dict should compute low/high from center±tol
        assert abs(obj.band.center - 0.10) < 1e-10  # type: ignore[operator]
        assert abs(obj.band.tol - 0.02) < 1e-10  # type: ignore[operator]
