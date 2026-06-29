"""Unit tests for TVL promotion gate."""

import pytest

from traigent.tvl.models import BandTarget, ChanceConstraint, PromotionPolicy
from traigent.tvl.promotion_gate import ObjectiveSpec, PromotionDecision, PromotionGate


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

    def test_invalid_direction_raises(self) -> None:
        """Axis D: a direct ObjectiveSpec caller with an invalid direction raises.

        Mirrors the spec-load guard (spec_loader.py:1980-1983). The compile-time
        ``Literal`` does not enforce at runtime, so an unsupported direction must
        fail loudly instead of silently falling into the minimize branch.
        """
        with pytest.raises(ValueError, match="direction must be one of"):
            ObjectiveSpec(name="accuracy", direction="upward")  # type: ignore[arg-type]


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

    def test_holm_adjustment_blocks_raw_p_value_promotion(self) -> None:
        """Holm adjustment prevents promotion that raw p-values would allow.

        Axis B (canonical promotion.py:233-244): the superiority (union)
        component is multiplicity-adjusted while non-inferiority stays
        unadjusted. Two identical, moderately-improved objectives are each
        superior at the raw p-value (p_super=0.0366 < 0.05) but Holm doubles
        the smallest p-value (2 x 0.0366 = 0.0731 >= 0.05), so neither is
        superior under Holm even though both remain non-inferior.
        """
        # Both objectives share the same candidate samples (incumbent = 0).
        # p_noninf == p_super here (epsilon = 0), so both stay non-inferior.
        cand_samples = [-0.45, -0.2, 0.05, 0.3, 0.55, 0.8, 1.05, -0.075, 0.425, 0.55]
        incumbent_metrics = {
            "a": [0.0] * 10,
            "b": [0.0] * 10,
        }
        candidate_metrics = {
            "a": list(cand_samples),
            "b": list(cand_samples),
        }
        objectives = [
            ObjectiveSpec("a", "maximize"),
            ObjectiveSpec("b", "maximize"),
        ]

        raw_gate = PromotionGate(
            PromotionPolicy(alpha=0.05, min_effect={"a": 0.0, "b": 0.0}, adjust="none"),
            objectives,
        )
        raw_decision = raw_gate.evaluate(incumbent_metrics, candidate_metrics)
        assert raw_decision.decision == "promote"
        assert raw_decision.adjusted_p_values["a"] < raw_gate.policy.alpha

        holm_gate = PromotionGate(
            PromotionPolicy(
                alpha=0.05,
                min_effect={"a": 0.0, "b": 0.0},
                adjust="holm",
            ),
            objectives,
        )
        holm_decision = holm_gate.evaluate(incumbent_metrics, candidate_metrics)
        # Non-inferiority is unadjusted, so the candidate is not rejected...
        assert all(
            r.p_value_noninf < holm_gate.policy.alpha
            for r in holm_decision.objective_results
        )
        # ...but the adjusted superiority p-value clears alpha -> no superiority.
        assert holm_decision.adjusted_p_values["a"] > holm_gate.policy.alpha
        assert holm_decision.decision == "no_decision"
        assert holm_decision.dominance_satisfied is False

    def test_iut_rejects_when_objective_not_noninferior(self) -> None:
        """Axis A: canonical IUT rejects when any objective is not non-inferior.

        Mirrors the canonical decision rule (promotion.py:768-773, :249-251):
        a candidate that is clearly superior on Y but cannot prove
        non-inferiority on a neutral objective X is REJECTED. The old SDK gate
        (any_better and not any_worse) PROMOTED this case, because the neutral
        objective X is not "worse beyond epsilon" and Y is better.
        """
        incumbent_metrics = {
            "x": [0.80, 0.82, 0.81, 0.79, 0.80, 0.81, 0.82, 0.80, 0.79, 0.81],
            "y": [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53, 0.47, 0.51, 0.50],
        }
        candidate_metrics = {
            # X: identical to incumbent -> non-inferiority unproven (p >= alpha),
            # yet NOT worse beyond epsilon (so the old gate ignored it).
            "x": list(incumbent_metrics["x"]),
            # Y: clearly better.
            "y": [0.80, 0.82, 0.78, 0.81, 0.79, 0.80, 0.83, 0.77, 0.81, 0.80],
        }
        objectives = [
            ObjectiveSpec("x", "maximize"),
            ObjectiveSpec("y", "maximize"),
        ]
        gate = PromotionGate(
            PromotionPolicy(alpha=0.05, min_effect={"x": 0.0, "y": 0.0}, adjust="none"),
            objectives,
        )

        decision = gate.evaluate(incumbent_metrics, candidate_metrics)

        assert decision.decision == "reject"
        assert decision.dominance_satisfied is False
        verdicts = {r.name: r.verdict for r in decision.objective_results}
        assert verdicts["x"] == "inferior"
        assert verdicts["y"] == "superior"
        x_result = next(r for r in decision.objective_results if r.name == "x")
        assert x_result.p_value_noninf >= gate.policy.alpha

    def test_unsupported_adjust_value_reaching_gate_raises(self) -> None:
        """The gate fails closed if an unsupported adjust value reaches it."""
        policy = PromotionPolicy(alpha=0.05, adjust="none")
        policy.adjust = "sidak"  # type: ignore[assignment]
        gate = PromotionGate(policy, [ObjectiveSpec("accuracy", "maximize")])

        with pytest.raises(ValueError, match="Unsupported promotion_policy.adjust"):
            gate.evaluate(
                incumbent_metrics={"accuracy": [0.0] * 5},
                candidate_metrics={"accuracy": [1.0] * 5},
            )

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
        """Chance constraint passes when the violation-rate upper bound <= threshold.

        Axis C (canonical promotion.py:677-678): constraint_data is
        ``(violations, trials)`` and the constraint is satisfied iff the
        upper confidence bound on the violation rate is at or below the
        threshold. 2/100 violations -> upper bound ~0.06 <= 0.10.
        """
        policy = PromotionPolicy(
            chance_constraints=[
                ChanceConstraint(name="error_rate", threshold=0.10, confidence=0.95)
            ]
        )
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        decision = gate.evaluate(
            incumbent_metrics={"accuracy": [0.8] * 5},
            candidate_metrics={"accuracy": [0.9] * 5},
            constraint_data={"error_rate": (2, 100)},  # (violations, trials)
        )

        assert len(decision.chance_results) == 1
        assert decision.chance_results[0].satisfied is True
        assert decision.chance_results[0].upper_bound <= 0.10

    def test_constraint_not_satisfied(self) -> None:
        """Candidate is rejected when the violation-rate upper bound > threshold.

        Axis C: 30/100 violations -> upper bound ~0.38 > 0.10 threshold.
        """
        policy = PromotionPolicy(
            chance_constraints=[
                ChanceConstraint(name="error_rate", threshold=0.10, confidence=0.95)
            ]
        )
        objectives = [ObjectiveSpec("accuracy", "maximize")]
        gate = PromotionGate(policy, objectives)

        decision = gate.evaluate(
            incumbent_metrics={"accuracy": [0.8] * 5},
            candidate_metrics={"accuracy": [0.9] * 5},
            constraint_data={"error_rate": (30, 100)},  # (violations, trials)
        )

        assert decision.decision == "reject"
        assert "Chance constraints not satisfied" in decision.reason

    def test_constraint_direction_matches_canonical_violation_bound(self) -> None:
        """Axis C flip: the SAME spec yields the canonical (violation-bound) verdict.

        Under the old SDK semantics (lower bound on a SUCCESS rate >= threshold)
        a high count would PASS; canonical treats the count as VIOLATIONS and
        bounds them from above, so a high count must FAIL and a low count must
        PASS. This asserts the verdict has flipped to the canonical one.
        """
        policy = PromotionPolicy(
            chance_constraints=[
                ChanceConstraint(name="error_rate", threshold=0.20, confidence=0.95)
            ]
        )
        gate = PromotionGate(policy, [ObjectiveSpec("accuracy", "maximize")])

        # 90/100: old success-rate logic (0.90 lower bound >= 0.20) PASSED;
        # canonical violation-rate upper bound (~0.95) > 0.20 -> FAIL.
        high = gate.evaluate(
            incumbent_metrics={"accuracy": [0.8] * 5},
            candidate_metrics={"accuracy": [0.9] * 5},
            constraint_data={"error_rate": (90, 100)},
        )
        assert high.chance_results[0].satisfied is False
        assert high.chance_results[0].upper_bound > 0.20

        # 5/100: canonical upper bound (~0.11) <= 0.20 -> PASS.
        low = gate.evaluate(
            incumbent_metrics={"accuracy": [0.8] * 5},
            candidate_metrics={"accuracy": [0.9] * 5},
            constraint_data={"error_rate": (5, 100)},
        )
        assert low.chance_results[0].satisfied is True
        assert low.chance_results[0].upper_bound <= 0.20

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

    def test_banded_objective_in_band_no_superiority(self) -> None:
        """A banded objective gates but never grants superiority.

        Matching the canonical gate (promotion.py:217-222, :790-795), a banded
        objective only contributes a pass/fail band constraint and is never
        "superior". With no standard superior objective, an in-band candidate
        yields NoDecision (the old SDK gate promoted on the band alone).
        """
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

        assert decision.decision == "no_decision"
        assert decision.objective_results[0].verdict == "in_band"


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
    """Tests for PromotionGate.from_spec_artifact using mocked ObjectiveSchema."""

    def test_from_policy_returns_none_without_policy(self) -> None:
        """from_policy returns None when policy is missing."""
        gate = PromotionGate.from_policy(promotion_policy=None, objective_schema=None)
        assert gate is None

    def test_from_policy_with_objective_schema(self) -> None:
        """from_policy builds objective specs without artifact duck typing."""
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

        schema = ObjectiveSchema.from_objectives(
            [ObjectiveDefinition(name="accuracy", orientation="maximize", weight=1.0)]
        )

        gate = PromotionGate.from_policy(
            promotion_policy=PromotionPolicy(alpha=0.05),
            objective_schema=schema,
        )

        assert gate is not None
        assert "accuracy" in gate.objectives
        assert gate.objectives["accuracy"].direction == "maximize"

    def test_no_promotion_policy_returns_none(self) -> None:
        """Returns None when artifact has no promotion policy."""
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

        class MockArtifact:
            promotion_policy = None
            objective_schema = ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition(
                        name="accuracy", orientation="maximize", weight=1.0
                    )
                ]
            )

        gate = PromotionGate.from_spec_artifact(MockArtifact())
        assert gate is None

    def test_with_none_objective_schema(self) -> None:
        """Returns gate with empty objectives when objective_schema is None."""

        class MockArtifact:
            promotion_policy = PromotionPolicy(alpha=0.05)
            objective_schema = None

        gate = PromotionGate.from_spec_artifact(MockArtifact())
        assert gate is not None
        assert len(gate.objectives) == 0

    def test_with_empty_objectives_list(self) -> None:
        """Returns gate with empty objectives when objectives list is empty."""

        # Create ObjectiveSchema with empty objectives list directly
        # (bypassing from_objectives which requires at least one objective)
        class MockObjectiveSchema:
            objectives: list = []

        class MockArtifact:
            promotion_policy = PromotionPolicy(alpha=0.05)
            objective_schema = MockObjectiveSchema()

        gate = PromotionGate.from_spec_artifact(MockArtifact())
        assert gate is not None
        assert len(gate.objectives) == 0

    def test_default_band_alpha_when_not_specified(self) -> None:
        """Default band_alpha is 0.05 when not specified."""
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

        class MockArtifact:
            promotion_policy = PromotionPolicy()
            objective_schema = ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition(
                        name="consistency",
                        orientation="band",
                        weight=1.0,
                        band=BandTarget(low=0.85, high=0.95),
                        # band_alpha not specified - should default to 0.05
                    ),
                ]
            )

        gate = PromotionGate.from_spec_artifact(MockArtifact())
        assert gate is not None
        obj = gate.objectives["consistency"]
        assert obj.direction == "band"
        assert abs(obj.band_alpha - 0.05) < 1e-10  # Default value

    def test_with_standard_objectives(self) -> None:
        """Creates gate from artifact with standard objectives."""
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

        class MockArtifact:
            promotion_policy = PromotionPolicy(alpha=0.05)
            objective_schema = ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition(
                        name="accuracy", orientation="maximize", weight=1.0
                    ),
                    ObjectiveDefinition(
                        name="latency", orientation="minimize", weight=1.0
                    ),
                ]
            )

        gate = PromotionGate.from_spec_artifact(MockArtifact())
        assert gate is not None
        assert len(gate.objectives) == 2
        assert gate.objectives["accuracy"].direction == "maximize"
        assert gate.objectives["latency"].direction == "minimize"

    def test_with_banded_objectives_list_target(self) -> None:
        """Creates gate from artifact with banded objectives using list target."""
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

        class MockArtifact:
            promotion_policy = PromotionPolicy()
            objective_schema = ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition(
                        name="consistency",
                        orientation="band",
                        weight=1.0,
                        band=BandTarget(low=0.85, high=0.95),
                        band_alpha=0.10,
                    ),
                ]
            )

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
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

        class MockArtifact:
            promotion_policy = PromotionPolicy()
            objective_schema = ObjectiveSchema.from_objectives(
                [
                    ObjectiveDefinition(
                        name="cost",
                        orientation="band",
                        weight=1.0,
                        band=BandTarget(center=0.10, tol=0.02),
                    ),
                ]
            )

        gate = PromotionGate.from_spec_artifact(MockArtifact())
        assert gate is not None
        obj = gate.objectives["cost"]
        assert obj.direction == "band"
        assert obj.band is not None
        # BandTarget should have center and tol preserved
        assert abs(obj.band.center - 0.10) < 1e-10  # type: ignore[operator]
        assert abs(obj.band.tol - 0.02) < 1e-10  # type: ignore[operator]


class TestFromSpecArtifactRealArtifact:
    """Tests for from_spec_artifact with real TVLSpecArtifact."""

    def test_from_spec_artifact_with_real_artifact(self) -> None:
        """Test from_spec_artifact with actual TVLSpecArtifact structure."""
        from pathlib import Path

        from traigent.tvl import load_tvl_spec

        spec_path = Path("examples/tvl/promotion_policy/promotion.tvl.yml")
        artifact = load_tvl_spec(spec_path=spec_path)

        gate = PromotionGate.from_spec_artifact(artifact)

        assert gate is not None
        # gate.objectives is a dict {name: ObjectiveSpec}, not a list!
        assert len(gate.objectives) > 0

        # Verify objectives extracted correctly
        # The promotion.tvl.yml has: task_accuracy, response_latency,
        # cost_per_request, safety_score
        assert "task_accuracy" in gate.objectives
        assert "response_latency" in gate.objectives
        assert "cost_per_request" in gate.objectives
        assert "safety_score" in gate.objectives

        # Verify objective directions
        assert gate.objectives["task_accuracy"].direction == "maximize"
        assert gate.objectives["response_latency"].direction == "minimize"
        assert gate.objectives["cost_per_request"].direction == "minimize"
        assert gate.objectives["safety_score"].direction == "maximize"

    def test_from_spec_artifact_returns_none_without_policy(self) -> None:
        """Test from_spec_artifact returns None when no promotion_policy."""
        from pathlib import Path

        from traigent.tvl import load_tvl_spec

        # hello_tvl has promotion gate: manual_review but no promotion_policy
        spec_path = Path("examples/tvl/hello_tvl/hello_tvl.tvl.yml")
        artifact = load_tvl_spec(spec_path=spec_path)

        gate = PromotionGate.from_spec_artifact(artifact)
        assert gate is None
