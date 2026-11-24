"""SDK-129 group A regressions for the TVL layer (N15, N124, N125, N126, N127)."""

from __future__ import annotations

import pytest

from traigent.tvl.models import BandTarget
from traigent.tvl.objectives import BandedObjectiveSpec, tost_equivalence_test
from traigent.tvl.promotion_gate import (
    ObjectiveSpec,
    PromotionGate,
    PromotionPolicy,
)
from traigent.tvl.spec_loader import load_tvl_spec
from traigent.utils.exceptions import TVLValidationError


def test_n15_singleton_is_insufficient_evidence_not_equivalence() -> None:
    band = BandTarget(low=0.8, high=0.95)
    for value in (0.8, 0.95):
        result = tost_equivalence_test([value], band)
        assert result.insufficient_evidence is True
        assert result.point_in_band is True  # point membership kept separate
        assert result.is_equivalent is False
        assert result.p_lower == 1.0 and result.p_upper == 1.0
    spec = BandedObjectiveSpec(name="quality", target=band)
    assert spec.is_satisfied([0.8]) is False


def test_n15_one_observation_cannot_promote_even_with_a_superior_objective() -> None:
    policy = PromotionPolicy(alpha=0.05, min_effect={"accuracy": 0.01})
    gate = PromotionGate(
        policy,
        [
            ObjectiveSpec("accuracy", "maximize"),
            ObjectiveSpec("quality", "band", band=BandTarget(low=0.8, high=0.95)),
        ],
    )
    decision = gate.evaluate(
        incumbent_metrics={
            "accuracy": [0.70, 0.71, 0.72, 0.70, 0.71, 0.69],
            "quality": [0.85, 0.86],
        },
        candidate_metrics={
            "accuracy": [0.90, 0.92, 0.91, 0.93, 0.90, 0.92],
            "quality": [0.8],
        },
    )
    assert decision.decision == "reject"
    band = next(r for r in decision.objective_results if r.name == "quality")
    assert band.verdict == "out_of_band" and band.p_value == 1.0


def _gate() -> PromotionGate:
    policy = PromotionPolicy(alpha=0.05, min_effect={"accuracy": 0.01})
    return PromotionGate(policy, [ObjectiveSpec("accuracy", "maximize")])


def test_n124_zero_variance_reject_says_it_is_untestable_not_inferior() -> None:
    decision = _gate().evaluate(
        incumbent_metrics={"accuracy": [0.8] * 30},
        candidate_metrics={"accuracy": [0.9] * 30},
    )
    # never promotes, and is NOT the "reject" of a measured-inferior candidate
    assert decision.decision == "no_decision"
    assert decision.dominance_satisfied is False
    assert "not proven" in decision.reason.lower()
    assert "zero-variance" in decision.reason
    assert decision.untestable_objectives == ["accuracy"]
    result = decision.objective_results[0]
    assert result.verdict == "untestable" and result.noninf_inconclusive


def test_n124_untestable_plus_measured_inferior_still_rejects() -> None:
    policy = PromotionPolicy(alpha=0.05, min_effect={"accuracy": 0.01, "cost": 0.01})
    gate = PromotionGate(
        policy,
        [ObjectiveSpec("accuracy", "maximize"), ObjectiveSpec("cost", "maximize")],
    )
    decision = gate.evaluate(
        incumbent_metrics={
            "accuracy": [0.8] * 30,
            "cost": [0.90, 0.92, 0.91, 0.89, 0.90, 0.91],
        },
        candidate_metrics={
            "accuracy": [0.9] * 30,
            "cost": [0.65, 0.74, 0.70, 0.60, 0.72, 0.68],
        },
    )
    assert decision.decision == "reject"


def test_n124_genuinely_inferior_still_rejects() -> None:
    decision = _gate().evaluate(
        incumbent_metrics={"accuracy": [0.90, 0.92, 0.91, 0.89, 0.90, 0.91]},
        candidate_metrics={"accuracy": [0.65, 0.74, 0.70, 0.60, 0.72, 0.68]},
    )
    assert decision.decision == "reject"
    # the test was computable (varying differences), so this is measured, not untestable
    assert decision.untestable_objectives == []
    assert all(not r.noninf_inconclusive for r in decision.objective_results)


def test_n125_empty_samples_fail_closed_not_zero_division() -> None:
    policy = PromotionPolicy(alpha=0.05, min_effect={"accuracy": 0.01, "latency": 1.0})
    gate = PromotionGate(
        policy,
        [ObjectiveSpec("accuracy", "maximize"), ObjectiveSpec("latency", "minimize")],
    )
    decision = gate.evaluate(
        incumbent_metrics={"accuracy": [], "latency": [80.0, 82.0, 81.0]},
        candidate_metrics={"accuracy": [0.9, 0.8, 0.85], "latency": [70.0, 71.0, 72.5]},
    )
    assert decision.decision == "reject"
    assert "accuracy" in decision.reason


def test_n125_numpy_arrays_are_not_truth_tested() -> None:
    np = pytest.importorskip("numpy")
    gate = _gate()
    decision = gate.evaluate(
        incumbent_metrics={"accuracy": np.array([0.70, 0.71, 0.72, 0.70, 0.71, 0.69])},
        candidate_metrics={"accuracy": np.array([0.90, 0.92, 0.91, 0.93, 0.90, 0.92])},
    )
    assert decision.decision == "promote"
    empty = gate.evaluate(
        incumbent_metrics={"accuracy": np.array([])},
        candidate_metrics={"accuracy": np.array([0.9, 0.8])},
    )
    assert empty.decision != "promote"


_SPEC = """
tvl_version: "0.9"
tvl:
  module: test.defaults
tvars:
  - name: n
    type: int
    domain:
      range: [1, 5]
    default: 1
{extra}
objectives:
  - name: accuracy
    direction: maximize
"""


def test_n126_top_level_default_outside_the_domain_is_refused(tmp_path) -> None:
    path = tmp_path / "s.tvl.yml"
    path.write_text(_SPEC.format(extra="defaults:\n  n: 9\n"))
    with pytest.raises(TVLValidationError):
        load_tvl_spec(spec_path=path)


def test_missing_data_reject_classifies_available_objectives_sdk129_bar_r2() -> None:
    gate = PromotionGate(
        PromotionPolicy(alpha=0.05, min_effect={"a": 0.05, "b": 0.05}),
        [ObjectiveSpec("a", "maximize"), ObjectiveSpec("b", "maximize")],
    )
    decision = gate.evaluate({"a": [1.0, 1.0]}, {"a": [0.0, 0.0]})
    assert decision.decision == "reject"
    by_name = {r.name: r for r in decision.objective_results}
    assert by_name["a"].verdict == "inferior"
    assert decision.untestable_objectives == []
