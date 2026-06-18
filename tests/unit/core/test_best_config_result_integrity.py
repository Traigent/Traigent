"""Regression tests for results-integrity bugs fixed in #1340.

F-C: best_metrics must match best_config's trial, not the max-score trial.
F-B: declared ObjectiveDefinition.orientation is respected in best-config
     selection; name-pattern heuristics are a fallback only.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from traigent.api.types import OptimizationResult, OptimizationStatus, TrialResult, TrialStatus
from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.core.result_selection import select_best_configuration
from traigent.utils.objectives import is_minimization_objective


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trial(
    trial_id: str,
    config: dict,
    metrics: dict,
    *,
    status: TrialStatus = TrialStatus.COMPLETED,
    successful_examples: int = 10,
) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        config=config,
        metrics=metrics,
        status=status,
        duration=0.1,
        timestamp=datetime.now(),
        metadata={"successful_examples": successful_examples, "examples_attempted": 10},
    )


def _make_result(
    trials: list[TrialResult],
    best_config: dict,
    best_score: float | None,
    objectives: list[str],
    best_trial_id: str | None = None,
) -> OptimizationResult:
    metadata: dict = {}
    if best_trial_id is not None:
        metadata["best_trial_id"] = best_trial_id
    return OptimizationResult(
        trials=trials,
        best_config=best_config,
        best_score=best_score,
        optimization_id="opt_test",
        duration=1.0,
        convergence_info={},
        status=OptimizationStatus.COMPLETED,
        objectives=objectives,
        algorithm="random",
        timestamp=datetime.now(),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# F-C: best_metrics uses best_config's trial
# ---------------------------------------------------------------------------


class TestBestMetricsMatchesBestConfig:
    """F-C regression: best_metrics must reflect the winning trial, not max()."""

    def test_minimize_primary_metrics_match_winning_cheap_trial(self):
        """Minimize latency: cheap wins; best_metrics must show cheap's numbers."""
        cheap = _make_trial(
            "cheap", {"model": "gpt-3.5"}, {"latency": 0.01, "accuracy": 0.70}
        )
        premium = _make_trial(
            "premium", {"model": "gpt-4"}, {"latency": 0.50, "accuracy": 0.95}
        )
        # latency is a minimize objective; cheap is the winner
        result = _make_result(
            trials=[cheap, premium],
            best_config=cheap.config,
            best_score=0.01,
            objectives=["latency"],
            best_trial_id="cheap",
        )
        bm = result.best_metrics
        assert bm["latency"] == pytest.approx(0.01), (
            "best_metrics latency should match cheap trial (the winner), not premium"
        )
        assert bm["accuracy"] == pytest.approx(0.70), (
            "best_metrics accuracy should match cheap trial (the winner)"
        )

    def test_maximize_primary_metrics_match_winning_accurate_trial(self):
        """Maximize accuracy: premium wins; best_metrics must show premium's numbers."""
        cheap = _make_trial(
            "cheap", {"model": "gpt-3.5"}, {"accuracy": 0.70, "cost": 0.01}
        )
        premium = _make_trial(
            "premium", {"model": "gpt-4"}, {"accuracy": 0.95, "cost": 0.50}
        )
        result = _make_result(
            trials=[cheap, premium],
            best_config=premium.config,
            best_score=0.95,
            objectives=["accuracy"],
            best_trial_id="premium",
        )
        bm = result.best_metrics
        assert bm["accuracy"] == pytest.approx(0.95)
        assert bm["cost"] == pytest.approx(0.50)

    def test_best_metrics_falls_back_to_best_trial_id_when_config_mismatch(self):
        """If config match fails, fall back to metadata best_trial_id."""
        trial_a = _make_trial("ta", {"x": 1}, {"score": 0.8})
        trial_b = _make_trial("tb", {"x": 2}, {"score": 0.6})
        # best_config points to an empty dict (aggregated path edge case);
        # best_trial_id in metadata should resolve to trial_a
        result = _make_result(
            trials=[trial_a, trial_b],
            best_config={},
            best_score=0.8,
            objectives=["score"],
            best_trial_id="ta",
        )
        bm = result.best_metrics
        assert bm["score"] == pytest.approx(0.8)

    def test_best_metrics_no_winner_returns_empty(self):
        """NO_CERTIFIED_SELECTION shape: empty best_config + None best_score."""
        trial = _make_trial("t1", {"x": 1}, {"accuracy": 0.9})
        result = _make_result(
            trials=[trial],
            best_config={},
            best_score=None,
            objectives=["accuracy"],
        )
        assert result.best_metrics == {}


# ---------------------------------------------------------------------------
# F-B: declared ObjectiveDefinition.orientation is respected
# ---------------------------------------------------------------------------


class TestDeclaredOrientationRespected:
    """F-B regression: declared orientation must override name-pattern heuristics."""

    def test_is_minimization_objective_respects_explicit_minimize(self):
        """Explicit minimize orientation beats name-pattern logic."""
        # "spend" is NOT in _MINIMIZE_OBJECTIVE_PATTERNS so heuristic → maximize
        assert is_minimization_objective("spend") is False
        # But explicit orientation must override:
        assert is_minimization_objective("spend", orientation="minimize") is True

    def test_is_minimization_objective_respects_explicit_maximize(self):
        """Explicit maximize orientation beats name-pattern logic."""
        # "cost" IS in patterns → heuristic says minimize
        assert is_minimization_objective("cost") is True
        # But explicit orientation=maximize must override:
        assert is_minimization_objective("cost", orientation="maximize") is False

    def test_is_minimization_objective_band_returns_false(self):
        """Banded objectives use deviation logic; is_minimization must return False."""
        assert is_minimization_objective("latency", orientation="band") is False

    def test_is_minimization_objective_none_orientation_uses_heuristic(self):
        """When orientation is None the heuristic fallback applies unchanged."""
        assert is_minimization_objective("cost", orientation=None) is True
        assert is_minimization_objective("accuracy", orientation=None) is False

    def test_select_best_config_honors_declared_minimize_for_custom_name(self):
        """Custom-named objective with declared minimize selects the lower value."""
        # "spend" is not in _MINIMIZE_OBJECTIVE_PATTERNS (heuristic → maximize)
        # but we declare orientation=minimize via objective_orientations.
        # Use comparability_mode="legacy" to bypass comparability-metadata
        # requirements and focus the test on the orientation fix.
        cheap = _make_trial("cheap", {"model": "gpt-3.5"}, {"spend": 0.001})
        premium = _make_trial("premium", {"model": "gpt-4"}, {"spend": 0.100})
        selection = select_best_configuration(
            trials=[cheap, premium],
            primary_objective="spend",
            config_space_keys={"model"},
            aggregate_configs=False,
            comparability_mode="legacy",
            objective_orientations={"spend": "minimize"},
        )
        assert selection.best_config == cheap.config, (
            "With declared minimize, cheap (spend=0.001) should win over premium (spend=0.100)"
        )
        assert selection.best_score == pytest.approx(0.001)

    def test_select_best_config_honors_declared_maximize_for_cost_name(self):
        """'cost' heuristic says minimize but declared maximize should pick the higher value."""
        low_cost = _make_trial("low", {"model": "gpt-3.5"}, {"cost": 0.01})
        high_cost = _make_trial("high", {"model": "gpt-4"}, {"cost": 0.99})
        selection = select_best_configuration(
            trials=[low_cost, high_cost],
            primary_objective="cost",
            config_space_keys={"model"},
            aggregate_configs=False,
            comparability_mode="legacy",
            objective_orientations={"cost": "maximize"},
        )
        assert selection.best_config == high_cost.config, (
            "With declared maximize on 'cost', high_cost should win"
        )
        assert selection.best_score == pytest.approx(0.99)

    def test_select_best_config_without_orientations_uses_heuristic(self):
        """Backward compat: no objective_orientations → heuristic applies."""
        cheap = _make_trial("cheap", {"model": "gpt-3.5"}, {"cost": 0.01})
        premium = _make_trial("premium", {"model": "gpt-4"}, {"cost": 0.99})
        # heuristic: 'cost' → minimize → cheap wins
        selection = select_best_configuration(
            trials=[cheap, premium],
            primary_objective="cost",
            config_space_keys={"model"},
            aggregate_configs=False,
            comparability_mode="legacy",
        )
        assert selection.best_config == cheap.config
        assert selection.best_score == pytest.approx(0.01)

    def test_objective_schema_orientation_end_to_end_via_select_best(self):
        """ObjectiveSchema with minimize on a custom name threads correctly."""
        obj_def = ObjectiveDefinition(name="spend", orientation="minimize", weight=1.0)
        schema = ObjectiveSchema.from_objectives([obj_def])
        orientations = {obj.name: str(obj.orientation) for obj in schema.objectives}

        cheap = _make_trial("cheap", {"model": "gpt-3.5"}, {"spend": 0.005})
        premium = _make_trial("premium", {"model": "gpt-4"}, {"spend": 0.200})
        selection = select_best_configuration(
            trials=[cheap, premium],
            primary_objective="spend",
            config_space_keys={"model"},
            aggregate_configs=False,
            comparability_mode="legacy",
            objective_orientations=orientations,
        )
        assert selection.best_config == cheap.config
        assert selection.best_score == pytest.approx(0.005)
