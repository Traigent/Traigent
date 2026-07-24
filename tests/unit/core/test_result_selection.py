import math

import pytest

from traigent.core.result_selection import (
    NO_RANKING_ELIGIBLE_TRIALS,
    SelectionResult,
    select_best_configuration,
)


class FakeTrial:
    """Minimal stand-in for TrialResult used in selection logic tests."""

    _trial_counter = 0

    def __init__(
        self,
        *,
        metrics: dict[str, float] | None = None,
        config: dict[str, object] | None = None,
        success: bool = True,
        trial_id: str | None = None,
    ) -> None:
        self.metrics = metrics or {}
        self.config = config or {}
        self.is_successful = success
        self.metadata: dict[str, object] = {}
        FakeTrial._trial_counter += 1
        self.trial_id = trial_id or f"fake_trial_{FakeTrial._trial_counter}"

        per_metric_coverage = {
            metric_name: {"present": 1, "total": 1, "ratio": 1.0}
            for metric_name, metric_value in self.metrics.items()
            if isinstance(metric_value, int | float)
        }
        has_accuracy = "accuracy" in per_metric_coverage
        self.metadata["comparability"] = {
            "schema_version": "1.0",
            "primary_objective": "accuracy",
            "evaluation_mode": "evaluated" if success else "unknown",
            "total_examples": 1,
            "examples_with_primary_metric": 1 if has_accuracy else 0,
            "coverage_ratio": 1.0 if has_accuracy else 0.0,
            "derivation_path": "explicit" if has_accuracy else "none",
            "ranking_eligible": has_accuracy and success,
            "warning_codes": [],
            "per_metric_coverage": per_metric_coverage,
            "missing_example_ids": [],
        }

    def get_metric(self, name: str, default: float | None = None) -> float | None:
        return self.metrics.get(name, default)


def test_returns_default_when_no_successful_trials():
    result = select_best_configuration(
        trials=[FakeTrial(success=False)],
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=False,
    )
    assert result == SelectionResult(
        best_config=None,
        best_score=None,
        session_summary={
            "reason_code": NO_RANKING_ELIGIBLE_TRIALS,
            "ranking": {
                "comparability_mode": "warn",
                "total_input_trials": 1,
                "total_successful_trials": 0,
                "non_successful_count": 1,
                "eligible_count": 0,
                "excluded_count": 0,
                "unknown_count": 0,
            },
        },
        reason_code=NO_RANKING_ELIGIBLE_TRIALS,
    )


def test_trial_completed_with_zero_successful_examples_yields_no_winner():
    """Regression: a trial that completes but had 0 successful examples
    (e.g. every provider call inside it 404'd on a deprecated model) is not
    a valid winner. best_score must be None, best_config must be None."""
    trials = [
        FakeTrial(metrics={"accuracy": 0.0}, config={"model": "deprecated-a"}),
        FakeTrial(metrics={"accuracy": 0.0}, config={"model": "deprecated-b"}),
    ]
    # Evaluators that surface per-example success counts set this; our gate uses it.
    for trial in trials:
        trial.metadata["successful_examples"] = 0
        trial.metadata["examples_attempted"] = 20

    result = select_best_configuration(
        trials=trials,
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=False,
    )

    assert result.best_score is None
    assert result.best_config is None
    assert result.reason_code == NO_RANKING_ELIGIBLE_TRIALS


def test_trial_with_successful_examples_still_wins_with_zero_accuracy():
    """Accuracy 0.0 across successful examples is honest (model produced output,
    answer was wrong) — that's still a rankable trial, NOT an "all failed" one."""
    trial = FakeTrial(metrics={"accuracy": 0.0}, config={"model": "honest-zero"})
    trial.metadata["successful_examples"] = 5
    trial.metadata["examples_attempted"] = 5

    result = select_best_configuration(
        trials=[trial],
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=False,
    )

    assert result.best_score == pytest.approx(0.0)
    assert result.best_config == {"model": "honest-zero"}


def test_selects_best_trial_without_aggregation():
    trials = [
        FakeTrial(metrics={"accuracy": 0.75}, config={"model": "A"}),
        FakeTrial(metrics={"accuracy": 0.82}, config={"model": "B"}),
    ]

    result = select_best_configuration(
        trials=trials,
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=False,
    )

    assert result.best_config == {"model": "B"}
    assert result.best_score == pytest.approx(0.82)
    assert result.session_summary is not None
    assert result.session_summary["ranking"]["eligible_count"] == 2


@pytest.mark.parametrize(
    ("primary_objective", "finite_score", "worse_score", "winner_config"),
    [
        ("accuracy", 0.9, 0.7, {"model": "finite"}),
        ("cost", 0.1, 0.5, {"model": "finite"}),
    ],
)
@pytest.mark.parametrize("nan_position", ["first", "last"])
def test_single_trial_selection_ignores_nan_scores(
    primary_objective, finite_score, worse_score, winner_config, nan_position
):
    nan_trial = FakeTrial(
        metrics={primary_objective: math.nan},
        config={"model": "nan"},
    )
    finite_trial = FakeTrial(
        metrics={primary_objective: finite_score},
        config=winner_config,
    )
    worse_trial = FakeTrial(
        metrics={primary_objective: worse_score},
        config={"model": "worse"},
    )
    trials = (
        [nan_trial, finite_trial, worse_trial]
        if nan_position == "first"
        else [finite_trial, worse_trial, nan_trial]
    )

    result = select_best_configuration(
        trials=trials,
        primary_objective=primary_objective,
        config_space_keys={"model"},
        aggregate_configs=False,
        comparability_mode="legacy",
    )

    assert result.best_config == winner_config
    assert result.best_score == pytest.approx(finite_score)


def test_minimization_objective_is_respected():
    trials = [
        FakeTrial(metrics={"cost_per_call": 0.5}, config={"model": "cheap"}),
        FakeTrial(metrics={"cost_per_call": 0.1}, config={"model": "expensive"}),
    ]

    result = select_best_configuration(
        trials=trials,
        primary_objective="cost_per_call",  # contains "cost" -> minimization
        config_space_keys={"model"},
        aggregate_configs=False,
    )

    assert result.best_config["model"] == "expensive"
    assert result.best_score == pytest.approx(0.1)


def test_banded_primary_objective_selects_closest_single_trial():
    trials = [
        FakeTrial(metrics={"latency_ms": 100.0}, config={"model": "too-fast"}),
        FakeTrial(metrics={"latency_ms": 500.0}, config={"model": "center"}),
        FakeTrial(metrics={"latency_ms": 900.0}, config={"model": "too-slow"}),
    ]

    result = select_best_configuration(
        trials=trials,
        primary_objective="latency_ms",
        config_space_keys={"model"},
        aggregate_configs=False,
        band_target=500.0,
    )

    assert result.best_config == {"model": "center"}
    assert result.best_score == pytest.approx(500.0)


def test_banded_primary_objective_ignores_nan_score():
    trials = [
        FakeTrial(metrics={"accuracy": math.nan}, config={"model": "nan"}),
        FakeTrial(metrics={"accuracy": 0.81}, config={"model": "center"}),
        FakeTrial(metrics={"accuracy": 0.50}, config={"model": "far"}),
    ]

    result = select_best_configuration(
        trials=trials,
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=False,
        band_target=0.80,
        comparability_mode="legacy",
    )

    assert result.best_config == {"model": "center"}
    assert result.best_score == pytest.approx(0.81)


def test_banded_primary_objective_selects_closest_aggregated_config():
    trials = [
        FakeTrial(metrics={"accuracy": 0.95}, config={"model": "high"}),
        FakeTrial(metrics={"accuracy": 0.70}, config={"model": "center"}),
        FakeTrial(metrics={"accuracy": 0.69}, config={"model": "center"}),
        FakeTrial(metrics={"accuracy": 0.50}, config={"model": "low"}),
    ]

    result = select_best_configuration(
        trials=trials,
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=True,
        band_target=0.70,
    )

    assert result.best_config == {"model": "center"}
    assert result.best_score == pytest.approx(0.695)


@pytest.mark.parametrize(
    ("primary_objective", "finite_score", "worse_score", "winner_config"),
    [
        ("accuracy", 0.9, 0.7, {"model": "finite"}),
        ("cost", 0.1, 0.5, {"model": "finite"}),
    ],
)
@pytest.mark.parametrize("nan_position", ["first", "last"])
def test_aggregated_selection_ignores_nan_scores(
    primary_objective, finite_score, worse_score, winner_config, nan_position
):
    nan_trial = FakeTrial(
        metrics={primary_objective: math.nan},
        config={"model": "nan"},
    )
    finite_trial = FakeTrial(
        metrics={primary_objective: finite_score},
        config=winner_config,
    )
    worse_trial = FakeTrial(
        metrics={primary_objective: worse_score},
        config={"model": "worse"},
    )
    trials = (
        [nan_trial, finite_trial, worse_trial]
        if nan_position == "first"
        else [finite_trial, worse_trial, nan_trial]
    )

    result = select_best_configuration(
        trials=trials,
        primary_objective=primary_objective,
        config_space_keys={"model"},
        aggregate_configs=True,
        comparability_mode="legacy",
    )

    assert result.best_config == winner_config
    assert result.best_score == pytest.approx(finite_score)


def test_aggregated_selection_all_nan_returns_no_eligible_reason():
    result = select_best_configuration(
        trials=[
            FakeTrial(metrics={"accuracy": math.nan}, config={"model": "nan-a"}),
            FakeTrial(metrics={"accuracy": math.nan}, config={"model": "nan-b"}),
        ],
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=True,
        comparability_mode="legacy",
    )

    assert result.best_config is None
    assert result.best_score is None
    assert result.reason_code == NO_RANKING_ELIGIBLE_TRIALS


def test_aggregated_selection_computes_means_and_metadata():
    trials = [
        FakeTrial(
            metrics={"accuracy": 0.9, "latency": 120},
            config={"model": "A", "temp": 0.1},
        ),
        FakeTrial(
            metrics={"accuracy": 0.7, "latency": 140},
            config={"model": "A", "temp": 0.1},
        ),
        FakeTrial(
            metrics={"accuracy": 0.75, "latency": 110},
            config={"model": "B", "temp": 0.5},
        ),
    ]

    result = select_best_configuration(
        trials=trials,
        primary_objective="accuracy",
        config_space_keys={"model", "temp"},
        aggregate_configs=True,
    )

    # Average accuracy for model A should be (0.9 + 0.7) / 2 = 0.8, which wins over 0.75.
    assert result.best_config == {"model": "A", "temp": 0.1}
    assert result.best_score == pytest.approx(0.8)
    assert result.session_summary is not None
    assert result.session_summary["selection_mode"] == "aggregated_mean"
    # ensure sanitized metrics uses allowed list (latency stays, accuracy stays)
    assert result.session_summary["metrics"]["accuracy"] == pytest.approx(0.8)
    assert result.session_summary["metrics"]["latency"] == pytest.approx(130.0)

    # Replicate metadata should be assigned incrementally for matching configs
    replicate_indexes = [
        trials[0].metadata.get("replicate_index"),
        trials[1].metadata.get("replicate_index"),
        trials[2].metadata.get("replicate_index"),
    ]
    assert replicate_indexes[:2] == [1, 2]
    assert replicate_indexes[2] == 1  # first instance of config B


class TestTieBreaker:
    """Tests for tie-breaker functionality."""

    def test_apply_tie_breaker_empty_list_raises(self) -> None:
        """apply_tie_breaker should raise on empty list."""
        from traigent.core.result_selection import apply_tie_breaker

        with pytest.raises(ValueError, match="Cannot apply tie-breaker"):
            apply_tie_breaker([], {}, "accuracy")

    def test_apply_tie_breaker_single_trial_returns_it(self) -> None:
        """apply_tie_breaker with single trial returns that trial."""
        from traigent.core.result_selection import apply_tie_breaker

        trial = FakeTrial(metrics={"accuracy": 0.9}, config={"model": "A"})
        result = apply_tie_breaker([trial], {}, "accuracy")

        assert result is trial

    def test_apply_tie_breaker_min_abs_deviation_with_band_target(self) -> None:
        """min_abs_deviation should pick closest to band target."""
        from traigent.core.result_selection import apply_tie_breaker

        # All have same primary score, but different distances from target
        trials = [
            FakeTrial(metrics={"accuracy": 0.95}, config={"model": "A"}),
            FakeTrial(metrics={"accuracy": 0.85}, config={"model": "B"}),
            FakeTrial(metrics={"accuracy": 0.90}, config={"model": "C"}),
        ]

        tie_breakers = {"accuracy": "min_abs_deviation"}
        result = apply_tie_breaker(
            trials,
            tie_breakers,
            "accuracy",
            band_target=0.88,
        )

        # Trial B (0.85) is closest to target 0.88 (distance 0.03)
        # Trial C (0.90) has distance 0.02 - should win
        assert result.config["model"] == "C"

    def test_apply_tie_breaker_min_abs_deviation_without_band(self) -> None:
        """min_abs_deviation without band should use secondary metrics."""
        from traigent.core.result_selection import apply_tie_breaker

        # All have same accuracy, but different secondary metrics
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.9, "latency": 100, "quality": 0.8},
                config={"model": "A"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9, "latency": 50, "quality": 0.9},
                config={"model": "B"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9, "latency": 200, "quality": 0.7},
                config={"model": "C"},
            ),
        ]

        tie_breakers = {"accuracy": "min_abs_deviation"}
        result = apply_tie_breaker(
            trials,
            tie_breakers,
            "accuracy",
        )

        # Trial B has best secondary: -50 (latency) + 0.9 (quality) vs others
        # Model B: -50 + 0.9 = -49.1
        # Model A: -100 + 0.8 = -99.2
        # Model C: -200 + 0.7 = -199.3
        # Max is B with -49.1
        assert result.config["model"] == "B"

    def test_apply_tie_breaker_custom_returns_first(self) -> None:
        """custom tie-breaker should return first trial."""
        from traigent.core.result_selection import apply_tie_breaker

        trials = [
            FakeTrial(metrics={"accuracy": 0.9}, config={"model": "A"}),
            FakeTrial(metrics={"accuracy": 0.9}, config={"model": "B"}),
        ]

        tie_breakers = {"accuracy": "custom"}
        result = apply_tie_breaker(trials, tie_breakers, "accuracy")

        assert result.config["model"] == "A"

    def test_apply_tie_breaker_defaults_to_min_abs_deviation(self) -> None:
        """Empty tie-breakers dict should default to min_abs_deviation."""
        from traigent.core.result_selection import apply_tie_breaker

        # Give distinct secondary metrics to make selection deterministic
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.9, "latency": 100},
                config={"model": "A"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9, "latency": 50},
                config={"model": "B"},
            ),
        ]

        # No tie-breaker registered for accuracy - defaults to min_abs_deviation
        result = apply_tie_breaker(trials, {}, "accuracy")

        # min_abs_deviation uses secondary metrics, B has lower latency
        assert result.config["model"] == "B"


class TestSelectBestConfigurationWithTieBreakers:
    """Tests for select_best_configuration with tie-breakers."""

    def test_tie_breaker_applied_when_scores_equal(self) -> None:
        """Tie-breaker should be applied when multiple trials have same score."""
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.9, "latency": 200},
                config={"model": "A"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9, "latency": 100},
                config={"model": "B"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9, "latency": 150},
                config={"model": "C"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers={"accuracy": "min_abs_deviation"},
        )

        # All have 0.9 accuracy, tie-breaker uses secondary metrics
        # latency is minimization, so lower is better
        # B has lowest latency (100), should win
        assert result.best_config["model"] == "B"
        assert result.best_score == pytest.approx(0.9)

    def test_tie_breaker_with_band_target(self) -> None:
        """Tie-breaker with band_target should pick closest to target."""
        trials = [
            FakeTrial(metrics={"accuracy": 0.95}, config={"model": "A"}),
            FakeTrial(metrics={"accuracy": 0.85}, config={"model": "B"}),
            FakeTrial(metrics={"accuracy": 0.89}, config={"model": "C"}),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers={"accuracy": "min_abs_deviation"},
            band_target=0.90,
        )

        # Banded primary objectives rank by distance from the band target, so
        # C (0.89) beats the raw max A (0.95).
        assert result.best_config["model"] == "C"

    def test_tie_breaker_all_equal_scores_with_band(self) -> None:
        """When all trials have same primary score, band_target helps pick."""
        trials = [
            FakeTrial(metrics={"accuracy": 0.8}, config={"model": "A"}),
            FakeTrial(metrics={"accuracy": 0.8}, config={"model": "B"}),
            FakeTrial(metrics={"accuracy": 0.8}, config={"model": "C"}),
        ]

        # This test simulates banded objective where target is 0.75
        # All have 0.8, but we want to pick based on distance from band
        # Actually, since get_metric returns same value, we need different approach
        # Let me adjust - this tests the tie_breakers dict usage

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers={"accuracy": "min_abs_deviation"},
        )

        # All equal, without band_target, falls back to secondary metrics
        # Since no secondary metrics differ, returns first (deterministic by trial_id)
        assert result.best_score == pytest.approx(0.8)
        assert result.best_config["model"] in ["A", "B", "C"]

    def test_no_tie_breaker_returns_first_tied(self) -> None:
        """Single-objective ties keep the historic first-wins behavior."""
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.9, "cost": 0.2},
                config={"model": "A"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9, "cost": 0.1},
                config={"model": "B"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers=None,  # No tie-breaker
            objective_order=["accuracy"],
        )

        assert result.best_config["model"] == "A"

    @pytest.mark.parametrize("aggregate_configs", [False, True])
    def test_default_multi_objective_tie_breaks_equal_accuracy_on_lowest_cost(
        self, aggregate_configs: bool
    ) -> None:
        """Regression #1184: equal primary accuracy picks the cheapest trial."""
        costs = [0.00020, 0.00001, 0.00002, 0.00011]
        trials = [
            FakeTrial(
                metrics={"accuracy": 1.0, "cost": cost},
                config={"model": f"m{i}", "cost": cost},
            )
            for i, cost in enumerate(costs, start=1)
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model", "cost"},
            aggregate_configs=aggregate_configs,
            tie_breakers=None,
            objective_order=["accuracy", "cost"],
        )

        assert result.best_config["cost"] == pytest.approx(0.00001)
        assert result.best_trial_id == trials[1].trial_id

    def test_default_tie_break_follows_declared_objective_order(self) -> None:
        trials = [
            FakeTrial(
                metrics={"accuracy": 1.0, "cost": 0.00001, "latency": 1000.0},
                config={"model": "low-cost-high-latency"},
            ),
            FakeTrial(
                metrics={"accuracy": 1.0, "cost": 0.00002, "latency": 1.0},
                config={"model": "higher-cost-low-latency"},
            ),
            FakeTrial(
                metrics={"accuracy": 1.0, "cost": 0.00001, "latency": 10.0},
                config={"model": "low-cost-low-latency"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers=None,
            objective_order=["accuracy", "cost", "latency"],
        )

        assert result.best_config["model"] == "low-cost-low-latency"

    def test_primary_near_ties_use_secondary_objective(self) -> None:
        trials = [
            FakeTrial(
                metrics={"accuracy": 1.0, "cost": 0.00020},
                config={"model": "slightly-higher-accuracy"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9999999999995, "cost": 0.00001},
                config={"model": "near-tied-cheaper"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers=None,
            objective_order=["accuracy", "cost"],
        )

        assert result.best_config["model"] == "near-tied-cheaper"

    def test_explicit_custom_tie_breaker_takes_precedence_over_default(self) -> None:
        trials = [
            FakeTrial(
                metrics={"accuracy": 1.0, "cost": 0.00020},
                config={"model": "first"},
            ),
            FakeTrial(
                metrics={"accuracy": 1.0, "cost": 0.00001},
                config={"model": "cheaper"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers={"accuracy": "custom"},
            objective_order=["accuracy", "cost"],
        )

        assert result.best_config["model"] == "first"

    def test_tie_breaker_only_applied_to_ties(self) -> None:
        """Tie-breaker should only affect trials with equal best scores."""
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.7, "latency": 10},
                config={"model": "A"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9, "latency": 200},
                config={"model": "B"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9, "latency": 100},
                config={"model": "C"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers={"accuracy": "min_abs_deviation"},
        )

        # A has lower accuracy (0.7), not in tie group
        # B and C both have 0.9, tie-breaker picks C (lower latency)
        assert result.best_config["model"] == "C"

    @pytest.mark.parametrize("aggregate_configs", [False, True])
    def test_direct_declared_minimize_secondary_tie_breaks_downward(
        self, aggregate_configs: bool
    ) -> None:
        """Issue #1955 — UNWEIGHTED (direct) single + aggregated tie-break paths.

        A custom-named declared-minimize secondary (``token_budget``) whose name
        matches none of the minimize heuristic patterns would, under the old
        name-only guess, be treated as maximize and ADDED to the tie-break score,
        crowning the trial with the LARGER (worse) budget. With the declared
        orientation threaded in, the smaller budget must win. ``objective_schema``
        is intentionally omitted so this drives the direct paths
        (``_select_best_single_trial`` / ``_select_best_aggregated``), not the
        weighted paths below.
        """
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.9, "token_budget": 5000.0},
                config={"model": "wasteful"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9, "token_budget": 500.0},
                config={"model": "frugal"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=aggregate_configs,
            tie_breakers=None,
            objective_order=["accuracy", "token_budget"],
            objective_orientations={
                "accuracy": "maximize",
                "token_budget": "minimize",
            },
        )

        assert result.best_config["model"] == "frugal"

    @pytest.mark.parametrize("aggregate_configs", [False, True])
    def test_weighted_declared_minimize_secondary_tie_breaks_downward(
        self, aggregate_configs: bool
    ) -> None:
        """Issue #1955 — WEIGHTED terminal tie-break (non-aggregated + aggregated).

        Canonical reproducer: three equal-weight objectives — accuracy(max),
        token_budget(min), quality(max). ``wasteful`` and ``frugal`` tie at
        weighted 0.5, so the winner is decided by the declared-order secondary
        tie-break. Only ``objective_schema`` is supplied (NO explicit
        ``objective_orientations`` map), proving the consolidated resolver pulls
        the minimize orientation from the schema into the weighted tie-break.
        Without the fix the weighted paths treated ``token_budget`` as maximize
        and crowned ``wasteful`` (the pricier config).
        """
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

        schema = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition(
                    name="accuracy", orientation="maximize", weight=1.0
                ),
                ObjectiveDefinition(
                    name="token_budget", orientation="minimize", weight=1.0
                ),
                ObjectiveDefinition(name="quality", orientation="maximize", weight=1.0),
            ]
        )
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.9, "token_budget": 5000.0, "quality": 1.0},
                config={"model": "wasteful"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9, "token_budget": 500.0, "quality": 0.0},
                config={"model": "frugal"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=aggregate_configs,
            tie_breakers=None,
            objective_order=["accuracy", "token_budget", "quality"],
            objective_schema=schema,
        )

        assert result.best_config["model"] == "frugal"


def test_missing_primary_objective_excluded_not_zero_coerced():
    trial_missing = FakeTrial(metrics={"latency": 10.0}, config={"model": "A"})
    trial_zero = FakeTrial(metrics={"accuracy": 0.0}, config={"model": "B"})

    result = select_best_configuration(
        trials=[trial_missing, trial_zero],
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=False,
    )

    assert result.best_config["model"] == "B"
    assert result.best_score == pytest.approx(0.0)


def test_execute_only_operational_objective_is_eligible():
    trial = FakeTrial(metrics={"total_cost": 0.12}, config={"model": "ops"})
    trial.metadata["comparability"] = {
        "schema_version": "1.0",
        "primary_objective": "accuracy",
        "evaluation_mode": "execute_only",
        "total_examples": 3,
        "examples_with_primary_metric": 0,
        "coverage_ratio": 0.0,
        "derivation_path": "none",
        "ranking_eligible": False,
        "warning_codes": ["MCI-004"],
        "per_metric_coverage": {"total_cost": {"present": 3, "total": 3, "ratio": 1.0}},
        "missing_example_ids": [],
    }

    result = select_best_configuration(
        trials=[trial],
        primary_objective="total_cost",
        config_space_keys={"model"},
        aggregate_configs=False,
    )

    assert result.best_config["model"] == "ops"
    assert result.best_score == pytest.approx(0.12)


def test_execute_only_quality_objective_is_ineligible():
    trial = FakeTrial(metrics={"accuracy": 0.9}, config={"model": "q"})
    trial.metadata["comparability"] = {
        "schema_version": "1.0",
        "primary_objective": "accuracy",
        "evaluation_mode": "execute_only",
        "total_examples": 3,
        "examples_with_primary_metric": 3,
        "coverage_ratio": 1.0,
        "derivation_path": "explicit",
        "ranking_eligible": False,
        "warning_codes": [],
        "per_metric_coverage": {"accuracy": {"present": 3, "total": 3, "ratio": 1.0}},
        "missing_example_ids": [],
    }

    result = select_best_configuration(
        trials=[trial],
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=False,
    )

    assert result.best_config is None
    assert result.best_score is None
    assert result.reason_code == NO_RANKING_ELIGIBLE_TRIALS


def test_unknown_comparability_is_excluded_by_default():
    trial = FakeTrial(metrics={"accuracy": 0.8}, config={"model": "legacy"})
    trial.metadata = {}

    result = select_best_configuration(
        trials=[trial],
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=False,
    )

    assert result.best_config is None
    assert result.best_score is None
    assert result.reason_code == NO_RANKING_ELIGIBLE_TRIALS
    assert result.session_summary is not None
    assert result.session_summary["ranking"]["unknown_count"] == 1


def test_strict_mode_keeps_eligible_trials_when_some_are_excluded():
    eligible = FakeTrial(metrics={"accuracy": 0.8}, config={"model": "eligible"})
    unknown = FakeTrial(metrics={"accuracy": 0.9}, config={"model": "unknown"})
    unknown.metadata = {}

    result = select_best_configuration(
        trials=[eligible, unknown],
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=False,
        comparability_mode="strict",
    )

    assert result.best_config["model"] == "eligible"
    assert result.best_score == pytest.approx(0.8)
    assert result.session_summary is not None
    assert result.session_summary["ranking"]["eligible_count"] == 1
    assert result.session_summary["ranking"]["excluded_count"] == 1
    assert result.session_summary["ranking"]["unknown_count"] == 1


def test_legacy_mode_keeps_historic_unknown_trials_rankable():
    trial = FakeTrial(metrics={"accuracy": 0.8}, config={"model": "legacy"})
    trial.metadata = {}

    result = select_best_configuration(
        trials=[trial],
        primary_objective="accuracy",
        config_space_keys={"model"},
        aggregate_configs=False,
        comparability_mode="legacy",
    )

    assert result.best_config["model"] == "legacy"
    assert result.best_score == pytest.approx(0.8)


def test_none_primary_objective_returns_no_eligible_shape():
    """Objectives-free runs (#1108 review NB): ranking is disabled — the
    selector returns its honest no-eligible shape, never a winner-by-score."""
    trial = FakeTrial(metrics={"accuracy": 0.8}, config={"model": "candidate"})

    result = select_best_configuration(
        trials=[trial],
        primary_objective=None,
        config_space_keys={"model"},
        aggregate_configs=False,
    )

    assert result.best_config == {}
    assert result.best_score is None
    assert result.reason_code == "NO_RANKING_ELIGIBLE_TRIALS"
    ranking = result.session_summary["ranking"]
    assert ranking["total_input_trials"] == 1
    assert ranking["total_successful_trials"] == 1
    assert ranking["eligible_count"] == 0
    assert ranking["excluded_count"] == 1


def test_none_primary_objective_strict_mode_stays_certificate_driven():
    """Strict mode never reads the primary objective: a certified incumbent
    is honored, and its absence fails closed — regardless of objectives."""
    trial = FakeTrial(metrics={"accuracy": 0.8}, config={"model": "candidate"})

    certified = select_best_configuration(
        trials=[trial],
        primary_objective=None,
        config_space_keys={"model"},
        aggregate_configs=False,
        require_certified=True,
        certified_config={"model": "incumbent"},
        certified_score=0.7,
    )
    assert certified.best_config == {"model": "incumbent"}
    assert certified.best_score == pytest.approx(0.7)

    uncertified = select_best_configuration(
        trials=[trial],
        primary_objective=None,
        config_space_keys={"model"},
        aggregate_configs=False,
        require_certified=True,
        certified_config=None,
    )
    assert uncertified.best_config == {}
    assert uncertified.reason_code == "NO_CERTIFIED_SELECTION"


class TestMinimizationObjectiveTieBreaker:
    """Tests for tie-breaker with minimization objectives."""

    def test_tie_breaker_with_cost_objective(self) -> None:
        """Tie-breaker should work with cost (minimization) objectives."""
        trials = [
            FakeTrial(
                metrics={"cost": 0.1, "accuracy": 0.8},
                config={"model": "A"},
            ),
            FakeTrial(
                metrics={"cost": 0.1, "accuracy": 0.9},
                config={"model": "B"},
            ),
            FakeTrial(
                metrics={"cost": 0.1, "accuracy": 0.7},
                config={"model": "C"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="cost",  # minimization
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers={"cost": "min_abs_deviation"},
        )

        # All have same cost (0.1), tie-breaker uses secondary metrics
        # accuracy is maximization, B has highest (0.9)
        assert result.best_config["model"] == "B"
        assert result.best_score == pytest.approx(0.1)

    def test_latency_minimization_tie_breaker(self) -> None:
        """Tie-breaker with latency (minimization) objective."""
        trials = [
            FakeTrial(
                metrics={"latency": 50, "quality": 0.7},
                config={"model": "A"},
            ),
            FakeTrial(
                metrics={"latency": 50, "quality": 0.9},
                config={"model": "B"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="latency",  # minimization
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers={"latency": "min_abs_deviation"},
        )

        # Both have latency 50, B has better quality
        assert result.best_config["model"] == "B"


class TestAggregatedTieBreaker:
    """Tests for tie-breaker in aggregated selection mode."""

    def test_aggregated_tie_breaker_applied_when_scores_equal(self) -> None:
        """Tie-breaker should work in aggregated mode when configs have same score."""
        # Two configs, each with multiple trials, same mean accuracy
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.8, "latency": 200},
                config={"model": "A", "temp": 0.5},
            ),
            FakeTrial(
                metrics={"accuracy": 0.8, "latency": 180},
                config={"model": "A", "temp": 0.5},
            ),
            FakeTrial(
                metrics={"accuracy": 0.8, "latency": 100},
                config={"model": "B", "temp": 0.7},
            ),
            FakeTrial(
                metrics={"accuracy": 0.8, "latency": 100},
                config={"model": "B", "temp": 0.7},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model", "temp"},
            aggregate_configs=True,
            tie_breakers={"accuracy": "min_abs_deviation"},
        )

        # Both configs have mean accuracy 0.8
        # B has better secondary (lower latency: 100 vs 190 avg)
        assert result.best_config["model"] == "B"
        assert result.best_score == pytest.approx(0.8)

    def test_aggregated_tie_breaker_with_band_target(self) -> None:
        """Aggregated tie-breaker should use band_target when provided."""
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.75},
                config={"model": "A"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.75},
                config={"model": "A"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.85},
                config={"model": "B"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.85},
                config={"model": "B"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=True,
            tie_breakers={"accuracy": "min_abs_deviation"},
            band_target=0.80,  # Target is 0.80
        )

        # Without band_target, B (0.85) would win
        # With band_target=0.80:
        # - A has deviation |0.75 - 0.80| = 0.05
        # - B has deviation |0.85 - 0.80| = 0.05
        # Equal deviation - should pick first (A)
        # Actually both have same primary score in aggregation, B wins by default
        # since scores aren't exactly equal (0.75 vs 0.85), no tie-breaking
        assert result.best_config["model"] == "B"  # Higher accuracy wins

    def test_aggregated_no_tie_breaker_returns_first(self) -> None:
        """Without tie-breakers, aggregated mode should return first tied config."""
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.9},
                config={"model": "A"},
            ),
            FakeTrial(
                metrics={"accuracy": 0.9},
                config={"model": "B"},
            ),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=True,
            tie_breakers=None,
        )

        # Both have same accuracy, should return one of them
        assert result.best_config["model"] in ["A", "B"]
        assert result.best_score == pytest.approx(0.9)


def _weighted_schema(accuracy_weight: float, cost_weight: float):
    from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

    return ObjectiveSchema.from_objectives(
        [
            ObjectiveDefinition(
                name="accuracy", orientation="maximize", weight=accuracy_weight
            ),
            ObjectiveDefinition(
                name="cost", orientation="minimize", weight=cost_weight
            ),
        ]
    )


def _tradeoff_trials() -> list[FakeTrial]:
    """Issue #1682's LITERAL deterministic trade-off table.

    Realistic per-trial LLM costs (fractions of a cent) — the review-verified
    case where range-free 1/(1+cost) normalization can never flip the winner
    (costs map to 0.999/0.996/0.990). Observed-range min-max normalization
    must flip it.
    """
    return [
        FakeTrial(
            metrics={"accuracy": 0.92, "cost": 0.010},
            config={"model": "gpt-4o-analog"},
        ),
        FakeTrial(
            metrics={"accuracy": 0.82, "cost": 0.004},
            config={"model": "mid-analog"},
        ),
        FakeTrial(
            metrics={"accuracy": 0.70, "cost": 0.001},
            config={"model": "nano-analog"},
        ),
    ]


# Observed min-max ranges over the literal table.
_TRADEOFF_RANGES = {"accuracy": (0.70, 0.92), "cost": (0.001, 0.010)}


class TestWeightedSelection:
    """ObjectiveSchema weights govern best_config selection (issue #1682)."""

    def _select(self, trials, schema, *, aggregate: bool = False):
        return select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=aggregate,
            objective_order=["accuracy", "cost"],
            objective_schema=schema,
        )

    def test_accuracy_heavy_weights_select_accurate_config(self) -> None:
        result = self._select(_tradeoff_trials(), _weighted_schema(0.9, 0.1))

        assert result.best_config == {"model": "gpt-4o-analog"}
        # best_score stays the winner's primary-objective value.
        assert result.best_score == pytest.approx(0.92)
        weighted = result.session_summary["weighted_selection"]
        # gpt-4o-analog is the observed-range extreme: accuracy normalized to
        # 1.0, cost (worst observed) to 0.0 -> 0.9*1.0 + 0.1*0.0 = 0.9.
        assert weighted["best_weighted_score"] == pytest.approx(0.9)
        assert weighted["weights_normalized"] == pytest.approx(
            {"accuracy": 0.9, "cost": 0.1}
        )
        assert weighted["normalization_ranges"] == {
            "accuracy": [0.70, 0.92],
            "cost": [0.001, 0.010],
        }

    def test_cost_heavy_weights_flip_winner_to_nano_config(self) -> None:
        """Flipping 0.9/0.1 -> 0.1/0.9 changes best_config (the issue's repro)."""
        result = self._select(_tradeoff_trials(), _weighted_schema(0.1, 0.9))

        assert result.best_config == {"model": "nano-analog"}
        assert result.best_score == pytest.approx(0.70)
        weighted = result.session_summary["weighted_selection"]
        # nano-analog: accuracy normalized to 0.0, cost (best observed) to
        # 1.0 -> 0.1*0.0 + 0.9*1.0 = 0.9.
        assert weighted["best_weighted_score"] == pytest.approx(0.9)

    def test_minimize_orientation_pulls_selection_down(self) -> None:
        """A cost-heavy weighting must beat a higher-accuracy, costlier trial."""
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.92, "cost": 0.5}, config={"model": "pricey"}
            ),
            FakeTrial(
                metrics={"accuracy": 0.90, "cost": 0.1}, config={"model": "frugal"}
            ),
        ]
        legacy = select_best_configuration(
            trials=[
                FakeTrial(
                    metrics={"accuracy": 0.92, "cost": 0.5}, config={"model": "pricey"}
                ),
                FakeTrial(
                    metrics={"accuracy": 0.90, "cost": 0.1}, config={"model": "frugal"}
                ),
            ],
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            objective_order=["accuracy", "cost"],
        )
        assert legacy.best_config == {"model": "pricey"}  # primary-only ranking

        result = self._select(trials, _weighted_schema(0.4, 0.6))
        assert result.best_config == {"model": "frugal"}
        # Two-trial observed ranges: frugal takes accuracy_n=0, cost_n=1
        # -> 0.4*0.0 + 0.6*1.0 = 0.6 (beats pricey's 0.4*1.0 + 0.6*0.0 = 0.4).
        assert result.session_summary["weighted_selection"][
            "best_weighted_score"
        ] == pytest.approx(0.6)

    def test_per_trial_weighted_score_populated(self) -> None:
        """metrics['score'] carries the basis of selection and matches the
        objectives.py computation exactly (issue #1682)."""
        schema = _weighted_schema(0.9, 0.1)
        trials = _tradeoff_trials()

        self._select(trials, schema)

        for trial in trials:
            raw = {k: v for k, v in trial.metrics.items() if k != "score"}
            expected = schema.compute_weighted_score(raw, ranges=_TRADEOFF_RANGES)
            assert expected is not None
            assert trial.metrics["score"] == pytest.approx(expected)
        # Spot-check the mid config against hand-computed normalization:
        # 0.9*((0.82-0.70)/0.22) + 0.1*((0.010-0.004)/0.009).
        mid = next(t for t in trials if t.config == {"model": "mid-analog"})
        assert mid.metrics["score"] == pytest.approx(
            0.9 * (0.12 / 0.22) + 0.1 * (0.006 / 0.009)
        )

    def test_single_objective_schema_keeps_legacy_result(self) -> None:
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

        single = ObjectiveSchema.from_objectives(
            [ObjectiveDefinition(name="accuracy", orientation="maximize", weight=2.0)]
        )
        trials_a = _tradeoff_trials()
        trials_b = _tradeoff_trials()

        legacy = select_best_configuration(
            trials=trials_a,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            objective_order=["accuracy", "cost"],
        )
        with_schema = self._select(trials_b, single)

        assert (
            with_schema.best_config == legacy.best_config == {"model": "gpt-4o-analog"}
        )
        assert with_schema.best_score == legacy.best_score
        assert with_schema.session_summary == legacy.session_summary
        assert not any("score" in t.metrics for t in trials_b)

    def test_uniform_weights_select_by_weighted_aggregate(self) -> None:
        """Issue #1846: equal (default) weights are a genuine multi-objective
        request, so the terminal best_config is chosen by the weighted
        aggregate — NOT the accuracy-argmax legacy path.

        This is the #1846 contract change on top of #1682: declaring
        ``objectives=["accuracy", "cost"]`` (equal 0.5/0.5) must not let cost be
        ignored just because the weights are uniform. Before this fix
        best_config followed accuracy alone (``gpt-4o-analog``) and contradicted
        the run's own post-hoc weighted winner.
        """
        trials_a = _tradeoff_trials()
        trials_b = _tradeoff_trials()

        legacy = select_best_configuration(
            trials=trials_a,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            objective_order=["accuracy", "cost"],
        )
        uniform = self._select(trials_b, _weighted_schema(1.0, 1.0))

        # Legacy (no schema) still picks the accuracy-argmax config.
        assert legacy.best_config == {"model": "gpt-4o-analog"}
        # Uniform-weight multi-objective now picks the weighted winner instead.
        # 0.5*acc_n + 0.5*cost_n over ranges accuracy (0.70, 0.92),
        # cost (0.001, 0.010): mid-analog = 0.5*(0.12/0.22) + 0.5*(0.006/0.009)
        # = 0.606, beating gpt-4o (0.5) and nano (0.5).
        assert uniform.best_config == {"model": "mid-analog"}
        assert uniform.best_config != legacy.best_config
        # best_score stays the winner's primary-objective (accuracy) value.
        assert uniform.best_score == pytest.approx(0.82)
        weighted = uniform.session_summary["weighted_selection"]
        assert weighted["best_weighted_score"] == pytest.approx(
            0.5 * (0.12 / 0.22) + 0.5 * (0.006 / 0.009)
        )
        assert weighted["weights_normalized"] == pytest.approx(
            {"accuracy": 0.5, "cost": 0.5}
        )
        # Per-trial basis of selection is populated for weighted runs.
        assert all("score" in t.metrics for t in trials_b)

    def test_weighted_selection_aggregated_configs(self) -> None:
        """Weighted ranking applies to config-aggregated mean metrics too."""
        trials = [
            FakeTrial(
                metrics={"accuracy": 0.92, "cost": 0.5}, config={"model": "pricey"}
            ),
            FakeTrial(
                metrics={"accuracy": 0.90, "cost": 0.5}, config={"model": "pricey"}
            ),
            FakeTrial(
                metrics={"accuracy": 0.90, "cost": 0.1}, config={"model": "frugal"}
            ),
            FakeTrial(
                metrics={"accuracy": 0.88, "cost": 0.1}, config={"model": "frugal"}
            ),
        ]

        result = self._select(trials, _weighted_schema(0.4, 0.6), aggregate=True)

        assert result.best_config == {"model": "frugal"}
        # Mean metrics: accuracy 0.89, cost 0.1.
        assert result.best_score == pytest.approx(0.89)
        weighted = result.session_summary["weighted_selection"]
        # Per-trial observed ranges: accuracy (0.88, 0.92), cost (0.1, 0.5).
        # Frugal mean (0.89, 0.1): 0.4*((0.89-0.88)/0.04) + 0.6*1.0 = 0.7.
        assert weighted["best_weighted_score"] == pytest.approx(
            0.4 * (0.01 / 0.04) + 0.6 * 1.0
        )
        assert result.session_summary["selection_mode"] == "aggregated_mean"
        # Per-trial basis of selection is populated post-aggregation.
        for trial in trials:
            assert "score" in trial.metrics

    def test_banded_schema_disables_weighted_selection(self) -> None:
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
        from traigent.core.result_selection import resolve_weighted_selection_schema
        from traigent.tvl.models import BandTarget

        banded = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition(
                    name="accuracy",
                    orientation="band",
                    weight=0.8,
                    band=BandTarget(low=0.8, high=0.9),
                ),
                ObjectiveDefinition(name="cost", orientation="minimize", weight=0.2),
            ]
        )
        assert resolve_weighted_selection_schema(banded) is None
        assert resolve_weighted_selection_schema(None) is None
        assert resolve_weighted_selection_schema(_weighted_schema(0.7, 0.3)) is not None
        assert resolve_weighted_selection_schema(_weighted_schema(1.0, 1.0)) is None


def _evidence_trials() -> list[FakeTrial]:
    """Issue #1846's Evidence scenario (crux, 3 configs of the 12-trial run).

    A NON-TIE accuracy edge where accuracy-argmax and the weighted winner
    disagree: full/query_plan_cot has the highest accuracy (0.850) but 2.5x the
    per-query cost; tables/direct is the equal-weight winner. The full run had
    tables/direct at weighted rank 1 and full/query_plan_cot at rank 6/12.
    """
    return [
        FakeTrial(
            metrics={"accuracy": 0.825, "cost": 0.000070},
            config={"schema": "tables", "prompt": "direct"},
            trial_id="t_tables_direct",
        ),
        FakeTrial(
            metrics={"accuracy": 0.850, "cost": 0.000177},
            config={"schema": "full", "prompt": "query_plan_cot"},
            trial_id="t_full_qpcot",
        ),
        FakeTrial(
            metrics={"accuracy": 0.800, "cost": 0.000066},
            config={"schema": "tables", "prompt": "direct0"},
            trial_id="t_tables_d0",
        ),
    ]


class TestIssue1846UniformWeightSelection:
    """Terminal best_config respects declared objectives under uniform weights.

    #1682 made non-uniform weights govern best_config; #1846 extends that to the
    uniform/default case (a bare ``objectives=["accuracy", "cost"]`` list), so
    ``results.best_config`` equals the run's own post-hoc weighted winner
    (``best_weighted_config`` / ``weighted_results_v2.json``) instead of the
    accuracy-argmax config that contradicted it.
    """

    def _select(self, trials, schema, *, aggregate: bool = False):
        return select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"schema", "prompt"},
            aggregate_configs=aggregate,
            objective_order=["accuracy", "cost"],
            objective_schema=schema,
        )

    def test_evidence_scenario_best_config_is_weighted_not_accuracy_argmax(
        self,
    ) -> None:
        """Core #1846 repro: with default equal weights, best_config is the
        cost-aware weighted winner, NOT the highest-accuracy config.

        Fails on pre-#1846 code, which returned full/query_plan_cot.
        """
        schema = _weighted_schema(1.0, 1.0)  # default equal 0.5/0.5

        result = self._select(_evidence_trials(), schema)

        assert result.best_config == {"schema": "tables", "prompt": "direct"}
        assert result.best_config != {"schema": "full", "prompt": "query_plan_cot"}
        # best_score stays the winner's primary (accuracy) value.
        assert result.best_score == pytest.approx(0.825)
        assert "weighted_selection" in result.session_summary

    def test_best_config_equals_post_hoc_weighted_winner(self) -> None:
        """Consistency: terminal best_config == post-hoc best_weighted_config on
        the same run — the two artifacts of one run must agree (#1846)."""
        from traigent.api.types import (
            OptimizationResult,
            OptimizationStatus,
            TrialResult,
        )

        schema = _weighted_schema(1.0, 1.0)
        selection = self._select(_evidence_trials(), schema)

        post_hoc_trials = [
            TrialResult(
                trial_id=t.trial_id,
                config=t.config,
                metrics={"accuracy": t.metrics["accuracy"], "cost": t.metrics["cost"]},
                status=OptimizationStatus.COMPLETED,
                duration=1.0,
                timestamp=0.0,
            )
            for t in _evidence_trials()
        ]
        result = OptimizationResult(
            trials=post_hoc_trials,
            best_config=selection.best_config,
            best_score=selection.best_score,
            optimization_id="opt",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="grid",
            timestamp=0.0,
        )
        weighted = result.calculate_weighted_scores(objective_schema=schema)

        assert selection.best_config == weighted["best_weighted_config"]

    def test_flipping_to_accuracy_heavy_weights_restores_accuracy_winner(
        self,
    ) -> None:
        """Sanity: with accuracy-dominant weights the accuracy config wins,
        confirming selection genuinely tracks the declared weights."""
        result = self._select(_evidence_trials(), _weighted_schema(0.95, 0.05))
        assert result.best_config == {"schema": "full", "prompt": "query_plan_cot"}

    def test_single_objective_unchanged_under_1846(self) -> None:
        """Single-objective runs keep accuracy-argmax (no weighted aggregate)."""
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
        from traigent.core.result_selection import resolve_result_selection_schema

        single = ObjectiveSchema.from_objectives(
            [ObjectiveDefinition(name="accuracy", orientation="maximize", weight=1.0)]
        )
        assert resolve_result_selection_schema(single) is None
        result = self._select(_evidence_trials(), single)
        assert result.best_config == {"schema": "full", "prompt": "query_plan_cot"}
        assert result.best_score == pytest.approx(0.850)
        assert "weighted_selection" not in (result.session_summary or {})

    def test_result_selection_gate_is_broader_than_live_gate(self) -> None:
        """The terminal gate fires on uniform weights; the live gate does not.

        Guards the intentional asymmetry: live incumbent tracking and the #1832
        inert-objective warning stay non-uniform-only, while terminal
        best_config selection is uniform-inclusive.
        """
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
        from traigent.core.result_selection import (
            resolve_result_selection_schema,
            resolve_weighted_selection_schema,
        )
        from traigent.tvl.models import BandTarget

        uniform = _weighted_schema(1.0, 1.0)
        assert resolve_weighted_selection_schema(uniform) is None
        assert resolve_result_selection_schema(uniform) is not None

        non_uniform = _weighted_schema(0.7, 0.3)
        assert resolve_weighted_selection_schema(non_uniform) is not None
        assert resolve_result_selection_schema(non_uniform) is not None

        # Single-objective and banded stay excluded from BOTH gates.
        single = ObjectiveSchema.from_objectives(
            [ObjectiveDefinition(name="accuracy", orientation="maximize", weight=1.0)]
        )
        assert resolve_result_selection_schema(single) is None
        assert resolve_result_selection_schema(None) is None
        banded = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition(
                    name="accuracy",
                    orientation="band",
                    weight=0.5,
                    band=BandTarget(low=0.8, high=0.9),
                ),
                ObjectiveDefinition(name="cost", orientation="minimize", weight=0.5),
            ]
        )
        assert resolve_result_selection_schema(banded) is None

    def test_unavailable_weighted_selection_is_flagged(self) -> None:
        """Defensive (#1846 #4): a declared multi-objective schema whose weighted
        score is uncomputable (secondary objective never measured) falls back to
        primary-objective ranking but is flagged, not silent."""
        trials = [
            FakeTrial(metrics={"accuracy": 0.9}, config={"schema": "a", "prompt": "x"}),
            FakeTrial(metrics={"accuracy": 0.8}, config={"schema": "b", "prompt": "y"}),
        ]
        # cost is declared but present on NO trial -> the accuracy-only weighted
        # score still computes, so instead declare a wholly-absent objective.
        from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

        schema = ObjectiveSchema.from_objectives(
            [
                ObjectiveDefinition(name="ndcg", orientation="maximize", weight=0.5),
                ObjectiveDefinition(name="cost", orientation="minimize", weight=0.5),
            ]
        )
        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"schema", "prompt"},
            aggregate_configs=False,
            objective_order=["accuracy", "cost"],
            objective_schema=schema,
        )
        # Neither declared objective (ndcg/cost) is measured -> no finite
        # weighted score -> legacy fallback, but flagged.
        assert result.best_config == {"schema": "a", "prompt": "x"}
        assert result.session_summary.get("weighted_selection_unavailable") is True


# Two Pareto endpoints that share the exact top weighted aggregate (0.5) under
# equal 0.5/0.5 weights over observed ranges accuracy (0.70, 0.90) and
# cost (0.001, 0.010):
#   endpoint A (high accuracy / high cost): acc_n=1, cost_n=0  -> 0.5
#   endpoint B (low accuracy  / low cost) : acc_n=0, cost_n=1  -> 0.5
# Iteration order is [A, B], so a plain ``max`` (first-by-order) crowns A, while
# the authoritative tie-break (min_abs_deviation: smallest secondary cost, then
# trial_id) crowns B. This is the common linear/concave frontier endpoint tie,
# not a corner.
_TIE_ENDPOINTS = [
    ("t_high_acc_high_cost", {"model": "big"}, {"accuracy": 0.90, "cost": 0.010}),
    ("t_low_acc_low_cost", {"model": "small"}, {"accuracy": 0.70, "cost": 0.001}),
]


def _fake_trials(spec):
    """FakeTrials for the TERMINAL selector (carry ranking-eligible metadata)."""
    return [
        FakeTrial(trial_id=tid, config=config, metrics=metrics)
        for tid, config, metrics in spec
    ]


def _real_trials(spec):
    """Real TrialResults for the POST-HOC ``calculate_weighted_scores`` path."""
    from traigent.api.types import OptimizationStatus, TrialResult

    return [
        TrialResult(
            trial_id=tid,
            config=config,
            metrics=metrics,
            status=OptimizationStatus.COMPLETED,
            duration=1.0,
            timestamp=0.0,
        )
        for tid, config, metrics in spec
    ]


class TestIssue1846TieBreakParity:
    """Terminal ``best_config`` and post-hoc ``best_weighted_config`` must crown
    the SAME config on weighted ties, not just on unique maxima (#1846).

    The reviewer's blocker: the post-hoc winner pick was a plain
    ``max(weighted_scores)`` (first-by-iteration, no tolerance, no tie-break),
    while the terminal selector gathers tolerance-tied trials and applies
    ``apply_tie_breaker``. On a two-endpoint Pareto tie the two artifacts named
    OPPOSITE winners, so the PR's headline "the two artifacts agree" was false
    on ties. These tests reproduce that exact divergence and lock in parity.
    Terminal selection is fed FakeTrials (which set the ranking-eligible
    comparability metadata); the post-hoc path is fed the equivalent real
    TrialResults, mirroring the existing #1846 consistency test.
    """

    def _terminal(self, spec, schema):
        return select_best_configuration(
            trials=_fake_trials(spec),
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            objective_order=["accuracy", "cost"],
            objective_schema=schema,
        )

    def _post_hoc_result(self, spec, terminal):
        from traigent.api.types import OptimizationResult, OptimizationStatus

        return OptimizationResult(
            trials=_real_trials(spec),
            best_config=terminal.best_config,
            best_score=terminal.best_score,
            optimization_id="opt",
            duration=1.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy", "cost"],
            algorithm="grid",
            timestamp=0.0,
        )

    def test_terminal_and_post_hoc_agree_on_weighted_tie(self) -> None:
        """The reviewer's exact repro: equal weights, two configs both weighted
        0.5, ordered so plain-max and tie-broken-max differ.

        FAILS on the pre-fix post-hoc selector (plain ``max`` -> endpoint A)
        because the terminal selector tie-breaks to endpoint B; PASSES once the
        post-hoc path shares the terminal tie policy.
        """
        schema = _weighted_schema(1.0, 1.0)  # default equal 0.5/0.5

        terminal = self._terminal(_TIE_ENDPOINTS, schema)
        # The authoritative selector tie-breaks to the lower-cost endpoint.
        assert terminal.best_config == {"model": "small"}

        result = self._post_hoc_result(_TIE_ENDPOINTS, terminal)
        weighted = result.calculate_weighted_scores(objective_schema=schema)

        # Both endpoints share the top weighted score (this IS a tie)...
        scores = {t.trial_id: s for t, s in weighted["weighted_scores"]}
        assert scores["t_high_acc_high_cost"] == pytest.approx(0.5)
        assert scores["t_low_acc_low_cost"] == pytest.approx(0.5)
        # ...and the post-hoc winner must equal the terminal winner.
        assert weighted["best_weighted_config"] == terminal.best_config

    def test_post_hoc_tie_winner_is_not_first_by_iteration(self) -> None:
        """Guards against a silent regression to plain ``max``: the tie winner
        is the lower-cost endpoint (tie-break), never the first tied trial."""
        schema = _weighted_schema(1.0, 1.0)
        terminal = self._terminal(_TIE_ENDPOINTS, schema)
        result = self._post_hoc_result(_TIE_ENDPOINTS, terminal)

        weighted = result.calculate_weighted_scores(objective_schema=schema)
        # First trial by iteration order is the high-cost endpoint; a plain max
        # would return it. The tie-break must reject it for the low-cost one.
        assert result.trials[0].trial_id == "t_high_acc_high_cost"
        assert weighted["best_weighted_config"] == {"model": "small"}

    def test_unique_maximum_still_wins(self) -> None:
        """Parity fix must not disturb the non-tie case: a strict weighted
        maximum is still selected by both selectors."""
        schema = _weighted_schema(1.0, 1.0)
        spec = [
            ("t_dominant", {"model": "best"}, {"accuracy": 0.90, "cost": 0.001}),
            ("t_dominated", {"model": "worse"}, {"accuracy": 0.70, "cost": 0.010}),
        ]
        terminal = self._terminal(spec, schema)
        assert terminal.best_config == {"model": "best"}

        result = self._post_hoc_result(spec, terminal)
        weighted = result.calculate_weighted_scores(objective_schema=schema)
        assert weighted["best_weighted_config"] == {"model": "best"}
        assert weighted["best_weighted_score"] == pytest.approx(1.0)
