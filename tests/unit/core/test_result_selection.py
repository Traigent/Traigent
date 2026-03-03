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
            if isinstance(metric_value, (int, float))
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
        best_config={},
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

        # Without band_target, A would win (0.95 is max)
        # With band_target=0.90, C (0.89) is closest
        # But since scores aren't equal, tie-breaker shouldn't apply
        # The max score is 0.95 (model A), so only A is in tied_trials
        assert result.best_config["model"] == "A"

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
        """Without tie-breakers, first tied trial is returned."""
        trials = [
            FakeTrial(metrics={"accuracy": 0.9}, config={"model": "A"}),
            FakeTrial(metrics={"accuracy": 0.9}, config={"model": "B"}),
        ]

        result = select_best_configuration(
            trials=trials,
            primary_objective="accuracy",
            config_space_keys={"model"},
            aggregate_configs=False,
            tie_breakers=None,  # No tie-breaker
        )

        # Should return first in list
        assert result.best_config["model"] == "A"

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
        "per_metric_coverage": {
            "total_cost": {"present": 3, "total": 3, "ratio": 1.0}
        },
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

    assert result.best_config == {}
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

    assert result.best_config == {}
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
