import pytest

from traigent.core.result_selection import SelectionResult, select_best_configuration


class FakeTrial:
    """Minimal stand-in for TrialResult used in selection logic tests."""

    def __init__(
        self,
        *,
        metrics: dict[str, float] | None = None,
        config: dict[str, object] | None = None,
        success: bool = True,
    ) -> None:
        self.metrics = metrics or {}
        self.config = config or {}
        self.is_successful = success
        self.metadata: dict[str, object] = {}

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
        best_config={}, best_score=0.0, session_summary=None
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
    assert result.session_summary is None


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
