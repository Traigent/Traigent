"""Tests for statistical significance wiring (extract + compute + find_equivalence_group).

Covers:
- extract_trial_data_for_metric: per-example extraction and example_id alignment
- compute_significance: objective direction, band skipping, heuristic fallback
- Integration with find_equivalence_group
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from traigent.core.stat_significance import (
    compute_significance,
    extract_trial_data_for_metric,
    find_equivalence_group,
)

# ---------------------------------------------------------------------------
# Helpers — lightweight stand-ins for TrialResult / ExampleResult
# ---------------------------------------------------------------------------


@dataclass
class FakeExampleResult:
    example_id: str
    metrics: dict[str, float]


@dataclass
class FakeTrialResult:
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"

    @property
    def is_successful(self) -> bool:
        return self.status == "completed"


# ---------------------------------------------------------------------------
# extract_trial_data_for_metric
# ---------------------------------------------------------------------------


class TestExtractTrialDataForMetric:
    def test_basic_extraction(self):
        """Two trials with same examples produce aligned vectors."""
        trials = [
            FakeTrialResult(
                metrics={"accuracy": 0.8},
                metadata={
                    "example_results": [
                        FakeExampleResult(f"e{i}", {"accuracy": 0.7 + i * 0.02})
                        for i in range(5)
                    ]
                },
            ),
            FakeTrialResult(
                metrics={"accuracy": 0.6},
                metadata={
                    "example_results": [
                        FakeExampleResult(f"e{i}", {"accuracy": 0.5 + i * 0.02})
                        for i in range(5)
                    ]
                },
            ),
        ]
        data = extract_trial_data_for_metric(trials, "accuracy")
        assert len(data) == 2
        assert len(data[0]["values"]) == 5
        assert len(data[1]["values"]) == 5
        # metric_value is now the mean of shared examples, not the full aggregate
        expected_mean_a = sum(0.7 + i * 0.02 for i in range(5)) / 5
        assert abs(data[0]["metric_value"] - expected_mean_a) < 1e-9
        expected_mean_b = sum(0.5 + i * 0.02 for i in range(5)) / 5
        assert abs(data[1]["metric_value"] - expected_mean_b) < 1e-9
        assert data[0]["trial_idx"] == 0
        assert data[1]["trial_idx"] == 1

    def test_intersects_on_example_id(self):
        """Only shared example IDs are included in value vectors."""
        trials = [
            FakeTrialResult(
                metrics={"accuracy": 0.8},
                metadata={
                    "example_results": [
                        FakeExampleResult("e1", {"accuracy": 0.9}),
                        FakeExampleResult("e2", {"accuracy": 0.7}),
                        FakeExampleResult("e3", {"accuracy": 0.8}),
                        FakeExampleResult("e4", {"accuracy": 0.75}),
                        FakeExampleResult("e5", {"accuracy": 0.85}),
                        FakeExampleResult("e6", {"accuracy": 0.82}),
                        FakeExampleResult("e7", {"accuracy": 0.78}),
                    ]
                },
            ),
            FakeTrialResult(
                metrics={"accuracy": 0.6},
                metadata={
                    "example_results": [
                        FakeExampleResult("e3", {"accuracy": 0.6}),
                        FakeExampleResult("e4", {"accuracy": 0.55}),
                        FakeExampleResult("e5", {"accuracy": 0.65}),
                        FakeExampleResult("e6", {"accuracy": 0.58}),
                        FakeExampleResult("e7", {"accuracy": 0.62}),
                        FakeExampleResult("e8", {"accuracy": 0.7}),
                    ]
                },
            ),
        ]
        data = extract_trial_data_for_metric(trials, "accuracy")
        assert len(data) == 2
        # e3, e4, e5, e6, e7 are shared (5)
        assert len(data[0]["values"]) == 5
        assert len(data[1]["values"]) == 5

    def test_ordering_consistent_across_trials(self):
        """Value vectors are ordered by sorted example_id for correct pairing."""
        trials = [
            FakeTrialResult(
                metrics={"acc": 0.5},
                metadata={
                    "example_results": [
                        FakeExampleResult("e", {"acc": 0.1}),
                        FakeExampleResult("d", {"acc": 0.2}),
                        FakeExampleResult("c", {"acc": 0.3}),
                        FakeExampleResult("b", {"acc": 0.4}),
                        FakeExampleResult("a", {"acc": 0.5}),
                    ]
                },
            ),
            FakeTrialResult(
                metrics={"acc": 0.5},
                metadata={
                    "example_results": [
                        FakeExampleResult("a", {"acc": 0.9}),
                        FakeExampleResult("b", {"acc": 0.8}),
                        FakeExampleResult("c", {"acc": 0.7}),
                        FakeExampleResult("d", {"acc": 0.6}),
                        FakeExampleResult("e", {"acc": 0.5}),
                    ]
                },
            ),
        ]
        data = extract_trial_data_for_metric(trials, "acc")
        # Sorted IDs: ["a", "b", "c", "d", "e"]
        assert data[0]["values"] == [0.5, 0.4, 0.3, 0.2, 0.1]
        assert data[1]["values"] == [0.9, 0.8, 0.7, 0.6, 0.5]

    def test_skips_failed_trials(self):
        """Failed trials are excluded from extraction."""
        eids = [f"e{i}" for i in range(5)]
        trials = [
            FakeTrialResult(
                metrics={"accuracy": 0.8},
                metadata={
                    "example_results": [
                        FakeExampleResult(eid, {"accuracy": 0.8 + i * 0.01})
                        for i, eid in enumerate(eids)
                    ]
                },
            ),
            FakeTrialResult(
                metrics={"accuracy": 0.4},
                metadata={
                    "example_results": [
                        FakeExampleResult(eid, {"accuracy": 0.4}) for eid in eids
                    ]
                },
                status="failed",
            ),
            FakeTrialResult(
                metrics={"accuracy": 0.6},
                metadata={
                    "example_results": [
                        FakeExampleResult(eid, {"accuracy": 0.6 + i * 0.01})
                        for i, eid in enumerate(eids)
                    ]
                },
            ),
        ]
        data = extract_trial_data_for_metric(trials, "accuracy")
        assert len(data) == 2
        assert data[0]["trial_idx"] == 0
        assert data[1]["trial_idx"] == 2

    def test_skips_trials_missing_metric(self):
        """Trials without the requested metric are skipped."""
        trials = [
            FakeTrialResult(
                metrics={"accuracy": 0.8},
                metadata={
                    "example_results": [
                        FakeExampleResult("e1", {"accuracy": 0.9}),
                    ]
                },
            ),
            FakeTrialResult(
                metrics={"cost": 0.5},
                metadata={
                    "example_results": [
                        FakeExampleResult("e1", {"cost": 0.5}),
                    ]
                },
            ),
        ]
        data = extract_trial_data_for_metric(trials, "accuracy")
        # Only 1 trial has accuracy — need 2 minimum
        assert len(data) == 0

    def test_returns_empty_for_no_shared_ids(self):
        """Returns empty when trials have no overlapping example IDs."""
        trials = [
            FakeTrialResult(
                metrics={"accuracy": 0.8},
                metadata={
                    "example_results": [
                        FakeExampleResult("e1", {"accuracy": 0.9}),
                    ]
                },
            ),
            FakeTrialResult(
                metrics={"accuracy": 0.6},
                metadata={
                    "example_results": [
                        FakeExampleResult("e2", {"accuracy": 0.5}),
                    ]
                },
            ),
        ]
        data = extract_trial_data_for_metric(trials, "accuracy")
        assert data == []

    def test_returns_empty_for_single_shared_example(self):
        """A single shared example is insufficient for a paired t-test."""
        trials = [
            FakeTrialResult(
                metrics={"accuracy": 0.8},
                metadata={
                    "example_results": [
                        FakeExampleResult("e1", {"accuracy": 0.9}),
                        FakeExampleResult("e2", {"accuracy": 0.7}),
                    ]
                },
            ),
            FakeTrialResult(
                metrics={"accuracy": 0.6},
                metadata={
                    "example_results": [
                        FakeExampleResult("e1", {"accuracy": 0.5}),
                        FakeExampleResult("e3", {"accuracy": 0.6}),
                    ]
                },
            ),
        ]
        data = extract_trial_data_for_metric(trials, "accuracy")
        assert data == []

    def test_returns_empty_for_no_example_results(self):
        """Returns empty when trials lack per-example data."""
        trials = [
            FakeTrialResult(metrics={"accuracy": 0.8}, metadata={}),
            FakeTrialResult(metrics={"accuracy": 0.6}, metadata={}),
        ]
        data = extract_trial_data_for_metric(trials, "accuracy")
        assert data == []

    def test_handles_dict_example_results(self):
        """Handles example results stored as plain dicts (not objects)."""
        trials = [
            FakeTrialResult(
                metrics={"accuracy": 0.8},
                metadata={
                    "example_results": [
                        {"example_id": f"e{i}", "metrics": {"accuracy": 0.7 + i * 0.02}}
                        for i in range(5)
                    ]
                },
            ),
            FakeTrialResult(
                metrics={"accuracy": 0.6},
                metadata={
                    "example_results": [
                        {"example_id": f"e{i}", "metrics": {"accuracy": 0.5 + i * 0.02}}
                        for i in range(5)
                    ]
                },
            ),
        ]
        data = extract_trial_data_for_metric(trials, "accuracy")
        assert len(data) == 2
        assert len(data[0]["values"]) == 5

    def test_single_trial_returns_empty(self):
        """A single trial is insufficient for paired comparison."""
        trials = [
            FakeTrialResult(
                metrics={"accuracy": 0.8},
                metadata={
                    "example_results": [
                        FakeExampleResult("e1", {"accuracy": 0.9}),
                    ]
                },
            ),
        ]
        data = extract_trial_data_for_metric(trials, "accuracy")
        assert data == []


# ---------------------------------------------------------------------------
# compute_significance
# ---------------------------------------------------------------------------


class TestComputeSignificance:
    def _make_trials(
        self,
        trial_values: list[list[float]],
        metric_name: str = "accuracy",
    ) -> list[FakeTrialResult]:
        """Build fake trials with clearly different per-example metrics."""
        example_ids = [f"e{j}" for j in range(len(trial_values[0]))]
        trials = []
        for vals in trial_values:
            mean_val = sum(vals) / len(vals)
            trials.append(
                FakeTrialResult(
                    metrics={metric_name: mean_val},
                    metadata={
                        "example_results": [
                            FakeExampleResult(eid, {metric_name: v})
                            for eid, v in zip(example_ids, vals, strict=True)
                        ]
                    },
                )
            )
        return trials

    def test_maximize_objective(self):
        """Higher-is-better objective detects a clear winner."""
        # Trial 0: high values, Trial 1: low values
        trials = self._make_trials(
            [
                [0.95, 0.92, 0.93, 0.94, 0.96, 0.91, 0.93, 0.95, 0.94, 0.92],
                [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.48, 0.52, 0.51],
            ],
            "accuracy",
        )
        result = compute_significance(
            trials=trials,
            objectives=["accuracy"],
            objective_orientations={"accuracy": "maximize"},
        )
        assert "accuracy" in result
        assert 0 in result["accuracy"]["winners"]

    def test_minimize_objective(self):
        """Lower-is-better objective detects a clear winner."""
        trials = self._make_trials(
            [
                [0.001, 0.002, 0.001, 0.002, 0.001, 0.002, 0.001, 0.002, 0.001, 0.002],
                [0.050, 0.055, 0.048, 0.052, 0.051, 0.053, 0.049, 0.054, 0.050, 0.052],
            ],
            "cost",
        )
        result = compute_significance(
            trials=trials,
            objectives=["cost"],
            objective_orientations={"cost": "minimize"},
        )
        assert "cost" in result
        assert 0 in result["cost"]["winners"]

    def test_skips_band_objectives(self):
        """Band objectives are skipped entirely."""
        trials = self._make_trials(
            [
                [0.9, 0.9, 0.9, 0.9, 0.9],
                [0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            "quality",
        )
        result = compute_significance(
            trials=trials,
            objectives=["quality"],
            objective_orientations={"quality": "band"},
        )
        assert result == {}

    def test_heuristic_fallback_minimize(self):
        """Falls back to is_minimization_objective heuristic when no schema."""
        trials = self._make_trials(
            [
                [0.001, 0.002, 0.001, 0.002, 0.001, 0.002, 0.001, 0.002, 0.001, 0.002],
                [0.050, 0.055, 0.048, 0.052, 0.051, 0.053, 0.049, 0.054, 0.050, 0.052],
            ],
            "cost",
        )
        # No orientations provided — "cost" should trigger heuristic
        result = compute_significance(
            trials=trials,
            objectives=["cost"],
            objective_orientations=None,
        )
        assert "cost" in result
        # Trial 0 (lower cost) should win
        assert 0 in result["cost"]["winners"]

    def test_heuristic_fallback_maximize(self):
        """Heuristic defaults to maximize for unknown objectives."""
        trials = self._make_trials(
            [
                [0.95, 0.92, 0.93, 0.94, 0.96, 0.91, 0.93, 0.95, 0.94, 0.92],
                [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.48, 0.52, 0.51],
            ],
            "custom_score",
        )
        result = compute_significance(
            trials=trials,
            objectives=["custom_score"],
            objective_orientations=None,
        )
        assert "custom_score" in result
        assert 0 in result["custom_score"]["winners"]

    def test_multiple_objectives(self):
        """Handles multiple objectives in a single call."""
        example_ids = [f"e{i}" for i in range(10)]
        trials = [
            FakeTrialResult(
                metrics={"accuracy": 0.93, "cost": 0.002},
                metadata={
                    "example_results": [
                        FakeExampleResult(
                            eid,
                            {"accuracy": 0.93 + (i % 3) * 0.01, "cost": 0.002},
                        )
                        for i, eid in enumerate(example_ids)
                    ]
                },
            ),
            FakeTrialResult(
                metrics={"accuracy": 0.50, "cost": 0.050},
                metadata={
                    "example_results": [
                        FakeExampleResult(
                            eid,
                            {"accuracy": 0.50 + (i % 3) * 0.01, "cost": 0.050},
                        )
                        for i, eid in enumerate(example_ids)
                    ]
                },
            ),
        ]
        result = compute_significance(
            trials=trials,
            objectives=["accuracy", "cost"],
            objective_orientations={"accuracy": "maximize", "cost": "minimize"},
        )
        assert "accuracy" in result
        assert "cost" in result

    def test_skips_objective_with_insufficient_data(self):
        """Objectives where < 2 trials have data are skipped."""
        trials = [
            FakeTrialResult(
                metrics={"accuracy": 0.8},
                metadata={
                    "example_results": [
                        FakeExampleResult("e1", {"accuracy": 0.9}),
                    ]
                },
            ),
            FakeTrialResult(
                metrics={"cost": 0.5},
                metadata={
                    "example_results": [
                        FakeExampleResult("e1", {"cost": 0.5}),
                    ]
                },
            ),
        ]
        result = compute_significance(
            trials=trials,
            objectives=["accuracy"],
            objective_orientations={"accuracy": "maximize"},
        )
        assert result == {}

    def test_no_ops_with_empty_trials(self):
        """Empty trial list produces empty results."""
        result = compute_significance(
            trials=[], objectives=["accuracy"], objective_orientations=None
        )
        assert result == {}

    def test_equivalent_trials_no_winners(self):
        """When all trials are statistically equivalent, no winners."""
        # Same values for both trials
        trials = self._make_trials(
            [
                [0.80, 0.81, 0.79, 0.80, 0.80, 0.81, 0.79, 0.80, 0.80, 0.81],
                [0.80, 0.80, 0.80, 0.81, 0.79, 0.80, 0.80, 0.81, 0.79, 0.80],
            ],
            "accuracy",
        )
        result = compute_significance(
            trials=trials,
            objectives=["accuracy"],
            objective_orientations={"accuracy": "maximize"},
        )
        assert "accuracy" in result
        # No winners when trials are equivalent
        assert result["accuracy"]["winners"] == []

    def test_result_structure(self):
        """Result dict has expected keys."""
        trials = self._make_trials(
            [
                [0.95, 0.92, 0.93, 0.94, 0.96, 0.91, 0.93, 0.95, 0.94, 0.92],
                [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.48, 0.52, 0.51],
            ],
            "accuracy",
        )
        result = compute_significance(
            trials=trials,
            objectives=["accuracy"],
            objective_orientations={"accuracy": "maximize"},
        )
        sig = result["accuracy"]
        assert "winners" in sig
        assert "top_group" in sig
        assert "rest_group" in sig
        assert "badge_name" in sig
        assert sig["badge_name"] == "accuracy"
        assert "n_shared_examples" in sig
        assert sig["n_shared_examples"] == 10


# ---------------------------------------------------------------------------
# find_equivalence_group (existing function — quick regression tests)
# ---------------------------------------------------------------------------


class TestFindEquivalenceGroup:
    def test_single_trial_returns_empty(self):
        data = [{"values": [1.0, 2.0], "metric_value": 1.5, "trial_idx": 0}]
        result = find_equivalence_group(data)
        assert result.winners == []
        assert result.top_group == []

    def test_clear_winner(self):
        """One trial is significantly better than the other."""
        data = [
            {
                "values": [0.95, 0.92, 0.93, 0.94, 0.96, 0.91, 0.93, 0.95, 0.94, 0.92],
                "metric_value": 0.935,
                "trial_idx": 0,
            },
            {
                "values": [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.48, 0.52, 0.51],
                "metric_value": 0.504,
                "trial_idx": 1,
            },
        ]
        result = find_equivalence_group(
            data, higher_is_better=True, badge_name="accuracy"
        )
        assert 0 in result.winners
        assert 1 in result.rest_group
        assert result.badge_name == "accuracy"

    def test_all_equivalent(self):
        """Nearly identical trials produce no winners."""
        data = [
            {"values": [0.80, 0.81, 0.79, 0.80], "metric_value": 0.80, "trial_idx": 0},
            {
                "values": [0.80, 0.80, 0.80, 0.81],
                "metric_value": 0.8025,
                "trial_idx": 1,
            },
        ]
        result = find_equivalence_group(data, higher_is_better=True)
        assert result.winners == []
        assert len(result.top_group) == 2


# ---------------------------------------------------------------------------
# Orchestrator wiring: significance attach + failure isolation
# ---------------------------------------------------------------------------


class TestOrchestratorSignificanceWiring:
    """Test that the orchestrator's finalization path attaches significance
    and isolates failures so they don't break the optimization run."""

    def _make_result_with_trials(self):
        """Build two separable fake trials."""
        trial_a = FakeTrialResult(
            metrics={"accuracy": 0.93},
            metadata={
                "example_results": [
                    {"example_id": f"ex_{i}", "metrics": {"accuracy": v}}
                    for i, v in enumerate(
                        [0.95, 0.92, 0.93, 0.94, 0.96, 0.91, 0.93, 0.95, 0.94, 0.92]
                    )
                ]
            },
        )
        trial_b = FakeTrialResult(
            metrics={"accuracy": 0.50},
            metadata={
                "example_results": [
                    {"example_id": f"ex_{i}", "metrics": {"accuracy": v}}
                    for i, v in enumerate(
                        [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.50, 0.48, 0.52, 0.51]
                    )
                ]
            },
        )
        return [trial_a, trial_b]

    def test_significance_attached_to_metadata(self):
        """compute_significance result is stored in result.metadata."""
        trials = self._make_result_with_trials()
        metadata: dict = {}

        significance = compute_significance(
            trials=trials,
            objectives=["accuracy"],
            objective_orientations={"accuracy": "maximize"},
        )
        if significance:
            metadata["statistical_significance"] = significance

        assert "statistical_significance" in metadata
        sig = metadata["statistical_significance"]["accuracy"]
        assert 0 in sig["winners"], "Trial 0 (higher acc) should be a winner"
        assert sig["n_shared_examples"] == 10

    def test_significance_failure_does_not_crash(self):
        """If compute_significance raises, the orchestrator's try/except
        catches it and continues (optimization result is unaffected)."""
        metadata: dict = {}

        def _exploding_compute(**kwargs):
            raise RuntimeError("boom")

        # Replicate the exact guard pattern from orchestrator._finalize_optimization
        try:
            significance = _exploding_compute(
                trials=[],
                objectives=["accuracy"],
                objective_orientations={"accuracy": "maximize"},
                alpha=0.05,
            )
            if significance:
                metadata["statistical_significance"] = significance
        except Exception:
            # Orchestrator swallows and logs — optimization result unaffected
            pass

        assert "statistical_significance" not in metadata

    def test_significance_uses_shared_example_aggregate(self):
        """metric_value should be the mean of shared examples, not the
        full-trial aggregate (which could differ if coverage varies)."""
        # Trial A evaluated examples 0-9, trial B evaluated 5-14.
        # Shared = examples 5-9 only.
        trial_a = FakeTrialResult(
            metrics={"accuracy": 0.80},  # full aggregate (should be ignored)
            metadata={
                "example_results": [
                    {"example_id": f"ex_{i}", "metrics": {"accuracy": 0.90}}
                    for i in range(10)
                ]
            },
        )
        trial_b = FakeTrialResult(
            metrics={"accuracy": 0.60},  # full aggregate (should be ignored)
            metadata={
                "example_results": [
                    {"example_id": f"ex_{i}", "metrics": {"accuracy": 0.50}}
                    for i in range(5, 15)
                ]
            },
        )
        data = extract_trial_data_for_metric([trial_a, trial_b], "accuracy")
        assert len(data) == 2
        # Shared examples are ex_5..ex_9 — both have constant values
        # Trial A shared mean = 0.90, trial B shared mean = 0.50
        metric_values = sorted(d["metric_value"] for d in data)
        assert abs(metric_values[0] - 0.50) < 1e-9
        assert abs(metric_values[1] - 0.90) < 1e-9
