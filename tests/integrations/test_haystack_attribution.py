"""Tests for Haystack attribution and insights module.

Tests Epic 6 stories:
- Story 6.1: Per-Node Metrics Collection
- Story 6.2: Quality Contribution Analysis
- Story 6.3: Cost and Latency Contribution Analysis
- Story 6.4: Parameter Sensitivity Analysis
- Story 6.5: Optimization Recommendations
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from traigent.integrations.haystack.attribution import (
    ComponentAttribution,
    NodeMetrics,
    compute_attribution,
    compute_cost_contribution,
    compute_latency_contribution,
    compute_quality_contribution,
    compute_sensitivity_scores,
    export_attribution,
    extract_node_metrics,
    generate_recommendation,
    get_attribution_ranked,
)


# Mock trial result for testing
@dataclass
class MockTrialResult:
    """Mock trial result for testing."""

    trial_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    node_metrics: dict[str, Any] | None = None
    is_successful: bool = True


class TestNodeMetrics:
    """Tests for NodeMetrics dataclass."""

    def test_default_values(self):
        """Test default values for NodeMetrics."""
        metrics = NodeMetrics(component_name="generator")
        assert metrics.component_name == "generator"
        assert metrics.invocation_count == 0
        assert metrics.total_latency_ms == 0.0
        assert metrics.total_cost == 0.0
        assert metrics.total_input_tokens == 0
        assert metrics.total_output_tokens == 0
        assert metrics.error_count == 0

    def test_avg_latency(self):
        """Test average latency calculation."""
        metrics = NodeMetrics(
            component_name="generator",
            invocation_count=5,
            total_latency_ms=500.0,
        )
        assert pytest.approx(metrics.avg_latency_ms) == 100.0

    def test_avg_latency_zero_invocations(self):
        """Test average latency with zero invocations."""
        metrics = NodeMetrics(component_name="generator")
        assert metrics.avg_latency_ms == 0.0

    def test_avg_cost(self):
        """Test average cost calculation."""
        metrics = NodeMetrics(
            component_name="generator",
            invocation_count=10,
            total_cost=0.5,
        )
        assert pytest.approx(metrics.avg_cost) == 0.05

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = NodeMetrics(
            component_name="generator",
            invocation_count=3,
            total_latency_ms=150.0,
            total_cost=0.03,
        )
        d = metrics.to_dict()
        assert d["component_name"] == "generator"
        assert d["invocation_count"] == 3
        assert pytest.approx(d["avg_latency_ms"]) == 50.0
        assert pytest.approx(d["avg_cost"]) == 0.01


class TestComponentAttribution:
    """Tests for ComponentAttribution dataclass."""

    def test_default_values(self):
        """Test default values for ComponentAttribution."""
        attr = ComponentAttribution(component_name="retriever")
        assert attr.component_name == "retriever"
        assert attr.quality_contribution == 0.0
        assert attr.cost_contribution == 0.0
        assert attr.latency_contribution == 0.0
        assert attr.sensitivity_scores == {}
        assert attr.most_sensitive_param is None
        assert attr.optimization_opportunity == "low"
        assert attr.recommendation is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        attr = ComponentAttribution(
            component_name="generator",
            quality_contribution=0.8,
            cost_contribution=0.6,
            latency_contribution=0.7,
            sensitivity_scores={"temperature": 0.9},
            most_sensitive_param="temperature",
            optimization_opportunity="high",
            recommendation="Expand temperature range",
        )
        d = attr.to_dict()
        assert d["component_name"] == "generator"
        assert pytest.approx(d["quality_contribution"]) == 0.8
        assert d["optimization_opportunity"] == "high"
        assert d["recommendation"] == "Expand temperature range"


class TestExtractNodeMetrics:
    """Tests for extract_node_metrics function."""

    def test_extract_from_node_metrics_dict(self):
        """Test extraction when node_metrics is a dict."""
        trial = MockTrialResult(
            trial_id="t1",
            config={},
            metrics={},
            node_metrics={
                "generator": {
                    "invocation_count": 5,
                    "total_latency_ms": 250.0,
                    "total_cost": 0.025,
                },
            },
        )

        result = extract_node_metrics(trial)
        assert "generator" in result
        assert result["generator"].invocation_count == 5
        assert pytest.approx(result["generator"].total_latency_ms) == 250.0

    def test_extract_from_node_metrics_objects(self):
        """Test extraction when node_metrics contains NodeMetrics objects."""
        trial = MockTrialResult(
            trial_id="t1",
            config={},
            metrics={},
            node_metrics={
                "retriever": NodeMetrics(
                    component_name="retriever",
                    invocation_count=10,
                    total_cost=0.01,
                ),
            },
        )

        result = extract_node_metrics(trial)
        assert "retriever" in result
        assert result["retriever"].invocation_count == 10

    def test_filter_by_component_names(self):
        """Test filtering by component names."""
        trial = MockTrialResult(
            trial_id="t1",
            config={},
            metrics={},
            node_metrics={
                "generator": {"invocation_count": 5},
                "retriever": {"invocation_count": 3},
            },
        )

        result = extract_node_metrics(trial, component_names=["generator"])
        assert "generator" in result
        assert "retriever" not in result


class TestComputeQualityContribution:
    """Tests for compute_quality_contribution function."""

    def test_empty_trials(self):
        """Test with empty trials list."""
        result = compute_quality_contribution([])
        assert result == {}

    def test_insufficient_trials(self):
        """Test with too few trials."""
        trials = [
            MockTrialResult("t1", {"gen.temp": 0.5}, {"accuracy": 0.8}),
            MockTrialResult("t2", {"gen.temp": 0.7}, {"accuracy": 0.85}),
        ]
        result = compute_quality_contribution(trials)
        assert result == {}

    def test_computes_contribution(self):
        """Test quality contribution computation."""
        trials = [
            MockTrialResult("t1", {"gen.temp": 0.3}, {"accuracy": 0.70}),
            MockTrialResult("t2", {"gen.temp": 0.5}, {"accuracy": 0.78}),
            MockTrialResult("t3", {"gen.temp": 0.7}, {"accuracy": 0.85}),
            MockTrialResult("t4", {"gen.temp": 0.9}, {"accuracy": 0.88}),
            MockTrialResult("t5", {"gen.temp": 0.6}, {"accuracy": 0.82}),
        ]

        result = compute_quality_contribution(trials, "accuracy")
        assert "gen" in result
        # Temperature correlates with accuracy, so should have contribution
        assert result["gen"] != 0.0

    def test_groups_by_component(self):
        """Test that parameters are grouped by component."""
        trials = [
            MockTrialResult(
                "t1", {"generator.temp": 0.3, "retriever.top_k": 3}, {"accuracy": 0.70}
            ),
            MockTrialResult(
                "t2", {"generator.temp": 0.5, "retriever.top_k": 5}, {"accuracy": 0.78}
            ),
            MockTrialResult(
                "t3", {"generator.temp": 0.7, "retriever.top_k": 7}, {"accuracy": 0.85}
            ),
            MockTrialResult(
                "t4", {"generator.temp": 0.9, "retriever.top_k": 10}, {"accuracy": 0.90}
            ),
            MockTrialResult(
                "t5", {"generator.temp": 0.6, "retriever.top_k": 6}, {"accuracy": 0.82}
            ),
        ]

        result = compute_quality_contribution(trials, "accuracy")
        assert "generator" in result
        assert "retriever" in result


class TestComputeCostContribution:
    """Tests for compute_cost_contribution function."""

    def test_empty_trials(self):
        """Test with empty trials list."""
        result = compute_cost_contribution([])
        assert result == {}

    def test_sums_to_one(self):
        """Test that contributions sum to 1.0."""
        trials = [
            MockTrialResult(
                "t1",
                {},
                {},
                node_metrics={
                    "generator": NodeMetrics("generator", total_cost=0.03),
                    "retriever": NodeMetrics("retriever", total_cost=0.01),
                },
            ),
            MockTrialResult(
                "t2",
                {},
                {},
                node_metrics={
                    "generator": NodeMetrics("generator", total_cost=0.04),
                    "retriever": NodeMetrics("retriever", total_cost=0.02),
                },
            ),
        ]

        result = compute_cost_contribution(trials)
        total = sum(result.values())
        assert pytest.approx(total) == 1.0


class TestComputeLatencyContribution:
    """Tests for compute_latency_contribution function."""

    def test_empty_trials(self):
        """Test with empty trials list."""
        result = compute_latency_contribution([])
        assert result == {}

    def test_sums_to_one(self):
        """Test that contributions sum to 1.0."""
        trials = [
            MockTrialResult(
                "t1",
                {},
                {},
                node_metrics={
                    "generator": NodeMetrics("generator", total_latency_ms=300),
                    "retriever": NodeMetrics("retriever", total_latency_ms=100),
                },
            ),
        ]

        result = compute_latency_contribution(trials)
        total = sum(result.values())
        assert pytest.approx(total) == 1.0

    def test_proportional_contribution(self):
        """Test that contributions are proportional."""
        trials = [
            MockTrialResult(
                "t1",
                {},
                {},
                node_metrics={
                    "generator": NodeMetrics("generator", total_latency_ms=750),
                    "retriever": NodeMetrics("retriever", total_latency_ms=250),
                },
            ),
        ]

        result = compute_latency_contribution(trials)
        assert pytest.approx(result["generator"]) == 0.75
        assert pytest.approx(result["retriever"]) == 0.25


class TestComputeSensitivityScores:
    """Tests for compute_sensitivity_scores function."""

    def test_empty_trials(self):
        """Test with empty trials list."""
        result = compute_sensitivity_scores([])
        assert result == {}

    def test_insufficient_trials(self):
        """Test with too few trials."""
        trials = [
            MockTrialResult("t1", {"gen.temp": 0.5}, {"accuracy": 0.8}),
        ]
        result = compute_sensitivity_scores(trials)
        assert result == {}

    def test_computes_sensitivity(self):
        """Test sensitivity computation with varied parameter values."""
        trials = [
            MockTrialResult("t1", {"gen.temp": 0.1}, {"accuracy": 0.60}),
            MockTrialResult("t2", {"gen.temp": 0.3}, {"accuracy": 0.70}),
            MockTrialResult("t3", {"gen.temp": 0.5}, {"accuracy": 0.80}),
            MockTrialResult("t4", {"gen.temp": 0.7}, {"accuracy": 0.85}),
            MockTrialResult("t5", {"gen.temp": 0.9}, {"accuracy": 0.88}),
        ]

        result = compute_sensitivity_scores(trials, "accuracy")
        assert "gen" in result
        assert "gen.temp" in result["gen"]
        # Strong correlation should yield high sensitivity
        assert result["gen"]["gen.temp"] > 0.5

    def test_zero_sensitivity_for_fixed_param(self):
        """Test zero sensitivity when parameter doesn't vary."""
        trials = [
            MockTrialResult("t1", {"gen.temp": 0.5}, {"accuracy": 0.70}),
            MockTrialResult("t2", {"gen.temp": 0.5}, {"accuracy": 0.80}),
            MockTrialResult("t3", {"gen.temp": 0.5}, {"accuracy": 0.75}),
            MockTrialResult("t4", {"gen.temp": 0.5}, {"accuracy": 0.85}),
            MockTrialResult("t5", {"gen.temp": 0.5}, {"accuracy": 0.78}),
        ]

        result = compute_sensitivity_scores(trials, "accuracy")
        # Fixed parameter has no variance, so sensitivity is 0
        if "gen" in result and "gen.temp" in result["gen"]:
            assert result["gen"]["gen.temp"] == 0.0


class TestGenerateRecommendation:
    """Tests for generate_recommendation function."""

    def test_low_opportunity_returns_none(self):
        """Test that low opportunity returns None."""
        result = generate_recommendation("generator", {"temp": 0.5}, "low")
        assert result is None

    def test_high_opportunity_temperature(self):
        """Test recommendation for temperature parameter."""
        result = generate_recommendation(
            "generator", {"generator.temperature": 0.9}, "high"
        )
        assert result is not None
        assert "temperature" in result.lower() or "range" in result.lower()

    def test_high_opportunity_model(self):
        """Test recommendation for model parameter."""
        result = generate_recommendation("generator", {"generator.model": 0.8}, "high")
        assert result is not None
        assert "model" in result.lower()

    def test_medium_opportunity(self):
        """Test recommendation for medium opportunity."""
        result = generate_recommendation(
            "retriever", {"retriever.top_k": 0.6}, "medium"
        )
        assert result is not None
        assert "moderate" in result.lower() or "potential" in result.lower()


class TestComputeAttribution:
    """Tests for compute_attribution function."""

    def test_empty_trials(self):
        """Test with empty trials list."""
        result = compute_attribution([])
        assert result == {}

    def test_full_attribution(self):
        """Test full attribution computation."""
        trials = [
            MockTrialResult(
                "t1",
                {"generator.temp": 0.3, "retriever.top_k": 3},
                {"accuracy": 0.70},
                node_metrics={
                    "generator": NodeMetrics(
                        "generator", total_cost=0.02, total_latency_ms=200
                    ),
                    "retriever": NodeMetrics(
                        "retriever", total_cost=0.01, total_latency_ms=100
                    ),
                },
            ),
            MockTrialResult(
                "t2",
                {"generator.temp": 0.5, "retriever.top_k": 5},
                {"accuracy": 0.78},
                node_metrics={
                    "generator": NodeMetrics(
                        "generator", total_cost=0.025, total_latency_ms=220
                    ),
                    "retriever": NodeMetrics(
                        "retriever", total_cost=0.012, total_latency_ms=110
                    ),
                },
            ),
            MockTrialResult(
                "t3",
                {"generator.temp": 0.7, "retriever.top_k": 7},
                {"accuracy": 0.85},
                node_metrics={
                    "generator": NodeMetrics(
                        "generator", total_cost=0.03, total_latency_ms=250
                    ),
                    "retriever": NodeMetrics(
                        "retriever", total_cost=0.015, total_latency_ms=120
                    ),
                },
            ),
            MockTrialResult(
                "t4",
                {"generator.temp": 0.9, "retriever.top_k": 10},
                {"accuracy": 0.90},
                node_metrics={
                    "generator": NodeMetrics(
                        "generator", total_cost=0.035, total_latency_ms=280
                    ),
                    "retriever": NodeMetrics(
                        "retriever", total_cost=0.018, total_latency_ms=130
                    ),
                },
            ),
            MockTrialResult(
                "t5",
                {"generator.temp": 0.6, "retriever.top_k": 6},
                {"accuracy": 0.82},
                node_metrics={
                    "generator": NodeMetrics(
                        "generator", total_cost=0.028, total_latency_ms=230
                    ),
                    "retriever": NodeMetrics(
                        "retriever", total_cost=0.013, total_latency_ms=115
                    ),
                },
            ),
        ]

        result = compute_attribution(trials, "accuracy")

        assert "generator" in result
        assert "retriever" in result
        assert isinstance(result["generator"], ComponentAttribution)
        assert result["generator"].component_name == "generator"


class TestGetAttributionRanked:
    """Tests for get_attribution_ranked function."""

    def test_empty_attribution(self):
        """Test with empty attribution."""
        result = get_attribution_ranked({})
        assert result == []

    def test_rank_by_quality(self):
        """Test ranking by quality contribution."""
        attribution = {
            "gen": ComponentAttribution("gen", quality_contribution=0.8),
            "ret": ComponentAttribution("ret", quality_contribution=0.3),
            "rer": ComponentAttribution("rer", quality_contribution=0.6),
        }

        result = get_attribution_ranked(attribution, by="quality")
        assert len(result) == 3
        assert result[0].component_name == "gen"
        assert result[1].component_name == "rer"
        assert result[2].component_name == "ret"

    def test_rank_by_opportunity(self):
        """Test ranking by optimization opportunity."""
        attribution = {
            "gen": ComponentAttribution("gen", optimization_opportunity="medium"),
            "ret": ComponentAttribution("ret", optimization_opportunity="high"),
            "rer": ComponentAttribution("rer", optimization_opportunity="low"),
        }

        result = get_attribution_ranked(attribution, by="opportunity")
        assert result[0].component_name == "ret"  # high first
        assert result[1].component_name == "gen"  # medium second
        assert result[2].component_name == "rer"  # low last


class TestExportAttribution:
    """Tests for export_attribution function."""

    def test_export_dict(self):
        """Test export to dict format."""
        attribution = {
            "generator": ComponentAttribution(
                "generator",
                quality_contribution=0.8,
                optimization_opportunity="high",
            ),
        }

        result = export_attribution(attribution, format="dict")
        assert "components" in result
        assert "summary" in result
        assert result["summary"]["high_opportunity"] == 1

    def test_export_json(self):
        """Test export to JSON format."""
        attribution = {
            "generator": ComponentAttribution("generator"),
        }

        result = export_attribution(attribution, format="json")
        assert isinstance(result, str)
        data = json.loads(result)
        assert "components" in data

    def test_export_json_to_file(self):
        """Test export to JSON file."""
        attribution = {
            "generator": ComponentAttribution("generator"),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attribution.json"
            export_attribution(attribution, format="json", output_path=str(path))

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
                assert "components" in data

    def test_export_csv(self):
        """Test export to CSV format."""
        attribution = {
            "generator": ComponentAttribution(
                "generator",
                quality_contribution=0.8,
                cost_contribution=0.6,
            ),
            "retriever": ComponentAttribution(
                "retriever",
                quality_contribution=0.4,
                cost_contribution=0.4,
            ),
        }

        result = export_attribution(attribution, format="csv")
        assert isinstance(result, str)
        lines = result.strip().split("\n")
        assert len(lines) == 3  # Header + 2 rows
        assert "component" in lines[0]

    def test_export_invalid_format(self):
        """Test export with invalid format."""
        with pytest.raises(ValueError, match="Unsupported format"):
            export_attribution({}, format="xml")
