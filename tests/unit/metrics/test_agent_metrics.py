"""Unit tests for agent-specific metric utilities.

Tests the per-agent metric computation, aggregation, and reference-free
metric selection utilities for multi-agent workflow evaluation.
"""

from __future__ import annotations

import pytest

from traigent.metrics.agent_metrics import (
    AGENT_PERFORMANCE_METRICS,
    AGENT_QUALITY_METRICS,
    ALL_RAGAS_METRICS,
    REFERENCE_FREE_RAGAS_METRICS,
    REFERENCE_REQUIRED_RAGAS_METRICS,
    AgentMetricsSummary,
    MultiAgentMetricsSummary,
    aggregate_agent_metrics,
    build_agent_objectives,
    compute_per_agent_metrics,
    extract_namespaced_config_for_agent,
    get_metrics_for_available_data,
    get_reference_free_metrics,
    validate_agent_id,
    validate_agent_metrics,
)


class TestMetricConstants:
    """Test metric constant definitions."""

    def test_reference_free_metrics_defined(self):
        """Verify reference-free RAGAS metrics are defined."""
        assert len(REFERENCE_FREE_RAGAS_METRICS) >= 2
        assert "answer_relevancy" in REFERENCE_FREE_RAGAS_METRICS
        assert "faithfulness" in REFERENCE_FREE_RAGAS_METRICS

    def test_reference_required_metrics_defined(self):
        """Verify reference-required RAGAS metrics are defined."""
        assert len(REFERENCE_REQUIRED_RAGAS_METRICS) >= 2
        assert "context_precision" in REFERENCE_REQUIRED_RAGAS_METRICS
        assert "answer_similarity" in REFERENCE_REQUIRED_RAGAS_METRICS

    def test_all_ragas_metrics_is_union(self):
        """ALL_RAGAS_METRICS should be union of reference-free and required."""
        all_set = set(ALL_RAGAS_METRICS)
        expected = set(REFERENCE_FREE_RAGAS_METRICS) | set(
            REFERENCE_REQUIRED_RAGAS_METRICS
        )
        assert all_set == expected

    def test_agent_performance_metrics_defined(self):
        """Verify agent performance metrics are defined."""
        assert "cost" in AGENT_PERFORMANCE_METRICS
        assert "latency_ms" in AGENT_PERFORMANCE_METRICS
        assert "total_tokens" in AGENT_PERFORMANCE_METRICS

    def test_agent_quality_metrics_defined(self):
        """Verify agent quality metrics are defined."""
        assert "tool_call_accuracy" in AGENT_QUALITY_METRICS
        assert "task_completion" in AGENT_QUALITY_METRICS


class TestAgentMetricsSummary:
    """Test AgentMetricsSummary dataclass."""

    def test_basic_creation(self):
        """Create basic agent metrics summary."""
        summary = AgentMetricsSummary(
            agent_id="grader",
            metrics={"cost": 0.002, "latency_ms": 150.0},
            sample_count=10,
        )
        assert summary.agent_id == "grader"
        assert summary.metrics["cost"] == 0.002
        assert summary.sample_count == 10

    def test_to_measures_dict_with_agent_id(self):
        """to_measures_dict includes agent prefix."""
        summary = AgentMetricsSummary(
            agent_id="grader",
            metrics={"cost": 0.002, "latency_ms": 150.0},
        )
        measures = summary.to_measures_dict()

        assert "grader_cost" in measures
        assert "grader_latency_ms" in measures
        assert measures["grader_cost"] == 0.002

    def test_to_measures_dict_with_prefix(self):
        """to_measures_dict with custom prefix."""
        summary = AgentMetricsSummary(
            agent_id="grader",
            metrics={"cost": 0.002},
        )
        measures = summary.to_measures_dict(prefix="langfuse_")

        assert "langfuse_grader_cost" in measures

    def test_to_measures_dict_empty_agent_id(self):
        """to_measures_dict with empty agent_id (global metrics)."""
        summary = AgentMetricsSummary(
            agent_id="",
            metrics={"total_cost": 0.006},
        )
        measures = summary.to_measures_dict()

        assert "total_cost" in measures


class TestMultiAgentMetricsSummary:
    """Test MultiAgentMetricsSummary dataclass."""

    def test_basic_creation(self):
        """Create multi-agent metrics summary."""
        grader = AgentMetricsSummary(
            agent_id="grader",
            metrics={"cost": 0.002, "latency_ms": 150.0},
        )
        generator = AgentMetricsSummary(
            agent_id="generator",
            metrics={"cost": 0.004, "latency_ms": 300.0},
        )

        summary = MultiAgentMetricsSummary(
            agents={"grader": grader, "generator": generator},
            global_metrics={"total_cost": 0.006},
            workflow_id="wf-123",
        )

        assert len(summary.agents) == 2
        assert summary.global_metrics["total_cost"] == 0.006
        assert summary.workflow_id == "wf-123"

    def test_to_measures_dict_includes_all(self):
        """to_measures_dict includes both global and per-agent metrics."""
        grader = AgentMetricsSummary(
            agent_id="grader",
            metrics={"cost": 0.002},
        )
        summary = MultiAgentMetricsSummary(
            agents={"grader": grader},
            global_metrics={"total_cost": 0.006},
        )

        measures = summary.to_measures_dict()

        assert "total_cost" in measures
        assert "grader_cost" in measures

    def test_to_measures_dict_without_totals(self):
        """to_measures_dict can exclude global metrics."""
        grader = AgentMetricsSummary(
            agent_id="grader",
            metrics={"cost": 0.002},
        )
        summary = MultiAgentMetricsSummary(
            agents={"grader": grader},
            global_metrics={"total_cost": 0.006},
        )

        measures = summary.to_measures_dict(include_totals=False)

        assert "total_cost" not in measures
        assert "grader_cost" in measures

    def test_get_agent(self):
        """get_agent returns correct agent summary."""
        grader = AgentMetricsSummary(agent_id="grader", metrics={"cost": 0.002})
        summary = MultiAgentMetricsSummary(agents={"grader": grader})

        assert summary.get_agent("grader") == grader
        assert summary.get_agent("unknown") is None


class TestComputePerAgentMetrics:
    """Test compute_per_agent_metrics function."""

    def test_extracts_metrics_for_all_agents(self):
        """Extract metrics for multiple agents."""
        measures = {
            "grader_cost": 0.002,
            "grader_latency_ms": 150,
            "generator_cost": 0.004,
            "generator_latency_ms": 300,
            "total_cost": 0.006,
        }

        per_agent = compute_per_agent_metrics(measures, ["grader", "generator"])

        assert "grader" in per_agent
        assert "generator" in per_agent
        assert per_agent["grader"]["cost"] == 0.002
        assert per_agent["grader"]["latency_ms"] == 150
        assert per_agent["generator"]["cost"] == 0.004

    def test_filters_by_metric_names(self):
        """Filter to specific metrics only."""
        measures = {
            "grader_cost": 0.002,
            "grader_latency_ms": 150,
            "grader_tokens": 100,
        }

        per_agent = compute_per_agent_metrics(
            measures, ["grader"], metric_names=["cost"]
        )

        assert per_agent["grader"] == {"cost": 0.002}

    def test_handles_missing_agent(self):
        """Agents with no metrics are not included."""
        measures = {"grader_cost": 0.002}

        per_agent = compute_per_agent_metrics(measures, ["grader", "unknown"])

        assert "grader" in per_agent
        assert "unknown" not in per_agent

    def test_empty_measures(self):
        """Empty measures returns empty result."""
        per_agent = compute_per_agent_metrics({}, ["grader"])
        assert per_agent == {}


class TestAggregateAgentMetrics:
    """Test aggregate_agent_metrics function."""

    def test_sum_aggregation(self):
        """Sum aggregation across agents."""
        per_agent = {
            "grader": {"cost": 0.002, "latency_ms": 150},
            "generator": {"cost": 0.004, "latency_ms": 300},
        }

        totals = aggregate_agent_metrics(per_agent, aggregation="sum")

        assert totals["total_cost"] == 0.006
        assert totals["total_latency_ms"] == 450

    def test_mean_aggregation(self):
        """Mean aggregation across agents."""
        per_agent = {
            "grader": {"cost": 0.002},
            "generator": {"cost": 0.004},
        }

        totals = aggregate_agent_metrics(per_agent, aggregation="mean")

        assert totals["mean_cost"] == 0.003

    def test_max_aggregation(self):
        """Max aggregation across agents."""
        per_agent = {
            "grader": {"latency_ms": 150},
            "generator": {"latency_ms": 300},
        }

        totals = aggregate_agent_metrics(per_agent, aggregation="max")

        assert totals["max_latency_ms"] == 300

    def test_min_aggregation(self):
        """Min aggregation across agents."""
        per_agent = {
            "grader": {"latency_ms": 150},
            "generator": {"latency_ms": 300},
        }

        totals = aggregate_agent_metrics(per_agent, aggregation="min")

        assert totals["min_latency_ms"] == 150

    def test_unknown_aggregation_raises(self):
        """Unknown aggregation method raises ValueError."""
        per_agent = {"grader": {"cost": 0.002}}

        with pytest.raises(ValueError, match="Unknown aggregation"):
            aggregate_agent_metrics(per_agent, aggregation="invalid")

    def test_empty_input(self):
        """Empty input returns empty result."""
        assert aggregate_agent_metrics({}) == {}


class TestGetReferenceFreeMetrics:
    """Test get_reference_free_metrics function."""

    def test_returns_all_when_no_filter(self):
        """Returns all reference-free metrics when no filter."""
        metrics = get_reference_free_metrics()
        assert set(metrics) == set(REFERENCE_FREE_RAGAS_METRICS)

    def test_filters_to_requested(self):
        """Filters to intersection with requested metrics."""
        metrics = get_reference_free_metrics(["faithfulness", "context_precision"])
        assert metrics == ["faithfulness"]

    def test_empty_when_no_overlap(self):
        """Returns empty when no overlap with reference-free metrics."""
        metrics = get_reference_free_metrics(["context_precision", "context_recall"])
        assert metrics == []


class TestGetMetricsForAvailableData:
    """Test get_metrics_for_available_data function."""

    def test_llm_only(self):
        """With only LLM, can compute LLM-based metrics."""
        metrics = get_metrics_for_available_data(
            has_reference=False,
            has_contexts=False,
            has_llm=True,
        )
        assert "answer_relevancy" in metrics
        assert "faithfulness" in metrics
        assert "context_precision" not in metrics

    def test_reference_only(self):
        """With only reference, can compute similarity metrics."""
        metrics = get_metrics_for_available_data(
            has_reference=True,
            has_contexts=False,
            has_llm=False,
        )
        assert "answer_similarity" in metrics
        assert "context_precision" not in metrics

    def test_reference_and_contexts(self):
        """With reference and contexts, can compute context metrics."""
        metrics = get_metrics_for_available_data(
            has_reference=True,
            has_contexts=True,
            has_llm=False,
        )
        assert "answer_similarity" in metrics
        assert "context_precision" in metrics
        assert "context_recall" in metrics

    def test_nothing_available(self):
        """With nothing available, no metrics can be computed."""
        metrics = get_metrics_for_available_data(
            has_reference=False,
            has_contexts=False,
            has_llm=False,
        )
        assert metrics == []

    def test_filters_by_requested(self):
        """Filters result by requested metrics."""
        metrics = get_metrics_for_available_data(
            has_reference=True,
            has_contexts=True,
            has_llm=True,
            requested_metrics=["faithfulness", "context_precision"],
        )
        assert set(metrics) == {"faithfulness", "context_precision"}


class TestBuildAgentObjectives:
    """Test build_agent_objectives function."""

    def test_includes_cost_and_latency_by_default(self):
        """Includes cost and latency metrics by default."""
        objectives = build_agent_objectives(["grader", "generator"])

        assert "total_cost" in objectives
        assert "total_latency_ms" in objectives
        assert "grader_cost" in objectives
        assert "generator_latency_ms" in objectives

    def test_excludes_quality_by_default(self):
        """Quality metrics excluded by default."""
        objectives = build_agent_objectives(["grader"])

        # Should not have quality metrics
        assert "grader_accuracy" not in objectives

    def test_includes_quality_when_requested(self):
        """Quality metrics included when requested."""
        objectives = build_agent_objectives(["grader"], include_quality=True)

        assert "grader_accuracy" in objectives
        assert "grader_quality_score" in objectives

    def test_excludes_totals_when_requested(self):
        """Can exclude total metrics."""
        objectives = build_agent_objectives(["grader"], include_totals=False)

        assert "total_cost" not in objectives
        assert "grader_cost" in objectives


class TestExtractNamespacedConfigForAgent:
    """Test extract_namespaced_config_for_agent function."""

    def test_extracts_agent_config(self):
        """Extract configuration for specific agent."""
        config = {
            "grader__temperature": 0.3,
            "grader__model": "gpt-4o-mini",
            "generator__temperature": 0.7,
            "max_retries": 3,
        }

        grader_config = extract_namespaced_config_for_agent(config, "grader")

        assert grader_config == {"temperature": 0.3, "model": "gpt-4o-mini"}

    def test_ignores_global_params(self):
        """Global params (no namespace) are not extracted."""
        config = {
            "grader__temperature": 0.3,
            "max_retries": 3,
        }

        grader_config = extract_namespaced_config_for_agent(config, "grader")

        assert "max_retries" not in grader_config

    def test_empty_when_no_matching_params(self):
        """Returns empty dict when no params match agent."""
        config = {"generator__temperature": 0.7}

        grader_config = extract_namespaced_config_for_agent(config, "grader")

        assert grader_config == {}

    def test_handles_nested_param_names(self):
        """Handles param names with underscores."""
        config = {"grader__max_tokens": 100}

        grader_config = extract_namespaced_config_for_agent(config, "grader")

        assert grader_config == {"max_tokens": 100}


class TestValidateAgentMetrics:
    """Test validate_agent_metrics function."""

    def test_valid_when_all_present(self):
        """Returns valid when all required metrics present."""
        measures = {
            "grader_cost": 0.002,
            "grader_latency_ms": 150,
            "generator_cost": 0.004,
            "generator_latency_ms": 300,
        }

        valid, missing = validate_agent_metrics(
            measures,
            ["grader", "generator"],
            ["cost", "latency_ms"],
        )

        assert valid is True
        assert missing == []

    def test_invalid_when_missing(self):
        """Returns invalid when metrics missing."""
        measures = {"grader_cost": 0.002, "generator_cost": 0.004}

        valid, missing = validate_agent_metrics(
            measures,
            ["grader", "generator"],
            ["cost", "latency_ms"],
        )

        assert valid is False
        assert "grader_latency_ms" in missing
        assert "generator_latency_ms" in missing

    def test_empty_agents(self):
        """Empty agents list is valid."""
        valid, missing = validate_agent_metrics({}, [], ["cost"])

        assert valid is True
        assert missing == []

    def test_empty_required_metrics(self):
        """Empty required metrics is valid."""
        measures = {"grader_cost": 0.002}

        valid, missing = validate_agent_metrics(measures, ["grader"], [])

        assert valid is True
        assert missing == []


class TestValidateAgentId:
    """Test validate_agent_id function."""

    def test_valid_simple_name(self):
        """Simple names are valid."""
        assert validate_agent_id("grader") is True
        assert validate_agent_id("generator") is True

    def test_valid_with_underscores(self):
        """Names with underscores are valid."""
        assert validate_agent_id("financial_advisor") is True
        assert validate_agent_id("grader_v2") is True
        assert validate_agent_id("_private") is True

    def test_valid_with_numbers(self):
        """Names with numbers (not at start) are valid."""
        assert validate_agent_id("agent1") is True
        assert validate_agent_id("v2_grader") is True

    def test_invalid_with_hyphen(self):
        """Names with hyphens are invalid."""
        assert validate_agent_id("grader-v2") is False
        assert validate_agent_id("my-agent") is False

    def test_invalid_starts_with_digit(self):
        """Names starting with digits are invalid."""
        assert validate_agent_id("123agent") is False
        assert validate_agent_id("1st_grader") is False

    def test_invalid_empty(self):
        """Empty string is invalid."""
        assert validate_agent_id("") is False

    def test_invalid_with_dot(self):
        """Names with dots are invalid."""
        assert validate_agent_id("grader.v2") is False

    def test_invalid_with_space(self):
        """Names with spaces are invalid."""
        assert validate_agent_id("my agent") is False


class TestAgentIdValidationInDataclass:
    """Test agent ID validation in AgentMetricsSummary."""

    def test_valid_agent_id_accepted(self):
        """Valid agent IDs are accepted."""
        summary = AgentMetricsSummary(agent_id="grader", metrics={"cost": 0.002})
        assert summary.agent_id == "grader"

    def test_empty_agent_id_accepted(self):
        """Empty agent ID is accepted (for global metrics)."""
        summary = AgentMetricsSummary(agent_id="", metrics={"total_cost": 0.006})
        assert summary.agent_id == ""

    def test_invalid_agent_id_raises(self):
        """Invalid agent IDs raise ValueError."""
        with pytest.raises(ValueError, match="Invalid agent_id"):
            AgentMetricsSummary(agent_id="grader-v2", metrics={"cost": 0.002})

    def test_invalid_agent_id_with_digit_start_raises(self):
        """Agent IDs starting with digit raise ValueError."""
        with pytest.raises(ValueError, match="Invalid agent_id"):
            AgentMetricsSummary(agent_id="123agent", metrics={"cost": 0.002})


class TestAgentIdValidationInFunctions:
    """Test agent ID validation in functions."""

    def test_compute_per_agent_metrics_rejects_invalid(self):
        """compute_per_agent_metrics rejects invalid agent IDs."""
        measures = {"grader_cost": 0.002}
        with pytest.raises(ValueError, match="Invalid agent IDs"):
            compute_per_agent_metrics(measures, ["grader-v2"])

    def test_build_agent_objectives_rejects_invalid(self):
        """build_agent_objectives rejects invalid agent IDs."""
        with pytest.raises(ValueError, match="Invalid agent IDs"):
            build_agent_objectives(["valid_agent", "invalid-agent"])

    def test_validate_agent_metrics_rejects_invalid(self):
        """validate_agent_metrics rejects invalid agent IDs."""
        measures = {"grader_cost": 0.002}
        with pytest.raises(ValueError, match="Invalid agent IDs"):
            validate_agent_metrics(measures, ["123invalid"], ["cost"])

    def test_functions_accept_valid_ids(self):
        """Functions accept valid agent IDs."""
        measures = {"grader_cost": 0.002, "generator_cost": 0.004}

        # compute_per_agent_metrics
        result = compute_per_agent_metrics(measures, ["grader", "generator"])
        assert "grader" in result

        # build_agent_objectives
        objectives = build_agent_objectives(["grader", "generator"])
        assert "grader_cost" in objectives

        # validate_agent_metrics
        valid, missing = validate_agent_metrics(
            measures, ["grader", "generator"], ["cost"]
        )
        assert valid is True
