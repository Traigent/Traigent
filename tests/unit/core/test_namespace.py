"""Unit tests for namespace parsing utilities.

Tests the multi-agent parameter namespace parsing functions that use
double underscore (__) as the delimiter.
"""

from __future__ import annotations

from traigent.core.namespace import (
    NAMESPACE_DELIMITER,
    ParsedNamespace,
    build_per_agent_objectives,
    create_namespaced_param,
    extract_agent_metrics,
    extract_agents_from_config,
    flatten_agent_config,
    group_params_by_agent,
    is_namespaced,
    parse_namespace,
    parse_namespaced_param,
    sanitize_metric_name,
)


class TestNamespaceDelimiter:
    """Test namespace delimiter constant."""

    def test_delimiter_is_double_underscore(self):
        """Verify delimiter is __ to avoid MeasuresDict conflicts."""
        assert NAMESPACE_DELIMITER == "__"


class TestIsNamespaced:
    """Test is_namespaced function."""

    def test_double_underscore_is_namespaced(self):
        """Parameters with __ are namespaced."""
        assert is_namespaced("grader__temperature") is True
        assert is_namespaced("generator__model") is True

    def test_single_underscore_not_namespaced(self):
        """Parameters with single underscore are NOT namespaced."""
        assert is_namespaced("total_cost") is False
        assert is_namespaced("latency_ms") is False
        assert is_namespaced("grader_cost") is False  # MeasuresDict format

    def test_no_underscore_not_namespaced(self):
        """Parameters without underscore are not namespaced."""
        assert is_namespaced("temperature") is False
        assert is_namespaced("model") is False

    def test_multiple_double_underscores(self):
        """Parameters with multiple __ are still namespaced."""
        assert is_namespaced("grader__nested__param") is True


class TestParseNamespacedParam:
    """Test parse_namespaced_param function."""

    def test_basic_namespaced_param(self):
        """Parse basic namespaced parameter."""
        agent, param = parse_namespaced_param("grader__temperature")
        assert agent == "grader"
        assert param == "temperature"

    def test_global_param(self):
        """Global (non-namespaced) parameters return empty agent."""
        agent, param = parse_namespaced_param("temperature")
        assert agent == ""
        assert param == "temperature"

    def test_single_underscore_not_split(self):
        """Single underscores are not treated as delimiters."""
        agent, param = parse_namespaced_param("total_cost")
        assert agent == ""
        assert param == "total_cost"

    def test_multiple_delimiters_split_first_only(self):
        """Multiple __ only splits on first occurrence."""
        agent, param = parse_namespaced_param("grader__nested__param")
        assert agent == "grader"
        assert param == "nested__param"

    def test_complex_agent_name(self):
        """Agent names can contain underscores."""
        agent, param = parse_namespaced_param("financial_advisor__model")
        assert agent == "financial_advisor"
        assert param == "model"


class TestParseNamespace:
    """Test parse_namespace function (returns ParsedNamespace)."""

    def test_returns_parsed_namespace_object(self):
        """parse_namespace returns ParsedNamespace dataclass."""
        result = parse_namespace("grader__temperature")
        assert isinstance(result, ParsedNamespace)
        assert result.agent == "grader"
        assert result.name == "temperature"
        assert result.original == "grader__temperature"
        assert result.is_namespaced is True

    def test_global_param(self):
        """Global parameters have empty agent."""
        result = parse_namespace("temperature")
        assert result.agent == ""
        assert result.name == "temperature"
        assert result.is_namespaced is False


class TestCreateNamespacedParam:
    """Test create_namespaced_param function."""

    def test_creates_namespaced_param(self):
        """Create namespaced parameter from agent and param name."""
        result = create_namespaced_param("grader", "temperature")
        assert result == "grader__temperature"

    def test_roundtrip(self):
        """Parse and create should roundtrip correctly."""
        original = "generator__model"
        agent, param = parse_namespaced_param(original)
        recreated = create_namespaced_param(agent, param)
        assert recreated == original


class TestExtractAgentsFromConfig:
    """Test extract_agents_from_config function."""

    def test_extracts_unique_agents(self):
        """Extract unique agents from config space."""
        config = {
            "grader__temperature": [0.0, 0.3],
            "grader__model": ["gpt-4"],
            "generator__temperature": [0.5, 0.7],
            "max_retries": [3],  # Global
        }
        agents = extract_agents_from_config(config)
        assert agents == {"grader", "generator"}

    def test_empty_config(self):
        """Empty config returns empty set."""
        assert extract_agents_from_config({}) == set()

    def test_all_global_params(self):
        """Config with only global params returns empty set."""
        config = {"temperature": [0.5], "max_retries": [3]}
        assert extract_agents_from_config(config) == set()


class TestGroupParamsByAgent:
    """Test group_params_by_agent function."""

    def test_groups_by_agent(self):
        """Group parameters by their agent prefix."""
        config = {
            "grader__temperature": [0.0, 0.3],
            "grader__model": ["gpt-4"],
            "generator__temperature": [0.5, 0.7],
            "max_retries": [3],  # Global
        }
        grouped = group_params_by_agent(config)

        assert "grader" in grouped
        assert grouped["grader"] == {"temperature": [0.0, 0.3], "model": ["gpt-4"]}

        assert "generator" in grouped
        assert grouped["generator"] == {"temperature": [0.5, 0.7]}

        assert "" in grouped  # Global params
        assert grouped[""] == {"max_retries": [3]}

    def test_empty_config(self):
        """Empty config returns empty dict."""
        assert group_params_by_agent({}) == {}


class TestFlattenAgentConfig:
    """Test flatten_agent_config function."""

    def test_flattens_to_namespaced(self):
        """Flatten agent-grouped config to namespaced format."""
        agent_configs = {
            "grader": {"temperature": 0.3, "model": "gpt-4"},
            "generator": {"temperature": 0.7},
            "": {"max_retries": 3},  # Global
        }
        flattened = flatten_agent_config(agent_configs)

        assert flattened["grader__temperature"] == 0.3
        assert flattened["grader__model"] == "gpt-4"
        assert flattened["generator__temperature"] == 0.7
        assert flattened["max_retries"] == 3

    def test_exclude_global(self):
        """Can exclude global params from flattening."""
        agent_configs = {
            "grader": {"temperature": 0.3},
            "": {"max_retries": 3},
        }
        flattened = flatten_agent_config(agent_configs, exclude_global=True)

        assert "grader__temperature" in flattened
        assert "max_retries" not in flattened

    def test_roundtrip(self):
        """Group and flatten should roundtrip correctly."""
        original = {
            "grader__temperature": [0.3],
            "generator__model": ["gpt-4"],
            "global_param": [1],
        }
        grouped = group_params_by_agent(original)
        flattened = flatten_agent_config(grouped)

        # Note: global_param has empty agent, so it stays as is
        assert set(flattened.keys()) == set(original.keys())


class TestSanitizeMetricName:
    """Test sanitize_metric_name function."""

    def test_replaces_dots_and_hyphens(self):
        """Dots and hyphens are replaced with underscores."""
        assert sanitize_metric_name("grader-v2.cost") == "grader_v2_cost"
        assert sanitize_metric_name("model-gpt-4") == "model_gpt_4"

    def test_preserves_valid_names(self):
        """Valid names are unchanged."""
        assert sanitize_metric_name("grader_cost") == "grader_cost"
        assert sanitize_metric_name("total_latency_ms") == "total_latency_ms"

    def test_handles_numeric_start(self):
        """Names starting with numbers get underscore prefix."""
        assert sanitize_metric_name("123invalid") == "_123invalid"
        assert sanitize_metric_name("4xx_errors") == "_4xx_errors"


class TestExtractAgentMetrics:
    """Test extract_agent_metrics function."""

    def test_extracts_agent_metrics(self):
        """Extract metrics for a specific agent."""
        measures = {
            "total_cost": 0.01,
            "grader_cost": 0.003,
            "grader_latency_ms": 150,
            "generator_cost": 0.007,
            "generator_latency_ms": 300,
        }
        grader_metrics = extract_agent_metrics(measures, "grader")

        assert grader_metrics == {
            "grader_cost": 0.003,
            "grader_latency_ms": 150,
        }

    def test_filter_by_suffix(self):
        """Can filter by metric suffix."""
        measures = {
            "grader_cost": 0.003,
            "grader_latency_ms": 150,
            "grader_tokens": 100,
        }
        cost_only = extract_agent_metrics(measures, "grader", metric_suffix="cost")

        assert cost_only == {"grader_cost": 0.003}

    def test_empty_for_missing_agent(self):
        """Returns empty dict for agents not in measures."""
        measures = {"grader_cost": 0.003}
        assert extract_agent_metrics(measures, "unknown") == {}


class TestBuildPerAgentObjectives:
    """Test build_per_agent_objectives function."""

    def test_builds_objectives_with_totals(self):
        """Build objectives including total metrics."""
        objectives = build_per_agent_objectives(
            agents=["grader", "generator"],
            metrics=["cost", "latency_ms"],
            include_totals=True,
        )

        expected = [
            "total_cost",
            "total_latency_ms",
            "grader_cost",
            "grader_latency_ms",
            "generator_cost",
            "generator_latency_ms",
        ]
        assert objectives == expected

    def test_builds_objectives_without_totals(self):
        """Build objectives without total metrics."""
        objectives = build_per_agent_objectives(
            agents=["grader"],
            metrics=["cost"],
            include_totals=False,
        )

        assert objectives == ["grader_cost"]

    def test_empty_agents(self):
        """Empty agents list returns only totals if enabled."""
        objectives = build_per_agent_objectives(
            agents=[],
            metrics=["cost"],
            include_totals=True,
        )
        assert objectives == ["total_cost"]
