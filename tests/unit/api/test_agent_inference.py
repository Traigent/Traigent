"""Unit tests for agent configuration inference."""

from __future__ import annotations

import pytest

from traigent.api.agent_inference import (
    _find_model_key,
    _humanize,
    build_agent_configuration,
    extract_parameter_agents,
)
from traigent.api.parameter_ranges import Choices, IntRange, Range
from traigent.api.types import AgentConfiguration, AgentDefinition, GlobalConfiguration


class TestBuildAgentConfiguration:
    """Tests for build_agent_configuration function."""

    def test_explicit_agents_override_all_inference(self):
        """Explicit agent definitions should override all inference."""
        explicit_agents = {
            "financial": AgentDefinition(
                display_name="Financial Agent",
                parameter_keys=["model", "temperature"],
                measure_ids=["accuracy"],
            )
        }

        result = build_agent_configuration(
            configuration_space={"model": ["gpt-4"], "temperature": (0, 1)},
            explicit_agents=explicit_agents,
            agent_prefixes=["other"],  # Should be ignored
        )

        assert result is not None
        assert result.auto_inferred is False
        assert "financial" in result.agents
        assert result.agents["financial"].display_name == "Financial Agent"
        assert result.agents["financial"].parameter_keys == ["model", "temperature"]

    def test_explicit_agents_with_global_measures(self):
        """Explicit agents should support global_measures."""
        result = build_agent_configuration(
            configuration_space={"model": ["gpt-4"]},
            explicit_agents={
                "router": AgentDefinition(
                    display_name="Router",
                    parameter_keys=["model"],
                    measure_ids=["routing_accuracy"],
                )
            },
            global_measures=["total_cost", "total_latency"],
        )

        assert result is not None
        assert result.global_config is not None
        assert result.global_config.measure_ids == ["total_cost", "total_latency"]

    def test_parameter_agents_from_range_objects(self):
        """Parameters with agent attribute should be grouped correctly."""
        result = build_agent_configuration(
            configuration_space={
                "model": Choices(["gpt-4"]),
                "temperature": Range(0.0, 1.0),
                "legal_model": Choices(["claude"]),
            },
            parameter_agents={
                "model": "financial",
                "temperature": "financial",
                "legal_model": "legal",
            },
        )

        assert result is not None
        assert len(result.agents) == 2
        assert "financial" in result.agents
        assert "legal" in result.agents
        assert sorted(result.agents["financial"].parameter_keys) == sorted(
            ["model", "temperature"]
        )
        assert result.agents["legal"].parameter_keys == ["legal_model"]
        assert result.auto_inferred is True

    def test_prefix_based_inference(self):
        """Parameters should be grouped by prefix when agent_prefixes is set."""
        result = build_agent_configuration(
            configuration_space={
                "financial_model": Choices(["gpt-4"]),
                "financial_temperature": Range(0.0, 1.0),
                "legal_model": Choices(["claude"]),
                "legal_temperature": Range(0.0, 0.5),
            },
            agent_prefixes=["financial", "legal"],
        )

        assert result is not None
        assert len(result.agents) == 2
        assert "financial" in result.agents
        assert "legal" in result.agents
        assert sorted(result.agents["financial"].parameter_keys) == sorted(
            ["financial_model", "financial_temperature"]
        )
        assert sorted(result.agents["legal"].parameter_keys) == sorted(
            ["legal_model", "legal_temperature"]
        )

    def test_invalid_prefix_raises_error(self):
        """Invalid prefix should raise ValueError."""
        with pytest.raises(ValueError, match="no parameters start with"):
            build_agent_configuration(
                configuration_space={"model": Choices(["gpt-4"])},
                agent_prefixes=["nonexistent"],
            )

    def test_single_agent_returns_none(self):
        """Single agent experiments should return None (no grouping needed)."""
        # All params assigned to same agent
        result = build_agent_configuration(
            configuration_space={"model": Choices(["gpt-4"]), "temp": Range(0, 1)},
            parameter_agents={"model": "single", "temp": "single"},
        )

        assert result is None

    def test_no_multi_agent_config_returns_none(self):
        """No multi-agent config should return None."""
        result = build_agent_configuration(
            configuration_space={"model": Choices(["gpt-4"]), "temp": Range(0, 1)},
        )

        assert result is None

    def test_global_params_collected(self):
        """Unassigned parameters should go to global_config."""
        result = build_agent_configuration(
            configuration_space={
                "financial_model": Choices(["gpt-4"]),
                "legal_model": Choices(["claude"]),
                "max_retries": IntRange(1, 5),  # Not assigned to any agent
            },
            agent_prefixes=["financial", "legal"],
        )

        assert result is not None
        assert result.global_config is not None
        assert "max_retries" in result.global_config.parameter_keys

    def test_agent_measures_mapping(self):
        """Agent measures should be assigned correctly."""
        result = build_agent_configuration(
            configuration_space={
                "financial_model": Choices(["gpt-4"]),
                "legal_model": Choices(["claude"]),
            },
            agent_prefixes=["financial", "legal"],
            agent_measures={
                "financial": ["financial_accuracy", "financial_latency"],
                "legal": ["legal_accuracy"],
            },
            global_measures=["total_cost"],
        )

        assert result is not None
        assert result.agents["financial"].measure_ids == [
            "financial_accuracy",
            "financial_latency",
        ]
        assert result.agents["legal"].measure_ids == ["legal_accuracy"]
        assert result.global_config is not None
        assert result.global_config.measure_ids == ["total_cost"]

    def test_primary_model_detection(self):
        """Primary model key should be detected from parameter names."""
        result = build_agent_configuration(
            configuration_space={
                "financial_model": Choices(["gpt-4"]),
                "financial_temperature": Range(0.0, 1.0),
                "legal_llm": Choices(["claude"]),
            },
            agent_prefixes=["financial", "legal"],
        )

        assert result is not None
        assert result.agents["financial"].primary_model == "financial_model"
        # "llm" doesn't match "_model" pattern
        assert result.agents["legal"].primary_model is None

    def test_display_name_humanization(self):
        """Display names should be humanized from agent IDs."""
        result = build_agent_configuration(
            configuration_space={
                "financial_agent_model": Choices(["gpt-4"]),
                "legal_team_model": Choices(["claude"]),
            },
            agent_prefixes=["financial_agent", "legal_team"],
        )

        assert result is not None
        assert result.agents["financial_agent"].display_name == "Financial Agent"
        assert result.agents["legal_team"].display_name == "Legal Team"

    def test_agent_ordering(self):
        """Agents should have consistent ordering."""
        result = build_agent_configuration(
            configuration_space={
                "zebra_model": Choices(["gpt-4"]),
                "alpha_model": Choices(["claude"]),
                "beta_model": Choices(["gemini"]),
            },
            agent_prefixes=["zebra", "alpha", "beta"],
        )

        assert result is not None
        # Sorted alphabetically: alpha=0, beta=1, zebra=2
        assert result.agents["alpha"].order == 0
        assert result.agents["beta"].order == 1
        assert result.agents["zebra"].order == 2

    def test_unknown_agent_measures_raises_error(self):
        """agent_measures referencing unknown agents should raise ValueError."""
        with pytest.raises(ValueError, match="unknown agents.*nonexistent"):
            build_agent_configuration(
                configuration_space={
                    "financial_model": Choices(["gpt-4"]),
                    "legal_model": Choices(["claude"]),
                },
                agent_prefixes=["financial", "legal"],
                agent_measures={
                    "financial": ["accuracy"],
                    "nonexistent": ["latency"],  # Not a valid agent
                },
            )


class TestExtractParameterAgents:
    """Tests for extract_parameter_agents function."""

    def test_extract_agents_from_range_objects(self):
        """Should extract agent from Range objects."""
        config_space = {
            "model": Choices(["gpt-4"], agent="financial"),
            "temperature": Range(0.0, 1.0, agent="financial"),
            "max_tokens": IntRange(100, 4096),  # No agent
        }

        result = extract_parameter_agents(config_space)

        assert result == {"model": "financial", "temperature": "financial"}

    def test_empty_when_no_agents(self):
        """Should return empty dict when no agents set."""
        config_space = {
            "model": Choices(["gpt-4"]),
            "temperature": Range(0.0, 1.0),
        }

        result = extract_parameter_agents(config_space)

        assert result == {}

    def test_mixed_range_types(self):
        """Should work with all range types."""
        config_space = {
            "model": Choices(["gpt-4"], agent="router"),
            "temperature": Range(0.0, 1.0, agent="financial"),
            "tokens": IntRange(100, 4096, agent="financial"),
            "learning_rate": Range(
                1e-5, 1e-1, log=True
            ),  # LogRange equivalent, no agent
        }

        result = extract_parameter_agents(config_space)

        assert result == {
            "model": "router",
            "temperature": "financial",
            "tokens": "financial",
        }


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_humanize_simple(self):
        """Test humanize with simple names."""
        assert _humanize("financial") == "Financial"
        assert _humanize("router") == "Router"

    def test_humanize_underscores(self):
        """Test humanize with underscores."""
        assert _humanize("financial_agent") == "Financial Agent"
        assert _humanize("my_llm_router") == "My Llm Router"

    def test_find_model_key_found(self):
        """Test finding model key."""
        assert _find_model_key(["temperature", "model", "max_tokens"]) == "model"
        assert _find_model_key(["financial_model", "temp"]) == "financial_model"
        assert _find_model_key(["MODEL_NAME", "temp"]) == "MODEL_NAME"

    def test_find_model_key_not_found(self):
        """Test when no model key exists."""
        assert _find_model_key(["temperature", "max_tokens"]) is None
        assert _find_model_key([]) is None


class TestAgentConfigurationSerialization:
    """Tests for AgentConfiguration to_dict and from_dict."""

    def test_to_dict_basic(self):
        """Test basic serialization."""
        config = AgentConfiguration(
            agents={
                "financial": AgentDefinition(
                    display_name="Financial",
                    parameter_keys=["model"],
                    measure_ids=["accuracy"],
                )
            }
        )

        result = config.to_dict()

        assert result["version"] == "1.0"
        assert result["auto_inferred"] is False
        assert "financial" in result["agents"]
        assert result["agents"]["financial"]["display_name"] == "Financial"

    def test_to_dict_with_all_fields(self):
        """Test serialization with all optional fields."""
        from traigent.api.types import AgentMeta

        config = AgentConfiguration(
            agents={
                "financial": AgentDefinition(
                    display_name="Financial Agent",
                    parameter_keys=["model", "temp"],
                    measure_ids=["accuracy"],
                    primary_model="model",
                    order=0,
                    agent_type="llm",
                    meta=AgentMeta(
                        color="#4299E1",
                        icon="robot",
                        description="Handles financial queries",
                    ),
                )
            },
            global_config=GlobalConfiguration(
                parameter_keys=["max_retries"],
                measure_ids=["total_cost"],
                order=99,
            ),
            auto_inferred=True,
        )

        result = config.to_dict()

        assert result["auto_inferred"] is True
        agent = result["agents"]["financial"]
        assert agent["primary_model"] == "model"
        assert agent["order"] == 0
        assert agent["agent_type"] == "llm"
        assert agent["meta"]["color"] == "#4299E1"
        assert result["global"]["parameter_keys"] == ["max_retries"]
        assert result["global"]["order"] == 99

    def test_from_dict_roundtrip(self):
        """Test from_dict can reconstruct from to_dict."""
        from traigent.api.types import AgentMeta

        original = AgentConfiguration(
            agents={
                "financial": AgentDefinition(
                    display_name="Financial",
                    parameter_keys=["model"],
                    measure_ids=["accuracy"],
                    primary_model="model",
                    order=0,
                    agent_type="llm",
                    meta=AgentMeta(color="#4299E1"),
                )
            },
            global_config=GlobalConfiguration(
                parameter_keys=["retries"],
                measure_ids=["cost"],
            ),
            auto_inferred=True,
        )

        serialized = original.to_dict()
        reconstructed = AgentConfiguration.from_dict(serialized)

        assert reconstructed.version == original.version
        assert reconstructed.auto_inferred == original.auto_inferred
        assert "financial" in reconstructed.agents
        assert reconstructed.agents["financial"].display_name == "Financial"
        assert reconstructed.agents["financial"].meta is not None
        assert reconstructed.agents["financial"].meta.color == "#4299E1"
        assert reconstructed.global_config is not None
        assert reconstructed.global_config.parameter_keys == ["retries"]
