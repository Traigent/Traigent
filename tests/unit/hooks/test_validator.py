"""Unit tests for traigent.hooks.validator.

Tests for agent configuration validation against constraints defined in traigent.yml,
including model validation, cost estimation, and pre-push hook validation.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from traigent.hooks.config import (
    HooksConfig,
)
from traigent.hooks.validator import (
    DEFAULT_TOKENS_PER_QUERY,
    MODEL_COST_PER_1K,
    AgentInfo,
    AgentValidator,
    ValidationIssue,
    ValidationResult,
    validate_agents_for_push,
)


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_default_initialization(self) -> None:
        """Test ValidationIssue initializes with required fields."""
        issue = ValidationIssue(
            severity="error",
            code="TEST_ERROR",
            message="Test error message",
        )
        assert issue.severity == "error"
        assert issue.code == "TEST_ERROR"
        assert issue.message == "Test error message"
        assert issue.suggestion is None

    def test_initialization_with_suggestion(self) -> None:
        """Test ValidationIssue initializes with suggestion field."""
        issue = ValidationIssue(
            severity="warning",
            code="TEST_WARNING",
            message="Test warning",
            suggestion="Fix by doing X",
        )
        assert issue.severity == "warning"
        assert issue.code == "TEST_WARNING"
        assert issue.message == "Test warning"
        assert issue.suggestion == "Fix by doing X"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_default_initialization(self) -> None:
        """Test ValidationResult initializes with minimal required fields."""
        result = ValidationResult(
            function_name="test_func",
            is_valid=True,
        )
        assert result.function_name == "test_func"
        assert result.is_valid is True
        assert result.issues == []
        assert result.warnings == []
        assert result.models_found == []
        assert result.estimated_cost_per_query is None

    def test_initialization_with_all_fields(self) -> None:
        """Test ValidationResult initializes with all fields provided."""
        issues = [
            ValidationIssue(severity="error", code="ERR1", message="Error 1"),
        ]
        warnings = [
            ValidationIssue(severity="warning", code="WARN1", message="Warning 1"),
        ]
        result = ValidationResult(
            function_name="test_func",
            is_valid=False,
            issues=issues,
            warnings=warnings,
            models_found=["gpt-4o", "claude-3-sonnet"],
            estimated_cost_per_query=0.005,
        )
        assert result.function_name == "test_func"
        assert result.is_valid is False
        assert len(result.issues) == 1
        assert len(result.warnings) == 1
        assert result.models_found == ["gpt-4o", "claude-3-sonnet"]
        assert result.estimated_cost_per_query == 0.005

    def test_has_errors_returns_true_for_errors(self) -> None:
        """Test has_errors property returns True when errors present."""
        issues = [
            ValidationIssue(severity="error", code="ERR1", message="Error 1"),
        ]
        result = ValidationResult(
            function_name="test_func",
            is_valid=False,
            issues=issues,
        )
        assert result.has_errors is True

    def test_has_errors_returns_false_without_errors(self) -> None:
        """Test has_errors property returns False when no errors."""
        result = ValidationResult(
            function_name="test_func",
            is_valid=True,
        )
        assert result.has_errors is False

    def test_has_warnings_returns_true_for_warnings(self) -> None:
        """Test has_warnings property returns True when warnings present."""
        warnings = [
            ValidationIssue(severity="warning", code="WARN1", message="Warning 1"),
        ]
        result = ValidationResult(
            function_name="test_func",
            is_valid=True,
            warnings=warnings,
        )
        assert result.has_warnings is True

    def test_has_warnings_returns_false_without_warnings(self) -> None:
        """Test has_warnings property returns False when no warnings."""
        result = ValidationResult(
            function_name="test_func",
            is_valid=True,
        )
        assert result.has_warnings is False

    def test_get_summary_for_passed_validation(self) -> None:
        """Test get_summary returns PASSED for valid result."""
        result = ValidationResult(
            function_name="test_func",
            is_valid=True,
        )
        assert result.get_summary() == "test_func: PASSED"

    def test_get_summary_for_passed_with_warnings(self) -> None:
        """Test get_summary includes warning count when present."""
        warnings = [
            ValidationIssue(severity="warning", code="WARN1", message="Warning 1"),
            ValidationIssue(severity="warning", code="WARN2", message="Warning 2"),
        ]
        result = ValidationResult(
            function_name="test_func",
            is_valid=True,
            warnings=warnings,
        )
        assert result.get_summary() == "test_func: PASSED (2 warnings)"

    def test_get_summary_for_failed_validation(self) -> None:
        """Test get_summary returns FAILED with issue count."""
        issues = [
            ValidationIssue(severity="error", code="ERR1", message="Error 1"),
            ValidationIssue(severity="error", code="ERR2", message="Error 2"),
            ValidationIssue(severity="error", code="ERR3", message="Error 3"),
        ]
        result = ValidationResult(
            function_name="test_func",
            is_valid=False,
            issues=issues,
        )
        assert result.get_summary() == "test_func: FAILED (3 issues)"


class TestAgentInfo:
    """Tests for AgentInfo dataclass."""

    @pytest.fixture
    def basic_agent(self) -> AgentInfo:
        """Create basic AgentInfo for testing."""
        return AgentInfo(
            name="test_agent",
            file_path="/path/to/test.py",
            configuration_space={"model": "gpt-4o"},
            objectives=["accuracy"],
            constraints=[],
        )

    def test_initialization(self, basic_agent: AgentInfo) -> None:
        """Test AgentInfo initializes with all required fields."""
        assert basic_agent.name == "test_agent"
        assert basic_agent.file_path == "/path/to/test.py"
        assert basic_agent.configuration_space == {"model": "gpt-4o"}
        assert basic_agent.objectives == ["accuracy"]
        assert basic_agent.constraints == []

    def test_models_property_with_single_model(self) -> None:
        """Test models property extracts single model from configuration space."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model": "gpt-4o"},
            objectives=[],
            constraints=[],
        )
        assert agent.models == ["gpt-4o"]

    def test_models_property_with_list_of_models(self) -> None:
        """Test models property extracts list of models."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"models": ["gpt-4o", "claude-3-sonnet"]},
            objectives=[],
            constraints=[],
        )
        assert agent.models == ["gpt-4o", "claude-3-sonnet"]

    def test_models_property_with_llm_model_key(self) -> None:
        """Test models property extracts model from llm_model key."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"llm_model": "claude-3-haiku"},
            objectives=[],
            constraints=[],
        )
        assert agent.models == ["claude-3-haiku"]

    def test_models_property_with_model_name_key(self) -> None:
        """Test models property extracts model from model_name key."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model_name": "gpt-3.5-turbo"},
            objectives=[],
            constraints=[],
        )
        assert agent.models == ["gpt-3.5-turbo"]

    def test_models_property_with_multiple_keys(self) -> None:
        """Test models property extracts from multiple model keys."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={
                "model": "gpt-4o",
                "models": ["claude-3-sonnet", "claude-3-opus"],
            },
            objectives=[],
            constraints=[],
        )
        # Should extract from all keys
        assert len(agent.models) == 3
        assert "gpt-4o" in agent.models
        assert "claude-3-sonnet" in agent.models

    def test_models_property_with_no_models(self) -> None:
        """Test models property returns empty list when no models found."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"temperature": 0.7},
            objectives=[],
            constraints=[],
        )
        assert agent.models == []

    def test_max_tokens_property_with_max_tokens_key(self) -> None:
        """Test max_tokens property extracts from max_tokens key."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"max_tokens": 2000},
            objectives=[],
            constraints=[],
        )
        assert agent.max_tokens == 2000

    def test_max_tokens_property_with_tuple_range(self) -> None:
        """Test max_tokens property uses upper bound from tuple."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"max_tokens": (100, 2000)},
            objectives=[],
            constraints=[],
        )
        assert agent.max_tokens == 2000

    def test_max_tokens_property_with_list_values(self) -> None:
        """Test max_tokens property uses maximum from list."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"max_tokens": [500, 1000, 2000, 1500]},
            objectives=[],
            constraints=[],
        )
        assert agent.max_tokens == 2000

    def test_max_tokens_property_with_maxTokens_camel_case(self) -> None:
        """Test max_tokens property extracts from maxTokens camelCase key."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"maxTokens": 1500},
            objectives=[],
            constraints=[],
        )
        assert agent.max_tokens == 1500

    def test_max_tokens_property_with_max_output_tokens(self) -> None:
        """Test max_tokens property extracts from max_output_tokens key."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"max_output_tokens": 3000},
            objectives=[],
            constraints=[],
        )
        assert agent.max_tokens == 3000

    def test_max_tokens_property_default_value(self) -> None:
        """Test max_tokens property returns default when no token config found."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model": "gpt-4o"},
            objectives=[],
            constraints=[],
        )
        assert agent.max_tokens == DEFAULT_TOKENS_PER_QUERY


class TestAgentValidator:
    """Tests for AgentValidator class."""

    @pytest.fixture
    def default_config(self) -> HooksConfig:
        """Create default HooksConfig for testing."""
        return HooksConfig()

    @pytest.fixture
    def validator(self, default_config: HooksConfig) -> AgentValidator:
        """Create AgentValidator with default config."""
        return AgentValidator(default_config)

    @pytest.fixture
    def basic_agent(self) -> AgentInfo:
        """Create basic valid AgentInfo for testing."""
        return AgentInfo(
            name="test_func",
            file_path="/path/to/test.py",
            configuration_space={"model": "gpt-4o-mini"},
            objectives=["accuracy"],
            constraints=[],
        )

    def test_initialization_with_config(self, default_config: HooksConfig) -> None:
        """Test AgentValidator initializes with provided config."""
        validator = AgentValidator(default_config)
        assert validator.config == default_config

    @patch("traigent.hooks.validator.load_hooks_config")
    def test_initialization_without_config_loads_default(
        self, mock_load: MagicMock
    ) -> None:
        """Test AgentValidator loads config when none provided."""
        mock_config = HooksConfig()
        mock_load.return_value = mock_config
        validator = AgentValidator(None)
        assert validator.config == mock_config
        mock_load.assert_called_once()

    def test_validate_agent_returns_valid_for_compliant_agent(
        self, validator: AgentValidator, basic_agent: AgentInfo
    ) -> None:
        """Test validate_agent returns valid result for compliant agent."""
        result = validator.validate_agent(basic_agent)
        assert result.is_valid is True
        assert result.function_name == "test_func"
        assert len(result.issues) == 0

    def test_validate_models_blocked_model_creates_error(
        self, validator: AgentValidator
    ) -> None:
        """Test _validate_models creates error for blocked model."""
        validator.config.constraints.models.blocked_models = ["gpt-4-32k"]
        validator.config.constraints.models.blocked_reasons = {
            "gpt-4-32k": "Too expensive"
        }
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model": "gpt-4-32k"},
            objectives=[],
            constraints=[],
        )
        issues = validator._validate_models(agent)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].code == "BLOCKED_MODEL"
        assert "gpt-4-32k" in issues[0].message
        assert "Too expensive" in issues[0].suggestion

    def test_validate_models_blocked_model_without_reason(
        self, validator: AgentValidator
    ) -> None:
        """Test _validate_models handles blocked model without explicit reason."""
        validator.config.constraints.models.blocked_models = ["gpt-4-32k"]
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model": "gpt-4-32k"},
            objectives=[],
            constraints=[],
        )
        issues = validator._validate_models(agent)
        assert len(issues) == 1
        assert "Listed in blocked_models" in issues[0].suggestion

    def test_validate_models_not_in_allowed_list_creates_error(
        self, validator: AgentValidator
    ) -> None:
        """Test _validate_models creates error when model not in allowed list."""
        validator.config.constraints.models.allowed_models = [
            "gpt-4o-mini",
            "claude-3-haiku",
        ]
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model": "gpt-4-turbo"},
            objectives=[],
            constraints=[],
        )
        issues = validator._validate_models(agent)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].code == "MODEL_NOT_ALLOWED"
        assert "gpt-4-turbo" in issues[0].message

    def test_validate_models_in_allowed_list_passes(
        self, validator: AgentValidator
    ) -> None:
        """Test _validate_models passes when model is in allowed list."""
        validator.config.constraints.models.allowed_models = ["gpt-4o", "gpt-4o-mini"]
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model": "gpt-4o"},
            objectives=[],
            constraints=[],
        )
        issues = validator._validate_models(agent)
        assert len(issues) == 0

    def test_validate_models_empty_allowed_list_accepts_all(
        self, validator: AgentValidator
    ) -> None:
        """Test _validate_models accepts all models when allowed list is empty."""
        validator.config.constraints.models.allowed_models = []
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model": "any-model"},
            objectives=[],
            constraints=[],
        )
        issues = validator._validate_models(agent)
        assert len(issues) == 0

    def test_validate_models_multiple_models_all_checked(
        self, validator: AgentValidator
    ) -> None:
        """Test _validate_models checks all models in configuration."""
        validator.config.constraints.models.blocked_models = ["gpt-4-32k", "gpt-4"]
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"models": ["gpt-4-32k", "gpt-4", "gpt-4o-mini"]},
            objectives=[],
            constraints=[],
        )
        issues = validator._validate_models(agent)
        assert len(issues) == 2

    def test_validate_cost_no_cost_limit_returns_empty(
        self, validator: AgentValidator, basic_agent: AgentInfo
    ) -> None:
        """Test _validate_cost returns empty when no cost limit configured."""
        validator.config.constraints.cost.max_cost_per_query = None
        errors, warnings = validator._validate_cost(basic_agent)
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_validate_cost_exceeds_max_creates_error(
        self, validator: AgentValidator
    ) -> None:
        """Test _validate_cost creates error when cost exceeds maximum."""
        validator.config.constraints.cost.max_cost_per_query = 0.001
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={
                "model": "gpt-4o",  # Expensive model
                "max_tokens": 5000,
            },
            objectives=[],
            constraints=[],
        )
        errors, warnings = validator._validate_cost(agent)
        assert len(errors) == 1
        assert errors[0].severity == "error"
        assert errors[0].code == "COST_EXCEEDED"

    def test_validate_cost_near_threshold_creates_warning(
        self, validator: AgentValidator
    ) -> None:
        """Test _validate_cost creates warning when cost near threshold."""
        validator.config.constraints.cost.max_cost_per_query = 0.01
        validator.config.constraints.cost.warn_threshold_pct = 0.8
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={
                "model": "gpt-4o",
                "max_tokens": 1750,  # Cost around 0.00875, 87.5% of limit (above 80% threshold)
            },
            objectives=[],
            constraints=[],
        )
        errors, warnings = validator._validate_cost(agent)
        assert len(errors) == 0
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"
        assert warnings[0].code == "COST_WARNING"

    def test_validate_cost_below_threshold_no_warning(
        self, validator: AgentValidator
    ) -> None:
        """Test _validate_cost does not warn when cost well below threshold."""
        validator.config.constraints.cost.max_cost_per_query = 1.0
        validator.config.constraints.cost.warn_threshold_pct = 0.8
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={
                "model": "gpt-4o-mini",
                "max_tokens": 1000,
            },
            objectives=[],
            constraints=[],
        )
        errors, warnings = validator._validate_cost(agent)
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_validate_cost_cannot_estimate_returns_empty(
        self, validator: AgentValidator
    ) -> None:
        """Test _validate_cost returns empty when cost cannot be estimated."""
        validator.config.constraints.cost.max_cost_per_query = 0.01
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={},  # No model specified
            objectives=[],
            constraints=[],
        )
        errors, warnings = validator._validate_cost(agent)
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_estimate_cost_per_query_with_known_model(
        self, validator: AgentValidator
    ) -> None:
        """Test _estimate_cost_per_query calculates cost for known model."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={
                "model": "gpt-4o-mini",
                "max_tokens": 1000,
            },
            objectives=[],
            constraints=[],
        )
        cost = validator._estimate_cost_per_query(agent)
        expected = MODEL_COST_PER_1K["gpt-4o-mini"] * (1000 / 1000)
        assert cost == expected

    def test_estimate_cost_per_query_with_unknown_model(
        self, validator: AgentValidator
    ) -> None:
        """Test _estimate_cost_per_query returns None for unknown model."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={
                "model": "unknown-model-xyz",
                "max_tokens": 1000,
            },
            objectives=[],
            constraints=[],
        )
        cost = validator._estimate_cost_per_query(agent)
        assert cost is None

    def test_estimate_cost_per_query_with_partial_model_match(
        self, validator: AgentValidator
    ) -> None:
        """Test _estimate_cost_per_query handles partial model name matches."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={
                "model": "gpt-4o-mini-2024-07-18",  # Versioned model name
                "max_tokens": 1000,
            },
            objectives=[],
            constraints=[],
        )
        cost = validator._estimate_cost_per_query(agent)
        # Should match "gpt-4o-mini" partially
        assert cost is not None
        assert cost > 0

    def test_estimate_cost_per_query_with_multiple_models_returns_max(
        self, validator: AgentValidator
    ) -> None:
        """Test _estimate_cost_per_query returns maximum cost across models."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={
                "models": ["gpt-4o-mini", "gpt-4o"],  # Different costs
                "max_tokens": 1000,
            },
            objectives=[],
            constraints=[],
        )
        cost = validator._estimate_cost_per_query(agent)
        expected_max = MODEL_COST_PER_1K["gpt-4o"] * (1000 / 1000)
        assert cost == expected_max

    def test_estimate_cost_per_query_with_no_models(
        self, validator: AgentValidator
    ) -> None:
        """Test _estimate_cost_per_query returns None when no models specified."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"temperature": 0.7},
            objectives=[],
            constraints=[],
        )
        cost = validator._estimate_cost_per_query(agent)
        assert cost is None

    def test_estimate_cost_per_query_uses_max_tokens(
        self, validator: AgentValidator
    ) -> None:
        """Test _estimate_cost_per_query scales with max_tokens."""
        agent_1k = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model": "gpt-4o-mini", "max_tokens": 1000},
            objectives=[],
            constraints=[],
        )
        agent_2k = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model": "gpt-4o-mini", "max_tokens": 2000},
            objectives=[],
            constraints=[],
        )
        cost_1k = validator._estimate_cost_per_query(agent_1k)
        cost_2k = validator._estimate_cost_per_query(agent_2k)
        assert cost_2k == cost_1k * 2

    def test_validate_agent_populates_models_found(
        self, validator: AgentValidator
    ) -> None:
        """Test validate_agent populates models_found field."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"models": ["gpt-4o", "claude-3-sonnet"]},
            objectives=[],
            constraints=[],
        )
        result = validator.validate_agent(agent)
        assert result.models_found == ["gpt-4o", "claude-3-sonnet"]

    def test_validate_agent_populates_estimated_cost(
        self, validator: AgentValidator
    ) -> None:
        """Test validate_agent populates estimated_cost_per_query field."""
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={"model": "gpt-4o-mini", "max_tokens": 1000},
            objectives=[],
            constraints=[],
        )
        result = validator.validate_agent(agent)
        assert result.estimated_cost_per_query is not None
        assert result.estimated_cost_per_query > 0

    def test_validate_agent_with_multiple_issues(
        self, validator: AgentValidator
    ) -> None:
        """Test validate_agent collects multiple issues from different validators."""
        validator.config.constraints.models.blocked_models = ["gpt-4-32k"]
        validator.config.constraints.cost.max_cost_per_query = 0.001
        agent = AgentInfo(
            name="test",
            file_path="test.py",
            configuration_space={
                "model": "gpt-4-32k",
                "max_tokens": 5000,
            },
            objectives=[],
            constraints=[],
        )
        result = validator.validate_agent(agent)
        assert result.is_valid is False
        # Should have both model blocked and cost exceeded errors
        assert len(result.issues) >= 2

    @patch("traigent.cli.function_discovery.discover_optimized_functions")
    def test_validate_file_discovers_and_validates_functions(
        self, mock_discover: MagicMock, validator: AgentValidator
    ) -> None:
        """Test validate_file discovers functions and validates them."""
        mock_func = MagicMock()
        mock_func.name = "test_func"
        mock_func.decorator_config = {
            "configuration_space": {"model": "gpt-4o-mini"},
        }
        mock_func.objectives = ["accuracy"]
        mock_discover.return_value = [mock_func]

        results = validator.validate_file("/path/to/test.py")
        assert len(results) == 1
        assert results[0].function_name == "test_func"
        mock_discover.assert_called_once_with("/path/to/test.py")

    @patch("traigent.cli.function_discovery.discover_optimized_functions")
    def test_validate_file_handles_discovery_error(
        self, mock_discover: MagicMock, validator: AgentValidator
    ) -> None:
        """Test validate_file handles errors during function discovery."""
        mock_discover.side_effect = Exception("Syntax error")
        results = validator.validate_file("/path/to/test.py")
        assert len(results) == 1
        assert results[0].is_valid is False
        assert results[0].issues[0].code == "DISCOVERY_ERROR"

    @patch("traigent.cli.function_discovery.discover_optimized_functions")
    def test_validate_file_accepts_path_object(
        self, mock_discover: MagicMock, validator: AgentValidator
    ) -> None:
        """Test validate_file accepts Path objects."""
        mock_discover.return_value = []
        path = Path("/path/to/test.py")
        validator.validate_file(path)
        mock_discover.assert_called_once_with("/path/to/test.py")

    def test_validate_directory_finds_python_files(
        self, validator: AgentValidator
    ) -> None:
        """Test validate_directory discovers Python files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "test1.py").write_text("# test file")
            (tmp_path / "test2.py").write_text("# test file")
            (tmp_path / "readme.txt").write_text("# not python")

            with patch.object(validator, "validate_file") as mock_validate:
                mock_validate.return_value = []
                validator.validate_directory(tmp_path, recursive=False)
                # Should validate 2 Python files
                assert mock_validate.call_count == 2

    def test_validate_directory_recursive_finds_nested_files(
        self, validator: AgentValidator
    ) -> None:
        """Test validate_directory finds files in subdirectories when recursive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            subdir = tmp_path / "subdir"
            subdir.mkdir()
            (tmp_path / "test1.py").write_text("# test")
            (subdir / "test2.py").write_text("# test")

            with patch.object(validator, "validate_file") as mock_validate:
                mock_validate.return_value = []
                validator.validate_directory(tmp_path, recursive=True)
                assert mock_validate.call_count == 2

    def test_validate_directory_non_recursive_skips_subdirs(
        self, validator: AgentValidator
    ) -> None:
        """Test validate_directory skips subdirectories when not recursive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            subdir = tmp_path / "subdir"
            subdir.mkdir()
            (tmp_path / "test1.py").write_text("# test")
            (subdir / "test2.py").write_text("# test")

            with patch.object(validator, "validate_file") as mock_validate:
                mock_validate.return_value = []
                validator.validate_directory(tmp_path, recursive=False)
                assert mock_validate.call_count == 1

    def test_validate_directory_skips_hidden_directories(
        self, validator: AgentValidator
    ) -> None:
        """Test validate_directory skips hidden directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            hidden_dir = tmp_path / ".hidden"
            hidden_dir.mkdir()
            (tmp_path / "test.py").write_text("# test")
            (hidden_dir / "hidden.py").write_text("# test")

            with patch.object(validator, "validate_file") as mock_validate:
                mock_validate.return_value = []
                validator.validate_directory(tmp_path, recursive=True)
                # Should only validate test.py, not hidden.py
                assert mock_validate.call_count == 1

    def test_validate_directory_skips_common_non_source_dirs(
        self, validator: AgentValidator
    ) -> None:
        """Test validate_directory skips venv, node_modules, __pycache__."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for dirname in ["venv", "node_modules", "__pycache__"]:
                dir_path = tmp_path / dirname
                dir_path.mkdir()
                (dir_path / "test.py").write_text("# test")

            (tmp_path / "real.py").write_text("# test")

            with patch.object(validator, "validate_file") as mock_validate:
                mock_validate.return_value = []
                validator.validate_directory(tmp_path, recursive=True)
                # Should only validate real.py
                assert mock_validate.call_count == 1

    def test_validate_directory_respects_skip_patterns(
        self, validator: AgentValidator
    ) -> None:
        """Test validate_directory skips files matching skip_patterns."""
        validator.config.skip_patterns = ["**/test_*.py", "**/*_test.py"]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "test_file.py").write_text("# test")
            (tmp_path / "file_test.py").write_text("# test")
            (tmp_path / "normal.py").write_text("# test")

            with patch.object(validator, "validate_file") as mock_validate:
                mock_validate.return_value = []
                validator.validate_directory(tmp_path, recursive=False)
                # Should only validate normal.py
                assert mock_validate.call_count == 1

    def test_validate_directory_accepts_path_object(
        self, validator: AgentValidator
    ) -> None:
        """Test validate_directory accepts Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            with patch.object(validator, "validate_file") as mock_validate:
                mock_validate.return_value = []
                validator.validate_directory(path)

    def test_should_skip_matches_pattern(self, validator: AgentValidator) -> None:
        """Test _should_skip returns True for files matching skip patterns."""
        validator.config.skip_patterns = ["**/test_*.py"]
        assert validator._should_skip(Path("/path/to/test_file.py")) is True

    def test_should_skip_no_match(self, validator: AgentValidator) -> None:
        """Test _should_skip returns False for files not matching patterns."""
        validator.config.skip_patterns = ["**/test_*.py"]
        assert validator._should_skip(Path("/path/to/main.py")) is False

    def test_should_skip_with_empty_patterns(self, validator: AgentValidator) -> None:
        """Test _should_skip returns False when no skip patterns configured."""
        validator.config.skip_patterns = []
        assert validator._should_skip(Path("/path/to/any.py")) is False

    def test_should_skip_with_multiple_patterns(
        self, validator: AgentValidator
    ) -> None:
        """Test _should_skip checks all patterns."""
        validator.config.skip_patterns = ["**/test_*.py", "**/*.tmp.py"]
        assert validator._should_skip(Path("/path/test_file.py")) is True
        assert validator._should_skip(Path("/path/backup.tmp.py")) is True
        assert validator._should_skip(Path("/path/normal.py")) is False


class TestValidateAgentsForPush:
    """Tests for validate_agents_for_push function."""

    @patch("traigent.hooks.validator.load_hooks_config")
    @patch("traigent.hooks.validator.AgentValidator")
    def test_validate_agents_for_push_disabled_allows_push(
        self, mock_validator_class: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test validate_agents_for_push allows push when hooks disabled."""
        mock_config = HooksConfig(enabled=False)
        mock_load_config.return_value = mock_config

        should_allow, results = validate_agents_for_push()
        assert should_allow is True
        assert results == []
        mock_validator_class.assert_not_called()

    @patch("traigent.hooks.validator.load_hooks_config")
    @patch("traigent.hooks.validator.AgentValidator")
    def test_validate_agents_for_push_with_target_files(
        self, mock_validator_class: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test validate_agents_for_push validates specified target files."""
        mock_config = HooksConfig(enabled=True)
        mock_load_config.return_value = mock_config

        mock_validator = MagicMock()
        mock_validator.validate_file.return_value = [
            ValidationResult(function_name="test", is_valid=True)
        ]
        mock_validator_class.return_value = mock_validator

        target_files = [Path("/path/to/file1.py"), Path("/path/to/file2.py")]
        should_allow, results = validate_agents_for_push(target_files)

        assert should_allow is True
        assert len(results) == 2
        assert mock_validator.validate_file.call_count == 2

    @patch("traigent.hooks.validator.load_hooks_config")
    @patch("traigent.hooks.validator.AgentValidator")
    @patch("traigent.hooks.validator.Path.cwd")
    def test_validate_agents_for_push_without_target_files_validates_cwd(
        self,
        mock_cwd: MagicMock,
        mock_validator_class: MagicMock,
        mock_load_config: MagicMock,
    ) -> None:
        """Test validate_agents_for_push validates current directory when no files specified."""
        mock_config = HooksConfig(enabled=True)
        mock_load_config.return_value = mock_config

        mock_cwd.return_value = Path("/current/dir")

        mock_validator = MagicMock()
        mock_validator.validate_directory.return_value = [
            ValidationResult(function_name="test", is_valid=True)
        ]
        mock_validator_class.return_value = mock_validator

        should_allow, results = validate_agents_for_push(None)

        assert should_allow is True
        mock_validator.validate_directory.assert_called_once()

    @patch("traigent.hooks.validator.load_hooks_config")
    @patch("traigent.hooks.validator.AgentValidator")
    def test_validate_agents_for_push_blocks_on_errors(
        self, mock_validator_class: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test validate_agents_for_push blocks push when errors present."""
        mock_config = HooksConfig(enabled=True, fail_on_warning=False)
        mock_load_config.return_value = mock_config

        mock_validator = MagicMock()
        mock_validator.validate_file.return_value = [
            ValidationResult(
                function_name="test",
                is_valid=False,
                issues=[ValidationIssue("error", "ERR", "Error message")],
            )
        ]
        mock_validator_class.return_value = mock_validator

        should_allow, results = validate_agents_for_push([Path("/path/test.py")])

        assert should_allow is False

    @patch("traigent.hooks.validator.load_hooks_config")
    @patch("traigent.hooks.validator.AgentValidator")
    def test_validate_agents_for_push_blocks_on_warnings_if_configured(
        self, mock_validator_class: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test validate_agents_for_push blocks on warnings when fail_on_warning is True."""
        mock_config = HooksConfig(enabled=True, fail_on_warning=True)
        mock_load_config.return_value = mock_config

        mock_validator = MagicMock()
        mock_validator.validate_file.return_value = [
            ValidationResult(
                function_name="test",
                is_valid=True,
                warnings=[ValidationIssue("warning", "WARN", "Warning message")],
            )
        ]
        mock_validator_class.return_value = mock_validator

        should_allow, results = validate_agents_for_push([Path("/path/test.py")])

        assert should_allow is False

    @patch("traigent.hooks.validator.load_hooks_config")
    @patch("traigent.hooks.validator.AgentValidator")
    def test_validate_agents_for_push_allows_warnings_by_default(
        self, mock_validator_class: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test validate_agents_for_push allows warnings when fail_on_warning is False."""
        mock_config = HooksConfig(enabled=True, fail_on_warning=False)
        mock_load_config.return_value = mock_config

        mock_validator = MagicMock()
        mock_validator.validate_file.return_value = [
            ValidationResult(
                function_name="test",
                is_valid=True,
                warnings=[ValidationIssue("warning", "WARN", "Warning message")],
            )
        ]
        mock_validator_class.return_value = mock_validator

        should_allow, results = validate_agents_for_push([Path("/path/test.py")])

        assert should_allow is True

    @patch("traigent.hooks.validator.load_hooks_config")
    def test_validate_agents_for_push_loads_config_from_path(
        self, mock_load_config: MagicMock
    ) -> None:
        """Test validate_agents_for_push loads config from specified path."""
        mock_config = HooksConfig(enabled=False)
        mock_load_config.return_value = mock_config

        config_path = Path("/path/to/traigent.yml")
        validate_agents_for_push(None, config_path)

        mock_load_config.assert_called_once_with(config_path)

    @patch("traigent.hooks.validator.load_hooks_config")
    @patch("traigent.hooks.validator.AgentValidator")
    def test_validate_agents_for_push_returns_all_results(
        self, mock_validator_class: MagicMock, mock_load_config: MagicMock
    ) -> None:
        """Test validate_agents_for_push returns all validation results."""
        mock_config = HooksConfig(enabled=True)
        mock_load_config.return_value = mock_config

        result1 = ValidationResult(function_name="func1", is_valid=True)
        result2 = ValidationResult(function_name="func2", is_valid=False)

        mock_validator = MagicMock()
        mock_validator.validate_file.side_effect = [[result1], [result2]]
        mock_validator_class.return_value = mock_validator

        should_allow, results = validate_agents_for_push(
            [Path("/file1.py"), Path("/file2.py")]
        )

        assert len(results) == 2
        assert results[0].function_name == "func1"
        assert results[1].function_name == "func2"


class TestModelCostConstants:
    """Tests for MODEL_COST_PER_1K constants."""

    def test_model_cost_per_1k_contains_openai_models(self) -> None:
        """Test MODEL_COST_PER_1K includes OpenAI models."""
        assert "gpt-4o-mini" in MODEL_COST_PER_1K
        assert "gpt-4o" in MODEL_COST_PER_1K
        assert "gpt-4-turbo" in MODEL_COST_PER_1K
        assert "gpt-3.5-turbo" in MODEL_COST_PER_1K

    def test_model_cost_per_1k_contains_claude_models(self) -> None:
        """Test MODEL_COST_PER_1K includes Claude models."""
        assert "claude-3-haiku" in MODEL_COST_PER_1K
        assert "claude-3-sonnet" in MODEL_COST_PER_1K
        assert "claude-3-opus" in MODEL_COST_PER_1K

    def test_model_cost_per_1k_all_positive_values(self) -> None:
        """Test MODEL_COST_PER_1K contains only positive cost values."""
        for model, cost in MODEL_COST_PER_1K.items():
            assert cost > 0, f"Model {model} has non-positive cost: {cost}"

    def test_default_tokens_per_query_is_positive(self) -> None:
        """Test DEFAULT_TOKENS_PER_QUERY is a positive integer."""
        assert DEFAULT_TOKENS_PER_QUERY > 0
        assert isinstance(DEFAULT_TOKENS_PER_QUERY, int)
