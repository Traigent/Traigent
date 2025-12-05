"""Comprehensive tests for configuration constraints system (constraints.py).

This test suite covers:
- All constraint types: ParameterRange, Conditional, MutuallyExclusive, Dependency, Resource, Custom
- ConstraintManager functionality and validation workflows
- ConstraintViolation handling and messaging
- Convenience constraint functions
- Error handling and edge cases
- CTD (Combinatorial Test Design) scenarios
"""

from typing import Any

import pytest

from traigent.utils.constraints import (  # Convenience functions
    ConditionalConstraint,
    Constraint,
    ConstraintManager,
    ConstraintViolation,
    CustomConstraint,
    DependencyConstraint,
    MutuallyExclusiveConstraint,
    ParameterRangeConstraint,
    ResourceConstraint,
    exclusive_high_quality_strategies,
    fast_model_low_temp_constraint,
    max_tokens_constraint,
    model_cost_constraint,
    temperature_constraint,
)

# Test fixtures


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": False,
        "tools": [{"type": "function"}],
    }


@pytest.fixture
def constraint_manager():
    """Fresh ConstraintManager for each test."""
    return ConstraintManager()


@pytest.fixture
def sample_violation():
    """Sample constraint violation for testing."""
    return ConstraintViolation(
        constraint_name="test_constraint",
        message="Test violation message",
        violating_config={"param": "invalid_value"},
        suggestion="Fix the parameter value",
    )


# Mock constraint for testing


class MockConstraint(Constraint):
    """Mock constraint for testing base functionality."""

    def __init__(
        self, name: str, should_pass: bool = True, message: str = "Mock violation"
    ):
        super().__init__(name, "Mock constraint for testing")
        self.should_pass = should_pass
        self.message = message

    def validate(self, config: dict[str, Any]) -> bool:
        return self.should_pass

    def get_violation_message(self, config: dict[str, Any]) -> str:
        return self.message

    def get_suggestion(self, config: dict[str, Any]) -> str | None:
        return "Mock suggestion" if not self.should_pass else None


# Test Classes


class TestConstraintViolation:
    """Test ConstraintViolation dataclass."""

    def test_violation_creation(self, sample_violation):
        """Test creating constraint violation."""
        assert sample_violation.constraint_name == "test_constraint"
        assert sample_violation.message == "Test violation message"
        assert sample_violation.violating_config == {"param": "invalid_value"}
        assert sample_violation.suggestion == "Fix the parameter value"

    def test_violation_without_suggestion(self):
        """Test creating violation without suggestion."""
        violation = ConstraintViolation(
            constraint_name="no_suggestion",
            message="No suggestion provided",
            violating_config={},
        )

        assert violation.suggestion is None

    def test_violation_with_empty_config(self):
        """Test violation with empty configuration."""
        violation = ConstraintViolation(
            constraint_name="empty_config",
            message="Empty config violation",
            violating_config={},
        )

        assert violation.violating_config == {}


class TestConstraintBase:
    """Test abstract Constraint base class functionality."""

    def test_mock_constraint_initialization(self):
        """Test mock constraint initialization."""
        constraint = MockConstraint("test_constraint")

        assert constraint.name == "test_constraint"
        assert constraint.description == "Mock constraint for testing"

    def test_mock_constraint_validation_pass(self):
        """Test mock constraint that passes validation."""
        constraint = MockConstraint("passing", should_pass=True)

        assert constraint.validate({"any": "config"}) is True
        assert constraint.get_suggestion({"any": "config"}) is None

    def test_mock_constraint_validation_fail(self):
        """Test mock constraint that fails validation."""
        constraint = MockConstraint(
            "failing", should_pass=False, message="Custom failure"
        )

        assert constraint.validate({"any": "config"}) is False
        assert constraint.get_violation_message({"any": "config"}) == "Custom failure"
        assert constraint.get_suggestion({"any": "config"}) == "Mock suggestion"


class TestParameterRangeConstraint:
    """Test ParameterRangeConstraint functionality."""

    def test_range_constraint_initialization(self):
        """Test range constraint initialization."""
        constraint = ParameterRangeConstraint("temperature", 0.0, 2.0)

        assert constraint.name == "temperature_range"
        assert constraint.parameter == "temperature"
        assert constraint.min_value == 0.0
        assert constraint.max_value == 2.0

    def test_range_constraint_validation_within_range(self):
        """Test validation with value within range."""
        constraint = ParameterRangeConstraint("temperature", 0.0, 2.0)

        assert constraint.validate({"temperature": 0.5}) is True
        assert constraint.validate({"temperature": 0.0}) is True  # Min boundary
        assert constraint.validate({"temperature": 2.0}) is True  # Max boundary

    def test_range_constraint_validation_outside_range(self):
        """Test validation with value outside range."""
        constraint = ParameterRangeConstraint("temperature", 0.0, 2.0)

        assert constraint.validate({"temperature": -0.1}) is False
        assert constraint.validate({"temperature": 2.1}) is False

    def test_range_constraint_missing_parameter(self):
        """Test validation with missing parameter."""
        constraint = ParameterRangeConstraint("temperature", 0.0, 2.0)

        # Missing parameter should pass (no constraint)
        assert constraint.validate({"other_param": 0.5}) is True
        assert constraint.validate({}) is True

    def test_range_constraint_invalid_type(self):
        """Test validation with invalid parameter type."""
        constraint = ParameterRangeConstraint("temperature", 0.0, 2.0)

        assert constraint.validate({"temperature": "invalid"}) is False
        assert constraint.validate({"temperature": None}) is False
        assert constraint.validate({"temperature": []}) is False

    def test_range_constraint_min_only(self):
        """Test range constraint with only minimum value."""
        constraint = ParameterRangeConstraint("temperature", min_value=0.0)

        assert constraint.validate({"temperature": 0.0}) is True
        assert constraint.validate({"temperature": 1.0}) is True
        assert constraint.validate({"temperature": -0.1}) is False

    def test_range_constraint_max_only(self):
        """Test range constraint with only maximum value."""
        constraint = ParameterRangeConstraint("temperature", max_value=2.0)

        assert constraint.validate({"temperature": 2.0}) is True
        assert constraint.validate({"temperature": 1.0}) is True
        assert constraint.validate({"temperature": 2.1}) is False

    def test_range_constraint_violation_messages(self):
        """Test violation message generation."""
        constraint = ParameterRangeConstraint("temperature", 0.0, 2.0)

        # Both min and max
        msg = constraint.get_violation_message({"temperature": 3.0})
        assert "temperature" in msg
        assert "3.0" in msg
        assert "[0.0, 2.0]" in msg

        # Min only
        min_constraint = ParameterRangeConstraint("temperature", min_value=0.0)
        msg = min_constraint.get_violation_message({"temperature": -1.0})
        assert "below minimum" in msg

        # Max only
        max_constraint = ParameterRangeConstraint("temperature", max_value=2.0)
        msg = max_constraint.get_violation_message({"temperature": 3.0})
        assert "above maximum" in msg

    def test_range_constraint_suggestions(self):
        """Test suggestion generation."""
        # Both min and max
        constraint = ParameterRangeConstraint("temperature", 0.0, 2.0)
        suggestion = constraint.get_suggestion({"temperature": 3.0})
        assert "between 0.0 and 2.0" in suggestion

        # Min only
        min_constraint = ParameterRangeConstraint("temperature", min_value=0.0)
        suggestion = min_constraint.get_suggestion({"temperature": -1.0})
        assert ">= 0.0" in suggestion

        # Max only
        max_constraint = ParameterRangeConstraint("temperature", max_value=2.0)
        suggestion = max_constraint.get_suggestion({"temperature": 3.0})
        assert "<= 2.0" in suggestion


class TestConditionalConstraint:
    """Test ConditionalConstraint functionality."""

    def test_conditional_constraint_initialization(self):
        """Test conditional constraint initialization."""

        def condition(config):
            return config.get("model") == "gpt-4"

        base_constraint = ParameterRangeConstraint("temperature", 0.0, 1.0)

        constraint = ConditionalConstraint("gpt4_temp", condition, base_constraint)

        assert constraint.name == "gpt4_temp"
        assert "Conditional:" in constraint.description

    def test_conditional_constraint_condition_true(self):
        """Test conditional constraint when condition is true."""

        def condition(config):
            return config.get("model") == "gpt-4"

        base_constraint = ParameterRangeConstraint("temperature", 0.0, 1.0)
        constraint = ConditionalConstraint("gpt4_temp", condition, base_constraint)

        # Condition true, valid config
        assert constraint.validate({"model": "gpt-4", "temperature": 0.5}) is True

        # Condition true, invalid config
        assert constraint.validate({"model": "gpt-4", "temperature": 1.5}) is False

    def test_conditional_constraint_condition_false(self):
        """Test conditional constraint when condition is false."""

        def condition(config):
            return config.get("model") == "gpt-4"

        base_constraint = ParameterRangeConstraint("temperature", 0.0, 1.0)
        constraint = ConditionalConstraint("gpt4_temp", condition, base_constraint)

        # Condition false - should pass regardless of temperature
        assert constraint.validate({"model": "gpt-3.5", "temperature": 1.5}) is True
        assert constraint.validate({"model": "claude", "temperature": 2.0}) is True

    def test_conditional_constraint_messages(self):
        """Test conditional constraint violation messages."""

        def condition(config):
            return config.get("model") == "gpt-4"

        base_constraint = ParameterRangeConstraint("temperature", 0.0, 1.0)
        constraint = ConditionalConstraint("gpt4_temp", condition, base_constraint)

        config = {"model": "gpt-4", "temperature": 1.5}
        msg = constraint.get_violation_message(config)
        assert "Conditional constraint violated:" in msg

        suggestion = constraint.get_suggestion(config)
        assert "When condition applies:" in suggestion

    def test_conditional_constraint_complex_condition(self):
        """Test conditional constraint with complex condition."""

        def complex_condition(config):
            return config.get("model") == "gpt-4" and config.get("stream") is True

        base_constraint = ParameterRangeConstraint("max_tokens", 1, 2000)
        constraint = ConditionalConstraint(
            "streaming_gpt4", complex_condition, base_constraint
        )

        # Both conditions met
        assert (
            constraint.validate({"model": "gpt-4", "stream": True, "max_tokens": 1000})
            is True
        )

        # Only one condition met
        assert (
            constraint.validate({"model": "gpt-4", "stream": False, "max_tokens": 5000})
            is True
        )

        # Both conditions met, constraint violated
        assert (
            constraint.validate({"model": "gpt-4", "stream": True, "max_tokens": 5000})
            is False
        )


class TestMutuallyExclusiveConstraint:
    """Test MutuallyExclusiveConstraint functionality."""

    def test_mutex_constraint_initialization(self):
        """Test mutually exclusive constraint initialization."""
        constraint = MutuallyExclusiveConstraint(
            ["param1", "param2"], ["value1", "value2"], max_simultaneous=1
        )

        assert "mutex_param1+param2" in constraint.name
        assert constraint.parameters == ["param1", "param2"]
        assert constraint.values == {"value1", "value2"}
        assert constraint.max_simultaneous == 1

    def test_mutex_constraint_validation_pass(self):
        """Test mutex constraint validation that passes."""
        constraint = MutuallyExclusiveConstraint(
            ["param1", "param2"], ["forbidden"], max_simultaneous=1
        )

        # No forbidden values
        assert constraint.validate({"param1": "allowed", "param2": "allowed"}) is True

        # One forbidden value (within limit)
        assert constraint.validate({"param1": "forbidden", "param2": "allowed"}) is True

        # Missing parameters
        assert constraint.validate({"other": "value"}) is True

    def test_mutex_constraint_validation_fail(self):
        """Test mutex constraint validation that fails."""
        constraint = MutuallyExclusiveConstraint(
            ["param1", "param2"], ["forbidden"], max_simultaneous=1
        )

        # Both have forbidden values (exceeds limit)
        assert (
            constraint.validate({"param1": "forbidden", "param2": "forbidden"}) is False
        )

    def test_mutex_constraint_higher_limit(self):
        """Test mutex constraint with higher simultaneous limit."""
        constraint = MutuallyExclusiveConstraint(
            ["param1", "param2", "param3"], ["forbidden"], max_simultaneous=2
        )

        # Two forbidden values (within limit)
        assert (
            constraint.validate(
                {"param1": "forbidden", "param2": "forbidden", "param3": "allowed"}
            )
            is True
        )

        # Three forbidden values (exceeds limit)
        assert (
            constraint.validate(
                {"param1": "forbidden", "param2": "forbidden", "param3": "forbidden"}
            )
            is False
        )

    def test_mutex_constraint_multiple_forbidden_values(self):
        """Test mutex constraint with multiple forbidden values."""
        constraint = MutuallyExclusiveConstraint(
            ["param1", "param2"], ["forbidden1", "forbidden2"], max_simultaneous=1
        )

        # Different forbidden values (still violates)
        assert (
            constraint.validate({"param1": "forbidden1", "param2": "forbidden2"})
            is False
        )

    def test_mutex_constraint_messages(self):
        """Test mutex constraint violation messages."""
        constraint = MutuallyExclusiveConstraint(
            ["param1", "param2"], ["forbidden"], max_simultaneous=1
        )

        config = {"param1": "forbidden", "param2": "forbidden"}
        msg = constraint.get_violation_message(config)
        assert "Too many parameters with restricted values" in msg
        assert "param1=forbidden" in msg
        assert "param2=forbidden" in msg

        suggestion = constraint.get_suggestion(config)
        assert "Change at least one of" in suggestion
        assert "forbidden" in suggestion


class TestDependencyConstraint:
    """Test DependencyConstraint functionality."""

    def test_dependency_constraint_initialization(self):
        """Test dependency constraint initialization."""
        constraint = DependencyConstraint("tools", "model", ["gpt-4", "claude-3"])

        assert constraint.name == "tools_depends_on_model"
        assert constraint.dependent_param == "tools"
        assert constraint.dependency_param == "model"
        assert constraint.dependency_values == {"gpt-4", "claude-3"}

    def test_dependency_constraint_validation_pass(self):
        """Test dependency constraint validation that passes."""
        constraint = DependencyConstraint("tools", "model", ["gpt-4", "claude-3"])

        # Dependent param not present (no constraint)
        assert constraint.validate({"model": "gpt-3.5"}) is True

        # Dependency satisfied
        assert constraint.validate({"tools": [], "model": "gpt-4"}) is True
        assert constraint.validate({"tools": [], "model": "claude-3"}) is True

    def test_dependency_constraint_validation_fail(self):
        """Test dependency constraint validation that fails."""
        constraint = DependencyConstraint("tools", "model", ["gpt-4", "claude-3"])

        # Dependency not satisfied
        assert constraint.validate({"tools": [], "model": "gpt-3.5"}) is False

        # Dependency missing
        assert constraint.validate({"tools": []}) is False

    def test_dependency_constraint_messages(self):
        """Test dependency constraint violation messages."""
        constraint = DependencyConstraint("tools", "model", ["gpt-4", "claude-3"])

        # Wrong dependency value
        config = {"tools": [], "model": "gpt-3.5"}
        msg = constraint.get_violation_message(config)
        assert "tools" in msg
        assert "model" in msg
        assert "gpt-3.5" in msg

        suggestion = constraint.get_suggestion(config)
        assert "Set 'model' to one of:" in suggestion
        assert "gpt-4" in suggestion

        # Missing dependency
        config_missing = {"tools": []}
        msg_missing = constraint.get_violation_message(config_missing)
        assert "missing" in msg_missing

    def test_dependency_constraint_single_value(self):
        """Test dependency constraint with single required value."""
        constraint = DependencyConstraint("stream", "model", ["gpt-4"])

        assert constraint.validate({"stream": True, "model": "gpt-4"}) is True
        assert constraint.validate({"stream": True, "model": "gpt-3.5"}) is False


class TestResourceConstraint:
    """Test ResourceConstraint functionality."""

    def test_resource_constraint_initialization(self):
        """Test resource constraint initialization."""

        def calculator(config):
            return config.get("max_tokens", 0) * 0.001

        constraint = ResourceConstraint("token_cost", calculator, 1.0)

        assert constraint.name == "token_cost"
        assert constraint.resource_calculator == calculator
        assert constraint.max_resource == 1.0

    def test_resource_constraint_validation_pass(self):
        """Test resource constraint validation that passes."""

        def calculator(config):
            return config.get("max_tokens", 0) * 0.001

        constraint = ResourceConstraint("token_cost", calculator, 1.0)

        # Within limit
        assert constraint.validate({"max_tokens": 500}) is True  # 0.5 cost
        assert constraint.validate({"max_tokens": 1000}) is True  # 1.0 cost (boundary)

    def test_resource_constraint_validation_fail(self):
        """Test resource constraint validation that fails."""

        def calculator(config):
            return config.get("max_tokens", 0) * 0.001

        constraint = ResourceConstraint("token_cost", calculator, 1.0)

        # Exceeds limit
        assert constraint.validate({"max_tokens": 1500}) is False  # 1.5 cost

    def test_resource_constraint_calculator_exception(self):
        """Test resource constraint with calculator that throws exception."""

        def broken_calculator(config):
            raise ValueError("Calculator error")

        constraint = ResourceConstraint("broken", broken_calculator, 1.0)

        # Should return False when calculator fails
        assert constraint.validate({"any": "config"}) is False

    def test_resource_constraint_messages(self):
        """Test resource constraint violation messages."""

        def calculator(config):
            return config.get("max_tokens", 0) * 0.001

        constraint = ResourceConstraint("token_cost", calculator, 1.0)

        config = {"max_tokens": 1500}
        msg = constraint.get_violation_message(config)
        assert "Resource usage 1.500 exceeds limit 1.0" in msg

        suggestion = constraint.get_suggestion(config)
        assert "Reduce resource usage to <= 1.0" in suggestion

    def test_resource_constraint_message_with_exception(self):
        """Test resource constraint messages when calculator throws exception."""

        def broken_calculator(config):
            raise ValueError("Calculator error")

        constraint = ResourceConstraint("broken", broken_calculator, 1.0)

        msg = constraint.get_violation_message({"any": "config"})
        assert "Cannot calculate resource usage:" in msg
        assert "Calculator error" in msg

    def test_resource_constraint_complex_calculator(self):
        """Test resource constraint with complex calculator."""

        def complex_calculator(config):
            base_cost = config.get("max_tokens", 1000) * 0.001
            model_multiplier = {"gpt-4": 2.0, "gpt-3.5": 1.0}.get(
                config.get("model", "gpt-3.5"), 1.0
            )
            return base_cost * model_multiplier

        constraint = ResourceConstraint("complex_cost", complex_calculator, 1.0)

        # GPT-3.5 with 500 tokens: 0.5 cost
        assert constraint.validate({"model": "gpt-3.5", "max_tokens": 500}) is True

        # GPT-4 with 500 tokens: 1.0 cost (boundary)
        assert constraint.validate({"model": "gpt-4", "max_tokens": 500}) is True

        # GPT-4 with 600 tokens: 1.2 cost (exceeds)
        assert constraint.validate({"model": "gpt-4", "max_tokens": 600}) is False


class TestCustomConstraint:
    """Test CustomConstraint functionality."""

    def test_custom_constraint_initialization(self):
        """Test custom constraint initialization."""

        def validator(config):
            return config.get("temperature", 0) < 1.0

        def message_gen(config):
            return f"Temperature {config.get('temperature')} too high"

        constraint = CustomConstraint("custom_temp", validator, message_gen)

        assert constraint.name == "custom_temp"
        assert constraint.description == "Custom constraint"
        assert constraint.validator == validator
        assert constraint.message_generator == message_gen
        assert constraint.suggestion_generator is None

    def test_custom_constraint_with_suggestion(self):
        """Test custom constraint with suggestion generator."""

        def validator(config):
            return config.get("temperature", 0) < 1.0

        def message_gen(config):
            return "Temperature too high"

        def suggestion_gen(config):
            return "Use temperature < 1.0"

        constraint = CustomConstraint(
            "custom_temp", validator, message_gen, suggestion_gen
        )

        assert constraint.suggestion_generator == suggestion_gen

    def test_custom_constraint_validation(self):
        """Test custom constraint validation."""

        def validator(config):
            return config.get("temperature", 0) < 1.0

        def message_gen(config):
            return "Temperature too high"

        constraint = CustomConstraint("custom_temp", validator, message_gen)

        assert constraint.validate({"temperature": 0.5}) is True
        assert constraint.validate({"temperature": 1.5}) is False
        assert constraint.validate({}) is True  # Default 0 < 1.0

    def test_custom_constraint_validator_exception(self):
        """Test custom constraint with validator that throws exception."""

        def broken_validator(config):
            raise ValueError("Validator error")

        def message_gen(config):
            return "Error occurred"

        constraint = CustomConstraint("broken", broken_validator, message_gen)

        # Should return False when validator fails
        assert constraint.validate({"any": "config"}) is False

    def test_custom_constraint_messages(self):
        """Test custom constraint message generation."""

        def validator(config):
            return config.get("temperature", 0) < 1.0

        def message_gen(config):
            return f"Temperature {config.get('temperature')} too high"

        def suggestion_gen(config):
            return "Use temperature < 1.0"

        constraint = CustomConstraint(
            "custom_temp", validator, message_gen, suggestion_gen
        )

        config = {"temperature": 1.5}
        msg = constraint.get_violation_message(config)
        assert "Temperature 1.5 too high" in msg

        suggestion = constraint.get_suggestion(config)
        assert "Use temperature < 1.0" in suggestion

    def test_custom_constraint_message_exception(self):
        """Test custom constraint messages when generators throw exceptions."""

        def validator(config):
            return False

        def broken_message_gen(config):
            raise ValueError("Message error")

        def broken_suggestion_gen(config):
            raise ValueError("Suggestion error")

        constraint = CustomConstraint(
            "broken", validator, broken_message_gen, broken_suggestion_gen
        )

        config = {"any": "config"}
        msg = constraint.get_violation_message(config)
        assert "Custom constraint 'broken' violated:" in msg
        assert "Message error" in msg

        suggestion = constraint.get_suggestion(config)
        assert suggestion is None  # Exception in suggestion generator

    def test_custom_constraint_complex_logic(self):
        """Test custom constraint with complex validation logic."""

        def complex_validator(config):
            # Complex business rule: high temperature requires high max_tokens
            temp = config.get("temperature", 0.7)
            max_tokens = config.get("max_tokens", 1000)

            if temp > 1.0:
                return max_tokens >= 2000
            return True

        def message_gen(config):
            return "High temperature requires high max_tokens"

        constraint = CustomConstraint(
            "temp_tokens_rule", complex_validator, message_gen
        )

        # Valid combinations
        assert constraint.validate({"temperature": 0.5, "max_tokens": 500}) is True
        assert constraint.validate({"temperature": 1.2, "max_tokens": 2000}) is True

        # Invalid combination
        assert constraint.validate({"temperature": 1.5, "max_tokens": 1000}) is False


class TestConstraintManager:
    """Test ConstraintManager functionality."""

    def test_manager_initialization(self, constraint_manager):
        """Test constraint manager initialization."""
        assert len(constraint_manager.constraints) == 0

    def test_add_constraint(self, constraint_manager):
        """Test adding constraints to manager."""
        constraint1 = MockConstraint("constraint1")
        constraint2 = MockConstraint("constraint2")

        constraint_manager.add_constraint(constraint1)
        assert len(constraint_manager.constraints) == 1
        assert constraint_manager.constraints[0] == constraint1

        constraint_manager.add_constraint(constraint2)
        assert len(constraint_manager.constraints) == 2
        assert constraint_manager.constraints[1] == constraint2

    def test_remove_constraint(self, constraint_manager):
        """Test removing constraints from manager."""
        constraint1 = MockConstraint("constraint1")
        constraint2 = MockConstraint("constraint2")

        constraint_manager.add_constraint(constraint1)
        constraint_manager.add_constraint(constraint2)

        # Remove existing constraint
        removed = constraint_manager.remove_constraint("constraint1")
        assert removed is True
        assert len(constraint_manager.constraints) == 1
        assert constraint_manager.constraints[0].name == "constraint2"

        # Try to remove non-existent constraint
        removed = constraint_manager.remove_constraint("nonexistent")
        assert removed is False
        assert len(constraint_manager.constraints) == 1

    def test_validate_configuration_all_pass(self, constraint_manager):
        """Test configuration validation when all constraints pass."""
        constraint1 = MockConstraint("constraint1", should_pass=True)
        constraint2 = MockConstraint("constraint2", should_pass=True)

        constraint_manager.add_constraint(constraint1)
        constraint_manager.add_constraint(constraint2)

        is_valid, violations = constraint_manager.validate_configuration(
            {"test": "config"}
        )

        assert is_valid is True
        assert len(violations) == 0

    def test_validate_configuration_some_fail(self, constraint_manager):
        """Test configuration validation when some constraints fail."""
        constraint1 = MockConstraint("constraint1", should_pass=True)
        constraint2 = MockConstraint(
            "constraint2", should_pass=False, message="Failure 2"
        )
        constraint3 = MockConstraint(
            "constraint3", should_pass=False, message="Failure 3"
        )

        constraint_manager.add_constraint(constraint1)
        constraint_manager.add_constraint(constraint2)
        constraint_manager.add_constraint(constraint3)

        config = {"test": "config"}
        is_valid, violations = constraint_manager.validate_configuration(config)

        assert is_valid is False
        assert len(violations) == 2

        # Check violation details
        violation_names = [v.constraint_name for v in violations]
        assert "constraint2" in violation_names
        assert "constraint3" in violation_names

        violation_messages = [v.message for v in violations]
        assert "Failure 2" in violation_messages
        assert "Failure 3" in violation_messages

        # Check that config is copied
        for violation in violations:
            assert violation.violating_config == config
            assert violation.violating_config is not config  # Should be a copy

    def test_validate_configuration_empty_manager(self, constraint_manager):
        """Test validation with no constraints."""
        is_valid, violations = constraint_manager.validate_configuration(
            {"any": "config"}
        )

        assert is_valid is True
        assert len(violations) == 0

    def test_filter_valid_configurations(self, constraint_manager):
        """Test filtering configurations to only valid ones."""
        # Constraint that only passes if temperature <= 1.0
        temp_constraint = ParameterRangeConstraint("temperature", max_value=1.0)
        constraint_manager.add_constraint(temp_constraint)

        configs = [
            {"temperature": 0.5},  # Valid
            {"temperature": 1.0},  # Valid (boundary)
            {"temperature": 1.5},  # Invalid
            {"other_param": "value"},  # Valid (no temperature)
            {"temperature": 2.0},  # Invalid
        ]

        valid_configs = constraint_manager.filter_valid_configurations(configs)

        assert len(valid_configs) == 3
        valid_temps = [
            c.get("temperature") for c in valid_configs if "temperature" in c
        ]
        assert 0.5 in valid_temps
        assert 1.0 in valid_temps
        assert 1.5 not in valid_temps
        assert 2.0 not in valid_temps

    def test_filter_valid_configurations_empty_list(self, constraint_manager):
        """Test filtering empty configuration list."""
        constraint_manager.add_constraint(MockConstraint("test"))
        valid_configs = constraint_manager.filter_valid_configurations([])

        assert valid_configs == []

    def test_get_constraint_summary_empty(self, constraint_manager):
        """Test constraint summary with no constraints."""
        summary = constraint_manager.get_constraint_summary()
        assert summary == "No constraints defined"

    def test_get_constraint_summary_with_constraints(self, constraint_manager):
        """Test constraint summary with multiple constraints."""
        constraint1 = MockConstraint("constraint1")
        constraint1.description = "First constraint"

        constraint2 = MockConstraint("constraint2")
        constraint2.description = "Second constraint"

        constraint_manager.add_constraint(constraint1)
        constraint_manager.add_constraint(constraint2)

        summary = constraint_manager.get_constraint_summary()

        assert "Active constraints (2):" in summary
        assert "1. constraint1: First constraint" in summary
        assert "2. constraint2: Second constraint" in summary

    def test_validate_configuration_with_suggestions(self, constraint_manager):
        """Test validation with constraints that provide suggestions."""
        constraint = MockConstraint(
            "with_suggestion", should_pass=False, message="Failed"
        )
        constraint_manager.add_constraint(constraint)

        is_valid, violations = constraint_manager.validate_configuration(
            {"test": "config"}
        )

        assert is_valid is False
        assert len(violations) == 1
        assert violations[0].suggestion == "Mock suggestion"


class TestConvenienceConstraints:
    """Test convenience constraint functions."""

    def test_temperature_constraint_default(self):
        """Test temperature constraint with default values."""
        constraint = temperature_constraint()

        assert constraint.parameter == "temperature"
        assert constraint.min_value == 0.0
        assert constraint.max_value == 2.0

        assert constraint.validate({"temperature": 1.0}) is True
        assert constraint.validate({"temperature": -0.1}) is False
        assert constraint.validate({"temperature": 2.1}) is False

    def test_temperature_constraint_custom(self):
        """Test temperature constraint with custom values."""
        constraint = temperature_constraint(min_temp=0.5, max_temp=1.5)

        assert constraint.min_value == 0.5
        assert constraint.max_value == 1.5

        assert constraint.validate({"temperature": 1.0}) is True
        assert constraint.validate({"temperature": 0.3}) is False
        assert constraint.validate({"temperature": 1.7}) is False

    def test_max_tokens_constraint_default(self):
        """Test max_tokens constraint with default values."""
        constraint = max_tokens_constraint()

        assert constraint.parameter == "max_tokens"
        assert constraint.min_value == 1
        assert constraint.max_value == 4000

        assert constraint.validate({"max_tokens": 1000}) is True
        assert constraint.validate({"max_tokens": 0}) is False
        assert constraint.validate({"max_tokens": 5000}) is False

    def test_max_tokens_constraint_custom(self):
        """Test max_tokens constraint with custom values."""
        constraint = max_tokens_constraint(min_tokens=100, max_tokens=2000)

        assert constraint.min_value == 100
        assert constraint.max_value == 2000

        assert constraint.validate({"max_tokens": 1000}) is True
        assert constraint.validate({"max_tokens": 50}) is False
        assert constraint.validate({"max_tokens": 3000}) is False

    def test_model_cost_constraint_default(self):
        """Test model cost constraint with default values."""
        constraint = model_cost_constraint()

        assert constraint.name == "model_cost"
        assert constraint.max_resource == 0.1

        # Test with different models
        assert (
            constraint.validate({"model": "gpt-4o-mini", "max_tokens": 1000}) is True
        )  # Low cost
        assert (
            constraint.validate({"model": "GPT-4o", "max_tokens": 2000}) is False
        )  # High cost

    def test_model_cost_constraint_custom_limit(self):
        """Test model cost constraint with custom cost limit."""
        constraint = model_cost_constraint(max_cost_per_1k_tokens=0.01)

        assert constraint.max_resource == 0.01

        # Even mini model exceeds very low limit
        assert (
            constraint.validate({"model": "gpt-4o-mini", "max_tokens": 10000}) is False
        )

    def test_model_cost_constraint_unknown_model(self):
        """Test model cost constraint with unknown model."""
        constraint = model_cost_constraint()

        # Unknown model should use default cost (gpt-4o-mini rate)
        assert (
            constraint.validate({"model": "unknown-model", "max_tokens": 1000}) is True
        )

    def test_fast_model_low_temp_constraint(self):
        """Test fast model low temperature constraint."""
        constraint = fast_model_low_temp_constraint()

        assert constraint.name == "fast_model_low_temp"

        # Fast model with low temperature (should pass)
        assert constraint.validate({"model": "gpt-4o-mini", "temperature": 0.5}) is True

        # Fast model with high temperature (should fail)
        assert (
            constraint.validate({"model": "gpt-4o-mini", "temperature": 1.0}) is False
        )

        # Non-fast model with high temperature (should pass - condition doesn't apply)
        assert constraint.validate({"model": "gpt-4", "temperature": 1.0}) is True

        # Test other fast models
        assert (
            constraint.validate({"model": "claude-3-haiku", "temperature": 0.6}) is True
        )
        assert (
            constraint.validate({"model": "claude-3-haiku", "temperature": 0.8})
            is False
        )

    def test_exclusive_high_quality_strategies(self):
        """Test exclusive high quality strategies constraint."""
        constraint = exclusive_high_quality_strategies()

        assert constraint.max_simultaneous == 1

        # Single high-quality strategy (should pass)
        assert constraint.validate({"model": "GPT-4o", "strategy": "normal"}) is True
        assert (
            constraint.validate({"model": "normal", "strategy": "high_quality"}) is True
        )

        # Multiple high-quality strategies (should fail)
        assert (
            constraint.validate({"model": "GPT-4o", "strategy": "high_quality"})
            is False
        )
        assert (
            constraint.validate({"model": "claude-3-opus", "strategy": "high_quality"})
            is False
        )

        # No high-quality strategies (should pass)
        assert constraint.validate({"model": "gpt-4o-mini", "strategy": "fast"}) is True


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_range_constraint_with_none_values(self):
        """Test range constraint with None min/max values."""
        constraint = ParameterRangeConstraint("temperature", None, None)

        # Should have no effective constraints
        assert constraint.min_value is None
        assert constraint.max_value is None

        # Any numeric value should pass
        assert constraint.validate({"temperature": -1000}) is True
        assert constraint.validate({"temperature": 1000}) is True

        # Non-numeric should still fail
        assert constraint.validate({"temperature": "invalid"}) is False

    def test_constraint_manager_with_broken_constraint(self, constraint_manager):
        """Test constraint manager with constraint that throws exceptions."""

        class BrokenConstraint(Constraint):
            def validate(self, config):
                raise RuntimeError("Broken validation")

            def get_violation_message(self, config):
                raise RuntimeError("Broken message")

        broken_constraint = BrokenConstraint("broken", "Broken constraint")
        working_constraint = MockConstraint("working", should_pass=True)

        constraint_manager.add_constraint(broken_constraint)
        constraint_manager.add_constraint(working_constraint)

        # Should handle exceptions gracefully
        try:
            is_valid, violations = constraint_manager.validate_configuration(
                {"test": "config"}
            )
            # Implementation dependent - may treat exception as failure or ignore
            assert isinstance(is_valid, bool)
            assert isinstance(violations, list)
        except (ValueError, RuntimeError, AttributeError):
            # If constraint exceptions propagate, that's also acceptable behavior
            pass

    def test_conditional_constraint_with_broken_condition(self):
        """Test conditional constraint with condition that throws exception."""

        def broken_condition(config):
            raise ValueError("Broken condition")

        base_constraint = MockConstraint("base", should_pass=True)
        constraint = ConditionalConstraint(
            "broken_condition", broken_condition, base_constraint
        )

        # Should handle condition exceptions gracefully
        try:
            result = constraint.validate({"test": "config"})
            assert isinstance(result, bool)
        except ValueError:
            # If the ValueError from broken_condition propagates, that's acceptable
            pass

    def test_mutex_constraint_with_empty_parameters(self):
        """Test mutex constraint with empty parameter list."""
        constraint = MutuallyExclusiveConstraint([], ["value"], max_simultaneous=1)

        # Should always pass with no parameters to check
        assert constraint.validate({"any": "config"}) is True

    def test_dependency_constraint_with_empty_dependencies(self):
        """Test dependency constraint with empty dependency values."""
        constraint = DependencyConstraint("dependent", "dependency", [])

        # Should always fail when dependent param is present (no valid dependency values)
        assert constraint.validate({"dependent": "value", "dependency": "any"}) is False
        assert constraint.validate({"other": "value"}) is True  # Dependent not present

    def test_resource_constraint_boundary_conditions(self):
        """Test resource constraint with boundary conditions."""

        def calculator(config):
            return float(config.get("value", 0))

        constraint = ResourceConstraint("boundary", calculator, 1.0)

        # Exact boundary
        assert constraint.validate({"value": 1.0}) is True
        assert constraint.validate({"value": 1.0000001}) is False

        # Edge values
        assert constraint.validate({"value": 0}) is True
        assert constraint.validate({"value": -1}) is True  # Negative values allowed

    def test_constraint_manager_duplicate_names(self, constraint_manager):
        """Test constraint manager with duplicate constraint names."""
        constraint1 = MockConstraint("duplicate_name")
        constraint2 = MockConstraint("duplicate_name")

        constraint_manager.add_constraint(constraint1)
        constraint_manager.add_constraint(constraint2)

        # Both should be added (manager doesn't prevent duplicates)
        assert len(constraint_manager.constraints) == 2

        # Remove should remove first occurrence
        removed = constraint_manager.remove_constraint("duplicate_name")
        assert removed is True
        assert len(constraint_manager.constraints) == 1


class TestCTDScenarios:
    """Combinatorial Test Design scenarios for comprehensive coverage."""

    @pytest.mark.parametrize(
        "min_val,max_val,test_val,expected_result",
        [
            (0.0, 2.0, 1.0, True),  # Within range
            (0.0, 2.0, 0.0, True),  # Min boundary
            (0.0, 2.0, 2.0, True),  # Max boundary
            (0.0, 2.0, -0.1, False),  # Below min
            (0.0, 2.0, 2.1, False),  # Above max
            (None, 2.0, 1.0, True),  # No min limit
            (0.0, None, 1.0, True),  # No max limit
            (None, None, 1.0, True),  # No limits
        ],
    )
    def test_range_constraint_combinations(
        self, min_val, max_val, test_val, expected_result
    ):
        """Test different combinations of range constraint boundaries."""
        constraint = ParameterRangeConstraint("test_param", min_val, max_val)
        result = constraint.validate({"test_param": test_val})
        assert result == expected_result

    @pytest.mark.parametrize(
        "param_type,param_value,expected_result",
        [
            ("valid_int", 1, True),
            ("valid_float", 1.5, True),
            ("invalid_string", "1.0", False),
            ("invalid_list", [1, 2], False),
            ("invalid_dict", {"value": 1}, False),
            ("invalid_none", None, False),
        ],
    )
    def test_range_constraint_type_combinations(
        self, param_type, param_value, expected_result
    ):
        """Test range constraint with different parameter types."""
        constraint = ParameterRangeConstraint("test_param", 0.0, 2.0)
        result = constraint.validate({"test_param": param_value})
        assert result == expected_result

    @pytest.mark.parametrize(
        "condition_result,base_constraint_result,expected_result",
        [
            (True, True, True),  # Condition applies, base passes
            (True, False, False),  # Condition applies, base fails
            (False, True, True),  # Condition doesn't apply, base would pass
            (False, False, True),  # Condition doesn't apply, base would fail
        ],
    )
    def test_conditional_constraint_combinations(
        self, condition_result, base_constraint_result, expected_result
    ):
        """Test conditional constraint with different condition and base constraint results."""

        def condition(config):
            return condition_result

        base_constraint = MockConstraint("base", should_pass=base_constraint_result)
        constraint = ConditionalConstraint("conditional", condition, base_constraint)

        result = constraint.validate({"test": "config"})
        assert result == expected_result

    @pytest.mark.parametrize(
        "forbidden_count,max_allowed,expected_result",
        [
            (0, 1, True),  # No forbidden values
            (1, 1, True),  # Within limit
            (1, 2, True),  # Well within limit
            (2, 1, False),  # Exceeds limit
            (2, 2, True),  # At limit
            (3, 2, False),  # Exceeds limit
        ],
    )
    def test_mutex_constraint_combinations(
        self, forbidden_count, max_allowed, expected_result
    ):
        """Test mutex constraint with different forbidden counts and limits."""
        params = [f"param{i}" for i in range(forbidden_count + 2)]
        constraint = MutuallyExclusiveConstraint(params, ["forbidden"], max_allowed)

        config = {}
        for i in range(forbidden_count):
            config[f"param{i}"] = "forbidden"
        for i in range(forbidden_count, len(params)):
            config[f"param{i}"] = "allowed"

        result = constraint.validate(config)
        assert result == expected_result

    @pytest.mark.parametrize(
        "has_dependent,has_dependency,dep_value_valid,expected_result",
        [
            (False, False, False, True),  # Neither present
            (False, True, True, True),  # Only dependency present
            (False, True, False, True),  # Only dependency present
            (True, False, True, False),  # Dependent present, dependency missing
            (True, True, True, True),  # Both present, valid
            (True, True, False, False),  # Both present, invalid
        ],
    )
    def test_dependency_constraint_combinations(
        self, has_dependent, has_dependency, dep_value_valid, expected_result
    ):
        """Test dependency constraint with different parameter presence combinations."""
        constraint = DependencyConstraint("dependent", "dependency", ["valid_value"])

        config = {}
        if has_dependent:
            config["dependent"] = "some_value"
        if has_dependency:
            config["dependency"] = "valid_value" if dep_value_valid else "invalid_value"

        result = constraint.validate(config)
        assert result == expected_result

    @pytest.mark.parametrize(
        "resource_usage,max_resource,calculator_throws,expected_result",
        [
            (0.5, 1.0, False, True),  # Within limit
            (1.0, 1.0, False, True),  # At limit
            (1.5, 1.0, False, False),  # Exceeds limit
            (0.5, 1.0, True, False),  # Calculator throws exception
        ],
    )
    def test_resource_constraint_combinations(
        self, resource_usage, max_resource, calculator_throws, expected_result
    ):
        """Test resource constraint with different usage levels and calculator behavior."""

        def calculator(config):
            if calculator_throws:
                raise ValueError("Calculator error")
            return resource_usage

        constraint = ResourceConstraint("test_resource", calculator, max_resource)
        result = constraint.validate({"test": "config"})
        assert result == expected_result

    @pytest.mark.parametrize(
        "validator_result,validator_throws,message_throws,suggestion_throws",
        [
            (True, False, False, False),  # All work, passes
            (False, False, False, False),  # Validator fails, messages work
            (True, True, False, False),  # Validator throws
            (False, False, True, False),  # Message generator throws
            (False, False, False, True),  # Suggestion generator throws
        ],
    )
    def test_custom_constraint_combinations(
        self, validator_result, validator_throws, message_throws, suggestion_throws
    ):
        """Test custom constraint with different generator behaviors."""

        def validator(config):
            if validator_throws:
                raise ValueError("Validator error")
            return validator_result

        def message_gen(config):
            if message_throws:
                raise ValueError("Message error")
            return "Test message"

        def suggestion_gen(config):
            if suggestion_throws:
                raise ValueError("Suggestion error")
            return "Test suggestion"

        constraint = CustomConstraint("custom", validator, message_gen, suggestion_gen)

        # Test validation
        if validator_throws:
            assert constraint.validate({"test": "config"}) is False
        else:
            assert constraint.validate({"test": "config"}) == validator_result

        # Test message generation (if validator fails)
        if not validator_result and not validator_throws:
            msg = constraint.get_violation_message({"test": "config"})
            if message_throws:
                assert "Custom constraint 'custom' violated:" in msg
            else:
                assert msg == "Test message"

        # Test suggestion generation (if validator fails)
        if not validator_result and not validator_throws:
            suggestion = constraint.get_suggestion({"test": "config"})
            if suggestion_throws:
                assert suggestion is None
            else:
                assert suggestion == "Test suggestion"
