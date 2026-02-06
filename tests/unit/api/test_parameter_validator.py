"""
Unit tests for parameter validation module.
"""

import pytest

from traigent.api.parameter_validator import (
    OptimizeParameters,
    ParameterValidator,
    validate_optimize_parameters,
)
from traigent.config.types import InjectionMode
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.exceptions import ValidationError


class TestOptimizeParameters:
    """Test OptimizeParameters dataclass."""

    def test_default_values(self):
        """Test default parameter values."""
        params = OptimizeParameters()

        assert params.eval_dataset is None
        assert params.objectives is None
        assert params.injection_mode == InjectionMode.CONTEXT
        assert params.execution_mode == "edge_analytics"
        assert params.minimal_logging is True
        assert params.kwargs == {}


class TestParameterValidator:
    """Test ParameterValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ParameterValidator()

    def test_validate_execution_mode_valid(self):
        """Test validation of valid execution modes.

        Only edge_analytics is currently supported. cloud/hybrid raise
        ConfigurationError (not yet supported), privacy/standard raise
        ConfigurationError (removed).
        """
        # Only edge_analytics is valid
        result = self.validator._validate_execution_mode("edge_analytics")
        assert result is None, "Expected None for valid mode edge_analytics"

    def test_validate_execution_mode_invalid(self):
        """Test validation of invalid execution modes."""
        invalid_modes = ["invalid", "remote", "unsupported"]

        for mode in invalid_modes:
            with pytest.raises(ValidationError) as exc_info:
                self.validator._validate_execution_mode(mode)
            assert "Invalid execution_mode" in str(exc_info.value)
            assert mode in str(exc_info.value)

    def test_validate_injection_mode_enum(self):
        """Test validation of injection mode enum values."""
        # Note: InjectionMode.ATTRIBUTE was removed in v2.x
        valid_modes = [
            InjectionMode.CONTEXT,
            InjectionMode.PARAMETER,
            InjectionMode.SEAMLESS,
        ]

        for mode in valid_modes:
            # Should not raise exception - validate returns None on success
            result = self.validator._validate_injection_mode(mode)
            assert result is None, f"Expected None for valid mode {mode}"

    def test_validate_injection_mode_string(self):
        """Test validation of injection mode string values."""
        # Note: "attribute" was removed in v2.x
        valid_strings = ["context", "parameter", "seamless"]

        for mode_str in valid_strings:
            # Should not raise exception - validate returns None on success
            result = self.validator._validate_injection_mode(mode_str)
            assert result is None, f"Expected None for valid mode {mode_str}"

    def test_validate_injection_mode_invalid(self):
        """Test validation of invalid injection modes."""
        invalid_modes = ["invalid", "direct", "override"]

        for mode in invalid_modes:
            with pytest.raises(ValidationError) as exc_info:
                self.validator._validate_injection_mode(mode)
            assert "Invalid injection_mode" in str(exc_info.value)

    def test_validate_dataset_string(self):
        """Test validation of string dataset paths."""
        valid_paths = ["test.jsonl", "/path/to/data.jsonl", "dataset.json"]

        for path in valid_paths:
            # Should not raise exception - validate returns None on success
            result = self.validator._validate_dataset(path)
            assert result is None, f"Expected None for valid path {path}"

    def test_validate_dataset_list(self):
        """Test validation of dataset path lists."""
        valid_lists = [
            ["test1.jsonl", "test2.jsonl"],
            ["/path/to/data1.jsonl"],
            ["a.json", "b.json", "c.json"],
            [
                Dataset(
                    examples=[
                        EvaluationExample(
                            input_data="prompt",
                            expected_output="response",
                            metadata={},
                        )
                    ],
                    name="single",
                )
            ],
        ]

        for dataset_list in valid_lists:
            # Should not raise exception - validate returns None on success
            result = self.validator._validate_dataset(dataset_list)
            assert result is None, "Expected None for valid dataset list"

    def test_validate_dataset_sequence_with_dataset_objects(self):
        """Dataset iterables can contain Dataset instances."""
        dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data="prompt",
                    expected_output="answer",
                    metadata={"source": "unit-test"},
                )
            ],
            name="example",
        )

        # Should accept tuples and mixed entries containing Dataset objects
        result1 = self.validator._validate_dataset((dataset,))
        result2 = self.validator._validate_dataset(["path.jsonl", dataset])
        assert result1 is None, "Expected None for tuple with Dataset"
        assert result2 is None, "Expected None for mixed list with Dataset"

    def test_validate_dataset_invalid_list(self):
        """Test validation of invalid dataset lists."""
        invalid_lists = [
            [123, "test.jsonl"],  # Mixed types
            [None, "test.jsonl"],  # None in list
            ["test.jsonl", 456],  # Mixed types
        ]

        for invalid_list in invalid_lists:
            with pytest.raises(ValidationError) as exc_info:
                self.validator._validate_dataset(invalid_list)
            assert "must contain only string paths" in str(exc_info.value)

    def test_validate_objectives_valid(self):
        """Test validation of valid objectives."""
        valid_objectives = [
            ["accuracy"],
            ["accuracy", "cost"],
            ["accuracy", "cost", "latency"],
            ["f1_score", "precision", "recall"],
            ["custom_metric1", "custom_metric2"],
        ]

        for objectives in valid_objectives:
            # Should not raise exception - validate returns None on success
            result = self.validator._validate_objectives(objectives)
            assert result is None, f"Expected None for valid objectives {objectives}"

    def test_validate_objectives_invalid_type(self):
        """Test validation of invalid objective types."""
        invalid_objectives = [
            "accuracy",  # String instead of list
            ["accuracy", 123],  # Mixed types
            [None, "accuracy"],  # None in list
        ]

        for objectives in invalid_objectives:
            with pytest.raises(ValidationError):
                self.validator._validate_objectives(objectives)

    def test_validate_configuration_space_valid(self):
        """Test validation of valid configuration spaces."""
        valid_spaces = [
            {"model": ["gpt-3.5", "gpt-4"]},
            {"temperature": (0.1, 1.0)},
            {"nested": {"param": ["a", "b"]}},
            {"multi": ["a", "b"], "range": (1, 10)},
        ]

        for space in valid_spaces:
            # Should not raise exception - validate returns None on success
            result = self.validator._validate_configuration_space(space)
            assert result is None, f"Expected None for valid config space {space}"

    def test_validate_configuration_space_invalid(self):
        """Test validation of invalid configuration spaces."""
        invalid_spaces = [
            {"param": []},  # Empty list
            {"param": (1,)},  # Single tuple value
            {"param": "invalid"},  # String value
            {123: ["a", "b"]},  # Non-string key
        ]

        for space in invalid_spaces:
            with pytest.raises(ValidationError):
                self.validator._validate_configuration_space(space)

    def test_validate_constraints_valid(self):
        """Test validation of valid constraint functions."""

        def constraint1(config):
            return True

        def constraint2(config, metrics):
            return False

        valid_constraints = [
            [constraint1],
            [constraint1, constraint2],
            [lambda x: True, lambda x, y: False],
        ]

        for constraints in valid_constraints:
            # Should not raise exception - validate returns None on success
            result = self.validator._validate_constraints(constraints)
            assert result is None, "Expected None for valid constraints"

    def test_validate_constraints_invalid(self):
        """Test validation of invalid constraints."""
        invalid_constraints = [
            ["not_callable"],  # String instead of function
            [lambda x: True, "not_callable"],  # Mixed types
            [None, lambda x: True],  # None in list
        ]

        for constraints in invalid_constraints:
            with pytest.raises(ValidationError) as exc_info:
                self.validator._validate_constraints(constraints)
            assert "not callable" in str(exc_info.value)

    def test_normalize_injection_mode_string(self):
        """Test normalization of string injection modes."""
        # Note: "attribute" was removed in v2.x
        string_modes = ["context", "parameter", "seamless"]
        expected_enums = [
            InjectionMode.CONTEXT,
            InjectionMode.PARAMETER,
            InjectionMode.SEAMLESS,
        ]

        for string_mode, expected_enum in zip(
            string_modes, expected_enums, strict=False
        ):
            result = self.validator._normalize_injection_mode(string_mode)
            assert result == expected_enum

    def test_normalize_injection_mode_enum(self):
        """Test normalization of enum injection modes."""
        enum_mode = InjectionMode.CONTEXT
        result = self.validator._normalize_injection_mode(enum_mode)
        assert result == enum_mode

    def test_validate_parameters_integration(self):
        """Test complete parameter validation integration."""
        # Valid parameters
        params = OptimizeParameters(
            eval_dataset="test.jsonl",
            objectives=["accuracy", "cost"],
            configuration_space={"model": ["gpt-3.5", "gpt-4"]},
            execution_mode="cloud",
            kwargs={"parallel_config": {"example_concurrency": 10}},
        )

        result = self.validator.validate_parameters(params)
        assert result.injection_mode == InjectionMode.CONTEXT
        assert isinstance(result, OptimizeParameters)


class TestValidateOptimizeParameters:
    """Test the convenience function."""

    def test_validate_optimize_parameters_basic(self):
        """Test basic parameter validation."""
        result = validate_optimize_parameters(
            eval_dataset="test.jsonl",
            objectives=["accuracy"],
            execution_mode="edge_analytics",
        )

        assert isinstance(result, OptimizeParameters)
        assert result.eval_dataset == "test.jsonl"
        assert result.objectives == ["accuracy"]
        assert result.execution_mode == "edge_analytics"

    def test_validate_optimize_parameters_with_kwargs(self):
        """Test parameter validation with extra kwargs."""
        result = validate_optimize_parameters(
            eval_dataset="test.jsonl", custom_param="value", another_param=123
        )

        assert result.kwargs["custom_param"] == "value"
        assert result.kwargs["another_param"] == 123

    def test_validate_optimize_parameters_invalid(self):
        """Test parameter validation with invalid parameters."""
        with pytest.raises(ValidationError):
            validate_optimize_parameters(execution_mode="invalid_mode")


@pytest.mark.integration
class TestParameterValidatorIntegration:
    """Integration tests for parameter validator."""

    def test_real_world_parameters(self):
        """Test validation with realistic parameter combinations."""
        # Example from documentation
        params = validate_optimize_parameters(
            eval_dataset=["qa_test.jsonl", "qa_validation.jsonl"],
            objectives=["accuracy", "cost", "latency"],
            configuration_space={
                "model": ["gpt-3.5-turbo", "gpt-4"],
                "temperature": (0.1, 1.0),
                "max_tokens": [100, 500, 1000],
            },
            constraints=[lambda config: config.get("temperature", 0) < 0.9],
            execution_mode="privacy",
            parallel_config={
                "example_concurrency": 5,
                "trial_concurrency": 3,
            },
            custom_evaluator="my_evaluator",
        )

        assert len(params.eval_dataset) == 2
        assert len(params.objectives) == 3
        assert "model" in params.configuration_space
        assert len(params.constraints) == 1
        assert params.execution_mode == "privacy"
        assert params.kwargs["parallel_config"]["example_concurrency"] == 5
        assert params.kwargs["parallel_config"]["trial_concurrency"] == 3
        assert params.kwargs["custom_evaluator"] == "my_evaluator"


class TestConfigValueHelpers:
    """Test helper methods for configuration value validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ParameterValidator()

    def test_validate_list_value_valid(self):
        """Test validation of valid list values."""
        assert self.validator._validate_list_value("key", ["a", "b"]) is None
        assert self.validator._validate_list_value("key", [1]) is None

    def test_validate_list_value_empty(self):
        """Test validation of empty list."""
        error = self.validator._validate_list_value("model", [])
        assert error is not None
        assert "empty list" in error
        assert "model" in error

    def test_validate_tuple_value_valid(self):
        """Test validation of valid tuple ranges."""
        assert self.validator._validate_tuple_value("temp", (0.0, 1.0)) is None
        assert self.validator._validate_tuple_value("count", (1, 100)) is None

    def test_validate_tuple_value_wrong_length(self):
        """Test validation of tuples with wrong number of elements."""
        error = self.validator._validate_tuple_value("key", (1,))
        assert error is not None
        assert "invalid range tuple" in error
        assert "1 elements" in error

        error = self.validator._validate_tuple_value("key", (1, 2, 3))
        assert error is not None
        assert "3 elements" in error

    def test_validate_tuple_value_non_numeric(self):
        """Test validation of tuples with non-numeric values."""
        error = self.validator._validate_tuple_value("key", ("a", "b"))
        assert error is not None
        assert "non-numeric" in error

    def test_validate_tuple_value_inverted_range(self):
        """Test validation of tuple with min > max."""
        error = self.validator._validate_tuple_value("temp", (1.0, 0.0))
        assert error is not None
        assert "min > max" in error
        assert "Did you mean" in error

    def test_validate_tuple_value_point_range(self):
        """Test validation of tuple with min == max."""
        error = self.validator._validate_tuple_value("temp", (0.5, 0.5))
        assert error is not None
        assert "point range" in error
        assert "min equals max" in error

    def test_validate_dict_value_valid(self):
        """Test validation of valid dict values."""
        # Nested config
        assert self.validator._validate_dict_value("nested", {"param": ["a"]}) is None
        # Typed parameter
        assert (
            self.validator._validate_dict_value(
                "temp", {"type": "float", "low": 0.0, "high": 1.0}
            )
            is None
        )
        assert (
            self.validator._validate_dict_value(
                "count", {"type": "int", "low": 1, "high": 10}
            )
            is None
        )
        assert (
            self.validator._validate_dict_value(
                "model", {"type": "categorical", "values": ["a", "b"]}
            )
            is None
        )
        assert (
            self.validator._validate_dict_value("fixed", {"type": "fixed", "value": 1})
            is None
        )

    def test_validate_dict_value_empty(self):
        """Test validation of empty dict."""
        error = self.validator._validate_dict_value("key", {})
        assert error is not None
        assert "empty nested configuration" in error

    def test_validate_dict_value_unknown_type(self):
        """Test validation of dict with unknown type."""
        error = self.validator._validate_dict_value("key", {"type": "unknown"})
        assert error is not None
        assert "unknown type" in error

    def test_validate_typed_dict_bounds_missing(self):
        """Test validation of typed dict missing bounds."""
        error = self.validator._validate_typed_dict_bounds(
            "temp", {"low": 0.0}, "float"
        )
        assert error is not None
        assert "'low' and 'high'" in error

        error = self.validator._validate_typed_dict_bounds(
            "temp", {"high": 1.0}, "float"
        )
        assert error is not None

    def test_validate_typed_dict_bounds_non_numeric(self):
        """Test validation of typed dict with non-numeric bounds."""
        error = self.validator._validate_typed_dict_bounds(
            "key", {"low": "a", "high": 1.0}, "float"
        )
        assert error is not None
        assert "non-numeric bounds" in error

    def test_validate_typed_dict_bounds_invalid(self):
        """Test validation of typed dict with low >= high."""
        error = self.validator._validate_typed_dict_bounds(
            "key", {"low": 1.0, "high": 0.5}, "float"
        )
        assert error is not None
        assert "less than high" in error

        # Equal bounds
        error = self.validator._validate_typed_dict_bounds(
            "key", {"low": 0.5, "high": 0.5}, "int"
        )
        assert error is not None

    def test_get_config_value_error_invalid_type(self):
        """Test error message for invalid config value types."""
        error = self.validator._get_config_value_error("key", "string_value")
        assert error is not None
        assert "invalid type" in error
        assert "str" in error

        error = self.validator._get_config_value_error("key", 123)
        assert error is not None
        assert "int" in error

    def test_is_valid_config_value(self):
        """Test _is_valid_config_value method."""
        # Valid list
        assert self.validator._is_valid_config_value(["a", "b"]) is True
        # Empty list
        assert self.validator._is_valid_config_value([]) is False
        # Valid tuple
        assert self.validator._is_valid_config_value((0.0, 1.0)) is True
        # Invalid tuple length
        assert self.validator._is_valid_config_value((1,)) is False
        # Non-numeric tuple
        assert self.validator._is_valid_config_value(("a", "b")) is False
        # Invalid range (min >= max)
        assert self.validator._is_valid_config_value((1.0, 0.0)) is False
        assert self.validator._is_valid_config_value((0.5, 0.5)) is False
        # Valid dict
        assert self.validator._is_valid_config_value({"a": [1]}) is True
        # Empty dict
        assert self.validator._is_valid_config_value({}) is False
        # Invalid type
        assert self.validator._is_valid_config_value("string") is False
        assert self.validator._is_valid_config_value(123) is False


class TestDeprecatedParameters:
    """Test handling of deprecated/removed parameters."""

    def test_removed_auto_optimize_parameter(self):
        """Test that removed auto_optimize parameter raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optimize_parameters(auto_optimize=True)
        assert "auto_optimize" in str(exc_info.value)
        assert "removed" in str(exc_info.value)

    def test_removed_trigger_parameter(self):
        """Test that removed trigger parameter raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optimize_parameters(trigger="on_call")
        assert "trigger" in str(exc_info.value)

    def test_removed_batch_size_parameter(self):
        """Test that removed batch_size parameter raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optimize_parameters(batch_size=10)
        assert "batch_size" in str(exc_info.value)

    def test_removed_parallel_trials_parameter(self):
        """Test that removed parallel_trials parameter raises error."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optimize_parameters(parallel_trials=4)
        assert "parallel_trials" in str(exc_info.value)

    def test_multiple_removed_parameters(self):
        """Test error message with multiple removed parameters."""
        with pytest.raises(ValidationError) as exc_info:
            validate_optimize_parameters(auto_optimize=True, trigger="on_call")
        error_msg = str(exc_info.value)
        assert "auto_optimize" in error_msg
        assert "trigger" in error_msg


class TestDatasetValidationEdgeCases:
    """Test edge cases for dataset validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ParameterValidator()

    def test_validate_dataset_none(self):
        """Test that None dataset is valid."""
        assert self.validator._validate_dataset(None) is None

    def test_validate_dataset_single_dataset_object(self):
        """Test validation of single Dataset object."""
        dataset = Dataset(
            examples=[
                EvaluationExample(
                    input_data="test", expected_output="response", metadata={}
                )
            ],
            name="test",
        )
        assert self.validator._validate_dataset(dataset) is None

    def test_validate_dataset_invalid_type(self):
        """Test validation of completely invalid dataset type."""
        with pytest.raises(ValidationError) as exc_info:
            self.validator._validate_dataset(123)
        assert "must be a string path" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            self.validator._validate_dataset({"key": "value"})
        assert "must be a string path" in str(exc_info.value)


class TestConfigurationSpaceEdgeCases:
    """Test edge cases for configuration space validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ParameterValidator()

    def test_empty_configuration_space(self):
        """Test that empty configuration space raises error with helpful message."""
        with pytest.raises(ValidationError) as exc_info:
            self.validator._validate_configuration_space({})
        error_msg = str(exc_info.value)
        assert "cannot be empty" in error_msg
        assert "Example:" in error_msg

    def test_configuration_space_none(self):
        """Test that None configuration space is valid."""
        assert self.validator._validate_configuration_space(None) is None

    def test_configuration_space_not_dict(self):
        """Test that non-dict configuration space raises error."""
        with pytest.raises(ValidationError) as exc_info:
            self.validator._validate_configuration_space(["a", "b"])
        assert "must be a dictionary" in str(exc_info.value)

    def test_configuration_space_integer_type_dict(self):
        """Test typed dict with 'integer' type alias."""
        # 'integer' should be accepted as alias for 'int'
        assert (
            self.validator._validate_dict_value(
                "count", {"type": "integer", "low": 1, "high": 10}
            )
            is None
        )
