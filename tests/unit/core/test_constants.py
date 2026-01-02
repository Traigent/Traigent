"""Comprehensive tests for traigent.core.constants module.

Tests verify all constants are defined, have correct types and values,
and maintain consistency across the system.
"""

from __future__ import annotations

from traigent.core import constants


class TestTimingConstants:
    """Test timing and performance constants."""

    def test_default_timeout_defined(self):
        """Test DEFAULT_TIMEOUT is defined and reasonable."""
        assert hasattr(constants, "DEFAULT_TIMEOUT")
        assert constants.DEFAULT_TIMEOUT == 60.0
        assert isinstance(constants.DEFAULT_TIMEOUT, float)

    def test_default_execution_time_defined(self):
        """Test DEFAULT_EXECUTION_TIME is defined."""
        assert hasattr(constants, "DEFAULT_EXECUTION_TIME")
        assert constants.DEFAULT_EXECUTION_TIME == 0.1
        assert constants.DEFAULT_EXECUTION_TIME > 0

    def test_max_retries_defined(self):
        """Test MAX_RETRIES is defined."""
        assert hasattr(constants, "MAX_RETRIES")
        assert constants.MAX_RETRIES == 3
        assert isinstance(constants.MAX_RETRIES, int)
        assert constants.MAX_RETRIES > 0

    def test_epsilon_defined(self):
        """Test EPSILON is defined for floating point comparisons."""
        assert hasattr(constants, "EPSILON")
        assert constants.EPSILON == 1e-10
        assert constants.EPSILON > 0
        assert constants.EPSILON < 1e-5


class TestLLMConstants:
    """Test LLM and model constants."""

    def test_default_model_defined(self):
        """Test DEFAULT_MODEL is defined."""
        assert hasattr(constants, "DEFAULT_MODEL")
        assert constants.DEFAULT_MODEL == "gpt-4o-mini"
        assert isinstance(constants.DEFAULT_MODEL, str)

    def test_default_prompt_style_defined(self):
        """Test DEFAULT_PROMPT_STYLE is defined."""
        assert hasattr(constants, "DEFAULT_PROMPT_STYLE")
        assert constants.DEFAULT_PROMPT_STYLE == "direct"

    def test_default_temperature_defined(self):
        """Test DEFAULT_TEMPERATURE is defined."""
        assert hasattr(constants, "DEFAULT_TEMPERATURE")
        assert constants.DEFAULT_TEMPERATURE == 0.0
        assert 0.0 <= constants.DEFAULT_TEMPERATURE <= 2.0

    def test_default_max_tokens_defined(self):
        """Test DEFAULT_MAX_TOKENS is defined."""
        assert hasattr(constants, "DEFAULT_MAX_TOKENS")
        assert constants.DEFAULT_MAX_TOKENS == 1000
        assert constants.DEFAULT_MAX_TOKENS > 0

    def test_teach_max_tokens_defined(self):
        """Test TEACH_MAX_TOKENS is defined and larger than default."""
        assert hasattr(constants, "TEACH_MAX_TOKENS")
        assert constants.TEACH_MAX_TOKENS == 1500
        assert constants.TEACH_MAX_TOKENS > constants.DEFAULT_MAX_TOKENS


class TestConfigurationConstants:
    """Test configuration and validation constants."""

    def test_min_objective_weight_defined(self):
        """Test MIN_OBJECTIVE_WEIGHT is defined."""
        assert hasattr(constants, "MIN_OBJECTIVE_WEIGHT")
        assert constants.MIN_OBJECTIVE_WEIGHT == 0.0
        assert constants.MIN_OBJECTIVE_WEIGHT >= 0.0

    def test_default_objective_weight_defined(self):
        """Test DEFAULT_OBJECTIVE_WEIGHT is defined."""
        assert hasattr(constants, "DEFAULT_OBJECTIVE_WEIGHT")
        assert constants.DEFAULT_OBJECTIVE_WEIGHT == 1.0
        assert constants.DEFAULT_OBJECTIVE_WEIGHT > constants.MIN_OBJECTIVE_WEIGHT

    def test_max_test_questions_defined(self):
        """Test MAX_TEST_QUESTIONS is defined."""
        assert hasattr(constants, "MAX_TEST_QUESTIONS")
        assert constants.MAX_TEST_QUESTIONS == 10
        assert constants.MAX_TEST_QUESTIONS > 0

    def test_default_test_questions_defined(self):
        """Test DEFAULT_TEST_QUESTIONS is defined."""
        assert hasattr(constants, "DEFAULT_TEST_QUESTIONS")
        assert constants.DEFAULT_TEST_QUESTIONS == 2
        assert constants.DEFAULT_TEST_QUESTIONS <= constants.MAX_TEST_QUESTIONS


class TestFilePathConstants:
    """Test file and path constants."""

    def test_dataset_file_extension_defined(self):
        """Test DATASET_FILE_EXTENSION is defined."""
        assert hasattr(constants, "DATASET_FILE_EXTENSION")
        assert constants.DATASET_FILE_EXTENSION == ".jsonl"
        assert constants.DATASET_FILE_EXTENSION.startswith(".")

    def test_config_file_name_defined(self):
        """Test CONFIG_FILE_NAME is defined."""
        assert hasattr(constants, "CONFIG_FILE_NAME")
        assert constants.CONFIG_FILE_NAME == "traigent_config.yaml"
        assert ".yaml" in constants.CONFIG_FILE_NAME

    def test_log_directory_defined(self):
        """Test LOG_DIRECTORY is defined."""
        assert hasattr(constants, "LOG_DIRECTORY")
        assert constants.LOG_DIRECTORY == "optimization_logs"
        assert isinstance(constants.LOG_DIRECTORY, str)


class TestMetricsConstants:
    """Test metrics and aggregation constants."""

    def test_token_metrics_defined(self):
        """Test TOKEN_METRICS is defined."""
        assert hasattr(constants, "TOKEN_METRICS")
        assert isinstance(constants.TOKEN_METRICS, list)
        assert "input_tokens" in constants.TOKEN_METRICS
        assert "output_tokens" in constants.TOKEN_METRICS
        assert "total_tokens" in constants.TOKEN_METRICS

    def test_cost_metrics_defined(self):
        """Test COST_METRICS is defined."""
        assert hasattr(constants, "COST_METRICS")
        assert isinstance(constants.COST_METRICS, list)
        assert "input_cost" in constants.COST_METRICS
        assert "output_cost" in constants.COST_METRICS
        assert "total_cost" in constants.COST_METRICS

    def test_response_time_metric_defined(self):
        """Test RESPONSE_TIME_METRIC is defined."""
        assert hasattr(constants, "RESPONSE_TIME_METRIC")
        assert constants.RESPONSE_TIME_METRIC == "avg_response_time"

    def test_aggregation_methods_defined(self):
        """Test AGGREGATION_METHODS is defined."""
        assert hasattr(constants, "AGGREGATION_METHODS")
        assert isinstance(constants.AGGREGATION_METHODS, list)
        assert "sum" in constants.AGGREGATION_METHODS
        assert "mean" in constants.AGGREGATION_METHODS
        assert "max" in constants.AGGREGATION_METHODS
        assert "min" in constants.AGGREGATION_METHODS
        assert "median" in constants.AGGREGATION_METHODS


class TestConstantTypes:
    """Test that constants have correct types."""

    def test_numeric_constants_are_numbers(self):
        """Test numeric constants are int or float."""
        numeric_constants = [
            "DEFAULT_TIMEOUT",
            "DEFAULT_EXECUTION_TIME",
            "MAX_RETRIES",
            "EPSILON",
            "DEFAULT_TEMPERATURE",
            "DEFAULT_MAX_TOKENS",
            "TEACH_MAX_TOKENS",
            "MIN_OBJECTIVE_WEIGHT",
            "DEFAULT_OBJECTIVE_WEIGHT",
            "MAX_TEST_QUESTIONS",
            "DEFAULT_TEST_QUESTIONS",
        ]

        for const_name in numeric_constants:
            value = getattr(constants, const_name)
            assert isinstance(value, (int, float)), f"{const_name} should be numeric"

    def test_string_constants_are_strings(self):
        """Test string constants are str."""
        string_constants = [
            "DEFAULT_MODEL",
            "DEFAULT_PROMPT_STYLE",
            "DATASET_FILE_EXTENSION",
            "CONFIG_FILE_NAME",
            "LOG_DIRECTORY",
            "RESPONSE_TIME_METRIC",
        ]

        for const_name in string_constants:
            value = getattr(constants, const_name)
            assert isinstance(value, str), f"{const_name} should be string"

    def test_list_constants_are_lists(self):
        """Test list constants are lists."""
        list_constants = [
            "TOKEN_METRICS",
            "COST_METRICS",
            "AGGREGATION_METHODS",
        ]

        for const_name in list_constants:
            value = getattr(constants, const_name)
            assert isinstance(value, list), f"{const_name} should be list"


class TestConstantConsistency:
    """Test consistency between related constants."""

    def test_weights_consistency(self):
        """Test objective weight constants are consistent."""
        assert constants.DEFAULT_OBJECTIVE_WEIGHT >= constants.MIN_OBJECTIVE_WEIGHT

    def test_test_questions_consistency(self):
        """Test test question constants are consistent."""
        assert constants.DEFAULT_TEST_QUESTIONS <= constants.MAX_TEST_QUESTIONS
        assert constants.DEFAULT_TEST_QUESTIONS > 0

    def test_token_limits_consistency(self):
        """Test token limit constants are consistent."""
        assert constants.TEACH_MAX_TOKENS > constants.DEFAULT_MAX_TOKENS
        assert constants.DEFAULT_MAX_TOKENS > 0

    def test_timing_consistency(self):
        """Test timing constants are reasonable."""
        assert constants.DEFAULT_TIMEOUT > constants.DEFAULT_EXECUTION_TIME
        assert constants.MAX_RETRIES > 0

    def test_epsilon_reasonable(self):
        """Test EPSILON is small but not zero."""
        assert 0 < constants.EPSILON < 0.001


class TestConstantImmutability:
    """Test that constants module encourages immutability."""

    def test_constants_exist(self):
        """Test that expected constants exist."""
        expected_constants = [
            "DEFAULT_TIMEOUT",
            "DEFAULT_MODEL",
            "TOKEN_METRICS",
            "COST_METRICS",
            "AGGREGATION_METHODS",
        ]

        for const_name in expected_constants:
            assert hasattr(constants, const_name), f"Missing constant: {const_name}"

    def test_list_constants_not_empty(self):
        """Test that list constants are not empty."""
        list_constants = ["TOKEN_METRICS", "COST_METRICS", "AGGREGATION_METHODS"]

        for const_name in list_constants:
            value = getattr(constants, const_name)
            assert len(value) > 0, f"{const_name} should not be empty"


class TestConstantValues:
    """Test specific constant values."""

    def test_retry_count_reasonable(self):
        """Test MAX_RETRIES is reasonable."""
        assert 1 <= constants.MAX_RETRIES <= 10

    def test_timeout_reasonable(self):
        """Test DEFAULT_TIMEOUT is reasonable."""
        assert 1 <= constants.DEFAULT_TIMEOUT <= 600

    def test_temperature_valid_range(self):
        """Test DEFAULT_TEMPERATURE is in valid range."""
        assert 0.0 <= constants.DEFAULT_TEMPERATURE <= 2.0

    def test_file_extensions_valid(self):
        """Test file extensions start with dot."""
        assert constants.DATASET_FILE_EXTENSION.startswith(".")

    def test_aggregation_methods_valid(self):
        """Test aggregation methods are standard statistical operations."""
        valid_methods = ["sum", "mean", "average", "max", "min", "median", "std", "var"]

        for method in constants.AGGREGATION_METHODS:
            assert any(method in valid for valid in valid_methods)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_epsilon_prevents_zero_division(self):
        """Test EPSILON can be used to prevent zero division."""
        # Should not raise ZeroDivisionError
        result = 1.0 / (0.0 + constants.EPSILON)
        assert result > 0

    def test_constants_not_none(self):
        """Test that critical constants are not None."""
        critical_constants = [
            "DEFAULT_TIMEOUT",
            "DEFAULT_MODEL",
            "MAX_RETRIES",
            "EPSILON",
        ]

        for const_name in critical_constants:
            value = getattr(constants, const_name)
            assert value is not None

    def test_positive_numeric_constants(self):
        """Test that positive constants are indeed positive."""
        positive_constants = [
            "DEFAULT_TIMEOUT",
            "DEFAULT_EXECUTION_TIME",
            "MAX_RETRIES",
            "EPSILON",
            "DEFAULT_MAX_TOKENS",
            "TEACH_MAX_TOKENS",
            "DEFAULT_OBJECTIVE_WEIGHT",
            "MAX_TEST_QUESTIONS",
            "DEFAULT_TEST_QUESTIONS",
        ]

        for const_name in positive_constants:
            value = getattr(constants, const_name)
            assert value > 0, f"{const_name} should be positive"

    def test_non_negative_constants(self):
        """Test that non-negative constants are >= 0."""
        non_negative = ["MIN_OBJECTIVE_WEIGHT", "DEFAULT_TEMPERATURE"]

        for const_name in non_negative:
            value = getattr(constants, const_name)
            assert value >= 0, f"{const_name} should be non-negative"


class TestConstantDocumentation:
    """Test that constants module is well-documented."""

    def test_module_has_docstring(self):
        """Test that module has a docstring."""
        assert constants.__doc__ is not None
        assert len(constants.__doc__) > 50

    def test_critical_constants_exist(self):
        """Test that all critical constants are defined."""
        critical = [
            "DEFAULT_TIMEOUT",
            "DEFAULT_MODEL",
            "DEFAULT_TEMPERATURE",
            "MAX_RETRIES",
            "TOKEN_METRICS",
            "COST_METRICS",
        ]

        for const in critical:
            assert hasattr(constants, const)
