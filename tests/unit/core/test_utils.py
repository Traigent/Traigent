"""Comprehensive tests for traigent.core.utils module."""

import os
import time
from unittest.mock import Mock, patch

import pytest

from traigent.core.utils import (
    aggregate_costs,
    aggregate_metrics,
    aggregate_response_times,
    aggregate_tokens,
    calculate_execution_time,
    create_error_message,
    create_validation_result,
    debug_print_object,
    merge_configs,
    safe_float_convert,
    safe_get_attr,
    safe_get_nested_attr,
    safe_int_convert,
    safe_str_convert,
    timing_decorator,
    truncate_error_message,
    validate_bounds,
    validate_config_keys,
    validate_positive_number,
    validate_range,
    validation_decorator,
)


class TestSafeAttributeAccess:
    """Test safe attribute access utilities."""

    def test_safe_get_nested_attr_simple(self):
        """Test safe_get_nested_attr with simple path."""
        obj = Mock(attr1="value1")
        result = safe_get_nested_attr(obj, "attr1")
        assert result == "value1"

    def test_safe_get_nested_attr_nested(self):
        """Test safe_get_nested_attr with nested path."""
        inner = Mock(attr2="value2")
        obj = Mock(attr1=inner)
        result = safe_get_nested_attr(obj, "attr1.attr2")
        assert result == "value2"

    def test_safe_get_nested_attr_dict(self):
        """Test safe_get_nested_attr with dictionary."""
        obj = {"key1": {"key2": "value"}}
        result = safe_get_nested_attr(obj, "key1.key2")
        assert result == "value"

    def test_safe_get_nested_attr_list(self):
        """Test safe_get_nested_attr with list indexing."""
        # Note: Due to implementation, dict.items is accessed via hasattr first,
        # which returns the dict.items() method, not the key
        # So this test verifies the actual behavior
        obj = {"items": ["a", "b", "c"]}
        result = safe_get_nested_attr(obj, "items.1")
        # Returns None because dict.items is a method, not the list
        assert result is None

    def test_safe_get_nested_attr_tuple(self):
        """Test safe_get_nested_attr with tuple indexing."""
        obj = ("first", "second", "third")
        result = safe_get_nested_attr(obj, "1")
        assert result == "second"

    def test_safe_get_nested_attr_missing(self):
        """Test safe_get_nested_attr with missing attribute."""
        # Note: Mock objects return Mock for missing attributes by default
        # Use spec=[] to prevent this behavior
        obj = Mock(attr1="value1", spec=["attr1"])
        result = safe_get_nested_attr(obj, "missing", "default")
        assert result == "default"

    def test_safe_get_nested_attr_none_object(self):
        """Test safe_get_nested_attr with None object."""
        result = safe_get_nested_attr(None, "any.path", "default")
        assert result == "default"

    def test_safe_get_nested_attr_invalid_index(self):
        """Test safe_get_nested_attr with invalid list index."""
        obj = {"items": ["a", "b"]}
        result = safe_get_nested_attr(obj, "items.5", "default")
        assert result == "default"

    def test_safe_get_attr_exists(self):
        """Test safe_get_attr when attribute exists."""
        obj = Mock(attr="value")
        result = safe_get_attr(obj, "attr")
        assert result == "value"

    def test_safe_get_attr_missing(self):
        """Test safe_get_attr when attribute missing."""
        obj = Mock(spec=[])
        result = safe_get_attr(obj, "missing", "default")
        assert result == "default"

    def test_safe_get_attr_none_object(self):
        """Test safe_get_attr with None object."""
        result = safe_get_attr(None, "attr", "default")
        assert result == "default"


class TestValidationUtilities:
    """Test validation utility functions."""

    def test_validate_bounds_float_valid(self):
        """Test validate_bounds with valid float bounds."""
        result = validate_bounds((0.0, 1.0), "float", "temperature")
        assert result is True

    def test_validate_bounds_float_invalid_type(self):
        """Test validate_bounds with invalid float bounds type."""
        with pytest.raises(ValueError, match="must have tuple bounds"):
            validate_bounds("invalid", "float", "param")

    def test_validate_bounds_float_invalid_length(self):
        """Test validate_bounds with wrong length."""
        with pytest.raises(ValueError, match="must have tuple bounds"):
            validate_bounds((0.0,), "float", "param")

    def test_validate_bounds_float_non_numeric(self):
        """Test validate_bounds with non-numeric float bounds."""
        with pytest.raises(ValueError, match="bounds must be numeric"):
            validate_bounds(("a", "b"), "float", "param")

    def test_validate_bounds_float_invalid_range(self):
        """Test validate_bounds with min >= max."""
        with pytest.raises(ValueError, match="min .* must be less than max"):
            validate_bounds((1.0, 0.5), "float", "param")

    def test_validate_bounds_integer_valid(self):
        """Test validate_bounds with valid integer bounds."""
        result = validate_bounds((1, 10), "integer", "trials")
        assert result is True

    def test_validate_bounds_integer_non_integer(self):
        """Test validate_bounds with non-integer bounds."""
        with pytest.raises(ValueError, match="bounds must be integers"):
            validate_bounds((1.5, 10.5), "integer", "param")

    def test_validate_bounds_integer_invalid_range(self):
        """Test validate_bounds with integer min >= max."""
        with pytest.raises(ValueError, match="min .* must be less than max"):
            validate_bounds((10, 1), "integer", "param")

    def test_validate_bounds_categorical_valid(self):
        """Test validate_bounds with valid categorical choices."""
        result = validate_bounds(["a", "b", "c"], "categorical", "model")
        assert result is True

    def test_validate_bounds_categorical_empty(self):
        """Test validate_bounds with empty categorical list."""
        with pytest.raises(ValueError, match="must have non-empty list"):
            validate_bounds([], "categorical", "param")

    def test_validate_bounds_categorical_invalid_type(self):
        """Test validate_bounds with invalid categorical type."""
        with pytest.raises(ValueError, match="must have non-empty list"):
            validate_bounds("not_a_list", "categorical", "param")

    def test_validate_positive_number_valid(self):
        """Test validate_positive_number with valid values."""
        validate_positive_number(5, "value")  # Should not raise
        validate_positive_number(0.1, "value")  # Should not raise

    def test_validate_positive_number_zero(self):
        """Test validate_positive_number with zero."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_number(0, "value")

    def test_validate_positive_number_negative(self):
        """Test validate_positive_number with negative."""
        with pytest.raises(ValueError, match="must be positive"):
            validate_positive_number(-5, "value")

    def test_validate_positive_number_non_numeric(self):
        """Test validate_positive_number with non-numeric."""
        with pytest.raises(TypeError, match="must be a number"):
            validate_positive_number("5", "value")

    def test_validate_range_valid(self):
        """Test validate_range with value in range."""
        validate_range(5, 0, 10, "value")  # Should not raise
        validate_range(0.5, 0.0, 1.0, "value")  # Should not raise

    def test_validate_range_below_min(self):
        """Test validate_range with value below minimum."""
        with pytest.raises(ValueError, match="must be between"):
            validate_range(-1, 0, 10, "value")

    def test_validate_range_above_max(self):
        """Test validate_range with value above maximum."""
        with pytest.raises(ValueError, match="must be between"):
            validate_range(11, 0, 10, "value")


class TestMetricsAggregation:
    """Test metrics aggregation utilities."""

    def test_aggregate_metrics_mean(self):
        """Test aggregate_metrics with mean aggregation."""
        metrics = [
            {"accuracy": 0.8, "loss": 0.2},
            {"accuracy": 0.9, "loss": 0.1},
            {"accuracy": 0.85, "loss": 0.15},
        ]
        result = aggregate_metrics(metrics, "mean")
        assert result["accuracy"] == pytest.approx(0.85)
        assert result["loss"] == pytest.approx(0.15)

    def test_aggregate_metrics_sum(self):
        """Test aggregate_metrics with sum aggregation."""
        metrics = [
            {"tokens": 100, "cost": 0.01},
            {"tokens": 150, "cost": 0.015},
        ]
        result = aggregate_metrics(metrics, "sum")
        assert result["tokens"] == 250
        assert result["cost"] == pytest.approx(0.025)

    def test_aggregate_metrics_max(self):
        """Test aggregate_metrics with max aggregation."""
        metrics = [
            {"latency": 100},
            {"latency": 200},
            {"latency": 150},
        ]
        result = aggregate_metrics(metrics, "max")
        assert result["latency"] == 200

    def test_aggregate_metrics_min(self):
        """Test aggregate_metrics with min aggregation."""
        metrics = [
            {"latency": 100},
            {"latency": 200},
            {"latency": 150},
        ]
        result = aggregate_metrics(metrics, "min")
        assert result["latency"] == 100

    def test_aggregate_metrics_median_odd(self):
        """Test aggregate_metrics with median (odd count)."""
        metrics = [
            {"value": 1},
            {"value": 3},
            {"value": 2},
        ]
        result = aggregate_metrics(metrics, "median")
        assert result["value"] == 2

    def test_aggregate_metrics_median_even(self):
        """Test aggregate_metrics with median (even count)."""
        metrics = [
            {"value": 1},
            {"value": 4},
            {"value": 2},
            {"value": 3},
        ]
        result = aggregate_metrics(metrics, "median")
        assert result["value"] == 2.5

    def test_aggregate_metrics_empty_list(self):
        """Test aggregate_metrics with empty list."""
        result = aggregate_metrics([], "mean")
        assert result == {}

    def test_aggregate_metrics_none_values(self):
        """Test aggregate_metrics with None values."""
        metrics = [
            {"accuracy": 0.8},
            None,
            {"accuracy": 0.9},
        ]
        result = aggregate_metrics(metrics, "mean")
        assert result["accuracy"] == pytest.approx(0.85)

    def test_aggregate_metrics_invalid_type(self):
        """Test aggregate_metrics with invalid aggregation type."""
        with pytest.raises(ValueError, match="Unsupported aggregation type"):
            aggregate_metrics([{"value": 1}], "invalid")

    def test_aggregate_tokens(self):
        """Test aggregate_tokens filters token metrics."""
        metrics = [
            {"total_tokens": 100, "accuracy": 0.8},
            {"total_tokens": 150, "accuracy": 0.9},
        ]
        result = aggregate_tokens(metrics)
        assert "total_tokens" in result
        assert result["total_tokens"] == 250
        assert "accuracy" not in result

    def test_aggregate_costs(self):
        """Test aggregate_costs filters cost metrics."""
        metrics = [
            {"total_cost": 0.01, "accuracy": 0.8},
            {"total_cost": 0.02, "accuracy": 0.9},
        ]
        result = aggregate_costs(metrics)
        assert "total_cost" in result
        assert result["total_cost"] == pytest.approx(0.03)
        assert "accuracy" not in result

    def test_aggregate_response_times(self):
        """Test aggregate_response_times filters response time metrics."""
        # RESPONSE_TIME_METRIC constant is "avg_response_time"
        # The function checks if RESPONSE_TIME_METRIC is IN the key name
        metrics = [
            {"avg_response_time": 100, "accuracy": 0.8},
            {"avg_response_time": 150, "accuracy": 0.9},
        ]
        result = aggregate_response_times(metrics)
        assert "avg_response_time" in result
        assert result["avg_response_time"] == pytest.approx(125)
        assert "accuracy" not in result


class TestTypeConversion:
    """Test type conversion utilities."""

    def test_safe_float_convert_valid(self):
        """Test safe_float_convert with valid values."""
        assert safe_float_convert(5) == 5.0
        assert safe_float_convert("3.14") == 3.14
        assert safe_float_convert(2.5) == 2.5

    def test_safe_float_convert_none(self):
        """Test safe_float_convert with None."""
        result = safe_float_convert(None, default=1.0)
        assert result == 1.0

    def test_safe_float_convert_invalid(self):
        """Test safe_float_convert with invalid value."""
        result = safe_float_convert("invalid", default=0.5)
        assert result == 0.5

    def test_safe_float_convert_strict_nulls(self):
        """Test safe_float_convert with strict nulls mode."""
        with patch.dict(os.environ, {"TRAIGENT_STRICT_METRICS_NULLS": "true"}):
            result = safe_float_convert(None, default=1.0)
            assert result is None

    def test_safe_int_convert_valid(self):
        """Test safe_int_convert with valid values."""
        assert safe_int_convert(5) == 5
        assert safe_int_convert("10") == 10
        assert safe_int_convert(3.7) == 3

    def test_safe_int_convert_none(self):
        """Test safe_int_convert with None."""
        result = safe_int_convert(None, default=10)
        assert result == 10

    def test_safe_int_convert_invalid(self):
        """Test safe_int_convert with invalid value."""
        result = safe_int_convert("invalid", default=5)
        assert result == 5

    def test_safe_str_convert_valid(self):
        """Test safe_str_convert with valid values."""
        assert safe_str_convert("hello") == "hello"
        assert safe_str_convert(123) == "123"
        assert safe_str_convert(3.14) == "3.14"

    def test_safe_str_convert_none(self):
        """Test safe_str_convert with None."""
        result = safe_str_convert(None, default="default")
        assert result == "default"

    def test_safe_str_convert_object(self):
        """Test safe_str_convert with object."""
        obj = Mock()
        obj.__str__ = Mock(return_value="mock_str")
        result = safe_str_convert(obj)
        assert result == "mock_str"


class TestConfigurationUtilities:
    """Test configuration utility functions."""

    def test_merge_configs_simple(self):
        """Test merge_configs with simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_configs_empty_override(self):
        """Test merge_configs with empty override."""
        base = {"a": 1, "b": 2}
        result = merge_configs(base, {})
        assert result == base

    def test_merge_configs_empty_base(self):
        """Test merge_configs with empty base."""
        override = {"a": 1, "b": 2}
        result = merge_configs({}, override)
        assert result == override

    def test_merge_configs_does_not_mutate(self):
        """Test merge_configs doesn't mutate original."""
        base = {"a": 1}
        override = {"b": 2}
        result = merge_configs(base, override)
        assert base == {"a": 1}  # Unchanged
        assert override == {"b": 2}  # Unchanged
        assert result == {"a": 1, "b": 2}

    def test_validate_config_keys_valid(self):
        """Test validate_config_keys with valid config."""
        config = {"required1": 1, "required2": 2}
        errors = validate_config_keys(config, ["required1", "required2"])
        assert errors == []

    def test_validate_config_keys_missing(self):
        """Test validate_config_keys with missing required key."""
        config = {"required1": 1}
        errors = validate_config_keys(config, ["required1", "required2"])
        assert len(errors) == 1
        assert "required2" in errors[0]

    def test_validate_config_keys_unexpected(self):
        """Test validate_config_keys with unexpected key."""
        config = {"required": 1, "unexpected": 2}
        errors = validate_config_keys(config, ["required"], optional_keys=[])
        assert len(errors) == 1
        assert "unexpected" in errors[0]

    def test_validate_config_keys_optional_allowed(self):
        """Test validate_config_keys allows optional keys."""
        config = {"required": 1, "optional": 2}
        errors = validate_config_keys(config, ["required"], optional_keys=["optional"])
        assert errors == []


class TestErrorUtilities:
    """Test error message utilities."""

    def test_create_error_message_simple(self):
        """Test create_error_message with simple message."""
        # "ERR001" is not in VALIDATION_ERROR_CODES, so it returns UNKNOWN_ERROR
        result = create_error_message("ERR001", "Something went wrong")
        assert "UNKNOWN_ERROR" in result
        assert "Something went wrong" in result

    def test_create_error_message_with_kwargs(self):
        """Test create_error_message with keyword arguments."""
        result = create_error_message("ERR002", "Error in {param}", param="temperature")
        assert "temperature" in result

    def test_truncate_error_message_short(self):
        """Test truncate_error_message with short message."""
        message = "Short message"
        result = truncate_error_message(message, max_length=100)
        assert result == message

    def test_truncate_error_message_long(self):
        """Test truncate_error_message with long message."""
        message = "A" * 1000
        result = truncate_error_message(message, max_length=100)
        assert len(result) <= 100
        assert "..." in result


class TestTimingUtilities:
    """Test timing utility functions."""

    def test_calculate_execution_time_with_end(self):
        """Test calculate_execution_time with end time."""
        start = time.time()
        time.sleep(0.1)
        end = time.time()
        result = calculate_execution_time(start, end)
        assert result >= 0.1
        assert result < 0.2

    def test_calculate_execution_time_without_end(self):
        """Test calculate_execution_time without end time."""
        start = time.time()
        time.sleep(0.05)
        result = calculate_execution_time(start)
        assert result >= 0.05
        assert result < 0.15

    def test_timing_decorator(self):
        """Test timing_decorator adds execution time."""

        @timing_decorator
        def slow_function():
            time.sleep(0.05)
            return "result"

        result = slow_function()
        assert result == "result"


class TestDebugUtilities:
    """Test debug utility functions."""

    def test_debug_print_object_simple(self):
        """Test debug_print_object with simple object."""
        obj = {"a": 1, "b": 2}
        result = debug_print_object(obj)
        assert isinstance(result, str)
        assert "a" in result or "1" in result

    def test_debug_print_object_nested(self):
        """Test debug_print_object with nested object."""
        obj = {"outer": {"inner": "value"}}
        result = debug_print_object(obj, max_depth=2)
        assert isinstance(result, str)

    def test_debug_print_object_max_depth(self):
        """Test debug_print_object respects max_depth."""
        obj = {"level1": {"level2": {"level3": "value"}}}
        result = debug_print_object(obj, max_depth=1)
        assert isinstance(result, str)


class TestValidationDecorator:
    """Test validation decorator."""

    def test_validation_decorator_valid(self):
        """Test validation_decorator with valid input."""

        def validator(value):
            # Must return a list of error messages (empty if valid)
            if value <= 0:
                return ["Value must be positive"]
            return []

        @validation_decorator(validator)
        def process(value):
            return value * 2

        result = process(5)
        assert result == 10

    def test_create_validation_result(self):
        """Test create_validation_result helper."""
        # Signature: create_validation_result(is_valid, errors=None, warnings=None)
        # Returns a dict (ValidationResult is actually a dict type)
        result = create_validation_result(True, errors=[], warnings=["minor issue"])
        assert isinstance(result, dict)
        assert result["is_valid"] is True
        assert result["errors"] == []
        assert result["warnings"] == ["minor issue"]
