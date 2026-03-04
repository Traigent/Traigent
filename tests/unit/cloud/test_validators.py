"""Unit tests for cloud validation functions.

Tests for OptiGen Backend integration validation according to schema specifications.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID REQ-CLOUD-009 SYNC-CloudHybrid

from __future__ import annotations

import pytest

from traigent.cloud.validators import (
    validate_comparability_metadata,
    validate_configuration_run_submission,
    validate_measure_results,
    validate_measures_array,
    validate_summary_stats,
)


class TestValidateMeasureResults:
    """Tests for validate_measure_results function."""

    def test_valid_measure_with_numeric_values(self) -> None:
        """Test measure validation with valid numeric values."""
        measure = {"accuracy": 0.95, "latency": 100, "cost": 0.05}
        assert validate_measure_results(measure) is True

    def test_valid_measure_with_string_values(self) -> None:
        """Test measure validation with valid string values."""
        measure = {"model_name": "gpt-4", "status": "success"}
        assert validate_measure_results(measure) is True

    def test_valid_measure_with_null_values(self) -> None:
        """Test measure validation with null values."""
        measure = {"accuracy": 0.95, "error": None, "warning": None}
        assert validate_measure_results(measure) is True

    def test_valid_measure_with_mixed_types(self) -> None:
        """Test measure validation with mixed valid types."""
        measure = {
            "accuracy": 0.95,
            "model": "gpt-4",
            "tokens": 150,
            "error": None,
        }
        assert validate_measure_results(measure) is True

    def test_valid_measure_with_underscore_prefix(self) -> None:
        """Test measure validation with underscore-prefixed keys."""
        measure = {"_private_metric": 1.0, "_internal_score": 0.5}
        assert validate_measure_results(measure) is True

    def test_valid_measure_with_alphanumeric_keys(self) -> None:
        """Test measure validation with alphanumeric keys."""
        measure = {"metric_v2": 0.95, "score123": 100, "test_ABC_123": 50.5}
        assert validate_measure_results(measure) is True

    def test_empty_measure_dict(self) -> None:
        """Test measure validation with empty dictionary."""
        measure = {}
        assert validate_measure_results(measure) is True

    def test_measure_not_dict_raises_error(self) -> None:
        """Test that non-dict measure raises ValueError."""
        with pytest.raises(ValueError, match="MeasureResults must be a dict"):
            validate_measure_results([1, 2, 3])

    def test_measure_list_raises_error(self) -> None:
        """Test that list measure raises ValueError."""
        with pytest.raises(
            ValueError, match="MeasureResults must be a dict, got <class 'list'>"
        ):
            validate_measure_results(["accuracy", "latency"])

    def test_measure_string_raises_error(self) -> None:
        """Test that string measure raises ValueError."""
        with pytest.raises(ValueError, match="MeasureResults must be a dict"):
            validate_measure_results("not_a_dict")

    def test_measure_none_raises_error(self) -> None:
        """Test that None measure raises ValueError."""
        with pytest.raises(ValueError, match="MeasureResults must be a dict"):
            validate_measure_results(None)

    def test_invalid_key_starting_with_number(self) -> None:
        """Test that key starting with number raises ValueError."""
        measure = {"123invalid": 0.95}
        with pytest.raises(ValueError, match="Invalid measure key '123invalid'"):
            validate_measure_results(measure)

    def test_invalid_key_with_hyphen(self) -> None:
        """Test that key with hyphen raises ValueError."""
        measure = {"invalid-key": 0.95}
        with pytest.raises(ValueError, match="Invalid measure key 'invalid-key'"):
            validate_measure_results(measure)

    def test_invalid_key_with_space(self) -> None:
        """Test that key with space raises ValueError."""
        measure = {"invalid key": 0.95}
        with pytest.raises(ValueError, match="Invalid measure key 'invalid key'"):
            validate_measure_results(measure)

    def test_invalid_key_with_special_chars(self) -> None:
        """Test that key with special characters raises ValueError."""
        measure = {"invalid@key": 0.95}
        with pytest.raises(ValueError, match="Invalid measure key 'invalid@key'"):
            validate_measure_results(measure)

    def test_invalid_key_with_dot(self) -> None:
        """Test that key with dot raises ValueError."""
        measure = {"invalid.key": 0.95}
        with pytest.raises(ValueError, match="Invalid measure key 'invalid.key'"):
            validate_measure_results(measure)

    def test_invalid_value_type_list(self) -> None:
        """Test that list value raises ValueError."""
        measure = {"metrics": [1, 2, 3]}
        with pytest.raises(
            ValueError,
            match="Measure value for 'metrics' must be number, string, or null",
        ):
            validate_measure_results(measure)

    def test_invalid_value_type_dict(self) -> None:
        """Test that dict value raises ValueError."""
        measure = {"nested": {"key": "value"}}
        with pytest.raises(
            ValueError,
            match="Measure value for 'nested' must be number, string, or null",
        ):
            validate_measure_results(measure)

    def test_boolean_value_accepted_as_int(self) -> None:
        """Test that boolean values are accepted (bool is subclass of int)."""
        measure = {"flag": True, "enabled": False}
        assert validate_measure_results(measure) is True

    def test_integer_value_accepted(self) -> None:
        """Test that integer values are accepted."""
        measure = {"count": 42, "tokens": 1000}
        assert validate_measure_results(measure) is True

    def test_float_value_accepted(self) -> None:
        """Test that float values are accepted."""
        measure = {"accuracy": 0.95, "latency": 123.456}
        assert validate_measure_results(measure) is True

    def test_zero_values_accepted(self) -> None:
        """Test that zero values are accepted."""
        measure = {"cost": 0, "errors": 0.0}
        assert validate_measure_results(measure) is True

    def test_negative_values_accepted(self) -> None:
        """Test that negative values are accepted."""
        measure = {"delta": -5, "change": -0.15}
        assert validate_measure_results(measure) is True

    def test_nested_format_with_example_id_and_metrics(self) -> None:
        """Test nested format with example_id and metrics."""
        measure = {
            "example_id": "example_1",
            "metrics": {"accuracy": 0.95, "latency": 100},
        }
        assert validate_measure_results(measure) is True

    def test_nested_format_with_null_example_id(self) -> None:
        """Test nested format with null example_id."""
        measure = {
            "example_id": None,
            "metrics": {"accuracy": 0.95},
        }
        assert validate_measure_results(measure) is True

    def test_nested_format_with_empty_metrics(self) -> None:
        """Test nested format with empty metrics dict."""
        measure = {
            "example_id": "example_1",
            "metrics": {},
        }
        assert validate_measure_results(measure) is True

    def test_nested_format_with_mixed_metric_types(self) -> None:
        """Test nested format with mixed metric value types."""
        measure = {
            "example_id": "example_1",
            "metrics": {
                "accuracy": 0.95,
                "model": "gpt-4",
                "tokens": 150,
                "error": None,
            },
        }
        assert validate_measure_results(measure) is True

    def test_nested_format_invalid_example_id_type_raises_error(self) -> None:
        """Test that non-string example_id raises ValueError."""
        measure = {
            "example_id": 123,
            "metrics": {"accuracy": 0.95},
        }
        with pytest.raises(ValueError, match="example_id must be a string"):
            validate_measure_results(measure)

    def test_nested_format_metrics_not_dict_raises_error(self) -> None:
        """Test that non-dict metrics raises ValueError."""
        measure = {
            "example_id": "example_1",
            "metrics": [0.95, 100],
        }
        with pytest.raises(
            ValueError, match="metrics must be a dict, got <class 'list'>"
        ):
            validate_measure_results(measure)

    def test_nested_format_invalid_metric_key_raises_error(self) -> None:
        """Test that invalid metric key in nested format raises ValueError."""
        measure = {
            "example_id": "example_1",
            "metrics": {"invalid-key": 0.95},
        }
        with pytest.raises(ValueError, match="Invalid metric key 'invalid-key'"):
            validate_measure_results(measure)

    def test_nested_format_invalid_metric_value_type_raises_error(self) -> None:
        """Test that invalid metric value type in nested format raises ValueError."""
        measure = {
            "example_id": "example_1",
            "metrics": {"metrics": [1, 2, 3]},
        }
        with pytest.raises(
            ValueError,
            match="Metric value for 'metrics' must be number, string, or null",
        ):
            validate_measure_results(measure)


class TestValidateMeasuresArray:
    """Tests for validate_measures_array function."""

    def test_valid_measures_array(self) -> None:
        """Test validation of valid measures array."""
        measures = [
            {"accuracy": 0.95, "latency": 100},
            {"accuracy": 0.90, "latency": 120},
        ]
        assert validate_measures_array(measures) is True


class TestValidateComparabilityMetadata:
    """Tests for validate_comparability_metadata function."""

    def test_valid_comparability_metadata(self) -> None:
        payload = {
            "total_examples": 3,
            "examples_with_primary_metric": 3,
            "coverage_ratio": 1.0,
            "ranking_eligible": True,
            "warning_codes": ["MCI-005"],
            "missing_example_ids": [],
            "per_metric_coverage": {
                "accuracy": {"present": 3, "total": 3, "ratio": 1.0}
            },
        }
        assert validate_comparability_metadata(payload) is True

    @pytest.mark.parametrize(
        "payload,match",
        [
            ("not_a_dict", "comparability metadata must be an object"),
            ({"total_examples": "3"}, "comparability.total_examples must be an integer"),
            (
                {"examples_with_primary_metric": "2"},
                "comparability.examples_with_primary_metric must be an integer",
            ),
            ({"coverage_ratio": "0.5"}, "comparability.coverage_ratio must be numeric"),
            ({"coverage_ratio": 1.2}, "comparability.coverage_ratio must be between 0 and 1"),
            ({"ranking_eligible": "yes"}, "comparability.ranking_eligible must be boolean"),
            ({"warning_codes": "MCI-001"}, "comparability.warning_codes must be an array"),
            (
                {"warning_codes": [1, "MCI-002"]},
                "comparability.warning_codes must contain only strings",
            ),
            (
                {"missing_example_ids": [1, "ex_2"]},
                "comparability.missing_example_ids must contain only strings",
            ),
            (
                {"per_metric_coverage": []},
                "comparability.per_metric_coverage must be an object",
            ),
            (
                {"per_metric_coverage": {1: {"present": 1}}},
                "comparability.per_metric_coverage keys must be strings",
            ),
            (
                {"per_metric_coverage": {"accuracy": []}},
                "comparability.per_metric_coverage entries must be objects",
            ),
            (
                {"per_metric_coverage": {"accuracy": {"present": "1"}}},
                "comparability.per_metric_coverage.present must be integer",
            ),
            (
                {"per_metric_coverage": {"accuracy": {"total": "3"}}},
                "comparability.per_metric_coverage.total must be integer",
            ),
            (
                {"per_metric_coverage": {"accuracy": {"ratio": "0.9"}}},
                "comparability.per_metric_coverage.ratio must be numeric",
            ),
        ],
    )
    def test_invalid_comparability_metadata_raises(self, payload: object, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            validate_comparability_metadata(payload)

    def test_empty_measures_array(self) -> None:
        """Test validation of empty measures array."""
        measures = []
        assert validate_measures_array(measures) is True

    def test_null_measures_array(self) -> None:
        """Test validation of null measures array."""
        assert validate_measures_array(None) is True

    def test_single_measure_in_array(self) -> None:
        """Test validation of array with single measure."""
        measures = [{"accuracy": 0.95}]
        assert validate_measures_array(measures) is True

    def test_measures_array_with_empty_dicts(self) -> None:
        """Test validation of array containing empty dicts."""
        measures = [{}, {}, {}]
        assert validate_measures_array(measures) is True

    def test_measures_array_not_list_raises_error(self) -> None:
        """Test that non-list measures array raises ValueError."""
        with pytest.raises(ValueError, match="measures must be an array or null"):
            validate_measures_array({"not": "a list"})

    def test_measures_array_string_raises_error(self) -> None:
        """Test that string measures array raises ValueError."""
        with pytest.raises(
            ValueError, match="measures must be an array or null, got <class 'str'>"
        ):
            validate_measures_array("not_an_array")

    def test_measures_array_with_invalid_measure(self) -> None:
        """Test that array with invalid measure raises ValueError."""
        measures = [
            {"accuracy": 0.95},
            {"invalid-key": 100},  # Invalid key
        ]
        with pytest.raises(ValueError, match="Invalid measure at index 1"):
            validate_measures_array(measures)

    def test_measures_array_with_invalid_value_type(self) -> None:
        """Test that array with invalid value type raises ValueError."""
        measures = [
            {"accuracy": 0.95},
            {"metrics": [1, 2, 3]},  # Invalid value type
        ]
        with pytest.raises(ValueError, match="Invalid measure at index 1"):
            validate_measures_array(measures)

    def test_measures_array_first_element_invalid(self) -> None:
        """Test that invalid first element is caught."""
        measures = [
            {"123invalid": 0.95},  # Invalid key
            {"accuracy": 0.90},
        ]
        with pytest.raises(ValueError, match="Invalid measure at index 0"):
            validate_measures_array(measures)

    def test_measures_array_last_element_invalid(self) -> None:
        """Test that invalid last element is caught."""
        measures = [
            {"accuracy": 0.95},
            {"latency": 100},
            None,  # Invalid measure (not a dict)
        ]
        with pytest.raises(ValueError, match="Invalid measure at index 2"):
            validate_measures_array(measures)

    def test_large_measures_array(self) -> None:
        """Test validation of large measures array."""
        measures = [{"metric": i, "value": i * 0.1} for i in range(100)]
        assert validate_measures_array(measures) is True


class TestValidateSummaryStats:
    """Tests for validate_summary_stats function."""

    def test_null_summary_stats(self) -> None:
        """Test that null summary_stats is valid."""
        assert validate_summary_stats(None) is True

    def test_empty_summary_stats(self) -> None:
        """Test that empty summary_stats dict is valid."""
        assert validate_summary_stats({}) is True

    def test_valid_metrics_simple_values(self) -> None:
        """Test summary_stats with simple metric values."""
        summary_stats = {
            "metrics": {
                "accuracy": 0.95,
                "latency": 100,
                "status": "success",
            }
        }
        assert validate_summary_stats(summary_stats) is True

    def test_valid_metrics_with_null(self) -> None:
        """Test summary_stats with null metric values."""
        summary_stats = {
            "metrics": {
                "accuracy": 0.95,
                "error": None,
            }
        }
        assert validate_summary_stats(summary_stats) is True

    def test_valid_metrics_statistics_dict(self) -> None:
        """Test summary_stats with pandas.describe-style statistics."""
        summary_stats = {
            "metrics": {
                "accuracy": {
                    "count": 10,
                    "mean": 0.95,
                    "std": 0.02,
                    "min": 0.90,
                    "25%": 0.93,
                    "50%": 0.95,
                    "75%": 0.97,
                    "max": 0.99,
                }
            }
        }
        assert validate_summary_stats(summary_stats) is True

    def test_valid_metrics_partial_statistics(self) -> None:
        """Test summary_stats with partial statistics dict."""
        summary_stats = {
            "metrics": {
                "accuracy": {
                    "mean": 0.95,
                    "std": 0.02,
                }
            }
        }
        assert validate_summary_stats(summary_stats) is True

    def test_valid_metrics_mixed_formats(self) -> None:
        """Test summary_stats with mixed simple and statistics values."""
        summary_stats = {
            "metrics": {
                "accuracy": {"mean": 0.95, "std": 0.02},
                "latency": 100,
                "status": "success",
            }
        }
        assert validate_summary_stats(summary_stats) is True

    def test_valid_execution_time_float(self) -> None:
        """Test summary_stats with float execution_time."""
        summary_stats = {"execution_time": 123.456}
        assert validate_summary_stats(summary_stats) is True

    def test_valid_execution_time_int(self) -> None:
        """Test summary_stats with integer execution_time."""
        summary_stats = {"execution_time": 100}
        assert validate_summary_stats(summary_stats) is True

    def test_valid_execution_time_null(self) -> None:
        """Test summary_stats with null execution_time."""
        summary_stats = {"execution_time": None}
        assert validate_summary_stats(summary_stats) is True

    def test_valid_total_examples_int(self) -> None:
        """Test summary_stats with integer total_examples."""
        summary_stats = {"total_examples": 50}
        assert validate_summary_stats(summary_stats) is True

    def test_valid_total_examples_null(self) -> None:
        """Test summary_stats with null total_examples."""
        summary_stats = {"total_examples": None}
        assert validate_summary_stats(summary_stats) is True

    def test_valid_metadata_object(self) -> None:
        """Test summary_stats with metadata object."""
        summary_stats = {
            "metadata": {
                "version": "1.0",
                "timestamp": "2025-12-12T10:00:00Z",
            }
        }
        assert validate_summary_stats(summary_stats) is True

    def test_valid_all_fields_combined(self) -> None:
        """Test summary_stats with all valid fields combined."""
        summary_stats = {
            "metrics": {"accuracy": 0.95, "latency": 100},
            "execution_time": 123.456,
            "total_examples": 50,
            "metadata": {"version": "1.0"},
        }
        assert validate_summary_stats(summary_stats) is True

    def test_summary_stats_not_dict_raises_error(self) -> None:
        """Test that non-dict summary_stats raises ValueError."""
        with pytest.raises(ValueError, match="summary_stats must be an object or null"):
            validate_summary_stats([1, 2, 3])

    def test_summary_stats_string_raises_error(self) -> None:
        """Test that string summary_stats raises ValueError."""
        with pytest.raises(
            ValueError,
            match="summary_stats must be an object or null, got <class 'str'>",
        ):
            validate_summary_stats("not_a_dict")

    def test_unexpected_property_raises_error(self) -> None:
        """Test that unexpected property raises ValueError."""
        summary_stats = {"invalid_field": "value"}
        with pytest.raises(
            ValueError,
            match="summary_stats has unexpected properties: {'invalid_field'}",
        ):
            validate_summary_stats(summary_stats)

    def test_multiple_unexpected_properties_raises_error(self) -> None:
        """Test that multiple unexpected properties raises ValueError."""
        summary_stats = {
            "metrics": {"accuracy": 0.95},
            "unknown_field": "value",
            "another_invalid": 123,
        }
        with pytest.raises(ValueError, match="summary_stats has unexpected properties"):
            validate_summary_stats(summary_stats)

    def test_metrics_not_dict_raises_error(self) -> None:
        """Test that non-dict metrics raises ValueError."""
        summary_stats = {"metrics": [1, 2, 3]}
        with pytest.raises(
            ValueError,
            match="summary_stats.metrics must be an object, got <class 'list'>",
        ):
            validate_summary_stats(summary_stats)

    def test_metrics_invalid_value_type_raises_error(self) -> None:
        """Test that invalid metric value type raises ValueError."""
        summary_stats = {
            "metrics": {
                "accuracy": [0.95, 0.90],  # Invalid type
            }
        }
        with pytest.raises(
            ValueError,
            match="summary_stats.metrics\\['accuracy'\\] must be number, string, null, or statistics dict",
        ):
            validate_summary_stats(summary_stats)

    def test_metrics_statistics_invalid_key_raises_error(self) -> None:
        """Test that invalid statistics key raises ValueError."""
        summary_stats = {
            "metrics": {
                "accuracy": {
                    "mean": 0.95,
                    "invalid_stat": 0.02,  # Invalid key
                }
            }
        }
        with pytest.raises(
            ValueError,
            match="summary_stats.metrics\\['accuracy'\\] has invalid statistics keys: {'invalid_stat'}",
        ):
            validate_summary_stats(summary_stats)

    def test_metrics_statistics_invalid_value_type_raises_error(self) -> None:
        """Test that invalid statistic value type raises ValueError."""
        summary_stats = {
            "metrics": {
                "accuracy": {
                    "mean": "not_a_number",  # Invalid type
                }
            }
        }
        with pytest.raises(
            ValueError,
            match="summary_stats.metrics\\['accuracy'\\]\\['mean'\\] must be a number",
        ):
            validate_summary_stats(summary_stats)

    def test_metrics_statistics_null_value_accepted(self) -> None:
        """Test that null statistic values are accepted."""
        summary_stats = {
            "metrics": {
                "accuracy": {
                    "mean": 0.95,
                    "std": None,
                }
            }
        }
        assert validate_summary_stats(summary_stats) is True

    def test_execution_time_string_raises_error(self) -> None:
        """Test that string execution_time raises ValueError."""
        summary_stats = {"execution_time": "123"}
        with pytest.raises(
            ValueError, match="summary_stats.execution_time must be number or null"
        ):
            validate_summary_stats(summary_stats)

    def test_execution_time_list_raises_error(self) -> None:
        """Test that list execution_time raises ValueError."""
        summary_stats = {"execution_time": [123]}
        with pytest.raises(
            ValueError,
            match="summary_stats.execution_time must be number or null, got <class 'list'>",
        ):
            validate_summary_stats(summary_stats)

    def test_total_examples_float_raises_error(self) -> None:
        """Test that float total_examples raises ValueError."""
        summary_stats = {"total_examples": 50.5}
        with pytest.raises(
            ValueError, match="summary_stats.total_examples must be integer or null"
        ):
            validate_summary_stats(summary_stats)

    def test_total_examples_string_raises_error(self) -> None:
        """Test that string total_examples raises ValueError."""
        summary_stats = {"total_examples": "50"}
        with pytest.raises(
            ValueError, match="summary_stats.total_examples must be integer or null"
        ):
            validate_summary_stats(summary_stats)

    def test_metadata_not_dict_raises_error(self) -> None:
        """Test that non-dict metadata raises ValueError."""
        summary_stats = {"metadata": "not_a_dict"}
        with pytest.raises(
            ValueError,
            match="summary_stats.metadata must be an object, got <class 'str'>",
        ):
            validate_summary_stats(summary_stats)

    def test_metadata_list_raises_error(self) -> None:
        """Test that list metadata raises ValueError."""
        summary_stats = {"metadata": [1, 2, 3]}
        with pytest.raises(
            ValueError,
            match="summary_stats.metadata must be an object, got <class 'list'>",
        ):
            validate_summary_stats(summary_stats)

    def test_empty_metrics_dict(self) -> None:
        """Test summary_stats with empty metrics dict."""
        summary_stats = {"metrics": {}}
        assert validate_summary_stats(summary_stats) is True

    def test_zero_execution_time(self) -> None:
        """Test summary_stats with zero execution_time."""
        summary_stats = {"execution_time": 0}
        assert validate_summary_stats(summary_stats) is True

    def test_zero_total_examples(self) -> None:
        """Test summary_stats with zero total_examples."""
        summary_stats = {"total_examples": 0}
        assert validate_summary_stats(summary_stats) is True

    def test_negative_execution_time(self) -> None:
        """Test summary_stats with negative execution_time."""
        summary_stats = {"execution_time": -123.456}
        assert validate_summary_stats(summary_stats) is True


class TestValidateConfigurationRunSubmission:
    """Tests for validate_configuration_run_submission function."""

    def test_empty_submission(self) -> None:
        """Test validation of empty submission."""
        assert validate_configuration_run_submission({}) is True

    def test_submission_with_valid_measures(self) -> None:
        """Test submission with valid measures array."""
        data = {
            "measures": [
                {"accuracy": 0.95, "latency": 100},
                {"accuracy": 0.90, "latency": 120},
            ]
        }
        assert validate_configuration_run_submission(data) is True

    def test_submission_with_null_measures(self) -> None:
        """Test submission with null measures."""
        data = {"measures": None}
        assert validate_configuration_run_submission(data) is True

    def test_submission_with_valid_summary_stats(self) -> None:
        """Test submission with valid summary_stats."""
        data = {
            "summary_stats": {
                "metrics": {"accuracy": 0.95},
                "execution_time": 123.456,
                "total_examples": 50,
            }
        }
        assert validate_configuration_run_submission(data) is True

    def test_submission_with_null_summary_stats(self) -> None:
        """Test submission with null summary_stats."""
        data = {"summary_stats": None}
        assert validate_configuration_run_submission(data) is True

    def test_submission_with_both_measures_and_summary_stats(self) -> None:
        """Test submission with both measures and summary_stats."""
        data = {
            "measures": [{"accuracy": 0.95}],
            "summary_stats": {
                "metrics": {"mean_accuracy": 0.95},
                "total_examples": 50,
            },
        }
        assert validate_configuration_run_submission(data) is True

    def test_submission_with_other_fields(self) -> None:
        """Test submission with other allowed fields."""
        data = {
            "measures": [{"accuracy": 0.95}],
            "summary_stats": {"metrics": {"accuracy": 0.95}},
            "configuration": {"model": "gpt-4"},
            "metadata": {"version": "1.0"},
        }
        assert validate_configuration_run_submission(data) is True

    def test_submission_with_invalid_measures_raises_error(self) -> None:
        """Test that invalid measures raises ValueError."""
        data = {
            "measures": [
                {"accuracy": 0.95},
                {"invalid-key": 100},  # Invalid key
            ]
        }
        with pytest.raises(ValueError, match="Invalid measure at index 1"):
            validate_configuration_run_submission(data)

    def test_submission_with_invalid_summary_stats_raises_error(self) -> None:
        """Test that invalid summary_stats raises ValueError."""
        data = {
            "summary_stats": {
                "invalid_field": "value",  # Unexpected property
            }
        }
        with pytest.raises(ValueError, match="summary_stats has unexpected properties"):
            validate_configuration_run_submission(data)

    def test_metrics_with_measures_field_raises_error(self) -> None:
        """Test that metrics containing measures field raises ValueError."""
        data = {
            "metrics": {
                "measures": [{"accuracy": 0.95}],  # Should be separate field
            }
        }
        with pytest.raises(
            ValueError, match="metrics dict should not contain 'measures'"
        ):
            validate_configuration_run_submission(data)

    def test_metrics_with_summary_stats_field_raises_error(self) -> None:
        """Test that metrics containing summary_stats field raises ValueError."""
        data = {
            "metrics": {
                "summary_stats": {"mean": 0.95},  # Should be separate field
            }
        }
        with pytest.raises(
            ValueError, match="metrics dict should not contain 'summary_stats'"
        ):
            validate_configuration_run_submission(data)

    def test_metrics_with_both_nested_fields_raises_error(self) -> None:
        """Test that metrics containing both nested fields raises ValueError."""
        data = {
            "metrics": {
                "measures": [{"accuracy": 0.95}],
                "summary_stats": {"mean": 0.95},
            }
        }
        with pytest.raises(
            ValueError, match="metrics dict should not contain 'measures'"
        ):
            validate_configuration_run_submission(data)

    def test_metrics_with_valid_content(self) -> None:
        """Test that metrics with valid content passes."""
        data = {
            "metrics": {
                "accuracy": 0.95,
                "latency": 100,
            }
        }
        assert validate_configuration_run_submission(data) is True

    def test_metrics_not_dict_is_ignored(self) -> None:
        """Test that non-dict metrics is ignored in validation."""
        data = {
            "metrics": [
                1,
                2,
                3,
            ],  # Not a dict, but validation only checks if it IS a dict
        }
        assert validate_configuration_run_submission(data) is True

    def test_metrics_null_is_ignored(self) -> None:
        """Test that null metrics is ignored in validation."""
        data = {"metrics": None}
        assert validate_configuration_run_submission(data) is True

    def test_comprehensive_valid_submission(self) -> None:
        """Test comprehensive valid submission with all fields."""
        data = {
            "measures": [
                {"accuracy": 0.95, "latency": 100, "cost": 0.05},
                {"accuracy": 0.90, "latency": 120, "cost": 0.06},
            ],
            "summary_stats": {
                "metrics": {
                    "accuracy": {"mean": 0.925, "std": 0.025},
                    "latency": 110,
                },
                "execution_time": 123.456,
                "total_examples": 50,
                "metadata": {"version": "1.0"},
            },
            "metrics": {
                "overall_accuracy": 0.95,
            },
        }
        assert validate_configuration_run_submission(data) is True

    def test_submission_with_metadata_comparability_invalid_raises_error(self) -> None:
        """Top-level metadata.comparability is validated when provided."""
        data = {
            "metadata": {
                "comparability": {
                    "coverage_ratio": "0.5",
                }
            }
        }
        with pytest.raises(ValueError, match="comparability.coverage_ratio must be numeric"):
            validate_configuration_run_submission(data)

    def test_submission_with_summary_stats_comparability_invalid_raises_error(self) -> None:
        """summary_stats.metadata.comparability is validated when provided."""
        data = {
            "summary_stats": {
                "metrics": {"accuracy": 0.95},
                "metadata": {
                    "comparability": {
                        "ranking_eligible": "true",
                    }
                },
            }
        }
        with pytest.raises(
            ValueError, match="comparability.ranking_eligible must be boolean"
        ):
            validate_configuration_run_submission(data)
