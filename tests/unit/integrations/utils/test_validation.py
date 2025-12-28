"""Unit tests for traigent.integrations.utils.validation.

Tests for parameter validation utilities used in framework integrations.
"""

# Traceability: CONC-Layer-Integration CONC-Quality-Compatibility
# FUNC-INTEGRATIONS REQ-INT-008 SYNC-IntegrationHook

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Literal, get_type_hints
from unittest.mock import MagicMock, patch

from traigent.integrations.utils.validation import ParameterValidator


class TestParameterValidator:
    """Tests for ParameterValidator class."""

    # ========== validate_parameter_types tests ==========

    def test_validate_parameter_types_with_valid_types(self) -> None:
        """Test validation passes with correct parameter types."""

        # Use get_type_hints to resolve string annotations
        def sample_func(name: str, age: int, score: float) -> None:
            pass

        # Get type hints to resolve annotations
        hints = get_type_hints(sample_func)
        # Create signature with resolved types
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    "name",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=hints["name"],
                ),
                inspect.Parameter(
                    "age",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=hints["age"],
                ),
                inspect.Parameter(
                    "score",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=hints["score"],
                ),
            ]
        )

        params = {"name": "Alice", "age": 30, "score": 95.5}
        validated, issues = ParameterValidator.validate_parameter_types(params, sig)

        assert validated == params
        assert len(issues) == 0

    def test_validate_parameter_types_with_type_mismatch(self) -> None:
        """Test validation detects type mismatches."""
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    "count", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int
                ),
            ]
        )
        params = {"count": "not_an_int"}

        validated, issues = ParameterValidator.validate_parameter_types(params, sig)

        assert "count" not in validated
        assert len(issues) == 1
        assert "count" in issues[0]
        assert "expected type" in issues[0]

    def test_validate_parameter_types_with_convertible_types(self) -> None:
        """Test validation converts compatible types."""
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    "count", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=int
                ),
                inspect.Parameter(
                    "ratio", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=float
                ),
                inspect.Parameter(
                    "flag", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=bool
                ),
            ]
        )
        params = {"count": "42", "ratio": "3.14", "flag": "yes"}

        validated, issues = ParameterValidator.validate_parameter_types(params, sig)

        assert validated["count"] == 42
        assert validated["ratio"] == 3.14
        assert validated["flag"] is True
        assert len(issues) == 3
        assert all("(converted)" in issue for issue in issues)

    def test_validate_parameter_types_with_no_annotation(self) -> None:
        """Test validation accepts parameters with no type annotation."""
        sig = inspect.Signature(
            [
                inspect.Parameter("data", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            ]
        )
        params = {"data": [1, 2, 3]}

        validated, issues = ParameterValidator.validate_parameter_types(params, sig)

        assert validated["data"] == [1, 2, 3]
        assert len(issues) == 0

    def test_validate_parameter_types_with_unknown_param(self) -> None:
        """Test validation skips unknown parameters."""
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    "name", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
                ),
            ]
        )
        params = {"name": "Alice", "unknown": "value"}

        validated, issues = ParameterValidator.validate_parameter_types(params, sig)

        assert "name" in validated
        assert "unknown" not in validated
        assert len(issues) == 0

    def test_validate_parameter_types_with_empty_params(self) -> None:
        """Test validation handles empty parameter dict."""
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    "name", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str
                ),
            ]
        )
        params = {}

        validated, issues = ParameterValidator.validate_parameter_types(params, sig)

        assert validated == {}
        assert len(issues) == 0

    # ========== _check_type_compatibility tests ==========

    def test_check_type_compatibility_with_any(self) -> None:
        """Test compatibility check with Any type."""
        assert ParameterValidator._check_type_compatibility("anything", Any) is True
        assert ParameterValidator._check_type_compatibility(42, Any) is True
        assert ParameterValidator._check_type_compatibility(None, Any) is True

    def test_check_type_compatibility_with_union(self) -> None:
        """Test compatibility check with Union types."""
        assert ParameterValidator._check_type_compatibility(42, int | str) is True
        assert ParameterValidator._check_type_compatibility("hello", int | str) is True
        assert ParameterValidator._check_type_compatibility(3.14, int | str) is False

    def test_check_type_compatibility_with_pipe_union(self) -> None:
        """Test compatibility check with pipe union syntax (int | str)."""
        assert ParameterValidator._check_type_compatibility(42, int | str) is True
        assert ParameterValidator._check_type_compatibility("hello", int | str) is True
        assert ParameterValidator._check_type_compatibility(3.14, int | str) is False

    def test_check_type_compatibility_with_list(self) -> None:
        """Test compatibility check with list types."""
        assert (
            ParameterValidator._check_type_compatibility([1, 2, 3], list[int]) is True
        )
        assert (
            ParameterValidator._check_type_compatibility([1, "a"], list[int]) is False
        )
        assert ParameterValidator._check_type_compatibility([], list[int]) is True
        assert ParameterValidator._check_type_compatibility([1, 2], list) is True

    def test_check_type_compatibility_with_set(self) -> None:
        """Test compatibility check with set types."""
        assert (
            ParameterValidator._check_type_compatibility({"a", "b"}, set[str]) is True
        )
        assert ParameterValidator._check_type_compatibility({"a", 1}, set[str]) is False
        assert ParameterValidator._check_type_compatibility(set(), set[str]) is True

    def test_check_type_compatibility_with_frozenset(self) -> None:
        """Test compatibility check with frozenset types."""
        assert (
            ParameterValidator._check_type_compatibility(
                frozenset({"a", "b"}), frozenset[str]
            )
            is True
        )
        assert (
            ParameterValidator._check_type_compatibility(
                frozenset({"a", 1}), frozenset[str]
            )
            is False
        )

    def test_check_type_compatibility_tuple_fixed_length(self) -> None:
        """Test compatibility check with fixed-length tuple types."""
        assert (
            ParameterValidator._check_type_compatibility((1, "a"), tuple[int, str])
            is True
        )
        assert (
            ParameterValidator._check_type_compatibility(("a", 1), tuple[int, str])
            is False
        )
        assert (
            ParameterValidator._check_type_compatibility((1,), tuple[int, str]) is False
        )

    def test_check_type_compatibility_tuple_ellipsis(self) -> None:
        """Test compatibility check with variable-length tuple types."""
        assert (
            ParameterValidator._check_type_compatibility((1, 2, 3), tuple[int, ...])
            is True
        )
        assert (
            ParameterValidator._check_type_compatibility((1, "a"), tuple[int, ...])
            is False
        )
        assert ParameterValidator._check_type_compatibility((), tuple[int, ...]) is True

    def test_check_type_compatibility_tuple_no_args(self) -> None:
        """Test compatibility check with untyped tuple."""
        assert ParameterValidator._check_type_compatibility((1, "a"), tuple) is True

    def test_check_type_compatibility_with_dict(self) -> None:
        """Test compatibility check with dict types."""
        assert (
            ParameterValidator._check_type_compatibility(
                {"a": 1, "b": 2}, dict[str, int]
            )
            is True
        )
        assert (
            ParameterValidator._check_type_compatibility({"a": "bad"}, dict[str, int])
            is False
        )
        assert (
            ParameterValidator._check_type_compatibility({1: "a"}, dict[str, int])
            is False
        )
        assert ParameterValidator._check_type_compatibility({}, dict[str, int]) is True

    def test_check_type_compatibility_with_mapping(self) -> None:
        """Test compatibility check with Mapping types."""
        assert (
            ParameterValidator._check_type_compatibility(
                {"a": 1, "b": 2}, Mapping[str, int]
            )
            is True
        )
        assert (
            ParameterValidator._check_type_compatibility(
                {"a": "bad"}, Mapping[str, int]
            )
            is False
        )

    def test_check_type_compatibility_mapping_no_args(self) -> None:
        """Test compatibility check with untyped Mapping."""
        assert ParameterValidator._check_type_compatibility({"a": 1}, Mapping) is True

    def test_check_type_compatibility_with_sequence(self) -> None:
        """Test compatibility check with Sequence types."""
        assert (
            ParameterValidator._check_type_compatibility([1, 2, 3], Sequence[int])
            is True
        )
        assert (
            ParameterValidator._check_type_compatibility("abc", Sequence[str]) is True
        )
        assert (
            ParameterValidator._check_type_compatibility([1, "a"], Sequence[int])
            is False
        )

    def test_check_type_compatibility_sequence_no_args(self) -> None:
        """Test compatibility check with untyped Sequence."""
        assert ParameterValidator._check_type_compatibility([1, 2], Sequence) is True

    def test_check_type_compatibility_with_literal(self) -> None:
        """Test compatibility check with Literal types."""
        assert (
            ParameterValidator._check_type_compatibility("abc", Literal["abc"]) is True
        )
        assert (
            ParameterValidator._check_type_compatibility("xyz", Literal["abc"]) is False
        )
        assert ParameterValidator._check_type_compatibility(1, Literal[1, 2, 3]) is True

    def test_check_type_compatibility_with_basic_types(self) -> None:
        """Test compatibility check with basic Python types."""
        assert ParameterValidator._check_type_compatibility("hello", str) is True
        assert ParameterValidator._check_type_compatibility(42, int) is True
        assert ParameterValidator._check_type_compatibility(3.14, float) is True
        assert ParameterValidator._check_type_compatibility(True, bool) is True
        assert ParameterValidator._check_type_compatibility(42, str) is False

    def test_check_type_compatibility_with_type_error(self) -> None:
        """Test compatibility check handles TypeError gracefully."""
        # Create a mock type that raises TypeError on isinstance
        mock_type = MagicMock()
        mock_type.__origin__ = None

        # This should not raise, just return False
        result = ParameterValidator._check_type_compatibility("value", mock_type)
        assert result is False

    def test_check_type_compatibility_tuple_as_expected(self) -> None:
        """Test compatibility check with tuple as expected type."""
        assert ParameterValidator._check_type_compatibility("hello", (str, int)) is True
        assert ParameterValidator._check_type_compatibility(42, (str, int)) is True
        assert ParameterValidator._check_type_compatibility(3.14, (str, int)) is False

    def test_check_type_compatibility_non_instance_origin(self) -> None:
        """Test compatibility check with origin type."""
        # Test with a generic type that has an origin
        assert (
            ParameterValidator._check_type_compatibility([1, 2, 3], list[int]) is True
        )

    # ========== _check_union_compatibility tests ==========

    def test_check_union_compatibility_first_arg(self) -> None:
        """Test union compatibility when value matches first type."""
        result = ParameterValidator._check_union_compatibility(42, (int, str))
        assert result is True

    def test_check_union_compatibility_second_arg(self) -> None:
        """Test union compatibility when value matches second type."""
        result = ParameterValidator._check_union_compatibility("hello", (int, str))
        assert result is True

    def test_check_union_compatibility_no_match(self) -> None:
        """Test union compatibility when value matches no types."""
        result = ParameterValidator._check_union_compatibility(3.14, (int, str))
        assert result is False

    # ========== _check_collection_compatibility tests ==========

    def test_check_collection_compatibility_wrong_origin(self) -> None:
        """Test collection compatibility with wrong origin type."""
        result = ParameterValidator._check_collection_compatibility(
            "not_a_list", list, (int,)
        )
        assert result is False

    def test_check_collection_compatibility_no_args(self) -> None:
        """Test collection compatibility with no type arguments."""
        result = ParameterValidator._check_collection_compatibility([1, 2], list, ())
        assert result is True

    def test_check_collection_compatibility_matching(self) -> None:
        """Test collection compatibility with matching item types."""
        result = ParameterValidator._check_collection_compatibility(
            [1, 2, 3], list, (int,)
        )
        assert result is True

    def test_check_collection_compatibility_mismatched(self) -> None:
        """Test collection compatibility with mismatched item types."""
        result = ParameterValidator._check_collection_compatibility(
            [1, "a"], list, (int,)
        )
        assert result is False

    # ========== _check_tuple_compatibility tests ==========

    def test_check_tuple_compatibility_not_tuple(self) -> None:
        """Test tuple compatibility with non-tuple value."""
        result = ParameterValidator._check_tuple_compatibility([1, 2], (int, int))
        assert result is False

    def test_check_tuple_compatibility_no_args(self) -> None:
        """Test tuple compatibility with no type arguments."""
        result = ParameterValidator._check_tuple_compatibility((1, "a"), ())
        assert result is True

    def test_check_tuple_compatibility_ellipsis_matching(self) -> None:
        """Test tuple compatibility with ellipsis and matching items."""
        result = ParameterValidator._check_tuple_compatibility(
            (1, 2, 3), (int, Ellipsis)
        )
        assert result is True

    def test_check_tuple_compatibility_ellipsis_mismatched(self) -> None:
        """Test tuple compatibility with ellipsis, mismatched items."""
        result = ParameterValidator._check_tuple_compatibility(
            (1, "a"), (int, Ellipsis)
        )
        assert result is False

    def test_check_tuple_compatibility_length_mismatch(self) -> None:
        """Test tuple compatibility with length mismatch."""
        result = ParameterValidator._check_tuple_compatibility((1,), (int, int))
        assert result is False

    def test_check_tuple_compatibility_matching_types(self) -> None:
        """Test tuple compatibility with matching types."""
        result = ParameterValidator._check_tuple_compatibility((1, "a"), (int, str))
        assert result is True

    def test_check_tuple_compatibility_mismatched_types(self) -> None:
        """Test tuple compatibility with mismatched types."""
        result = ParameterValidator._check_tuple_compatibility((1, 2), (int, str))
        assert result is False

    # ========== _check_mapping_compatibility tests ==========

    def test_check_mapping_compatibility_not_mapping(self) -> None:
        """Test mapping compatibility with non-mapping value."""
        result = ParameterValidator._check_mapping_compatibility([1, 2], (str, int))
        assert result is False

    def test_check_mapping_compatibility_no_key_val_types(self) -> None:
        """Test mapping compatibility without key/value type spec."""
        result = ParameterValidator._check_mapping_compatibility({"a": 1}, (str,))
        assert result is True

    def test_check_mapping_compatibility_matching_types(self) -> None:
        """Test mapping compatibility with matching key/value types."""
        result = ParameterValidator._check_mapping_compatibility(
            {"a": 1, "b": 2}, (str, int)
        )
        assert result is True

    def test_check_mapping_compatibility_mismatched_key(self) -> None:
        """Test mapping compatibility with mismatched key type."""
        result = ParameterValidator._check_mapping_compatibility({1: "a"}, (str, str))
        assert result is False

    def test_check_mapping_compatibility_mismatched_value(self) -> None:
        """Test mapping compatibility with mismatched value type."""
        result = ParameterValidator._check_mapping_compatibility({"a": 1}, (str, str))
        assert result is False

    # ========== _check_sequence_compatibility tests ==========

    def test_check_sequence_compatibility_not_sequence(self) -> None:
        """Test sequence compatibility with non-sequence value."""
        result = ParameterValidator._check_sequence_compatibility(42, (int,))
        assert result is False

    def test_check_sequence_compatibility_no_args(self) -> None:
        """Test sequence compatibility with no type arguments."""
        result = ParameterValidator._check_sequence_compatibility([1, 2], ())
        assert result is True

    def test_check_sequence_compatibility_matching_items(self) -> None:
        """Test sequence compatibility with matching item types."""
        result = ParameterValidator._check_sequence_compatibility([1, 2, 3], (int,))
        assert result is True

    def test_check_sequence_compatibility_mismatched_items(self) -> None:
        """Test sequence compatibility with mismatched item types."""
        result = ParameterValidator._check_sequence_compatibility([1, "a"], (int,))
        assert result is False

    # ========== _try_convert_type tests ==========

    def test_try_convert_type_to_str(self) -> None:
        """Test type conversion to string."""
        result = ParameterValidator._try_convert_type(42, str)
        assert result == "42"

    def test_try_convert_type_to_int(self) -> None:
        """Test type conversion to int."""
        result = ParameterValidator._try_convert_type("42", int)
        assert result == 42

    def test_try_convert_type_to_float(self) -> None:
        """Test type conversion to float."""
        result = ParameterValidator._try_convert_type("3.14", float)
        assert result == 3.14

    def test_try_convert_type_to_bool(self) -> None:
        """Test type conversion to bool."""
        result = ParameterValidator._try_convert_type("yes", bool)
        assert result is True
        result = ParameterValidator._try_convert_type("", bool)
        assert result is False

    def test_try_convert_type_invalid_conversion(self) -> None:
        """Test type conversion with invalid value."""
        result = ParameterValidator._try_convert_type("not_a_number", int)
        assert result is None

    def test_try_convert_type_unsupported_type(self) -> None:
        """Test type conversion to unsupported type."""
        result = ParameterValidator._try_convert_type("value", list)
        assert result is None

    def test_try_convert_type_type_error(self) -> None:
        """Test type conversion that raises TypeError."""
        result = ParameterValidator._try_convert_type(None, int)
        assert result is None

    # ========== validate_parameter_values tests ==========

    def test_validate_parameter_values_within_constraints(self) -> None:
        """Test parameter value validation with values in constraints."""
        params = {"temperature": 0.5, "top_p": 0.8}
        constraints = {
            "temperature": {"min": 0.0, "max": 1.0},
            "top_p": {"min": 0.0, "max": 1.0},
        }

        issues = ParameterValidator.validate_parameter_values(params, constraints)
        assert len(issues) == 0

    def test_validate_parameter_values_below_min(self) -> None:
        """Test parameter value validation detects value below minimum."""
        params = {"temperature": -0.5}
        constraints = {"temperature": {"min": 0.0, "max": 1.0}}

        issues = ParameterValidator.validate_parameter_values(params, constraints)
        assert len(issues) == 1
        assert "temperature" in issues[0]
        assert "below minimum" in issues[0]

    def test_validate_parameter_values_above_max(self) -> None:
        """Test parameter value validation detects value above maximum."""
        params = {"temperature": 1.5}
        constraints = {"temperature": {"min": 0.0, "max": 1.0}}

        issues = ParameterValidator.validate_parameter_values(params, constraints)
        assert len(issues) == 1
        assert "temperature" in issues[0]
        assert "above maximum" in issues[0]

    def test_validate_parameter_values_not_in_allowed(self) -> None:
        """Test parameter value validation detects value not in allowed."""
        params = {"model": "invalid-model"}
        constraints = {"model": {"allowed": ["gpt-3.5-turbo", "gpt-4"]}}

        issues = ParameterValidator.validate_parameter_values(params, constraints)
        assert len(issues) == 1
        assert "model" in issues[0]
        assert "not in allowed values" in issues[0]

    def test_validate_parameter_values_in_allowed(self) -> None:
        """Test parameter value validation accepts value in allowed list."""
        params = {"model": "gpt-4"}
        constraints = {"model": {"allowed": ["gpt-3.5-turbo", "gpt-4"]}}

        issues = ParameterValidator.validate_parameter_values(params, constraints)
        assert len(issues) == 0

    def test_validate_parameter_values_unknown_param(self) -> None:
        """Test parameter value validation skips unknown parameters."""
        params = {"unknown": "value"}
        constraints = {"temperature": {"min": 0.0, "max": 1.0}}

        issues = ParameterValidator.validate_parameter_values(params, constraints)
        assert len(issues) == 0

    def test_validate_parameter_values_empty_params(self) -> None:
        """Test parameter value validation handles empty parameters."""
        params = {}
        constraints = {"temperature": {"min": 0.0, "max": 1.0}}

        issues = ParameterValidator.validate_parameter_values(params, constraints)
        assert len(issues) == 0

    def test_validate_parameter_values_multiple_issues(self) -> None:
        """Test parameter value validation detects multiple issues."""
        params = {"temperature": 2.0, "top_p": -0.5, "model": "bad"}
        constraints = {
            "temperature": {"min": 0.0, "max": 1.0},
            "top_p": {"min": 0.0, "max": 1.0},
            "model": {"allowed": ["gpt-3.5-turbo"]},
        }

        issues = ParameterValidator.validate_parameter_values(params, constraints)
        assert len(issues) == 3

    # ========== sanitize_parameters tests ==========

    def test_sanitize_parameters_valid_params(self) -> None:
        """Test parameter sanitization with valid parameters."""

        class SampleClass:
            def __init__(self, name: str, age: int) -> None:
                pass

        params = {"name": "Alice", "age": 30}
        sanitized = ParameterValidator.sanitize_parameters(params, SampleClass)

        assert sanitized == params

    def test_sanitize_parameters_removes_unknown(self) -> None:
        """Test parameter sanitization removes unknown parameters."""

        class SampleClass:
            def __init__(self, name: str) -> None:
                pass

        params = {"name": "Alice", "unknown": "value"}
        sanitized = ParameterValidator.sanitize_parameters(params, SampleClass)

        assert "name" in sanitized
        assert "unknown" not in sanitized

    def test_sanitize_parameters_removes_none_with_default(self) -> None:
        """Test parameter sanitization removes None with default."""

        class SampleClass:
            def __init__(self, name: str = "default") -> None:
                pass

        params = {"name": None}
        sanitized = ParameterValidator.sanitize_parameters(params, SampleClass)

        assert "name" not in sanitized

    def test_sanitize_parameters_keeps_none_without_default(self) -> None:
        """Test parameter sanitization keeps None without default."""

        class SampleClass:
            def __init__(self, name: str) -> None:
                pass

        params = {"name": None}
        sanitized = ParameterValidator.sanitize_parameters(params, SampleClass)

        assert "name" in sanitized
        assert sanitized["name"] is None

    def test_sanitize_parameters_empty_params(self) -> None:
        """Test parameter sanitization handles empty parameters."""

        class SampleClass:
            def __init__(self, name: str) -> None:
                pass

        params = {}
        sanitized = ParameterValidator.sanitize_parameters(params, SampleClass)

        assert sanitized == {}

    def test_sanitize_parameters_no_signature(self) -> None:
        """Test parameter sanitization when signature cannot be obtained."""
        # Create a class that will raise an exception when getting signature
        mock_class = MagicMock()
        with patch("inspect.signature", side_effect=ValueError("No sig")):
            params = {"name": "Alice"}
            sanitized = ParameterValidator.sanitize_parameters(params, mock_class)

            # Should return params as-is when signature fails
            assert sanitized == params

    def test_sanitize_parameters_with_logger_debug(self) -> None:
        """Test sanitization logs debug message for unknown parameters."""

        class SampleClass:
            def __init__(self, name: str) -> None:
                pass

        params = {"name": "Alice", "unknown": "value"}

        with patch("traigent.integrations.utils.validation.logger.debug") as mock_log:
            sanitized = ParameterValidator.sanitize_parameters(params, SampleClass)

            assert "name" in sanitized
            assert "unknown" not in sanitized
            # Verify logger was called
            mock_log.assert_called_once()
            assert "unknown" in str(mock_log.call_args)

    # ========== get_common_constraints tests ==========

    def test_get_common_constraints_structure(self) -> None:
        """Test get_common_constraints returns expected structure."""
        constraints = ParameterValidator.get_common_constraints()

        assert isinstance(constraints, dict)
        assert "temperature" in constraints
        assert "top_p" in constraints
        assert "max_tokens" in constraints

    def test_get_common_constraints_temperature(self) -> None:
        """Test get_common_constraints has correct temperature."""
        constraints = ParameterValidator.get_common_constraints()

        temp = constraints["temperature"]
        assert temp["min"] == 0.0
        assert temp["max"] == 2.0
        assert temp["type"] is float

    def test_get_common_constraints_top_p(self) -> None:
        """Test get_common_constraints has correct top_p constraints."""
        constraints = ParameterValidator.get_common_constraints()

        top_p = constraints["top_p"]
        assert top_p["min"] == 0.0
        assert top_p["max"] == 1.0
        assert top_p["type"] is float

    def test_get_common_constraints_top_k(self) -> None:
        """Test get_common_constraints has correct top_k constraints."""
        constraints = ParameterValidator.get_common_constraints()

        top_k = constraints["top_k"]
        assert top_k["min"] == 1
        assert top_k["type"] is int
        assert "max" not in top_k

    def test_get_common_constraints_max_tokens(self) -> None:
        """Test get_common_constraints has correct max_tokens."""
        constraints = ParameterValidator.get_common_constraints()

        max_tokens = constraints["max_tokens"]
        assert max_tokens["min"] == 1
        assert max_tokens["type"] is int

    def test_get_common_constraints_frequency_penalty(self) -> None:
        """Test get_common_constraints has correct frequency_penalty."""
        constraints = ParameterValidator.get_common_constraints()

        freq = constraints["frequency_penalty"]
        assert freq["min"] == -2.0
        assert freq["max"] == 2.0
        assert freq["type"] is float

    def test_get_common_constraints_presence_penalty(self) -> None:
        """Test get_common_constraints has correct presence_penalty."""
        constraints = ParameterValidator.get_common_constraints()

        pres = constraints["presence_penalty"]
        assert pres["min"] == -2.0
        assert pres["max"] == 2.0
        assert pres["type"] is float

    def test_get_common_constraints_n(self) -> None:
        """Test get_common_constraints has correct n constraints."""
        constraints = ParameterValidator.get_common_constraints()

        n = constraints["n"]
        assert n["min"] == 1
        assert n["type"] is int

    # ========== Integration tests ==========

    def test_validate_parameter_types_complex_nested_types(self) -> None:
        """Test validation with complex nested type structures."""
        # Create signature with complex types
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    "data",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=dict[str, list[int]],
                ),
                inspect.Parameter(
                    "options",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=list[tuple[str, float]],
                ),
            ]
        )

        params = {
            "data": {"scores": [1, 2, 3], "counts": [4, 5]},
            "options": [("opt1", 0.5), ("opt2", 0.8)],
        }

        validated, issues = ParameterValidator.validate_parameter_types(params, sig)

        assert validated == params
        assert len(issues) == 0

    def test_validate_parameter_types_with_optional(self) -> None:
        """Test validation with Optional types."""
        # Create signature with optional type
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    "value",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str | None,
                ),
            ]
        )

        # Test with string value
        validated, issues = ParameterValidator.validate_parameter_types(
            {"value": "test"}, sig
        )
        assert validated["value"] == "test"
        assert len(issues) == 0

        # Test with None value
        validated, issues = ParameterValidator.validate_parameter_types(
            {"value": None}, sig
        )
        assert validated["value"] is None
        assert len(issues) == 0

    def test_full_workflow_with_llm_parameters(self) -> None:
        """Test complete validation workflow with typical LLM parameters."""
        # Create signature for typical LLM function
        sig = inspect.Signature(
            [
                inspect.Parameter(
                    "temperature",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=float,
                ),
                inspect.Parameter(
                    "max_tokens",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=int,
                ),
                inspect.Parameter(
                    "model",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=str,
                ),
            ]
        )
        params = {"temperature": "0.7", "max_tokens": 100, "model": "gpt-4"}

        # Type validation
        validated, type_issues = ParameterValidator.validate_parameter_types(
            params, sig
        )

        # Value validation
        constraints = ParameterValidator.get_common_constraints()
        value_issues = ParameterValidator.validate_parameter_values(
            validated, constraints
        )

        assert validated["temperature"] == 0.7  # Converted from string
        assert len(type_issues) == 1  # Temperature was converted
        assert "(converted)" in type_issues[0]
        assert len(value_issues) == 0  # All values within constraints

    def test_sanitize_parameters_with_complex_class(self) -> None:
        """Test sanitization with a class that has many parameters."""

        class ComplexClass:
            def __init__(
                self,
                required: str,
                optional: int = 10,
                with_default: float = 0.5,
                *args: Any,
                **kwargs: Any,
            ) -> None:
                pass

        params = {
            "required": "value",
            "optional": 20,
            "with_default": None,  # Should be removed
            "unknown": "ignored",  # Should be removed
        }

        sanitized = ParameterValidator.sanitize_parameters(params, ComplexClass)

        assert "required" in sanitized
        assert "optional" in sanitized
        assert "with_default" not in sanitized  # None with default removed
        assert "unknown" not in sanitized  # Unknown parameter removed
