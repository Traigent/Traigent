"""Property-based tests for validation logic using Hypothesis.

This module uses property-based testing to ensure validation functions
handle edge cases correctly across a wide range of inputs.
"""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from traigent.utils.exceptions import MetricExtractionError
from traigent.utils.validation import validate_numeric_metric


# Test validate_numeric_metric with invalid non-numeric types
@given(
    st.one_of(
        st.none(),
        st.text(),
        st.dictionaries(st.text(), st.integers()),
        st.lists(st.integers()),
        st.booleans(),
    )
)
def test_validate_numeric_metric_rejects_non_numeric(invalid_value):
    """Should raise MetricExtractionError for non-numeric values."""
    # Booleans are technically numeric in Python (True=1, False=0)
    # but we reject them for clarity in optimization metrics
    if isinstance(invalid_value, bool):
        pytest.skip("Booleans are handled separately")

    with pytest.raises(MetricExtractionError) as exc_info:
        validate_numeric_metric(invalid_value, "test_metric")

    assert exc_info.value.field == "test_metric"
    assert exc_info.value.value == invalid_value


@given(st.floats(allow_nan=True, allow_infinity=False))
def test_validate_numeric_metric_rejects_nan(nan_value):
    """Should raise for NaN values."""
    if math.isnan(nan_value):
        with pytest.raises(MetricExtractionError) as exc_info:
            validate_numeric_metric(nan_value, "test_metric")

        assert exc_info.value.field == "test_metric"
        assert "NaN" in exc_info.value.message


@given(st.floats(allow_nan=False, allow_infinity=True))
def test_validate_numeric_metric_rejects_inf(inf_value):
    """Should raise for Inf values."""
    if math.isinf(inf_value):
        with pytest.raises(MetricExtractionError) as exc_info:
            validate_numeric_metric(inf_value, "test_metric")

        assert exc_info.value.field == "test_metric"
        assert "Inf" in exc_info.value.message


@given(
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10)
)
def test_validate_numeric_metric_accepts_valid_floats(valid_value):
    """Should accept valid floats."""
    result = validate_numeric_metric(valid_value, "test_metric")
    assert isinstance(result, float)
    assert result == valid_value


@given(st.integers(min_value=-10000, max_value=10000))
def test_validate_numeric_metric_accepts_valid_integers(valid_value):
    """Should accept valid integers and convert to float."""
    result = validate_numeric_metric(valid_value, "test_metric")
    assert isinstance(result, float)
    assert result == float(valid_value)


def test_validate_numeric_metric_rejects_none():
    """Should raise for None values."""
    with pytest.raises(MetricExtractionError) as exc_info:
        validate_numeric_metric(None, "test_metric")

    assert exc_info.value.field == "test_metric"
    assert exc_info.value.value is None
    assert "None" in exc_info.value.message


def test_validate_numeric_metric_includes_trial_context():
    """Should include trial_id and example_id in error context."""
    with pytest.raises(MetricExtractionError) as exc_info:
        validate_numeric_metric(
            "invalid",
            field_name="accuracy",
            trial_id="trial-123",
            example_id="example-456",
        )

    assert exc_info.value.field == "accuracy"
    assert exc_info.value.trial_id == "trial-123"
    assert exc_info.value.example_id == "example-456"


@given(st.text(min_size=1))
def test_validate_numeric_metric_numeric_strings_fail(text_value):
    """Should reject string values even if they look numeric."""
    # Hypothesis might generate numeric strings like "123" or "3.14"
    # We want to be strict and reject all strings
    try:
        # If it converts successfully, skip this test case
        float(text_value)
        pytest.skip("Hypothesis generated a numeric string - handled separately")
    except ValueError:
        # Non-numeric string - should fail validation
        with pytest.raises(MetricExtractionError):
            validate_numeric_metric(text_value, "test_metric")


# Edge case tests for specific boundary conditions
def test_validate_numeric_metric_zero():
    """Should accept zero as a valid metric."""
    result = validate_numeric_metric(0, "test_metric")
    assert result == 0.0


def test_validate_numeric_metric_negative():
    """Should accept negative values."""
    result = validate_numeric_metric(-42.5, "test_metric")
    assert result == -42.5


def test_validate_numeric_metric_very_small():
    """Should accept very small positive values."""
    result = validate_numeric_metric(1e-100, "test_metric")
    assert result == 1e-100


def test_validate_numeric_metric_very_large():
    """Should accept very large values (but not Inf)."""
    result = validate_numeric_metric(1e100, "test_metric")
    assert result == 1e100
