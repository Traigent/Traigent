"""Characterize FrameworkConstraints.validate_parameter before extraction."""

from unittest.mock import patch

from traigent.integrations.config import FrameworkConstraints, ParameterConstraints


def test_unknown_parameter_returns_no_issues_without_constraints_lookup():
    with patch.object(
        FrameworkConstraints,
        "get_constraints_for_framework",
        return_value={"known": ParameterConstraints(type=int)},
    ) as get_constraints:
        assert FrameworkConstraints.validate_parameter("made-up", "unknown", 1) == []
    get_constraints.assert_called_once_with("made-up")


def test_unknown_framework_uses_common_constraints():
    assert FrameworkConstraints.validate_parameter(
        "made-up", "temperature", -0.5
    ) == ["Parameter 'temperature' value -0.5 is below minimum 0.0"]


def test_alias_uses_canonical_name_in_validation_issue():
    assert FrameworkConstraints.validate_parameter("openai", "max_length", 0) == [
        "Parameter 'max_tokens' value 0 is below minimum 1"
    ]


def test_wrong_type_returns_before_range_and_deprecation_warning():
    constraints = {
        "count": ParameterConstraints(
            type=int,
            min_value=1,
            deprecated=True,
            deprecated_message="count is deprecated",
        )
    }
    with (
        patch.object(
            FrameworkConstraints,
            "get_constraints_for_framework",
            return_value=constraints,
        ),
        patch("traigent.integrations.config.logger.warning") as warning,
    ):
        assert FrameworkConstraints.validate_parameter("custom", "count", "0") == [
            "Parameter 'count' expected type int, got str"
        ]
    warning.assert_not_called()


def test_range_and_allowed_value_issues_keep_order_and_warning_follows_checks():
    events = []

    class TracedValue:
        def __lt__(self, other):
            events.append("minimum")
            return True

        def __gt__(self, other):
            events.append("maximum")
            return False

        def __repr__(self):
            return "sample"

    constraints = {
        "mode": ParameterConstraints(
            type=TracedValue,
            min_value=1,
            max_value=9,
            allowed_values={"allowed"},
            deprecated=True,
            deprecated_message="mode is deprecated",
        )
    }
    with (
        patch.object(
            FrameworkConstraints,
            "get_constraints_for_framework",
            return_value=constraints,
        ),
        patch(
            "traigent.integrations.config.logger.warning",
            side_effect=lambda message: events.append(("warning", message)),
        ),
    ):
        issues = FrameworkConstraints.validate_parameter("custom", "mode", TracedValue())

    assert issues == [
        "Parameter 'mode' value sample is below minimum 1",
        "Parameter 'mode' value sample not in allowed values",
    ]
    assert events == [
        "minimum",
        "maximum",
        ("warning", "mode is deprecated"),
    ]


def test_none_range_bounds_are_skipped_and_valid_value_has_no_issues():
    constraints = {
        "value": ParameterConstraints(type=int, min_value=None, max_value=None)
    }
    with patch.object(
        FrameworkConstraints,
        "get_constraints_for_framework",
        return_value=constraints,
    ):
        assert FrameworkConstraints.validate_parameter("custom", "value", 0) == []
