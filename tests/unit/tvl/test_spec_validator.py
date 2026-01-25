"""Tests for TVL spec drift validation (G-3).

Tests cover:
- Parameter alignment detection
- Type mismatch detection
- Report generation and severity levels
- Edge cases (empty spaces, partial matches)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from traigent.tvl.spec_validator import (
    DriftSeverity,
    SpecDriftIssue,
    SpecDriftReport,
    extract_config_space_params,
    extract_decorator_params,
    validate_spec_code_alignment,
    validate_tvar_types_match,
)


@dataclass
class MockTVLSpecArtifact:
    """Mock TVL spec artifact for testing."""

    path: Path = Path("test.tvl.yaml")
    environment: str | None = None
    configuration_space: dict[str, Any] | None = None
    objective_schema: Any = None
    constraints: list[Any] | None = None
    default_config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    budget: Any = None
    algorithm: str | None = None


class TestExtractConfigSpaceParams:
    """Tests for extract_config_space_params helper."""

    def test_none_returns_empty_set(self) -> None:
        """None config space returns empty set."""
        result = extract_config_space_params(None)
        assert result == set()

    def test_empty_dict_returns_empty_set(self) -> None:
        """Empty dict returns empty set."""
        result = extract_config_space_params({})
        assert result == set()

    def test_extracts_keys_from_dict(self) -> None:
        """Extracts parameter names from config space dict."""
        config_space = {
            "temperature": (0.0, 1.0),
            "max_tokens": [100, 200, 500],
            "model": ["gpt-4", "claude"],
        }
        result = extract_config_space_params(config_space)
        assert result == {"temperature", "max_tokens", "model"}


class TestExtractDecoratorParams:
    """Tests for extract_decorator_params helper."""

    def test_plain_function_returns_empty(self) -> None:
        """Plain undecorated function returns empty set."""

        def plain_func(x: int) -> int:
            return x * 2

        result = extract_decorator_params(plain_func)
        assert result == set()

    def test_function_with_configuration_space_attr(self) -> None:
        """Function with configuration_space attribute extracts params."""

        def decorated_func(x: int) -> int:
            return x * 2

        decorated_func.configuration_space = {  # type: ignore[attr-defined]
            "temperature": (0.0, 1.0),
            "model": ["gpt-4"],
        }

        result = extract_decorator_params(decorated_func)
        assert result == {"temperature", "model"}

    def test_function_with_traigent_attr(self) -> None:
        """Function with __traigent__ attribute extracts params."""

        def decorated_func(x: int) -> int:
            return x * 2

        decorated_func.__traigent__ = {  # type: ignore[attr-defined]
            "configuration_space": {
                "k": [1, 3, 5],
                "strategy": ["bm25", "dense"],
            }
        }

        result = extract_decorator_params(decorated_func)
        assert result == {"k", "strategy"}


class TestValidateSpecCodeAlignment:
    """Tests for validate_spec_code_alignment function."""

    def test_no_drift_when_params_match(self) -> None:
        """No drift detected when spec and code params match."""
        spec = MockTVLSpecArtifact(
            configuration_space={
                "temperature": (0.0, 1.0),
                "model": ["gpt-4", "claude"],
            }
        )
        code_space = {
            "temperature": (0.0, 1.0),
            "model": ["gpt-4", "claude"],
        }

        report = validate_spec_code_alignment(spec, configuration_space=code_space)

        assert len(report.issues) == 0
        assert not report.has_errors
        assert not report.has_warnings

    def test_drift_when_spec_has_extra_params(self) -> None:
        """Warning when spec has params not in code."""
        spec = MockTVLSpecArtifact(
            configuration_space={
                "temperature": (0.0, 1.0),
                "model": ["gpt-4"],
                "extra_param": [1, 2, 3],  # Not in code
            }
        )
        code_space = {
            "temperature": (0.0, 1.0),
            "model": ["gpt-4"],
        }

        report = validate_spec_code_alignment(spec, configuration_space=code_space)

        assert len(report.issues) == 1
        assert report.has_warnings
        assert not report.has_errors
        issue = report.issues[0]
        assert issue.severity == DriftSeverity.WARNING
        assert "extra_param" in str(issue.missing_in_code)

    def test_drift_when_code_has_extra_params(self) -> None:
        """Info when code has params not in spec."""
        spec = MockTVLSpecArtifact(
            configuration_space={
                "temperature": (0.0, 1.0),
            }
        )
        code_space = {
            "temperature": (0.0, 1.0),
            "new_param": ["a", "b"],  # Not in spec
        }

        report = validate_spec_code_alignment(spec, configuration_space=code_space)

        assert len(report.issues) == 1
        # Extra params in code are INFO by default (not blocking)
        issue = report.issues[0]
        assert issue.severity == DriftSeverity.INFO
        assert "new_param" in str(issue.missing_in_spec)

    def test_strict_mode_treats_all_drift_as_error(self) -> None:
        """Strict mode treats any drift as error."""
        spec = MockTVLSpecArtifact(
            configuration_space={
                "temperature": (0.0, 1.0),
                "spec_only": [1, 2],
            }
        )
        code_space = {
            "temperature": (0.0, 1.0),
            "code_only": ["a", "b"],
        }

        report = validate_spec_code_alignment(
            spec, configuration_space=code_space, strict=True
        )

        assert report.has_errors
        assert len(report.issues) == 2
        for issue in report.issues:
            assert issue.severity == DriftSeverity.ERROR

    def test_no_code_params_provided(self) -> None:
        """Info message when no code params provided."""
        spec = MockTVLSpecArtifact(configuration_space={"temperature": (0.0, 1.0)})

        report = validate_spec_code_alignment(spec)

        assert len(report.issues) == 1
        assert report.issues[0].severity == DriftSeverity.INFO
        assert "No configuration_space" in report.issues[0].message

    def test_empty_spec_with_code_params(self) -> None:
        """Empty spec with code params reports drift."""
        spec = MockTVLSpecArtifact(configuration_space={})
        code_space = {"temperature": (0.0, 1.0)}

        report = validate_spec_code_alignment(spec, configuration_space=code_space)

        assert len(report.issues) == 1
        assert "temperature" in str(report.issues[0].missing_in_spec)

    def test_raise_if_errors_raises_on_strict(self) -> None:
        """raise_if_errors raises ValueError when errors exist."""
        spec = MockTVLSpecArtifact(configuration_space={"spec_param": (0.0, 1.0)})
        code_space = {"code_param": (0.0, 1.0)}

        report = validate_spec_code_alignment(
            spec, configuration_space=code_space, strict=True
        )

        with pytest.raises(ValueError, match="Spec drift errors"):
            report.raise_if_errors()

    def test_summary_format(self) -> None:
        """Report summary is human-readable."""
        spec = MockTVLSpecArtifact(configuration_space={"spec_param": (0.0, 1.0)})
        code_space = {"code_param": (0.0, 1.0)}

        report = validate_spec_code_alignment(spec, configuration_space=code_space)
        summary = report.summary()

        assert "Spec Drift Report:" in summary
        assert "[WARNING]" in summary or "[INFO]" in summary


class TestValidateTvarTypesMatch:
    """Tests for validate_tvar_types_match function."""

    def test_no_issues_when_types_match(self) -> None:
        """No issues when spec and code types match."""
        spec = MockTVLSpecArtifact(
            configuration_space={
                "temperature": (0.0, 1.0),  # Numeric
                "model": ["gpt-4", "claude"],  # Categorical
            }
        )
        code_space = {
            "temperature": (0.0, 1.0),  # Numeric
            "model": ["gpt-4", "claude"],  # Categorical
        }

        report = validate_tvar_types_match(spec, code_space)

        assert len(report.issues) == 0

    def test_type_mismatch_numeric_vs_categorical(self) -> None:
        """Warning when spec has numeric but code has categorical."""
        spec = MockTVLSpecArtifact(
            configuration_space={
                "param": (0.0, 1.0),  # Numeric in spec
            }
        )
        code_space = {
            "param": ["a", "b", "c"],  # Categorical in code
        }

        report = validate_tvar_types_match(spec, code_space)

        assert len(report.issues) == 1
        assert report.issues[0].severity == DriftSeverity.WARNING
        assert "Type mismatch" in report.issues[0].message
        assert "param" in report.issues[0].message

    def test_type_mismatch_categorical_vs_numeric(self) -> None:
        """Warning when spec has categorical but code has numeric."""
        spec = MockTVLSpecArtifact(
            configuration_space={
                "param": ["a", "b", "c"],  # Categorical in spec
            }
        )
        code_space = {
            "param": (0.0, 1.0),  # Numeric in code
        }

        report = validate_tvar_types_match(spec, code_space)

        assert len(report.issues) == 1
        assert "Type mismatch" in report.issues[0].message

    def test_none_code_space_returns_empty(self) -> None:
        """None code space returns empty report."""
        spec = MockTVLSpecArtifact(configuration_space={"param": (0.0, 1.0)})

        report = validate_tvar_types_match(spec, None)

        assert len(report.issues) == 0

    def test_only_checks_common_params(self) -> None:
        """Only checks params that exist in both spec and code."""
        spec = MockTVLSpecArtifact(
            configuration_space={
                "shared": (0.0, 1.0),
                "spec_only": ["a", "b"],
            }
        )
        code_space = {
            "shared": (0.0, 1.0),
            "code_only": (10, 100),
        }

        report = validate_tvar_types_match(spec, code_space)

        # Only "shared" is checked - it matches, so no issues
        assert len(report.issues) == 0


class TestSpecDriftReportMethods:
    """Tests for SpecDriftReport helper methods."""

    def test_has_errors_true_when_error_exists(self) -> None:
        """has_errors is True when ERROR severity issue exists."""
        report = SpecDriftReport(
            issues=[SpecDriftIssue(severity=DriftSeverity.ERROR, message="Error msg")]
        )
        assert report.has_errors is True

    def test_has_errors_false_when_no_errors(self) -> None:
        """has_errors is False when no ERROR severity issues."""
        report = SpecDriftReport(
            issues=[
                SpecDriftIssue(severity=DriftSeverity.WARNING, message="Warning msg")
            ]
        )
        assert report.has_errors is False

    def test_has_warnings_true_when_warning_exists(self) -> None:
        """has_warnings is True when WARNING severity issue exists."""
        report = SpecDriftReport(
            issues=[
                SpecDriftIssue(severity=DriftSeverity.WARNING, message="Warning msg")
            ]
        )
        assert report.has_warnings is True

    def test_summary_no_issues(self) -> None:
        """Summary for empty report."""
        report = SpecDriftReport(issues=[])
        assert report.summary() == "No spec drift detected"

    def test_warn_if_issues_emits_warnings(self) -> None:
        """warn_if_issues emits Python warnings for WARNING severity."""
        report = SpecDriftReport(
            issues=[
                SpecDriftIssue(severity=DriftSeverity.WARNING, message="Test warning")
            ]
        )

        with pytest.warns(UserWarning, match="Test warning"):
            report.warn_if_issues()
