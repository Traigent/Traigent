"""Spec drift detection for TVL specifications.

Validates alignment between TVL spec configuration and decorator parameters.

Traceability: CONC-Quality-Reliability REQ-TVLSPEC-012
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from traigent.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class DriftSeverity(str, Enum):
    """Severity level for spec drift issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class SpecDriftIssue:
    """Represents a single spec drift issue."""

    severity: DriftSeverity
    message: str
    spec_params: set[str] | None = None
    code_params: set[str] | None = None
    missing_in_code: set[str] | None = None
    missing_in_spec: set[str] | None = None


@dataclass
class SpecDriftReport:
    """Report of all spec drift issues found."""

    issues: list[SpecDriftIssue]

    @property
    def has_errors(self) -> bool:
        """Check if report contains any errors."""
        return any(issue.severity == DriftSeverity.ERROR for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if report contains any warnings."""
        return any(issue.severity == DriftSeverity.WARNING for issue in self.issues)

    def raise_if_errors(self) -> None:
        """Raise ValueError if there are any errors in the report."""
        if self.has_errors:
            error_messages = [
                issue.message
                for issue in self.issues
                if issue.severity == DriftSeverity.ERROR
            ]
            raise ValueError(f"Spec drift errors detected: {'; '.join(error_messages)}")

    def warn_if_issues(self) -> None:
        """Emit warnings for all issues in the report."""
        for issue in self.issues:
            if issue.severity == DriftSeverity.WARNING:
                warnings.warn(f"Spec drift warning: {issue.message}", stacklevel=3)
            elif issue.severity == DriftSeverity.INFO:
                logger.info(f"Spec drift info: {issue.message}")

    def summary(self) -> str:
        """Generate human-readable summary of drift issues."""
        if not self.issues:
            return "No spec drift detected"

        lines = ["Spec Drift Report:"]
        for issue in self.issues:
            lines.append(f"  [{issue.severity.value.upper()}] {issue.message}")
        return "\n".join(lines)


def extract_config_space_params(config_space: dict[str, Any] | None) -> set[str]:
    """Extract parameter names from a configuration space.

    Handles both raw dict configuration spaces and typed ParameterRange objects.

    Args:
        config_space: Configuration space dictionary (may contain ParameterRange objects)

    Returns:
        Set of parameter names
    """
    if config_space is None:
        return set()

    return set(config_space.keys())


def extract_decorator_params(func: Callable[..., Any]) -> set[str]:
    """Extract configuration space parameter names from a decorated function.

    Looks for the __traigent_config__ or similar attributes that store
    the decorator's configuration space.

    Args:
        func: Decorated function

    Returns:
        Set of parameter names from the decorator's configuration space
    """
    # Check for OptimizedFunction wrapper
    if hasattr(func, "configuration_space"):
        config_space = func.configuration_space
        if config_space is not None:
            return extract_config_space_params(config_space)

    # Check for __wrapped__ attribute (functools.wraps)
    if hasattr(func, "__wrapped__"):
        return extract_decorator_params(func.__wrapped__)

    # Check for stored config in __traigent__ attribute
    if hasattr(func, "__traigent__"):
        traigent_meta = func.__traigent__
        if isinstance(traigent_meta, dict) and "configuration_space" in traigent_meta:
            return extract_config_space_params(traigent_meta["configuration_space"])

    return set()


class SpecArtifactLike(Protocol):
    """Protocol for spec-like objects consumed by drift validators."""

    configuration_space: dict[str, Any] | None


def validate_spec_code_alignment(
    spec: SpecArtifactLike,
    configuration_space: dict[str, Any] | None = None,
    *,
    decorated_func: Callable[..., Any] | None = None,
    strict: bool = False,
) -> SpecDriftReport:
    """Validate that TVL spec parameters match decorator configuration space.

    Detects "spec drift" where the TVL specification has diverged from the
    actual code implementation.

    Args:
        spec: The loaded TVL specification artifact
        configuration_space: Explicit configuration space to compare against.
            If provided, takes precedence over decorated_func.
        decorated_func: Function decorated with @traigent.optimize.
            Used to extract configuration space if not provided explicitly.
        strict: If True, any drift is treated as an error. If False,
            missing params are warnings, extra params are info.

    Returns:
        SpecDriftReport containing all detected issues

    Example::

        spec = load_tvl_spec("optimization.tvl.yaml")
        report = validate_spec_code_alignment(spec, decorated_func=my_agent)
        if report.has_errors:
            print(report.summary())
    """
    issues: list[SpecDriftIssue] = []

    # Get spec parameters
    spec_params = extract_config_space_params(spec.configuration_space)

    # Get code parameters
    if configuration_space is not None:
        code_params = extract_config_space_params(configuration_space)
    elif decorated_func is not None:
        code_params = extract_decorator_params(decorated_func)
    else:
        # No code parameters provided - can't validate
        issues.append(
            SpecDriftIssue(
                severity=DriftSeverity.INFO,
                message="No configuration_space or decorated_func provided for comparison",
            )
        )
        return SpecDriftReport(issues=issues)

    # Find drift
    missing_in_code = spec_params - code_params
    missing_in_spec = code_params - spec_params

    # No drift detected
    if not missing_in_code and not missing_in_spec:
        return SpecDriftReport(issues=[])

    # Report missing in code (spec has params not in code)
    if missing_in_code:
        severity = DriftSeverity.ERROR if strict else DriftSeverity.WARNING
        issues.append(
            SpecDriftIssue(
                severity=severity,
                message=(
                    f"TVL spec defines parameters not found in code: {missing_in_code}"
                ),
                spec_params=spec_params,
                code_params=code_params,
                missing_in_code=missing_in_code,
            )
        )

    # Report missing in spec (code has params not in spec)
    if missing_in_spec:
        severity = DriftSeverity.ERROR if strict else DriftSeverity.INFO
        issues.append(
            SpecDriftIssue(
                severity=severity,
                message=(
                    f"Code defines parameters not found in TVL spec: {missing_in_spec}"
                ),
                spec_params=spec_params,
                code_params=code_params,
                missing_in_spec=missing_in_spec,
            )
        )

    return SpecDriftReport(issues=issues)


def validate_tvar_types_match(
    spec: SpecArtifactLike,
    configuration_space: dict[str, Any] | None,
) -> SpecDriftReport:
    """Validate that TVL spec tvar types match code configuration types.

    Checks that:
    - Numeric tvars in spec correspond to Range/IntRange in code
    - Categorical tvars in spec correspond to Choices in code
    - Type constraints are compatible

    Args:
        spec: The loaded TVL specification artifact
        configuration_space: Code's configuration space dictionary

    Returns:
        SpecDriftReport containing type mismatch issues
    """
    issues: list[SpecDriftIssue] = []

    if configuration_space is None:
        return SpecDriftReport(issues=[])

    spec_space = spec.configuration_space or {}
    code_space = configuration_space

    # Check each parameter in both spec and code
    common_params = set(spec_space.keys()) & set(code_space.keys())

    for param in common_params:
        spec_config = spec_space[param]
        code_config = code_space[param]

        # Determine spec type
        spec_is_numeric = _is_numeric_config(spec_config)
        spec_is_categorical = _is_categorical_config(spec_config)

        # Determine code type
        code_is_numeric = _is_numeric_config(code_config)
        code_is_categorical = _is_categorical_config(code_config)

        # Check for type mismatch
        if spec_is_numeric and code_is_categorical:
            issues.append(
                SpecDriftIssue(
                    severity=DriftSeverity.WARNING,
                    message=(
                        f"Type mismatch for '{param}': "
                        f"spec defines numeric range, code defines categorical"
                    ),
                )
            )
        elif spec_is_categorical and code_is_numeric:
            issues.append(
                SpecDriftIssue(
                    severity=DriftSeverity.WARNING,
                    message=(
                        f"Type mismatch for '{param}': "
                        f"spec defines categorical, code defines numeric range"
                    ),
                )
            )

    return SpecDriftReport(issues=issues)


def _is_numeric_config(config: Any) -> bool:
    """Check if a config value represents a numeric range."""
    # Handle ParameterRange types (Range, IntRange, LogRange)
    if hasattr(config, "low") and hasattr(config, "high"):
        return True

    # Handle raw tuple (low, high)
    if isinstance(config, tuple) and len(config) == 2:
        return all(isinstance(v, (int, float)) for v in config)

    return False


def _is_categorical_config(config: Any) -> bool:
    """Check if a config value represents categorical choices."""
    # Handle Choices type
    if hasattr(config, "values") and isinstance(
        getattr(config, "values", None), (list, tuple)
    ):
        return True

    # Handle raw list
    if isinstance(config, list):
        return True

    return False
