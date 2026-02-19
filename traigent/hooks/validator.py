"""Agent configuration validator for Traigent hooks.

Validates @traigent.optimize decorated functions against constraints
defined in traigent.yml.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability CONC-Quality-Maintainability FUNC-ORCH-LIFECYCLE REQ-ORCH-003

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traigent.hooks.config import HooksConfig, load_hooks_config
from traigent.utils.cost_calculator import FALLBACK_MODEL_PRICING
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


def _build_model_cost_per_1k() -> dict[str, float]:
    """Derive MODEL_COST_PER_1K from the canonical FALLBACK_MODEL_PRICING.

    Computes a blended (input+output)/2 average per-1K-token cost for each
    model, then adds short-name aliases so that user configs in traigent.yml
    using names like ``"gpt-4"`` or ``"claude-3-haiku"`` still resolve.
    """
    result: dict[str, float] = {}
    for model, pricing in FALLBACK_MODEL_PRICING.items():
        avg_per_token = (
            pricing["input_cost_per_token"] + pricing["output_cost_per_token"]
        ) / 2
        result[model] = avg_per_token * 1000

    # Compatibility aliases — short names that users may configure.
    # Maps legacy/short names to their closest canonical equivalent.
    _aliases = {
        "gpt-4": "gpt-4-turbo",
        "gpt-4-32k": "gpt-4-turbo",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-sonnet-20240229": "claude-3-5-sonnet-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    }
    for alias, canonical in _aliases.items():
        if alias not in result and canonical in result:
            result[alias] = result[canonical]

    return result


# Model cost estimates per 1K tokens (input + output average).
# Derived from the canonical FALLBACK_MODEL_PRICING in cost_calculator.py.
MODEL_COST_PER_1K: dict[str, float] = _build_model_cost_per_1k()

# Default tokens per query estimate
DEFAULT_TOKENS_PER_QUERY = 1000


@dataclass
class ValidationIssue:
    """A single validation issue found during agent validation."""

    severity: str  # "error" or "warning"
    code: str  # Issue code (e.g., "BLOCKED_MODEL", "COST_EXCEEDED")
    message: str  # Human-readable message
    suggestion: str | None = None  # How to fix the issue


@dataclass
class ValidationResult:
    """Result of validating an agent configuration."""

    function_name: str
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    # Configuration details
    models_found: list[str] = field(default_factory=list)
    estimated_cost_per_query: float | None = None

    @property
    def has_errors(self) -> bool:
        """Check if there are any blocking errors."""
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return bool(self.warnings)

    def get_summary(self) -> str:
        """Get a one-line summary of validation result."""
        if self.is_valid:
            warnings_text = f" ({len(self.warnings)} warnings)" if self.warnings else ""
            return f"{self.function_name}: PASSED{warnings_text}"
        else:
            return f"{self.function_name}: FAILED ({len(self.issues)} issues)"


@dataclass
class AgentInfo:
    """Information about a discovered @traigent.optimize decorated function."""

    name: str
    file_path: str
    configuration_space: dict[str, Any]
    objectives: list[str]
    constraints: list[Any]

    @property
    def models(self) -> list[str]:
        """Extract model names from configuration space."""
        models: list[str] = []
        for key in ["model", "models", "llm_model", "model_name"]:
            if key in self.configuration_space:
                value = self.configuration_space[key]
                if isinstance(value, list):
                    models.extend(str(v) for v in value)
                else:
                    models.append(str(value))
        return models

    @property
    def max_tokens(self) -> int:
        """Get maximum tokens from configuration or default."""
        for key in ["max_tokens", "maxTokens", "max_output_tokens"]:
            if key in self.configuration_space:
                value = self.configuration_space[key]
                if isinstance(value, tuple):
                    return int(value[1])  # Use upper bound
                elif isinstance(value, list):
                    return int(max(value))
                else:
                    return int(value)
        return DEFAULT_TOKENS_PER_QUERY


class AgentValidator:
    """Validates agent configurations against Traigent constraints."""

    def __init__(self, config: HooksConfig | None = None) -> None:
        """Initialize validator with configuration.

        Args:
            config: HooksConfig to use, or None to auto-load from traigent.yml
        """
        if config is None:
            config = load_hooks_config()
        self.config = config

    def validate_agent(self, agent: AgentInfo) -> ValidationResult:
        """Validate a single agent against all constraints.

        Args:
            agent: AgentInfo describing the decorated function

        Returns:
            ValidationResult with all issues found
        """
        issues: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []

        # Run all validation checks
        issues.extend(self._validate_models(agent))
        cost_issues, cost_warnings = self._validate_cost(agent)
        issues.extend(cost_issues)
        warnings.extend(cost_warnings)

        # Calculate estimated cost
        estimated_cost = self._estimate_cost_per_query(agent)

        # Determine if valid (no errors)
        is_valid = not any(issue.severity == "error" for issue in issues)

        return ValidationResult(
            function_name=agent.name,
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            models_found=agent.models,
            estimated_cost_per_query=estimated_cost,
        )

    def _validate_models(self, agent: AgentInfo) -> list[ValidationIssue]:
        """Validate model constraints.

        Args:
            agent: AgentInfo to validate

        Returns:
            List of validation issues
        """
        issues = []
        model_constraints = self.config.constraints.models

        for model in agent.models:
            # Check blocked models
            if model in model_constraints.blocked_models:
                reason = model_constraints.blocked_reasons.get(
                    model, "Listed in blocked_models"
                )
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="BLOCKED_MODEL",
                        message=f"Model '{model}' is in blocked_models list",
                        suggestion=f"Reason: {reason}. Use an allowed model: {', '.join(model_constraints.allowed_models[:3]) or 'any non-blocked model'}",
                    )
                )

            # Check allowed models (if whitelist is specified)
            if (
                model_constraints.allowed_models
                and model not in model_constraints.allowed_models
            ):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="MODEL_NOT_ALLOWED",
                        message=f"Model '{model}' is not in allowed_models list",
                        suggestion=f"Use an allowed model: {', '.join(model_constraints.allowed_models)}",
                    )
                )

        return issues

    def _validate_cost(
        self, agent: AgentInfo
    ) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
        """Validate cost constraints.

        Args:
            agent: AgentInfo to validate

        Returns:
            Tuple of (errors, warnings)
        """
        errors: list[ValidationIssue] = []
        warnings: list[ValidationIssue] = []

        cost_constraints = self.config.constraints.cost
        estimated_cost = self._estimate_cost_per_query(agent)

        if estimated_cost is None:
            return errors, warnings

        # Check max cost per query
        if cost_constraints.max_cost_per_query is not None:
            if estimated_cost > cost_constraints.max_cost_per_query:
                errors.append(
                    ValidationIssue(
                        severity="error",
                        code="COST_EXCEEDED",
                        message=f"Estimated cost ${estimated_cost:.4f}/query exceeds max_cost_per_query (${cost_constraints.max_cost_per_query})",
                        suggestion="Use a more cost-effective model or reduce max_tokens",
                    )
                )
            elif (
                estimated_cost
                > cost_constraints.max_cost_per_query
                * cost_constraints.warn_threshold_pct
            ):
                warnings.append(
                    ValidationIssue(
                        severity="warning",
                        code="COST_WARNING",
                        message=f"Estimated cost ${estimated_cost:.4f}/query is close to limit (${cost_constraints.max_cost_per_query})",
                        suggestion="Consider cost optimization for production",
                    )
                )

        return errors, warnings

    def _estimate_cost_per_query(self, agent: AgentInfo) -> float | None:
        """Estimate cost per query for an agent.

        Args:
            agent: AgentInfo to estimate cost for

        Returns:
            Estimated cost in dollars, or None if cannot estimate
        """
        if not agent.models:
            return None

        max_tokens = agent.max_tokens

        # Calculate max cost across all configured models
        costs = []
        for model in agent.models:
            cost_per_1k = MODEL_COST_PER_1K.get(model.lower())
            if cost_per_1k is None:
                # Try partial match for model variants
                for known_model, known_cost in MODEL_COST_PER_1K.items():
                    if known_model in model.lower() or model.lower() in known_model:
                        cost_per_1k = known_cost
                        break

            if cost_per_1k is not None:
                costs.append(cost_per_1k * (max_tokens / 1000))

        return max(costs) if costs else None

    def validate_file(self, file_path: Path | str) -> list[ValidationResult]:
        """Validate all @traigent.optimize functions in a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            List of ValidationResults for each function found
        """
        from traigent.cli.function_discovery import discover_optimized_functions

        file_path = Path(file_path)
        results = []

        try:
            functions = discover_optimized_functions(str(file_path))

            for func_info in functions:
                agent = AgentInfo(
                    name=func_info.name,
                    file_path=str(file_path),
                    configuration_space=func_info.decorator_config.get(
                        "configuration_space", {}
                    ),
                    objectives=func_info.objectives,
                    constraints=func_info.decorator_config.get("constraints", []),
                )
                results.append(self.validate_agent(agent))

        except Exception as e:
            logger.warning(f"Failed to validate {file_path}: {e}")
            results.append(
                ValidationResult(
                    function_name=str(file_path),
                    is_valid=False,
                    issues=[
                        ValidationIssue(
                            severity="error",
                            code="DISCOVERY_ERROR",
                            message=f"Failed to discover functions: {e}",
                            suggestion="Check file syntax and imports",
                        )
                    ],
                )
            )

        return results

    def validate_directory(
        self, directory: Path | str, recursive: bool = True
    ) -> list[ValidationResult]:
        """Validate all Python files in a directory.

        Args:
            directory: Directory to scan
            recursive: Whether to scan subdirectories

        Returns:
            List of ValidationResults for all functions found
        """
        directory = Path(directory)
        results = []

        pattern = "**/*.py" if recursive else "*.py"
        for py_file in directory.glob(pattern):
            # Skip common non-source directories
            if any(
                part.startswith((".", "_", "venv", "node_modules", "__pycache__"))
                for part in py_file.parts
            ):
                continue

            # Check skip patterns
            if self._should_skip(py_file):
                continue

            results.extend(self.validate_file(py_file))

        return results

    def _should_skip(self, file_path: Path) -> bool:
        """Check if a file should be skipped based on patterns.

        Args:
            file_path: File path to check

        Returns:
            True if file should be skipped
        """
        import fnmatch

        file_str = str(file_path)

        for pattern in self.config.skip_patterns:
            if fnmatch.fnmatch(file_str, pattern):
                return True

        return False


def validate_agents_for_push(
    target_files: list[Path] | None = None,
    config_path: Path | str | None = None,
) -> tuple[bool, list[ValidationResult]]:
    """Validate agent configurations for a Git push operation.

    This is the main entry point for the pre-push hook.

    Args:
        target_files: Specific files to validate (None = auto-detect changed files)
        config_path: Path to traigent.yml (None = auto-detect)

    Returns:
        Tuple of (should_allow_push, list_of_results)
    """
    config = load_hooks_config(config_path)

    if not config.enabled:
        logger.info("Traigent hooks are disabled")
        return True, []

    validator = AgentValidator(config)
    all_results = []

    if target_files:
        for file_path in target_files:
            all_results.extend(validator.validate_file(file_path))
    else:
        # Validate current directory
        all_results = validator.validate_directory(Path.cwd())

    # Determine if push should be allowed
    has_errors = any(not result.is_valid for result in all_results)
    has_warnings = any(result.has_warnings for result in all_results)

    should_block = has_errors or (config.fail_on_warning and has_warnings)

    return not should_block, all_results
