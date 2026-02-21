"""Configuration constraints system for Traigent optimization."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE FUNC-INVOKERS REQ-ORCH-003 REQ-INJ-002 SYNC-OptimizationFlow

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from traigent.core.constants import DEFAULT_MODEL
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConstraintViolation:
    """Represents a constraint violation."""

    constraint_name: str
    message: str
    violating_config: dict[str, Any]
    suggestion: str | None = None


class Constraint(ABC):
    """Abstract base class for configuration constraints."""

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize constraint.

        Args:
            name: Constraint name for identification
            description: Human-readable description
        """
        self.name = name
        self.description = description

    @abstractmethod
    def validate(self, config: dict[str, Any]) -> bool:
        """Check if configuration satisfies the constraint.

        Args:
            config: Configuration to validate

        Returns:
            True if constraint is satisfied
        """
        pass

    @abstractmethod
    def get_violation_message(self, config: dict[str, Any]) -> str:
        """Get violation message for failed constraint.

        Args:
            config: Configuration that violated the constraint

        Returns:
            Human-readable violation message
        """
        pass

    def get_suggestion(self, config: dict[str, Any]) -> str | None:
        """Get suggestion for fixing constraint violation.

        Args:
            config: Configuration that violated the constraint

        Returns:
            Optional suggestion for fixing the violation
        """
        return None


class ParameterRangeConstraint(Constraint):
    """Constraint that limits parameter values to specific ranges."""

    def __init__(
        self,
        parameter: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """Initialize parameter range constraint.

        Args:
            parameter: Parameter name
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
        """
        name = f"{parameter}_range"
        description = f"Parameter '{parameter}' must be in range"
        super().__init__(name, description)

        self.parameter = parameter
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, config: dict[str, Any]) -> bool:
        """Check if parameter value is in allowed range."""
        if self.parameter not in config:
            return True  # No constraint if parameter not present

        value = config[self.parameter]

        if not isinstance(value, (int, float)):
            return False

        if self.min_value is not None and value < self.min_value:
            return False

        if self.max_value is not None and value > self.max_value:
            return False

        return True

    def get_violation_message(self, config: dict[str, Any]) -> str:
        """Get violation message."""
        value = config.get(self.parameter, "missing")

        if self.min_value is not None and self.max_value is not None:
            return f"Parameter '{self.parameter}' value {value} not in range [{self.min_value}, {self.max_value}]"
        elif self.min_value is not None:
            return f"Parameter '{self.parameter}' value {value} below minimum {self.min_value}"
        elif self.max_value is not None:
            return f"Parameter '{self.parameter}' value {value} above maximum {self.max_value}"
        else:
            return f"Parameter '{self.parameter}' has invalid value {value}"

    def get_suggestion(self, config: dict[str, Any]) -> str | None:
        """Get suggestion for fixing violation."""
        if self.min_value is not None and self.max_value is not None:
            return f"Use value between {self.min_value} and {self.max_value}"
        elif self.min_value is not None:
            return f"Use value >= {self.min_value}"
        elif self.max_value is not None:
            return f"Use value <= {self.max_value}"
        return None


class ConditionalConstraint(Constraint):
    """Constraint that applies only when certain conditions are met."""

    def __init__(
        self,
        name: str,
        condition: Callable[[dict[str, Any]], bool],
        constraint: Constraint,
    ) -> None:
        """Initialize conditional constraint.

        Args:
            name: Constraint name
            condition: Function that returns True when constraint should apply
            constraint: The constraint to apply when condition is True
        """
        super().__init__(name, f"Conditional: {constraint.description}")
        self.condition = condition
        self.constraint = constraint

    def validate(self, config: dict[str, Any]) -> bool:
        """Check constraint only if condition is met."""
        if not self.condition(config):
            return True  # Constraint doesn't apply

        return self.constraint.validate(config)

    def get_violation_message(self, config: dict[str, Any]) -> str:
        """Get violation message."""
        return f"Conditional constraint violated: {self.constraint.get_violation_message(config)}"

    def get_suggestion(self, config: dict[str, Any]) -> str | None:
        """Get suggestion for fixing violation."""
        suggestion = self.constraint.get_suggestion(config)
        if suggestion:
            return f"When condition applies: {suggestion}"
        return None


class MutuallyExclusiveConstraint(Constraint):
    """Constraint ensuring certain parameter combinations don't occur together."""

    def __init__(
        self, parameters: list[str], values: list[Any], max_simultaneous: int = 1
    ) -> None:
        """Initialize mutually exclusive constraint.

        Args:
            parameters: List of parameter names
            values: List of values that can't occur together
            max_simultaneous: Maximum number of parameters that can have these values
        """
        name = f"mutex_{'+'.join(parameters)}"
        description = (
            f"Parameters {parameters} cannot have values {values} simultaneously"
        )
        super().__init__(name, description)

        self.parameters = parameters
        self.values: set[Any] = set(values)
        self.max_simultaneous = max_simultaneous

    def validate(self, config: dict[str, Any]) -> bool:
        """Check that not too many parameters have forbidden values."""
        count = 0

        for param in self.parameters:
            if param in config and config[param] in self.values:
                count += 1

        return count <= self.max_simultaneous

    def get_violation_message(self, config: dict[str, Any]) -> str:
        """Get violation message."""
        violating_params = []
        for param in self.parameters:
            if param in config and config[param] in self.values:
                violating_params.append(f"{param}={config[param]}")

        return (
            f"Too many parameters with restricted values: {', '.join(violating_params)}"
        )

    def get_suggestion(self, config: dict[str, Any]) -> str | None:
        """Get suggestion for fixing violation."""
        return f"Change at least one of {self.parameters} to avoid values {list(self.values)}"


class DependencyConstraint(Constraint):
    """Constraint ensuring parameter dependencies are satisfied."""

    def __init__(
        self, dependent_param: str, dependency_param: str, dependency_values: list[Any]
    ) -> None:
        """Initialize dependency constraint.

        Args:
            dependent_param: Parameter that depends on another
            dependency_param: Parameter that must have specific values
            dependency_values: Required values for dependency parameter
        """
        name = f"{dependent_param}_depends_on_{dependency_param}"
        description = f"Parameter '{dependent_param}' requires '{dependency_param}' to be in {dependency_values}"
        super().__init__(name, description)

        self.dependent_param = dependent_param
        self.dependency_param = dependency_param
        self.dependency_values: set[Any] = set(dependency_values)

    def validate(self, config: dict[str, Any]) -> bool:
        """Check dependency is satisfied when dependent parameter is present."""
        if self.dependent_param not in config:
            return True  # No constraint if dependent param not present

        if self.dependency_param not in config:
            return False  # Dependency missing

        return config[self.dependency_param] in self.dependency_values

    def get_violation_message(self, config: dict[str, Any]) -> str:
        """Get violation message."""
        dep_value = config.get(self.dependency_param, "missing")
        return (
            f"Parameter '{self.dependent_param}' requires '{self.dependency_param}' "
            f"to be in {list(self.dependency_values)}, but got {dep_value}"
        )

    def get_suggestion(self, config: dict[str, Any]) -> str | None:
        """Get suggestion for fixing violation."""
        return (
            f"Set '{self.dependency_param}' to one of: {list(self.dependency_values)}"
        )


class ResourceConstraint(Constraint):
    """Constraint based on resource consumption (cost, memory, etc.)."""

    def __init__(
        self,
        name: str,
        resource_calculator: Callable[[dict[str, Any]], float],
        max_resource: float,
    ) -> None:
        """Initialize resource constraint.

        Args:
            name: Constraint name
            resource_calculator: Function that calculates resource usage from config
            max_resource: Maximum allowed resource usage
        """
        super().__init__(name, f"Resource usage must be <= {max_resource}")
        self.resource_calculator = resource_calculator
        self.max_resource = max_resource

    def validate(self, config: dict[str, Any]) -> bool:
        """Check if resource usage is within limits."""
        try:
            usage = self.resource_calculator(config)
            return usage <= self.max_resource
        except KeyError:
            # Missing keys mean we can't calculate usage, treat as invalid
            return False
        except Exception as e:
            logger.warning(
                f"Resource constraint validation error (treating as invalid): {e}",
                exc_info=True,
            )
            return False  # Invalid config for resource calculation

    def get_violation_message(self, config: dict[str, Any]) -> str:
        """Get violation message."""
        try:
            usage = self.resource_calculator(config)
            return f"Resource usage {usage:.3f} exceeds limit {self.max_resource}"
        except Exception as e:
            return f"Cannot calculate resource usage: {e}"

    def get_suggestion(self, config: dict[str, Any]) -> str | None:
        """Get suggestion for fixing violation."""
        return f"Reduce resource usage to <= {self.max_resource}"


class CustomConstraint(Constraint):
    """Custom constraint defined by user function."""

    def __init__(
        self,
        name: str,
        validator: Callable[[dict[str, Any]], bool],
        message_generator: Callable[[dict[str, Any]], str],
        suggestion_generator: Callable[[dict[str, Any]], str] | None = None,
    ) -> None:
        """Initialize custom constraint.

        Args:
            name: Constraint name
            validator: Function that returns True if constraint is satisfied
            message_generator: Function that generates violation message
            suggestion_generator: Optional function that generates suggestions
        """
        super().__init__(name, "Custom constraint")
        self.validator = validator
        self.message_generator = message_generator
        self.suggestion_generator = suggestion_generator

    def validate(self, config: dict[str, Any]) -> bool:
        """Check constraint using custom validator."""
        try:
            return self.validator(config)
        except Exception as e:
            logger.debug(
                f"Custom constraint '{self.name}' validation failed (treating as invalid): {e}"
            )
            return False

    def get_violation_message(self, config: dict[str, Any]) -> str:
        """Get violation message using custom generator."""
        try:
            return self.message_generator(config)
        except Exception as e:
            return f"Custom constraint '{self.name}' violated: {e}"

    def get_suggestion(self, config: dict[str, Any]) -> str | None:
        """Get suggestion using custom generator."""
        if self.suggestion_generator:
            try:
                return self.suggestion_generator(config)
            except Exception as e:
                logger.debug(f"Custom constraint suggestion generation failed: {e}")
        return None


class ConstraintManager:
    """Manages and validates configuration constraints."""

    def __init__(self) -> None:
        """Initialize constraint manager."""
        self.constraints: list[Constraint] = []

    def add_constraint(self, constraint: Constraint) -> None:
        """Add constraint to manager.

        Args:
            constraint: Constraint to add
        """
        self.constraints.append(constraint)

    def remove_constraint(self, name: str) -> bool:
        """Remove constraint by name.

        Args:
            name: Constraint name to remove

        Returns:
            True if constraint was found and removed
        """
        for i, constraint in enumerate(self.constraints):
            if constraint.name == name:
                del self.constraints[i]
                return True
        return False

    def validate_configuration(
        self, config: dict[str, Any]
    ) -> tuple[bool, list[ConstraintViolation]]:
        """Validate configuration against all constraints.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []

        for constraint in self.constraints:
            if not constraint.validate(config):
                violation = ConstraintViolation(
                    constraint_name=constraint.name,
                    message=constraint.get_violation_message(config),
                    violating_config=config.copy(),
                    suggestion=constraint.get_suggestion(config),
                )
                violations.append(violation)

        is_valid = len(violations) == 0
        return is_valid, violations

    def filter_valid_configurations(
        self, configs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter list of configurations to only include valid ones.

        Args:
            configs: List of configurations to filter

        Returns:
            List of valid configurations
        """
        valid_configs = []

        for config in configs:
            is_valid, _ = self.validate_configuration(config)
            if is_valid:
                valid_configs.append(config)

        return valid_configs

    def get_constraint_summary(self) -> str:
        """Get summary of all active constraints.

        Returns:
            Human-readable summary of constraints
        """
        if not self.constraints:
            return "No constraints defined"

        lines = [f"Active constraints ({len(self.constraints)}):"]
        for i, constraint in enumerate(self.constraints, 1):
            lines.append(f"  {i}. {constraint.name}: {constraint.description}")

        return "\n".join(lines)


# Convenience functions for creating common constraints
def temperature_constraint(
    min_temp: float = 0.0, max_temp: float = 2.0
) -> ParameterRangeConstraint:
    """Create constraint for temperature parameter."""
    return ParameterRangeConstraint("temperature", min_temp, max_temp)


def max_tokens_constraint(
    min_tokens: int = 1, max_tokens: int = 4000
) -> ParameterRangeConstraint:
    """Create constraint for max_tokens parameter."""
    return ParameterRangeConstraint("max_tokens", min_tokens, max_tokens)


def model_cost_constraint(max_cost_per_1k_tokens: float = 0.1) -> ResourceConstraint:
    """Create constraint based on model cost using litellm library.

    Uses litellm for cost calculation with fallback to simplified pricing
    for unknown models. The constraint validates that estimated cost per
    1k tokens is within the specified limit.

    IMPORTANT: Unknown models that can't be priced will FAIL the constraint
    (return infinity) to prevent them from silently passing.
    """

    def calculate_cost(config: dict[str, Any]) -> float:
        """Calculate cost based on model and token usage.

        Returns float('inf') for unknown models to ensure they fail the constraint.
        """
        model = config.get("model", DEFAULT_MODEL)
        max_tokens = config.get("max_tokens", 150)

        # Method 1: Try litellm.cost_per_token directly with 1000 tokens
        # This is the most accurate method
        try:
            import litellm

            # Get cost per token directly from litellm (1000 input + 1000 output)
            input_cost, output_cost = litellm.cost_per_token(
                model=model, prompt_tokens=1000, completion_tokens=1000
            )

            if input_cost > 0 or output_cost > 0:
                # Calculate average cost per 1k tokens
                avg_cost_per_1k = (input_cost + output_cost) / 2
                return float(avg_cost_per_1k * (max_tokens / 1000))

            # Zero cost could mean unknown model or legitimately free
            # Check if model is actually known to litellm
            from traigent.utils.cost_calculator import _is_model_known_to_litellm

            if _is_model_known_to_litellm(model):
                # Known model with 0 cost (free tier) - allow it
                return 0.0

        except ImportError:
            pass  # litellm not available
        except Exception as e:
            logger.debug(
                "litellm cost_per_token failed for model %r: %s",
                model,
                e,
                exc_info=True,
            )
            pass  # Fall through to fallback

        # Method 2: Fallback to ESTIMATION_MODEL_PRICING from cost_calculator
        try:
            from traigent.utils.cost_calculator import (
                ESTIMATION_MODEL_PRICING,
                _normalize_model_for_fallback,
            )

            base_model = _normalize_model_for_fallback(model)

            # Try exact match first
            pricing = None
            for key, value in ESTIMATION_MODEL_PRICING.items():
                if key.lower() == base_model:
                    pricing = value
                    break

            # Try prefix matching - prefer longest match
            if not pricing:
                best_match_len = 0
                for model_key, model_pricing in ESTIMATION_MODEL_PRICING.items():
                    key_lower = model_key.lower()
                    if base_model.startswith(key_lower):
                        if len(key_lower) > best_match_len:
                            best_match_len = len(key_lower)
                            pricing = model_pricing
                    elif key_lower.startswith(base_model):
                        if len(base_model) > best_match_len:
                            best_match_len = len(base_model)
                            pricing = model_pricing

            if pricing:
                # Calculate cost per 1k tokens from per-token pricing
                input_cost_per_1k = pricing["input_cost_per_token"] * 1000
                output_cost_per_1k = pricing["output_cost_per_token"] * 1000
                # Average input/output cost
                avg_cost_per_1k = (input_cost_per_1k + output_cost_per_1k) / 2
                return float(avg_cost_per_1k * (max_tokens / 1000))

        except ImportError:
            pass  # ESTIMATION_MODEL_PRICING not available

        # Ultimate fallback: Unknown model - return infinity to FAIL the constraint
        # This prevents unknown models from silently passing cost checks
        logger.warning(
            "Unknown model %r has no pricing info - failing cost constraint", model
        )
        return float("inf")

    return ResourceConstraint("model_cost", calculate_cost, max_cost_per_1k_tokens)


def fast_model_low_temp_constraint() -> ConditionalConstraint:
    """Create constraint: if using fast model, temperature should be low for consistency."""

    def is_fast_model(config: dict[str, Any]) -> bool:
        fast_models = ["gpt-4o-mini", "claude-3-haiku"]
        return config.get("model") in fast_models

    temp_constraint = ParameterRangeConstraint("temperature", max_value=0.7)

    return ConditionalConstraint("fast_model_low_temp", is_fast_model, temp_constraint)


def exclusive_high_quality_strategies() -> MutuallyExclusiveConstraint:
    """Create constraint: can't use multiple high-quality strategies simultaneously."""
    return MutuallyExclusiveConstraint(
        ["model", "strategy"],
        ["GPT-4o", "claude-3-opus", "high_quality"],
        max_simultaneous=1,
    )
