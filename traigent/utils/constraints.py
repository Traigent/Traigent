"""Configuration constraints system for Traigent optimization.

Model-cost pricing precedence (``model_cost_constraint``, issue #1958):
canonical estimation FIRST (curated table + ``gpt-4`` -> ``gpt-4-turbo``
aliasing), then litellm as a FALLBACK for models outside that table, then
``inf`` for models unknown to both. A model IN the canonical table is ALWAYS
priced by canonical, so its constraint cost is deterministic and equal to the
canonical estimate regardless of whether litellm is installed -- litellm cannot
reintroduce the #1958 divergence for those models.
"""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability CONC-Quality-Reliability FUNC-ORCH-LIFECYCLE FUNC-INVOKERS REQ-ORCH-003 REQ-INJ-002 SYNC-OptimizationFlow

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from traigent.core.constants import DEFAULT_MODEL
from traigent.utils.logging import configure_litellm_logging, get_logger

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


def _try_litellm_pricing(model: str, max_tokens: int) -> float | None:
    """Price the ``max_tokens`` budget via litellm. Returns None if unavailable.

    This is the FALLBACK pricing source for the cost constraint, consulted ONLY
    when the canonical estimation path (:func:`_try_fallback_pricing`) cannot
    price the model (i.e. it is outside Traigent's curated table and alias map).
    It exists so exotic / newer models that litellm knows but the curated table
    does not stay priceable instead of unconditionally failing the constraint.

    The whole ``max_tokens`` budget is modeled as an INPUT-token budget so the
    convention matches the canonical path (both price the budget as input
    tokens). Because canonical runs FIRST, this branch never prices a
    canonical-known model, so it cannot reintroduce the issue #1958 divergence
    (where a litellm-first order priced raw ``gpt-4`` at $0.03 instead of the
    canonical gpt-4->gpt-4-turbo $0.010). Its result is env-dependent (it needs
    litellm installed), which is acceptable for models canonical could not price
    at all.
    """
    try:
        import litellm

        configure_litellm_logging(litellm_module=litellm)
        input_cost, output_cost = litellm.cost_per_token(
            model=model, prompt_tokens=1000, completion_tokens=1000
        )
        if input_cost > 0 or output_cost > 0:
            input_cost_per_1k = input_cost  # cost of 1000 prompt tokens
            return float(input_cost_per_1k * (max_tokens / 1000))

        from traigent.utils.cost_calculator import _is_model_known_to_litellm

        if _is_model_known_to_litellm(model):
            return 0.0
    except ImportError:
        pass
    except Exception as e:
        logger.debug(
            "litellm cost_per_token failed for model %r: %s", model, e, exc_info=True
        )
    return None


def _try_fallback_pricing(model: str, max_tokens: int) -> float | None:
    """Price the ``max_tokens`` budget via the canonical estimation path.

    Delegates to :func:`traigent.utils.cost_calculator._estimation_cost_from_tokens`
    so the constraint price AGREES BY CONSTRUCTION with the canonical estimation
    path (issue #1958): the whole ``max_tokens`` budget is priced as input tokens
    (``output_tokens=0``) and summed. This canonical path applies Traigent's model
    aliasing (e.g. ``gpt-4`` -> ``gpt-4-turbo``) against a static curated table and
    does NOT consult litellm, so it returns the SAME number whether or not litellm
    is installed.

    This is the constraint's PRIMARY pricing source: ``calculate_cost`` calls it
    FIRST, so every model the curated table (plus alias map) can price is
    deterministic and equal to canonical regardless of litellm. litellm is only a
    fallback (:func:`_try_litellm_pricing`) for models this path cannot price.
    That ordering is what keeps litellm from reintroducing the issue #1958
    divergence for canonical-known models (a litellm-first order priced raw
    ``gpt-4`` at $0.03 instead of the canonical $0.010).

    Returns ``None`` for models the canonical path cannot price (it returns
    ``(0.0, 0.0)``) so ``calculate_cost`` can then try the litellm fallback.
    """
    try:
        from traigent.utils.cost_calculator import _estimation_cost_from_tokens
    except ImportError:
        return None

    input_cost, output_cost = _estimation_cost_from_tokens(
        model, max_tokens, 0, _quiet=True
    )
    if input_cost <= 0 and output_cost <= 0:
        return None
    return float(input_cost + output_cost)


def model_cost_constraint(max_cost_per_1k_tokens: float = 0.1) -> ResourceConstraint:
    """Create constraint based on model cost.

    Pricing precedence (issue #1958):

    1. **Canonical FIRST** — :func:`_try_fallback_pricing`, delegating to
       :func:`traigent.utils.cost_calculator._estimation_cost_from_tokens`
       (curated table + ``gpt-4`` -> ``gpt-4-turbo`` aliasing). A model in the
       canonical table (or resolvable via its aliases) is ALWAYS priced here, so
       its cost is deterministic and equal to canonical REGARDLESS of whether
       litellm is installed. This is why litellm can no longer reintroduce the
       #1958 divergence for canonical-known models.
    2. **litellm FALLBACK** — :func:`_try_litellm_pricing`, consulted ONLY when
       canonical returns ``None`` (model outside the curated table). This keeps
       exotic / newer models litellm knows priceable instead of failing outright.
       It is env-dependent (needs litellm), which is acceptable for models
       canonical could not price at all.
    3. **inf** — if BOTH return ``None`` the model is unknown to every source and
       FAILS the constraint (returns infinity) to prevent silent passing.

    The constraint validates that this estimated cost per 1k-token budget is
    within the specified limit.
    """

    def calculate_cost(config: dict[str, Any]) -> float:
        """Calculate cost based on model and token usage.

        Canonical-first, litellm-fallback, then float('inf') for models unknown
        to both sources (so they fail the constraint).
        """
        model = config.get("model", DEFAULT_MODEL)
        max_tokens = config.get("max_tokens", 150)

        # 1. Canonical first: deterministic for the curated table + aliases.
        result = _try_fallback_pricing(model, max_tokens)
        if result is not None:
            return result

        # 2. litellm fallback: coverage for models outside the curated table.
        result = _try_litellm_pricing(model, max_tokens)
        if result is not None:
            return result

        # 3. Unknown to both -> fail the constraint.
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
