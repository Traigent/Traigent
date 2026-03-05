"""Load and normalize TVL specifications for the Traigent SDK."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-TVLSPEC REQ-TVLSPEC-012 SYNC-OptimizationFlow

from __future__ import annotations

import ast
import copy
import math
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.utils.exceptions import TVLValidationError
from traigent.utils.logging import get_logger

from .models import (
    ConvergenceCriteria,
    DerivedConstraint,
    DomainSpec,
    EnvironmentSnapshot,
    EvaluationSet,
    ExplorationBudgets,
    PromotionPolicy,
    RegistryResolver,
    StructuralConstraint,
    TVarDecl,
    TVarType,
    TVLHeader,
    normalize_tvar_type,
    parse_domain_spec,
)

logger = get_logger(__name__)

# Constant for metrics reference detection in constraint expressions
_METRICS_PREFIX = "metrics."


@dataclass(slots=True)
class TVLBudget:
    """Structured representation of the optimization budget in a TVL spec."""

    max_trials: int | None = None
    parallel_trials: int | None = None
    timeout_seconds: float | None = None
    max_total_examples: int | None = None
    samples_include_pruned: bool | None = None


@dataclass(slots=True)
class CompiledConstraint:
    """Internal representation of a TVL constraint."""

    identifier: str
    description: str
    requires_metrics: bool
    evaluator: Callable[[dict[str, Any], dict[str, Any] | None], bool]
    constraint_type: str

    def to_callable(self) -> Callable[[dict[str, Any], dict[str, Any] | None], bool]:
        """Expose a decorator-compatible callable with metadata attached."""

        def _constraint(
            config: dict[str, Any], metrics: dict[str, Any] | None = None
        ) -> bool:
            return self.evaluator(config, metrics)

        _constraint.__name__ = f"tvl_constraint_{self.identifier}"
        _constraint.__tvl_constraint__ = {  # type: ignore[attr-defined]
            "id": self.identifier,
            "message": self.description,
            "requires_metrics": self.requires_metrics,
            "type": self.constraint_type,
        }
        return _constraint


@dataclass(slots=True)
class TVLSpecArtifact:
    """Normalized view of a TVL spec ready for the decorator/runtime.

    TVL 0.9 compatible artifact containing all parsed spec components.
    """

    path: Path
    environment: str | None
    configuration_space: dict[str, Any]
    objective_schema: ObjectiveSchema | None
    constraints: list[Callable[[dict[str, Any], dict[str, Any] | None], bool]]
    default_config: dict[str, Any]
    metadata: dict[str, Any]
    budget: TVLBudget
    algorithm: str | None
    promotion_policy: PromotionPolicy | None = None
    tvars: list[TVarDecl] | None = None
    derived_constraints: list[DerivedConstraint] | None = None
    # TVL 0.9 additions
    tvl_header: TVLHeader | None = None
    environment_snapshot: EnvironmentSnapshot | None = None
    evaluation_set: EvaluationSet | None = None
    tvl_version: str | None = None
    convergence: ConvergenceCriteria | None = None
    exploration_budgets: ExplorationBudgets | None = None
    exploration_parallelism: int | None = None
    # Multi-agent support: parameter name -> agent ID mappings
    parameter_agents: dict[str, str] | None = None

    def _apply_legacy_budget_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply legacy budget fields to overrides."""
        if self.algorithm:
            overrides.setdefault("algorithm", self.algorithm)
        if self.budget.max_trials is not None:
            overrides.setdefault("max_trials", self.budget.max_trials)
        if self.budget.timeout_seconds is not None:
            overrides.setdefault("timeout", self.budget.timeout_seconds)
        if self.budget.parallel_trials is not None:
            overrides.setdefault("parallel_trials", self.budget.parallel_trials)
        if self.budget.max_total_examples is not None:
            overrides.setdefault("max_total_examples", self.budget.max_total_examples)
        if self.budget.samples_include_pruned is not None:
            overrides.setdefault(
                "samples_include_pruned", self.budget.samples_include_pruned
            )

    def _apply_exploration_budget_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply TVL 0.9 exploration_budgets to overrides."""
        if self.exploration_budgets is None:
            return
        if self.exploration_budgets.max_trials is not None:
            overrides.setdefault("max_trials", self.exploration_budgets.max_trials)
        if self.exploration_budgets.max_spend_usd is not None:
            overrides.setdefault("cost_limit", self.exploration_budgets.max_spend_usd)
        if self.exploration_budgets.max_wallclock_s is not None:
            overrides.setdefault("timeout", self.exploration_budgets.max_wallclock_s)

    def _convert_parallel_trials_to_config(self, overrides: dict[str, Any]) -> None:
        """Convert parallel_trials to parallel_config structure."""
        if "parallel_trials" not in overrides:
            return
        parallel_trials_value = overrides.get("parallel_trials")
        if "parallel_config" not in overrides:
            if isinstance(parallel_trials_value, int) and parallel_trials_value > 0:
                from traigent.config.parallel import ParallelConfig

                overrides["parallel_config"] = ParallelConfig.from_legacy(
                    parallel_trials=parallel_trials_value
                )
        overrides.pop("parallel_trials", None)

    def runtime_overrides(self) -> dict[str, Any]:
        """Return decorator/runtime overrides derived from the spec."""
        overrides: dict[str, Any] = {}

        # Legacy budget fields first (they take precedence if set)
        self._apply_legacy_budget_overrides(overrides)

        # TVL 0.9 exploration_budgets (only set if not already set by legacy)
        self._apply_exploration_budget_overrides(overrides)

        # TVL 0.9 exploration parallelism
        if self.exploration_parallelism is not None:
            overrides.setdefault("parallel_trials", self.exploration_parallelism)

        # TVL 0.9 convergence criteria
        if self.convergence is not None:
            overrides["convergence_metric"] = self.convergence.metric
            overrides["convergence_window"] = self.convergence.window
            overrides["convergence_threshold"] = self.convergence.threshold

        # TVL 0.9 promotion policy: tie_breakers
        if self.promotion_policy is not None and self.promotion_policy.tie_breakers:
            overrides["tie_breakers"] = self.promotion_policy.tie_breakers

        # Multi-agent: pass TVL parameter_agents for agent configuration
        if self.parameter_agents:
            overrides["tvl_parameter_agents"] = self.parameter_agents

        # Convert parallel_trials to unified parallel_config structure
        self._convert_parallel_trials_to_config(overrides)

        return overrides


def _load_and_parse_yaml(path: Path) -> dict:
    """Load and parse YAML file, returning the raw dictionary."""
    if not path.is_file():
        raise TVLValidationError(f"TVL spec not found: {path}")

    try:
        raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive guard
        raise TVLValidationError(f"Failed to parse TVL spec: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise TVLValidationError("TVL specification must be a mapping")
    return raw_data


# ===== T-5: Early Schema Validation =====

# Known TVL 0.9 section names
_KNOWN_TVL_SECTIONS = frozenset(
    {
        "tvl",
        "tvl_version",
        "tvars",
        "exploration",
        "objectives",
        "constraints",
        "defaults",
        "metadata",
        "environment",  # Environment snapshot (single env)
        "environments",  # Multi-environment definitions
        "env_snapshot",
        "evaluation_set",
        "promotion_policy",
        "extends",  # Spec-level inheritance
        "safety_constraints",  # Safety constraint definitions
        "datasets",  # Dataset references for evaluation
        "evaluators",  # Evaluator definitions
        # Legacy sections (deprecated but still valid)
        "configuration_space",
        "optimization",
    }
)

# Required types for top-level sections
_SECTION_TYPES: dict[str, type | tuple[type, ...]] = {
    "tvl": dict,
    "tvl_version": (str, int, float),
    "tvars": (list, dict),  # TVL 0.9 array or legacy dict format
    "exploration": dict,
    "objectives": (list, dict),
    "constraints": (list, dict),
    "defaults": dict,
    "metadata": dict,
    "environment": dict,  # Environment snapshot
    "environments": dict,  # Multi-environment definitions
    "env_snapshot": dict,
    "evaluation_set": dict,
    "promotion_policy": dict,
    "extends": str,  # Path to base spec (spec-level inheritance)
    "safety_constraints": list,  # Safety constraint definitions
    "datasets": dict,  # Dataset references for evaluation
    "evaluators": dict,  # Evaluator definitions
    "configuration_space": dict,
    "optimization": dict,
}


def validate_tvl_schema(
    data: dict[str, Any],
    *,
    strict: bool = False,
) -> list[str]:
    """Validate TVL spec structure before detailed parsing (T-5).

    Performs early validation to catch obvious errors like invalid section
    names, missing required fields, incorrect types, etc. This provides
    clearer error messages than later parsing failures.

    Args:
        data: Parsed YAML data (raw dictionary).
        strict: If True, raise TVLValidationError on any issue.
                If False, return list of warnings but don't raise.

    Returns:
        List of validation warnings/errors found.

    Raises:
        TVLValidationError: If strict=True and validation fails.

    Example::

        data = yaml.safe_load(path.read_text())
        issues = validate_tvl_schema(data)
        if issues:
            print("Warnings:", issues)
    """
    issues: list[str] = []

    # Check for unknown top-level sections
    unknown_sections = set(data.keys()) - _KNOWN_TVL_SECTIONS
    if unknown_sections:
        issues.append(
            f"Unknown top-level sections: {sorted(unknown_sections)}. "
            f"Valid sections: {sorted(_KNOWN_TVL_SECTIONS)}"
        )

    # Check types of known sections
    for section, expected_type in _SECTION_TYPES.items():
        if section in data:
            value = data[section]
            if not isinstance(value, expected_type):
                type_names = (
                    expected_type.__name__
                    if isinstance(expected_type, type)
                    else " or ".join(t.__name__ for t in expected_type)
                )
                issues.append(
                    f"Section '{section}' should be {type_names}, "
                    f"got {type(value).__name__}"
                )

    # Validate tvars structure if present
    if "tvars" in data:
        tvars = data["tvars"]
        if isinstance(tvars, dict):
            issues.extend(_validate_tvars_structure(tvars))
        elif isinstance(tvars, list):
            issues.extend(_validate_tvars_list_structure(tvars))

    # Validate objectives structure if present
    if "objectives" in data:
        objectives = data["objectives"]
        issues.extend(_validate_objectives_structure(objectives))

    # Validate constraints structure if present
    if "constraints" in data:
        constraints = data["constraints"]
        issues.extend(_validate_constraints_structure(constraints))

    # Validate exploration section if present
    if "exploration" in data:
        exploration = data["exploration"]
        if isinstance(exploration, dict):
            issues.extend(_validate_exploration_structure(exploration))

    if strict and issues:
        raise TVLValidationError(
            "TVL schema validation failed:\n  - " + "\n  - ".join(issues)
        )

    return issues


def _validate_tvars_structure(tvars: dict[str, Any]) -> list[str]:
    """Validate structure of tvars section (legacy dict format)."""
    issues: list[str] = []

    for name, tvar_def in tvars.items():
        if not isinstance(name, str):
            issues.append(f"TVAR name must be string, got {type(name).__name__}")
            continue

        if not isinstance(tvar_def, dict):
            issues.append(f"TVAR '{name}' definition must be a dict")
            continue

        # Check for required fields
        if "domain" not in tvar_def:
            issues.append(f"TVAR '{name}' missing required 'domain' field")

        # Validate domain structure
        domain = tvar_def.get("domain")
        if domain is not None:
            if isinstance(domain, dict):
                if "type" in domain:
                    valid_types = {"range", "choices", "registry"}
                    if domain["type"] not in valid_types:
                        issues.append(
                            f"TVAR '{name}' has invalid domain type '{domain['type']}'. "
                            f"Valid: {valid_types}"
                        )
            elif not isinstance(domain, (list, tuple)):
                issues.append(
                    f"TVAR '{name}' domain should be dict, list, or tuple, "
                    f"got {type(domain).__name__}"
                )

    return issues


def _validate_tvars_list_structure(tvars: list[Any]) -> list[str]:
    """Validate structure of tvars section (TVL 0.9 array format)."""
    issues: list[str] = []

    for idx, tvar_def in enumerate(tvars):
        if not isinstance(tvar_def, dict):
            issues.append(
                f"TVAR at index {idx} must be a dict, got {type(tvar_def).__name__}"
            )
            continue

        # Check for required 'name' field
        name = tvar_def.get("name")
        if not isinstance(name, str) or not name:
            issues.append(f"TVAR at index {idx} missing required 'name' string")
            name = f"<index {idx}>"  # Use placeholder for further validation

        # Check for required 'type' field
        tvar_type = tvar_def.get("type")
        if not isinstance(tvar_type, str):
            issues.append(f"TVAR '{name}' missing required 'type' string")
        else:
            valid_types = {"float", "int", "str", "bool", "categorical", "registry"}
            if tvar_type.lower() not in valid_types:
                issues.append(
                    f"TVAR '{name}' has invalid type '{tvar_type}'. "
                    f"Valid: {valid_types}"
                )

        # Validate domain structure if present
        domain = tvar_def.get("domain")
        if domain is not None:
            if isinstance(domain, dict):
                if "type" in domain:
                    valid_domain_types = {"range", "choices", "registry"}
                    if domain["type"] not in valid_domain_types:
                        issues.append(
                            f"TVAR '{name}' has invalid domain type '{domain['type']}'. "
                            f"Valid: {valid_domain_types}"
                        )
            elif not isinstance(domain, (list, tuple)):
                issues.append(
                    f"TVAR '{name}' domain should be dict, list, or tuple, "
                    f"got {type(domain).__name__}"
                )

    return issues


def _validate_objectives_structure(objectives: Any) -> list[str]:
    """Validate structure of objectives section."""
    issues: list[str] = []

    if isinstance(objectives, list):
        for i, obj in enumerate(objectives):
            if isinstance(obj, dict):
                if "name" not in obj:
                    issues.append(f"Objective {i} missing 'name' field")
                if "direction" in obj and obj["direction"] not in {
                    "maximize",
                    "minimize",
                }:
                    issues.append(
                        f"Objective {i} has invalid direction '{obj['direction']}'"
                    )
            elif not isinstance(obj, str):
                issues.append(
                    f"Objective {i} should be string or dict, got {type(obj).__name__}"
                )
    elif isinstance(objectives, dict):
        for name, obj_def in objectives.items():
            if isinstance(obj_def, dict) and "direction" in obj_def:
                if obj_def["direction"] not in {"maximize", "minimize"}:
                    issues.append(
                        f"Objective '{name}' has invalid direction '{obj_def['direction']}'"
                    )

    return issues


def _validate_constraints_structure(constraints: Any) -> list[str]:
    """Validate structure of constraints section."""
    issues: list[str] = []

    if isinstance(constraints, list):
        for i, constraint in enumerate(constraints):
            if isinstance(constraint, dict):
                if "type" in constraint:
                    valid_types = {"structural", "derived"}
                    if constraint["type"] not in valid_types:
                        issues.append(
                            f"Constraint {i} has invalid type '{constraint['type']}'"
                        )
                if (
                    constraint.get("type") == "structural"
                    and "require" not in constraint
                ):
                    issues.append(f"Structural constraint {i} missing 'require' field")
                if constraint.get("type") == "derived" and "require" not in constraint:
                    issues.append(f"Derived constraint {i} missing 'require' field")
            elif not isinstance(constraint, str):
                issues.append(
                    f"Constraint {i} should be string or dict, got {type(constraint).__name__}"
                )
    elif isinstance(constraints, dict):
        if "structural" in constraints and not isinstance(
            constraints["structural"], list
        ):
            issues.append("constraints.structural should be a list")
        if "derived" in constraints and not isinstance(constraints["derived"], list):
            issues.append("constraints.derived should be a list")

    return issues


def _validate_exploration_structure(exploration: dict[str, Any]) -> list[str]:
    """Validate structure of exploration section."""
    issues: list[str] = []

    # Known exploration fields
    known_fields = {
        "budget",
        "algorithm",
        "convergence",
        "budgets",
        "parallelism",
        "max_trials",
        "timeout",
        "parallel_trials",
    }
    unknown = set(exploration.keys()) - known_fields
    if unknown:
        # Just a warning, not critical
        pass  # Allow unknown fields for extensibility

    # Validate budget if present
    if "budget" in exploration:
        budget = exploration["budget"]
        if isinstance(budget, dict):
            if "max_trials" in budget and not isinstance(budget["max_trials"], int):
                issues.append("exploration.budget.max_trials should be integer")

    # Validate convergence if present
    if "convergence" in exploration:
        convergence = exploration["convergence"]
        if (
            isinstance(convergence, dict)
            and "threshold" in convergence
            and not isinstance(convergence["threshold"], (int, float))
        ):
            issues.append("exploration.convergence.threshold should be numeric")

    return issues


def _parse_config_space_format(
    resolved: dict,
    registry_resolver: RegistryResolver | None,
) -> tuple[list | None, dict, dict, dict]:
    """Parse configuration space in either TVL 0.9 (tvars) or legacy format."""
    has_tvars = "tvars" in resolved
    has_config_space = "configuration_space" in resolved

    if has_tvars and has_config_space:
        raise TVLValidationError(
            "TVL spec contains both 'tvars' (TVL 0.9) and 'configuration_space' "
            "(legacy). Please remove 'configuration_space' to use the new format."
        )

    if has_tvars:
        tvars, config_space, defaults, units = _parse_tvars(
            resolved, registry_resolver=registry_resolver
        )
        return tvars, config_space, defaults, units

    if has_config_space:
        warnings.warn(
            "TVL spec uses legacy 'configuration_space' format. "
            "This format is deprecated in TVL 0.9; migrate to 'tvars'.",
            DeprecationWarning,
            stacklevel=3,
        )
        config_space, defaults, units = _parse_configuration_space(resolved)
        return None, config_space, defaults, units

    # Empty spec - no configuration space defined
    return None, {}, {}, {}


def _parse_exploration_section(
    resolved: dict,
) -> tuple[
    TVLBudget,
    str | None,
    ConvergenceCriteria | None,
    ExplorationBudgets | None,
    int | None,
]:
    """Parse exploration/optimization section, handling legacy format."""
    has_exploration = "exploration" in resolved
    has_optimization = "optimization" in resolved

    if has_exploration and has_optimization:
        raise TVLValidationError(
            "TVL spec contains both 'exploration' and 'optimization' sections. "
            "Use only 'exploration' (TVL 0.9) or 'optimization' (legacy), not both."
        )

    if has_optimization and not has_exploration:
        warnings.warn(
            "TVL spec uses legacy 'optimization' section. "
            "This format is deprecated in TVL 0.9; migrate to 'exploration'.",
            DeprecationWarning,
            stacklevel=3,
        )

    exploration_section = resolved.get("exploration") or resolved.get("optimization")
    budget = _parse_budget(exploration_section)
    algorithm = _resolve_algorithm(exploration_section)
    convergence = _parse_convergence(exploration_section)
    exploration_budgets = _parse_exploration_budgets(exploration_section)
    exploration_parallelism = _parse_exploration_parallelism(exploration_section)
    return budget, algorithm, convergence, exploration_budgets, exploration_parallelism


def load_tvl_spec(
    *,
    spec_path: str | Path,
    environment: str | None = None,
    validate_constraints: bool = True,
    validator: str = "python",
    validate_schema: bool = True,
    registry_resolver: RegistryResolver | None = None,
) -> TVLSpecArtifact:
    """Load and normalize a TVL specification from disk.

    Supports both legacy format (configuration_space) and TVL 0.9 format (tvars).
    Auto-detects format based on presence of 'tvars' vs 'configuration_space'.

    Args:
        spec_path: Path to the TVL spec file.
        environment: Optional environment overlay key to apply.
        validate_constraints: Whether to compile and validate structural constraints.
        validator: Constraint validator plugin name (``traigent.validators`` entry point).
        validate_schema: Whether to perform early schema validation (T-5).
            When True, validates the TVL spec structure before detailed parsing
            to catch errors early with clearer error messages.
        registry_resolver: Optional resolver used to materialize registry domains
            into concrete configuration values. If a spec contains a registry
            domain and no resolver is provided, spec loading fails fast.
    """
    path = Path(spec_path).expanduser()
    raw_data = _load_and_parse_yaml(path)

    # Resolve spec-level inheritance (extends) before any other processing
    raw_data = _resolve_spec_extends(raw_data, path)

    # T-5: Early schema validation before detailed parsing
    if validate_schema:
        schema_issues = validate_tvl_schema(raw_data, strict=False)
        for issue in schema_issues:
            logger.warning("TVL schema issue in %s: %s", path.name, issue)

    resolved = _apply_environment(raw_data, environment)

    # Parse TVL 0.9 header (tvl section)
    tvl_header = _parse_tvl_header(resolved.get("tvl"))
    tvl_version = resolved.get("tvl_version")
    if isinstance(tvl_version, (int, float)):
        tvl_version = str(tvl_version)

    # Parse TVL 0.9 environment snapshot and evaluation set
    environment_snapshot = _parse_environment_snapshot(resolved.get("environment"))
    evaluation_set = _parse_evaluation_set(resolved.get("evaluation_set"))

    # Parse configuration space (tvars or legacy)
    tvars, config_space, defaults, units = _parse_config_space_format(
        resolved, registry_resolver
    )

    # Extract multi-agent parameter mappings from tvars
    parameter_agents = _extract_parameter_agents_from_tvars(tvars)

    objective_schema = _parse_objectives(resolved)

    # Warn about legacy constraints format
    constraints_section = resolved.get("constraints")
    if isinstance(constraints_section, list) and constraints_section:
        warnings.warn(
            "TVL spec uses legacy 'constraints' list format. "
            "This format is deprecated in TVL 0.9; use 'constraints: {structural: [...], derived: [...]}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    _validate_structural_constraints_if_enabled(
        constraints_section=constraints_section,
        configuration_space=config_space,
        validate_constraints=validate_constraints,
        validator_name=validator,
    )

    # Parse constraints - support both legacy and TVL 0.9 format
    compiled_constraints, derived_constraints = _compile_constraints_unified(
        resolved.get("constraints", []) or [], path
    )
    constraint_wrappers = [
        constraint.to_callable() for constraint in compiled_constraints
    ]

    # Compile derived constraints and add to constraint_wrappers for runtime evaluation
    if derived_constraints:
        for dc in derived_constraints:
            label = f"derived_constraint_{dc.index}"
            try:
                compiled_derived = compile_constraint_expression(
                    dc.require, label=label
                )
                constraint_wrappers.append(compiled_derived)
            except TVLValidationError as exc:
                # Re-raise with more context about derived constraint
                raise TVLValidationError(
                    f"Invalid derived constraint at index {dc.index}: "
                    f"expression '{dc.require}' is not valid"
                ) from exc

    # Parse exploration section
    budget, algorithm, convergence, exploration_budgets, exploration_parallelism = (
        _parse_exploration_section(resolved)
    )

    # Parse promotion policy (TVL 0.9)
    promotion_policy = _parse_promotion_policy(resolved.get("promotion_policy"))

    metadata = _build_metadata(
        resolved,
        path,
        environment,
        units,
        compiled_constraints,
    )

    return TVLSpecArtifact(
        path=path,
        environment=environment,
        configuration_space=config_space,
        objective_schema=objective_schema,
        constraints=constraint_wrappers,
        default_config=defaults,
        metadata=metadata,
        budget=budget,
        algorithm=algorithm,
        promotion_policy=promotion_policy,
        tvars=tvars,
        derived_constraints=derived_constraints,
        tvl_header=tvl_header,
        environment_snapshot=environment_snapshot,
        evaluation_set=evaluation_set,
        tvl_version=tvl_version,
        convergence=convergence,
        exploration_budgets=exploration_budgets,
        exploration_parallelism=exploration_parallelism,
        parameter_agents=parameter_agents,
    )


def compile_constraint_expression(
    expression: str, *, label: str
) -> Callable[[dict[str, Any], dict[str, Any] | None], bool]:
    """Compile a CEL-like constraint into a safe Python callable.

    The expression language supports a subset of Python syntax for safety:
    - Identifiers: `params.x`, `metrics.accuracy`
    - Literals: Numbers, strings, booleans (`True`, `False`)
    - Operators: `+`, `-`, `*`, `/`, `==`, `!=`, `<`, `<=`, `>`, `>=`
    - Logical: `and`, `or`, `not`
    - Functions: `min`, `max`, `abs`, `len`, `sum`, `any`, `all`

    Note on Equality:
    Constraint expressions are parsed as Python expressions, so equality must
    be written as `==` (not `=`).

    Args:
        expression: The constraint expression string.
        label: Label for the compiled code object (for tracebacks).

    Returns:
        A callable that takes (config, metrics) and returns a boolean.

    Raises:
        TVLValidationError: If the expression is invalid or unsafe.
    """

    translated = _translate_expression(expression)
    try:
        parsed = ast.parse(translated, mode="eval")
    except SyntaxError as exc:
        raise TVLValidationError(
            f"Invalid constraint expression '{expression}': {exc}"
        ) from exc
    _validate_expression_ast(parsed, source=expression)
    code = compile(parsed, filename=label, mode="eval")

    def _evaluate(
        config: dict[str, Any], metrics: dict[str, Any] | None = None
    ) -> bool:
        context = {
            "params": _AttributeView(config or {}),
            "metrics": _AttributeView(metrics or {}),
            "math": math,
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "any": any,
            "all": all,
        }
        try:
            # SECURITY: This eval() is safe because:
            # 1. Expression is parsed and validated by _validate_expression_ast()
            # 2. Only whitelisted AST nodes are allowed (no imports, lambdas)
            # 3. __builtins__ is empty, removing access to dangerous functions
            # 4. Context only exposes params, metrics, and safe math functions
            result = eval(code, {"__builtins__": {}}, context)  # nosec B307
        except Exception as exc:  # pragma: no cover - defensive guard
            raise TVLValidationError(
                f"Failed to evaluate constraint '{expression}': {exc}"
            ) from exc
        if not isinstance(result, bool):
            raise TVLValidationError(
                f"Constraint '{expression}' must evaluate to a boolean"
            )
        return bool(result)

    return _evaluate


def _parse_configuration_space(
    resolved: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    space_section = resolved.get("configuration_space")
    if not isinstance(space_section, dict) or not space_section:
        raise TVLValidationError("configuration_space must be a non-empty mapping")

    configuration_space: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    units: dict[str, str] = {}

    for name, definition in space_section.items():
        if not isinstance(name, str):
            raise TVLValidationError("Configuration parameter names must be strings")
        if not isinstance(definition, dict):
            raise TVLValidationError(f"Parameter '{name}' must be defined as a mapping")
        parameter_type = (definition.get("type") or "").lower()
        if not parameter_type:
            if "values" in definition:
                parameter_type = "categorical"
            elif "range" in definition:
                parameter_type = "continuous"
            else:
                raise TVLValidationError(
                    f"Parameter '{name}' must define 'type', 'values', or 'range'"
                )

        units_value = definition.get("unit")
        if isinstance(units_value, str):
            units[name] = units_value

        configuration_space[name] = _normalize_parameter(
            name, parameter_type, definition
        )

        if "default" in definition:
            defaults[name] = definition["default"]

    return configuration_space, defaults, units


def _normalize_parameter(
    name: str, parameter_type: str, definition: dict[str, Any]
) -> list[Any] | tuple[Any, Any]:
    if parameter_type in {"categorical", "discrete"}:
        values = definition.get("values")
        if not isinstance(values, list) or not values:
            raise TVLValidationError(
                f"Parameter '{name}' must define a non-empty 'values' list"
            )
        return values

    if parameter_type in {"continuous", "float"}:
        lower, upper = _resolve_range(definition.get("range"), name)
        return float(lower), float(upper)

    if parameter_type == "integer":
        lower, upper = _resolve_range(definition.get("range"), name)
        return int(lower), int(upper)

    if parameter_type == "boolean":
        return [True, False]

    raise TVLValidationError(
        f"Unsupported parameter type '{parameter_type}' for '{name}'"
    )


def _resolve_registry_domain(
    name: str,
    domain: DomainSpec,
    registry_resolver: RegistryResolver | None,
) -> list[Any]:
    """Resolve a registry domain to concrete values."""
    if registry_resolver is None:
        raise TVLValidationError(
            f"TVAR '{name}' uses a registry domain ('{domain.registry}') "
            "but no registry_resolver was provided."
        )
    if domain.registry is None:
        raise TVLValidationError(
            f"TVAR '{name}' has a registry domain but registry field is None"
        )
    resolved_values = registry_resolver.resolve(
        domain.registry, filter_expr=domain.filter, version=domain.version
    )
    if not isinstance(resolved_values, list):
        raise TVLValidationError(
            f"RegistryResolver.resolve() must return a list; got {type(resolved_values)}"
        )
    if not resolved_values:
        raise TVLValidationError(
            f"Registry domain '{domain.registry}' for TVAR '{name}' resolved to no values"
        )
    return resolved_values


def _parse_single_tvar(
    idx: int, decl: dict[str, Any]
) -> tuple[str, TVarType, str, DomainSpec, Any, str | None, str | None]:
    """Parse a single TVAR declaration and validate its fields.

    Returns:
        Tuple of (name, tvar_type, raw_type, domain, default, unit, agent).
    """
    if not isinstance(decl, dict):
        raise TVLValidationError(f"TVAR declaration at index {idx} must be a mapping")

    name = decl.get("name")
    if not isinstance(name, str) or not name:
        raise TVLValidationError(f"TVAR at index {idx} requires a 'name' string")

    raw_type = decl.get("type")
    if not isinstance(raw_type, str):
        raise TVLValidationError(f"TVAR '{name}' requires a 'type' string")

    tvar_type = normalize_tvar_type(raw_type)
    if tvar_type is None:
        raise TVLValidationError(f"TVAR '{name}' has unsupported type '{raw_type}'")

    domain_data = decl.get("domain")
    try:
        domain = parse_domain_spec(name, tvar_type, domain_data)
    except ValueError as exc:
        raise TVLValidationError(str(exc)) from exc

    unit = decl.get("unit") if isinstance(decl.get("unit"), str) else None
    default = decl.get("default")
    agent = decl.get("agent") if isinstance(decl.get("agent"), str) else None

    return name, tvar_type, raw_type, domain, default, unit, agent


def _parse_tvars(
    resolved: dict[str, Any],
    *,
    registry_resolver: RegistryResolver | None = None,
) -> tuple[list[TVarDecl], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Parse TVL 0.9 tvars array format.

    Args:
        resolved: The resolved TVL spec dictionary.
        registry_resolver: Optional resolver used to materialize registry domains.

    Returns:
        Tuple of (tvars, configuration_space, defaults, units).
    """
    tvars_section = resolved.get("tvars")
    if not isinstance(tvars_section, list) or not tvars_section:
        raise TVLValidationError("tvars must be a non-empty array")

    tvars: list[TVarDecl] = []
    configuration_space: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    units: dict[str, str] = {}

    for idx, decl in enumerate(tvars_section):
        name, tvar_type, raw_type, domain, default, unit, agent = _parse_single_tvar(
            idx, decl
        )

        if unit:
            units[name] = unit

        tvar = TVarDecl(
            name=name,
            type=tvar_type,
            raw_type=raw_type,
            domain=domain,
            default=default,
            unit=unit,
            agent=agent,
        )
        tvars.append(tvar)

        # Convert to configuration space format
        if domain.kind == "registry":
            configuration_space[name] = _resolve_registry_domain(
                name, domain, registry_resolver
            )
        else:
            configuration_space[name] = domain.to_configuration_space_entry()

        if default is not None:
            defaults[name] = default

    return tvars, configuration_space, defaults, units


def _extract_parameter_agents_from_tvars(
    tvars: list[TVarDecl] | None,
) -> dict[str, str] | None:
    """Extract parameter-to-agent mappings from TVarDecl objects.

    Args:
        tvars: List of parsed TVarDecl objects from TVL spec.

    Returns:
        Dictionary mapping parameter names to agent IDs for parameters
        that have explicit agent assignments, or None if no agents specified.
    """
    if not tvars:
        return None

    result: dict[str, str] = {}
    for tvar in tvars:
        if tvar.agent:
            result[tvar.name] = tvar.agent

    return result if result else None


def _parse_promotion_policy(policy_data: Any) -> PromotionPolicy | None:
    """Parse TVL 0.9 promotion_policy section.

    Args:
        policy_data: The raw promotion_policy dictionary.

    Returns:
        PromotionPolicy if present, None otherwise.
    """
    if policy_data is None:
        return None

    if not isinstance(policy_data, dict):
        raise TVLValidationError("promotion_policy must be a mapping")

    try:
        return PromotionPolicy.from_dict(policy_data)
    except (ValueError, KeyError) as exc:
        raise TVLValidationError(f"Invalid promotion_policy: {exc}") from exc


def _parse_tvl_header(header_data: Any) -> TVLHeader | None:
    """Parse TVL module header.

    Args:
        header_data: The raw tvl header dictionary.

    Returns:
        TVLHeader if present, None otherwise.
    """
    if header_data is None:
        return None

    if not isinstance(header_data, dict):
        raise TVLValidationError("tvl header must be a mapping")

    try:
        return TVLHeader.from_dict(header_data)
    except ValueError as exc:
        raise TVLValidationError(f"Invalid tvl header: {exc}") from exc


def _parse_environment_snapshot(env_data: Any) -> EnvironmentSnapshot | None:
    """Parse TVL 0.9 environment section.

    Args:
        env_data: The raw environment dictionary.

    Returns:
        EnvironmentSnapshot if present, None otherwise.
    """
    if env_data is None:
        return None

    if not isinstance(env_data, dict):
        raise TVLValidationError("environment must be a mapping")

    try:
        return EnvironmentSnapshot.from_dict(env_data)
    except ValueError as exc:
        raise TVLValidationError(f"Invalid environment: {exc}") from exc


def _parse_evaluation_set(eval_data: Any) -> EvaluationSet | None:
    """Parse TVL 0.9 evaluation_set section.

    Args:
        eval_data: The raw evaluation_set dictionary.

    Returns:
        EvaluationSet if present, None otherwise.
    """
    if eval_data is None:
        return None

    if not isinstance(eval_data, dict):
        raise TVLValidationError("evaluation_set must be a mapping")

    try:
        return EvaluationSet.from_dict(eval_data)
    except ValueError as exc:
        raise TVLValidationError(f"Invalid evaluation_set: {exc}") from exc


def _parse_convergence(exploration_data: Any) -> ConvergenceCriteria | None:
    """Parse convergence criteria from exploration section.

    Args:
        exploration_data: The raw exploration dictionary.

    Returns:
        ConvergenceCriteria if present, None otherwise.
    """
    if not isinstance(exploration_data, dict):
        return None

    convergence_data = exploration_data.get("convergence")
    if convergence_data is None:
        return None

    if not isinstance(convergence_data, dict):
        raise TVLValidationError("convergence must be a mapping")

    try:
        return ConvergenceCriteria.from_dict(convergence_data)
    except ValueError as exc:
        raise TVLValidationError(f"Invalid convergence: {exc}") from exc


def _parse_exploration_budgets(exploration_data: Any) -> ExplorationBudgets | None:
    """Parse exploration budgets from exploration section.

    Args:
        exploration_data: The raw exploration dictionary.

    Returns:
        ExplorationBudgets if present, None otherwise.
    """
    if not isinstance(exploration_data, dict):
        return None

    budgets_data = exploration_data.get("budgets")
    if budgets_data is None:
        return None

    if not isinstance(budgets_data, dict):
        raise TVLValidationError("budgets must be a mapping")

    try:
        return ExplorationBudgets.from_dict(budgets_data)
    except ValueError as exc:
        raise TVLValidationError(f"Invalid budgets: {exc}") from exc


def _parse_exploration_parallelism(exploration_data: Any) -> int | None:
    """Parse parallelism config from exploration section.

    Args:
        exploration_data: The raw exploration dictionary.

    Returns:
        max_parallel_trials if present, None otherwise.
    """
    if not isinstance(exploration_data, dict):
        return None

    parallelism = exploration_data.get("parallelism")
    if parallelism is None:
        return None

    if isinstance(parallelism, dict):
        max_parallel = parallelism.get("max_parallel_trials")
        if isinstance(max_parallel, int):
            return max_parallel
    return None


def _extract_structural_constraints_for_validation(
    constraints_section: Any,
) -> list[StructuralConstraint]:
    """Normalize constraints into StructuralConstraint objects for plugin validation."""
    if isinstance(constraints_section, dict):
        entries = constraints_section.get("structural", [])
    elif isinstance(constraints_section, list):
        entries = constraints_section
    else:
        return []

    if not isinstance(entries, list):
        return []

    normalized: list[StructuralConstraint] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue

        # TVL 0.9 structural format already aligns with StructuralConstraint.
        if any(key in entry for key in ("expr", "when", "then")):
            try:
                normalized.append(StructuralConstraint.from_dict(entry, index))
            except ValueError:
                # Let the main compile path emit the canonical validation error.
                continue
            continue

        # Legacy list format maps ``rule``/``type`` into expression or implication.
        constraint_type = (entry.get("type") or "expression").lower()
        if constraint_type == "conditional":
            when_expr = entry.get("when")
            then_expr = entry.get("then")
            if isinstance(when_expr, str) and isinstance(then_expr, str):
                normalized.append(
                    StructuralConstraint(when=when_expr, then=then_expr, index=index)
                )
            continue

        rule_expr = entry.get("rule")
        if isinstance(rule_expr, str):
            normalized.append(StructuralConstraint(expr=rule_expr, index=index))

    return normalized


def _validate_structural_constraints_if_enabled(
    *,
    constraints_section: Any,
    configuration_space: dict[str, Any],
    validate_constraints: bool,
    validator_name: str,
) -> None:
    """Validate structural constraints through optional validator plugins."""
    if not validate_constraints:
        return

    structural_constraints = _extract_structural_constraints_for_validation(
        constraints_section
    )
    if not structural_constraints:
        return

    try:
        from traigent_validation.plugins import get_validator
    except ImportError:
        logger.warning(
            "Constraint validation package is unavailable; skipping structural validation."
        )
        return

    validator = get_validator(validator_name)
    if validator is None:
        logger.warning(
            "No constraint validator named '%s' is registered; skipping structural validation.",
            validator_name,
        )
        return

    validate_fn = getattr(validator, "validate_structural_constraints", None)
    if not callable(validate_fn):
        logger.warning(
            "Validator '%s' does not implement validate_structural_constraints(); skipping.",
            validator_name,
        )
        return

    available_parameters = set(configuration_space.keys())
    try:
        issues = validate_fn(structural_constraints, available_parameters)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise TVLValidationError(
            f"Constraint validator '{validator_name}' failed: {exc}"
        ) from exc

    if issues:
        rendered = "\n".join(f"- {issue}" for issue in issues)
        raise TVLValidationError(
            "Structural constraint validation failed:\n"
            f"{rendered}\n"
            f"Available parameters: {sorted(available_parameters)}"
        )


def _compile_constraints_unified(
    entries: list[Any] | dict[str, Any],
    path: Path,
) -> tuple[list[CompiledConstraint], list[DerivedConstraint] | None]:
    """Compile constraints supporting both legacy and TVL 0.9 formats.

    Legacy format: list of constraint objects with type/rule/when/then
    TVL 0.9 format: dict with 'structural' and/or 'derived' arrays

    Returns:
        Tuple of (compiled_constraints, derived_constraints).
        derived_constraints is None for legacy format.
    """
    # Handle TVL 0.9 format (dict with structural/derived)
    if isinstance(entries, dict):
        structural = entries.get("structural", [])
        derived_raw = entries.get("derived", [])
        compiled = _compile_structural_constraints(structural, path)
        derived = _parse_derived_constraints(derived_raw)
        return compiled, derived

    # Handle legacy format (list)
    if isinstance(entries, list):
        return _compile_constraints(entries, path), None

    return [], None


def _parse_derived_constraints(entries: list[Any]) -> list[DerivedConstraint] | None:
    """Parse TVL 0.9 derived constraints.

    Derived constraints are linear arithmetic expressions over environment
    symbols. They are stored as data only - actual SMT compilation is
    handled by TVL tools, not the runtime.

    Args:
        entries: List of derived constraint dictionaries.

    Returns:
        List of DerivedConstraint objects, or None if empty.
    """
    if not entries:
        return None

    derived: list[DerivedConstraint] = []

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TVLValidationError(
                f"Derived constraint at index {index} must be a mapping"
            )

        try:
            constraint = DerivedConstraint.from_dict(entry, index)
        except ValueError as exc:
            raise TVLValidationError(str(exc)) from exc

        derived.append(constraint)

    return derived if derived else None


def _compile_structural_constraints(
    entries: list[Any],
    path: Path,
) -> list[CompiledConstraint]:
    """Compile TVL 0.9 structural constraints.

    Structural constraints are typed DNF clauses that support:
    - expr: standalone expression
    - when/then: conditional implication
    """
    compiled: list[CompiledConstraint] = []

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TVLValidationError(
                f"Structural constraint at index {index} must be a mapping"
            )

        # Use explicit id from YAML if provided, otherwise generate from index
        constraint_id = entry.get("id") or f"structural_{index}"
        error_message = (
            entry.get("error_message")
            or f"Structural constraint {constraint_id} violated"
        )

        try:
            struct_constraint = StructuralConstraint.from_dict(entry, index)
        except ValueError as exc:
            raise TVLValidationError(str(exc)) from exc

        # Convert to rule expression
        rule_expr = struct_constraint.to_rule_expression()

        # Check if metrics are referenced (structural constraints shouldn't use metrics)
        requires_metrics = _METRICS_PREFIX in rule_expr

        compiled_rule = compile_constraint_expression(
            rule_expr, label=f"{path}:{constraint_id}"
        )

        compiled.append(
            CompiledConstraint(
                identifier=constraint_id,
                description=error_message,
                requires_metrics=requires_metrics,
                evaluator=compiled_rule,
                constraint_type="structural",
            )
        )

    return compiled


def _resolve_range(range_value: Any, name: str) -> tuple[float, float]:
    if isinstance(range_value, (list, tuple)) and len(range_value) == 2:
        return float(range_value[0]), float(range_value[1])
    if isinstance(range_value, dict):
        if "min" in range_value and "max" in range_value:
            return float(range_value["min"]), float(range_value["max"])
        if "start" in range_value and "end" in range_value:
            return float(range_value["start"]), float(range_value["end"])
    raise TVLValidationError(
        f"Parameter '{name}' requires a numeric range with min/max values"
    )


def _parse_banded_objective(
    name: str, band_spec: dict[str, Any], weight: float, unit: str | None
) -> ObjectiveDefinition:
    """Parse a banded objective (TVL 0.9)."""
    from .models import BandTarget

    if not isinstance(band_spec, dict):
        raise TVLValidationError(f"Objective '{name}' band must be a mapping")

    target = band_spec.get("target")
    if target is None:
        raise TVLValidationError(f"Banded objective '{name}' requires a 'target'")

    try:
        band_target = BandTarget.from_dict(target)
    except ValueError as exc:
        raise TVLValidationError(
            f"Invalid band target for objective '{name}': {exc}"
        ) from exc

    test_type = band_spec.get("test", "TOST")
    if test_type != "TOST":
        raise TVLValidationError(
            f"Banded objective '{name}' test must be 'TOST', got '{test_type}'"
        )

    alpha = float(band_spec.get("alpha", 0.05))
    if not 0 < alpha < 1:
        raise TVLValidationError(
            f"Banded objective '{name}' alpha must be in (0, 1), got {alpha}"
        )

    return ObjectiveDefinition(
        name=name,
        orientation="band",
        weight=weight,
        unit=unit,
        band=band_target,
        band_test="TOST",
        band_alpha=alpha,
    )


def _parse_standard_objective(
    name: str, entry: dict[str, Any], weight: float, unit: str | None
) -> ObjectiveDefinition:
    """Parse a standard objective with direction."""
    direction = (entry.get("direction") or "maximize").lower()
    if direction not in {"maximize", "minimize"}:
        raise TVLValidationError(
            f"Objective '{name}' direction must be 'maximize' or 'minimize'"
        )
    return ObjectiveDefinition(
        name=name,
        orientation=cast(Literal["maximize", "minimize"], direction),
        weight=weight,
        unit=unit,
    )


def _parse_objectives(resolved: dict[str, Any]) -> ObjectiveSchema | None:
    """Parse objectives supporting both standard and banded objectives (TVL 0.9).

    Standard objectives have a direction (maximize/minimize).
    Banded objectives have a band with target, test, and alpha.
    """
    objectives = resolved.get("objectives")
    if objectives is None:
        return None
    if not isinstance(objectives, list) or not objectives:
        raise TVLValidationError("objectives must be a non-empty list when provided")

    definitions: list[ObjectiveDefinition] = []
    for entry in objectives:
        if not isinstance(entry, dict):
            raise TVLValidationError("Each objective entry must be a mapping")
        name = entry.get("name")
        if not isinstance(name, str):
            raise TVLValidationError("Each objective requires a 'name'")

        weight = float(entry.get("weight", 1.0))
        unit = entry.get("unit") if isinstance(entry.get("unit"), str) else None

        band_spec = entry.get("band")
        if band_spec is not None:
            definitions.append(_parse_banded_objective(name, band_spec, weight, unit))
        else:
            definitions.append(_parse_standard_objective(name, entry, weight, unit))

    return ObjectiveSchema.from_objectives(definitions)


def _parse_budget(optimization_section: Any) -> TVLBudget:
    if not isinstance(optimization_section, dict):
        return TVLBudget()

    budget_section = optimization_section.get("budget") or {}
    if not isinstance(budget_section, dict):
        budget_section = {}

    max_trials_raw = _extract_numeric_field(budget_section.get("max_trials"))
    max_trials = int(max_trials_raw) if max_trials_raw is not None else None

    timeout_seconds = _extract_numeric_field(budget_section.get("timeout_seconds"))

    max_total_examples_raw = _extract_numeric_field(
        budget_section.get("max_total_examples")
    )
    max_total_examples = (
        int(max_total_examples_raw) if max_total_examples_raw is not None else None
    )

    parallel_trials_value = budget_section.get("parallel_trials")
    parallel_trials = (
        int(parallel_trials_value)
        if isinstance(parallel_trials_value, (int, float))
        else None
    )
    samples_include_pruned = budget_section.get("samples_include_pruned")
    if not isinstance(samples_include_pruned, bool):
        samples_include_pruned = None

    return TVLBudget(
        max_trials=max_trials,
        parallel_trials=parallel_trials,
        timeout_seconds=timeout_seconds,
        max_total_examples=max_total_examples,
        samples_include_pruned=samples_include_pruned,
    )


def _extract_numeric_field(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict) and "value" in value:
        raw = value["value"]
        if isinstance(raw, (int, float)):
            return raw
    return None


def _resolve_algorithm(optimization_section: Any) -> str | None:
    if not isinstance(optimization_section, dict):
        return None
    strategy = optimization_section.get("strategy")

    # Handle TVL 0.9 dict format: {type: "nsga2", ...}
    if isinstance(strategy, dict):
        strategy = strategy.get("type")

    if not isinstance(strategy, str):
        return None

    normalized = strategy.lower()
    mapping = {
        "pareto_optimal": "nsga2",
        "nsga2": "nsga2",
        "grid": "grid",
        "grid_search": "grid",
        "random": "random",
        "random_search": "random",
        "bayesian": "bayesian",
        "tpe": "optuna",
    }
    return mapping.get(normalized, normalized)


def _compile_constraints(
    entries: list[Any],
    path: Path,
) -> list[CompiledConstraint]:
    compiled: list[CompiledConstraint] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TVLValidationError("constraint entries must be mappings")
        constraint_id = entry.get("id") or f"constraint_{index}"
        constraint_type = (entry.get("type") or "expression").lower()
        error_message = (
            entry.get("error_message") or f"Constraint {constraint_id} violated"
        )
        requires_metrics = False

        if constraint_type == "conditional":
            when_expr = entry.get("when")
            then_expr = entry.get("then")
            if not isinstance(when_expr, str) or not isinstance(then_expr, str):
                raise TVLValidationError(
                    f"Conditional constraint '{constraint_id}' requires 'when' and 'then' expressions"
                )
            when_compiled = compile_constraint_expression(
                when_expr, label=f"{path}:{constraint_id}:when"
            )
            then_compiled = compile_constraint_expression(
                then_expr, label=f"{path}:{constraint_id}:then"
            )
            requires_metrics = (
                _METRICS_PREFIX in when_expr or _METRICS_PREFIX in then_expr
            )

            def _conditional(
                config: dict[str, Any],
                metrics: dict[str, Any] | None = None,
                *,
                _when: Callable[..., Any] = when_compiled,
                _then: Callable[..., Any] = then_compiled,
            ) -> bool:
                if not _when(config, metrics):
                    return True
                return bool(_then(config, metrics))

            evaluator = _conditional

        else:
            rule_expr = entry.get("rule")
            if not isinstance(rule_expr, str):
                raise TVLValidationError(
                    f"Constraint '{constraint_id}' requires a string 'rule' expression"
                )
            requires_metrics = _METRICS_PREFIX in rule_expr
            compiled_rule = compile_constraint_expression(
                rule_expr, label=f"{path}:{constraint_id}:rule"
            )

            if constraint_type == "forbidden":

                def _forbidden(
                    config: dict[str, Any],
                    metrics: dict[str, Any] | None = None,
                    *,
                    _rule: Callable[..., Any] = compiled_rule,
                ) -> bool:
                    return not bool(_rule(config, metrics))

                evaluator = _forbidden  # type: ignore[assignment]
            else:
                evaluator = compiled_rule  # type: ignore[assignment]

        compiled.append(
            CompiledConstraint(
                identifier=constraint_id,
                description=error_message,
                requires_metrics=requires_metrics,
                evaluator=evaluator,
                constraint_type=constraint_type,
            )
        )

    return compiled


def _build_metadata(
    resolved: dict[str, Any],
    path: Path,
    environment: str | None,
    units: dict[str, str],
    constraints: list[CompiledConstraint],
) -> dict[str, Any]:
    spec_section = resolved.get("spec") or {}
    metadata_section = resolved.get("metadata") or {}
    triagent_section = resolved.get("triagent") or {}
    evaluation_section = resolved.get("evaluation") or {}

    metadata: dict[str, Any] = {
        "spec_id": spec_section.get("id"),
        "spec_version": spec_section.get("version"),
        "spec_path": str(path),
        "environment": environment,
        "owner": metadata_section.get("owner"),
        "description": metadata_section.get("description"),
        "units": units,
        "triagent": triagent_section,
        "evaluation": evaluation_section,
        "constraints": [
            {
                "id": constraint.identifier,
                "type": constraint.constraint_type,
                "requires_metrics": constraint.requires_metrics,
            }
            for constraint in constraints
        ],
    }
    tags = metadata_section.get("tags")
    if isinstance(tags, list):
        metadata["tags"] = list(tags)
    return metadata


def _apply_environment(
    raw_data: dict[str, Any], environment: str | None
) -> dict[str, Any]:
    resolved = copy.deepcopy(raw_data)
    environments = resolved.get("environments") or {}
    if not environment:
        resolved.pop("environments", None)
        return resolved
    if not isinstance(environments, dict) or environment not in environments:
        raise TVLValidationError(
            f"Environment '{environment}' not defined in the TVL spec"
        )

    overlay = _resolve_environment_overlay(environments, environment, set())
    resolved.pop("environments", None)
    return _deep_merge(resolved, overlay)


def _resolve_environment_overlay(
    environments: dict[str, Any],
    target: str,
    seen: set[str],
) -> dict[str, Any]:
    if target in seen:
        raise TVLValidationError("Environment overlay cycle detected")
    seen.add(target)
    entry = environments.get(target) or {}
    if not isinstance(entry, dict):
        raise TVLValidationError(f"Environment '{target}' must be defined as a mapping")
    base_overlay: dict[str, Any] = {}
    extends = entry.get("extends")
    if isinstance(extends, str):
        base_overlay = _resolve_environment_overlay(environments, extends, seen)
    direct_overrides = {
        key: value
        for key, value in entry.items()
        if key not in {"extends", "overrides"}
    }
    overlay_content = entry.get("overrides") or {}
    if not isinstance(overlay_content, dict):
        overlay_content = {}
    merged = _deep_merge(base_overlay, overlay_content)
    if direct_overrides:
        merged = _deep_merge(merged, direct_overrides)
    return merged


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _deep_merge_lists(
    base: dict[str, Any], overlay: dict[str, Any], list_keys: set[str]
) -> dict[str, Any]:
    """Deep merge with list concatenation for specified keys.

    Unlike _deep_merge which replaces lists, this function concatenates lists
    for specified keys (e.g., safety_constraints, objectives).

    Args:
        base: Base dictionary to merge into.
        overlay: Overlay dictionary with values to merge.
        list_keys: Set of keys where lists should be concatenated rather than replaced.

    Returns:
        Merged dictionary with lists concatenated for specified keys.
    """
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in list_keys and isinstance(value, list):
            # Concatenate lists for inheritance
            base_list = result.get(key, [])
            if isinstance(base_list, list):
                result[key] = base_list + copy.deepcopy(value)
            else:
                result[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_lists(result[key], value, list_keys)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_spec_extends(
    raw_data: dict[str, Any],
    spec_path: Path,
    seen_paths: set[Path] | None = None,
) -> dict[str, Any]:
    """Resolve spec-level inheritance via 'extends' key.

    Supports inheriting from another TVL spec file. The base spec is loaded
    and deep-merged with the current spec. Lists for safety_constraints,
    objectives, tvars, and constraints are concatenated (not replaced).

    Args:
        raw_data: Parsed YAML data (raw dictionary).
        spec_path: Path to the current spec file (for relative path resolution).
        seen_paths: Set of already-visited paths for cycle detection.

    Returns:
        Resolved spec with inheritance applied.

    Raises:
        TVLValidationError: If there's a cycle in the extends chain or
            the base spec cannot be loaded.

    Example:
        # child.tvl.yml
        extends: ./base_safety.tvl.yml
        tvars:
          - name: model
            domain: ["gpt-4o"]

        # The child spec inherits all sections from base_safety.tvl.yml
        # and can override or extend them.
    """
    if seen_paths is None:
        seen_paths = set()

    extends = raw_data.get("extends")
    if not extends:
        return raw_data

    if not isinstance(extends, str):
        raise TVLValidationError(
            f"'extends' must be a string path, got {type(extends).__name__}"
        )

    # Resolve path relative to current spec
    base_path = (spec_path.parent / extends).resolve()

    # Cycle detection
    if base_path in seen_paths:
        cycle_list = " -> ".join(str(p) for p in seen_paths) + f" -> {base_path}"
        raise TVLValidationError(f"Circular extends detected: {cycle_list}")

    seen_paths.add(spec_path.resolve())

    # Load base spec
    if not base_path.is_file():
        raise TVLValidationError(
            f"Base spec not found: {extends} (resolved to {base_path})"
        )

    try:
        base_data = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TVLValidationError(f"Failed to parse base spec {extends}: {exc}") from exc

    if not isinstance(base_data, dict):
        raise TVLValidationError(f"Base spec {extends} must be a mapping")

    # Recursively resolve base spec's extends
    resolved_base = _resolve_spec_extends(base_data, base_path, seen_paths)

    # Remove 'extends' from current spec before merging
    current_data = {k: v for k, v in raw_data.items() if k != "extends"}

    # Keys where lists should be concatenated (inherited + child)
    list_concat_keys = {
        "safety_constraints",
        "objectives",
        "tvars",
        # Note: constraints can be list or dict, handle dict case separately
    }

    # Deep merge with list concatenation for specified keys
    merged = _deep_merge_lists(resolved_base, current_data, list_concat_keys)

    # Special handling for constraints: if both are lists, concatenate
    base_constraints = resolved_base.get("constraints")
    current_constraints = current_data.get("constraints")
    if isinstance(base_constraints, list) and isinstance(current_constraints, list):
        merged["constraints"] = base_constraints + current_constraints
    elif isinstance(base_constraints, dict) and isinstance(current_constraints, dict):
        # Merge structural and derived separately
        merged_constraints: dict[str, Any] = {}
        for section in ("structural", "derived"):
            base_list = base_constraints.get(section, [])
            current_list = current_constraints.get(section, [])
            if isinstance(base_list, list) and isinstance(current_list, list):
                merged_constraints[section] = base_list + current_list
            elif current_list:
                merged_constraints[section] = current_list
            elif base_list:
                merged_constraints[section] = base_list
        merged["constraints"] = merged_constraints

    logger.debug("Resolved spec extends: %s -> %s", spec_path.name, base_path.name)
    return merged


def _translate_expression(expression: str) -> str:
    translated = expression
    translated = re.sub(r"&&", " and ", translated)
    translated = re.sub(r"\|\|", " or ", translated)
    translated = re.sub(r"(?<![=!<>])!(?![=])", " not ", translated)
    translated = re.sub(r"\btrue\b", "True", translated, flags=re.IGNORECASE)
    translated = re.sub(r"\bfalse\b", "False", translated, flags=re.IGNORECASE)
    translated = re.sub(r"\bnull\b", "None", translated, flags=re.IGNORECASE)
    return translated


_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Call,
    ast.IfExp,
    ast.Attribute,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Subscript,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Is,
    ast.IsNot,
    ast.In,
    ast.NotIn,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
)

_SAFE_CONSTRAINT_CALLS = frozenset({"len", "min", "max", "sum", "abs", "any", "all"})
_SAFE_MATH_CALLS = frozenset(
    {
        "sqrt",
        "log",
        "log2",
        "log10",
        "exp",
        "expm1",
        "pow",
        "floor",
        "ceil",
        "trunc",
        "fabs",
        "factorial",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "degrees",
        "radians",
        "hypot",
        "isfinite",
        "isinf",
        "isnan",
        "prod",
        "fsum",
        "gcd",
        "lcm",
        "comb",
        "perm",
        "copysign",
        "fmod",
        "remainder",
        "dist",
        "erf",
        "erfc",
        "gamma",
        "lgamma",
        "cbrt",
    }
)


def _is_allowed_constraint_call(func_node: ast.AST) -> bool:
    """Return True when a call target is explicitly permitted in constraints."""
    if isinstance(func_node, ast.Name):
        return func_node.id in _SAFE_CONSTRAINT_CALLS

    if isinstance(func_node, ast.Attribute):
        # Allow explicit math.<fn> subset only (e.g., math.sqrt, math.log)
        if (
            isinstance(func_node.value, ast.Name)
            and func_node.value.id == "math"
            and func_node.attr in _SAFE_MATH_CALLS
        ):
            return True

    return False


def _validate_expression_ast(node: ast.AST, source: str) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_AST_NODES):
            raise TVLValidationError(
                f"Unsupported expression construct '{type(child).__name__}' in '{source}'"
            )
        if isinstance(child, ast.Attribute) and child.attr.startswith("__"):
            raise TVLValidationError(
                f"Unsafe attribute access '{child.attr}' in '{source}'"
            )
        if isinstance(child, ast.Name) and child.id.startswith("__"):
            raise TVLValidationError(f"Unsafe identifier '{child.id}' in '{source}'")
        if isinstance(child, ast.Call) and not _is_allowed_constraint_call(child.func):
            raise TVLValidationError(
                f"Unsupported function call in '{source}'; "
                "only len/min/max/sum/abs/any/all and math.<fn> are allowed"
            )


class _AttributeView:
    """Proxy that exposes dictionary keys as attributes."""

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, item: str) -> Any:
        if item in self._data:
            return self._wrap(self._data[item])
        return None

    def __getitem__(self, key: str) -> Any:
        return self._wrap(self._data[key])

    def get(self, key: str, default: Any = None) -> Any:
        return self._wrap(self._data.get(key, default))

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, dict):
            return _AttributeView(value)
        return value
