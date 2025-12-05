"""Load and normalize TVL specifications for the TraiGent SDK."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-TVLSPEC REQ-TVLSPEC-012 SYNC-OptimizationFlow

from __future__ import annotations

import ast
import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, cast

import yaml

from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema
from traigent.utils.exceptions import TVLValidationError
from traigent.utils.logging import get_logger

logger = get_logger(__name__)


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
    """Normalized view of a TVL spec ready for the decorator/runtime."""

    path: Path
    environment: str | None
    configuration_space: dict[str, Any]
    objective_schema: ObjectiveSchema | None
    constraints: list[Callable[[dict[str, Any], dict[str, Any] | None], bool]]
    default_config: dict[str, Any]
    metadata: dict[str, Any]
    budget: TVLBudget
    algorithm: str | None

    def runtime_overrides(self) -> dict[str, Any]:
        """Return decorator/runtime overrides derived from the spec."""

        overrides: dict[str, Any] = {}
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
        if self.metadata:
            overrides.setdefault("tvl_metadata", self.metadata)
        return overrides


def load_tvl_spec(
    *,
    spec_path: str | Path,
    environment: str | None = None,
    validate_constraints: bool = True,
) -> TVLSpecArtifact:
    """Load and normalize a TVL specification from disk."""

    path = Path(spec_path).expanduser()
    if not path.is_file():
        raise TVLValidationError(f"TVL spec not found: {path}")

    try:
        raw_data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - defensive guard
        raise TVLValidationError(f"Failed to parse TVL spec: {exc}") from exc

    if not isinstance(raw_data, dict):
        raise TVLValidationError("TVL specification must be a mapping")

    resolved = _apply_environment(raw_data, environment)
    config_space, defaults, units = _parse_configuration_space(resolved)
    objective_schema = _parse_objectives(resolved)
    compiled_constraints = _compile_constraints(
        resolved.get("constraints", []) or [], validate_constraints, path
    )
    constraint_wrappers = [
        constraint.to_callable() for constraint in compiled_constraints
    ]
    budget = _parse_budget(resolved.get("optimization"))
    algorithm = _resolve_algorithm(resolved.get("optimization"))
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
    )


def compile_constraint_expression(
    expression: str, *, label: str
) -> Callable[[dict[str, Any], dict[str, Any] | None], bool]:
    """Compile a CEL-like constraint into a safe Python callable."""

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
            # 2. Only whitelisted AST nodes are allowed (no imports, lambdas, etc.)
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


def _parse_objectives(resolved: dict[str, Any]) -> ObjectiveSchema | None:
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
        direction = (entry.get("direction") or "maximize").lower()
        if direction not in {"maximize", "minimize"}:
            raise TVLValidationError(
                f"Objective '{name}' direction must be 'maximize' or 'minimize'"
            )
        weight = float(entry.get("weight", 1.0))
        unit = entry.get("unit") if isinstance(entry.get("unit"), str) else None
        definitions.append(
            ObjectiveDefinition(
                name=name,
                orientation=cast(Literal["maximize", "minimize"], direction),
                weight=weight,
                unit=unit,
            )
        )

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
    validate_constraints: bool,
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
            requires_metrics = "metrics." in when_expr or "metrics." in then_expr

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
            requires_metrics = "metrics." in rule_expr
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


def _validate_expression_ast(node: ast.AST, source: str) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_AST_NODES):
            raise TVLValidationError(
                f"Unsupported expression construct '{type(child).__name__}' in '{source}'"
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
