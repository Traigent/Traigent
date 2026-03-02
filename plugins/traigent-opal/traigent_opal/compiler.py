"""Compile a pragmatic OPAL subset into Traigent optimize kwargs.

The parser is intentionally small and targets the current OPAL-in-Python workflow:
- Tuned variable declarations: `x in {...}` and `x in [lo, hi] step s`
- Fixed assignments: `x = literal`
- Optimization directives: `objective`, `constraint`, `chance_constraint`
- Python-valid directive form: `# opal: <directive ...>`
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from traigent.core.objectives import ObjectiveDefinition, ObjectiveSchema

if TYPE_CHECKING:
    from .api import (
        CallableTemplate,
        ChanceConstraintSpec,
        ConstraintSpec,
        DomainChoices,
        DomainRange,
        ObjectiveSpec,
        ProgramSpec,
    )

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TVAR_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+)$")
_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")
_RANGE_RE = re.compile(
    r"^\[\s*(?P<lo>.+?)\s*,\s*(?P<hi>.+?)\s*\](?:\s+step\s+(?P<step>.+))?$"
)
_OBJECTIVE_RE = re.compile(
    r"^objective\s+(maximize|minimize)\s+([A-Za-z_][A-Za-z0-9_\-.]*)(?:\s+on\s+([A-Za-z_][A-Za-z0-9_\-.]*))?$"
)
_DIRECTIVE_PREFIX = "# opal:"
_FLOW_PREFIXES = (
    "def ",
    "class ",
    "if ",
    "for ",
    "while ",
    "with ",
    "return ",
    "import ",
    "from ",
    "raise ",
    "assert ",
    "try:",
    "except ",
    "elif ",
    "else:",
    "finally:",
    "yield ",
    "async ",
    "await ",
)


class OpalCompileError(ValueError):
    """Raised when OPAL source cannot be compiled into a supported subset."""


@dataclass(slots=True)
class ObjectiveDecl:
    """A parsed objective directive."""

    direction: Literal["maximize", "minimize"]
    metric: str
    dataset: str | None = None
    raw: str = ""


@dataclass(slots=True)
class OpalCompilationArtifact:
    """Compiled OPAL declarations that can feed Traigent optimization APIs."""

    module_name: str | None
    tuned_variables: dict[str, list[Any] | tuple[float, float]]
    assignments: dict[str, Any]
    objectives: list[ObjectiveDecl] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    chance_constraints: list[str] = field(default_factory=list)
    objective_schema: ObjectiveSchema | None = None

    def to_optimize_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for `@traigent.optimize(...)`.

        The return shape mirrors the common Traigent decorator style where each
        tuned variable is a direct keyword argument and `objectives` is an
        explicit list.
        """
        overlap = sorted(set(self.assignments) & set(self.tuned_variables))
        if overlap:
            raise OpalCompileError(
                f"Assignment/tvar collision in artifact: {overlap}"
            )

        kwargs: dict[str, Any] = dict(self.assignments)
        kwargs.update(self.tuned_variables)

        if self.objective_schema is not None:
            kwargs["objectives"] = self.objective_schema
        elif self.objectives:
            ordered_metrics: list[str] = []
            for obj in self.objectives:
                if obj.metric not in ordered_metrics:
                    ordered_metrics.append(obj.metric)
            kwargs["objectives"] = ordered_metrics

        return kwargs

    def to_directive_metadata(self) -> dict[str, Any]:
        """Return parsed directives for callers that need governance metadata."""
        return {
            "module": self.module_name,
            "objectives": [
                {
                    "direction": obj.direction,
                    "metric": obj.metric,
                    "dataset": obj.dataset,
                    "raw": obj.raw,
                }
                for obj in self.objectives
            ],
            "constraints": list(self.constraints),
            "chance_constraints": list(self.chance_constraints),
        }


def compile_opal_source(source: str) -> OpalCompilationArtifact:
    """Compile OPAL subset declarations from source text.

    The compiler accepts both raw OPAL lines and Python-commented directives
    (`# opal: ...`) so examples can remain valid Python source.
    """
    module_name: str | None = None
    tuned_variables: dict[str, list[Any] | tuple[float, float]] = {}
    assignments: dict[str, Any] = {}
    objectives: list[ObjectiveDecl] = []
    constraints: list[str] = []
    chance_constraints: list[str] = []

    for line_no, raw_line in enumerate(source.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        opal_line = _normalize_directive_line(stripped)
        if opal_line is None:
            continue

        if opal_line.startswith("module "):
            module_name = _parse_module_name(opal_line, line_no)
            continue

        if opal_line.startswith("objective "):
            objectives.append(_parse_objective(opal_line, line_no))
            continue

        if opal_line.startswith("constraint "):
            expr = opal_line[len("constraint ") :].strip()
            if not expr:
                raise OpalCompileError(
                    f"Line {line_no}: empty constraint directive is not allowed"
                )
            constraints.append(expr)
            continue

        if opal_line.startswith("chance_constraint "):
            expr = opal_line[len("chance_constraint ") :].strip()
            if not expr:
                raise OpalCompileError(
                    f"Line {line_no}: empty chance_constraint directive is not allowed"
                )
            chance_constraints.append(expr)
            continue

        tvar_match = _TVAR_RE.match(opal_line)
        if tvar_match:
            name = tvar_match.group(1)
            if name in assignments:
                raise OpalCompileError(
                    f"Line {line_no}: '{name}' already declared with '='; cannot redeclare with 'in'"
                )
            if name in tuned_variables:
                raise OpalCompileError(
                    f"Line {line_no}: duplicate tuned variable declaration for '{name}'"
                )
            tuned_variables[name] = _parse_domain_expr(tvar_match.group(2).strip(), line_no)
            continue

        assign_match = _ASSIGN_RE.match(opal_line)
        if assign_match and not _looks_like_function_or_flow_statement(opal_line):
            name = assign_match.group(1)
            if name in tuned_variables:
                raise OpalCompileError(
                    f"Line {line_no}: '{name}' already declared with 'in'; cannot redeclare with '='"
                )

            rhs = assign_match.group(2).strip()
            try:
                assignments[name] = _parse_literal(rhs, line_no)
                continue
            except OpalCompileError:
                # Regular behavior-plane Python assignment (e.g. constructor calls)
                # is intentionally ignored by the declarative OPAL compiler.
                pass

        # Ignore regular Python behavior-plane statements.

    return OpalCompilationArtifact(
        module_name=module_name,
        tuned_variables=tuned_variables,
        assignments=assignments,
        objectives=objectives,
        constraints=constraints,
        chance_constraints=chance_constraints,
        objective_schema=_build_objective_schema_from_decls(objectives),
    )


def compile_opal_file(path: str | Path) -> OpalCompilationArtifact:
    """Compile OPAL subset declarations from a file path."""
    source_path = Path(path)
    if not source_path.is_file():
        raise OpalCompileError(f"OPAL file not found: {source_path}")
    return compile_opal_source(source_path.read_text(encoding="utf-8"))


def compile_opal_spec(spec: ProgramSpec) -> OpalCompilationArtifact:
    """Compile a native Python ProgramSpec into an OPAL artifact.

    This is the Python-first path for users who do not want to author OPAL
    source strings in regular Python code.
    """
    # Local import keeps module graph acyclic for tools that only use string mode.
    from .api import (
        CallableTemplate,
        ChanceConstraintSpec,
        ConstraintSpec,
        DomainChoices,
        DomainRange,
        ProgramSpec,
    )

    if not isinstance(spec, ProgramSpec):
        raise OpalCompileError(
            f"compile_opal_spec expects ProgramSpec, got {type(spec).__name__}"
        )

    tuned_variables: dict[str, list[Any] | tuple[float, float]] = {}
    objectives: list[ObjectiveDecl] = []
    constraints: list[str] = []
    chance_constraints: list[str] = []

    for tv in spec.tvars:
        domain = tv.domain
        if isinstance(domain, DomainChoices):
            if not domain.values:
                raise OpalCompileError(f"TVar '{tv.name}' has empty choices domain")
            tuned_variables[tv.name] = [_render_choice_value(v) for v in domain.values]
        elif isinstance(domain, DomainRange):
            if domain.hi < domain.lo:
                raise OpalCompileError(
                    f"TVar '{tv.name}' range invalid: hi ({domain.hi}) < lo ({domain.lo})"
                )
            if domain.step is not None and domain.step <= 0:
                raise OpalCompileError(
                    f"TVar '{tv.name}' range invalid: step must be > 0"
                )

            if domain.step is None:
                tuned_variables[tv.name] = (float(domain.lo), float(domain.hi))
            else:
                tuned_variables[tv.name] = _discretize_range(
                    float(domain.lo),
                    float(domain.hi),
                    float(domain.step),
                )
        else:
            raise OpalCompileError(
                f"TVar '{tv.name}' uses unsupported domain type: {type(domain).__name__}"
            )

    for obj in spec.objectives:
        dataset = obj.on if obj.on is not None else spec.evaluation_set
        objectives.append(
            ObjectiveDecl(
                direction=obj.direction,
                metric=obj.metric,
                dataset=dataset,
                raw=_render_objective_raw(obj.direction, obj.metric, dataset),
            )
        )

    for cons in spec.constraints:
        rendered = _render_constraint(cons)
        if not rendered:
            raise OpalCompileError("ConstraintSpec rendered to empty expression")
        constraints.append(rendered)

    for chance in spec.chance_constraints:
        rendered = _render_chance_constraint(chance)
        if not rendered:
            raise OpalCompileError("ChanceConstraintSpec rendered to empty expression")
        chance_constraints.append(rendered)

    objective_schema = _build_objective_schema(spec.objectives)

    return OpalCompilationArtifact(
        module_name=spec.module,
        tuned_variables=tuned_variables,
        assignments=dict(spec.assignments),
        objectives=objectives,
        constraints=constraints,
        chance_constraints=chance_constraints,
        objective_schema=objective_schema,
    )


def _normalize_directive_line(line: str) -> str | None:
    """Return the OPAL directive payload or None if line is irrelevant.

    Rules:
    - `# opal: ...` => return payload
    - Other comments => ignore
    - Raw OPAL directives / declarations => return as-is
    - Plain Python behavior code => return as-is for possible assignment parsing
    """
    lowered = line.lower()
    if lowered.startswith(_DIRECTIVE_PREFIX):
        payload = line[len(_DIRECTIVE_PREFIX) :].strip()
        return payload or None

    if line.startswith("#"):
        return None

    return line


def _parse_module_name(line: str, line_no: int) -> str:
    """Parse `module foo.bar` declaration."""
    module_name = line[len("module ") :].strip()
    if not module_name:
        raise OpalCompileError(f"Line {line_no}: module name cannot be empty")

    parts = module_name.split(".")
    if not all(_IDENTIFIER_RE.match(part) for part in parts):
        raise OpalCompileError(
            f"Line {line_no}: invalid module name '{module_name}'"
        )
    return module_name


def _parse_objective(line: str, line_no: int) -> ObjectiveDecl:
    """Parse an objective directive."""
    match = _OBJECTIVE_RE.match(line)
    if not match:
        raise OpalCompileError(
            "Line "
            f"{line_no}: invalid objective syntax; expected "
            "`objective maximize|minimize <metric> [on <dataset>]`"
        )

    direction = cast(Literal["maximize", "minimize"], match.group(1))
    metric = match.group(2)
    dataset = match.group(3)
    return ObjectiveDecl(direction=direction, metric=metric, dataset=dataset, raw=line)


def _parse_domain_expr(expr: str, line_no: int) -> list[Any] | tuple[float, float]:
    """Parse OPAL domain expressions for the supported subset."""
    if expr.startswith("{") and expr.endswith("}"):
        return _parse_set_domain(expr[1:-1], line_no)

    range_match = _RANGE_RE.match(expr)
    if range_match:
        lo = _parse_literal(range_match.group("lo"), line_no)
        hi = _parse_literal(range_match.group("hi"), line_no)
        step_raw = range_match.group("step")

        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            raise OpalCompileError(
                f"Line {line_no}: range bounds must be numeric, got {lo!r}, {hi!r}"
            )

        lo_f = float(lo)
        hi_f = float(hi)
        if hi_f < lo_f:
            raise OpalCompileError(
                f"Line {line_no}: range upper bound must be >= lower bound"
            )

        if step_raw is None:
            return (lo_f, hi_f)

        step = _parse_literal(step_raw, line_no)
        if not isinstance(step, (int, float)):
            raise OpalCompileError(f"Line {line_no}: range step must be numeric")

        step_f = float(step)
        if step_f <= 0:
            raise OpalCompileError(f"Line {line_no}: range step must be positive")

        return _discretize_range(lo_f, hi_f, step_f)

    raise OpalCompileError(
        f"Line {line_no}: unsupported domain syntax '{expr}'. "
        "Supported: {a, b} and [lo, hi] step s"
    )


def _parse_set_domain(content: str, line_no: int) -> list[Any]:
    """Parse `{...}` domain values.

    Values are parsed with `ast.literal_eval` when possible; bare identifiers are
    accepted as strings to support concise OPAL model names.
    """
    text = content.strip()
    if not text:
        raise OpalCompileError(f"Line {line_no}: set domain cannot be empty")

    try:
        parsed = ast.literal_eval(f"[{text}]")
        if not isinstance(parsed, list):
            raise OpalCompileError(f"Line {line_no}: invalid set domain")
        if not parsed:
            raise OpalCompileError(f"Line {line_no}: set domain cannot be empty")
        return parsed
    except Exception as parse_err:
        # Fallback for bare identifiers: {gpt_4o, claude_3_5}
        values: list[Any] = []
        for token in _split_csv_top_level(text):
            token = token.strip()
            if not token:
                continue
            if _IDENTIFIER_RE.match(token):
                values.append(token)
            else:
                values.append(_parse_literal(token, line_no))

        if not values:
            raise OpalCompileError(
                f"Line {line_no}: set domain cannot be empty"
            ) from parse_err
        return values


def _split_csv_top_level(text: str) -> list[str]:
    """Split by commas while respecting simple bracket and string nesting."""
    items: list[str] = []
    current: list[str] = []
    depth = 0
    in_single = False
    in_double = False

    prev_char = ""
    for char in text:
        if char == "'" and not in_double and prev_char != "\\":
            in_single = not in_single
        elif char == '"' and not in_single and prev_char != "\\":
            in_double = not in_double
        elif not in_single and not in_double:
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(depth - 1, 0)
            elif char == "," and depth == 0:
                items.append("".join(current))
                current = []
                prev_char = char
                continue

        current.append(char)
        prev_char = char

    if current:
        items.append("".join(current))
    return items


def _parse_literal(expr: str, line_no: int) -> Any:
    """Parse a literal assignment/expression value."""
    try:
        return ast.literal_eval(expr)
    except Exception as exc:
        raise OpalCompileError(
            f"Line {line_no}: expected a literal value in '{expr}'"
        ) from exc


def _discretize_range(lo: float, hi: float, step: float) -> list[float]:
    """Create an inclusive discretized list from [lo, hi] with step."""
    values: list[float] = []
    guard = 0
    max_steps = 100_000
    tolerance = max(abs(step) * 1e-9, 1e-12)

    while True:
        current = lo + guard * step
        if current > hi + tolerance:
            break
        rounded = _round_if_close_integer(current)
        values.append(rounded)
        guard += 1
        if guard > max_steps:
            raise OpalCompileError(
                "Range discretization produced too many values; check bounds/step"
            )

    # De-duplicate numeric artifacts while preserving order.
    deduped: list[float] = []
    seen: set[float] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _round_if_close_integer(value: float) -> float:
    """Normalize values that are numerically very close to clean decimal forms."""
    rounded_decimal = round(value, 12)
    if math.isclose(value, rounded_decimal, rel_tol=0.0, abs_tol=1e-12):
        value = rounded_decimal

    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-12):
        return float(nearest)
    return value


def _looks_like_function_or_flow_statement(line: str) -> bool:
    """Heuristic guard so regular Python statements are not treated as declarations."""
    return line.startswith(_FLOW_PREFIXES)


def _build_objective_schema(objectives: list[ObjectiveSpec]) -> ObjectiveSchema | None:
    if not objectives:
        return None

    defs = [
        ObjectiveDefinition(
            name=obj.metric,
            orientation=obj.direction,
            weight=float(getattr(obj, "weight", 1.0)),
        )
        for obj in objectives
    ]
    return ObjectiveSchema.from_objectives(defs)


def _build_objective_schema_from_decls(
    objectives: list[ObjectiveDecl],
) -> ObjectiveSchema | None:
    if not objectives:
        return None
    defs = [
        ObjectiveDefinition(name=obj.metric, orientation=obj.direction, weight=1.0)
        for obj in objectives
    ]
    return ObjectiveSchema.from_objectives(defs)


def _render_objective_raw(direction: str, metric: str, on: str | None) -> str:
    if on:
        return f"objective {direction} {metric} on {on}"
    return f"objective {direction} {metric}"


def _render_guard_value(value: Any) -> str:
    if isinstance(value, str) and _IDENTIFIER_RE.match(value):
        return value
    return repr(value)


def _render_constraint(cons: ConstraintSpec) -> str:
    if cons.expr:
        return cons.expr
    return f"when {cons.when_name} is {_render_guard_value(cons.is_value)}: {cons.then_expr}"


def _render_choice_value(value: Any) -> Any:
    # Callable templates are passed as structured dictionaries so downstream
    # consumers can still inspect callable and args metadata.
    from .api import CallableTemplate

    if isinstance(value, CallableTemplate):
        rendered_args: dict[str, Any] = {}
        for key, arg_val in value.args.items():
            rendered_args[key] = getattr(arg_val, "name", arg_val)
        return {"callable": value.name, "args": rendered_args}
    return value


def _render_chance_constraint(chance: ChanceConstraintSpec) -> str:
    return (
        f"P({chance.expr}) <= {chance.threshold:g} confidence {chance.confidence:g}"
    )
