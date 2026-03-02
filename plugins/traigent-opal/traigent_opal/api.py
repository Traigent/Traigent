"""Native Python object API for Traigent OPAL declarations."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from traigent.api.parameter_ranges import Choices as CoreChoices
    from traigent.api.parameter_ranges import IntRange as CoreIntRange
    from traigent.api.parameter_ranges import LogRange as CoreLogRange
    from traigent.api.parameter_ranges import Range as CoreRange

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class DomainChoices:
    """Finite domain for a tuned variable."""

    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("DomainChoices requires at least one value")


@dataclass(frozen=True, slots=True)
class DomainRange:
    """Numeric range domain for a tuned variable."""

    lo: float
    hi: float
    step: float | None = None

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError("DomainRange requires hi >= lo")
        if self.step is not None and self.step <= 0:
            raise ValueError("DomainRange step must be positive when provided")


@dataclass(frozen=True, slots=True)
class CallableTemplate:
    """Executable callable domain member with keyword arguments."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CallableTemplate name cannot be empty")


@dataclass(frozen=True, slots=True)
class TVarSpec:
    """Tuned-variable declaration in Python object form."""

    name: str
    domain: DomainChoices | DomainRange
    given: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TVarSpec name cannot be empty")


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Objective declaration with explicit direction."""

    direction: Literal["maximize", "minimize"]
    metric: str
    on: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("ObjectiveSpec direction must be 'maximize' or 'minimize'")
        if not self.metric:
            raise ValueError("ObjectiveSpec metric cannot be empty")
        if self.weight <= 0:
            raise ValueError("ObjectiveSpec weight must be > 0")


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    """Constraint declaration."""

    expr: str | None = None
    when_name: str | None = None
    is_value: Any | None = None
    then_expr: str | None = None

    def __post_init__(self) -> None:
        has_expr = bool(self.expr and self.expr.strip())
        has_guard = bool(
            self.when_name
            and self.is_value is not None
            and self.then_expr is not None
            and self.then_expr.strip()
        )

        if has_expr and has_guard:
            raise ValueError(
                "ConstraintSpec requires either expr, or when_name/is_value/then_expr (not both)"
            )
        if has_expr or has_guard:
            return
        raise ValueError(
            "ConstraintSpec requires either expr, or when_name/is_value/then_expr"
        )


@dataclass(frozen=True, slots=True)
class ChanceConstraintSpec:
    """Chance-constraint declaration."""

    expr: str
    threshold: float
    confidence: float

    def __post_init__(self) -> None:
        if not self.expr:
            raise ValueError("ChanceConstraintSpec expr cannot be empty")
        if not 0 <= self.threshold <= 1:
            raise ValueError("ChanceConstraintSpec threshold must be in [0, 1]")
        if not 0 < self.confidence < 1:
            raise ValueError("ChanceConstraintSpec confidence must be in (0, 1)")


@dataclass(slots=True)
class ProgramSpec:
    """Top-level optimization-plane declaration in Python object form."""

    module: str | None = None
    evaluation_set: str | None = None
    tvars: list[TVarSpec] = field(default_factory=list)
    assignments: dict[str, Any] = field(default_factory=dict)
    objectives: list[ObjectiveSpec] = field(default_factory=list)
    constraints: list[ConstraintSpec] = field(default_factory=list)
    chance_constraints: list[ChanceConstraintSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        tvar_names = [tv.name for tv in self.tvars]
        if len(tvar_names) != len(set(tvar_names)):
            duplicates = sorted(
                {name for name in tvar_names if tvar_names.count(name) > 1}
            )
            raise ValueError(f"Duplicate tvar names: {duplicates}")

        overlap = sorted(set(tvar_names) & set(self.assignments.keys()))
        if overlap:
            raise ValueError(
                f"Assignment/tvar name collision: {overlap}. Use either assignment or tvar declaration, not both."
            )


@dataclass(frozen=True, slots=True)
class SymbolRef:
    """Named declaration symbol used in scoped OPAL builder blocks."""

    builder: ProgramBuilder
    name: str


@dataclass(frozen=True, slots=True)
class TunedVariable:
    """Anonymous tuned-variable declaration for `v.name in tv(...)` syntax."""

    domain: DomainChoices | DomainRange
    given: str | None = None

    def __contains__(self, symbol: object) -> bool:
        if not isinstance(symbol, SymbolRef):
            raise TypeError(
                "Use `v.<name> in tv(...)` inside `with opal_program(...) as v:`."
            )
        symbol.builder._declare_tvar(symbol.name, self.domain, self.given)
        return True


@dataclass(slots=True)
class ProgramBuilder:
    """Scoped builder so OPAL declarations can read like OPAL in valid Python."""

    module: str | None = None
    evaluation_set: str | None = None
    _tvars: dict[str, TVarSpec] = field(default_factory=dict, init=False)
    _assignments: dict[str, Any] = field(default_factory=dict, init=False)
    _objectives: list[ObjectiveSpec] = field(default_factory=list, init=False)
    _constraints: list[ConstraintSpec] = field(default_factory=list, init=False)
    _chance_constraints: list[ChanceConstraintSpec] = field(
        default_factory=list, init=False
    )
    _active: bool = field(default=False, init=False)
    _runtime_config: ContextVar[dict[str, Any] | None] = field(
        default_factory=lambda: ContextVar("opal_runtime_config", default=None),
        init=False,
        repr=False,
    )
    _frozen_config: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> ProgramBuilder:
        self._active = True
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        self._active = False
        return False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or not _IDENTIFIER_RE.match(name):
            raise AttributeError(name)
        if self._active:
            return SymbolRef(self, name)

        if name in self._assignments:
            return self._assignments[name]
        if name in self._tvars:
            cfg = self._resolve_runtime_config()
            if cfg and name in cfg:
                return cfg[name]
            raise AttributeError(
                f"Tuned variable '{name}' is not bound. Use @v.optimize, v.using(config), or v.freeze(config)."
            )
        raise AttributeError(name)

    @property
    def spec(self) -> ProgramSpec:
        return self.build()

    def _require_active(self) -> None:
        if not self._active:
            raise ValueError(
                "ProgramBuilder declarations must run inside `with opal_program(...) as v:`."
            )

    def _symbol_name(self, name_or_symbol: str | SymbolRef) -> str:
        if isinstance(name_or_symbol, SymbolRef):
            return name_or_symbol.name
        if not isinstance(name_or_symbol, str) or not _IDENTIFIER_RE.match(name_or_symbol):
            raise ValueError(f"Invalid symbol name: {name_or_symbol!r}")
        return name_or_symbol

    def _declare_tvar(
        self, name: str, domain: DomainChoices | DomainRange, given: str | None
    ) -> None:
        self._require_active()
        if name in self._assignments:
            raise ValueError(
                f"Assignment/tvar name collision: ['{name}']. Use either assignment or tvar declaration, not both."
            )
        if name in self._tvars:
            raise ValueError(f"Duplicate tvar names: ['{name}']")
        self._tvars[name] = TVarSpec(name=name, domain=domain, given=given)

    def tvar(
        self, name_or_symbol: str | SymbolRef, domain: Any, *, given: str | None = None
    ) -> TVarSpec:
        name = self._symbol_name(name_or_symbol)
        normalized = _normalize_domain(domain)
        self._declare_tvar(name, normalized, given)
        return self._tvars[name]

    def assign(self, name_or_symbol: str | SymbolRef, value: Any) -> None:
        self._require_active()
        name = self._symbol_name(name_or_symbol)
        if name in self._tvars:
            raise ValueError(
                f"Assignment/tvar name collision: ['{name}']. Use either assignment or tvar declaration, not both."
            )
        self._assignments[name] = value

    def maximize(
        self, metric: str, on: str | None = None, *, weight: float = 1.0
    ) -> ObjectiveSpec:
        self._require_active()
        obj = maximize(
            metric,
            on=on if on is not None else self.evaluation_set,
            weight=weight,
        )
        self._objectives.append(obj)
        return obj

    def minimize(
        self, metric: str, on: str | None = None, *, weight: float = 1.0
    ) -> ObjectiveSpec:
        self._require_active()
        obj = minimize(
            metric,
            on=on if on is not None else self.evaluation_set,
            weight=weight,
        )
        self._objectives.append(obj)
        return obj

    def objective(self, spec: ObjectiveSpec) -> ObjectiveSpec:
        self._require_active()
        self._objectives.append(spec)
        return spec

    def constraint(self, expr_or_constraint: str | ConstraintSpec) -> ConstraintSpec:
        self._require_active()
        cons = (
            expr_or_constraint
            if isinstance(expr_or_constraint, ConstraintSpec)
            else constraint(expr_or_constraint)
        )
        self._constraints.append(cons)
        return cons

    def when(
        self, name_or_symbol: str | SymbolRef, is_value: Any, then_expr: str
    ) -> ConstraintSpec:
        self._require_active()
        cons = when(self._symbol_name(name_or_symbol), is_value, then_expr)
        self._constraints.append(cons)
        return cons

    def chance_constraint(
        self, expr: str, *, threshold: float, confidence: float
    ) -> ChanceConstraintSpec:
        self._require_active()
        cons = chance_constraint(expr, threshold=threshold, confidence=confidence)
        self._chance_constraints.append(cons)
        return cons

    def build(self) -> ProgramSpec:
        return ProgramSpec(
            module=self.module,
            evaluation_set=self.evaluation_set,
            tvars=list(self._tvars.values()),
            assignments=dict(self._assignments),
            objectives=list(self._objectives),
            constraints=list(self._constraints),
            chance_constraints=list(self._chance_constraints),
        )

    def _resolve_runtime_config(self) -> dict[str, Any] | None:
        config = self._runtime_config.get()
        if config is not None:
            return config
        if self._frozen_config is not None:
            return self._frozen_config
        try:
            import traigent

            external = traigent.get_config()
            if isinstance(external, dict):
                return external
        except (ImportError, AttributeError):
            pass
        return None

    @contextmanager
    def using(self, config: Mapping[str, Any]) -> Iterator[ProgramBuilder]:
        """Temporarily bind tuned-variable values for direct function execution."""
        token = self._runtime_config.set(dict(config))
        try:
            yield self
        finally:
            self._runtime_config.reset(token)

    def freeze(self, config: Mapping[str, Any]) -> ProgramBuilder:
        """Persistently bind tuned-variable values (e.g., best-known config)."""
        self._frozen_config = dict(config)
        return self

    def unfreeze(self) -> ProgramBuilder:
        """Clear persistent tuned-variable bindings."""
        self._frozen_config = None
        return self

    def optimize(
        self,
        fn: Callable[..., Any] | None = None,
        **override_kwargs: Any,
    ) -> Callable[..., Any]:
        """Traigent-backed decorator built from this builder's ProgramSpec."""
        import traigent

        from .compiler import compile_opal_spec

        artifact = compile_opal_spec(self.build())
        kwargs = artifact.to_optimize_kwargs()
        kwargs.update(override_kwargs)
        decorator = traigent.optimize(**kwargs)
        if fn is None:
            return decorator
        return decorator(fn)

    def deploy(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        config: Mapping[str, Any],
    ) -> Callable[..., Any]:
        """Decorator or wrapper to run with a fixed config without optimization."""

        def _decorate(func: Callable[..., Any]) -> Callable[..., Any]:
            @wraps(func)
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                with self.using(config):
                    return func(*args, **kwargs)

            return _wrapped

        if fn is None:
            return _decorate
        return _decorate(fn)


def _normalize_domain(domain: Any) -> DomainChoices | DomainRange:
    if isinstance(domain, (DomainChoices, DomainRange)):
        return domain
    if isinstance(domain, set):
        raise ValueError(
            "Set domains are unordered. Use list/tuple to keep deterministic ordering."
        )

    try:
        from traigent.api.parameter_ranges import Choices as CoreChoices
        from traigent.api.parameter_ranges import IntRange as CoreIntRange
        from traigent.api.parameter_ranges import LogRange as CoreLogRange
        from traigent.api.parameter_ranges import Range as CoreRange
    except ImportError:
        CoreChoices = CoreIntRange = CoreLogRange = CoreRange = None  # type: ignore[assignment]

    if CoreChoices is not None and isinstance(domain, CoreChoices):
        return DomainChoices(values=tuple(domain.values))
    if CoreRange is not None and isinstance(domain, CoreRange):
        return DomainRange(
            lo=float(domain.low),
            hi=float(domain.high),
            step=float(domain.step) if domain.step is not None else None,
        )
    if CoreIntRange is not None and isinstance(domain, CoreIntRange):
        return DomainRange(
            lo=float(domain.low),
            hi=float(domain.high),
            step=float(domain.step) if domain.step is not None else None,
        )
    if CoreLogRange is not None and isinstance(domain, CoreLogRange):
        # LogRange currently lowers to [lo, hi] in this plugin surface.
        return DomainRange(lo=float(domain.low), hi=float(domain.high), step=None)

    raise TypeError(
        "Unsupported domain type. Use DomainChoices/DomainRange or Traigent Choices/Range/IntRange/LogRange."
    )


def choices(*values: Any) -> DomainChoices:
    """Create a finite domain."""
    if len(values) == 1 and isinstance(values[0], set):
        raise ValueError(
            "Set domains are unordered. Use list/tuple for deterministic ordering."
        )
    if len(values) == 1 and isinstance(values[0], (list, tuple)):
        vals = tuple(values[0])
    else:
        vals = tuple(values)
    return DomainChoices(values=vals)


def frange(lo: float, hi: float, step: float | None = None) -> DomainRange:
    """Create a numeric range domain."""
    return DomainRange(
        lo=float(lo),
        hi=float(hi),
        step=float(step) if step is not None else None,
    )


def tv(domain: Any, *, given: str | None = None) -> TunedVariable:
    """Create an anonymous tuned variable for `v.<name> in tv(...)` declarations."""
    return TunedVariable(domain=_normalize_domain(domain), given=given)


def opal_program(
    module: str | None = None,
    evaluation_set: str | None = None,
) -> ProgramBuilder:
    """Create a scoped OPAL declaration builder."""
    return ProgramBuilder(module=module, evaluation_set=evaluation_set)


def callable_template(name: str, **kwargs: Any) -> CallableTemplate:
    """Create an executable callable domain member."""
    return CallableTemplate(name=name, args=dict(kwargs))


def tvar(name: str, domain: Any, *, given: str | None = None) -> TVarSpec:
    """Create a tuned-variable declaration."""
    return TVarSpec(name=name, domain=_normalize_domain(domain), given=given)


def maximize(metric: str, on: str | None = None, *, weight: float = 1.0) -> ObjectiveSpec:
    """Create a maximize objective."""
    return ObjectiveSpec(direction="maximize", metric=metric, on=on, weight=weight)


def minimize(metric: str, on: str | None = None, *, weight: float = 1.0) -> ObjectiveSpec:
    """Create a minimize objective."""
    return ObjectiveSpec(direction="minimize", metric=metric, on=on, weight=weight)


def constraint(expr: str) -> ConstraintSpec:
    """Create a plain constraint expression."""
    return ConstraintSpec(expr=expr)


def when(name: str, is_value: Any, then_expr: str) -> ConstraintSpec:
    """Create a structural guard constraint."""
    return ConstraintSpec(when_name=name, is_value=is_value, then_expr=then_expr)


def chance_constraint(
    expr: str, threshold: float, confidence: float
) -> ChanceConstraintSpec:
    """Create a chance-constraint declaration."""
    return ChanceConstraintSpec(expr=expr, threshold=threshold, confidence=confidence)


def program(
    *,
    module: str | None = None,
    evaluation_set: str | None = None,
    tvars: list[TVarSpec] | None = None,
    assignments: dict[str, Any] | None = None,
    objectives: list[ObjectiveSpec] | None = None,
    constraints: list[ConstraintSpec] | None = None,
    chance_constraints: list[ChanceConstraintSpec] | None = None,
) -> ProgramSpec:
    """Create a ProgramSpec with validated defaults."""
    return ProgramSpec(
        module=module,
        evaluation_set=evaluation_set,
        tvars=tvars or [],
        assignments=assignments or {},
        objectives=objectives or [],
        constraints=constraints or [],
        chance_constraints=chance_constraints or [],
    )
