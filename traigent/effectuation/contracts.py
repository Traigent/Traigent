"""Contracts and registry for executable tuned-variable effectuation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class KnobKind(StrEnum):
    """Supported tuned-variable effectuation kinds."""

    VALUE = "value"
    CARDINALITY = "cardinality"
    TOPOLOGY = "topology"
    POLICY = "policy"


VALID_KNOB_KINDS = frozenset(kind.value for kind in KnobKind)


@runtime_checkable
class CompiledEffect(Protocol):
    """Executable projection for one declared knob."""

    def project_for_optimizer(self) -> dict[str, Any]:
        """Return the flattened value set handed to the optimizer."""
        ...

    def wrap_callable(self, fn: Callable[..., Any], plan: Any) -> Callable[..., Any]:
        """Return a callable with this effect applied, or ``fn`` unchanged."""
        ...

    def emit_events(self) -> list[dict[str, Any]]:
        """Return aggregate, content-free effectuation events."""
        ...


@runtime_checkable
class KnobStrategy(Protocol):
    """Strategy that validates and compiles a declared knob."""

    strategy_id: str
    supported_kinds: set[str]

    def validate(self, knob: Any, runtime_ctx: Any) -> None:
        """Validate that ``knob`` can be compiled by this strategy."""
        ...

    def compile(self, knob: Any) -> CompiledEffect:
        """Compile ``knob`` into an executable effect."""
        ...


STRATEGY_REGISTRY: dict[str, KnobStrategy] = {}


def register_strategy(strategy: KnobStrategy) -> KnobStrategy:
    """Register a knob strategy by its stable strategy id."""

    strategy_id = getattr(strategy, "strategy_id", "")
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise ValueError("Knob strategy must define a non-empty strategy_id")

    supported_kinds = set(getattr(strategy, "supported_kinds", set()))
    invalid_kinds = supported_kinds - VALID_KNOB_KINDS
    if invalid_kinds:
        raise ValueError(
            f"Knob strategy {strategy_id!r} declares unsupported kinds: "
            f"{sorted(invalid_kinds)}"
        )

    STRATEGY_REGISTRY[strategy_id] = strategy
    return strategy


def get_strategy(strategy_id: str) -> KnobStrategy:
    """Resolve a registered knob strategy by id."""

    try:
        return STRATEGY_REGISTRY[strategy_id]
    except KeyError as exc:
        raise KeyError(f"No knob strategy registered for {strategy_id!r}") from exc


def knob_field(knob: Any, field_name: str, default: Any = None) -> Any:
    """Read a field from a dict-like or object-like knob declaration."""

    if isinstance(knob, Mapping):
        return knob.get(field_name, default)
    return getattr(knob, field_name, default)


def knob_name(knob: Any) -> str:
    """Return a knob name or raise a clear validation error."""

    name = knob_field(knob, "name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Knob declaration must include a non-empty name")
    return name


def knob_kind(knob: Any, default: str | None = None) -> str | None:
    """Return a normalized knob kind when present."""

    kind = knob_field(knob, "kind", default)
    if kind is None:
        return None
    kind_value = str(kind)
    if kind_value not in VALID_KNOB_KINDS:
        raise ValueError(f"Unsupported knob kind: {kind_value!r}")
    return kind_value


def project_knob_for_optimizer(knob: Any) -> dict[str, Any]:
    """Project a knob declaration to today's flattened optimizer value set."""

    name = knob_name(knob)

    if isinstance(knob, Mapping):
        for value_key in ("value_set", "config_value", "search_space", "range"):
            if value_key in knob:
                return {name: _normalize_config_value(knob[value_key])}
        if "values" in knob:
            return {name: list(knob["values"])}
        if "range_type" in knob and "range_kwargs" in knob:
            return {
                name: _project_catalog_range(
                    str(knob["range_type"]),
                    knob.get("range_kwargs", {}),
                )
            }
        if "value" in knob:
            return {name: knob["value"]}

    if hasattr(knob, "to_config_value"):
        return {name: knob.to_config_value()}

    raise ValueError(f"Knob {name!r} does not include an optimizer value set")


def _normalize_config_value(value: Any) -> Any:
    try:
        from traigent.api.parameter_ranges import normalize_config_value

        return normalize_config_value(value)
    except Exception:
        return value


def _project_catalog_range(range_type: str, range_kwargs: Any) -> Any:
    if not isinstance(range_kwargs, Mapping):
        raise ValueError("Catalog range_kwargs must be a mapping")

    kwargs = dict(range_kwargs)
    if range_type == "Choices":
        values = kwargs.get("values")
        if not isinstance(values, list):
            raise ValueError("Choices catalog range must include a values list")
        return list(values)

    if range_type == "Range":
        return _project_numeric_range(kwargs, type_name="float")
    if range_type == "IntRange":
        return _project_numeric_range(kwargs, type_name="int")
    if range_type == "LogRange":
        projected = _project_numeric_range(kwargs, type_name="float")
        if isinstance(projected, tuple):
            return {
                "type": "float",
                "low": projected[0],
                "high": projected[1],
                "log": True,
            }
        projected["log"] = True
        return projected

    raise ValueError(f"Unsupported catalog range_type: {range_type!r}")


def _project_numeric_range(kwargs: dict[str, Any], *, type_name: str) -> Any:
    if "low" not in kwargs or "high" not in kwargs:
        raise ValueError("Numeric catalog range must include low and high")

    if kwargs.get("step") is None and not kwargs.get("log"):
        return (kwargs["low"], kwargs["high"])

    projected = {"type": type_name, "low": kwargs["low"], "high": kwargs["high"]}
    if kwargs.get("step") is not None:
        projected["step"] = kwargs["step"]
    if kwargs.get("log"):
        projected["log"] = True
    return projected


__all__ = [
    "CompiledEffect",
    "KnobKind",
    "KnobStrategy",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "knob_field",
    "knob_kind",
    "knob_name",
    "project_knob_for_optimizer",
    "register_strategy",
]
