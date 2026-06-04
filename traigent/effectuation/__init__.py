"""Opt-in executable effectuation for declared tuned variables."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from traigent.effectuation.contracts import (
    STRATEGY_REGISTRY,
    CompiledEffect,
    KnobKind,
    KnobStrategy,
    get_strategy,
    knob_field,
    knob_kind,
    knob_name,
    register_strategy,
)
from traigent.effectuation.strategies.framework_param import FrameworkParamStrategy
from traigent.effectuation.strategies.self_consistency import (
    SELF_CONSISTENCY_NAMES,
    SelfConsistencyStrategy,
)

register_strategy(FrameworkParamStrategy())
register_strategy(SelfConsistencyStrategy())

EFFECTUATION_EVENTS_ATTR = "__traigent_effectuation_emit_events__"


@dataclass(frozen=True)
class EffectuationApplication:
    """Compiled effectuation result for one callable/config declaration."""

    wrapped_callable: Callable[..., Any]
    effects: tuple[CompiledEffect, ...] = ()

    def emit_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for effect in self.effects:
            events.extend(effect.emit_events())
        return events


def apply_effectuation(
    fn: Callable[..., Any],
    config: Any,
    *,
    enabled: bool = False,
) -> Callable[..., Any]:
    """Compose registered executable strategies for declared knobs.

    Effectuation is opt-in. With the default ``enabled=False``, the original
    callable is returned unchanged.
    """

    return compile_effectuation(fn, config, enabled=enabled).wrapped_callable


def compile_effectuation(
    fn: Callable[..., Any],
    config: Any,
    *,
    enabled: bool = False,
) -> EffectuationApplication:
    """Compose registered executable strategies and retain event emitters."""

    if not enabled:
        return EffectuationApplication(wrapped_callable=fn)

    wrapped = fn
    effects: list[CompiledEffect] = []
    for raw_knob in _iter_declared_knobs(config):
        strategy_id = _strategy_id_for_knob(raw_knob)
        if strategy_id is None:
            continue
        knob = _with_inferred_kind(raw_knob, strategy_id)
        strategy = get_strategy(strategy_id)
        strategy.validate(knob, runtime_ctx=config)
        effect = strategy.compile(knob)
        effects.append(effect)
        wrapped = effect.wrap_callable(wrapped, plan=config)

    application = EffectuationApplication(
        wrapped_callable=wrapped,
        effects=tuple(effects),
    )
    if effects:
        try:
            setattr(wrapped, EFFECTUATION_EVENTS_ATTR, application.emit_events)
        except Exception:
            pass
    return application


def _iter_declared_knobs(config: Any) -> list[Any]:
    if config is None:
        return []

    if isinstance(config, Mapping):
        if _looks_like_single_knob(config):
            return [dict(config)]

        section_knobs: list[Any] = []
        for section_key in (
            "knobs",
            "tvars",
            "recommendations",
            "configuration_space",
        ):
            if section_key in config and config[section_key] is not None:
                section_knobs.extend(_iter_declared_knobs(config[section_key]))
        if section_knobs:
            return section_knobs

        return [
            knob
            for name, value in config.items()
            if (knob := _knob_from_flat_entry(name, value)) is not None
        ]

    if isinstance(config, Sequence) and not isinstance(config, (str, bytes)):
        knobs: list[Any] = []
        for item in config:
            knobs.extend(_iter_declared_knobs(item))
        return knobs

    object_knobs: list[Any] = []
    for attr_name in ("knobs", "tvars", "recommendations", "configuration_space"):
        if hasattr(config, attr_name):
            object_knobs.extend(_iter_declared_knobs(getattr(config, attr_name)))
    if object_knobs:
        return object_knobs

    if hasattr(config, "name"):
        return [config]

    return []


def _looks_like_single_knob(value: Mapping[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "name",
            "kind",
            "effectuation_strategy",
            "strategy_id",
            "range_type",
            "range_kwargs",
        )
    )


def _knob_from_flat_entry(name: Any, value: Any) -> dict[str, Any] | None:
    if not isinstance(name, str) or name.startswith("_"):
        return None

    strategy_id = _strategy_id_for_name(name)
    if strategy_id is None:
        return None

    kind = (
        KnobKind.CARDINALITY.value
        if strategy_id == "self_consistency"
        else KnobKind.VALUE.value
    )
    return {
        "name": name,
        "kind": kind,
        "value_set": value,
        "effectuation_strategy": strategy_id,
    }


def _strategy_id_for_knob(knob: Any) -> str | None:
    explicit_strategy = knob_field(knob, "effectuation_strategy") or knob_field(
        knob, "strategy_id"
    )
    if explicit_strategy:
        return str(explicit_strategy)

    try:
        name = knob_name(knob)
    except ValueError:
        return None

    return _strategy_id_for_name(name, knob)


def _strategy_id_for_name(name: str, knob: Any | None = None) -> str | None:
    if name in SELF_CONSISTENCY_NAMES:
        return "self_consistency"

    try:
        kind = knob_kind(knob, default=None) if knob is not None else None
    except ValueError:
        return None

    if name in _framework_parameter_names():
        return "framework_param"
    if kind == KnobKind.VALUE.value and knob_field(knob, "effectuation_strategy"):
        return "framework_param"
    return None


def _with_inferred_kind(knob: Any, strategy_id: str) -> Any:
    if not isinstance(knob, Mapping) or "kind" in knob:
        return knob

    inferred_kind = (
        KnobKind.CARDINALITY.value
        if strategy_id == "self_consistency"
        else KnobKind.VALUE.value
    )
    updated = dict(knob)
    updated["kind"] = inferred_kind
    return updated


@lru_cache(maxsize=1)
def _framework_parameter_names() -> frozenset[str]:
    from traigent.integrations.mappings import PARAMETER_MAPPINGS

    names: set[str] = set()
    for mapping in PARAMETER_MAPPINGS.values():
        names.update(mapping)
    return frozenset(names)


__all__ = [
    "CompiledEffect",
    "EffectuationApplication",
    "EFFECTUATION_EVENTS_ATTR",
    "KnobKind",
    "KnobStrategy",
    "STRATEGY_REGISTRY",
    "apply_effectuation",
    "compile_effectuation",
    "get_strategy",
    "register_strategy",
]
