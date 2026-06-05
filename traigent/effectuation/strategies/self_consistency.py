"""Self-consistency effectuation strategy."""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

from traigent.config.context import get_config
from traigent.effectuation.contracts import (
    KnobKind,
    knob_field,
    knob_kind,
    knob_name,
    project_knob_for_optimizer,
)

SELF_CONSISTENCY_NAMES = frozenset({"candidate_count", "self_consistency_k"})
_WRAPPED_MARKER = "__traigent_effectuation_self_consistency_wrapped__"


def majority_vote(outputs: Sequence[Any]) -> Any:
    """Return the modal output using normalized string equality."""

    if not outputs:
        return None

    normalized = [_normalize_output(output) for output in outputs]
    counts = Counter(normalized)
    first_index = {value: normalized.index(value) for value in counts}
    selected = max(counts, key=lambda value: (counts[value], -first_index[value]))
    return outputs[first_index[selected]]


@dataclass
class SelfConsistencyEffect:
    """Compiled effect that executes the wrapped callable N times."""

    knob: Any
    aggregator: Callable[[Sequence[Any]], Any] = majority_vote
    _events: list[dict[str, Any]] = field(default_factory=list, init=False)

    @property
    def knob_name(self) -> str:
        return knob_name(self.knob)

    def project_for_optimizer(self) -> dict[str, Any]:
        return project_knob_for_optimizer(self.knob)

    def wrap_callable(self, fn: Callable[..., Any], plan: Any) -> Callable[..., Any]:
        if getattr(fn, _WRAPPED_MARKER, False):
            return fn

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                n = self._runtime_n()
                if n <= 1:
                    output = await fn(*args, **kwargs)
                    self._record_event(n=n, outputs=[output], passthrough=True)
                    return output

                outputs = []
                for _ in range(n):
                    outputs.append(await fn(*args, **kwargs))
                result = self.aggregator(outputs)
                self._record_event(n=n, outputs=outputs, passthrough=False)
                return result

            setattr(async_wrapped, _WRAPPED_MARKER, True)
            return async_wrapped

        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            n = self._runtime_n()
            if n <= 1:
                output = fn(*args, **kwargs)
                self._record_event(n=n, outputs=[output], passthrough=True)
                return output

            outputs = [fn(*args, **kwargs) for _ in range(n)]
            result = self.aggregator(outputs)
            self._record_event(n=n, outputs=outputs, passthrough=False)
            return result

        setattr(wrapped, _WRAPPED_MARKER, True)
        return wrapped

    def emit_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def _runtime_n(self) -> int:
        cfg = get_config()
        raw_value = _config_get(cfg, self.knob_name)
        if raw_value is None:
            raw_value = _config_get(cfg, _alternate_name(self.knob_name))
        if raw_value is None:
            return 1
        return _coerce_count(raw_value)

    def _record_event(
        self,
        *,
        n: int,
        outputs: Sequence[Any],
        passthrough: bool,
    ) -> None:
        self._events = [
            {
                "strategy": "self_consistency",
                "knob": self.knob_name,
                "n": n,
                "calls": len(outputs),
                "unique_outputs": len(
                    {_normalize_output(output) for output in outputs}
                ),
                "aggregator": _aggregator_label(self.aggregator),
                "passthrough": passthrough,
            }
        ]


class SelfConsistencyStrategy:
    """Strategy for candidate-count/self-consistency cardinality knobs."""

    strategy_id = "self_consistency"
    supported_kinds = {KnobKind.CARDINALITY.value}

    def validate(self, knob: Any, runtime_ctx: Any) -> None:
        name = knob_name(knob)
        if name not in SELF_CONSISTENCY_NAMES:
            raise ValueError(
                f"{self.strategy_id} only supports {sorted(SELF_CONSISTENCY_NAMES)}, "
                f"got {name!r}"
            )

        kind = knob_kind(knob, default=KnobKind.CARDINALITY.value)
        if kind != KnobKind.CARDINALITY.value:
            raise ValueError(
                f"{self.strategy_id} only supports cardinality knobs, got {kind!r}"
            )

        aggregator = knob_field(knob, "aggregator")
        if aggregator is not None and not callable(aggregator):
            raise ValueError("self_consistency aggregator must be callable")

        project_knob_for_optimizer(knob)

    def compile(self, knob: Any) -> SelfConsistencyEffect:
        self.validate(knob, runtime_ctx=None)
        aggregator = knob_field(knob, "aggregator", majority_vote)
        return SelfConsistencyEffect(knob=knob, aggregator=aggregator)


def _normalize_output(output: Any) -> str:
    return str(output).strip().casefold()


def _config_get(config: Any, key: str | None) -> Any:
    if key is None:
        return None
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key)
    try:
        return config[key]
    except Exception:
        return None


def _alternate_name(name: str) -> str | None:
    if name == "candidate_count":
        return "self_consistency_k"
    if name == "self_consistency_k":
        return "candidate_count"
    return None


def _coerce_count(raw_value: Any) -> int:
    if isinstance(raw_value, bool):
        raise ValueError("self_consistency count must be an integer, not bool")
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float) and raw_value.is_integer():
        return int(raw_value)
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
    raise ValueError(f"self_consistency count must be an integer, got {raw_value!r}")


def _aggregator_label(aggregator: Callable[[Sequence[Any]], Any]) -> str:
    if aggregator is majority_vote:
        return "majority_vote"
    return getattr(aggregator, "__name__", "custom")


__all__ = [
    "SELF_CONSISTENCY_NAMES",
    "SelfConsistencyEffect",
    "SelfConsistencyStrategy",
    "majority_vote",
]
