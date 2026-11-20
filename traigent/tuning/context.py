"""Module context: the TVL blocks the SDK derives from an ``@traigent.optimize`` function."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

DEFAULT_SNAPSHOT_ID = "1970-01-01T00:00:00Z"
_MINIMIZE_HINTS = ("cost", "latency", "error", "time", "loss", "token", "tokens")


def _tvar(name: str, values: Any) -> dict[str, Any] | None:
    if (
        isinstance(values, (list, tuple))
        and len(values) == 2
        and isinstance(values, tuple)
    ):
        lo, hi = values
        if all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in (lo, hi)
        ):
            kind = "int" if isinstance(lo, int) and isinstance(hi, int) else "float"
            return {"name": name, "type": kind, "domain": {"range": [lo, hi]}}
    if isinstance(values, (list, tuple)) and values:
        if all(isinstance(v, str) for v in values):
            kind = "str"
        elif all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            kind = "int"
        elif all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
        ):
            kind = "float"
        else:
            return None
        return {"name": name, "type": f"enum[{kind}]", "domain": list(values)}
    # ParameterRange objects (Choices/Range/IntRange/LogRange): reuse ConfigSpace's TVL export.
    try:
        from traigent.api.config_space import ConfigSpace

        spec = ConfigSpace(tvars={name: values}).to_tvl_spec()["tvars"][0]
        spec.pop("x_traigent_parameter_range", None)
        return dict(spec)
    except Exception:  # noqa: BLE001 - an unsupported tvar is left out; TVL lint reports the gap
        return None


def _objective(item: Any) -> dict[str, str] | None:
    if isinstance(item, str):
        name = item
        direction = (
            "minimize"
            if any(h in name.lower() for h in _MINIMIZE_HINTS)
            else "maximize"
        )
        return {"name": name, "direction": direction}
    attr_name = getattr(item, "name", None)
    if not isinstance(attr_name, str):
        return None
    orientation = str(getattr(item, "orientation", "") or "").lower()
    if orientation in ("minimize", "maximize"):
        return {"name": attr_name, "direction": orientation}
    return _objective(attr_name)


def derive_module_context(
    target: Any,
    *,
    module_name: str | None = None,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    dataset: str | None = None,
) -> dict[str, Any] | None:
    """Module blocks for an optimized function, or ``None`` when it is not one.

    The result holds ``tvl``, ``environment``, ``evaluation_set``, ``tvars``,
    ``objectives`` and ``promotion_policy`` (the defaults RFC 0011 slice uses),
    never the runtime blocks.  Placeholders (snapshot id, dataset) are
    deterministic and overridable.
    """
    space = getattr(target, "configuration_space", None)
    if not isinstance(space, Mapping) or not space:
        return None
    tvars = [t for t in (_tvar(str(k), v) for k, v in space.items()) if t is not None]
    objectives = [
        o
        for o in (
            _objective(i) for i in (getattr(target, "objectives", None) or ["accuracy"])
        )
        if o is not None
    ]
    eval_dataset = dataset or getattr(target, "eval_dataset", None)
    if not isinstance(eval_dataset, str) or not eval_dataset:
        eval_dataset = "evaluation_set"
    name = module_name or getattr(target, "__name__", None) or "agent"
    return {
        "tvl": {"module": f"traigent.{name}"},
        "environment": {"snapshot_id": snapshot_id},
        "evaluation_set": {"dataset": eval_dataset},
        "tvars": tvars,
        "objectives": objectives,
        "promotion_policy": {
            "dominance": "epsilon_pareto",
            "alpha": 0.05,
            "min_effect": {o["name"]: 0.02 for o in objectives},
        },
    }


def merge_context(
    base: Mapping[str, Any] | None, override: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    if base is None and override is None:
        return None
    out = copy.deepcopy(dict(base or {}))
    out.update(copy.deepcopy(dict(override or {})))
    return out
