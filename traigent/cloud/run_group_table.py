"""One table across every run in a run group (Traigent#1711, slice 1).

Pure projection of the backend's grouped configuration-run rows into a flat,
analysis-friendly table. Rows stay individual source runs: nothing is merged or
deduped by configuration hash. An unmeasured cost is ``None``, never ``0``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from traigent.cloud.dtos import GroupedConfigurationRunRowDTO
from traigent.utils.trial_costs import coerce_finite_cost, is_cost_unranked

__all__ = ["RunGroupRow", "RunGroupTable", "build_run_group_row"]

_COST_KEYS = ("total_cost", "cost")
_NAME_KEYS = ("run_name", "experiment_run_name", "name")
#: Measure keys that describe cost bookkeeping rather than a quality metric.
_NON_METRIC_KEYS = frozenset(
    {
        "cost",
        "total_cost",
        "input_cost",
        "output_cost",
        "cost_unmeasured",
        "cost_unpriced",
    }
)
_COLUMNS = (
    "run_id",
    "run_name",
    "configuration_run_id",
    "experiment_id",
    "config",
    "metrics",
    "cost",
    "cost_unmeasured",
    "cost_floor",
    "status",
    "created_at",
)


@dataclass(frozen=True)
class RunGroupRow:
    """One configuration-run in a run group."""

    run_id: str
    run_name: str | None
    configuration_run_id: str
    experiment_id: str
    config: dict[str, Any]
    metrics: dict[str, float]
    #: Measured cost; ``None`` when unmeasured or unpriced (never ``0``).
    cost: float | None
    cost_unmeasured: bool
    #: Known-spend floor; always ``None`` until the backend emits one.
    cost_floor: float | None
    status: str | None
    created_at: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "configuration_run_id": self.configuration_run_id,
            "experiment_id": self.experiment_id,
            "config": dict(self.config),
            "metrics": dict(self.metrics),
            "cost": self.cost,
            "cost_unmeasured": self.cost_unmeasured,
            "cost_floor": self.cost_floor,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class RunGroupTable:
    """Rows across all runs of one run group, in backend order."""

    group_id: str
    rows: list[RunGroupRow] = field(default_factory=list)
    #: ``True`` when ``max_rows`` stopped the fetch before the last row.
    truncated: bool = False

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[RunGroupRow]:
        return iter(self.rows)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [row.to_dict() for row in self.rows]

    def to_dataframe(self) -> Any:
        """Return a pandas DataFrame (needs the optional ``pandas`` extra).

        ``config`` and ``metrics`` columns hold dicts; an unmeasured cost is
        missing (``None``/``NaN``) in ``cost``, never ``0``.
        """
        from traigent.api.types import _require_pandas

        pd = _require_pandas()
        return pd.DataFrame(self.to_dicts(), columns=list(_COLUMNS))


def _first_str(
    sources: Iterable[Mapping[str, Any]], keys: tuple[str, ...]
) -> str | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def build_run_group_row(row: GroupedConfigurationRunRowDTO) -> RunGroupRow:
    """Project one backend row into a :class:`RunGroupRow`."""
    measures = row.measures
    unmeasured = is_cost_unranked(measures)
    cost: float | None = None
    if not unmeasured:
        for key in _COST_KEYS:
            cost = coerce_finite_cost(measures.get(key))
            if cost is not None:
                break
    floor = coerce_finite_cost(measures.get("cost_lower_bound")) if unmeasured else None
    metrics = {
        key: float(value)
        for key, value in measures.items()
        if key not in _NON_METRIC_KEYS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }
    return RunGroupRow(
        run_id=row.experiment_run_id,
        run_name=_first_str((row.extra_fields, row.metadata), _NAME_KEYS),
        configuration_run_id=row.configuration_run_id,
        experiment_id=row.experiment_id,
        config=dict(row.configuration),
        metrics=metrics,
        cost=cost,
        cost_unmeasured=unmeasured,
        cost_floor=floor,
        status=row.status,
        created_at=row.created_at,
    )
