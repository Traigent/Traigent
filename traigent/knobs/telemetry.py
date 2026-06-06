"""Composite telemetry → the measures wire channel (RFC 0002 §3.10).

The composite execution runtime (:mod:`traigent.knobs.runtime`) hands back a
:class:`~traigent.knobs.runtime.CompositeRunResult` whose ``measures`` is the
§3.10 *content-free* telemetry dict (counts, rates, enums, finite numbers only
— never anything output-derived). That dict is **structured**: it nests a
per-gate ``gate_margin_pass_rate`` map keyed by gate index.

The Traigent measures wire channel (the per-trial numeric metrics that ride a
trial submission — see :class:`traigent.cloud.dtos.MeasuresDict`) is **flat**:
≤ 50 entries, each key a Python identifier (``^[a-zA-Z_]\\w*$``), each value a
plain number. This module is the small, pure adapter between the two:

    >>> from traigent.knobs.runtime import execute_composite
    >>> from traigent.knobs.telemetry import composite_measures
    >>> run = execute_composite(knob, stages, config=cfg, calibrated_values=cv)
    >>> metrics = {"accuracy": score}
    >>> metrics.update(composite_measures(run))   # rides the same channel

:func:`composite_measures` flattens the run's measures into identifier-safe,
numeric-only keys so a user can merge them straight into the ``metrics`` dict
their evaluated function already returns. The per-gate map is flattened by
index (``composite_gate_0_margin_pass_rate``, ``composite_gate_1_...``). The
output is **content-free by construction**: the input is already content-free
per §3.10, and this adapter only ever copies finite numbers and emits integer
gate indices — it never reads ``run.output`` or any stage value.

This module adds NO new wire surface: it produces an ordinary numeric metrics
dict that the existing channel already validates and transports.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from traigent.utils.logging import get_logger

if TYPE_CHECKING:  # avoid importing the runtime module at import time
    from traigent.knobs.runtime import CompositeRunResult

__all__ = [
    "MAX_COMPOSITE_MEASURES",
    "composite_measures",
]

logger = get_logger(__name__)

#: Cap on the number of flattened composite keys, with headroom below the
#: backend ``MeasuresDict`` 50-key ceiling so a user's own metrics (accuracy,
#: cost, latency, ...) co-exist on the same trial without tripping the
#: cardinality guard. Truncation is deterministic and logged — never raised
#: mid-trial (a telemetry overflow must not fail the user's optimization).
MAX_COMPOSITE_MEASURES = 40


def _is_finite_number(value: object) -> bool:
    """True for a real, finite int/float (bool and non-finite excluded).

    ``bool`` is rejected for parity with the backend ``MeasuresDict`` contract
    (a bool is not a measure); NaN/±inf are dropped because the wire channel
    carries finite numbers only (§3.10).
    """
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def composite_measures(
    run: CompositeRunResult, *, prefix: str = "composite"
) -> dict[str, float | int]:
    """Flatten a composite run's §3.10 measures into MeasuresDict-safe keys.

    The result is a flat dict of identifier-safe keys → finite numbers, ready
    to merge into the ``metrics`` dict a decorated function returns (the keys
    then ride the existing measures wire channel; see the module docstring).

    Flattening rules:

    - a scalar measure ``m`` (e.g. ``escalation_rate``, ``stage_selected``)
      becomes ``{prefix}_{m}`` carrying the same finite number;
    - the per-gate ``gate_margin_pass_rate`` map (keyed by gate INDEX) is
      flattened per gate into ``{prefix}_gate_{index}_margin_pass_rate``;
    - any non-finite or non-numeric value is DROPPED (never coerced) — the
      channel carries finite numbers only (§3.10);
    - keys are emitted in a deterministic order (scalars in input order, then
      gate entries by ascending index) and the total is capped at
      :data:`MAX_COMPOSITE_MEASURES`. Over the cap, the lowest-priority keys
      are truncated deterministically and a warning is logged — this function
      NEVER raises (a telemetry overflow must not fail a trial).

    Args:
        run: the frozen :class:`~traigent.knobs.runtime.CompositeRunResult`.
            Only its ``measures`` is read — never ``output`` (content-free).
        prefix: the identifier-safe namespace for every emitted key (default
            ``"composite"``). A non-identifier prefix raises ``ValueError`` —
            this is a programming error at call time, not a per-trial fault.

    Returns:
        A flat ``dict[str, float | int]`` of content-free composite measures.
    """
    if not prefix.isidentifier():
        raise ValueError(
            f"composite_measures prefix {prefix!r} must be a Python identifier "
            r"(^[a-zA-Z_]\w*$) so emitted keys satisfy the MeasuresDict contract"
        )

    measures: Mapping[str, object] = run.measures or {}

    # Build (key, value) pairs in a deterministic, priority order:
    #   1. scalar measures in input order (these are the per-run summaries);
    #   2. per-gate entries by ascending gate index.
    ordered: list[tuple[str, float | int]] = []
    gate_entries: list[tuple[int, str, float | int]] = []

    for name, value in measures.items():
        if isinstance(value, Mapping):
            # The §3.10 per-gate map (gate_margin_pass_rate): index -> rate.
            # The map's own name already carries the `gate_` prefix; strip it so
            # the flattened key reads `..._gate_{index}_margin_pass_rate`, not a
            # doubled `..._gate_{index}_gate_margin_pass_rate`.
            suffix = name.removeprefix("gate_")
            for raw_index, rate in value.items():
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    # An index that is not integer-coercible is not a §3.10
                    # gate key; drop it rather than emit a bogus name.
                    continue
                if index < 0 or not _is_finite_number(rate):
                    continue
                key = f"{prefix}_gate_{index}_{suffix}"
                gate_entries.append((index, key, rate))  # type: ignore[arg-type]
            continue
        if _is_finite_number(value):
            ordered.append((f"{prefix}_{name}", value))  # type: ignore[arg-type]

    gate_entries.sort(key=lambda item: item[0])
    ordered.extend((key, rate) for _index, key, rate in gate_entries)

    if len(ordered) > MAX_COMPOSITE_MEASURES:
        dropped = [key for key, _ in ordered[MAX_COMPOSITE_MEASURES:]]
        logger.warning(
            "composite_measures: %d composite measure key(s) exceed the "
            "%d-key cap; truncating deterministically. Dropped: %s",
            len(ordered),
            MAX_COMPOSITE_MEASURES,
            sorted(dropped),
        )
        ordered = ordered[:MAX_COMPOSITE_MEASURES]

    return dict(ordered)
