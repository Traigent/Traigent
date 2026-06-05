"""Knob-kind taxonomy — reconciled with the effectuation module.

The canonical ``KnobKind`` lives in ``traigent.effectuation`` (the
tvar-executable-effectuation branch). Until that branch merges, this module
provides a values-identical fallback behind an import guard so the two
surfaces can never drift apart silently: ``tests/unit/knobs/test_kinds.py``
pins the member names and values, and once the effectuation module is
importable the guard resolves to it as the single source of truth.
"""

from __future__ import annotations

__all__ = ["KnobKind"]

try:  # pragma: no cover - exercised only once effectuation merges
    from traigent.effectuation.contracts import KnobKind  # type: ignore[no-redef]
except (ImportError, ModuleNotFoundError):  # pragma: no cover - tested path today
    from enum import StrEnum

    class KnobKind(StrEnum):  # type: ignore[no-redef]
        """What aspect of the agent a knob effectuates.

        Values mirror ``traigent.effectuation.contracts.KnobKind`` exactly.
        """

        VALUE = "value"
        CARDINALITY = "cardinality"
        TOPOLOGY = "topology"
        POLICY = "policy"
