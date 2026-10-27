"""Runtime requirements and reactive policies (PY-1: declare, apply, watch, respond).

Run one controller per agent: occurrences are reserved in a local ledger, so
cross-controller deduplication is not enforced until the backend reserves them.
"""

from __future__ import annotations

from traigent.runtime.decorator import RequirementList, requirement
from traigent.runtime.guards import LocalGuardError, LocalGuardSet, guarded
from traigent.runtime.model import (
    Issue,
    Receipt,
    RuntimeEvent,
    RuntimeRequirement,
    RuntimeRequirementError,
)

__all__ = [
    "Issue",
    "LocalGuardError",
    "LocalGuardSet",
    "Receipt",
    "RequirementList",
    "RuntimeEvent",
    "RuntimeRequirement",
    "RuntimeRequirementError",
    "guarded",
    "requirement",
]
