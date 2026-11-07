"""Crew declarations from code: validation, wire shape and the closed binding result.

``@traigent.optimize(crew=..., role=...)`` declares which crew and role an agent plays.
The crew and role keys are CUSTOMER CONTENT: they travel only as
``metadata.crew_declaration`` of the session-create request and never appear in
spans, logs, telemetry, local result files or error messages. Every error raised here
is a fixed string that does not repeat the supplied values.

The backend answers with a closed ``crew_binding`` (status plus identifiers, no names).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: Same rules the backend enforces on a declaration.
CREW_KEY_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")
ROLE_KEY_MAX_LENGTH = 128

DECLARATION_METADATA_KEY = "crew_declaration"

STATUS_BOUND = "bound"
STATUS_PENDING_PORTAL = "pending_portal"
STATUS_CONFLICT = "conflict"
STATUS_DISABLED = "disabled"
STATUS_UNKNOWN = "unknown"
BINDING_STATUSES = frozenset(
    {
        STATUS_BOUND,
        STATUS_PENDING_PORTAL,
        STATUS_CONFLICT,
        STATUS_DISABLED,
        STATUS_UNKNOWN,
    }
)

CONFLICT_WARNING = (
    "Crew declaration conflicts with portal membership; portal membership was retained."
)


@dataclass(frozen=True)
class CrewDeclaration:
    """A validated declaration. Both fields are content: the repr never shows them."""

    crew_key: str
    role_key: str

    def __repr__(self) -> str:
        return "CrewDeclaration(<content>)"

    def to_wire(self) -> dict[str, str]:
        return {"crew_key": self.crew_key, "role_key": self.role_key}


@dataclass(frozen=True)
class CrewBinding:
    """The backend's closed answer: a status and identifiers only, never names."""

    status: str = STATUS_UNKNOWN
    crew_id: str | None = None
    role_id: str | None = None
    role_token: str | None = None
    crew_revision_id: str | None = None
    revision_no: int | None = None


UNKNOWN_BINDING = CrewBinding()


def validate_crew_declaration(crew: Any, role: Any) -> CrewDeclaration | None:
    """Validate the decorator's ``crew`` / ``role`` pair.

    Returns None when neither is given. Raises ``ValueError`` with a fixed message
    (never echoing a value) for a half pair or a key the backend would refuse.
    """
    if crew is None and role is None:
        return None
    if crew is None or role is None:
        raise ValueError("crew and role must be declared together (both or neither).")
    if not isinstance(crew, str) or CREW_KEY_PATTERN.fullmatch(crew) is None:
        raise ValueError(
            "crew must be a string of 1-128 characters from A-Z, a-z, 0-9, '_' and '-'."
        )
    if not isinstance(role, str) or not 1 <= len(role) <= ROLE_KEY_MAX_LENGTH:
        raise ValueError("role must be a string of 1-128 characters.")
    return CrewDeclaration(crew_key=crew, role_key=role)


def declaration_of(request: Any) -> CrewDeclaration | None:
    value = getattr(request, "crew_declaration", None)
    return value if isinstance(value, CrewDeclaration) else None


def crew_declaration_to_wire(request: Any) -> dict[str, dict[str, str]]:
    """``{"crew_declaration": {...}}`` to merge into the request METADATA, or ``{}``."""
    declaration = declaration_of(request)
    if declaration is None:
        return {}
    return {DECLARATION_METADATA_KEY: declaration.to_wire()}


def parse_crew_binding(raw: Any) -> CrewBinding:
    """Read the response ``crew_binding``; anything missing or unrecognised is ``unknown``."""
    if not isinstance(raw, dict):
        return UNKNOWN_BINDING
    status = raw.get("status")
    if not isinstance(status, str) or status not in BINDING_STATUSES:
        return UNKNOWN_BINDING

    def text(name: str) -> str | None:
        value = raw.get(name)
        return value if isinstance(value, str) and value else None

    revision_no = raw.get("revision_no")
    if isinstance(revision_no, bool) or not isinstance(revision_no, int):
        revision_no = None
    return CrewBinding(
        status=status,
        crew_id=text("crew_id"),
        role_id=text("role_id"),
        role_token=text("role_token"),
        crew_revision_id=text("crew_revision_id"),
        revision_no=revision_no,
    )
