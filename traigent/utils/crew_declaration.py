"""Crew declarations from code: validation, wire shape and the closed binding result.

``@traigent.optimize(crew=..., role=...)`` declares which crew and role an agent plays.
The crew and role keys are CUSTOMER CONTENT: they travel only as
``metadata.crew_declaration`` of the session-create request and never appear in
spans, logs, telemetry, local result files or error messages. Every error raised here
is a fixed string that does not repeat the supplied values.

The backend answers with a closed ``crew_binding`` (status plus identifiers, no names).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

#: Same rules the backend enforces on a declaration.
CREW_KEY_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")
ROLE_KEY_MAX_LENGTH = 128

DECLARATION_METADATA_KEY = "crew_declaration"

logger = logging.getLogger(__name__)

#: Identifier formats of the closed binding result (TraigentSchema crew_common).
_CREW_ID = re.compile(r"crew_[0-9a-f]{8,64}")
_ROLE_ID = re.compile(r"crol_[0-9a-f]{8,64}")
_ROLE_TOKEN = re.compile(r"rtk_[a-z2-7]{26}")
_REVISION_ID = re.compile(r"crev_[0-9a-f]{8,64}")
_BINDING_FIELDS = frozenset(
    {"status", "crew_id", "role_id", "role_token", "crew_revision_id", "revision_no"}
)
_ID_PATTERNS = {
    "crew_id": _CREW_ID,
    "role_id": _ROLE_ID,
    "role_token": _ROLE_TOKEN,
    "crew_revision_id": _REVISION_ID,
}

NOT_SENT_MESSAGE = "Crew declaration not sent in this mode."

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

    def __repr__(self) -> str:
        return (
            f"CrewBinding(status={self.status!r}, crew_id={self.crew_id!r}, "
            f"role_id={self.role_id!r}, role_token={self.role_token!r}, "
            f"crew_revision_id={self.crew_revision_id!r}, "
            f"revision_no={self.revision_no!r})"
        )


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


def crew_declaration_to_wire(
    request: Any, *, send: bool = True
) -> dict[str, dict[str, str]]:
    """``{"crew_declaration": {...}}`` to merge into the request METADATA, or ``{}``.

    ``send=False`` is the content-less / optimization_only / legacy case: nothing is
    built (the keys are never copied into any body) and one fixed info line is logged.
    """
    declaration = declaration_of(request)
    if declaration is None:
        return {}
    if not send:
        logger.info(NOT_SENT_MESSAGE)
        return {}
    return {DECLARATION_METADATA_KEY: declaration.to_wire()}


def parse_crew_binding(raw: Any) -> CrewBinding:
    """Read the response ``crew_binding`` against the closed binding schema.

    Anything missing, unrecognised or malformed (bad id format, unknown status, ids
    present or absent contrary to the status, extra fields) is ``unknown``.
    """
    if not isinstance(raw, dict) or not set(raw) <= _BINDING_FIELDS:
        return UNKNOWN_BINDING
    status = raw.get("status")
    if not isinstance(status, str) or status not in BINDING_STATUSES:
        return UNKNOWN_BINDING
    ids: dict[str, str] = {}
    for name, pattern in _ID_PATTERNS.items():
        if name not in raw:
            continue
        value = raw[name]
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            return UNKNOWN_BINDING
        ids[name] = value
    revision_no = raw.get("revision_no")
    if "revision_no" in raw and (
        isinstance(revision_no, bool)
        or not isinstance(revision_no, int)
        or revision_no < 1
    ):
        return UNKNOWN_BINDING
    present = set(ids) | ({"revision_no"} if "revision_no" in raw else set())
    if status == STATUS_BOUND:
        if present != _BINDING_FIELDS - {"status"}:
            return UNKNOWN_BINDING
    elif status in (STATUS_PENDING_PORTAL, STATUS_CONFLICT):
        if not present <= {"crew_id"}:
            return UNKNOWN_BINDING
    elif present:
        return UNKNOWN_BINDING
    return CrewBinding(status=status, revision_no=revision_no, **ids)
