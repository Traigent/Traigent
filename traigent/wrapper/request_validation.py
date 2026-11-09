"""Request-schema validation for the hybrid wrapper ``/execute`` and ``/evaluate``.

Enforces the structural rules of the Hybrid Mode request contract before a
request is dispatched to user handlers: ``request_id`` (required UUID),
``tunable_id`` (required string), ``config`` (object, at most 200 properties),
and example ids (``^[a-zA-Z0-9][a-zA-Z0-9._-]{0,255}$``).

Every failure raises :class:`BadRequestError` with a FIXED message: client
supplied values, ids and keys are never echoed back (ids may be customer
content, and an echoed value could carry log or header injection).
"""

from __future__ import annotations

import re
from typing import Any

from traigent.wrapper.errors import BadRequestError

MAX_CONFIG_PROPERTIES = 200
MAX_TUNABLE_ID_LENGTH = 256
MAX_TIMEOUT_MS = 600000

_UUID_RE = re.compile(
    r"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)
# \Z, not $: "$" would accept a trailing newline.
_ID_RE = re.compile(r"\A[a-zA-Z0-9][a-zA-Z0-9._-]{0,255}\Z")


def is_valid_request_id(value: Any) -> bool:
    """True when ``value`` is a canonical hyphenated UUID string."""
    return isinstance(value, str) and _UUID_RE.match(value) is not None


def _fail(message: str) -> BadRequestError:
    return BadRequestError(message, error_code="INVALID_REQUEST")


def _check_common(request: dict[str, Any]) -> None:
    if "request_id" not in request:
        raise _fail("request_id is required")
    if not is_valid_request_id(request["request_id"]):
        raise _fail("request_id must be a UUID")

    tunable_id = request.get("tunable_id")
    if (
        not isinstance(tunable_id, str)
        or not tunable_id
        or len(tunable_id) > MAX_TUNABLE_ID_LENGTH
    ):
        raise _fail("tunable_id is required and must be a string of 1-256 characters")


def _check_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise _fail("config must be an object")
    if len(config) > MAX_CONFIG_PROPERTIES:
        raise _fail("config must have at most 200 properties")


def _check_item_ids(items: Any, label: str) -> None:
    """Validate ``example_id`` of each object item; leave other shapes to the service."""
    if not isinstance(items, list):
        return
    for item in items:
        if label == "evaluations" and not isinstance(item, dict):
            raise _fail("evaluations items must be objects")
        if isinstance(item, dict) and "example_id" in item:
            value = item["example_id"]
            if not isinstance(value, str) or _ID_RE.match(value) is None:
                raise _fail(f"{label} example_id must match the documented id pattern")


def validate_execute_request(request: dict[str, Any]) -> None:
    """Validate an ``/execute`` body; raises BadRequestError on any violation."""
    _check_common(request)
    if "config" not in request:
        raise _fail("config is required")
    _check_config(request["config"])
    _check_item_ids(request.get("examples"), "examples")


def validate_evaluate_request(request: dict[str, Any]) -> None:
    """Validate an ``/evaluate`` body; raises BadRequestError on any violation.

    ``config`` is only the deprecated alias of ``kwargs`` here, so it stays
    optional; when present it must still be an object within the size cap.
    """
    _check_common(request)
    if request.get("config") is not None:
        _check_config(request["config"])
    _check_item_ids(request.get("evaluations"), "evaluations")
