"""Release and undo of an agent head, from the SDK.

A *release* moves an agent's head to a configuration a person has already
accepted (owner ruling 1A: every release needs a recorded human acceptance).
``OptimizedFunction.apply_best_config()`` is NOT a release: it is a developer
shortcut with no gate and no acceptance, and it never moves the head.

Contracts (TraigentBackend ``src/contracts/optimization/release_acceptance_endpoints.json``):

* ``POST /api/v1/projects/{project_id}/agents/{agent_ref}/best-config-head/release``
  with ``{acceptance_id, expected_head_generation, agent_code_version,
  configuration_space_digest, gate_result_digest}``.
* ``POST .../best-config-head/rollback`` with ``{environment, target_generation,
  expected_head_generation}``.  This route is in review on the backend and not
  merged: it needs a backend that has it (a missing route is refused closed).

Every refusal is the closed ``ReleaseRefusal`` envelope.  Only its ``error_code``
is read, and only when the whole envelope matches the contract; nothing else of
the body is kept, so a server message is never echoed.  Every failure raises
:class:`ReleaseRefused`, whose message is SDK-authored from the closed code.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from traigent.runtime.client import AGENT_API_PREFIX, RuntimeClient
from traigent.runtime.model import RuntimeRequirementError

#: The fixed value for "no gate was run" (ReleaseGateResultDigest).
NO_GATE_RUN = "no_gate_run"

_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_AGENT_REF = re.compile(r"agentref_[0-9a-f]{32}")
_ENVIRONMENTS = ("production", "staging", "development", "test", "default")
_MAX_GENERATION = 9_007_199_254_740_991

#: Closed code -> (HTTP status the contract pairs with it, contract message, SDK message).
_REFUSALS: dict[str, tuple[int, str, str]] = {
    "invalid_request": (400, "Request is invalid.", "the request is invalid"),
    "unauthenticated": (401, "Authentication required.", "authentication is required"),
    "forbidden": (403, "Forbidden.", "this credential may not do that"),
    "reference_not_found": (
        404,
        "Reference not found.",
        "the project, agent or acceptance was not found (or the backend has no such route)",
    ),
    "acceptance_required": (
        409,
        "A release acceptance is required.",
        "a person must accept this configuration first",
    ),
    "acceptance_expired": (
        409,
        "The release acceptance has expired.",
        "the acceptance has expired; a person must accept again",
    ),
    "acceptance_consumed": (
        409,
        "The release acceptance was already used.",
        "the acceptance was already used; a person must accept again",
    ),
    "acceptance_revoked": (
        409,
        "The release acceptance was revoked.",
        "the acceptance was revoked; a person must accept again",
    ),
    "acceptance_binding_mismatch": (
        409,
        "The release does not match the acceptance.",
        "this release differs from what was accepted (candidate, gate result, "
        "configuration space or agent code version)",
    ),
    "head_generation_stale": (
        409,
        "Agent head moved since the acceptance.",
        "the agent head moved since the acceptance; a person must accept again",
    ),
    "rollback_target_not_accepted": (
        409,
        "The rollback target was not accepted.",
        "that generation was never accepted or released, or does not exist",
    ),
    "internal_error": (500, "Internal error.", "the backend failed; try again later"),
}

#: SDK-side codes (no server involved, or the server answered outside the contract).
_LOCAL_REFUSALS: dict[str, str] = {
    "egress_disabled": "backend egress is disabled",
    "credentials_missing": "no API key is configured",
    "backend_error": "the backend answered outside the release contract",
    "agent_ref_required": "set agent_ref or TRAIGENT_AGENT_REF to the agent's opaque reference",
    "project_required": "set project_id or TRAIGENT_PROJECT_ID",
    "environment_required": "name the environment (production, staging, development, test, default)",
    "code_version_unavailable": "the agent code version cannot be determined",
    "invalid_digest": "a configuration-space or gate-result digest is not a legal value",
}

#: Every code a :class:`ReleaseRefused` may carry.
REFUSAL_CODES = frozenset(_REFUSALS) | frozenset(_LOCAL_REFUSALS)

_ENVELOPE_KEYS = frozenset({"success", "error", "error_code", "message"})


class ReleaseRefused(Exception):
    """A release or undo did not happen.  ``code`` is a closed value; no server text.

    The message is ``"<code>: <fixed SDK text>"``, for example
    ``"acceptance_required: a person must accept this configuration first"``.
    """

    def __init__(self, code: str, *, status: int | None = None) -> None:
        if code not in REFUSAL_CODES:
            code = "backend_error"
        self.code = code
        self.status = status
        text = _REFUSALS[code][2] if code in _REFUSALS else _LOCAL_REFUSALS[code]
        super().__init__(f"{code}: {text}")

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (self.code, self.status))


def _rebuild(code: str, status: int | None) -> ReleaseRefused:
    return ReleaseRefused(code, status=status)


@dataclass(frozen=True)
class ReleaseReceipt:
    """What the backend recorded for a release or an undo."""

    agent_ref: str
    environment: str
    generation: int
    best_config_id: str | None
    best_config_version: int | None
    acceptance_id: str | None


# --- digests -----------------------------------------------------------------------------


def configuration_space_digest(space: Mapping[str, Any] | None) -> str:
    """Lowercase-hex sha256 of the canonical JSON of the normalized configuration space.

    Canonical form: sorted keys, compact separators, ASCII, UTF-8.  Customer content
    never leaves the process; only this digest does.
    """
    from traigent.api.parameter_ranges import normalize_configuration_space

    normalized, _defaults = normalize_configuration_space(dict(space or {}))
    try:
        text = json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
            default=_plain,
        )
    except (TypeError, ValueError):
        raise ReleaseRefused("invalid_digest") from None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError("not JSON-plain")


# --- wire ----------------------------------------------------------------------------------


def _refusal_from(status: int, body: Any) -> ReleaseRefused:
    """The closed refusal of a failed response; a body outside the envelope is ``backend_error``."""
    if isinstance(body, dict) and set(body) == _ENVELOPE_KEYS:
        raw = body.get("error_code")
        spec = _REFUSALS.get(raw) if isinstance(raw, str) else None
        if (
            spec is not None
            and body.get("success") is False
            and body.get("message") == spec[1]
            and status == spec[0]
        ):
            return ReleaseRefused(str(raw), status=status)
    return ReleaseRefused("backend_error", status=status)


def _send(
    client: RuntimeClient,
    method: str,
    agent_ref: str,
    suffix: str,
    body: dict[str, Any] | None,
    params: dict[str, Any] | None = None,
) -> Any:
    origin = client.origin
    project = quote(client.project_id, safe="")
    url = f"{origin}{AGENT_API_PREFIX}/{project}/agents/{agent_ref}{suffix}"
    try:
        status, parsed, _headers = client.send(method, url, body, params=params)
    except RuntimeRequirementError as exc:
        code = exc.code if exc.code in _LOCAL_REFUSALS else "backend_error"
        raise ReleaseRefused(code) from None
    except Exception:  # noqa: BLE001 - never surface transport text
        raise ReleaseRefused("backend_error") from None
    if status != 200:
        raise _refusal_from(status, parsed)
    return parsed


def _gen(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_GENERATION
    ):
        raise ReleaseRefused("invalid_request")
    return value


def _parse_receipt(parsed: Any, agent_ref: str) -> ReleaseReceipt:
    data = parsed.get("data") if isinstance(parsed, dict) else None
    if (
        not isinstance(data, dict)
        or parsed.get("success") is not True
        or data.get("agent_id", agent_ref) != agent_ref
        or data.get("environment") not in _ENVIRONMENTS
    ):
        raise ReleaseRefused("backend_error")
    generation = data.get("generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or not 0 <= generation <= _MAX_GENERATION
    ):
        raise ReleaseRefused("backend_error")
    config_id = data.get("best_config_id")
    version = data.get("best_config_version")
    acceptance = data.get("acceptance_id")
    return ReleaseReceipt(
        agent_ref=agent_ref,
        environment=data["environment"],
        generation=generation,
        best_config_id=(
            config_id
            if isinstance(config_id, str) and _ID.fullmatch(config_id)
            else None
        ),
        best_config_version=(
            version
            if isinstance(version, int) and not isinstance(version, bool)
            else None
        ),
        acceptance_id=(
            acceptance
            if isinstance(acceptance, str) and _ID.fullmatch(acceptance)
            else None
        ),
    )


def _need_agent_ref(agent_ref: str) -> str:
    if not isinstance(agent_ref, str) or not _AGENT_REF.fullmatch(agent_ref):
        raise ReleaseRefused("agent_ref_required")
    return agent_ref


def read_head_generation(
    client: RuntimeClient, agent_ref: str, environment: str
) -> int:
    """Current head generation (0 when never released): ``GET .../best-config-head``."""
    _need_agent_ref(agent_ref)
    if environment not in _ENVIRONMENTS:
        raise ReleaseRefused("environment_required")
    parsed = _send(
        client,
        "GET",
        agent_ref,
        "/best-config-head",
        None,
        params={"environment": environment},
    )
    data = parsed.get("data", parsed) if isinstance(parsed, dict) else None
    generation = data.get("generation") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or isinstance(generation, bool)
        or not isinstance(generation, int)
        or not 0 <= generation <= _MAX_GENERATION
    ):
        raise ReleaseRefused("backend_error")
    return generation


def release_agent_head(
    client: RuntimeClient,
    agent_ref: str,
    *,
    acceptance_id: str,
    expected_head_generation: int,
    agent_code_version: str,
    configuration_space_digest: str,
    gate_result_digest: str = NO_GATE_RUN,
) -> ReleaseReceipt:
    """``POST .../best-config-head/release``: move the head to an accepted candidate.

    Raises :class:`ReleaseRefused` on any refusal; the head is moved only on success.
    """
    _need_agent_ref(agent_ref)
    if not isinstance(acceptance_id, str) or not _ID.fullmatch(acceptance_id):
        raise ReleaseRefused("invalid_request")
    if not isinstance(agent_code_version, str) or not _ID.fullmatch(agent_code_version):
        raise ReleaseRefused("invalid_request")
    if not isinstance(configuration_space_digest, str) or not _DIGEST.fullmatch(
        configuration_space_digest
    ):
        raise ReleaseRefused("invalid_digest")
    if not isinstance(gate_result_digest, str) or not (
        gate_result_digest == NO_GATE_RUN or _DIGEST.fullmatch(gate_result_digest)
    ):
        raise ReleaseRefused("invalid_digest")
    body = {
        "acceptance_id": acceptance_id,
        "expected_head_generation": _gen(expected_head_generation),
        "agent_code_version": agent_code_version,
        "configuration_space_digest": configuration_space_digest,
        "gate_result_digest": gate_result_digest,
    }
    parsed = _send(client, "POST", agent_ref, "/best-config-head/release", body)
    return _parse_receipt(parsed, agent_ref)


def rollback_agent_head(
    client: RuntimeClient,
    agent_ref: str,
    *,
    environment: str,
    target_generation: int,
    expected_head_generation: int,
) -> ReleaseReceipt:
    """``POST .../best-config-head/rollback``: undo to an earlier generation.

    Needs a backend that has the rollback route (in review, not yet merged).  A person
    may return to any earlier generation; an API key only to one that was accepted or
    released (``rollback_target_not_accepted`` otherwise).
    """
    _need_agent_ref(agent_ref)
    if environment not in _ENVIRONMENTS:
        raise ReleaseRefused("environment_required")
    body = {
        "environment": environment,
        "target_generation": _gen(target_generation),
        "expected_head_generation": _gen(expected_head_generation),
    }
    parsed = _send(client, "POST", agent_ref, "/best-config-head/rollback", body)
    return _parse_receipt(parsed, agent_ref)


__all__ = [
    "NO_GATE_RUN",
    "REFUSAL_CODES",
    "ReleaseReceipt",
    "ReleaseRefused",
    "configuration_space_digest",
    "read_head_generation",
    "release_agent_head",
    "rollback_agent_head",
]
