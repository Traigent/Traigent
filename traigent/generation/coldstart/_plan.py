"""cold-start-plan request/response handling.

Builds the outbound request payload, strictly parses the backend's response,
and recomputes the descriptor digest so a buggy or compromised transport
can't smuggle a plan meant for a different descriptor past the caller. The
plan itself carries no signature -- it is NOT cryptographically verifiable,
only digest-consistent with what this process actually sent. Nothing here
pretends the plan is more trustworthy than that.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ._contract import (
    KNOWN_422_REASONS,
    MAX_CANDIDATE_LIMIT,
    MIN_CANDIDATE_LIMIT,
    PROTOCOL_VERSION,
    compute_descriptor_digest,
)
from .models import DiscoveryGap

_RESPONSE_KEYS = frozenset(
    {
        "plan_id",
        "protocol_version",
        "descriptor_digest",
        "candidate_limit",
        "expires_at",
    }
)


@dataclass(frozen=True, slots=True)
class TransportResponse:
    """What an injected transport hands back: HTTP status + JSON body.

    The transport owns the actual network call (auth headers, retries,
    timeouts); this executor only ever sees the parsed status/body pair.
    """

    status_code: int
    body: Mapping[str, Any]


Transport = Callable[[Mapping[str, Any]], TransportResponse]


@dataclass(frozen=True, slots=True)
class Plan:
    """A parsed, digest-checkable (but NOT cryptographically verifiable)
    cold-start plan."""

    plan_id: str
    protocol_version: str
    descriptor_digest: str
    candidate_limit: int
    expires_at: datetime


def build_request(
    descriptor: Mapping[str, Any], candidate_limit: int
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "descriptor": dict(descriptor),
        "budget": {"candidate_limit": candidate_limit},
    }


def validate_descriptor_arity(descriptor: Mapping[str, Any]) -> DiscoveryGap | None:
    """Client-side pre-flight for an invariant the JSON schema cannot express.

    ``additionalProperties: false`` and per-field enums are schema-checkable;
    "``len(input_kinds) == input_arity``" is a cross-field invariant the
    schema has no way to state, so it is caught here before a round trip is
    burned on a request the backend would reject anyway.
    """
    input_kinds = descriptor.get("input_kinds")
    input_arity = descriptor.get("input_arity")
    if not isinstance(input_kinds, list) or len(input_kinds) != input_arity:
        return DiscoveryGap(
            reason="descriptor_arity_mismatch",
            detail=(
                f"input_kinds has "
                f"{len(input_kinds) if isinstance(input_kinds, list) else 'n/a'} "
                f"entries but input_arity is {input_arity!r}"
            ),
        )
    return None


def parse_response(response: TransportResponse) -> Plan | DiscoveryGap:
    if not isinstance(response, TransportResponse):
        raise TypeError(
            "transport must return a coldstart._plan.TransportResponse, "
            f"got {type(response).__name__}"
        )
    status = response.status_code
    body = response.body
    if status == 200:
        return _parse_plan(body)
    if status == 422:
        return _parse_422(body)
    if status == 501:
        return DiscoveryGap(
            reason="feature_disabled",
            detail="cold-start-plan is disabled",
            http_status=501,
        )
    if status in (401, 403):
        return DiscoveryGap(
            reason="unauthorized",
            detail=f"cold-start-plan authorization failed (status {status})",
            http_status=status,
        )
    return DiscoveryGap(
        reason="malformed_response",
        detail=f"cold-start-plan returned unexpected status {status}",
        http_status=status,
    )


def _parse_plan(body: Mapping[str, Any]) -> Plan | DiscoveryGap:
    if not isinstance(body, Mapping) or set(body) != _RESPONSE_KEYS:
        return DiscoveryGap(
            reason="malformed_response",
            detail="malformed cold-start-plan response body",
            http_status=200,
        )
    try:
        expires_at = _parse_timestamp(body["expires_at"])
        raw_candidate_limit = body["candidate_limit"]
        if isinstance(raw_candidate_limit, bool) or not isinstance(
            raw_candidate_limit, int
        ):
            raise ValueError("candidate_limit must be an int")
        candidate_limit = raw_candidate_limit
        if not (MIN_CANDIDATE_LIMIT <= candidate_limit <= MAX_CANDIDATE_LIMIT):
            raise ValueError("candidate_limit out of range")
        plan_id = str(body["plan_id"])
        if not plan_id:
            raise ValueError("plan_id must not be empty")
        plan = Plan(
            plan_id=plan_id,
            protocol_version=str(body["protocol_version"]),
            descriptor_digest=str(body["descriptor_digest"]),
            candidate_limit=candidate_limit,
            expires_at=expires_at,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return DiscoveryGap(
            reason="malformed_response",
            detail=f"malformed cold-start-plan response: {exc}",
            http_status=200,
        )
    return plan


def _parse_422(body: Mapping[str, Any]) -> DiscoveryGap:
    if isinstance(body, Mapping) and body.get("reason") in KNOWN_422_REASONS:
        return DiscoveryGap(
            reason=str(body["reason"]),
            detail=str(body.get("error", "cold-start-plan declined the descriptor")),
            http_status=422,
        )
    return DiscoveryGap(
        reason="malformed_response",
        detail="cold-start-plan returned 422 with an unrecognized body",
        http_status=422,
    )


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("expires_at must be a string")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise ValueError("expires_at must be timezone-aware")
    return parsed


def check_digest(descriptor: Mapping[str, Any], plan: Plan) -> DiscoveryGap | None:
    expected = compute_descriptor_digest(descriptor)
    if expected != plan.descriptor_digest:
        return DiscoveryGap(
            reason="descriptor_digest_mismatch",
            detail="recomputed descriptor digest does not match the plan's",
        )
    return None


def check_not_expired(plan: Plan, *, now: datetime) -> DiscoveryGap | None:
    if plan.expires_at <= now:
        return DiscoveryGap(
            reason="plan_expired",
            detail=f"plan expired at {plan.expires_at.isoformat()} (now {now.isoformat()})",
        )
    return None
