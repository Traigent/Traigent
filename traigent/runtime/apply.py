"""``apply``: idempotent registration of a requirement.

* kwargs requirements use the native create route (provenance ``api``) and need
  no TVL compiler; a TVL-fragment requirement is compiled to the hosted subset
  and sent to ``tvl-import`` (provenance ``tvl_spec``).
* 201 is a new registration; an identical import (200) or a native 409 that
  matches exactly one ACTIVE head with an equal profile is a no-op.  Anything
  else (retired, missing, ambiguous, or a different profile) fails closed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from traigent.runtime import manifest
from traigent.runtime.client import RuntimeClient
from traigent.runtime.model import (
    E_EXISTS,
    PROFILE_FIELDS,
    Receipt,
    RuntimeRequirement,
    RuntimeRequirementError,
    _canonical_number,
    digest_of,
    normalise_environment,
)
from traigent.runtime.tvl_fragment import (
    compile_hosted_subset,
    module_name_for,
    token_entries,
)

E_AGENT_REQUIRED = "agent_id_required"
E_PROJECT_REQUIRED = "project_id_required"

EXIT_CONFLICT = 3


class ApplyConflict(RuntimeRequirementError):
    """The registered requirement differs from the declaration (CLI exit code 3)."""

    def __init__(self, diff: list[tuple[str, Any, Any]]) -> None:
        super().__init__(E_EXISTS)
        self.diff = diff


@dataclass(frozen=True)
class ApplyResult:
    outcome: str  # "created" | "noop" | "dry_run"
    route: str  # "native" | "tvl_import"
    receipt: Receipt | None
    policy_digest: str
    hosted_digest: str
    notes: list[str] = field(default_factory=list)


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a == b)
        if not (math.isfinite(a) and math.isfinite(b)):
            return False
        # Exact, after the shared digest's canonical number encoding (1 == 1.0).
        return _canonical_number(a) == _canonical_number(b)
    return bool(a == b)


def profile_diff(
    requirement: RuntimeRequirement, remote: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    diff: list[tuple[str, Any, Any]] = []
    for name, value in requirement.profile().items():
        if not _equal(value, remote.get(name)):
            diff.append((name, value, remote.get(name)))
    return diff


_REACTIVE_REF = re.compile(r"policyrevisionref_[0-9a-f]{32}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _receipt(item: dict[str, Any], created: bool | None) -> Receipt:
    if item.get("receipt_kind") == "reactive_import":
        # Zero-native contract (201 new / 200 replay): no production requirement exists.
        ref, digest = item.get("policy_revision_ref"), item.get("contract_digest")
        if (
            item.get("schema_version") != 1
            or not isinstance(ref, str)
            or not _REACTIVE_REF.fullmatch(ref)
            or not isinstance(digest, str)
            or not _DIGEST.fullmatch(digest)
            or item.get("created") is not created
        ):
            raise RuntimeRequirementError("backend_error", "response")
        return Receipt(
            head_id="",
            revision_id="",
            revision_number=0,
            provenance="tvl_spec",
            status="reactive_import",
            created=created,
            policy_revision_ref=ref,
            contract_digest=digest,
        )
    try:
        return Receipt(
            head_id=str(item["head_id"]),
            revision_id=str(item["revision_id"]),
            revision_number=int(item["revision_number"]),
            provenance=str(item["provenance"]),
            status=str(item["status"]),
            created=created,
        )
    except (KeyError, TypeError, ValueError):
        raise RuntimeRequirementError("backend_error", "response") from None


def matching_heads(
    items: list[dict[str, Any]],
    requirement: RuntimeRequirement,
    project_id: str,
    agent_id: str,
) -> list[dict[str, Any]]:
    """Heads (any status) that carry this requirement's key for this project and agent."""
    key = requirement.key()
    out = []
    for item in items:
        env = item.get("environment")
        if (
            item.get("scope") == "agent"
            and item.get("project_id") == project_id
            and item.get("agent_id") == agent_id
            and item.get("kind") == key["kind"]
            and item.get("latency_threshold_ms") == key["latency_threshold_ms"]
            and isinstance(env, str)
            and normalise_environment(env) == key["environment"]
        ):
            out.append(item)
    return out


def current_head(
    client: RuntimeClient,
    requirement: RuntimeRequirement,
    project_id: str,
    agent_id: str,
) -> dict[str, Any]:
    """The single ACTIVE head for the key, or a fail-closed error."""
    matches = matching_heads(
        client.list_requirements(), requirement, project_id, agent_id
    )
    active = [m for m in matches if m.get("status") == "active"]
    if len(active) != 1:
        raise RuntimeRequirementError(E_EXISTS, "active_head")
    return active[0]


def _resolve_apply_result(
    item: dict[str, Any] | None,
    status: int,
    api: RuntimeClient,
    requirement: RuntimeRequirement,
    project_id: str,
    agent: str,
    route: str,
) -> tuple[str, Receipt]:
    if item is not None:
        created = status == 201
        outcome = "created" if created else "noop"
        receipt = _receipt(item, created)
    else:  # 409 requirement_exists
        head = current_head(api, requirement, project_id, agent)
        diff = profile_diff(requirement, head)
        if diff:
            raise ApplyConflict(diff)
        if route == "tvl_import":
            # Equal profile but not the identical import: provenance differs.
            raise RuntimeRequirementError(E_EXISTS, "provenance")
        receipt = _receipt(head, False)
        outcome = "noop"

    return outcome, receipt


def apply_requirement(
    requirement: RuntimeRequirement,
    *,
    project_id: str,
    agent_id: str | None = None,
    client: RuntimeClient | None = None,
    state_dir: Path | None = None,
    dry_run: bool = False,
    module: str | None = None,
) -> ApplyResult:
    """Register ``requirement`` (see module docstring); never guesses the agent."""
    agent = agent_id or requirement.agent_id or ""
    if not dry_run:
        if not agent:
            raise RuntimeRequirementError(E_AGENT_REQUIRED, "agent_id")
        if not project_id:
            raise RuntimeRequirementError(E_PROJECT_REQUIRED, "project_id")
    errors = [i for i in requirement.validate() if i.severity == "error"]
    if errors:
        raise RuntimeRequirementError(errors[0].code, errors[0].path)
    state = state_dir or manifest.default_state_dir()
    policy_digest = requirement.policy_digest()

    body: dict[str, Any]
    if requirement.source == "tvl":
        module_ref = module or module_name_for(requirement.name)
        token_map = manifest.ensure_tokens(
            state, token_entries(requirement, module_ref), persist=not dry_run
        )
        contract, _local, _issues = compile_hosted_subset(
            requirement, token_map, module_ref
        )
        env_ref = manifest.token_for(
            token_map, "environment", normalise_environment(requirement.environment)
        )
        body = {
            "agent_id": agent,
            "contract": contract,
            "environment_binding": {
                "ref": env_ref,
                "label": requirement.environment.strip(),
            },
        }
        hosted_digest = digest_of(contract)
        route = "tvl_import"
    else:
        body = requirement.native_body(agent)
        hosted_digest = digest_of({k: v for k, v in body.items() if k != "agent_id"})
        route = "native"

    if dry_run:
        return ApplyResult("dry_run", route, None, policy_digest, hosted_digest)

    api = client or RuntimeClient(project_id)
    if route == "native":
        status, item = api.create_requirement(body)
    else:
        status, item = api.import_tvl(body)

    outcome, receipt = _resolve_apply_result(
        item, status, api, requirement, project_id, agent, route
    )

    manifest.record_apply(
        state,
        agent,
        name=requirement.name,
        policy_digest=policy_digest,
        hosted_digest=hosted_digest,
        key=requirement.key(),
        route=route,
        receipt=receipt.as_dict(),
    )
    return ApplyResult(outcome, route, receipt, policy_digest, hosted_digest)


__all__ = [
    "ApplyConflict",
    "ApplyResult",
    "EXIT_CONFLICT",
    "PROFILE_FIELDS",
    "apply_requirement",
    "current_head",
    "matching_heads",
    "profile_diff",
]
