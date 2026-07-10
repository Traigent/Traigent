"""Contract tests for the additive Planner V2 SDK client."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from traigent.analytics.planner import PlannerV2Client


@pytest.fixture(autouse=True)
def _online_backend(jwt_development_mode, monkeypatch):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")


@pytest.fixture()
def decision_payload() -> dict[str, object]:
    decision_id = "decision_0123456789abcdef"
    return {
        "schema_version": "2.0.0",
        "lifecycle_id": "lifecycle_0123456789abcdef",
        "run_id": "run_123",
        "decision": {
            "id": decision_id,
            "mode": "policy_override",
            "category": "run_optimization",
            "source_engine": "policy",
            "baseline_rule_category": "audit_evaluator_quality",
            "utility_profile": "balanced",
            "certificate_ref": "certificate_0123456789abcdef",
            "advantage_label": "certified_session_utility_advantage_no_kpi_guarantee",
            "evidence_snapshot_hash": "ev_0123456789abcdefghijklmnopqrstuv",
            "rationale": "certified session-utility advantage selected; no product KPI guarantee",
            "action": {
                "kind": "cli",
                "variant": "optimize_probe",
                "command_template": (
                    f"traigent guidance execute --decision {decision_id}"
                ),
            },
            "evidence_level": "high",
        },
        "meta": {
            "requested_variant": "policy_override",
            "served_variant": "policy_override",
            "selector_engine": "policy",
            "context_status": "complete",
            "policy_version": "override-table-v2",
            "rule_version": "rules-v1-frozen",
            "calibration_version": "cal-v2",
            "shield_version": "shield-v2",
            "fallback_reason": None,
        },
    }


def _client_with_response(payload: object) -> tuple[PlannerV2Client, AsyncMock]:
    client = PlannerV2Client(api_key="test")
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    transport = AsyncMock()
    transport.post.return_value = response
    client._client = transport
    return client, transport


def _execution_payload() -> dict[str, object]:
    decision_id = "decision_0123456789abcdef"
    attempt_id = "attempt_0123456789abcdef"
    return {
        "schema_version": "2.0.0",
        "decision_id": decision_id,
        "argv": [
            "traigent",
            "guidance",
            "execute-resolved",
            "--attempt",
            attempt_id,
        ],
        "execution_spec": {
            "operation_kind": "run_optimization",
            "variant": "optimize_probe",
            "target": "agent",
            "cost_units": 2,
            "reservation_units": 2,
            "max_budget_fraction": 0.15,
            "max_trials": 4,
            "sample_limit": None,
            "action_signature": (
                "run_optimization:optimize_probe:agent:2:2:0.15:4:None"
            ),
            "economics_hash": (
                "df7d6d7c95e61716a55eec42f829c7ca2bb66c285ce4013e66f5dc1cac956d02"
            ),
            "attempt_id": attempt_id,
            "receipt_url": (
                "/api/v2/lifecycles/lifecycle_0123456789abcdef/decisions/"
                f"{decision_id}/receipts"
            ),
            "lease_expires_at": "2099-07-10T10:30:00Z",
        },
        "signature": "sig_" + "A" * 43,
        "expires_at": "2099-07-10T10:30:00Z",
    }


@pytest.mark.asyncio
async def test_get_next_decision_uses_additive_v2_endpoint(
    decision_payload: dict[str, object],
) -> None:
    client, transport = _client_with_response(decision_payload)

    result = await client.get_next_decision("run_123")

    assert result == decision_payload
    transport.post.assert_awaited_once_with(
        "/api/v2/experiment-runs/run_123/next-decision",
        json={
            "utility_profile": "balanced",
            "requested_variant": "policy_override",
        },
    )


@pytest.mark.asyncio
async def test_get_next_decision_rejects_missing_planner_payload() -> None:
    client, _ = _client_with_response({})

    with pytest.raises(ValueError, match="missing"):
        await client.get_next_decision("run_123")


@pytest.mark.asyncio
async def test_get_next_decision_propagates_old_backend_404() -> None:
    client, _ = _client_with_response({"message": "not found"})
    request = httpx.Request("POST", "https://backend.test/api/v2/next-decision")
    response = httpx.Response(404, request=request)
    client._client.post.return_value.raise_for_status.side_effect = (
        httpx.HTTPStatusError(
            "Planner V2 endpoint is unavailable",
            request=request,
            response=response,
        )
    )

    with pytest.raises(httpx.HTTPStatusError, match="endpoint is unavailable"):
        await client.get_next_decision("run_123")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("decision", "mode"), "unmapped", "decision.mode"),
        (("decision", "category"), "invented", "decision.category"),
        (("decision", "advantage_label"), "proven", "advantage_label"),
        (("decision", "evidence_level"), "certain", "evidence_level"),
        (("meta", "selector_engine"), "oracle", "selector_engine"),
        (("meta", "context_status"), "mystery", "context_status"),
    ],
)
async def test_get_next_decision_rejects_unknown_enums(
    decision_payload: dict[str, object],
    path: tuple[str, str],
    value: str,
    message: str,
) -> None:
    payload = deepcopy(decision_payload)
    nested = payload[path[0]]
    assert isinstance(nested, dict)
    nested[path[1]] = value
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match=message):
        await client.get_next_decision("run_123")


@pytest.mark.asyncio
@pytest.mark.parametrize("container", ["root", "decision", "meta", "action"])
async def test_get_next_decision_rejects_non_public_fields(
    decision_payload: dict[str, object], container: str
) -> None:
    payload = deepcopy(decision_payload)
    if container == "root":
        target = payload
    elif container == "action":
        decision = payload["decision"]
        assert isinstance(decision, dict)
        target = decision["action"]
    else:
        target = payload[container]
    assert isinstance(target, dict)
    target["internal_signal"] = 0.99
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match="unexpected internal_signal"):
        await client.get_next_decision("run_123")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("id", "decision.with.dots", "valid opaque identifier"),
        ("evidence_snapshot_hash", "sha256.synthetic", "malformed"),
    ],
)
async def test_get_next_decision_rejects_schema_invalid_identifiers(
    decision_payload: dict[str, object],
    field: str,
    value: str,
    match: str,
) -> None:
    payload = deepcopy(decision_payload)
    decision = payload["decision"]
    assert isinstance(decision, dict)
    decision[field] = value
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match=match):
        await client.get_next_decision("run_123")


@pytest.mark.asyncio
async def test_policy_override_requires_certificate_and_positive_label(
    decision_payload: dict[str, object],
) -> None:
    payload = deepcopy(decision_payload)
    decision = payload["decision"]
    assert isinstance(decision, dict)
    decision["certificate_ref"] = None
    decision["advantage_label"] = "no_certified_override"
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match="certified-positive"):
        await client.get_next_decision("run_123")


@pytest.mark.asyncio
async def test_executable_action_must_use_static_opaque_command(
    decision_payload: dict[str, object],
) -> None:
    payload = deepcopy(decision_payload)
    decision = payload["decision"]
    assert isinstance(decision, dict)
    action = decision["action"]
    assert isinstance(action, dict)
    action["command_template"] = "traigent optimize --trials 30"
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match="static opaque guidance command"):
        await client.get_next_decision("run_123")


@pytest.mark.asyncio
async def test_action_kind_and_variant_must_match_public_category(
    decision_payload: dict[str, object],
) -> None:
    payload = deepcopy(decision_payload)
    decision = payload["decision"]
    assert isinstance(decision, dict)
    action = decision["action"]
    assert isinstance(action, dict)
    action["kind"] = "sdk"

    client, _ = _client_with_response(payload)
    with pytest.raises(ValueError, match="action.kind"):
        await client.get_next_decision("run_123")

    action["kind"] = "cli"
    action["variant"] = "audit_full"
    client, _ = _client_with_response(payload)
    with pytest.raises(ValueError, match="action.variant"):
        await client.get_next_decision("run_123")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "category"), [("pending_wait", "wait"), ("safety_stop", "stop")]
)
async def test_wait_and_stop_are_non_executable(
    decision_payload: dict[str, object], mode: str, category: str
) -> None:
    payload = deepcopy(decision_payload)
    decision = payload["decision"]
    meta = payload["meta"]
    assert isinstance(decision, dict) and isinstance(meta, dict)
    decision.update(
        {
            "mode": mode,
            "category": category,
            "baseline_rule_category": category,
            "source_engine": "rules" if category == "wait" else "safety",
            "certificate_ref": None,
            "advantage_label": "not_applicable",
            "evidence_level": "low" if category == "wait" else "high",
            "rationale": (
                "an operation is already active; wait for its authoritative result"
                if category == "wait"
                else "no positive affordable safe action remains"
            ),
            "action": {"kind": "none", "variant": category, "command_template": ""},
        }
    )
    meta["selector_engine"] = "safety"
    client, _ = _client_with_response(payload)

    result = await client.get_next_decision("run_123")

    assert result["decision"]["category"] == category


@pytest.mark.asyncio
async def test_strict_policy_experiment_rejects_fallback(
    decision_payload: dict[str, object],
) -> None:
    payload = deepcopy(decision_payload)
    decision = payload["decision"]
    meta = payload["meta"]
    assert isinstance(decision, dict) and isinstance(meta, dict)
    decision.update(
        {
            "mode": "rules_fallback",
            "baseline_rule_category": "run_optimization",
            "source_engine": "rules",
            "certificate_ref": None,
            "advantage_label": "unavailable",
            "evidence_level": "low",
            "rationale": "policy artifact or certificate unavailable; safe rules were used",
        }
    )
    meta["selector_engine"] = "rules"
    meta["fallback_reason"] = "calibration_unavailable"
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match="Strict experiment rejects"):
        await client.get_next_decision("run_123", strict_experiment=True)


@pytest.mark.asyncio
async def test_strict_policy_experiment_accepts_certified_rules_parity(
    decision_payload: dict[str, object],
) -> None:
    payload = deepcopy(decision_payload)
    decision = payload["decision"]
    assert isinstance(decision, dict)
    decision.update(
        {
            "mode": "rules_parity",
            "baseline_rule_category": "run_optimization",
            "source_engine": "rules",
            "certificate_ref": None,
            "advantage_label": "no_certified_override",
            "evidence_level": "medium",
            "rationale": "no certified override applies; safe rule action retained",
        }
    )
    client, _ = _client_with_response(payload)

    result = await client.get_next_decision("run_123", strict_experiment=True)

    assert result["decision"]["mode"] == "rules_parity"


@pytest.mark.asyncio
async def test_treatment_mode_isolation_is_unconditional(
    decision_payload: dict[str, object],
) -> None:
    payload = deepcopy(decision_payload)
    meta = payload["meta"]
    assert isinstance(meta, dict)
    meta.update(requested_variant="rules_control", served_variant="rules_control")
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match="contaminates the requested treatment"):
        await client.get_next_decision(
            "run_123", treatment="rules_control", strict_experiment=False
        )


@pytest.mark.asyncio
async def test_strict_experiment_rejects_incomplete_context(
    decision_payload: dict[str, object],
) -> None:
    payload = deepcopy(decision_payload)
    meta = payload["meta"]
    assert isinstance(meta, dict)
    meta["context_status"] = "incomplete"
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match="fallback or incomplete context"):
        await client.get_next_decision("run_123", strict_experiment=True)


@pytest.mark.asyncio
async def test_rules_parity_cannot_change_the_baseline_category(
    decision_payload: dict[str, object],
) -> None:
    payload = deepcopy(decision_payload)
    decision = payload["decision"]
    assert isinstance(decision, dict)
    decision.update(
        {
            "mode": "rules_parity",
            "source_engine": "rules",
            "certificate_ref": None,
            "advantage_label": "no_certified_override",
            "evidence_level": "medium",
            "rationale": "no certified override applies; safe rule action retained",
            "category": "adjust_configuration_space",
            "action": {
                "kind": "cli",
                "variant": "config_prune",
                "command_template": (
                    "traigent guidance execute --decision decision_0123456789abcdef"
                ),
            },
        }
    )
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match="frozen baseline category"):
        await client.get_next_decision("run_123")


@pytest.mark.asyncio
async def test_submitted_receipt_requires_result_ref_before_http() -> None:
    client = PlannerV2Client(api_key="test")
    client._client = AsyncMock()

    with pytest.raises(ValueError, match="requires result_ref"):
        await client.record_receipt(
            "lifecycle_0123456789abcdef",
            "decision_0123456789abcdef",
            status="submitted",
            attempt_id="attempt_0123456789abcdef",
        )
    client._client.post.assert_not_called()


@pytest.mark.asyncio
async def test_non_submitted_receipt_rejects_successor_before_http() -> None:
    client = PlannerV2Client(api_key="test")
    client._client = AsyncMock()

    with pytest.raises(ValueError, match="successor_run_id is valid only"):
        await client.record_receipt(
            "lifecycle_0123456789abcdef",
            "decision_0123456789abcdef",
            status="failed",
            attempt_id="attempt_0123456789abcdef",
            successor_run_id="run_456",
        )
    client._client.post.assert_not_called()


@pytest.mark.asyncio
async def test_receipt_preserves_pending_verification_semantics() -> None:
    payload = {
        "schema_version": "2.0.0",
        "receipt_id": "receipt_0123456789abcdef",
        "lifecycle_id": "lifecycle_0123456789abcdef",
        "decision_id": "decision_0123456789abcdef",
        "attempt_id": "attempt_0123456789abcdef",
        "status": "submitted",
        "verification_status": "pending",
        "idempotent_replay": False,
        "successor_run_id": "run_456",
        "result_ref": "result_0123456789abcdef",
        "updated_at": "2026-07-10T09:00:01Z",
    }
    client, transport = _client_with_response(payload)

    result = await client.record_receipt(
        "lifecycle_0123456789abcdef",
        "decision_0123456789abcdef",
        status="submitted",
        attempt_id="attempt_0123456789abcdef",
        successor_run_id="run_456",
        result_ref="result_0123456789abcdef",
    )

    assert result["verification_status"] == "pending"
    transport.post.assert_awaited_once_with(
        "/api/v2/lifecycles/lifecycle_0123456789abcdef/decisions/"
        "decision_0123456789abcdef/receipts",
        json={
            "status": "submitted",
            "attempt_id": "attempt_0123456789abcdef",
            "successor_run_id": "run_456",
            "result_ref": "result_0123456789abcdef",
        },
        headers={"Idempotency-Key": "receipt-af8d666b308e43f5ec1f452653c11485"},
    )


@pytest.mark.asyncio
async def test_receipt_accepts_explicit_idempotent_replay() -> None:
    payload = {
        "schema_version": "2.0.0",
        "receipt_id": "receipt_0123456789abcdef",
        "lifecycle_id": "lifecycle_0123456789abcdef",
        "decision_id": "decision_0123456789abcdef",
        "attempt_id": "attempt_0123456789abcdef",
        "status": "submitted",
        "verification_status": "pending",
        "idempotent_replay": True,
        "successor_run_id": "run_456",
        "result_ref": "result_0123456789abcdef",
        "updated_at": "2026-07-10T09:00:01Z",
    }
    client, _ = _client_with_response(payload)

    result = await client.record_receipt(
        "lifecycle_0123456789abcdef",
        "decision_0123456789abcdef",
        status="submitted",
        attempt_id="attempt_0123456789abcdef",
        successor_run_id="run_456",
        result_ref="result_0123456789abcdef",
    )

    assert result["idempotent_replay"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "verification_status"),
    [
        ("started", "verified"),
        ("failed", "pending"),
        ("failed", "verified"),
        ("skipped", "pending"),
    ],
)
async def test_receipt_rejects_inconsistent_verification_transition(
    status: str, verification_status: str
) -> None:
    payload = {
        "schema_version": "2.0.0",
        "receipt_id": "receipt_0123456789abcdef",
        "lifecycle_id": "lifecycle_0123456789abcdef",
        "decision_id": "decision_0123456789abcdef",
        "attempt_id": "attempt_0123456789abcdef",
        "status": status,
        "verification_status": verification_status,
        "idempotent_replay": False,
        "updated_at": "2026-07-10T09:00:01Z",
    }
    client, _ = _client_with_response(payload)
    with pytest.raises(ValueError, match="status and verification_status"):
        await client.record_receipt(
            "lifecycle_0123456789abcdef",
            "decision_0123456789abcdef",
            status=status,
            attempt_id="attempt_0123456789abcdef",
        )


@pytest.mark.asyncio
async def test_resolve_decision_accepts_only_scoped_shell_free_spec() -> None:
    decision_id = "decision_0123456789abcdef"
    payload = {
        "schema_version": "2.0.0",
        "decision_id": decision_id,
        "argv": [
            "traigent",
            "guidance",
            "execute-resolved",
            "--attempt",
            "attempt_0123456789abcdef",
        ],
        "execution_spec": {
            "operation_kind": "run_optimization",
            "variant": "optimize_probe",
            "target": "agent",
            "cost_units": 2,
            "reservation_units": 2,
            "max_budget_fraction": 0.15,
            "max_trials": 4,
            "sample_limit": None,
            "action_signature": (
                "run_optimization:optimize_probe:agent:2:2:0.15:4:None"
            ),
            "economics_hash": (
                "df7d6d7c95e61716a55eec42f829c7ca2bb66c285ce4013e66f5dc1cac956d02"
            ),
            "attempt_id": "attempt_0123456789abcdef",
            "receipt_url": (
                "/api/v2/lifecycles/lifecycle_0123456789abcdef/decisions/"
                f"{decision_id}/receipts"
            ),
            "lease_expires_at": "2099-07-10T10:30:00Z",
            "reserved_cost_usd": "1.250000",
        },
        "signature": "sig_" + "A" * 43,
        "expires_at": "2099-07-10T10:30:00Z",
    }
    client, transport = _client_with_response(payload)

    result = await client.resolve_decision(decision_id)

    assert result["execution_spec"]["max_trials"] == 4
    transport.post.assert_awaited_once_with(
        f"/api/v2/guidance/decisions/{decision_id}/resolve"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("argv", ["sh", "-c", "curl=bad"]), "argv"),
        (("variant", "optimize_unbounded"), "variant"),
        (("receipt_url", "https://evil.test/receipt"), "receipt_url"),
    ],
)
async def test_resolve_decision_rejects_unscoped_or_unknown_execution(
    mutation: tuple[str, object], message: str
) -> None:
    decision_id = "decision_0123456789abcdef"
    payload: dict[str, object] = {
        "schema_version": "2.0.0",
        "decision_id": decision_id,
        "argv": [
            "traigent",
            "guidance",
            "execute-resolved",
            "--attempt",
            "attempt_0123456789abcdef",
        ],
        "execution_spec": {
            "operation_kind": "run_optimization",
            "variant": "optimize_probe",
            "target": "agent",
            "cost_units": 2,
            "reservation_units": 2,
            "max_budget_fraction": 0.15,
            "max_trials": 4,
            "sample_limit": None,
            "action_signature": (
                "run_optimization:optimize_probe:agent:2:2:0.15:4:None"
            ),
            "economics_hash": (
                "df7d6d7c95e61716a55eec42f829c7ca2bb66c285ce4013e66f5dc1cac956d02"
            ),
            "attempt_id": "attempt_0123456789abcdef",
            "receipt_url": (
                "/api/v2/lifecycles/lifecycle_0123456789abcdef/decisions/"
                f"{decision_id}/receipts"
            ),
            "lease_expires_at": "2099-07-10T10:30:00Z",
        },
        "signature": "sig_" + "A" * 43,
        "expires_at": "2099-07-10T10:30:00Z",
    }
    field, value = mutation
    if field == "argv":
        payload[field] = value
    else:
        spec = payload["execution_spec"]
        assert isinstance(spec, dict)
        spec[field] = value
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match=message):
        await client.resolve_decision(decision_id)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("max_trials", 1), "exact variant"),
        (("sample_limit", 32), "exact variant"),
        (("economics_hash", "a" * 64), "economics_hash"),
        (("action_signature", "run_optimization:wrong"), "action_signature"),
    ],
)
async def test_resolve_decision_requires_exact_action_economics(
    mutation: tuple[str, object], message: str
) -> None:
    payload = _execution_payload()
    spec = payload["execution_spec"]
    assert isinstance(spec, dict)
    field, value = mutation
    spec[field] = value
    client, _ = _client_with_response(payload)

    with pytest.raises(ValueError, match=message):
        await client.resolve_decision("decision_0123456789abcdef")


@pytest.mark.asyncio
async def test_resolve_decision_rejects_expired_or_mismatched_lease() -> None:
    payload = _execution_payload()
    payload["expires_at"] = "2020-01-01T00:00:00Z"
    spec = payload["execution_spec"]
    assert isinstance(spec, dict)
    spec["lease_expires_at"] = "2020-01-01T00:00:00Z"
    client, _ = _client_with_response(payload)
    with pytest.raises(ValueError, match="expired"):
        await client.resolve_decision("decision_0123456789abcdef")

    payload = _execution_payload()
    spec = payload["execution_spec"]
    assert isinstance(spec, dict)
    spec["lease_expires_at"] = "2099-07-10T10:31:00Z"
    client, _ = _client_with_response(payload)
    with pytest.raises(ValueError, match="expiries must match"):
        await client.resolve_decision("decision_0123456789abcdef")


@pytest.mark.asyncio
async def test_resolve_decision_binds_exact_argv_attempt_and_receipt_path() -> None:
    payload = _execution_payload()
    argv = payload["argv"]
    assert isinstance(argv, list)
    argv[-1] = "attempt_abcdef0123456789"
    client, _ = _client_with_response(payload)
    with pytest.raises(ValueError, match="argv attempt"):
        await client.resolve_decision("decision_0123456789abcdef")

    payload = _execution_payload()
    spec = payload["execution_spec"]
    assert isinstance(spec, dict)
    spec["receipt_url"] = (
        "/api/v2/lifecycles/lifecycle_0123456789abcdef/extra/decisions/"
        "decision_0123456789abcdef/receipts"
    )
    client, _ = _client_with_response(payload)
    with pytest.raises(ValueError, match="receipt_url"):
        await client.resolve_decision("decision_0123456789abcdef")


@pytest.mark.asyncio
async def test_reopen_creates_joined_active_child_lifecycle() -> None:
    payload = {
        "schema_version": "2.0.0",
        "lifecycle_id": "lifecycle_abcdef0123456789",
        "predecessor_lifecycle_id": "lifecycle_0123456789abcdef",
        "reason": "new_artifact",
        "requested_variant": "policy_override",
        "utility_profile": "balanced",
        "status": "active",
        "idempotent_replay": False,
        "created_at": "2026-07-10T09:00:00Z",
    }
    client, transport = _client_with_response(payload)

    result = await client.reopen_lifecycle(
        "lifecycle_0123456789abcdef", reason="new_artifact"
    )

    assert result["lifecycle_id"] == "lifecycle_abcdef0123456789"
    transport.post.assert_awaited_once_with(
        "/api/v2/lifecycles/lifecycle_0123456789abcdef/reopen",
        json={"reason": "new_artifact"},
        headers={"Idempotency-Key": "reopen-6d33ebdb297158ed652304060927e50d"},
    )


@pytest.mark.asyncio
async def test_reopen_rejects_same_lifecycle_or_inherited_assignment_drift() -> None:
    payload = {
        "schema_version": "2.0.0",
        "lifecycle_id": "lifecycle_0123456789abcdef",
        "predecessor_lifecycle_id": "lifecycle_0123456789abcdef",
        "reason": "new_artifact",
        "requested_variant": "policy_override",
        "utility_profile": "balanced",
        "status": "active",
        "idempotent_replay": False,
        "created_at": "2026-07-10T09:00:00Z",
    }
    client, _ = _client_with_response(payload)
    with pytest.raises(ValueError, match="distinct child"):
        await client.reopen_lifecycle(
            "lifecycle_0123456789abcdef", reason="new_artifact"
        )

    payload["lifecycle_id"] = "lifecycle_abcdef0123456789"
    client, _ = _client_with_response(payload)
    with pytest.raises(ValueError, match="inherited treatment"):
        await client.reopen_lifecycle(
            "lifecycle_0123456789abcdef",
            reason="new_artifact",
            expected_treatment="rules_control",
        )

    client, _ = _client_with_response(payload)
    with pytest.raises(ValueError, match="inherited utility profile"):
        await client.reopen_lifecycle(
            "lifecycle_0123456789abcdef",
            reason="new_artifact",
            expected_treatment="policy_override",
            expected_profile="cost_first",
        )
