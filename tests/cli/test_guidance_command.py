"""CLI tests for the additive ``traigent guidance`` surface."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli


@pytest.fixture()
def payload() -> dict[str, object]:
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
            "policy_version": "policy-v2",
            "rule_version": "rules-v1",
            "calibration_version": "cal-v2",
            "shield_version": "shield-v2",
            "fallback_reason": None,
        },
    }


class _FakePlannerClient:
    payload: dict[str, object]
    captured: dict[str, object] = {}

    def __init__(self, backend_url: str, api_key: str | None = None) -> None:
        self.captured.update(backend_url=backend_url, api_key=api_key)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return None

    async def get_next_decision(
        self,
        run_id: str,
        *,
        utility_profile: str,
        treatment: str,
        strict_experiment: bool,
    ) -> dict[str, object]:
        self.captured.update(
            run_id=run_id,
            profile=utility_profile,
            treatment=treatment,
            strict=strict_experiment,
        )
        return self.payload

    async def resolve_decision(self, decision_id: str) -> dict[str, object]:
        self.captured["decision_id"] = decision_id
        return self.payload

    async def record_receipt(
        self,
        lifecycle_id: str,
        decision_id: str,
        *,
        attempt_id: str,
        status: str,
        successor_run_id: str | None,
        result_ref: str | None,
    ) -> dict[str, object]:
        self.captured.update(
            lifecycle_id=lifecycle_id,
            decision_id=decision_id,
            attempt_id=attempt_id,
            status=status,
            successor_run_id=successor_run_id,
            result_ref=result_ref,
        )
        return self.payload

    async def reopen_lifecycle(
        self,
        lifecycle_id: str,
        *,
        reason: str,
        expected_treatment: str,
        expected_profile: str,
    ) -> dict[str, object]:
        self.captured.update(
            lifecycle_id=lifecycle_id,
            reason=reason,
            expected_treatment=expected_treatment,
            expected_profile=expected_profile,
        )
        return self.payload


def test_execute_resolved_binding_token_fails_closed() -> None:
    result = CliRunner().invoke(
        cli,
        [
            "guidance",
            "execute-resolved",
            "--attempt",
            "attempt_0123456789abcdef",
        ],
    )

    assert result.exit_code != 0
    assert "binding token, not a shell command" in result.output


def test_guidance_next_json_forwards_precommitted_assignment(
    payload: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakePlannerClient.payload = payload
    _FakePlannerClient.captured = {}
    monkeypatch.setattr(
        "traigent.cli.guidance_command.PlannerV2Client", _FakePlannerClient
    )

    result = CliRunner().invoke(
        cli,
        [
            "guidance",
            "next",
            "run_123",
            "--profile",
            "balanced",
            "--treatment",
            "policy_override",
            "--strict-experiment",
            "--backend-url",
            "https://backend.example.test",
            "--api-key",
            "test-key",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == payload
    assert _FakePlannerClient.captured == {
        "backend_url": "https://backend.example.test",
        "api_key": "test-key",
        "run_id": "run_123",
        "profile": "balanced",
        "treatment": "policy_override",
        "strict": True,
    }


def test_guidance_next_table_presents_opaque_command_without_rewriting(
    payload: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakePlannerClient.payload = payload
    monkeypatch.setattr(
        "traigent.cli.guidance_command.PlannerV2Client", _FakePlannerClient
    )
    monkeypatch.setattr(
        "traigent.cli.guidance_command.BackendConfig.get_configured_backend_url",
        lambda: "https://backend.example.test",
    )
    monkeypatch.setattr(
        "traigent.cli.guidance_command.BackendConfig.get_api_key", lambda: "key"
    )

    result = CliRunner().invoke(cli, ["guidance", "next", "run_123"])

    assert result.exit_code == 0, result.output
    assert "policy_override" in result.output
    assert "certified_session_utility_advantage_no_kpi_guarantee" in result.output
    assert "lifecycle_0123456789abcdef" in result.output
    assert (
        "traigent guidance execute --decision decision_0123456789abcdef"
        in result.output
    )


def test_guidance_next_rejects_unknown_profile_before_client() -> None:
    result = CliRunner().invoke(
        cli, ["guidance", "next", "run_123", "--profile", "mystery"]
    )

    assert result.exit_code == 2
    assert "Invalid value for '--profile'" in result.output


def test_guidance_execute_resolves_typed_spec_without_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision_id = "decision_0123456789abcdef"
    _FakePlannerClient.payload = {
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
            "max_trials": 4,
            "reserved_cost_usd": "1.250000",
        },
        "signature": "sig_" + "A" * 43,
        "expires_at": "2099-07-10T10:30:00Z",
    }
    _FakePlannerClient.captured = {}
    monkeypatch.setattr(
        "traigent.cli.guidance_command.PlannerV2Client", _FakePlannerClient
    )
    monkeypatch.setattr(
        "traigent.cli.guidance_command.BackendConfig.get_configured_backend_url",
        lambda: "https://backend.example.test",
    )
    monkeypatch.setattr(
        "traigent.cli.guidance_command.BackendConfig.get_api_key", lambda: "key"
    )

    result = CliRunner().invoke(cli, ["guidance", "execute", "--decision", decision_id])

    assert result.exit_code == 0, result.output
    assert "run_optimization" in result.output
    assert "optimize_probe" in result.output
    assert "Receipt URL" in result.output
    assert "lifecycle_0123456789abcdef" in result.output
    assert "Lease expires" in result.output
    assert "Do not run server data through a shell" in result.output
    assert _FakePlannerClient.captured["decision_id"] == decision_id


def test_guidance_receipt_submits_result_for_server_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakePlannerClient.payload = {
        "schema_version": "2.0.0",
        "receipt_id": "receipt_0123456789abcdef",
        "lifecycle_id": "lifecycle_0123456789abcdef",
        "decision_id": "decision_0123456789abcdef",
        "attempt_id": "attempt_0123456789abcdef",
        "status": "submitted",
        "verification_status": "pending",
        "result_ref": "result_0123456789abcdef",
        "idempotent_replay": False,
        "updated_at": "2026-07-10T10:30:00Z",
    }
    _FakePlannerClient.captured = {}
    monkeypatch.setattr(
        "traigent.cli.guidance_command.PlannerV2Client", _FakePlannerClient
    )

    result = CliRunner().invoke(
        cli,
        [
            "guidance",
            "receipt",
            "--lifecycle",
            "lifecycle_0123456789abcdef",
            "--decision",
            "decision_0123456789abcdef",
            "--attempt",
            "attempt_0123456789abcdef",
            "--status",
            "submitted",
            "--result-ref",
            "result_0123456789abcdef",
            "--backend-url",
            "https://backend.example.test",
            "--api-key",
            "test-key",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["verification_status"] == "pending"
    assert _FakePlannerClient.captured == {
        "backend_url": "https://backend.example.test",
        "api_key": "test-key",
        "lifecycle_id": "lifecycle_0123456789abcdef",
        "decision_id": "decision_0123456789abcdef",
        "attempt_id": "attempt_0123456789abcdef",
        "status": "submitted",
        "successor_run_id": None,
        "result_ref": "result_0123456789abcdef",
    }


def test_guidance_reopen_preserves_server_assigned_arm_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakePlannerClient.payload = {
        "schema_version": "2.0.0",
        "lifecycle_id": "lifecycle_abcdef0123456789",
        "predecessor_lifecycle_id": "lifecycle_0123456789abcdef",
        "reason": "new_artifact",
        "requested_variant": "policy_override",
        "utility_profile": "balanced",
        "status": "active",
        "idempotent_replay": False,
        "created_at": "2026-07-10T10:30:00Z",
    }
    _FakePlannerClient.captured = {}
    monkeypatch.setattr(
        "traigent.cli.guidance_command.PlannerV2Client", _FakePlannerClient
    )

    result = CliRunner().invoke(
        cli,
        [
            "guidance",
            "reopen",
            "lifecycle_0123456789abcdef",
            "--reason",
            "new_artifact",
            "--expected-treatment",
            "policy_override",
            "--expected-profile",
            "balanced",
            "--backend-url",
            "https://backend.example.test",
            "--api-key",
            "test-key",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["requested_variant"] == "policy_override"
    assert _FakePlannerClient.captured == {
        "backend_url": "https://backend.example.test",
        "api_key": "test-key",
        "lifecycle_id": "lifecycle_0123456789abcdef",
        "reason": "new_artifact",
        "expected_treatment": "policy_override",
        "expected_profile": "balanced",
    }
