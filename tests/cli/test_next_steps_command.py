"""CLI smoke tests for traigent next-steps."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.config.backend_config import DEFAULT_LOCAL_URL


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def next_steps_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "experiment_run_id": "run_123",
        "caveat": "Recommendations are category-level and should be reviewed.",
        "summary": {"confidence_label": "medium"},
        "next_steps": [
            {
                "id": "step_1",
                "category": "compare_with_baseline",
                "priority": 1,
                "rationale": "Compare the candidate winner to a baseline.",
                "action": {
                    "kind": "cli",
                    "command_template": "traigent results compare run_123 baseline",
                },
                "evidence_level": "medium",
            }
        ],
    }


class _FakeNextStepsClient:
    def __init__(self, backend_url: str, api_key: str | None = None) -> None:
        self.backend_url = backend_url
        self.api_key = api_key

    async def __aenter__(self) -> _FakeNextStepsClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def get_next_steps(
        self,
        experiment_run_id: str,
        *,
        guidance_variant: str | None = None,
        strict_experiment: bool = False,
    ) -> dict[str, object]:
        assert experiment_run_id == "run_123"
        self.guidance_variant = guidance_variant
        self.strict_experiment = strict_experiment
        return _FakeNextStepsClient.payload


def test_next_steps_table_mode(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeNextStepsClient.payload = next_steps_payload
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient",
        _FakeNextStepsClient,
    )

    result = runner.invoke(cli, ["next-steps", "run_123"])

    assert result.exit_code == 0
    assert "Next Steps for run_123" in result.output
    assert "compare_with_baseline" in result.output
    assert "traigent results compare run_123 baseline" in result.output
    assert "Caveat:" in result.output
    assert "Recommendations are category-level" in result.output
    assert "Posture:" not in result.output


def test_next_steps_table_mode_prints_posture_summary(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_with_posture = {
        **next_steps_payload,
        "posture": {
            "summary_text": "Evidence is sufficient for a cautious promotion.",
            "generated_at": "2026-06-27T09:30:00Z",
        },
    }
    _FakeNextStepsClient.payload = payload_with_posture
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient",
        _FakeNextStepsClient,
    )

    result = runner.invoke(cli, ["next-steps", "run_123"])

    assert result.exit_code == 0
    assert "Posture:" in result.output
    assert "Evidence is sufficient for a cautious promotion." in result.output
    assert "compare_with_baseline" in result.output


def test_next_steps_table_mode_prints_guidance_meta_line(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload_with_guidance_meta = {
        **next_steps_payload,
        "guidance_meta": {
            "requested_variant": "policy",
            "served_variant": "policy",
            "engine": "policy",
            "policy_table_sha": "abc123",
            "smartopt_version": "0.90.0",
            "fallback_reason": None,
        },
    }
    _FakeNextStepsClient.payload = payload_with_guidance_meta
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient",
        _FakeNextStepsClient,
    )

    result = runner.invoke(cli, ["next-steps", "run_123"])

    assert result.exit_code == 0
    assert "guidance: requested=policy served=policy engine=policy" in result.output
    assert "fallback=none" in result.output


def test_next_steps_table_mode_prints_non_executable_wait_decision(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeNextStepsClient.payload = {
        **next_steps_payload,
        "next_steps": [],
        "decision": {
            "id": "decision_wait",
            "category": "wait",
            "source_engine": "rules",
            "evidence_snapshot_hash": "abc123",
            "rationale": "Wait for more evidence.",
            "action": {"kind": "none", "command_template": ""},
            "evidence_level": "low",
        },
    }
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient", _FakeNextStepsClient
    )

    result = runner.invoke(cli, ["next-steps", "run_123"])

    assert result.exit_code == 0
    assert "Decision:" in result.output
    assert "wait" in result.output
    assert "no action" in result.output


def test_next_steps_table_mode_prefers_authoritative_decision_over_legacy_rows(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeNextStepsClient.payload = {
        **next_steps_payload,
        "decision": {
            "id": "decision_1",
            "category": "audit_evaluator_quality",
            "source_engine": "rules",
            "evidence_snapshot_hash": "abc123",
            "rationale": "Audit the evaluator.",
            "action": {"kind": "skill", "command_template": "traigent-eval-audit"},
            "evidence_level": "high",
        },
    }
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient", _FakeNextStepsClient
    )

    result = runner.invoke(cli, ["next-steps", "run_123"])

    assert result.exit_code == 0
    assert "audit_evaluator_quality" in result.output
    assert "traigent-eval-audit" in result.output
    assert "compare_with_baseline" not in result.output


def test_next_steps_forwards_guidance_variant_and_strict_mode(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeNextStepsClient.payload = next_steps_payload
    captured: dict[str, object] = {}

    class _CapturingClient(_FakeNextStepsClient):
        async def get_next_steps(
            self,
            experiment_run_id: str,
            *,
            guidance_variant: str | None = None,
            strict_experiment: bool = False,
        ) -> dict[str, object]:
            captured["guidance_variant"] = guidance_variant
            captured["strict_experiment"] = strict_experiment
            return await super().get_next_steps(
                experiment_run_id,
                guidance_variant=guidance_variant,
                strict_experiment=strict_experiment,
            )

    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient", _CapturingClient
    )
    result = runner.invoke(
        cli,
        [
            "next-steps",
            "run_123",
            "--guidance-variant",
            "policy",
            "--strict-experiment",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {"guidance_variant": "policy", "strict_experiment": True}


def test_next_steps_rejects_invalid_guidance_variant_locally(
    runner: CliRunner,
) -> None:
    result = runner.invoke(
        cli,
        ["next-steps", "run_123", "--guidance-variant", "variant_b"],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--guidance-variant'" in result.output


def test_next_steps_table_mode_without_guidance_meta_omits_guidance_line(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward compat: payloads without guidance_meta print no guidance line."""
    _FakeNextStepsClient.payload = next_steps_payload
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient",
        _FakeNextStepsClient,
    )

    result = runner.invoke(cli, ["next-steps", "run_123"])

    assert result.exit_code == 0
    assert "guidance:" not in result.output


def test_next_steps_json_mode(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeNextStepsClient.payload = next_steps_payload
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient",
        _FakeNextStepsClient,
    )

    result = runner.invoke(cli, ["next-steps", "run_123", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.output) == next_steps_payload


def test_next_steps_uses_explicit_backend_url_and_api_key_over_stored_credentials(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit --backend-url/--api-key must win over stored credentials."""
    _FakeNextStepsClient.payload = next_steps_payload
    captured: dict[str, object] = {}

    class _CapturingClient(_FakeNextStepsClient):
        def __init__(self, backend_url: str, api_key: str | None = None) -> None:
            captured["backend_url"] = backend_url
            captured["api_key"] = api_key
            super().__init__(backend_url, api_key)

    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient", _CapturingClient
    )
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.BackendConfig.get_configured_backend_url",
        lambda: (_ for _ in ()).throw(
            AssertionError("should not fall back when --backend-url is explicit")
        ),
    )
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.BackendConfig.get_api_key",
        lambda: (_ for _ in ()).throw(
            AssertionError("should not fall back when --api-key is explicit")
        ),
    )

    result = runner.invoke(
        cli,
        [
            "next-steps",
            "run_123",
            "--backend-url",
            "https://explicit.example.test",
            "--api-key",
            "explicit-key",  # pragma: allowlist secret
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["backend_url"] == "https://explicit.example.test"
    assert captured["api_key"] == "explicit-key"  # pragma: allowlist secret


def test_next_steps_falls_back_to_stored_cli_credentials_when_unset(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Traigent#1721: next-steps must not silently ignore stored CLI credentials."""
    monkeypatch.delenv("TRAIGENT_BACKEND_URL", raising=False)
    monkeypatch.delenv("TRAIGENT_API_URL", raising=False)
    _FakeNextStepsClient.payload = next_steps_payload
    captured: dict[str, object] = {}

    class _CapturingClient(_FakeNextStepsClient):
        def __init__(self, backend_url: str, api_key: str | None = None) -> None:
            captured["backend_url"] = backend_url
            captured["api_key"] = api_key
            super().__init__(backend_url, api_key)

    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient", _CapturingClient
    )
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.BackendConfig.get_configured_backend_url",
        lambda: "https://stored.example.test",
    )
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.BackendConfig.get_api_key",
        lambda: "stored-key",  # pragma: allowlist secret
    )

    result = runner.invoke(cli, ["next-steps", "run_123"])

    assert result.exit_code == 0, result.output
    assert captured["backend_url"] == "https://stored.example.test"
    assert captured["api_key"] == "stored-key"  # pragma: allowlist secret
    assert "Using backend URL from stored CLI credentials" in result.output


def test_next_steps_falls_back_to_localhost_when_nothing_configured(
    runner: CliRunner,
    next_steps_payload: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Traigent#1721: with no flag/env/stored cred, next-steps must fall back to
    the local default, NOT the prod cloud."""
    monkeypatch.delenv("TRAIGENT_BACKEND_URL", raising=False)
    monkeypatch.delenv("TRAIGENT_API_URL", raising=False)
    _FakeNextStepsClient.payload = next_steps_payload
    captured: dict[str, object] = {}

    class _CapturingClient(_FakeNextStepsClient):
        def __init__(self, backend_url: str, api_key: str | None = None) -> None:
            captured["backend_url"] = backend_url
            captured["api_key"] = api_key
            super().__init__(backend_url, api_key)

    monkeypatch.setattr(
        "traigent.cli.next_steps_command.NextStepsClient", _CapturingClient
    )
    # Genuinely nothing configured: no env (deleted above), no stored cred.
    monkeypatch.setattr(
        "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
        classmethod(lambda cls: None),
    )
    monkeypatch.setattr(
        "traigent.cli.next_steps_command.BackendConfig.get_api_key",
        lambda: None,
    )

    result = runner.invoke(cli, ["next-steps", "run_123"])

    assert result.exit_code == 0, result.output
    assert captured["backend_url"] == DEFAULT_LOCAL_URL
    assert "portal.traigent.ai" not in str(captured["backend_url"])
