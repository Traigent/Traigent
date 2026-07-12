"""Unit-gate coverage for Planner V2 guidance presentation helpers."""

from __future__ import annotations

from unittest.mock import Mock

from traigent.cli import guidance_command


def _decision_payload(command_template: str | None) -> dict[str, object]:
    return {
        "run_id": "run_123",
        "lifecycle_id": "lifecycle_0123456789abcdef",
        "decision": {
            "category": "run_optimization",
            "mode": "policy_override",
            "utility_profile": "balanced",
            "rationale": "bounded session-utility advantage",
            "evidence_level": "high",
            "advantage_label": "no_kpi_guarantee",
            "action": {
                "variant": "optimize_probe",
                "command_template": command_template,
            },
        },
        "meta": {
            "served_variant": "policy_override",
            "selector_engine": "policy",
            "fallback_reason": None,
        },
    }


def test_print_decision_renders_executable_and_non_executable_actions(
    monkeypatch,
) -> None:
    printer = Mock()
    monkeypatch.setattr(guidance_command.console, "print", printer)

    guidance_command._print_decision(
        _decision_payload("traigent guidance execute --decision decision_123")
    )
    guidance_command._print_decision(_decision_payload(None))

    rendered = "\n".join(str(call.args[0]) for call in printer.call_args_list)
    assert "Planner V2 decision for run_123" in rendered
    assert "traigent guidance execute --decision decision_123" in rendered
    assert "This decision is non-executable" in rendered
    assert "fallback=none" in rendered


def test_backend_and_api_key_resolution_precedence(monkeypatch) -> None:
    monkeypatch.setattr(
        guidance_command.BackendConfig,
        "get_configured_backend_url",
        lambda: "https://stored.example.test",
    )
    monkeypatch.setattr(
        guidance_command.BackendConfig,
        "get_api_key",
        lambda: "stored-key",
    )

    assert (
        guidance_command._resolve_backend_url("https://explicit.example.test")
        == "https://explicit.example.test"
    )
    assert guidance_command._resolve_backend_url(None) == "https://stored.example.test"
    assert guidance_command._resolve_api_key("explicit-key") == "explicit-key"
    assert guidance_command._resolve_api_key(None) == "stored-key"
