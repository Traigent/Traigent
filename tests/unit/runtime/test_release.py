"""Release and undo of an agent head: SDK entry points against a local fake backend."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import traigent
from traigent.cli.main import cli
from traigent.runtime import release as rel
from traigent.runtime.client import RuntimeClient

from .conftest import FakeBackend

pytestmark = pytest.mark.backend_online

PROJECT = "proj-1"
REF = "agentref_" + "ab" * 16
BASE = f"/api/v1/projects/{PROJECT}/agents/{REF}/best-config-head"
DIGEST = "c" * 64
VERSION = "git_abc123"

MESSAGES = {
    "invalid_request": "Request is invalid.",
    "unauthenticated": "Authentication required.",
    "forbidden": "Forbidden.",
    "reference_not_found": "Reference not found.",
    "acceptance_required": "A release acceptance is required.",
    "acceptance_expired": "The release acceptance has expired.",
    "acceptance_consumed": "The release acceptance was already used.",
    "acceptance_revoked": "The release acceptance was revoked.",
    "acceptance_binding_mismatch": "The release does not match the acceptance.",
    "head_generation_stale": "Agent head moved since the acceptance.",
    "rollback_target_not_accepted": "The rollback target was not accepted.",
    "internal_error": "Internal error.",
}
STATUS = {
    "invalid_request": 400,
    "unauthenticated": 401,
    "forbidden": 403,
    "reference_not_found": 404,
    "internal_error": 500,
}


def refusal(code: str, **over: Any) -> dict[str, Any]:
    broad = {
        "invalid_request": "invalid_request",
        "unauthenticated": "unauthenticated",
        "forbidden": "forbidden",
        "reference_not_found": "reference_not_found",
        "internal_error": "internal_error",
    }.get(code, "release_precondition_refused")
    body = {
        "success": False,
        "error": broad,
        "error_code": code,
        "message": MESSAGES[code],
    }
    body.update(over)
    return body


def receipt(generation: int = 4, **over: Any) -> dict[str, Any]:
    data = {
        "agent_id": REF,
        "environment": "production",
        "generation": generation,
        "best_config_id": "bc_1",
        "best_config_version": 7,
        "acceptance_id": "acc_1",
    }
    data.update(over)
    return {"success": True, "message": "Release recorded", "data": data}


@pytest.fixture
def client(backend: FakeBackend) -> RuntimeClient:
    return RuntimeClient(PROJECT)


def release_kwargs(**over: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "acceptance_id": "acc_1",
        "expected_head_generation": 3,
        "agent_code_version": VERSION,
        "configuration_space_digest": DIGEST,
    }
    kw.update(over)
    return kw


def test_release_success_sends_the_closed_body_and_returns_a_typed_receipt(
    backend: FakeBackend, client: RuntimeClient
) -> None:
    backend.on("POST", f"{BASE}/release", 200, receipt(4))
    got = rel.release_agent_head(client, REF, **release_kwargs())
    assert got == rel.ReleaseReceipt(REF, "production", 4, "bc_1", 7, "acc_1")
    assert backend.bodies() == [
        {
            "acceptance_id": "acc_1",
            "expected_head_generation": 3,
            "agent_code_version": VERSION,
            "configuration_space_digest": DIGEST,
            "gate_result_digest": "no_gate_run",
        }
    ]


@pytest.mark.parametrize("code", sorted(MESSAGES))
def test_every_closed_refusal_code_maps_to_its_own_error(
    backend: FakeBackend, client: RuntimeClient, code: str
) -> None:
    backend.on("POST", f"{BASE}/release", STATUS.get(code, 409), refusal(code))
    with pytest.raises(rel.ReleaseRefused) as info:
        rel.release_agent_head(client, REF, **release_kwargs())
    assert info.value.code == code
    assert str(info.value).startswith(f"{code}: ")


def test_acceptance_required_message_names_the_acceptance(
    backend: FakeBackend, client: RuntimeClient
) -> None:
    backend.on("POST", f"{BASE}/release", 409, refusal("acceptance_required"))
    with pytest.raises(rel.ReleaseRefused) as info:
        rel.release_agent_head(client, REF, **release_kwargs())
    assert (
        str(info.value)
        == "acceptance_required: a person must accept this configuration first"
    )


def test_server_text_is_never_echoed(
    backend: FakeBackend, client: RuntimeClient
) -> None:
    """A well-formed code with a foreign message is outside the envelope: backend_error."""
    body = refusal("acceptance_required", message="Alice_has_diabetes SECRET")
    backend.on("POST", f"{BASE}/release", 409, body)
    with pytest.raises(rel.ReleaseRefused) as info:
        rel.release_agent_head(client, REF, **release_kwargs())
    assert info.value.code == "backend_error"
    assert "Alice" not in str(info.value) and "SECRET" not in str(info.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"error_code": "acceptance_required"},
        refusal("acceptance_required", extra="x"),
        {"unrelated": 1},
    ],
)
def test_a_body_outside_the_closed_envelope_is_backend_error(
    backend: FakeBackend, client: RuntimeClient, payload: dict[str, Any]
) -> None:
    backend.on("POST", f"{BASE}/release", 409, payload)
    with pytest.raises(rel.ReleaseRefused) as info:
        rel.release_agent_head(client, REF, **release_kwargs())
    assert info.value.code == "backend_error"


def test_a_code_paired_with_the_wrong_status_is_backend_error(
    backend: FakeBackend, client: RuntimeClient
) -> None:
    backend.on("POST", f"{BASE}/release", 401, refusal("forbidden"))
    with pytest.raises(rel.ReleaseRefused) as info:
        rel.release_agent_head(client, REF, **release_kwargs())
    assert info.value.code == "backend_error"


def test_a_malformed_success_body_is_backend_error(
    backend: FakeBackend, client: RuntimeClient
) -> None:
    backend.on(
        "POST", f"{BASE}/release", 200, receipt(4, agent_id="agentref_" + "0" * 32)
    )
    with pytest.raises(rel.ReleaseRefused) as info:
        rel.release_agent_head(client, REF, **release_kwargs())
    assert info.value.code == "backend_error"


def test_bad_inputs_are_refused_before_any_request(
    backend: FakeBackend, client: RuntimeClient
) -> None:
    for over in (
        {"acceptance_id": "bad id!"},
        {"agent_code_version": "src:" + "a" * 64},
        {"expected_head_generation": -1},
        {"configuration_space_digest": "nope"},
        {"gate_result_digest": "nope"},
    ):
        with pytest.raises(rel.ReleaseRefused):
            rel.release_agent_head(client, REF, **release_kwargs(**over))
    assert backend.requests == []


def test_rollback_sends_the_contract_body(
    backend: FakeBackend, client: RuntimeClient
) -> None:
    backend.on("POST", f"{BASE}/rollback", 200, receipt(5))
    got = rel.rollback_agent_head(
        client,
        REF,
        environment="production",
        target_generation=2,
        expected_head_generation=4,
    )
    assert got.generation == 5
    assert backend.bodies() == [
        {
            "environment": "production",
            "target_generation": 2,
            "expected_head_generation": 4,
        }
    ]


def test_rollback_refusal_maps(backend: FakeBackend, client: RuntimeClient) -> None:
    backend.on("POST", f"{BASE}/rollback", 409, refusal("rollback_target_not_accepted"))
    with pytest.raises(rel.ReleaseRefused) as info:
        rel.rollback_agent_head(
            client,
            REF,
            environment="production",
            target_generation=2,
            expected_head_generation=4,
        )
    assert info.value.code == "rollback_target_not_accepted"


def test_a_backend_without_the_rollback_route_is_a_closed_refusal(
    backend: FakeBackend, client: RuntimeClient
) -> None:
    # The fake answers its own non-envelope 404 for an unknown route.
    with pytest.raises(rel.ReleaseRefused) as info:
        rel.rollback_agent_head(
            client,
            REF,
            environment="production",
            target_generation=2,
            expected_head_generation=4,
        )
    assert info.value.code == "backend_error"


def test_refusal_error_survives_pickling() -> None:
    err = pickle.loads(
        pickle.dumps(rel.ReleaseRefused("head_generation_stale", status=409))
    )
    assert (err.code, err.status) == ("head_generation_stale", 409)


# --- OptimizedFunction ----------------------------------------------------------------------


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> Any:
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", PROJECT)
    monkeypatch.setenv("TRAIGENT_AGENT_REF", REF)
    monkeypatch.setenv("TRAIGENT_AGENT_CODE_VERSION", VERSION)

    @traigent.optimize(
        eval_dataset="data.jsonl",
        objectives=["accuracy"],
        configuration_space={"t": [0.1, 0.2]},
        injection_mode="seamless",
    )
    def answer(q: str, t: float = 0.1) -> str:
        return q

    return answer


def test_function_release_binds_space_digest_and_defaults_to_no_gate(
    backend: FakeBackend, agent: Any
) -> None:
    backend.on("POST", f"{BASE}/release", 200, receipt(4))
    got = agent.release(acceptance_id="acc_1", expected_head_generation=3)
    assert got.generation == 4
    body = backend.bodies()[0]
    assert body["configuration_space_digest"] == rel.configuration_space_digest(
        agent.configuration_space
    )
    assert body["agent_code_version"] == VERSION
    assert body["gate_result_digest"] == "no_gate_run"


def test_function_release_reads_the_live_head_when_no_generation_is_given(
    backend: FakeBackend, agent: Any
) -> None:
    backend.on("GET", BASE, 200, {"success": True, "data": {"generation": 3}})
    backend.on("POST", f"{BASE}/release", 200, receipt(4))
    agent.release(acceptance_id="acc_1", environment="production")
    assert backend.bodies()[0]["expected_head_generation"] == 3


def test_function_release_refusal_has_no_head_move_and_closed_text(
    backend: FakeBackend, agent: Any
) -> None:
    backend.on("POST", f"{BASE}/release", 409, refusal("acceptance_required"))
    with pytest.raises(
        traigent.runtime.release.ReleaseRefused, match="^acceptance_required: "
    ):
        agent.release(acceptance_id="acc_1", expected_head_generation=3)


def test_function_release_needs_an_agent_ref_and_a_project(
    monkeypatch: pytest.MonkeyPatch, agent: Any
) -> None:
    monkeypatch.delenv("TRAIGENT_AGENT_REF")
    with pytest.raises(rel.ReleaseRefused) as info:
        agent.release(acceptance_id="acc_1", expected_head_generation=3)
    assert info.value.code == "agent_ref_required"
    monkeypatch.setenv("TRAIGENT_AGENT_REF", REF)
    monkeypatch.delenv("TRAIGENT_PROJECT_ID")
    with pytest.raises(rel.ReleaseRefused) as info:
        agent.release(acceptance_id="acc_1", expected_head_generation=3)
    assert info.value.code == "project_required"


def test_function_rollback_release(backend: FakeBackend, agent: Any) -> None:
    backend.on("POST", f"{BASE}/rollback", 200, receipt(5))
    got = agent.rollback_release(
        target_generation=2, expected_head_generation=4, environment="production"
    )
    assert got.generation == 5


def test_apply_best_config_docstring_says_it_is_not_a_release() -> None:
    from traigent.core.optimized_function import OptimizedFunction

    doc = OptimizedFunction.apply_best_config.__doc__ or ""
    assert "no gate, no acceptance, not a release" in doc


# --- CLI ----------------------------------------------------------------------------------------


def invoke(*args: str) -> Any:
    return CliRunner().invoke(cli, ["release", *args])


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> FakeBackend:
    monkeypatch.setenv("TRAIGENT_PROJECT_ID", PROJECT)
    monkeypatch.setenv("TRAIGENT_AGENT_REF", REF)
    return backend


def test_cli_release_success(cli_env: FakeBackend) -> None:
    cli_env.on("POST", f"{BASE}/release", 200, receipt(4))
    result = invoke(
        "acc_1",
        "--expected-head-generation",
        "3",
        "--agent-code-version",
        VERSION,
        "--configuration-space-digest",
        DIGEST,
    )
    assert result.exit_code == 0, result.output
    assert "released: environment=production generation=4" in result.output


def test_cli_release_refusal_prints_the_closed_message(cli_env: FakeBackend) -> None:
    cli_env.on("POST", f"{BASE}/release", 409, refusal("acceptance_required"))
    result = invoke(
        "acc_1",
        "--expected-head-generation",
        "3",
        "--agent-code-version",
        VERSION,
        "--configuration-space-digest",
        DIGEST,
    )
    assert result.exit_code == 1
    assert (
        "error: acceptance_required: a person must accept this configuration first"
        in result.output
    )


def test_cli_release_from_a_function(
    cli_env: FakeBackend,
    user_module: Callable[[str], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_AGENT_CODE_VERSION", VERSION)
    cli_env.on("POST", f"{BASE}/release", 200, receipt(4))
    path = user_module(
        "import traigent\n\n"
        "@traigent.optimize(eval_dataset='data.jsonl', objectives=['accuracy'],\n"
        "    configuration_space={'t': [0.1, 0.2]}, injection_mode='seamless')\n"
        "def answer(q: str, t: float = 0.1) -> str:\n    return q\n"
    )
    result = invoke(
        "acc_1", "--file", str(path), "-f", "answer", "--expected-head-generation", "3"
    )
    assert result.exit_code == 0, result.output
    assert cli_env.bodies()[0]["agent_code_version"] == VERSION


def test_cli_release_without_inputs_is_a_usage_error(cli_env: FakeBackend) -> None:
    result = invoke("acc_1")
    assert result.exit_code == 2
    assert cli_env.requests == []


def test_cli_rollback(cli_env: FakeBackend) -> None:
    cli_env.on("POST", f"{BASE}/rollback", 200, receipt(5))
    result = invoke(
        "rollback",
        "2",
        "--expected-head-generation",
        "4",
        "--environment",
        "production",
    )
    assert result.exit_code == 0, result.output
    assert "rolled back: environment=production generation=5" in result.output
    assert cli_env.bodies() == [
        {
            "environment": "production",
            "target_generation": 2,
            "expected_head_generation": 4,
        }
    ]


def test_cli_rollback_refusal(cli_env: FakeBackend) -> None:
    cli_env.on("POST", f"{BASE}/rollback", 409, refusal("rollback_target_not_accepted"))
    result = invoke(
        "rollback",
        "2",
        "--expected-head-generation",
        "4",
        "--environment",
        "production",
    )
    assert result.exit_code == 1
    assert "error: rollback_target_not_accepted: " in result.output
