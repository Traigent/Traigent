"""apply: idempotency against a real local HTTP server, fail-closed ambiguity, CLI exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import traigent
from traigent.cli.main import cli
from traigent.runtime import apply as apply_mod
from traigent.runtime import manifest
from traigent.runtime.apply import ApplyConflict, apply_requirement
from traigent.runtime.model import RuntimeRequirementError

from .conftest import (
    AGENT,
    FIXTURES,
    FIX_MODULE,
    FRAGMENT_MODULE,
    PROJECT,
    FakeBackend,
    backend_parse_import,
    head,
)

pytestmark = pytest.mark.backend_online

CREATE = ("POST", "/production-requirements")
IMPORT = ("POST", "/production-requirements/tvl-import")
LIST = ("GET", "/production-requirements")
EXISTS = (
    409,
    {
        "success": False,
        "error": "requirement_exists",
        "error_code": "requirement_exists",
    },
)


def _req(**kw):
    @traigent.requirement(
        **{"latency_ms": 2000, "max_violation_rate": 0.05, "on_breach": "alert", **kw}
    )
    def fn() -> None: ...

    return fn.requirements[0]


def _apply(req, **kw):
    return apply_requirement(req, project_id=PROJECT, agent_id=AGENT, **kw)


def test_native_create_201_then_noop_on_409_with_equal_active_profile(
    backend: FakeBackend,
) -> None:
    backend.on(*CREATE, 201, head())
    result = _apply(_req())
    assert (result.outcome, result.route, result.receipt.created) == (
        "created",
        "native",
        True,
    )
    assert result.receipt.provenance == "api"
    sent = backend.requests[0]["body"]
    assert sent["agent_id"] == AGENT and sent["latency_threshold_ms"] == 2000
    assert sent["max_violation_rate"] == 0.05 and sent["window_hours"] == 6

    backend.routes[CREATE] = lambda _b: EXISTS
    backend.on(*LIST, 200, {"items": [head()]})
    again = _apply(_req())
    assert (again.outcome, again.receipt.created) == ("noop", False)
    manifest_doc = manifest.read_manifest(manifest.default_state_dir(), AGENT)
    entry = manifest_doc["requirements"]["latency"]
    assert entry["receipt"]["revision_id"] == "rev-1" and entry["route"] == "native"
    assert entry["full_policy_digest"] == again.policy_digest


@pytest.mark.parametrize(
    "items",
    [
        [head(status="retired")],  # equal profile but retired
        [],  # missing
        [head(), head(head_id="head-2")],  # ambiguous: two active heads
        [head(project_id="other-project")],  # not this project
        [head(agent_id="other-agent")],  # not this agent
        [head(latency_threshold_ms=2500)],  # other key
    ],
    ids=[
        "retired",
        "missing",
        "ambiguous",
        "other_project",
        "other_agent",
        "other_key",
    ],
)
def test_409_without_exactly_one_matching_active_head_fails_closed(
    backend: FakeBackend, items: list
) -> None:
    backend.routes[CREATE] = lambda _b: EXISTS
    backend.on(*LIST, 200, {"items": items})
    with pytest.raises(RuntimeRequirementError) as err:
        _apply(_req())
    assert err.value.code == "requirement_exists"
    assert not isinstance(err.value, ApplyConflict)


def test_409_with_different_profile_reports_difference(backend: FakeBackend) -> None:
    backend.routes[CREATE] = lambda _b: EXISTS
    backend.on(*LIST, 200, {"items": [head(max_violation_rate=0.1)]})
    with pytest.raises(ApplyConflict) as err:
        _apply(_req())
    assert err.value.diff == [("max_violation_rate", 0.05, 0.1)]


def test_dry_run_makes_no_request(backend: FakeBackend) -> None:
    result = _apply(_req(), dry_run=True)
    assert result.outcome == "dry_run" and result.receipt is None
    assert backend.requests == []
    assert not (manifest.default_state_dir() / "tvl-token-map.json").exists()


def test_dry_run_needs_no_agent_or_project(backend: FakeBackend) -> None:
    result = apply_requirement(_req(), project_id="", dry_run=True)
    assert result.outcome == "dry_run" and backend.requests == []


def test_agent_is_never_guessed(backend: FakeBackend) -> None:
    with pytest.raises(RuntimeRequirementError) as err:
        apply_requirement(_req(), project_id=PROJECT)
    assert err.value.code == "agent_id_required"
    assert backend.requests == []


def test_flag_off_is_a_closed_error(backend: FakeBackend) -> None:
    backend.on(
        *CREATE, 404, {"error_code": "production_requirements_disabled", "error": "x"}
    )
    with pytest.raises(RuntimeRequirementError) as err:
        _apply(_req())
    assert err.value.code == "production_requirements_disabled"


def test_unknown_backend_code_and_body_are_never_echoed(backend: FakeBackend) -> None:
    backend.on(
        *CREATE, 500, {"error_code": "SECRET_BODY_TEXT", "error": "SECRET_BODY_TEXT"}
    )
    with pytest.raises(RuntimeRequirementError) as err:
        _apply(_req())
    assert err.value.code == "backend_error" and "SECRET_BODY_TEXT" not in str(
        err.value
    )


def test_redirects_are_refused(backend: FakeBackend) -> None:
    backend.routes[CREATE] = lambda _b: (302, {"error_code": "x"})
    with pytest.raises(RuntimeRequirementError) as err:
        _apply(_req())
    assert err.value.code == "backend_error"  # a 302 is an error, never followed
    assert len(backend.requests) == 1


# --- TVL route (compiler stubbed with the checked-in compiled fixture) ---------------------


@pytest.fixture
def fragment_requirement(monkeypatch: pytest.MonkeyPatch):
    fixture = json.loads((FIXTURES / "hosted_contract.json").read_text())

    def honoring_stub(requirement, token_map, module):
        """The checked-in compiled contract, re-tokenized with the refs apply minted."""
        actual = {
            (t["namespace"], t["local"]): t["hosted"] for t in token_map["tokens"]
        }
        wanted = {
            "module": module,
            "requirement": requirement.name,
            "monitor": requirement.monitor_name,
            "environment": requirement.environment.strip().lower(),
        }
        text = json.dumps(fixture["contract"])
        for entry in fixture["token_map"]["tokens"]:
            ns = entry["namespace"]
            if ns in wanted:
                text = text.replace(
                    f'"{entry["hosted"]}"', f'"{actual[(ns, wanted[ns])]}"'
                )
        return json.loads(text), {}, []

    monkeypatch.setattr(apply_mod, "compile_hosted_subset", honoring_stub)

    @traigent.requirement(tvl=str(FIXTURES / "latency.tvl.yml"), max_fix_cost_usd=5.0)
    def fn() -> None: ...

    return fn.requirements[0]


def test_tvl_route_imports_201_then_200(
    backend: FakeBackend, fragment_requirement
) -> None:
    backend.on(*IMPORT, 201, head(provenance="tvl_spec"))
    first = _apply(fragment_requirement)
    assert (first.outcome, first.route) == ("created", "tvl_import")
    body = backend.requests[0]["body"]
    assert backend_parse_import(body) == "OK"  # the complete request, real importer
    assert set(body) == {"agent_id", "contract", "environment_binding"}
    assert body["contract"]["declarations"]["responses"] == []
    assert body["environment_binding"]["label"] == "production"

    backend.on(*IMPORT, 200, head(provenance="tvl_spec"))
    second = _apply(fragment_requirement)
    assert (second.outcome, second.receipt.created) == ("noop", False)


def test_tvl_409_reports_difference_or_fails_closed(
    backend: FakeBackend, fragment_requirement
) -> None:
    backend.routes[IMPORT] = lambda _b: EXISTS
    backend.on(*LIST, 200, {"items": [head(window_hours=12)]})
    with pytest.raises(ApplyConflict):
        _apply(fragment_requirement)
    backend.on(*LIST, 200, {"items": [head()]})  # equal profile, different provenance
    with pytest.raises(RuntimeRequirementError) as err:
        _apply(fragment_requirement)
    assert err.value.code == "requirement_exists" and not isinstance(
        err.value, ApplyConflict
    )


# --- CLI ---------------------------------------------------------------------------------------


def _invoke(path: Path, *args: str, command: str = "apply"):
    return CliRunner().invoke(
        cli,
        [
            "requirements",
            command,
            str(path),
            "--function",
            "answer",
            "--project-id",
            PROJECT,
            *args,
        ],
    )


def test_cli_apply_idempotent_and_exit_codes(backend: FakeBackend, user_module) -> None:
    path = user_module(FIX_MODULE)
    backend.on(*CREATE, 201, head())
    ok = _invoke(path)
    assert ok.exit_code == 0, ok.output
    assert "created via native" in ok.output and "head=head-1" in ok.output

    backend.routes[CREATE] = lambda _b: EXISTS
    backend.on(*LIST, 200, {"items": [head()]})
    noop = _invoke(path)
    assert noop.exit_code == 0 and "noop" in noop.output

    backend.on(*LIST, 200, {"items": [head(confidence=0.99)]})
    diff = _invoke(path)
    assert diff.exit_code == 3
    assert "confidence" in diff.output and "declared=0.95" in diff.output

    backend.on(*LIST, 200, {"items": [head(status="retired")]})
    closed = _invoke(path)
    assert closed.exit_code == 1 and "requirement_exists" in closed.output


def test_cli_flag_off_exits_nonzero_with_closed_code(
    backend: FakeBackend, user_module
) -> None:
    backend.on(*CREATE, 404, {"error_code": "production_requirements_disabled"})
    result = _invoke(user_module(FIX_MODULE))
    assert result.exit_code == 1
    assert "production_requirements_disabled" in result.output


def test_cli_dry_run_and_validate_make_no_request(
    backend: FakeBackend, user_module
) -> None:
    path = user_module(FIX_MODULE)
    assert _invoke(path, "--dry-run").exit_code == 0
    validated = CliRunner().invoke(
        cli, ["requirements", "validate", str(path), "--function", "answer"]
    )
    assert validated.exit_code == 0 and "tvl_version" in validated.output
    assert backend.requests == []


def test_cli_function_discovery_errors(backend: FakeBackend, user_module) -> None:
    path = user_module(FIX_MODULE)
    missing = CliRunner().invoke(
        cli, ["requirements", "validate", str(path), "--function", "nope"]
    )
    assert missing.exit_code == 1 and "function_not_found" in missing.output


def test_cli_fragment_module_apply_needs_compiler_or_reports_closed_error(
    backend: FakeBackend, user_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "tvl", None)
    monkeypatch.setitem(sys.modules, "tvl.runtime_compiler", None)
    path = user_module(
        FRAGMENT_MODULE.format(
            fragment=str(FIXTURES / "latency.tvl.yml"), extra=", agent_id='agent-1'"
        )
    )
    result = _invoke(path)
    assert result.exit_code == 1 and "tvl_compiler_unavailable" in result.output
    assert backend.requests == []


def test_explain_resolves_ref_locally(backend: FakeBackend, state_dir: Path) -> None:
    doc = manifest.ensure_tokens(state_dir, [("requirement", "my_local_name")])
    ref = doc["tokens"][0]["hosted"]
    result = CliRunner().invoke(cli, ["requirements", "explain", ref])
    assert result.exit_code == 0 and result.output.strip() == "my_local_name"
    assert (
        CliRunner().invoke(cli, ["requirements", "explain", "unknownref"]).exit_code
        == 1
    )


def test_cli_import_time_closed_error_keeps_its_code(
    backend: FakeBackend, user_module
) -> None:
    source = FIX_MODULE.replace("max_fix_cost_usd=5.0, ", "")
    result = CliRunner().invoke(
        cli,
        ["requirements", "validate", str(user_module(source)), "--function", "answer"],
    )
    assert result.exit_code == 1 and "fix_ceiling_required" in result.output
    assert "module_load_failed" not in result.output
