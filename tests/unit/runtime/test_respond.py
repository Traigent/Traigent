"""respond: human approval, re-checks, one occurrence, existing .optimize() with a cost limit."""

from __future__ import annotations

import dataclasses
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from click.testing import CliRunner

import traigent
from traigent.cli import requirements_commands
from traigent.cli.main import cli
from traigent.core.optimized_function import OptimizedFunction
from traigent.runtime import manifest
from traigent.runtime.apply import apply_requirement
from traigent.runtime.model import RuntimeRequirementError
from traigent.runtime.respond import respond

from .conftest import (
    AGENT,
    FIXTURES,
    FIX_MODULE,
    FRAGMENT_MODULE,
    PROJECT,
    FakeBackend,
    breach,
    head,
)

CREATE = ("POST", "/production-requirements")
LIST = ("GET", "/production-requirements")
BREACHES = ("GET", f"/agents/{AGENT}/production-breaches")
NEXT = ("GET", f"/agent-readiness/{AGENT}/next-step")
RESTORE = {"step": {"code": "RESTORE_PRODUCTION_REQUIREMENT"}}


@pytest.fixture
def live(backend: FakeBackend) -> FakeBackend:
    """A registered requirement with one open breach and a RESTORE next step."""
    backend.on(*CREATE, 201, head())
    backend.on(*LIST, 200, {"items": [head()]})
    backend.on(*BREACHES, 200, {"items": [breach("b1")], "has_more": False})
    backend.on(*NEXT, 200, RESTORE)
    return backend


@pytest.fixture
def optimize_mock(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=SimpleNamespace(optimization_id="run-42"))
    monkeypatch.setattr(OptimizedFunction, "optimize", mock)
    return mock


@pytest.fixture(autouse=True)
def _tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requirements_commands, "_interactive", lambda: True)


def _cli(path: Path, command: str, *args: str, input: str | None = None):
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
            "--agent-id",
            AGENT,
            *args,
        ],
        input=input,
    )


def _applied(path: Path) -> None:
    assert _cli(path, "apply").exit_code == 0


def test_happy_path_runs_existing_optimize_with_the_cost_limit(
    live: FakeBackend, optimize_mock: AsyncMock, user_module
) -> None:
    path = user_module(FIX_MODULE)
    _applied(path)
    result = _cli(path, "respond", input="y\n")
    assert result.exit_code == 0, result.output
    assert "run id: run-42" in result.output and "cost ceiling: 5.0" in result.output
    optimize_mock.assert_awaited_once_with(cost_limit=5.0)
    state = manifest.default_state_dir()
    approvals = json.loads(
        (next((state / "runtime").iterdir()) / "approvals.json").read_text()
    )
    (record,) = approvals.values()
    assert record["cost_ceiling_usd"] == 5.0 and record["rule_id"] == "fix"
    assert record["breach_id"] == "b1" and record["occurrence"] == 1


def test_same_occurrence_twice_is_refused(
    live: FakeBackend, optimize_mock: AsyncMock, user_module
) -> None:
    path = user_module(FIX_MODULE)
    _applied(path)
    assert _cli(path, "respond", input="y\n").exit_code == 0
    again = _cli(path, "respond", input="y\n")
    assert again.exit_code == 1
    assert "occurrence_already_used" in again.output and "run-42" in again.output
    assert optimize_mock.await_count == 1


def test_declining_the_prompt_runs_nothing(
    live: FakeBackend, optimize_mock: AsyncMock, user_module
) -> None:
    path = user_module(FIX_MODULE)
    _applied(path)
    result = _cli(path, "respond", input="n\n")
    assert result.exit_code == 1 and "approval_required" in result.output
    optimize_mock.assert_not_awaited()


def test_no_tty_means_approval_required(
    live: FakeBackend,
    optimize_mock: AsyncMock,
    user_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(requirements_commands, "_interactive", lambda: False)
    path = user_module(FIX_MODULE)
    _applied(path)
    result = _cli(path, "respond", input="y\n")
    assert result.exit_code == 1 and "approval_required" in result.output
    optimize_mock.assert_not_awaited()


def _fragment_module(user_module, extra: str = "") -> Path:
    return user_module(
        FRAGMENT_MODULE.format(
            fragment=str(FIXTURES / "latency.tvl.yml"),
            extra=extra + ", agent_id='agent-1'",
        )
    )


@pytest.mark.parametrize("ceiling", ["0", "-1", "nan", "inf"])
def test_missing_or_invalid_ceiling_is_refused(
    live: FakeBackend, optimize_mock: AsyncMock, user_module, ceiling: str
) -> None:
    path = _fragment_module(
        user_module
    )  # fragment with an optimize rule and no ceiling
    result = _cli(path, "respond", "--max-fix-cost-usd", ceiling, input="y\n")
    assert result.exit_code == 1 and "fix_ceiling_required" in result.output
    no_flag = _cli(path, "respond", input="y\n")
    assert no_flag.exit_code == 1 and "fix_ceiling_required" in no_flag.output
    optimize_mock.assert_not_awaited()


def test_cli_ceiling_flag_is_bound_into_the_run(
    live: FakeBackend,
    optimize_mock: AsyncMock,
    user_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from traigent.runtime import apply as apply_mod

    fixture = json.loads((FIXTURES / "hosted_contract.json").read_text())
    monkeypatch.setattr(
        apply_mod,
        "compile_hosted_subset",
        lambda *_a, **_k: (fixture["contract"], {}, []),
    )
    live.on(
        "POST", "/production-requirements/tvl-import", 201, head(provenance="tvl_spec")
    )
    path = _fragment_module(user_module)
    _applied(path)
    result = _cli(path, "respond", "--max-fix-cost-usd", "3", input="y\n")
    assert result.exit_code == 0, result.output
    optimize_mock.assert_awaited_once_with(cost_limit=3.0)


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        (
            lambda b: b.on(*LIST, 200, {"items": [head(status="retired")]}),
            "requirement_retired",
        ),
        (
            lambda b: b.on(*LIST, 200, {"items": [head(revision_id="rev-2")]}),
            "receipt_not_current",
        ),
        (lambda b: b.on(*LIST, 200, {"items": []}), "receipt_not_current"),
        (
            lambda b: b.on(*BREACHES, 200, {"items": [], "has_more": False}),
            "no_open_breach",
        ),
        (
            lambda b: b.on(
                *BREACHES,
                200,
                {"items": [breach("c", outcome="resolved", closed_at="x")]},
            ),
            "no_open_breach",
        ),
        (
            lambda b: b.on(*NEXT, 200, {"step": {"code": "WAIT_FOR_EVALUATION_RUN"}}),
            "next_step_not_restore",
        ),
        (lambda b: b.on(*NEXT, 404, {"error_code": "not_found"}), "not_found"),
    ],
    ids=[
        "retired",
        "revision_moved",
        "head_missing",
        "no_breach",
        "breach_closed",
        "not_restore",
        "step_unreadable",
    ],
)
def test_failed_check_holds_and_nothing_runs(
    live: FakeBackend, optimize_mock: AsyncMock, user_module, setup, code: str
) -> None:
    path = user_module(FIX_MODULE)
    _applied(path)
    setup(live)
    result = _cli(path, "respond", input="y\n")
    assert result.exit_code == 1 and code in result.output
    optimize_mock.assert_not_awaited()


def test_local_policy_changed_since_apply_holds(
    live: FakeBackend, optimize_mock: AsyncMock, user_module
) -> None:
    path = user_module(FIX_MODULE)
    _applied(path)
    state = manifest.default_state_dir()
    mpath = manifest.manifest_path(state, AGENT)
    doc = json.loads(mpath.read_text())
    doc["requirements"]["latency"]["full_policy_digest"] = "0" * 64
    mpath.write_text(json.dumps(doc))
    result = _cli(path, "respond", input="y\n")
    assert result.exit_code == 1 and "policy_changed" in result.output
    optimize_mock.assert_not_awaited()


def test_never_applied_holds(
    live: FakeBackend, optimize_mock: AsyncMock, user_module
) -> None:
    result = _cli(user_module(FIX_MODULE), "respond", input="y\n")
    assert result.exit_code == 1 and "requirement_not_applied" in result.output


def test_alert_only_policy_has_nothing_to_respond_with(
    live: FakeBackend, optimize_mock: AsyncMock, user_module
) -> None:
    path = user_module(FIX_MODULE.replace('on_breach="fix"', 'on_breach="alert"'))
    _applied(path)
    result = _cli(path, "respond", input="y\n")
    assert result.exit_code == 1 and "no_optimize_rule" in result.output


# --- direct API: expiry and rule removal after approval ----------------------------------------


def _fn_and_req():
    @traigent.requirement(
        latency_ms=2000, max_violation_rate=0.05, on_breach="fix", max_fix_cost_usd=2.0
    )
    def fn() -> None: ...

    fn.optimize = AsyncMock(return_value=SimpleNamespace(optimization_id="r"))
    return fn, fn.requirements[0]


async def test_expired_approval_holds(live: FakeBackend, state_dir: Path) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    base = manifest.utcnow()
    ticks = iter([base, base + timedelta(hours=25), base + timedelta(hours=25)])
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(
            fn,
            req,
            project_id=PROJECT,
            agent_id=AGENT,
            confirm=lambda _f: True,
            interactive=True,
            state_dir=state_dir,
            clock=lambda: next(ticks, base + timedelta(hours=25)),
        )
    assert err.value.code == "approval_expired"
    fn.optimize.assert_not_awaited()


async def test_policy_edit_after_approval_invalidates_it(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)

    def confirm_then_reapply_a_changed_policy(_facts) -> bool:
        mpath = manifest.manifest_path(state_dir, AGENT)
        doc = json.loads(mpath.read_text())
        doc["requirements"][req.name]["full_policy_digest"] = "f" * 64
        mpath.write_text(json.dumps(doc))
        return True

    with pytest.raises(RuntimeRequirementError) as err:
        await respond(
            fn,
            req,
            project_id=PROJECT,
            agent_id=AGENT,
            confirm=confirm_then_reapply_a_changed_policy,
            interactive=True,
            state_dir=state_dir,
        )
    assert err.value.code == "policy_changed"
    fn.optimize.assert_not_awaited()


async def test_facts_shown_to_the_person_are_closed_numbers_and_ids(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    shown: dict = {}

    def confirm(facts) -> bool:
        shown.update(facts)
        return True

    result = await respond(
        fn,
        req,
        project_id=PROJECT,
        agent_id=AGENT,
        confirm=confirm,
        interactive=True,
        state_dir=state_dir,
    )
    assert result.run_id == "r" and shown["cost_ceiling_usd"] == 2.0
    assert set(shown) == {
        "requirement",
        "breach_id",
        "opened_at",
        "latency_threshold_ms",
        "environment",
        "rule_id",
        "occurrence",
        "cost_ceiling_usd",
    }
    fn.optimize.assert_awaited_once_with(cost_limit=2.0)


async def test_optimize_failure_is_closed_and_a_retry_needs_a_new_approval(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)
    sentinel = "SECRET_USER_ERROR_TEXT"
    fn.optimize = AsyncMock(side_effect=RuntimeError(sentinel))
    approvals: list[int] = []

    def confirm(_f) -> bool:
        approvals.append(1)
        return True

    kw = {
        "project_id": PROJECT,
        "agent_id": AGENT,
        "confirm": confirm,
        "interactive": True,
        "state_dir": state_dir,
    }
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(fn, req, **kw)
    assert err.value.code == "optimize_failed" and sentinel not in repr(err.value)
    key = manifest.occurrence_key("b1", "fix", 1)
    record = manifest.read_occurrence(state_dir, AGENT, key)
    assert record["status"] == "failed" and record["run_id"] is None

    fn.optimize = AsyncMock(return_value=SimpleNamespace(optimization_id="retry-run"))
    result = await respond(fn, req, **kw)  # explicit retry, a person is asked again
    assert result.run_id == "retry-run" and len(approvals) == 2
    assert manifest.read_occurrence(state_dir, AGENT, key)["attempts"] == 2


async def test_declaration_changed_while_the_person_decides_holds(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)

    def confirm_then_remove_rule(_f) -> bool:
        fn.requirements.clear()  # the optimize rule vanishes from the live declaration
        return True

    with pytest.raises(RuntimeRequirementError) as err:
        await respond(
            fn,
            req,
            project_id=PROJECT,
            agent_id=AGENT,
            confirm=confirm_then_remove_rule,
            interactive=True,
            state_dir=state_dir,
        )
    assert err.value.code == "policy_changed"
    fn.optimize.assert_not_awaited()


def _rule_on(event: str):
    from dataclasses import replace

    from traigent.runtime.model import ResponseRule

    _fn, req = _fn_and_req()
    rules = tuple(
        ResponseRule(r.id, event, r.action, r.severity, r.approval)
        if r.id == "fix"
        else r
        for r in req.rules
    )
    return replace(req, rules=rules)


@pytest.mark.parametrize("event", ["episode_resolved", "episode_withdrawn"])
def test_optimize_and_next_step_rules_only_on_opened_episodes(event: str) -> None:
    issues = [i for i in _rule_on(event).validate() if i.severity == "error"]
    assert issues and issues[0].code == "response_action_unsupported"
    assert issues[0].path.endswith("trigger.event")


@pytest.mark.parametrize("event", ["episode_resolved", "episode_withdrawn"])
async def test_respond_refuses_a_fix_rule_on_a_closing_event(
    live: FakeBackend, state_dir: Path, event: str
) -> None:
    fn, _ = _fn_and_req()
    with pytest.raises(RuntimeRequirementError) as err:
        await respond(
            fn,
            _rule_on(event),
            project_id=PROJECT,
            agent_id=AGENT,
            confirm=lambda _f: True,
            interactive=True,
            state_dir=state_dir,
        )
    assert err.value.code == "response_action_unsupported"
    fn.optimize.assert_not_awaited()


def test_cli_prints_closed_optimize_failed(
    live: FakeBackend, user_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        OptimizedFunction, "optimize", AsyncMock(side_effect=RuntimeError("SECRET_X"))
    )
    path = user_module(FIX_MODULE)
    _applied(path)
    result = _cli(path, "respond", input="y\n")
    assert result.exit_code == 1 and "optimize_failed" in result.output
    assert "SECRET_X" not in result.output


async def test_declared_ceiling_changed_during_confirm_is_refused(
    live: FakeBackend, state_dir: Path
) -> None:
    fn, req = _fn_and_req()
    apply_requirement(req, project_id=PROJECT, agent_id=AGENT, state_dir=state_dir)

    def confirm_then_raise_the_ceiling(_f) -> bool:
        # Same policy digest (the ceiling is not in it), different declared ceiling.
        fn.requirements[:] = [dataclasses.replace(req, max_fix_cost_usd=50.0)]
        return True

    with pytest.raises(RuntimeRequirementError) as err:
        await respond(
            fn,
            req,
            project_id=PROJECT,
            agent_id=AGENT,
            confirm=confirm_then_raise_the_ceiling,
            interactive=True,
            state_dir=state_dir,
        )
    assert err.value.code == "policy_changed"
    fn.optimize.assert_not_awaited()
