"""Contract checks for the live Redis session test's CI wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _workflow(name: str) -> dict[str, Any]:
    workflow = yaml.safe_load(
        (_REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")
    )
    assert isinstance(workflow, dict)
    return workflow


def _trigger_map(workflow: dict[str, Any]) -> dict[str, Any]:
    # PyYAML 1.1 parses the YAML 1.2 `on` key as boolean True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict)
    return triggers


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_required_pr_gate_unit_owns_live_redis_contract() -> None:
    workflow = _workflow("pr-gate.yml")
    triggers = _trigger_map(workflow)
    assert "pull_request" in triggers
    assert "merge_group" in triggers

    jobs = workflow["jobs"]
    unit = jobs["unit"]
    unit_condition = unit["if"]
    assert "needs.changes.outputs.code_changed == 'true'" in unit_condition
    assert "github.event_name == 'merge_group'" in unit_condition
    assert (
        "github.event.pull_request.head.repo.full_name == github.repository"
        in unit_condition
    )
    assert unit["services"]["redis"]["image"] == "redis:7-alpine"
    assert "6379:6379" in unit["services"]["redis"]["ports"]

    install = _step(unit, "Install test deps")["run"]
    assert 'pip install "redis>=8,<9"' in install
    verify = _step(unit, "Verify redis-py 8")["run"]
    assert "print" in verify and "major == 8" in verify

    live_test = _step(unit, "Run live Redis session test")
    assert "tests/integration/test_session_manager_live_redis.py" in live_test["run"]
    assert live_test["env"]["TRAIGENT_TEST_REDIS_URL"] == "redis://localhost:6379/15"

    required_gate = jobs["required-pr-gate"]
    assert "unit" in required_gate["needs"]


def test_release_review_keeps_live_redis_compatibility_contract() -> None:
    workflow = _workflow("release-review.yml")
    assert "workflow_dispatch" in _trigger_map(workflow)

    integration = workflow["jobs"]["tests_integration"]
    assert integration["services"]["redis"]["image"] == "redis:7-alpine"
    install = _step(integration, "Install test deps")["run"]
    assert 'pip install "redis>=8,<9"' in install
    verify = _step(integration, "Verify redis-py 8")["run"]
    assert "print" in verify and "major == 8" in verify
    run = _step(integration, "Run integration tests")["run"]
    assert 'TRAIGENT_TEST_REDIS_URL="redis://localhost:6379/15"' in run
    assert "pytest tests/integration" in run
