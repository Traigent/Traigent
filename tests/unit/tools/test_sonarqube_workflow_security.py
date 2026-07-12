"""Security invariants for the privileged local SonarQube workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "sonarqube-local.yml"


def _workflow() -> dict[str, Any]:
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict), "sonarqube-local.yml must parse as a mapping"
    return workflow


def test_privileged_sonarqube_job_has_only_trusted_triggers() -> None:
    workflow = _workflow()
    job = workflow["jobs"]["sonarqube-quality-gate"]
    triggers = workflow.get("on", workflow.get(True))

    assert job["runs-on"] == "large"
    assert "SONAR_TOKEN" in workflow["env"]
    assert isinstance(triggers, dict)
    assert "pull_request" not in triggers
    assert "pull_request_target" not in triggers
    assert triggers["push"]["branches"] == ["main"]
    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers
    assert "if" not in job


def test_checkout_uses_only_the_exact_trusted_event_sha() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["sonarqube-quality-gate"]["steps"]

    checkouts = [step for step in steps if step.get("uses") == "actions/checkout@v7"]
    assert len(checkouts) == 1
    trusted_checkout = checkouts[0]
    assert trusted_checkout["name"] == "Checkout trusted event ref"
    assert "if" not in trusted_checkout
    assert trusted_checkout["with"]["ref"] == "${{ github.sha }}"
    assert trusted_checkout["with"]["persist-credentials"] is False

    analyze = next(
        step for step in steps if step["name"] == "Analyze and enforce quality gate"
    )
    assert "${{ github.head_ref }}" not in analyze["run"]
    assert "-Dsonar.qualitygate.wait=true" in analyze["run"]
