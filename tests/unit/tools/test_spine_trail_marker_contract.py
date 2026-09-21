"""Execution-level contract tests for the hosted spine marker gate."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml


WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / ".github"
    / "workflows"
    / "spine-trail-gate.yml"
)


def _gate_script(event_name: str = "pull_request") -> str:
    """Extract the workflow's actual blocking shell step and render its event."""
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["spine-trail"]["steps"]
    step = next(
        step for step in steps if step["name"] == "Check PR body for a spine mark"
    )
    return step["run"].replace("${{ github.event_name }}", event_name)


def _run_gate(
    body: str, *, author: str = "contributor"
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"PR_BODY": body, "PR_AUTHOR": author})
    return subprocess.run(
        ["bash", "-c", _gate_script()],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "body",
    [
        "Spine-Session: cs_9836c68d6634d1c1",
        "Spine: cs_9836c68d6634d1c1",
        "Spine-Trail: st_012345abcdef",
        "Spine: none (reason: documentation-only correction)",
        "Summary before the marker\n  Spine-Session: cs_9836c68d6634d1c1  \n",
    ],
)
def test_gate_accepts_supported_markers(body: str) -> None:
    result = _run_gate(body)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "body",
    [
        "",
        "Spine-Session: cs_12345",
        "Spine Session: cs_9836c68d6634d1c1",
        "Spine-Session: cs_9836c68d6634d1c1 trailing-text",
        "Spine-Trail: st_012345abcde",
        "Spine: none (reason:)",
    ],
)
def test_gate_rejects_missing_or_malformed_markers(body: str) -> None:
    result = _run_gate(body)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Spine trail missing" in result.stdout


def test_bodyless_exemption_remains_limited_to_dependabot() -> None:
    dependabot = _run_gate("", author="dependabot[bot]")
    other_bot = _run_gate("", author="renovate[bot]")

    assert dependabot.returncode == 0, dependabot.stdout + dependabot.stderr
    assert other_bot.returncode == 1, other_bot.stdout + other_bot.stderr
