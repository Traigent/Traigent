"""Dispatch-only workflows must not depend on push/pull_request event data.

Regression for Traigent#3168: workflows cut to ``workflow_dispatch`` kept
conditions and refs that only exist on push/pull_request events, so jobs were
silently skipped (tests.yml integration-tests) or compared a commit with
itself (traigent-ci-gates.yml).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

_WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"

# Workflows named in #3168. Other dispatch-only workflows are covered by the
# generic discovery below only for the event-dependent patterns.
_FORBIDDEN = [
    re.compile(r"github\.event_name\s*[!=]=\s*'(push|pull_request)'"),
    re.compile(r"\$GITHUB_EVENT_NAME\"?\s*=\s*\"(push|pull_request)\""),
    re.compile(r"github\.event\.pull_request"),
    re.compile(r"github\.event\.head_commit"),
]


def _triggers(path: Path) -> list[str]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    on = doc.get(True, doc.get("on"))
    if isinstance(on, dict):
        return list(on)
    return [on] if isinstance(on, str) else list(on)


def _dispatch_only() -> list[Path]:
    return sorted(
        p for p in _WORKFLOWS.glob("*.yml") if _triggers(p) == ["workflow_dispatch"]
    )


_NAMED = [
    "tests.yml",
    "traigent-ci-gates.yml",
    "release-review.yml",
    "quality.yml",
]


@pytest.mark.parametrize("name", _NAMED)
def test_named_workflows_are_dispatch_only(name: str) -> None:
    assert _triggers(_WORKFLOWS / name) == ["workflow_dispatch"]


@pytest.mark.parametrize("name", _NAMED)
def test_no_push_or_pull_request_logic(name: str) -> None:
    lines = (_WORKFLOWS / name).read_text(encoding="utf-8").splitlines()
    hits = [
        f"{name}:{i}: {line.strip()}"
        for i, line in enumerate(lines, 1)
        if not line.lstrip().startswith("#")
        for pat in _FORBIDDEN
        if pat.search(line)
    ]
    assert not hits, "event-dependent logic in dispatch-only workflow:\n" + "\n".join(
        hits
    )


def test_quality_scans_do_not_swallow_failures() -> None:
    text = (_WORKFLOWS / "quality.yml").read_text(encoding="utf-8")
    bad = [
        f"quality.yml:{i}: {line.strip()}"
        for i, line in enumerate(text.splitlines(), 1)
        if "|| true" in line and not line.lstrip().startswith("#")
    ]
    assert not bad, "scan steps must fail on findings:\n" + "\n".join(bad)


def _guard_script() -> str:
    doc = yaml.safe_load((_WORKFLOWS / "traigent-ci-gates.yml").read_text("utf-8"))
    steps = doc["jobs"]["optimization-gates"]["steps"]
    return next(s["run"] for s in steps if "distinct" in s.get("name", ""))


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _run_guard(tmp_path: Path, same: bool) -> subprocess.CompletedProcess[str]:
    base, head = tmp_path / "baseline", tmp_path / "work"
    for d in (base, head):
        d.mkdir()
        _git(d, "init", "-q")
        (d / "f").write_text("x" if same else d.name)
        _git(d, "add", "f")
        _git(d, "commit", "-q", "-m", "c", "--date=2020-01-01T00:00:00")
    if same:  # make both repos hold the identical commit
        _git(head, "fetch", "-q", str(base), "HEAD")
        _git(head, "reset", "-q", "--hard", "FETCH_HEAD")
    return subprocess.run(
        ["bash", "-e", "-c", _guard_script()],
        cwd=tmp_path,
        env={**os.environ, "BASE_REF": "develop"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_ci_gates_base_is_fixed_to_develop() -> None:
    doc = yaml.safe_load((_WORKFLOWS / "traigent-ci-gates.yml").read_text("utf-8"))
    on = doc.get(True, doc.get("on"))
    # No dispatch input may feed a checkout ref (user-controlled checkout).
    assert not (on["workflow_dispatch"] or {}).get("inputs")
    steps = doc["jobs"]["optimization-gates"]["steps"]
    for step in steps:
        ref = str(step.get("with", {}).get("ref", ""))
        assert "inputs." not in ref
    base = next(s for s in steps if s.get("with", {}).get("path") == "baseline")
    assert base["with"]["ref"] == "develop"


def test_ci_gates_guard_fails_when_base_equals_head(tmp_path: Path) -> None:
    result = _run_guard(tmp_path, same=True)
    assert result.returncode != 0
    assert "same commit" in result.stdout


def test_ci_gates_guard_passes_when_commits_differ(tmp_path: Path) -> None:
    result = _run_guard(tmp_path, same=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_integration_tests_gated_on_dispatch_input() -> None:
    doc = yaml.safe_load((_WORKFLOWS / "tests.yml").read_text("utf-8"))
    on = doc.get(True, doc.get("on"))
    inp = on["workflow_dispatch"]["inputs"]["run_integration"]
    assert inp["type"] == "boolean" and inp["default"] is True
    assert doc["jobs"]["integration-tests"]["if"] == "inputs.run_integration"


def test_discovery_finds_named_workflows() -> None:
    names = {p.name for p in _dispatch_only()}
    assert set(_NAMED) <= names
