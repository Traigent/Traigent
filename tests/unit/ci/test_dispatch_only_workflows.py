"""Dispatch-only workflows must not depend on push/pull_request event data.

Regression for Traigent#3168: workflows cut to ``workflow_dispatch`` kept
conditions and refs that only exist on push/pull_request events, so jobs were
silently skipped (tests.yml integration-tests) or compared a commit with
itself (traigent-ci-gates.yml).
"""

from __future__ import annotations

import re
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


def test_ci_gates_base_is_an_input_and_guarded() -> None:
    text = (_WORKFLOWS / "traigent-ci-gates.yml").read_text(encoding="utf-8")
    assert "inputs.base_ref" in text
    assert "base_ref:" in text
    assert "same commit" in text, "gate must fail when base == head"


def test_discovery_finds_named_workflows() -> None:
    names = {p.name for p in _dispatch_only()}
    assert set(_NAMED) <= names
