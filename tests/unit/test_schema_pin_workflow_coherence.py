"""The SDK pins TraigentSchema in exactly one place.

``scripts/ci/schema-pin.txt`` is the pin the SDK installs from. The pr-gate
workflow's DTO-regeneration job checks TraigentSchema out at a ``ref`` of its
own to prove ``traigent/generated/schema_types.py`` is current. If those two
refs drift, the regeneration gate validates the DTOs against a Schema revision
the SDK does not install -- it fails on every honest pin move and would pass a
stale one. Measured 2026-09-09: the pin moved to ``8f8bb5882`` while the
workflow still checked out ``3f0529c1``, and ``--check`` failed in CI for the
wrong reason. This test makes the two refs one fact.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIN_FILE = _REPO_ROOT / "scripts" / "ci" / "schema-pin.txt"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "pr-gate.yml"
_SCHEMA_REPO = "Traigent/TraigentSchema"
_PIN_RE = re.compile(
    r"^traigent-schema\s*@\s*git\+https://github\.com/Traigent/TraigentSchema\.git@([0-9a-f]{40})\s*$",
    re.MULTILINE,
)


def _pinned_schema_sha() -> str:
    matches = _PIN_RE.findall(_PIN_FILE.read_text(encoding="utf-8"))
    assert len(matches) == 1, (
        f"expected exactly one uncommented 40-hex pin line, found {matches}"
    )
    return matches[0]


def _schema_checkout_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    steps: list[dict[str, Any]] = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            with_block = step.get("with") or {}
            if with_block.get("repository") == _SCHEMA_REPO:
                steps.append(step)
    return steps


def test_pin_file_names_one_full_sha() -> None:
    sha = _pinned_schema_sha()
    assert len(sha) == 40


def test_workflow_checks_out_traigent_schema_at_the_pinned_sha() -> None:
    steps = _schema_checkout_steps()
    assert steps, f"no actions/checkout step for {_SCHEMA_REPO} in {_WORKFLOW.name}"
    pinned = _pinned_schema_sha()
    refs = {str(step["with"].get("ref")) for step in steps}
    assert refs == {pinned}, (
        f"pr-gate.yml checks out {_SCHEMA_REPO} at {sorted(refs)} but "
        f"scripts/ci/schema-pin.txt pins {pinned}; bump both in the same PR"
    )
