"""Contract checks for the publish workflow's GitHub Release step.

GitHub's "Latest" release drifted from PyPI (stuck at v0.12.0 while PyPI
published up to v0.27.0, issue #2267) because `.github/workflows/publish.yml`
verified PyPI installs but never created a GitHub Release. This test pins the
structure of the fix so the drift cannot recur silently: a dedicated job that
runs only after a real, verified tag-triggered publish, creates or (on a
rerun) idempotently updates the Release for the exact published tag, requests
no more than the `contents: write` permission it needs, and asserts the tag,
PyPI version, and GitHub Latest all agree before the run is considered green.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "publish.yml"


def _workflow() -> dict[str, Any]:
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(workflow, dict)
    return workflow


def _release_job() -> dict[str, Any]:
    return _workflow()["jobs"]["github-release"]


def test_workflow_top_level_permission_stays_read_only() -> None:
    workflow = _workflow()
    assert workflow["permissions"] == {"contents": "read"}


def test_release_job_only_runs_after_a_verified_tag_publish() -> None:
    job = _release_job()
    assert set(job["needs"]) == {"publish", "verify-publication"}
    condition = job["if"]
    assert "needs.publish.result == 'success'" in condition
    assert "needs.verify-publication.result == 'success'" in condition
    assert "startsWith(github.ref, 'refs/tags/v')" in condition


def test_release_job_requests_only_contents_write() -> None:
    job = _release_job()
    # Minimum additional permission beyond the workflow-level `contents: read`
    # (see test above): this job is the only writer, and it only writes.
    assert job["permissions"] == {"contents": "write"}


def test_release_job_is_idempotent_create_or_update() -> None:
    job = _release_job()
    steps_text = "\n".join(step.get("run", "") for step in job["steps"])
    assert "gh release view" in steps_text
    assert "gh release edit" in steps_text
    assert "gh release create" in steps_text
    assert "--verify-tag" in steps_text


def test_release_job_asserts_tag_pypi_and_latest_agree() -> None:
    job = _release_job()
    steps_text = "\n".join(step.get("run", "") for step in job["steps"])
    assert "EXPECTED_VERSION" in steps_text
    assert "gh release view --repo" in steps_text
    assert "json tagName" in steps_text
    # The published package version flows from the `publish` job's output,
    # which is where PyPI's actual version is read from pyproject.toml.
    assert (
        job["env"]["EXPECTED_VERSION"] == "${{ needs.publish.outputs.package_version }}"
    )
