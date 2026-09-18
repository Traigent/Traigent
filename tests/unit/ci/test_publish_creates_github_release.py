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


def test_release_job_also_runs_for_a_workflow_dispatch_pypi_publish() -> None:
    # Regression for #2267 recurring on the workflow_dispatch->PyPI path:
    # in this repo's history, 12 of 27 successful publishes were manual
    # dispatches straight to production PyPI (never a tag push), and the
    # `publish` job's own "Publish to PyPI" step already treats that path as
    # a real production release (same condition, `publish.yml` line ~199).
    # The release job must mirror it, not just the tag-push half.
    job = _release_job()
    condition = job["if"]
    assert "github.event_name == 'workflow_dispatch'" in condition
    assert "inputs.environment == 'pypi'" in condition

    publish_job = _workflow()["jobs"]["publish"]
    publish_step = next(
        step for step in publish_job["steps"] if step.get("name") == "Publish to PyPI"
    )
    # Same production-publish test the upstream step already uses, so the
    # release job can never drift narrower (or wider) than it again.
    assert publish_step["if"] == (
        "startsWith(github.ref, 'refs/tags/v') || "
        "(github.event_name == 'workflow_dispatch' && inputs.environment == 'pypi')"
    )
    assert "startsWith(github.ref, 'refs/tags/v')" in publish_step["if"]
    assert "github.event_name == 'workflow_dispatch'" in publish_step["if"]
    assert "inputs.environment == 'pypi'" in publish_step["if"]


def test_release_tag_is_derived_from_the_published_version_not_the_ref() -> None:
    # A workflow_dispatch production publish never sets github.ref to a tag
    # (github.ref_name would be a branch name), so the tag used to create
    # the Release must come from the same version source verify-publication
    # already trusts: needs.publish.outputs.package_version.
    job = _release_job()
    assert job["env"]["TAG"] == "v${{ needs.publish.outputs.package_version }}"


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
    # Not --verify-tag: on the workflow_dispatch->pypi path the derived tag
    # (see test above) has no matching git tag yet, and --verify-tag would
    # abort the release rather than mint one. --target lets `gh release
    # create` mint the tag from this run's commit; for a real tag push the
    # tag already exists there, so --target is a no-op.
    assert "--target" in steps_text
    assert "--verify-tag" not in steps_text


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


def test_release_job_authenticates_by_env_not_by_writing_a_credential_to_disk() -> None:
    """``gh`` gets its token from the job env, never from ``gh auth login``.

    ``gh auth login --with-token`` persists the token to
    ``~/.config/gh/hosts.yml`` on the runner -- a credential at rest for the
    life of the job -- whereas ``GH_TOKEN`` in the job env is read directly by
    ``gh`` and never touches the filesystem. Both are masked in logs, so the
    difference is invisible in a run and easy to reintroduce.
    """
    job = _release_job()

    assert job["env"]["GH_TOKEN"] == "${{ github.token }}"

    steps_text = "\n".join(step.get("run", "") for step in job["steps"])
    assert "gh auth login" not in steps_text, (
        "use the GH_TOKEN job env instead of gh auth login, which writes the "
        "credential to the runner's gh config"
    )
