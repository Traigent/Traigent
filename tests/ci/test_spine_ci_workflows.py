from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SPINE_SHA = "e14f105c876a30b270d4939088d8361b720af555"


def _load(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def test_spine_core_is_immutable_explicit_and_merge_group_safe() -> None:
    workflow = _load("spine-core.yml")
    trigger = workflow["on"]
    job = workflow["jobs"]["spine-core"]

    assert trigger["pull_request"]["branches"] == ["develop", "main"]
    assert "edited" in trigger["pull_request"]["types"]
    assert trigger["merge_group"]["branches"] == ["develop", "main"]
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert job["uses"].endswith(f"spine-core-reusable.yml@{SPINE_SHA}")
    assert job["with"] == {
        "repo_id": "Traigent",
        "base_ref": "${{ github.event_name == 'merge_group' && github.event.merge_group.base_ref || github.event.pull_request.base.ref }}",
        "head_ref": "${{ github.event_name == 'merge_group' && github.event.merge_group.head_ref || github.event.pull_request.head.ref }}",
        "head_sha": "${{ github.event_name == 'merge_group' && github.event.merge_group.head_sha || github.event.pull_request.head.sha }}",
        "mode": "advisory",
    }
    assert job["secrets"] == {
        "VALIDATION_CI_PAT": "${{ secrets.VALIDATION_CI_PAT }}"
    }


def test_g8_dispatch_uses_real_aggregate_and_exact_provenance() -> None:
    path = WORKFLOWS / "validation-g8-dispatch.yml"
    workflow = _load(path.name)
    trigger = workflow["on"]["workflow_run"]
    job = workflow["jobs"]["dispatch"]
    step = job["steps"][0]

    assert trigger["workflows"] == ["SDK Required PR Gate"]
    assert "workflow_dispatch" not in workflow["on"]
    assert "pull_request_target" not in path.read_text()
    assert "conclusion == 'success'" in job["if"]
    assert step["env"]["SOURCE_HEAD_SHA"] == "${{ github.event.workflow_run.head_sha }}"
    assert step["env"]["SOURCE_BASE_BRANCH"] == "${{ github.event.workflow_run.pull_requests[0].base.ref }}"
    assert step["env"]["SOURCE_HEAD_BRANCH"] == "${{ github.event.workflow_run.head_branch }}"
    assert step["env"]["SOURCE_WORKFLOW"] == "${{ github.event.workflow_run.name }}"
    assert step["env"]["SOURCE_RUN_ID"] == "${{ github.event.workflow_run.id }}"
    assert '::error::validation-g8: $1' in step["run"]
    assert "VALIDATION_CI_PAT is not configured" in step["run"]
    assert "skipping validation-spine dispatch" not in step["run"]


def test_policy_surface_link_preserves_marker_and_queue_contract() -> None:
    path = WORKFLOWS / "spine-session-link.yml"
    workflow = _load(path.name)
    trigger = workflow["on"]
    job = workflow["jobs"]["policy-surface-link"]
    source = path.read_text()

    assert "edited" in trigger["pull_request"]["types"]
    assert "labeled" in trigger["pull_request"]["types"]
    assert trigger["merge_group"]["types"] == ["checks_requested"]
    assert trigger["merge_group"]["branches"] == ["develop", "main"]
    assert workflow["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert "traigent/security/**" in job["env"]["POLICY_SURFACE_GLOBS"]
    assert "Spine-Trail:" in source
    assert "a linked/promoted ChangeSession is also required" in source
    assert "Spine(?:-Session)?" in source
    assert "Spine:\\s*none" in source
    assert "spine-exempt" in source
    assert "pull_request_target" not in source
