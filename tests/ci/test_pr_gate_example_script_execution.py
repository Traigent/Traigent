"""Exercise the real changed-path classifier and required unit-job gates."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "pr-gate.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _classify_changed_paths(paths: list[str], tmp_path: Path) -> dict[str, str]:
    """Run the exact has_match/set_output_bool block from the workflow shell."""
    workflow = _workflow()
    changes = workflow["jobs"]["changes"]
    step = next(step for step in changes["steps"] if step.get("name") == "Classify changed files")
    command = step["run"]
    lines = command.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("has_match()"))
    end = next(
        index
        for index, line in enumerate(lines[start:], start)
        if line.startswith("set_output_bool runtime_tvl_changed")
    )
    output = tmp_path / "github-output"
    script = "\n".join(
        [
            "set -euo pipefail",
            'changed_files="${CHANGED_FILES:-}"',
            'GITHUB_OUTPUT="$OUTPUT_FILE"',
            *lines[start:end],
        ]
    )
    env = {
        **os.environ,
        "CHANGED_FILES": "\n".join(paths),
        "OUTPUT_FILE": str(output),
    }
    subprocess.run(["bash", "-c", script], check=True, cwd=ROOT, env=env)
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


def _unit_job_runs(unit_job: dict, outputs: dict[str, str]) -> bool:
    """Evaluate the current required prerequisites for a same-repo PR."""
    condition = unit_job["if"]
    assert " ".join(condition.split()) == (
        "${{ !cancelled() && needs.changes.outputs.code_changed == 'true' && "
        "needs.preflight.result == 'success' && (github.event_name == 'merge_group' || "
        "github.event.pull_request.head.repo.full_name == github.repository) }}"
    )

    event_name = "pull_request"
    head_repo = repository = "Traigent/Traigent"
    preflight_result = "success"
    cancelled = False
    return (
        not cancelled
        and outputs.get("code_changed") == "true"
        and preflight_result == "success"
        and (event_name == "merge_group" or head_repo == repository)
    )


def test_relevant_changes_run_the_example_suite_through_existing_unit_job(
    tmp_path: Path,
) -> None:
    workflow = _workflow()
    unit_job = workflow["jobs"]["unit"]
    assert unit_job["name"] == "unit"
    assert "pytest tests/unit -q --tb=short" in "\n".join(
        step.get("run", "") for step in unit_job["steps"]
    )
    example_step = next(
        step
        for step in unit_job["steps"]
        if step.get("name") == "Run shipped example script regressions"
    )
    assert example_step["if"] == "${{ needs.changes.outputs.example_scripts_changed == 'true' }}"
    assert example_step["env"]["TRAIGENT_MOCK_LLM"] == "true"
    assert example_step["env"]["TRAIGENT_OFFLINE_MODE"] == "true"
    assert example_step["run"] == "pytest tests/examples/test_example_scripts_run.py -q --tb=short"

    changed_path_groups = [
        ["examples/core/simple-prompt/run.py"],
        ["tests/examples/test_example_scripts_run.py"],
        ["walkthrough/mock/advanced/02_prompt_optimization.py"],
        ["walkthrough/utils/mock_answers.py"],
        ["walkthrough/datasets/simple_questions.jsonl"],
        ["traigent/core/optimizer.py"],
        ["tests/conftest.py"],
        ["tests/fixtures/rate_limit_fixtures.py"],
        ["pyproject.toml"],
        ["requirements-dev.txt"],
        ["uv.lock"],
        ["scripts/ci/install_sdk_requirements.sh"],
        ["scripts/ci/schema-pin.txt"],
        [".github/workflows/pr-gate.yml"],
    ]
    for paths in changed_path_groups:
        outputs = _classify_changed_paths(paths, tmp_path)
        assert outputs["code_changed"] == "true", paths
        assert outputs["example_scripts_changed"] == "true", paths
        assert _unit_job_runs(unit_job, outputs), paths
        assert example_step["if"] == "${{ needs.changes.outputs.example_scripts_changed == 'true' }}"


def test_unrelated_docs_do_not_select_the_example_script_step(tmp_path: Path) -> None:
    outputs = _classify_changed_paths(["docs/usage.md"], tmp_path)
    assert outputs["code_changed"] == "false"
    assert outputs["example_scripts_changed"] == "false"
