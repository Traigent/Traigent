"""Tests for release-review verdict hard peer-review completeness gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _infer_agent_type(model: str) -> str:
    lowered = model.lower()
    if "claude" in lowered or "opus" in lowered:
        return "claude_cli"
    if "copilot" in lowered or "gemini" in lowered:
        return "copilot_cli"
    return "codex_cli"


def _load_verdict_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[3]
    module_path = (
        repo_root / ".release_review" / "automation" / "build_release_verdict.py"
    )
    spec = importlib.util.spec_from_file_location("build_release_verdict", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_check_results(run_dir: Path) -> Path:
    checks_file = run_dir / "gate_results" / "check_results.json"
    checks_file.parent.mkdir(parents=True, exist_ok=True)
    checks_payload = {
        "checks": [
            {"key": "lint-type", "status": "pass", "required": True},
            {"key": "tests-unit", "status": "pass", "required": True},
            {"key": "tests-integration", "status": "pass", "required": True},
            {"key": "security", "status": "pass", "required": True},
            {"key": "dependency-review", "status": "pass", "required": True},
            {"key": "codeql", "status": "pass", "required": True},
            {"key": "release-review-consistency", "status": "pass", "required": True},
        ]
    }
    checks_file.write_text(json.dumps(checks_payload, indent=2) + "\n")
    return checks_file


def _write_scope_files(run_dir: Path, files: list[str]) -> None:
    inventory_dir = run_dir / "inventories"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(files) + ("\n" if files else "")
    (inventory_dir / "review_scope_files.txt").write_text(payload)


def _write_run_manifest(
    run_dir: Path,
    *,
    release_id: str | None = None,
    review_mode: str = "strict",
    baseline_sha: str = "abc1234",
) -> None:
    payload = {
        "release_id": release_id or run_dir.name,
        "baseline_sha": baseline_sha,
        "base_branch": "main",
        "review_mode": review_mode,
        "protocol_version": 2,
        "generated_at_utc": "2026-03-07T00:00:00Z",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_evidence(
    run_dir: Path,
    *,
    component: str,
    review_type: str,
    reviewer_model: str,
    decision: str,
    commit_sha: str,
    stamp: str,
    filename: str,
    files_reviewed: list[str] | None = None,
    agent_type: str | None = None,
) -> None:
    path = run_dir / "components" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "component": component,
        "review_type": review_type,
        "agent_type": agent_type or _infer_agent_type(reviewer_model),
        "reviewer_model": reviewer_model,
        "commit_sha": commit_sha,
        "files_reviewed": files_reviewed or ["docs/placeholder.md"],
        "findings": [],
        "strengths": [
            {
                "id": f"{review_type}-strength-1",
                "severity": "S1",
                "file": (files_reviewed or ["docs/placeholder.md"])[0],
                "line": 1,
                "title": "Validated guardrail",
                "description": "Verified this file preserves expected release behavior.",
            }
        ],
        "checks_performed": [
            {
                "check_id": "determinism-check",
                "category": "correctness",
                "result": "pass",
                "evidence": "Reviewed control flow and error handling for deterministic behavior.",
            }
        ],
        "tests": [{"command": "pytest -q", "exit_code": 0, "summary": "ok"}],
        "review_summary": (
            "Reviewed assigned files end-to-end, validated correctness, and recorded both "
            "defects and strengths to support meta-analysis."
        ),
        "decision": decision,
        "timestamp_utc": stamp,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_file_review_artifact(
    run_dir: Path,
    *,
    component: str,
    review_type: str,
    agent_type: str,
    reviewer_model: str,
    decision: str,
    commit_sha: str,
    stamp: str,
    file_path: str,
    filename: str,
    angles_reviewed: list[str] | None = None,
) -> None:
    path = run_dir / "file_reviews" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "component": component,
        "review_type": review_type,
        "agent_type": agent_type,
        "reviewer_model": reviewer_model,
        "file": file_path,
        "angles_reviewed": angles_reviewed
        or [
            "security_authz",
            "correctness_regression",
            "async_concurrency_performance",
            "dto_api_contract",
        ],
        "commit_sha": commit_sha,
        "decision": decision,
        "notes": "Reviewed the file in full, traced error paths, and validated expected behavior.",
        "findings": [],
        "strengths": [
            {
                "id": f"{review_type}-strength-file-1",
                "severity": "S1",
                "line": 1,
                "title": "Contract consistency",
                "description": "Confirmed this file maintains expected release contract behavior.",
            }
        ],
        "checks_performed": [
            {
                "check_id": "file-contract-check",
                "category": "correctness",
                "result": "pass",
                "evidence": "Inspected control flow, edge cases, and test expectations for this file.",
            }
        ],
        "timestamp_utc": stamp,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_verdict_blocks_ready_without_peer_review_evidence(tmp_path: Path) -> None:
    module = _load_verdict_module()

    run_dir = tmp_path / "run-missing-peer"
    (run_dir / "components").mkdir(parents=True)
    (run_dir / "waivers").mkdir(parents=True)
    checks_file = _write_check_results(run_dir)
    _write_scope_files(run_dir, ["traigent/api/decorators.py"])

    exit_code = module.main(
        [
            "--release-id",
            "run-missing-peer",
            "--run-dir",
            str(run_dir),
            "--checks-file",
            str(checks_file),
            "--baseline-sha",
            "abc1234",
        ]
    )

    verdict = json.loads((run_dir / "gate_results" / "verdict.json").read_text())
    assert exit_code == 1
    assert verdict["status"] == "NOT_READY"
    assert verdict["failed_required_reviews"]
    assert any(
        item["reason"] == "missing_component_evidence"
        for item in verdict["failed_required_reviews"]
    )


def test_verdict_ready_when_all_required_peer_reviews_are_complete(
    tmp_path: Path,
) -> None:
    module = _load_verdict_module()

    run_dir = tmp_path / "run-complete-peer"
    (run_dir / "components").mkdir(parents=True)
    (run_dir / "waivers").mkdir(parents=True)
    checks_file = _write_check_results(run_dir)

    baseline_sha = "abc1234"
    component_scope_files = {
        "Public API + Safety": "traigent/api/decorators.py",
        "Core Orchestration + Config": "traigent/core/optimized_function.py",
        "Integrations + Invokers": "traigent/integrations/plugin_registry.py",
        "Optimizers + Evaluators": "traigent/optimizers/random.py",
        "Packaging + CI": ".github/workflows/release-review.yml",
        "Docs + Release Ops": ".release_review/CAPTAIN_PROTOCOL.md",
    }
    _write_scope_files(run_dir, list(component_scope_files.values()))

    for idx, component in enumerate(module.REQUIRED_COMPONENTS):
        component_name = component.name
        prefix = f"c{idx:02d}"
        reviewed_file = component_scope_files[component_name]
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="primary",
            reviewer_model="codex-cli-5.3-high",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:00:{idx:02d}Z",
            filename=f"{prefix}_primary.json",
            files_reviewed=[reviewed_file],
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="secondary",
            reviewer_model="claude-opus-4.6-extended",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:01:{idx:02d}Z",
            filename=f"{prefix}_secondary.json",
            files_reviewed=[reviewed_file],
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="tertiary",
            reviewer_model="codex-cli-5.3-medium",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:01:{(idx + 30):02d}Z",
            filename=f"{prefix}_tertiary.json",
            files_reviewed=[reviewed_file],
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="reconciliation",
            reviewer_model="codex-cli-5.3-xhigh",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:02:{idx:02d}Z",
            filename=f"{prefix}_reconciliation.json",
            files_reviewed=[reviewed_file],
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="primary",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-high",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:03:{idx:02d}Z",
            file_path=reviewed_file,
            filename=f"{prefix}/primary/codex_cli/{idx:03d}.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="secondary",
            agent_type="claude_cli",
            reviewer_model="claude-opus-4.6-extended",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:04:{idx:02d}Z",
            file_path=reviewed_file,
            filename=f"{prefix}/secondary/claude_cli/{idx:03d}.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="tertiary",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-medium",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:05:{idx:02d}Z",
            file_path=reviewed_file,
            filename=f"{prefix}/tertiary/codex_cli/{idx:03d}.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="reconciliation",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-xhigh",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:06:{idx:02d}Z",
            file_path=reviewed_file,
            filename=f"{prefix}/reconciliation/codex_cli/{idx:03d}.json",
        )

    exit_code = module.main(
        [
            "--release-id",
            "run-complete-peer",
            "--run-dir",
            str(run_dir),
            "--checks-file",
            str(checks_file),
            "--baseline-sha",
            baseline_sha,
        ]
    )

    verdict = json.loads((run_dir / "gate_results" / "verdict.json").read_text())
    assert exit_code == 0
    assert verdict["status"] == "READY"
    assert verdict["failed_required_reviews"] == []


def test_verdict_blocks_p0_p1_when_primary_and_secondary_are_same_family(
    tmp_path: Path,
) -> None:
    module = _load_verdict_module()

    run_dir = tmp_path / "run-same-family"
    (run_dir / "components").mkdir(parents=True)
    (run_dir / "waivers").mkdir(parents=True)
    checks_file = _write_check_results(run_dir)
    _write_scope_files(run_dir, ["traigent/api/decorators.py"])

    baseline_sha = "abc1234"
    component_name = module.REQUIRED_COMPONENTS[0].name  # P0

    _write_evidence(
        run_dir,
        component=component_name,
        review_type="primary",
        reviewer_model="codex-cli-5.3-high",
        decision="approved",
        commit_sha=baseline_sha,
        stamp="2026-03-06T00:00:00Z",
        filename="p0_primary.json",
        files_reviewed=["traigent/api/decorators.py"],
    )
    _write_evidence(
        run_dir,
        component=component_name,
        review_type="secondary",
        reviewer_model="codex-cli-5.3-xhigh",
        decision="approved",
        commit_sha=baseline_sha,
        stamp="2026-03-06T00:01:00Z",
        filename="p0_secondary.json",
        files_reviewed=["traigent/api/decorators.py"],
    )
    _write_evidence(
        run_dir,
        component=component_name,
        review_type="reconciliation",
        reviewer_model="claude-opus-4.6-extended",
        decision="approved",
        commit_sha=baseline_sha,
        stamp="2026-03-06T00:02:00Z",
        filename="p0_reconciliation.json",
        files_reviewed=["traigent/api/decorators.py"],
    )
    _write_evidence(
        run_dir,
        component=component_name,
        review_type="tertiary",
        reviewer_model="codex-cli-5.3-medium",
        decision="approved",
        commit_sha=baseline_sha,
        stamp="2026-03-06T00:03:00Z",
        filename="p0_tertiary.json",
        files_reviewed=["traigent/api/decorators.py"],
    )

    exit_code = module.main(
        [
            "--release-id",
            "run-same-family",
            "--run-dir",
            str(run_dir),
            "--checks-file",
            str(checks_file),
            "--baseline-sha",
            baseline_sha,
        ]
    )

    verdict = json.loads((run_dir / "gate_results" / "verdict.json").read_text())
    assert exit_code == 1
    assert verdict["status"] == "NOT_READY"
    assert any(
        item.get("reason") == "primary_secondary_same_family"
        for item in verdict["failed_required_reviews"]
    )


def test_verdict_blocks_when_component_file_is_missing_from_secondary_coverage(
    tmp_path: Path,
) -> None:
    module = _load_verdict_module()

    run_dir = tmp_path / "run-missing-file-coverage"
    (run_dir / "components").mkdir(parents=True)
    (run_dir / "waivers").mkdir(parents=True)
    checks_file = _write_check_results(run_dir)

    baseline_sha = "abc1234"
    _write_scope_files(
        run_dir,
        [
            "traigent/api/decorators.py",
            "traigent/core/optimized_function.py",
            "traigent/integrations/plugin_registry.py",
            "traigent/optimizers/random.py",
            ".github/workflows/release-review.yml",
            ".release_review/CAPTAIN_PROTOCOL.md",
        ],
    )

    component_files = {
        "Public API + Safety": "traigent/api/decorators.py",
        "Core Orchestration + Config": "traigent/core/optimized_function.py",
        "Integrations + Invokers": "traigent/integrations/plugin_registry.py",
        "Optimizers + Evaluators": "traigent/optimizers/random.py",
        "Packaging + CI": ".github/workflows/release-review.yml",
        "Docs + Release Ops": ".release_review/CAPTAIN_PROTOCOL.md",
    }

    for idx, component in enumerate(module.REQUIRED_COMPONENTS):
        component_name = component.name
        file_path = component_files[component_name]
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="primary",
            reviewer_model="codex-cli-5.3-high",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:10:{idx:02d}Z",
            filename=f"p_{idx:02d}.json",
            files_reviewed=[file_path],
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="secondary",
            reviewer_model="claude-opus-4.6-extended",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:11:{idx:02d}Z",
            filename=f"s_{idx:02d}.json",
            files_reviewed=[file_path],
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="tertiary",
            reviewer_model="codex-cli-5.3-medium",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:11:{(idx + 30):02d}Z",
            filename=f"t_{idx:02d}.json",
            files_reviewed=[file_path],
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="reconciliation",
            reviewer_model="codex-cli-5.3-xhigh",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:12:{idx:02d}Z",
            filename=f"r_{idx:02d}.json",
            files_reviewed=[file_path],
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="primary",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-high",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:13:{idx:02d}Z",
            file_path=file_path,
            filename=f"c_{idx:02d}/primary/codex_cli/review.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="secondary",
            agent_type="claude_cli",
            reviewer_model="claude-opus-4.6-extended",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:14:{idx:02d}Z",
            file_path=file_path,
            filename=f"c_{idx:02d}/secondary/claude_cli/review.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="tertiary",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-medium",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:15:{idx:02d}Z",
            file_path=file_path,
            filename=f"c_{idx:02d}/tertiary/codex_cli/review.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="reconciliation",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-xhigh",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:16:{idx:02d}Z",
            file_path=file_path,
            filename=f"c_{idx:02d}/reconciliation/codex_cli/review.json",
        )

    # Remove one required file from secondary coverage for first component.
    secondary_file = run_dir / "components" / "s_00.json"
    payload = json.loads(secondary_file.read_text())
    payload["files_reviewed"] = []
    secondary_file.write_text(json.dumps(payload, indent=2) + "\n")

    exit_code = module.main(
        [
            "--release-id",
            "run-missing-file-coverage",
            "--run-dir",
            str(run_dir),
            "--checks-file",
            str(checks_file),
            "--baseline-sha",
            baseline_sha,
        ]
    )

    verdict = json.loads((run_dir / "gate_results" / "verdict.json").read_text())
    assert exit_code == 1
    assert verdict["status"] == "NOT_READY"
    assert any(
        item.get("reason") == "missing_file_peer_review_coverage"
        for item in verdict["failed_required_reviews"]
    )


def test_verdict_blocks_when_file_review_artifact_lane_is_missing(tmp_path: Path) -> None:
    module = _load_verdict_module()

    run_dir = tmp_path / "run-missing-file-artifact-lane"
    (run_dir / "components").mkdir(parents=True)
    (run_dir / "waivers").mkdir(parents=True)
    checks_file = _write_check_results(run_dir)

    baseline_sha = "abc1234"
    component_scope_files = {
        "Public API + Safety": "traigent/api/decorators.py",
        "Core Orchestration + Config": "traigent/core/optimized_function.py",
        "Integrations + Invokers": "traigent/integrations/plugin_registry.py",
        "Optimizers + Evaluators": "traigent/optimizers/random.py",
        "Packaging + CI": ".github/workflows/release-review.yml",
        "Docs + Release Ops": ".release_review/CAPTAIN_PROTOCOL.md",
    }
    _write_scope_files(run_dir, list(component_scope_files.values()))

    for idx, component in enumerate(module.REQUIRED_COMPONENTS):
        component_name = component.name
        file_path = component_scope_files[component_name]
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="primary",
            reviewer_model="codex-cli-5.3-high",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:20:{idx:02d}Z",
            filename=f"pa_{idx:02d}.json",
            files_reviewed=[file_path],
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="secondary",
            reviewer_model="claude-opus-4.6-extended",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:21:{idx:02d}Z",
            filename=f"sa_{idx:02d}.json",
            files_reviewed=[file_path],
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="tertiary",
            reviewer_model="codex-cli-5.3-medium",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:22:{idx:02d}Z",
            filename=f"ta_{idx:02d}.json",
            files_reviewed=[file_path],
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="reconciliation",
            reviewer_model="codex-cli-5.3-xhigh",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:23:{idx:02d}Z",
            filename=f"ra_{idx:02d}.json",
            files_reviewed=[file_path],
        )

        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="primary",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-high",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:24:{idx:02d}Z",
            file_path=file_path,
            filename=f"a_{idx:02d}/primary/codex_cli/review.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="tertiary",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-medium",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:25:{idx:02d}Z",
            file_path=file_path,
            filename=f"a_{idx:02d}/tertiary/codex_cli/review.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="reconciliation",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-xhigh",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:26:{idx:02d}Z",
            file_path=file_path,
            filename=f"a_{idx:02d}/reconciliation/codex_cli/review.json",
        )
        # Intentionally skip the required secondary/claude lane.

    exit_code = module.main(
        [
            "--release-id",
            "run-missing-file-artifact-lane",
            "--run-dir",
            str(run_dir),
            "--checks-file",
            str(checks_file),
            "--baseline-sha",
            baseline_sha,
        ]
    )

    verdict = json.loads((run_dir / "gate_results" / "verdict.json").read_text())
    assert exit_code == 1
    assert verdict["status"] == "NOT_READY"
    assert any(
        item.get("reason") == "missing_file_review_artifact"
        and item.get("role") == "secondary"
        for item in verdict["failed_required_reviews"]
    )


def test_verdict_blocks_when_positive_findings_are_missing(tmp_path: Path) -> None:
    module = _load_verdict_module()

    run_dir = tmp_path / "run-missing-strengths"
    (run_dir / "components").mkdir(parents=True)
    (run_dir / "waivers").mkdir(parents=True)
    checks_file = _write_check_results(run_dir)

    baseline_sha = "abc1234"
    component_scope_files = {
        "Public API + Safety": "traigent/api/decorators.py",
        "Core Orchestration + Config": "traigent/core/optimized_function.py",
        "Integrations + Invokers": "traigent/integrations/plugin_registry.py",
        "Optimizers + Evaluators": "traigent/optimizers/random.py",
        "Packaging + CI": ".github/workflows/release-review.yml",
        "Docs + Release Ops": ".release_review/CAPTAIN_PROTOCOL.md",
    }
    _write_scope_files(run_dir, list(component_scope_files.values()))

    for idx, component in enumerate(module.REQUIRED_COMPONENTS):
        component_name = component.name
        file_path = component_scope_files[component_name]
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="primary",
            reviewer_model="codex-cli-5.3-high",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:30:{idx:02d}Z",
            filename=f"pp_{idx:02d}.json",
            files_reviewed=[file_path],
            agent_type="codex_cli",
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="secondary",
            reviewer_model="claude-opus-4.6-extended",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:31:{idx:02d}Z",
            filename=f"ss_{idx:02d}.json",
            files_reviewed=[file_path],
            agent_type="claude_cli",
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="tertiary",
            reviewer_model="codex-cli-5.3-medium",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:32:{idx:02d}Z",
            filename=f"tt_{idx:02d}.json",
            files_reviewed=[file_path],
            agent_type="codex_cli",
        )
        _write_evidence(
            run_dir,
            component=component_name,
            review_type="reconciliation",
            reviewer_model="codex-cli-5.3-xhigh",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:33:{idx:02d}Z",
            filename=f"rr_{idx:02d}.json",
            files_reviewed=[file_path],
            agent_type="codex_cli",
        )

        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="primary",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-high",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:34:{idx:02d}Z",
            file_path=file_path,
            filename=f"x_{idx:02d}/primary/codex_cli/review.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="secondary",
            agent_type="claude_cli",
            reviewer_model="claude-opus-4.6-extended",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:35:{idx:02d}Z",
            file_path=file_path,
            filename=f"x_{idx:02d}/secondary/claude_cli/review.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="tertiary",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-medium",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:36:{idx:02d}Z",
            file_path=file_path,
            filename=f"x_{idx:02d}/tertiary/codex_cli/review.json",
        )
        _write_file_review_artifact(
            run_dir,
            component=component_name,
            review_type="reconciliation",
            agent_type="codex_cli",
            reviewer_model="codex-cli-5.3-xhigh",
            decision="approved",
            commit_sha=baseline_sha,
            stamp=f"2026-03-06T00:37:{idx:02d}Z",
            file_path=file_path,
            filename=f"x_{idx:02d}/reconciliation/codex_cli/review.json",
        )

    evidence_file = run_dir / "components" / "pp_00.json"
    evidence_payload = json.loads(evidence_file.read_text())
    evidence_payload["strengths"] = []
    evidence_file.write_text(json.dumps(evidence_payload, indent=2) + "\n")

    artifact_file = (
        run_dir / "file_reviews" / "x_00" / "primary" / "codex_cli" / "review.json"
    )
    artifact_payload = json.loads(artifact_file.read_text())
    artifact_payload["strengths"] = []
    artifact_file.write_text(json.dumps(artifact_payload, indent=2) + "\n")

    exit_code = module.main(
        [
            "--release-id",
            "run-missing-strengths",
            "--run-dir",
            str(run_dir),
            "--checks-file",
            str(checks_file),
            "--baseline-sha",
            baseline_sha,
        ]
    )

    verdict = json.loads((run_dir / "gate_results" / "verdict.json").read_text())
    assert exit_code == 1
    assert verdict["status"] == "NOT_READY"
    assert any(
        item.get("reason") == "component_evidence_missing_strengths"
        for item in verdict["failed_required_reviews"]
    )
    assert any(
        item.get("reason") == "file_review_artifact_missing_strengths"
        for item in verdict["failed_required_reviews"]
    )


def test_quick_mode_is_ready_with_single_primary_lane_and_reduced_angles(
    tmp_path: Path,
) -> None:
    module = _load_verdict_module()

    run_dir = tmp_path / "run-quick-mode"
    (run_dir / "components").mkdir(parents=True)
    (run_dir / "waivers").mkdir(parents=True)
    checks_file = _write_check_results(run_dir)
    baseline_sha = "abc1234"

    _write_run_manifest(
        run_dir,
        release_id="run-quick-mode",
        review_mode="quick",
        baseline_sha=baseline_sha,
    )
    _write_scope_files(run_dir, ["traigent/api/decorators.py"])
    _write_file_review_artifact(
        run_dir,
        component="Public API + Safety",
        review_type="primary",
        agent_type="codex_cli",
        reviewer_model="codex-cli-5.3-high",
        decision="approved",
        commit_sha=baseline_sha,
        stamp="2026-03-07T00:10:00Z",
        file_path="traigent/api/decorators.py",
        filename="quick/primary/codex_cli/review.json",
        angles_reviewed=[
            "security_authz",
            "correctness_regression",
            "async_concurrency_performance",
        ],
    )

    exit_code = module.main(
        [
            "--release-id",
            "run-quick-mode",
            "--run-dir",
            str(run_dir),
            "--checks-file",
            str(checks_file),
            "--baseline-sha",
            baseline_sha,
        ]
    )

    verdict = json.loads((run_dir / "gate_results" / "verdict.json").read_text())
    assert exit_code == 0
    assert verdict["status"] == "QUICK_READY"
    assert verdict["review_mode"] == "quick"
    assert verdict["failed_required_reviews"] == []


def test_quick_mode_reuses_previous_run_file_artifact_when_file_is_unchanged(
    tmp_path: Path,
) -> None:
    module = _load_verdict_module()

    previous_run = tmp_path / "previous-run"
    current_run = tmp_path / "current-run"
    for run_dir in (previous_run, current_run):
        (run_dir / "components").mkdir(parents=True)
        (run_dir / "waivers").mkdir(parents=True)

    checks_file = _write_check_results(current_run)
    _write_run_manifest(
        previous_run,
        release_id="previous-run",
        review_mode="quick",
        baseline_sha="abc1234",
    )
    _write_run_manifest(
        current_run,
        release_id="current-run",
        review_mode="quick",
        baseline_sha="def5678",
    )
    _write_scope_files(current_run, ["traigent/api/decorators.py"])
    _write_file_review_artifact(
        previous_run,
        component="Public API + Safety",
        review_type="primary",
        agent_type="codex_cli",
        reviewer_model="codex-cli-5.3-high",
        decision="approved",
        commit_sha="abc1234",
        stamp="2026-03-06T23:59:00Z",
        file_path="traigent/api/decorators.py",
        filename="previous/primary/codex_cli/review.json",
        angles_reviewed=[
            "security_authz",
            "correctness_regression",
            "async_concurrency_performance",
        ],
    )

    module.git_file_unchanged_since = lambda commit_sha, baseline_sha, file_path: True

    exit_code = module.main(
        [
            "--release-id",
            "current-run",
            "--run-dir",
            str(current_run),
            "--checks-file",
            str(checks_file),
            "--baseline-sha",
            "def5678",
        ]
    )

    verdict = json.loads((current_run / "gate_results" / "verdict.json").read_text())
    assert exit_code == 0
    assert verdict["status"] == "QUICK_READY"
    assert verdict["failed_required_reviews"] == []
