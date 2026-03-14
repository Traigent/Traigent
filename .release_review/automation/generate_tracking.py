#!/usr/bin/env python3
"""Initialize release-review v2 run workspace and slim tracking board."""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
ALL_FILE_REVIEW_ANGLES = (
    "security_authz",
    "correctness_regression",
    "async_concurrency_performance",
    "dto_api_contract",
)
DEFAULT_REVIEW_MODES: dict[str, dict[str, object]] = {
    "strict": {
        "required_review_types": ["primary", "secondary", "tertiary", "reconciliation"],
        "required_angles": list(ALL_FILE_REVIEW_ANGLES),
        "allow_previous_artifact_reuse": True,
        "skip_if_file_unchanged_and_artifact_exists": True,
        "single_model": False,
    },
    "quick": {
        "required_review_types": ["primary"],
        "required_angles": [
            "security_authz",
            "correctness_regression",
            "async_concurrency_performance",
        ],
        "allow_previous_artifact_reuse": True,
        "skip_if_file_unchanged_and_artifact_exists": True,
        "single_model": True,
    },
}


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_iso_utc(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def git_merge_base(base_branch: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "merge-base", base_branch, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def git_changed_files(base_branch: str) -> list[str]:
    merge_base = git_merge_base(base_branch)
    if not merge_base:
        return []
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", f"{merge_base}..HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _matches_any(path: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def normalize_repo_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def load_scope_config() -> dict:
    scope_file = Path(".release_review/scope.yml")
    if not scope_file.exists():
        return {}
    data = yaml.safe_load(scope_file.read_text())
    return data if isinstance(data, dict) else {}


def resolve_review_mode_config(review_mode: str) -> dict[str, object]:
    scope_config = load_scope_config()
    configured_modes = scope_config.get("review_modes")
    mode_name = review_mode or str(scope_config.get("default_review_mode") or "strict")
    if mode_name not in DEFAULT_REVIEW_MODES:
        mode_name = "strict"

    merged = dict(DEFAULT_REVIEW_MODES[mode_name])
    if isinstance(configured_modes, dict) and isinstance(configured_modes.get(mode_name), dict):
        merged.update(configured_modes[mode_name])
    merged["name"] = mode_name
    return merged


def git_file_unchanged_since(commit_sha: str, file_path: str) -> bool:
    if not commit_sha.strip():
        return False
    result = subprocess.run(
        ["git", "diff", "--quiet", commit_sha, "HEAD", "--", file_path],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def iter_previous_run_dirs(runs_root: Path, current_release_id: str) -> list[Path]:
    manifests: list[tuple[tuple[int, str], Path]] = []
    for manifest_path in runs_root.glob("*/run_manifest.json"):
        run_dir = manifest_path.parent
        if run_dir.name == current_release_id:
            continue
        data = json.loads(manifest_path.read_text())
        if not isinstance(data, dict):
            continue
        stamp = str(
            data.get("generated_at_utc") or data.get("created_at_utc") or ""
        ).strip()
        parsed = parse_iso_utc(stamp) if stamp else None
        sort_key = (1, parsed.isoformat()) if parsed else (0, stamp)
        manifests.append((sort_key, run_dir))
    manifests.sort(reverse=True)
    return [item[1] for item in manifests]


def artifact_covers_requirement(
    payload: dict[str, object],
    *,
    file_path: str,
    required_role: str,
    required_angles: set[str],
) -> set[str]:
    artifact_file = normalize_repo_path(str(payload.get("file") or payload.get("file_path") or ""))
    if artifact_file != normalize_repo_path(file_path):
        return set()
    artifact_role = str(payload.get("review_type") or "").strip().lower()
    if artifact_role != required_role:
        return set()
    if str(payload.get("decision") or "").strip().lower() != "approved":
        return set()

    raw_angles = payload.get("angles_reviewed", [])
    if not isinstance(raw_angles, list):
        return set()

    return {
        str(angle).strip()
        for angle in raw_angles
        if str(angle).strip() in required_angles
    }


def _force_rereview_requested(file_path: str, force_patterns: list[str]) -> bool:
    return bool(force_patterns) and _matches_any(file_path, force_patterns)


def _source_release_id(previous_run_dir: Path) -> str:
    manifest_path = previous_run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return str(manifest.get("release_id") or previous_run_dir.name)


def _record_reused_artifact_coverage(
    *,
    pending: list[str],
    force_patterns: list[str],
    required_roles: set[str],
    required_angles: set[str],
    coverage: dict[str, dict[str, set[str]]],
    reuse_rows: list[dict[str, object]],
    source_release_id: str,
    artifact_path: Path,
    payload: dict[str, object],
) -> None:
    commit_sha = str(payload.get("commit_sha") or "").strip()
    for file_path in pending:
        if _force_rereview_requested(file_path, force_patterns):
            continue
        if not git_file_unchanged_since(commit_sha, file_path):
            continue
        for role in required_roles:
            covered_angles = artifact_covers_requirement(
                payload,
                file_path=file_path,
                required_role=role,
                required_angles=required_angles,
            )
            new_angles = covered_angles - coverage[file_path][role]
            if not new_angles:
                continue
            coverage[file_path][role].update(new_angles)
            reuse_rows.append(
                {
                    "file": file_path,
                    "review_type": role,
                    "angles_reused": sorted(new_angles),
                    "source_run_id": source_release_id,
                    "source_artifact": str(artifact_path),
                    "commit_sha": commit_sha,
                }
            )


def _partition_pending_files(
    *,
    pending: list[str],
    force_patterns: list[str],
    required_roles: set[str],
    required_angles: set[str],
    coverage: dict[str, dict[str, set[str]]],
) -> tuple[list[str], list[str]]:
    skipped: list[str] = []
    remaining: list[str] = []
    for file_path in pending:
        if _force_rereview_requested(file_path, force_patterns):
            remaining.append(file_path)
            continue
        if all(required_angles.issubset(coverage[file_path][role]) for role in required_roles):
            skipped.append(file_path)
        else:
            remaining.append(file_path)
    return sorted(remaining), sorted(skipped)


def build_reuse_plan(
    runs_root: Path,
    current_release_id: str,
    review_scope_files: list[str],
    review_mode_config: dict[str, object],
    force_rereview: list[str],
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    pending = sorted(set(normalize_repo_path(path) for path in review_scope_files))
    if not pending:
        return [], [], []

    if not review_mode_config.get("allow_previous_artifact_reuse", True):
        return pending, [], []

    required_roles = {
        str(item).strip().lower()
        for item in review_mode_config.get("required_review_types", [])
        if str(item).strip()
    }
    required_angles = {
        str(item).strip()
        for item in review_mode_config.get("required_angles", [])
        if str(item).strip()
    }
    force_patterns = [pattern for pattern in force_rereview if pattern.strip()]
    coverage: dict[str, dict[str, set[str]]] = {
        file_path: {role: set() for role in required_roles} for file_path in pending
    }
    reuse_rows: list[dict[str, object]] = []

    for previous_run_dir in iter_previous_run_dirs(runs_root, current_release_id):
        source_release_id = _source_release_id(previous_run_dir)
        for artifact_path in sorted((previous_run_dir / "file_reviews").rglob("*.json")):
            payload = json.loads(artifact_path.read_text())
            if not isinstance(payload, dict):
                continue
            _record_reused_artifact_coverage(
                pending=pending,
                force_patterns=force_patterns,
                required_roles=required_roles,
                required_angles=required_angles,
                coverage=coverage,
                reuse_rows=reuse_rows,
                source_release_id=source_release_id,
                artifact_path=artifact_path,
                payload=payload,
            )

    remaining, skipped = _partition_pending_files(
        pending=pending,
        force_patterns=force_patterns,
        required_roles=required_roles,
        required_angles=required_angles,
        coverage=coverage,
    )
    return remaining, skipped, reuse_rows


def filter_review_scope_files(changed_files: list[str]) -> list[str]:
    scope_file = Path(".release_review/scope.yml")
    if not scope_file.exists():
        return sorted(changed_files)

    data = yaml.safe_load(scope_file.read_text())
    if not isinstance(data, dict):
        return sorted(changed_files)

    include = [str(x) for x in data.get("include", [])]
    exclude = [str(x) for x in data.get("exclude", [])]
    shared_files = [str(x) for x in data.get("shared_files", [])]

    scoped: list[str] = []
    for file_path in changed_files:
        if _matches_any(file_path, exclude):
            continue
        if file_path in shared_files:
            scoped.append(file_path)
            continue
        if include and not _matches_any(file_path, include):
            continue
        scoped.append(file_path)

    return sorted(scoped)


def archive_tracking_file(path: Path) -> str | None:
    if not path.exists():
        return None

    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    suffix = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    archived = archive_dir / f"PRE_RELEASE_REVIEW_TRACKING_{suffix}.md"
    shutil.move(str(path), str(archived))
    return str(archived)


def validate_release_id(release_id: str) -> str:
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError(
            "release_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ "
            "(alphanumeric, dot, underscore, hyphen only)"
        )
    return release_id


def resolve_run_dir(output_root: str, release_id: str) -> tuple[Path, Path]:
    root = Path(output_root).resolve()
    run_dir = (root / release_id).resolve()
    try:
        run_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("resolved run directory escapes output root") from exc
    return root, run_dir


def resolve_inventory_path(run_dir: Path, filename: str) -> Path:
    inventories_dir = (run_dir / "inventories").resolve()
    candidate = Path(filename)
    if candidate.name != filename:
        raise ValueError("inventory filename must not include directory components")
    target = (inventories_dir / candidate).resolve()
    try:
        target.relative_to(inventories_dir)
    except ValueError as exc:
        raise ValueError("resolved inventory path escapes inventories directory") from exc
    return target


def resolve_run_child_path(run_dir: Path, filename: str) -> Path:
    run_dir_resolved = run_dir.resolve()
    candidate = Path(filename)
    if candidate.name != filename:
        raise ValueError("run child filename must not include directory components")
    target = (run_dir_resolved / candidate).resolve()
    try:
        target.relative_to(run_dir_resolved)
    except ValueError as exc:
        raise ValueError("resolved run child path escapes run directory") from exc
    return target


def write_inventory_text(run_dir: Path, filename: str, content: str) -> Path:
    target = resolve_inventory_path(run_dir, filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return target


def render_tracking_board(
    release_id: str,
    version: str,
    baseline_sha: str,
    captain: str,
    owner: str,
    review_mode: str,
) -> str:
    return f"""# Pre-Release Review Tracking (Slim Board v3)

**Protocol Version**: 3
**Status Source of Truth**: this board + `.release_review/runs/{release_id}/gate_results/verdict.json`

## Active Run

- Release ID: `{release_id}`
- Version: `{version}`
- Baseline SHA: `{baseline_sha}`
- Captain: `{captain}`
- Release owner: `{owner}`
- Review mode: `{review_mode}`
- Run root: `.release_review/runs/{release_id}/`

## Gate Status

| Check | Status | Evidence |
|---|---|---|
| release-gate/lint-type | Not started | |
| release-gate/tests-unit | Not started | |
| release-gate/tests-integration | Not started | |
| release-gate/security | Not started | |
| release-gate/dependency-review | Not started | |
| release-gate/codeql | Not started | |
| release-gate/release-review-consistency | Not started | |
| release-verdict/peer-review-completeness | Not started | `.release_review/runs/{release_id}/gate_results/verdict.json` |

## Component Board

| Component | Priority | Owner | Secondary | Tertiary | Status | Gate | Primary Evidence | Secondary Evidence | Tertiary Evidence | Reconciliation Evidence | Approved At |
|---|---:|---|---|---|---|---|---|---|---|---|---|
| Public API + Safety | P0 | - | - | - | Not started | Pending | | | | | |
| Core Orchestration + Config | P0 | - | - | - | Not started | Pending | | | | | |
| Integrations + Invokers | P1 | - | - | - | Not started | Pending | | | | | |
| Optimizers + Evaluators | P1 | - | - | - | Not started | Pending | | | | | |
| Packaging + CI | P1 | - | - | - | Not started | Pending | | | | | |
| Docs + Release Ops | P2 | - | - | - | Not started | Pending | | | | | |

## Decision Summary

- Verdict: `NOT_READY`
- Unresolved P0: 0
- Unresolved P1: 0
- Failed required checks: 0
- Failed required reviews: 0
- Waivers: 0

## Review Notes Log

- `{utc_now()}`: Run initialized.
"""


def write_inventories(run_dir: Path, base_branch: str) -> list[str]:
    inventories = run_dir / "inventories"
    inventories.mkdir(parents=True, exist_ok=True)

    src_files: list[str] = []
    for root in ("traigent", "traigent_validation"):
        root_path = Path(root)
        if root_path.exists():
            src_files.extend(str(path) for path in root_path.rglob("*.py"))

    test_files = (
        sorted(str(path) for path in Path("tests").rglob("*.py"))
        if Path("tests").exists()
        else []
    )

    src_files = sorted(src_files)
    write_inventory_text(
        run_dir,
        "src_files.txt",
        "\n".join(src_files) + ("\n" if src_files else ""),
    )
    write_inventory_text(
        run_dir,
        "tests_files.txt",
        "\n".join(test_files) + ("\n" if test_files else ""),
    )

    changed_files = git_changed_files(base_branch)
    review_scope_files = filter_review_scope_files(changed_files)
    write_inventory_text(
        run_dir,
        "changed_files.txt",
        "\n".join(sorted(changed_files)) + ("\n" if changed_files else ""),
    )
    write_inventory_text(
        run_dir,
        "review_scope_files.txt",
        "\n".join(review_scope_files) + ("\n" if review_scope_files else ""),
    )
    return review_scope_files


def write_run_manifest(
    run_dir: Path,
    release_id: str,
    version: str,
    base_branch: str,
    baseline_sha: str,
    captain: str,
    owner: str,
    review_mode: str,
    force_rereview: list[str],
) -> None:
    payload = {
        "release_id": release_id,
        "version": version,
        "base_branch": base_branch,
        "baseline_sha": baseline_sha,
        "captain": captain,
        "release_owner": owner,
        "review_mode": review_mode,
        "force_rereview": force_rereview,
        "protocol_version": 3,
        "generated_at_utc": utc_now(),
    }
    resolve_run_child_path(run_dir, "run_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate release-review v3 tracking and workspace"
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--captain", default="Codex")
    parser.add_argument("--owner", default="@release-owner")
    parser.add_argument("--output-root", default=".release_review/runs")
    parser.add_argument("--review-mode", choices=["strict", "quick"], default="strict")
    parser.add_argument(
        "--force-rereview",
        action="append",
        default=[],
        help="Repeatable glob pattern; matching files are never skipped via artifact reuse.",
    )
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    release_id = validate_release_id(args.release_id)
    release_review_dir = Path(".release_review")
    tracking_file = release_review_dir / "PRE_RELEASE_REVIEW_TRACKING.md"

    if not args.no_archive:
        archived = archive_tracking_file(tracking_file)
        if archived:
            print(f"Archived previous tracking file: {archived}")

    _, run_dir = resolve_run_dir(args.output_root, release_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gate_results").mkdir(parents=True, exist_ok=True)
    (run_dir / "components").mkdir(parents=True, exist_ok=True)
    (run_dir / "file_reviews").mkdir(parents=True, exist_ok=True)
    (run_dir / "waivers").mkdir(parents=True, exist_ok=True)

    baseline_sha = git_short_sha()
    review_mode_config = resolve_review_mode_config(args.review_mode)

    review_scope_files = write_inventories(run_dir, args.base_branch)
    pending_files, skipped_files, reuse_rows = build_reuse_plan(
        Path(args.output_root).resolve(),
        release_id,
        review_scope_files,
        review_mode_config,
        args.force_rereview,
    )
    write_inventory_text(
        run_dir,
        "review_pending_files.txt",
        "\n".join(pending_files) + ("\n" if pending_files else ""),
    )
    write_inventory_text(
        run_dir,
        "review_skipped_files.txt",
        "\n".join(skipped_files) + ("\n" if skipped_files else ""),
    )
    write_inventory_text(
        run_dir,
        "reused_file_review_artifacts.json",
        json.dumps(reuse_rows, indent=2) + "\n",
    )
    write_run_manifest(
        run_dir=run_dir,
        release_id=release_id,
        version=args.version,
        base_branch=args.base_branch,
        baseline_sha=baseline_sha,
        captain=args.captain,
        owner=args.owner,
        review_mode=str(review_mode_config["name"]),
        force_rereview=[normalize_repo_path(pattern) for pattern in args.force_rereview],
    )

    review_log_template = Path(".release_review/templates/REVIEW_LOG.md")
    review_log_path = run_dir / "REVIEW_LOG.md"
    if review_log_template.exists():
        review_log_path.write_text(
            review_log_template.read_text()
            .replace("<release_id>", release_id)
            .replace("<version>", args.version)
            .replace("<sha>", baseline_sha)
            .replace("<timestamp>", utc_now())
        )

    tracking_file.write_text(
        render_tracking_board(
            release_id=release_id,
            version=args.version,
            baseline_sha=baseline_sha,
            captain=args.captain,
            owner=args.owner,
            review_mode=str(review_mode_config["name"]),
        )
    )

    (release_review_dir / "CURRENT_RUN").write_text(f"{release_id}\n")

    print(f"Tracking board created: {tracking_file}")
    print(f"Run directory created: {run_dir}")
    print(f"Review mode: {review_mode_config['name']}")
    print(f"Pending review files: {len(pending_files)}")
    print(f"Skipped via artifact reuse: {len(skipped_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
