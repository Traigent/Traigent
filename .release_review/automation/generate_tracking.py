#!/usr/bin/env python3
"""Initialize release-review v2 run workspace and slim tracking board."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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


def archive_tracking_file(path: Path) -> str | None:
    if not path.exists():
        return None

    archive_dir = path.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archived = archive_dir / f"PRE_RELEASE_REVIEW_TRACKING_{suffix}.md"
    shutil.move(str(path), str(archived))
    return str(archived)


def render_tracking_board(
    release_id: str,
    version: str,
    baseline_sha: str,
    captain: str,
    owner: str,
) -> str:
    return f"""# Pre-Release Review Tracking (Slim Board v2)

**Protocol Version**: 2
**Status Source of Truth**: this board + `.release_review/runs/{release_id}/gate_results/verdict.json`

## Active Run

- Release ID: `{release_id}`
- Version: `{version}`
- Baseline SHA: `{baseline_sha}`
- Captain: `{captain}`
- Release owner: `{owner}`
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

## Component Board

| Component | Priority | Owner | Secondary | Status | Gate | Primary Evidence | Secondary Evidence | Approved At |
|---|---:|---|---|---|---|---|---|---|
| Public API + Safety | P0 | - | - | Not started | Pending | | | |
| Core Orchestration + Config | P0 | - | - | Not started | Pending | | | |
| Integrations + Invokers | P1 | - | - | Not started | Pending | | | |
| Optimizers + Evaluators | P1 | - | - | Not started | Pending | | | |
| Packaging + CI | P1 | - | - | Not started | Pending | | | |
| Docs + Release Ops | P2 | - | - | Not started | Pending | | | |

## Decision Summary

- Verdict: `NOT_READY`
- Unresolved P0: 0
- Unresolved P1: 0
- Failed required checks: 0
- Waivers: 0

## Review Notes Log

- `{utc_now()}`: Run initialized.
"""


def write_inventories(run_dir: Path) -> None:
    inventories = run_dir / "inventories"
    inventories.mkdir(parents=True, exist_ok=True)

    src_files: list[str] = []
    for root in ("traigent", "traigent_validation"):
        root_path = Path(root)
        if root_path.exists():
            src_files.extend(str(path) for path in root_path.rglob("*.py"))

    test_files = sorted(str(path) for path in Path("tests").rglob("*.py")) if Path("tests").exists() else []

    src_files = sorted(src_files)
    (inventories / "src_files.txt").write_text("\n".join(src_files) + ("\n" if src_files else ""))
    (inventories / "tests_files.txt").write_text("\n".join(test_files) + ("\n" if test_files else ""))


def write_run_manifest(
    run_dir: Path,
    release_id: str,
    version: str,
    base_branch: str,
    baseline_sha: str,
    captain: str,
    owner: str,
) -> None:
    payload = {
        "release_id": release_id,
        "version": version,
        "base_branch": base_branch,
        "baseline_sha": baseline_sha,
        "captain": captain,
        "release_owner": owner,
        "protocol_version": 2,
        "generated_at_utc": utc_now(),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate release-review v2 tracking and workspace")
    parser.add_argument("--version", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--captain", default="Codex")
    parser.add_argument("--owner", default="@release-owner")
    parser.add_argument("--output-root", default=".release_review/runs")
    parser.add_argument("--no-archive", action="store_true")
    args = parser.parse_args()

    release_review_dir = Path(".release_review")
    tracking_file = release_review_dir / "PRE_RELEASE_REVIEW_TRACKING.md"

    if not args.no_archive:
        archived = archive_tracking_file(tracking_file)
        if archived:
            print(f"Archived previous tracking file: {archived}")

    run_root = Path(args.output_root)
    run_dir = run_root / args.release_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gate_results").mkdir(parents=True, exist_ok=True)
    (run_dir / "components").mkdir(parents=True, exist_ok=True)
    (run_dir / "waivers").mkdir(parents=True, exist_ok=True)

    baseline_sha = git_short_sha()

    write_inventories(run_dir)
    write_run_manifest(
        run_dir=run_dir,
        release_id=args.release_id,
        version=args.version,
        base_branch=args.base_branch,
        baseline_sha=baseline_sha,
        captain=args.captain,
        owner=args.owner,
    )

    review_log_template = Path(".release_review/templates/REVIEW_LOG.md")
    review_log_path = run_dir / "REVIEW_LOG.md"
    if review_log_template.exists():
        review_log_path.write_text(
            review_log_template.read_text()
            .replace("<release_id>", args.release_id)
            .replace("<version>", args.version)
            .replace("<sha>", baseline_sha)
            .replace("<timestamp>", utc_now())
        )

    tracking_file.write_text(
        render_tracking_board(
            release_id=args.release_id,
            version=args.version,
            baseline_sha=baseline_sha,
            captain=args.captain,
            owner=args.owner,
        )
    )

    (release_review_dir / "CURRENT_RUN").write_text(f"{args.release_id}\n")

    print(f"Tracking board created: {tracking_file}")
    print(f"Run directory created: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
