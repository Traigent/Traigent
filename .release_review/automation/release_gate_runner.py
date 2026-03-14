#!/usr/bin/env python3
"""Execute release gate checks and emit check results for verdict generation."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from build_release_verdict import main as build_verdict_main

RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CHECK_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class GateCheck:
    key: str
    description: str
    command: list[str]
    required: bool
    timeout_seconds: int


@dataclass
class GateResult:
    key: str
    description: str
    command: list[str]
    required: bool
    status: str
    exit_code: int | None
    duration_seconds: float
    reason: str | None
    stdout_log: str
    stderr_log: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_release_id(release_id: str) -> str:
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError(
            "release_id must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ "
            "(alphanumeric, dot, underscore, hyphen only)"
        )
    return release_id


def validate_check_key(check_key: str) -> str:
    if not CHECK_KEY_PATTERN.fullmatch(check_key):
        raise ValueError(
            "check key must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ "
            "(alphanumeric, dot, underscore, hyphen only)"
        )
    return check_key


def resolve_child_path(root: Path, filename: str) -> Path:
    root_resolved = root.resolve()
    candidate = Path(filename)
    if candidate.name != filename:
        raise ValueError("child filename must not include directory components")
    target = (root_resolved / candidate).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("resolved child path escapes root directory") from exc
    return target


def resolve_run_dir(run_dir_arg: str | None, release_id: str) -> Path:
    runs_root = Path(".release_review") / "runs"
    runs_root_resolved = runs_root.resolve()

    if run_dir_arg:
        candidate = Path(run_dir_arg)
        if candidate.is_absolute():
            run_dir = candidate.resolve()
        else:
            run_dir = (Path.cwd() / candidate).resolve()
    else:
        run_dir = (runs_root_resolved / release_id).resolve()

    try:
        run_dir.relative_to(runs_root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"run directory must remain under {runs_root_resolved}"
        ) from exc
    return run_dir


def command_exists(command: list[str]) -> bool:
    if not command:
        return False
    executable = command[0]
    if "/" in executable:
        return Path(executable).exists()
    return shutil.which(executable) is not None


def get_default_checks(mode: str) -> list[GateCheck]:
    checks = [
        GateCheck(
            key="lint-type",
            description="Lint and type checks",
            command=[
                "bash",
                "-lc",
                "python3 -m black --check --diff traigent/ traigent_validation/ "
                "&& python3 -m isort --check-only --diff traigent/ traigent_validation/ "
                "&& python3 -m ruff check traigent/ traigent_validation/ "
                "&& python3 -m mypy traigent/ traigent_validation/ --install-types --non-interactive",
            ],
            required=True,
            timeout_seconds=1800,
        ),
        GateCheck(
            key="tests-unit",
            description="Unit tests",
            command=[
                "bash",
                "-lc",
                "TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true PYTHONPATH=. "
                "python3 -m pytest tests/unit -q --tb=short",
            ],
            required=True,
            timeout_seconds=2400,
        ),
        GateCheck(
            key="tests-integration",
            description="Integration tests",
            command=[
                "bash",
                "-lc",
                "TRAIGENT_MOCK_LLM=true TRAIGENT_OFFLINE_MODE=true PYTHONPATH=. "
                "python3 -m pytest tests/integration -q --tb=short",
            ],
            required=True,
            timeout_seconds=3000,
        ),
        GateCheck(
            key="security",
            description="Security baseline checks",
            command=["make", "security-check"],
            required=True,
            timeout_seconds=1800,
        ),
        GateCheck(
            key="dependency-review",
            description="Dependency vulnerability review",
            command=["pip-audit", "--progress-spinner", "off", "--desc", "-r", "requirements/requirements.txt"],
            required=True,
            timeout_seconds=900,
        ),
        GateCheck(
            key="codeql",
            description="CodeQL (CI-native); local mode marks as not applicable",
            command=["python3", "-c", "print('codeql check delegated to CI workflow')"],
            required=(mode == "ci"),
            timeout_seconds=60,
        ),
        GateCheck(
            key="release-review-consistency",
            description="Release-review docs/protocol consistency",
            command=["python3", ".release_review/automation/validate_release_review_consistency.py"],
            required=True,
            timeout_seconds=180,
        ),
    ]

    if mode == "local":
        adjusted: list[GateCheck] = []
        for check in checks:
            if check.key == "codeql":
                adjusted.append(
                    GateCheck(
                        key=check.key,
                        description=check.description,
                        command=check.command,
                        required=False,
                        timeout_seconds=check.timeout_seconds,
                    )
                )
            else:
                adjusted.append(check)
        return adjusted

    return checks


def run_check(check: GateCheck, logs_dir: Path, strict: bool) -> GateResult:
    check_key = validate_check_key(check.key)
    stdout_log = resolve_child_path(logs_dir, f"{check_key}.stdout.log")
    stderr_log = resolve_child_path(logs_dir, f"{check_key}.stderr.log")

    if not command_exists(check.command):
        stdout_log.write_text("")
        stderr_log.write_text(f"Missing command: {check.command[0]}\n")
        status = "fail" if strict and check.required else "skipped_missing_tool"
        return GateResult(
            key=check.key,
            description=check.description,
            command=check.command,
            required=check.required,
            status=status,
            exit_code=None,
            duration_seconds=0.0,
            reason=f"Command not found: {check.command[0]}",
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
        )

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            check.command,
            capture_output=True,
            text=True,
            check=False,
            timeout=check.timeout_seconds,
        )
        elapsed = time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout_log.write_text(exc.stdout or "")
        stderr_log.write_text((exc.stderr or "") + f"\nTimeout after {check.timeout_seconds}s\n")
        return GateResult(
            key=check.key,
            description=check.description,
            command=check.command,
            required=check.required,
            status="timeout",
            exit_code=None,
            duration_seconds=elapsed,
            reason=f"Timeout after {check.timeout_seconds}s",
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
        )

    stdout_log.write_text(proc.stdout or "")
    stderr_log.write_text(proc.stderr or "")

    status = "pass" if proc.returncode == 0 else "fail"
    reason = None if proc.returncode == 0 else f"Exit code {proc.returncode}"
    return GateResult(
        key=check.key,
        description=check.description,
        command=check.command,
        required=check.required,
        status=status,
        exit_code=proc.returncode,
        duration_seconds=elapsed,
        reason=reason,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
    )


def write_summary_markdown(
    path: Path,
    release_id: str,
    mode: str,
    strict: bool,
    started_at: str,
    finished_at: str,
    results: list[GateResult],
) -> None:
    lines = [
        f"# Release Gate Summary: {release_id}",
        "",
        f"- Mode: {mode}",
        f"- Strict: {'yes' if strict else 'no'}",
        f"- Started: {started_at}",
        f"- Finished: {finished_at}",
        "",
        "| Check | Required | Status | Exit | Duration (s) | Command |",
        "|---|---|---|---|---|---|",
    ]

    for result in results:
        exit_code = "-" if result.exit_code is None else str(result.exit_code)
        lines.append(
            f"| {result.key} | {'yes' if result.required else 'no'} | {result.status} | "
            f"{exit_code} | {result.duration_seconds:.1f} | `{' '.join(result.command)}` |"
        )

    lines.append("")
    lines.append("Artifacts:")
    lines.append("- `gate_results/check_results.json`")
    lines.append("- `gate_results/verdict.json`")
    lines.append("- `logs/*.stdout.log`, `logs/*.stderr.log`")

    path.write_text("\n".join(lines) + "\n")


def ensure_run_workspace(run_dir: Path, release_id: str, base_branch: str) -> None:
    gate_results = run_dir / "gate_results"
    inventories = run_dir / "inventories"
    components = run_dir / "components"
    file_reviews = run_dir / "file_reviews"
    waivers = run_dir / "waivers"
    logs = run_dir / "logs"

    gate_results.mkdir(parents=True, exist_ok=True)
    inventories.mkdir(parents=True, exist_ok=True)
    components.mkdir(parents=True, exist_ok=True)
    file_reviews.mkdir(parents=True, exist_ok=True)
    waivers.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    manifest = resolve_child_path(run_dir, "run_manifest.json")
    if not manifest.exists():
        manifest_payload = {
            "release_id": release_id,
            "base_branch": base_branch,
            "created_at_utc": utc_now(),
            "protocol_version": 3,
        }
        manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n")


def write_inventories(run_dir: Path) -> None:
    inventories = run_dir / "inventories"

    src_files: list[str] = []
    for root in ("traigent", "traigent_validation"):
        root_path = Path(root)
        if root_path.exists():
            src_files.extend(str(path) for path in root_path.rglob("*.py"))

    test_files = sorted(str(path) for path in Path("tests").rglob("*.py")) if Path("tests").exists() else []

    src_files = sorted(src_files)
    resolve_child_path(inventories, "src_files.txt").write_text(
        "\n".join(src_files) + ("\n" if src_files else "")
    )
    resolve_child_path(inventories, "tests_files.txt").write_text(
        "\n".join(test_files) + ("\n" if test_files else "")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run release gate checks")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--run-dir", help="Defaults to .release_review/runs/<release_id>")
    parser.add_argument("--mode", choices=["local", "ci"], default="local")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--base-branch", default="main")
    args = parser.parse_args()

    release_id = validate_release_id(args.release_id)
    run_dir = resolve_run_dir(args.run_dir, release_id)
    ensure_run_workspace(run_dir, release_id, args.base_branch)
    write_inventories(run_dir)

    checks = get_default_checks(args.mode)
    logs_dir = run_dir / "logs"

    started_at = utc_now()
    results = [run_check(check, logs_dir, args.strict) for check in checks]
    finished_at = utc_now()

    checks_payload = {
        "release_id": release_id,
        "mode": args.mode,
        "strict": args.strict,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "checks": [asdict(result) for result in results],
    }

    checks_file = run_dir / "gate_results" / "check_results.json"
    checks_file.write_text(json.dumps(checks_payload, indent=2) + "\n")

    write_summary_markdown(
        path=run_dir / "gate_results" / "summary.md",
        release_id=release_id,
        mode=args.mode,
        strict=args.strict,
        started_at=started_at,
        finished_at=finished_at,
        results=results,
    )

    required_keys = [check.key for check in checks if check.required]

    verdict_args = [
        "--release-id",
        release_id,
        "--run-dir",
        str(run_dir),
        "--checks-file",
        str(checks_file),
    ]
    for key in required_keys:
        verdict_args.extend(["--required-check", key])

    verdict_exit = int(build_verdict_main(verdict_args))

    failures = [r for r in results if r.required and r.status != "pass"]
    if args.strict and failures:
        return 1
    return verdict_exit


if __name__ == "__main__":
    raise SystemExit(main())
