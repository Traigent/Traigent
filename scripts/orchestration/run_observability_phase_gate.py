#!/usr/bin/env python3
"""Run observability phase-gate checks and emit a markdown verification report."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class Stage:
    name: str
    command: list[str]
    timeout_seconds: int = 300


@dataclass
class StageResult:
    stage: Stage
    returncode: int
    duration_seconds: float
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _summary_line(stdout: str, stderr: str) -> str:
    merged = [line.strip() for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
    for line in reversed(merged):
        lowered = line.lower()
        if any(token in lowered for token in ("passed", "failed", "error", "warnings")):
            return line
    return merged[-1] if merged else "No output captured"


def _run_stage(stage: Stage, *, cwd: Path, env: dict[str, str]) -> StageResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            stage.command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=stage.timeout_seconds,
        )
        duration = time.perf_counter() - started
        return StageResult(
            stage=stage,
            returncode=completed.returncode,
            duration_seconds=duration,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started
        timeout_stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        timeout_stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timeout_stderr = (
            timeout_stderr
            + f"\\nStage timed out after {stage.timeout_seconds} seconds."
        ).strip()
        return StageResult(
            stage=stage,
            returncode=124,
            duration_seconds=duration,
            stdout=timeout_stdout,
            stderr=timeout_stderr,
        )


def _write_report(
    *,
    phase_name: str,
    report_path: Path,
    started_at: datetime,
    finished_at: datetime,
    branch_name: str,
    commit_sha: str,
    stage_results: list[StageResult],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Verification Report - {phase_name}")
    lines.append("")
    lines.append(f"- Phase: `{phase_name}`")
    lines.append(f"- Started at: `{started_at.isoformat()}`")
    lines.append(f"- Finished at: `{finished_at.isoformat()}`")
    lines.append(f"- Branch: `{branch_name}`")
    lines.append(f"- Commit: `{commit_sha}`")
    lines.append("")

    for result in stage_results:
        status_text = "PASS" if result.passed else "FAIL"
        command_text = " ".join(result.stage.command)
        lines.append(f"## {result.stage.name}")
        lines.append(f"- command: `{command_text}`")
        lines.append(f"- status: **{status_text}**")
        lines.append(f"- duration_seconds: `{result.duration_seconds:.2f}`")
        lines.append(f"- summary: `{_summary_line(result.stdout, result.stderr)}`")
        if not result.passed:
            tail = (result.stdout + "\n" + result.stderr).splitlines()[-25:]
            lines.append("- output_tail:")
            lines.append("```text")
            lines.extend(tail)
            lines.append("```")
        lines.append("")

    lines.append("## Result")
    overall = "PASS" if all(item.passed for item in stage_results) else "FAIL"
    lines.append(f"- overall: **{overall}**")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        default="phase-4-automation-hardening",
        help="Phase label used in the report filename and header.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    python_bin = project_root / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)

    env = dict(os.environ)
    env["TRAIGENT_MOCK_LLM"] = "true"
    env["TRAIGENT_OFFLINE_MODE"] = "true"

    verification_dir = (
        project_root
        / ".release_review"
        / "observability_orchestration"
        / "verification"
    )
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    report_path = verification_dir / f"{args.phase}_{timestamp}.md"

    branch_name = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root)
        .decode()
        .strip()
    )
    commit_sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root)
        .decode()
        .strip()
    )

    stages = [
        Stage(
            name="Smoke",
            command=[
                str(python_bin),
                "-m",
                "pytest",
                "-m",
                "smoke",
                "tests/smoke/test_observability_phase4_smoke.py",
                "-q",
            ],
        ),
        Stage(
            name="Targeted",
            command=[
                str(python_bin),
                "-m",
                "pytest",
                "tests/unit/core/test_trial_lifecycle.py",
                "tests/unit/integrations/observability/test_workflow_traces.py",
                "-q",
            ],
        ),
        Stage(
            name="Broader",
            command=[
                str(python_bin),
                "-m",
                "pytest",
                "tests/unit/core/test_workflow_trace_manager.py",
                "tests/unit/integrations/observability",
                "-q",
            ],
        ),
    ]

    started_at = datetime.now(UTC)
    results: list[StageResult] = []

    for stage in stages:
        result = _run_stage(stage, cwd=project_root, env=env)
        results.append(result)
        if not result.passed:
            break

    finished_at = datetime.now(UTC)
    _write_report(
        phase_name=args.phase,
        report_path=report_path,
        started_at=started_at,
        finished_at=finished_at,
        branch_name=branch_name,
        commit_sha=commit_sha,
        stage_results=results,
    )

    print(f"Verification report: {report_path}")
    return 0 if all(item.passed for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
