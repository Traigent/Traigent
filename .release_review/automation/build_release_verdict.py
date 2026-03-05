#!/usr/bin/env python3
"""Build canonical release verdict from checks, evidence, and waivers."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REQUIRED_CHECKS = [
    "lint-type",
    "tests-unit",
    "tests-integration",
    "security",
    "dependency-review",
    "codeql",
    "release-review-consistency",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def resolve_within(base_dir: Path, candidate: Path, label: str) -> Path:
    """Resolve candidate path and ensure it stays under base_dir."""
    base_resolved = base_dir.resolve()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (base_resolved / candidate).resolve()

    try:
        resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ValueError(f"{label} must remain under {base_resolved}") from exc

    return resolved


def get_git_sha_short() -> str:
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


def parse_iso_utc(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


@dataclass
class Finding:
    finding_id: str
    severity: str
    file: str
    line: int
    title: str
    status: str


@dataclass
class Waiver:
    waiver_id: str
    finding_id: str
    severity: str
    expires_at: str
    approved_by: list[str]
    reason: str
    valid: bool


def normalize_check_results(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and isinstance(raw.get("checks"), list):
        raw = raw["checks"]
    if not isinstance(raw, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            continue
        status = str(item.get("status") or "unknown").strip().lower()
        normalized.append(
            {
                "key": key,
                "status": status,
                "required": bool(item.get("required", False)),
            }
        )
    return normalized


def is_check_pass(status: str) -> bool:
    return status in {"pass", "success"}


def collect_evidence_findings(evidence_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    if not evidence_root.exists():
        return findings

    for file in sorted(evidence_root.rglob("*.json")):
        data = load_json(file, {})
        if not isinstance(data, dict):
            continue
        raw_findings = data.get("findings", [])
        if not isinstance(raw_findings, list):
            continue
        for item in raw_findings:
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity") or "").upper()
            finding_id = str(item.get("id") or "").strip()
            if not finding_id or severity not in {"P0", "P1", "P2", "P3"}:
                continue
            findings.append(
                Finding(
                    finding_id=finding_id,
                    severity=severity,
                    file=str(item.get("file") or ""),
                    line=int(item.get("line") or 0),
                    title=str(item.get("title") or ""),
                    status=str(item.get("status") or "open").lower(),
                )
            )
    return findings


def collect_waivers(waiver_dir: Path) -> list[Waiver]:
    waivers: list[Waiver] = []
    now = datetime.now(timezone.utc)

    if not waiver_dir.exists():
        return waivers

    for file in sorted(waiver_dir.glob("*.json")):
        data = load_json(file, {})
        if not isinstance(data, dict):
            continue

        approved_by = data.get("approved_by", [])
        if not isinstance(approved_by, list):
            approved_by = []

        expires_at = str(data.get("expires_at") or "")
        expiry_dt = parse_iso_utc(expires_at)
        valid = (
            bool(str(data.get("finding_id") or "").strip())
            and len(approved_by) >= 2
            and expiry_dt is not None
            and expiry_dt > now
        )

        waivers.append(
            Waiver(
                waiver_id=str(data.get("waiver_id") or file.stem),
                finding_id=str(data.get("finding_id") or ""),
                severity=str(data.get("severity") or "").upper(),
                expires_at=expires_at,
                approved_by=[str(x) for x in approved_by],
                reason=str(data.get("reason") or ""),
                valid=valid,
            )
        )

    return waivers


def build_verdict_payload(
    release_id: str,
    baseline_sha: str,
    checks: list[dict[str, Any]],
    findings: list[Finding],
    waivers: list[Waiver],
    required_checks: list[str],
) -> dict[str, Any]:
    waiver_by_finding = {w.finding_id: w for w in waivers if w.valid}

    unresolved_p0: list[dict[str, Any]] = []
    unresolved_p1: list[dict[str, Any]] = []

    for finding in findings:
        if finding.status == "resolved":
            continue
        if finding.finding_id in waiver_by_finding:
            continue

        row = {
            "id": finding.finding_id,
            "severity": finding.severity,
            "file": finding.file,
            "line": finding.line,
            "title": finding.title,
        }
        if finding.severity == "P0":
            unresolved_p0.append(row)
        elif finding.severity == "P1":
            unresolved_p1.append(row)

    check_map = {item["key"]: item for item in checks}
    failed_required_checks: list[dict[str, str]] = []
    for key in required_checks:
        item = check_map.get(key)
        if item is None:
            failed_required_checks.append({"key": key, "status": "missing"})
            continue
        if not is_check_pass(str(item.get("status", "unknown"))):
            failed_required_checks.append({"key": key, "status": str(item.get("status", "unknown"))})

    accepted_waivers = [
        {
            "waiver_id": w.waiver_id,
            "finding_id": w.finding_id,
            "severity": w.severity,
            "expires_at": w.expires_at,
            "approved_by": w.approved_by,
            "reason": w.reason,
        }
        for w in waivers
        if w.valid
    ]

    if unresolved_p0 or unresolved_p1 or failed_required_checks:
        status = "NOT_READY"
    elif accepted_waivers:
        status = "READY_WITH_ACCEPTED_RISKS"
    else:
        status = "READY"

    return {
        "release_id": release_id,
        "baseline_sha": baseline_sha,
        "status": status,
        "unresolved_p0": unresolved_p0,
        "unresolved_p1": unresolved_p1,
        "failed_required_checks": failed_required_checks,
        "waivers": accepted_waivers,
        "generated_at_utc": utc_now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build release verdict JSON")
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--run-dir", help="Defaults to .release_review/runs/<release_id>")
    parser.add_argument("--checks-file", help="Defaults to <run_dir>/gate_results/check_results.json")
    parser.add_argument("--output", help="Defaults to <run_dir>/gate_results/verdict.json")
    parser.add_argument("--baseline-sha", help="Defaults to git HEAD short SHA")
    parser.add_argument(
        "--required-check",
        action="append",
        default=[],
        help="Repeatable required check key override",
    )
    args = parser.parse_args(argv)

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else Path(".release_review") / "runs" / args.release_id
    ).resolve()
    checks_candidate = (
        Path(args.checks_file)
        if args.checks_file
        else Path("gate_results") / "check_results.json"
    )
    output_candidate = (
        Path(args.output)
        if args.output
        else Path("gate_results") / "verdict.json"
    )

    try:
        checks_file = resolve_within(run_dir, checks_candidate, "--checks-file")
        output = resolve_within(run_dir, output_candidate, "--output")
    except ValueError as err:
        print(f"Invalid path argument: {err}")
        return 2

    checks_raw = load_json(checks_file, [])
    checks = normalize_check_results(checks_raw)

    findings = collect_evidence_findings(run_dir / "components")
    waivers = collect_waivers(run_dir / "waivers")

    required_checks = args.required_check if args.required_check else DEFAULT_REQUIRED_CHECKS

    payload = build_verdict_payload(
        release_id=args.release_id,
        baseline_sha=args.baseline_sha or get_git_sha_short(),
        checks=checks,
        findings=findings,
        waivers=waivers,
        required_checks=required_checks,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Wrote verdict: {output}")
    print(f"Status: {payload['status']}")
    return 1 if payload["status"] == "NOT_READY" else 0


if __name__ == "__main__":
    raise SystemExit(main())
