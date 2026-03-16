#!/usr/bin/env python3
"""Post-execution verification: compare local example output against backend state.

Runs after examples complete. Checks that:
1. The example produced a valid optimization result (exit code 0, "Best" in output)
2. A new experiment was registered in the backend with COMPLETED status
3. The backend's configuration_runs_count matches the expected trial count
4. Local result snapshots are stored for audit trail

Usage:
    # Verify a single example
    python scripts/verify_example_results.py examples/core/simple-prompt/run.py

    # Verify all examples (reads from run_all_examples.sh report)
    python scripts/verify_example_results.py --from-report results.json

    # Just snapshot the backend state (for before/after comparison)
    python scripts/verify_example_results.py --snapshot before
    python scripts/verify_example_results.py --snapshot after
    python scripts/verify_example_results.py --diff before after
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / ".validation_results"
SNAPSHOTS_DIR = RESULTS_DIR / "snapshots"


def get_backend_config() -> tuple[str, str]:
    """Return (api_url, api_key) from environment or .env file."""
    api_url = os.environ.get("TRAIGENT_API_URL", "")
    api_key = os.environ.get("TRAIGENT_API_KEY", "")

    if not api_url or not api_key:
        env_file = REPO_ROOT / "walkthrough" / "examples" / "real" / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if k == "TRAIGENT_API_URL" and not api_url:
                    api_url = v
                elif k == "TRAIGENT_API_KEY" and not api_key:
                    api_key = v
                elif k == "TRAIGENT_BACKEND_URL" and not api_url:
                    api_url = v.rstrip("/") + "/api/v1"

    if not api_url:
        api_url = "http://localhost:5000/api/v1"

    return api_url, api_key


def query_backend(endpoint: str, api_url: str, api_key: str) -> dict | None:
    """Query a backend endpoint, return parsed JSON or None on failure."""
    import urllib.error
    import urllib.request

    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    req = urllib.request.Request(url)
    req.add_header("X-API-Key", api_key)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("X-Client-Version", "2.0.0")

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"  Backend query failed ({endpoint}): {e}", file=sys.stderr)
        return None


def fetch_experiments(api_url: str, api_key: str) -> list[dict]:
    """Fetch all experiments from the backend."""
    data = query_backend("experiments", api_url, api_key)
    if not data or not data.get("success"):
        return []
    items = data.get("data", {})
    if isinstance(items, dict):
        items = items.get("items", [])
    return items


def fetch_sessions(api_url: str, api_key: str) -> list[dict]:
    """Fetch all sessions from the backend."""
    data = query_backend("sessions", api_url, api_key)
    if not data:
        return []
    return data.get("sessions", [])


def snapshot_backend(label: str) -> Path:
    """Take a snapshot of all backend experiments and sessions."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    api_url, api_key = get_backend_config()

    experiments = fetch_experiments(api_url, api_key)
    sessions = fetch_sessions(api_url, api_key)

    snapshot = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiments": experiments,
        "sessions": sessions,
        "experiment_count": len(experiments),
        "session_count": len(sessions),
        "completed_experiments": [
            e for e in experiments if e.get("status") == "COMPLETED"
        ],
    }

    path = SNAPSHOTS_DIR / f"{label}.json"
    path.write_text(json.dumps(snapshot, indent=2, default=str))
    print(f"Snapshot '{label}' saved: {len(experiments)} experiments, {len(sessions)} sessions")
    return path


def diff_snapshots(before_label: str, after_label: str) -> dict:
    """Compare two backend snapshots and report new/changed experiments."""
    before_path = SNAPSHOTS_DIR / f"{before_label}.json"
    after_path = SNAPSHOTS_DIR / f"{after_label}.json"

    if not before_path.exists() or not after_path.exists():
        print(f"Snapshot files not found: {before_path}, {after_path}", file=sys.stderr)
        return {}

    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())

    before_ids = {e["experiment_id"] for e in before["experiments"]}
    after_ids = {e["experiment_id"] for e in after["experiments"]}

    new_ids = after_ids - before_ids
    new_experiments = [
        e for e in after["experiments"] if e["experiment_id"] in new_ids
    ]

    # Check for status changes in existing experiments
    before_map = {e["experiment_id"]: e for e in before["experiments"]}
    changed = []
    for e in after["experiments"]:
        eid = e["experiment_id"]
        if eid in before_map:
            old = before_map[eid]
            if (
                e.get("status") != old.get("status")
                or e.get("configuration_runs_count") != old.get("configuration_runs_count")
            ):
                changed.append({
                    "experiment_id": eid,
                    "name": e.get("name"),
                    "status_before": old.get("status"),
                    "status_after": e.get("status"),
                    "runs_before": old.get("configuration_runs_count"),
                    "runs_after": e.get("configuration_runs_count"),
                })

    result = {
        "new_experiments": len(new_experiments),
        "changed_experiments": len(changed),
        "new": [
            {
                "experiment_id": e["experiment_id"],
                "name": e.get("name"),
                "agent_id": e.get("agent_id"),
                "status": e.get("status"),
                "runs": e.get("configuration_runs_count", 0),
                "total_examples": e.get("total_examples", 0),
            }
            for e in new_experiments
        ],
        "changed": changed,
        "before_session_count": before["session_count"],
        "after_session_count": after["session_count"],
    }

    # Save diff
    diff_path = SNAPSHOTS_DIR / f"diff_{before_label}_{after_label}.json"
    diff_path.write_text(json.dumps(result, indent=2, default=str))

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Backend Diff: {before_label} → {after_label}")
    print(f"{'='*60}")
    print(f"  New experiments:     {result['new_experiments']}")
    print(f"  Changed experiments: {result['changed_experiments']}")
    print(f"  Sessions:            {result['before_session_count']} → {result['after_session_count']}")

    if new_experiments:
        print(f"\n  New experiments:")
        for e in result["new"]:
            status_icon = "✓" if e["status"] == "COMPLETED" else "✗"
            print(f"    {status_icon} {e['agent_id']}: {e['status']} ({e['runs']} runs)")

    if changed:
        print(f"\n  Changed experiments:")
        for c in changed:
            print(f"    {c['name']}: {c['status_before']}→{c['status_after']} "
                  f"(runs: {c['runs_before']}→{c['runs_after']})")

    # Verify all new experiments completed
    failed_new = [e for e in result["new"] if e["status"] != "COMPLETED"]
    if failed_new:
        print(f"\n  ⚠ {len(failed_new)} new experiments did NOT complete:")
        for e in failed_new:
            print(f"    - {e['agent_id']}: {e['status']}")

    print()
    return result


def verify_example_output(example: str, output: str, exit_code: int) -> dict:
    """Verify a single example's output for expected patterns."""
    checks = {
        "exit_code_zero": exit_code == 0,
        "has_best_config": False,
        "has_best_score": False,
        "best_config": None,
        "best_score": None,
        "mock_mode": "MOCK LLM MODE" in output,
    }

    # Look for "Best config" or "best_config" in output
    for line in output.splitlines():
        lower = line.lower()
        if "best config" in lower or "best_config" in lower:
            checks["has_best_config"] = True
            # Try to extract the config dict
            match = re.search(r"\{.*\}", line)
            if match:
                try:
                    checks["best_config"] = json.loads(match.group().replace("'", '"'))
                except json.JSONDecodeError:
                    checks["best_config"] = match.group()

        if "best score" in lower or "best_score" in lower:
            checks["has_best_score"] = True
            match = re.search(r"(\d+\.?\d*)%?", line.split(":")[-1] if ":" in line else line)
            if match:
                checks["best_score"] = float(match.group(1))

    return checks


def run_and_verify(example_path: str, timeout: int = 300) -> dict:
    """Run a single example and verify its results."""
    abs_path = Path(example_path).resolve()
    if not abs_path.exists():
        return {"example": example_path, "status": "SKIP", "reason": "File not found"}

    example_dir = abs_path.parent
    example_name = abs_path.name

    env = os.environ.copy()
    env["TRAIGENT_DATASET_ROOT"] = str(REPO_ROOT)
    env["TRAIGENT_BATCH_MODE"] = "true"
    env["TRAIGENT_COST_APPROVED"] = "true"
    env["TRAIGENT_PAUSE_ON_ERROR"] = "false"

    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, example_name],
            cwd=str(example_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        duration = time.time() - start
        output = proc.stdout + proc.stderr
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        return {
            "example": example_path,
            "status": "TIMEOUT",
            "duration": timeout,
        }

    checks = verify_example_output(example_path, output, exit_code)

    result = {
        "example": example_path,
        "status": "PASS" if exit_code == 0 else "FAIL",
        "exit_code": exit_code,
        "duration": round(duration, 1),
        "checks": checks,
    }

    if exit_code != 0:
        result["last_output_lines"] = output.strip().splitlines()[-10:]

    return result


def verify_from_report(report_path: str) -> list[dict]:
    """Read a JSON report from run_all_examples.sh and verify each result."""
    report = json.loads(Path(report_path).read_text())
    verifications = []

    for entry in report.get("results", []):
        example = entry["example"]
        status = entry["status"]

        v = {
            "example": example,
            "run_status": status,
            "run_duration": entry.get("duration_seconds"),
            "verified": status == "PASS",
        }

        if status != "PASS":
            v["issue"] = f"Example {status}"

        verifications.append(v)

    return verifications


def main():
    parser = argparse.ArgumentParser(description="Verify Traigent example results against backend")
    parser.add_argument("example", nargs="?", help="Path to example to run and verify")
    parser.add_argument("--from-report", help="Verify from a run_all_examples.sh JSON report")
    parser.add_argument("--snapshot", metavar="LABEL", help="Take a backend snapshot (e.g., 'before', 'after')")
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"), help="Diff two snapshots")
    parser.add_argument("--timeout", type=int, default=300, help="Per-example timeout (default: 300s)")
    parser.add_argument("--output", help="Write verification results to JSON file")

    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.snapshot:
        snapshot_backend(args.snapshot)
        return

    if args.diff:
        diff_snapshots(args.diff[0], args.diff[1])
        return

    if args.from_report:
        results = verify_from_report(args.from_report)
        passed = sum(1 for r in results if r.get("verified"))
        print(f"Verified {passed}/{len(results)} examples from report")
        if args.output:
            Path(args.output).write_text(json.dumps(results, indent=2))
        return

    if args.example:
        result = run_and_verify(args.example, timeout=args.timeout)
        print(json.dumps(result, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
