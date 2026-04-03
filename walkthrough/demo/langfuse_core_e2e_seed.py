#!/usr/bin/env python3
"""Seed deterministic Langfuse-core E2E data and emit a manifest.

This wrapper runs the guided optimize+observe demo non-interactively across
baseline -> optimize -> post phases, then records a consolidated manifest that
frontend Playwright tests can use for full-stack assertions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
GUIDED_RUNNER = SCRIPT_DIR / "guided_optimize_and_observe.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the guided optimize+observe demo and emit a single E2E manifest.",
    )
    parser.add_argument("--run-id", required=True, help="Stable run identifier.")
    parser.add_argument(
        "--mode",
        choices=("mock", "real"),
        default="mock",
        help="Execution mode for the guided demo.",
    )
    parser.add_argument(
        "--scale",
        default="tiny",
        help="Optimization scale preset passed to the guided demo.",
    )
    parser.add_argument(
        "--observability",
        choices=("backend", "memory", "auto"),
        default="backend",
        help="Observability delivery mode for the guided demo.",
    )
    parser.add_argument(
        "--frontend-url",
        default="http://localhost:3000",
        help="Frontend base URL stored in the manifest.",
    )
    parser.add_argument(
        "--backend-url",
        default=os.getenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:5001"),
        help="Backend base URL used for canonical verification.",
    )
    parser.add_argument(
        "--baseline-runs",
        type=int,
        default=1,
        help="Number of observed baseline runs.",
    )
    parser.add_argument(
        "--post-runs",
        type=int,
        default=1,
        help="Number of observed post-optimization runs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the manifest JSON.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional artifact root for phase summaries and backend snapshots.",
    )
    return parser.parse_args()


def ensure_api_key() -> str:
    api_key = os.getenv("TRAIGENT_API_KEY")
    if api_key:
        return api_key
    raise RuntimeError("TRAIGENT_API_KEY must be set for langfuse core E2E seeding")


def run_phase(args: argparse.Namespace, phase: str, artifacts_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(GUIDED_RUNNER),
        "--phase",
        phase,
        "--run-id",
        args.run_id,
        "--mode",
        args.mode,
        "--scale",
        args.scale,
        "--observability",
        args.observability,
        "--artifacts-dir",
        str(artifacts_dir),
        "--baseline-runs",
        str(args.baseline_runs),
        "--post-runs",
        str(args.post_runs),
        "--frontend-url",
        args.frontend_url,
    ]

    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    phase_log = artifacts_dir / args.run_id / f"{phase}.log"
    phase_log.parent.mkdir(parents=True, exist_ok=True)
    phase_log.write_text(
        "\n".join(
            [
                f"command: {' '.join(cmd)}",
                f"returncode: {completed.returncode}",
                "",
                "[stdout]",
                completed.stdout,
                "",
                "[stderr]",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"guided phase {phase!r} failed with exit code {completed.returncode}. "
            f"See {phase_log}"
        )

    summary_path = artifacts_dir / args.run_id / f"{phase}_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def extract_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], (dict, list)):
            return payload["data"]
    return payload


def extract_items(payload: Any) -> list[Any]:
    data = extract_payload(payload)
    if isinstance(data, dict):
        for key in ("items", "configuration_runs", "experiment_runs", "benchmarks"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    if isinstance(data, list):
        return data
    return []


def api_get_json(base_url: str, api_key: str, path: str) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    response = requests.get(
        url,
        headers={
            "X-API-Key": api_key,
            "X-Request-ID": f"langfuse-core-e2e-{path.rsplit('/', 1)[-1]}",
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def collect_configuration_run_ids(results_payload: Any) -> list[str]:
    ids: set[str] = set()

    def walk(value: Any, parent_key: str | None = None) -> None:
        if isinstance(value, dict):
            if parent_key in {"configuration_runs", "configurations"}:
                for item in value.values():
                    walk(item, parent_key)
            for key, nested in value.items():
                if key in {"configuration_run_id", "config_run_id"} and nested:
                    ids.add(str(nested))
                elif key == "id" and parent_key in {"configuration_runs", "configurations"} and nested:
                    ids.add(str(nested))
                walk(nested, key)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    if item.get("configuration_run_id"):
                        ids.add(str(item["configuration_run_id"]))
                    elif item.get("id") and parent_key in {"configuration_runs", "configurations"}:
                        ids.add(str(item["id"]))
                walk(item, parent_key)

    walk(results_payload)
    return sorted(ids)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve()
    artifacts_dir = (
        Path(args.artifacts_dir).resolve()
        if args.artifacts_dir
        else output_path.parent / "guided-artifacts"
    )
    api_key = ensure_api_key()

    baseline_summary = run_phase(args, "baseline", artifacts_dir)
    optimize_summary = run_phase(args, "optimize", artifacts_dir)
    post_summary = run_phase(args, "post", artifacts_dir)

    experiment_id = (
        post_summary.get("experiment_id")
        or optimize_summary.get("experiment_id")
        or baseline_summary.get("experiment_id")
    )
    experiment_run_id = (
        post_summary.get("experiment_run_id")
        or optimize_summary.get("experiment_run_id")
        or baseline_summary.get("experiment_run_id")
    )

    if not experiment_id or not experiment_run_id:
        raise RuntimeError("Unable to resolve experiment_id / experiment_run_id from guided summaries")

    experiment_payload = api_get_json(args.backend_url, api_key, f"/api/v1/experiments/{experiment_id}")
    run_results_payload = api_get_json(
        args.backend_url,
        api_key,
        f"/api/v1/experiment-runs/runs/{experiment_run_id}/results",
    )
    runs_payload = api_get_json(
        args.backend_url,
        api_key,
        f"/api/v1/experiment-runs/{experiment_id}/runs?per_page=25",
    )

    write_json(output_path.parent / "backend-experiment.json", experiment_payload)
    write_json(output_path.parent / "backend-run-results.json", run_results_payload)
    write_json(output_path.parent / "backend-runs.json", runs_payload)

    experiment = extract_payload(experiment_payload)
    run_results = extract_payload(run_results_payload)
    configuration_run_ids = collect_configuration_run_ids(run_results_payload)

    dataset_id = None
    agent_id = None
    agent_name = None
    dataset_name = None
    if isinstance(experiment, dict):
        dataset_id = (
            experiment.get("dataset_id")
            or experiment.get("eval_dataset_id")
            or experiment.get("evaluation_set_id")
            or experiment.get("benchmark_id")
        )
        agent_id = experiment.get("agent_id")
        agent_name = experiment.get("agent_name") or experiment.get("agent", {}).get("name")
        dataset_name = (
            experiment.get("dataset_name")
            or experiment.get("dataset", {}).get("name")
            or experiment.get("eval_dataset", {}).get("name")
        )

    manifest = {
        "scenario": "guided-optimize-observe",
        "run_id": args.run_id,
        "mode": args.mode,
        "scale": args.scale,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "experiment_id": experiment_id,
        "experiment_run_id": experiment_run_id,
        "configuration_run_ids": configuration_run_ids,
        "baseline_trace_ids": [f"guided-trace:{args.run_id}:baseline"],
        "post_trace_ids": [f"guided-trace:{args.run_id}:post"],
        "best_config": post_summary.get("best_config") or optimize_summary.get("best_config"),
        "best_metrics": post_summary.get("best_metrics") or optimize_summary.get("best_metrics"),
        "frontend_urls": {
            "base": args.frontend_url,
            "datasets": f"{args.frontend_url.rstrip('/')}/datasets",
            "agents": f"{args.frontend_url.rstrip('/')}/agents",
            "experiments": f"{args.frontend_url.rstrip('/')}/experiments",
            "experiment": (
                f"{args.frontend_url.rstrip('/')}/experiments/view/{experiment_id}"
                f"?results_tab=overview&run_id={experiment_run_id}"
            ),
        },
        "backend_urls": {
            "base": args.backend_url,
            "experiment": f"{args.backend_url.rstrip('/')}/api/v1/experiments/{experiment_id}",
            "experiment_run_results": (
                f"{args.backend_url.rstrip('/')}/api/v1/experiment-runs/runs/{experiment_run_id}/results"
            ),
            "experiment_runs": (
                f"{args.backend_url.rstrip('/')}/api/v1/experiment-runs/{experiment_id}/runs?per_page=25"
            ),
        },
        "phase_summaries": {
            "baseline": baseline_summary,
            "optimize": optimize_summary,
            "post": post_summary,
        },
        "backend_snapshots": {
            "experiment": str((output_path.parent / "backend-experiment.json").resolve()),
            "run_results": str((output_path.parent / "backend-run-results.json").resolve()),
            "runs": str((output_path.parent / "backend-runs.json").resolve()),
        },
    }

    write_json(output_path, manifest)
    print(str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
