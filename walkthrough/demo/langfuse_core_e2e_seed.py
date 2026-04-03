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
        "--scenario",
        choices=(
            "guided-optimize-observe",
            "dataset-version-lineage",
            "experiment-auto-evaluators",
            "feedback-observability-roundtrip",
            "variant-compare",
            "trace-to-dataset-curation",
        ),
        default="guided-optimize-observe",
        help="Scenario to seed for the frontend Playwright suite.",
    )
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


def api_post_json(base_url: str, api_key: str, path: str, payload: Any) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "X-Request-ID": f"langfuse-core-e2e-post-{path.rsplit('/', 1)[-1]}",
        },
        json=payload,
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


def build_frontend_urls(frontend_url: str, experiment_id: str, experiment_run_id: str, dataset_id: str | None) -> dict[str, str]:
    urls = {
        "base": frontend_url,
        "datasets": f"{frontend_url.rstrip('/')}/datasets",
        "agents": f"{frontend_url.rstrip('/')}/agents",
        "experiments": f"{frontend_url.rstrip('/')}/experiments",
        "experiment": (
            f"{frontend_url.rstrip('/')}/experiments/view/{experiment_id}"
            f"?results_tab=overview&run_id={experiment_run_id}"
        ),
    }
    if dataset_id:
        urls["dataset"] = f"{frontend_url.rstrip('/')}/datasets/view/{dataset_id}"
        urls["legacy_dataset"] = f"{frontend_url.rstrip('/')}/benchmarks/view/{dataset_id}"
    return urls


def build_backend_urls(
    backend_url: str,
    experiment_id: str,
    experiment_run_id: str,
    dataset_id: str | None,
) -> dict[str, str]:
    urls = {
        "base": backend_url,
        "experiment": f"{backend_url.rstrip('/')}/api/v1/experiments/{experiment_id}",
        "experiment_run_results": (
            f"{backend_url.rstrip('/')}/api/v1/experiment-runs/runs/{experiment_run_id}/results"
        ),
        "experiment_runs": (
            f"{backend_url.rstrip('/')}/api/v1/experiment-runs/{experiment_id}/runs?per_page=25"
        ),
    }
    if dataset_id:
        urls["dataset"] = f"{backend_url.rstrip('/')}/api/v1/datasets/{dataset_id}"
        urls["dataset_examples"] = f"{backend_url.rstrip('/')}/api/v1/datasets/{dataset_id}/examples"
    return urls


def add_dataset_version_lineage(
    *,
    args: argparse.Namespace,
    api_key: str,
    dataset_id: str | None,
    run_id: str,
    output_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    if not dataset_id:
        return [], {"skipped": "dataset_id unavailable"}

    v1_label = f"{run_id}-v1"
    v2_label = f"{run_id}-v2"
    v1_payload = api_post_json(
        args.backend_url,
        api_key,
        f"/api/v1/datasets/{dataset_id}/versions",
        {"version_label": v1_label, "description": "Pre-mutation snapshot"},
    )
    api_post_json(
        args.backend_url,
        api_key,
        f"/api/v1/datasets/{dataset_id}/examples",
        {
            "input_text": f"What changed in {run_id}?",
            "expected_output": f"{run_id} version two",
            "metadata": {"scenario": args.scenario, "run_id": run_id},
        },
    )
    v2_payload = api_post_json(
        args.backend_url,
        api_key,
        f"/api/v1/datasets/{dataset_id}/versions",
        {"version_label": v2_label, "description": "Post-mutation snapshot"},
    )
    versions_payload = api_get_json(
        args.backend_url,
        api_key,
        f"/api/v1/datasets/{dataset_id}/versions",
    )

    write_json(output_dir / "dataset-versions.json", versions_payload)
    version_ids = [
        extract_payload(v1_payload).get("id"),
        extract_payload(v2_payload).get("id"),
    ]
    return [value for value in version_ids if value], {
        "versions_listed": len(extract_payload(versions_payload) or []),
        "version_labels": [v1_label, v2_label],
    }


def add_feedback_roundtrip(
    *,
    args: argparse.Namespace,
    api_key: str,
    run_id: str,
    trace_id: str,
    output_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    if not trace_id:
        return [], {"skipped": "trace_id unavailable"}

    ingest_payload = api_post_json(
        args.backend_url,
        api_key,
        "/api/v1beta/observability/ingest",
        {
            "traces": [
                {
                    "id": trace_id,
                    "name": "feedback-roundtrip",
                    "status": "completed",
                    "custom_trace_id": trace_id,
                    "metadata": {
                        "scenario": args.scenario,
                        "run_id": run_id,
                        "phase": "post",
                    },
                }
            ]
        },
    )
    scores_payload = api_post_json(
        args.backend_url,
        api_key,
        "/api/v1beta/scores",
        {
            "trace_id": trace_id,
            "name": "helpfulness",
            "value": 0.91,
            "source": "HUMAN",
            "comment": f"Feedback submitted for {run_id}",
            "metadata": {"scenario": args.scenario},
        },
    )
    listed_scores = api_get_json(
        args.backend_url,
        api_key,
        f"/api/v1beta/scores?trace_id={trace_id}",
    )

    write_json(output_dir / "observability-ingest.json", ingest_payload)
    write_json(output_dir / "observability-scores.json", listed_scores)
    score_ids = extract_payload(scores_payload).get("score_ids") or []
    return [str(score_id) for score_id in score_ids], {
        "ingested_trace_id": trace_id,
        "listed_scores": extract_payload(listed_scores).get("total", 0),
    }


def add_trace_curation(
    *,
    args: argparse.Namespace,
    api_key: str,
    dataset_id: str | None,
    trace_id: str,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    if not dataset_id or not trace_id:
        return {"skipped": "dataset_id or trace_id unavailable"}

    curated_input = f"What should be curated for {run_id}?"
    curated_output = f"Curated output for {run_id}"
    curated_payload = api_post_json(
        args.backend_url,
        api_key,
        f"/api/v1beta/traces/{trace_id}/curate",
        {
            "dataset_id": dataset_id,
            "input_text": curated_input,
            "expected_output": curated_output,
            "metadata": {"scenario": args.scenario, "run_id": run_id},
        },
    )
    examples_payload = api_get_json(
        args.backend_url,
        api_key,
        f"/api/v1/datasets/{dataset_id}/examples",
    )

    write_json(output_dir / "curated-example.json", curated_payload)
    write_json(output_dir / "dataset-examples-after-curation.json", examples_payload)
    return {
        "curated_example_id": extract_payload(curated_payload).get("example_id"),
        "curated_input_text": curated_input,
        "curated_expected_output": curated_output,
        "example_count_after_curation": len(extract_items(examples_payload)),
    }


def add_auto_evaluator_artifacts(
    *,
    args: argparse.Namespace,
    api_key: str,
    experiment_run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    features_payload = api_post_json(
        args.backend_url,
        api_key,
        f"/api/v1/analytics/example-scoring/{experiment_run_id}/features",
        {
            "feature_kind": "scenario_tokens",
            "features": {
                "scenario-root": {
                    "tokens_in": 8,
                    "tokens_out": 5,
                }
            },
        },
    )
    write_json(output_dir / "example-features.json", features_payload)
    return {
        "feature_upload_acknowledged": True,
        "feature_kind": "scenario_tokens",
    }


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

    scenario_id = f"T-E2E-{args.scenario}"
    trace_ids = [
        f"guided-trace:{args.run_id}:baseline",
        f"guided-trace:{args.run_id}:optimize",
        f"guided-trace:{args.run_id}:post",
    ]
    session_ids = [
        f"guided-session:{args.run_id}:baseline",
        f"guided-session:{args.run_id}:optimize",
        f"guided-session:{args.run_id}:post",
    ]
    tag_values = sorted(
        {
            tag
            for summary in (baseline_summary, optimize_summary, post_summary)
            for tag in summary.get("tags", [])
        }
    )

    dataset_version_ids: list[str] = []
    score_ids: list[str] = []
    feedback_ids: list[str] = []
    assertions: dict[str, Any] = {}

    if args.scenario == "dataset-version-lineage":
        dataset_version_ids, assertions = add_dataset_version_lineage(
            args=args,
            api_key=api_key,
            dataset_id=dataset_id,
            run_id=args.run_id,
            output_dir=output_path.parent,
        )
    elif args.scenario == "feedback-observability-roundtrip":
        score_ids, assertions = add_feedback_roundtrip(
            args=args,
            api_key=api_key,
            run_id=args.run_id,
            trace_id=trace_ids[-1],
            output_dir=output_path.parent,
        )
        feedback_ids = list(score_ids)
    elif args.scenario == "trace-to-dataset-curation":
        assertions = add_trace_curation(
            args=args,
            api_key=api_key,
            dataset_id=dataset_id,
            trace_id=trace_ids[-1],
            run_id=args.run_id,
            output_dir=output_path.parent,
        )
    elif args.scenario == "experiment-auto-evaluators":
        assertions = add_auto_evaluator_artifacts(
            args=args,
            api_key=api_key,
            experiment_run_id=experiment_run_id,
            output_dir=output_path.parent,
        )
    elif args.scenario == "variant-compare":
        assertions = {
            "configuration_run_count": len(configuration_run_ids),
            "best_config_present": bool(post_summary.get("best_config") or optimize_summary.get("best_config")),
        }
    else:
        assertions = {"guided_flow": True}

    manifest = {
        "scenario_id": scenario_id,
        "scenario_type": args.scenario,
        "scenario": args.scenario,
        "run_id": args.run_id,
        "mode": args.mode,
        "scale": args.scale,
        "dataset_id": dataset_id,
        "dataset_version_ids": dataset_version_ids,
        "dataset_name": dataset_name,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "experiment_id": experiment_id,
        "experiment_run_id": experiment_run_id,
        "configuration_run_ids": configuration_run_ids,
        "trace_ids": trace_ids,
        "baseline_trace_ids": [f"guided-trace:{args.run_id}:baseline"],
        "post_trace_ids": [f"guided-trace:{args.run_id}:post"],
        "score_ids": score_ids,
        "feedback_ids": feedback_ids,
        "session_ids": session_ids,
        "user_ids": ["guided-demo-user"],
        "tag_values": tag_values,
        "prompt_version_ids": [],
        "best_config": post_summary.get("best_config") or optimize_summary.get("best_config"),
        "best_metrics": post_summary.get("best_metrics") or optimize_summary.get("best_metrics"),
        "frontend_urls": build_frontend_urls(args.frontend_url, experiment_id, experiment_run_id, dataset_id),
        "backend_urls": build_backend_urls(args.backend_url, experiment_id, experiment_run_id, dataset_id),
        "assertions": assertions,
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
