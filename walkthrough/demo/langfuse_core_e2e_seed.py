#!/usr/bin/env python3
"""Seed deterministic Langfuse-core E2E data and emit a manifest.

This wrapper runs the guided optimize+observe demo non-interactively across
baseline -> optimize -> post phases, then records a consolidated manifest that
frontend Playwright tests can use for full-stack assertions.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
GUIDED_RUNNER = SCRIPT_DIR / "guided_optimize_and_observe.py"
TOKEN_CACHE_PATH = Path(
    os.getenv("LANGFUSE_CORE_E2E_TOKEN_CACHE", "/tmp/langfuse_core_e2e_token.json")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the guided optimize+observe demo and emit a single E2E manifest.",
    )
    parser.add_argument("--run-id", required=True, help="Stable run identifier.")
    parser.add_argument(
        "--scenario",
        choices=(
            "guided-optimize-observe",
            "access-control-isolation",
            "dataset-version-lineage",
            "experiment-auto-evaluators",
            "feedback-observability-roundtrip",
            "tool-calling-multistep-trace",
            "variant-compare",
            "trace-to-dataset-curation",
            "trace-session-user-browse",
            "trace-feedback-collaboration",
            "prompt-version-lineage",
            "playground-run-and-compare",
            "trace-to-prompt-lineage",
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


def _decode_jwt_expiration(token: str) -> int | None:
    try:
        _header, payload, _sig = token.split(".", 2)
        padded = payload + "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        exp = json.loads(decoded).get("exp")
        return int(exp) if exp is not None else None
    except Exception:
        return None


def _load_cached_bearer_token() -> str | None:
    if not TOKEN_CACHE_PATH.exists():
        return None
    try:
        payload = json.loads(TOKEN_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

    token = payload.get("token")
    if not isinstance(token, str) or not token:
        return None

    expires_at = payload.get("expires_at")
    now = int(time.time())
    if isinstance(expires_at, int) and expires_at - now > 60:
        os.environ["TRAIGENT_JWT_TOKEN"] = token
        return token
    if isinstance(expires_at, float) and expires_at - now > 60:
        os.environ["TRAIGENT_JWT_TOKEN"] = token
        return token
    return None


def _store_cached_bearer_token(token: str) -> None:
    expires_at = _decode_jwt_expiration(token)
    TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_CACHE_PATH.write_text(
        json.dumps(
            {
                "token": token,
                "expires_at": expires_at,
                "cached_at": int(time.time()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def ensure_bearer_token(base_url: str) -> str | None:
    token = os.getenv("TRAIGENT_JWT_TOKEN")
    if token:
        return token

    cached_token = _load_cached_bearer_token()
    if cached_token:
        return cached_token

    email = os.getenv("LANGFUSE_CORE_E2E_EMAIL")
    password = os.getenv("LANGFUSE_CORE_E2E_PASSWORD")
    if not email or not password:
        return None

    for attempt in range(4):
        response = requests.post(
            f"{base_url.rstrip('/')}/api/v1/auth/login",
            headers={"Content-Type": "application/json"},
            json={"email": email, "password": password},
            timeout=30,
        )
        if response.status_code == 429:
            cached_token = _load_cached_bearer_token()
            if cached_token:
                return cached_token
            if attempt < 3:
                time.sleep(1.0 * (attempt + 1))
                continue
        response.raise_for_status()
        payload = response.json()
        token = payload.get("data", {}).get("access_token")
        if isinstance(token, str) and token:
            os.environ["TRAIGENT_JWT_TOKEN"] = token
            _store_cached_bearer_token(token)
            return token
        return None
    return None


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


def api_request_json(base_url: str, api_key: str, method: str, path: str, payload: Any | None = None) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    jwt_token = ensure_bearer_token(base_url)
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
        "X-Request-ID": f"langfuse-core-e2e-{method.lower()}-{path.rsplit('/', 1)[-1]}",
    }
    if jwt_token:
        headers["Authorization"] = f"Bearer {jwt_token}"
    response = requests.request(
        method.upper(),
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def api_get_json(base_url: str, api_key: str, path: str) -> Any:
    return api_request_json(base_url, api_key, "GET", path)


def api_post_json(base_url: str, api_key: str, path: str, payload: Any) -> Any:
    return api_request_json(base_url, api_key, "POST", path, payload)


def api_put_json(base_url: str, api_key: str, path: str, payload: Any) -> Any:
    return api_request_json(base_url, api_key, "PUT", path, payload)


def api_patch_json(base_url: str, api_key: str, path: str, payload: Any) -> Any:
    return api_request_json(base_url, api_key, "PATCH", path, payload)


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


def build_frontend_urls(
    frontend_url: str,
    experiment_id: str | None,
    experiment_run_id: str | None,
    dataset_id: str | None,
) -> dict[str, str]:
    urls = {
        "base": frontend_url,
        "datasets": f"{frontend_url.rstrip('/')}/datasets",
        "agents": f"{frontend_url.rstrip('/')}/agents",
        "experiments": f"{frontend_url.rstrip('/')}/experiments",
    }
    if experiment_id and experiment_run_id:
        urls["experiment"] = (
            f"{frontend_url.rstrip('/')}/experiments/view/{experiment_id}"
            f"?results_tab=overview&run_id={experiment_run_id}"
        )
    if dataset_id:
        urls["dataset"] = f"{frontend_url.rstrip('/')}/datasets/view/{dataset_id}"
        urls["legacy_dataset"] = f"{frontend_url.rstrip('/')}/benchmarks/view/{dataset_id}"
    return urls


def build_backend_urls(
    backend_url: str,
    experiment_id: str | None,
    experiment_run_id: str | None,
    dataset_id: str | None,
) -> dict[str, str]:
    urls = {
        "base": backend_url,
    }
    if experiment_id:
        urls["experiment"] = f"{backend_url.rstrip('/')}/api/v1/experiments/{experiment_id}"
        urls["experiment_runs"] = (
            f"{backend_url.rstrip('/')}/api/v1/experiment-runs/{experiment_id}/runs?per_page=25"
        )
    if experiment_run_id:
        urls["experiment_run_results"] = (
            f"{backend_url.rstrip('/')}/api/v1/experiment-runs/runs/{experiment_run_id}/results"
        )
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


def seed_observability_fixture(
    *,
    args: argparse.Namespace,
    api_key: str,
    run_id: str,
    output_dir: Path,
    prompt_name: str | None = None,
    prompt_version: int | None = None,
) -> dict[str, Any]:
    trace_id = f"obs-trace:{run_id}"
    session_id = f"obs-session:{run_id}"
    user_id = f"obs-user:{run_id}"
    trace_name = f"langfuse-observability-{run_id}"
    tags = ["langfuse-core", "observability", run_id]

    prompt_reference = None
    if prompt_name:
        prompt_reference = {
            "name": prompt_name,
            "version": prompt_version,
            "variables": {"topic": run_id},
        }

    ingest_payload = api_post_json(
        args.backend_url,
        api_key,
        "/api/v1beta/observability/ingest",
        {
            "traces": [
                {
                    "id": trace_id,
                    "name": trace_name,
                    "status": "completed",
                    "session_id": session_id,
                    "user_id": user_id,
                    "environment": "playground",
                    "release": "langfuse-e2e",
                    "tags": tags,
                    "metadata": {
                        "scenario": args.scenario,
                        "run_id": run_id,
                        "source": "sdk-seed",
                    },
                    "prompt_reference": prompt_reference,
                    "session": {
                        "id": session_id,
                        "user_id": user_id,
                        "environment": "playground",
                        "release": "langfuse-e2e",
                        "tags": tags,
                        "metadata": {"scenario": args.scenario, "run_id": run_id},
                    },
                    "observations": [
                        {
                            "id": f"{trace_id}:root",
                            "type": "span",
                            "name": "pipeline",
                            "status": "completed",
                            "latency_ms": 42,
                            "input_tokens": 4,
                            "output_tokens": 3,
                            "metadata": {"stage": "pipeline"},
                            "children": [
                                {
                                    "id": f"{trace_id}:generation",
                                    "type": "generation",
                                    "name": "answer",
                                    "status": "completed",
                                    "latency_ms": 37,
                                    "input_tokens": 12,
                                    "output_tokens": 18,
                                    "cost_usd": 0.0012,
                                    "model_name": "mock-model",
                                    "input_data": {"question": f"How did {run_id} perform?"},
                                    "output_data": {"answer": f"{run_id} completed successfully."},
                                    "metadata": {"scenario": args.scenario},
                                    "prompt_reference": prompt_reference,
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    )
    trace_payload = api_get_json(args.backend_url, api_key, f"/api/v1beta/observability/traces/{quote(trace_id, safe='')}")
    session_payload = api_get_json(args.backend_url, api_key, f"/api/v1beta/observability/sessions/{quote(session_id, safe='')}")
    user_payload = api_get_json(args.backend_url, api_key, f"/api/v1beta/observability/users/{quote(user_id, safe='')}")

    write_json(output_dir / "observability-fixture-ingest.json", ingest_payload)
    write_json(output_dir / "observability-fixture-trace.json", trace_payload)
    write_json(output_dir / "observability-fixture-session.json", session_payload)
    write_json(output_dir / "observability-fixture-user.json", user_payload)

    return {
        "trace_id": trace_id,
        "trace_name": trace_name,
        "session_id": session_id,
        "user_id": user_id,
        "tags": tags,
    }


def seed_tool_calling_multistep_trace_fixture(
    *,
    args: argparse.Namespace,
    api_key: str,
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    trace_id = f"tool-trace:{run_id}"
    session_id = f"tool-session:{run_id}"
    user_id = f"tool-user:{run_id}"
    trace_name = f"langfuse-tool-calling-{run_id}"
    custom_trace_id = f"tool-calling:{run_id}"
    tool_names = ["crm_lookup", "shipping_eta_api"]
    tags = ["langfuse-core", "observability", "tool-calling", run_id]

    ingest_payload = api_post_json(
        args.backend_url,
        api_key,
        "/api/v1beta/observability/ingest",
        {
            "traces": [
                {
                    "id": trace_id,
                    "name": trace_name,
                    "status": "completed",
                    "session_id": session_id,
                    "user_id": user_id,
                    "environment": "support",
                    "release": "langfuse-e2e",
                    "custom_trace_id": custom_trace_id,
                    "tags": tags,
                    "input_data": {
                        "ticket_id": f"TCK-{run_id[-6:].upper()}",
                        "question": "Where is the enterprise shipment and can we expedite it?",
                    },
                    "output_data": {
                        "answer": "The shipment is still on schedule for tomorrow morning and has already been escalated for enterprise handling."
                    },
                    "metadata": {
                        "scenario": args.scenario,
                        "run_id": run_id,
                        "workflow": "support-escalation",
                        "tool_decision_count": 2,
                        "tool_strategy": "sequential",
                        "traigent_active_config": {
                            "model": "gpt-4o-mini",
                            "tool_choice": "auto",
                            "tools": tool_names,
                        },
                    },
                    "session": {
                        "id": session_id,
                        "user_id": user_id,
                        "environment": "support",
                        "release": "langfuse-e2e",
                        "tags": tags,
                        "metadata": {
                            "scenario": args.scenario,
                            "run_id": run_id,
                            "channel": "enterprise-support",
                        },
                    },
                    "observations": [
                        {
                            "id": f"{trace_id}:root",
                            "type": "span",
                            "name": "agent-orchestrator",
                            "status": "completed",
                            "latency_ms": 290,
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "metadata": {"stage": "orchestration"},
                            "children": [
                                {
                                    "id": f"{trace_id}:plan",
                                    "type": "generation",
                                    "name": "plan-next-actions",
                                    "status": "completed",
                                    "latency_ms": 91,
                                    "input_tokens": 22,
                                    "output_tokens": 18,
                                    "model_name": "gpt-4o-mini",
                                    "input_data": {
                                        "question": "Where is the enterprise shipment and can we expedite it?"
                                    },
                                    "output_data": {
                                        "plan": [
                                            "Look up the customer tier and shipment id.",
                                            "Fetch the latest ETA from the shipping provider.",
                                            "Compose the final answer for the operator.",
                                        ]
                                    },
                                    "metadata": {"phase": "planning"},
                                },
                                {
                                    "id": f"{trace_id}:crm",
                                    "type": "tool_call",
                                    "name": "lookup-account-context",
                                    "status": "completed",
                                    "latency_ms": 63,
                                    "input_tokens": 6,
                                    "output_tokens": 12,
                                    "tool_name": "crm_lookup",
                                    "input_data": {"customer_id": "acct-enterprise-42"},
                                    "output_data": {
                                        "account_tier": "enterprise",
                                        "shipment_id": "ship-4458",
                                    },
                                    "metadata": {"provider": "salesforce"},
                                    "children": [
                                        {
                                            "id": f"{trace_id}:crm:event",
                                            "type": "event",
                                            "name": "crm-lookup.result",
                                            "status": "completed",
                                            "latency_ms": 0,
                                            "output_data": {
                                                "matched_records": 1,
                                                "escalation_policy": "priority-routing",
                                            },
                                            "metadata": {"matched_records": 1},
                                        }
                                    ],
                                },
                                {
                                    "id": f"{trace_id}:shipping",
                                    "type": "tool_call",
                                    "name": "fetch-shipment-timeline",
                                    "status": "completed",
                                    "latency_ms": 77,
                                    "input_tokens": 8,
                                    "output_tokens": 10,
                                    "tool_name": "shipping_eta_api",
                                    "input_data": {"shipment_id": "ship-4458"},
                                    "output_data": {
                                        "eta": "2026-04-05T09:30:00Z",
                                        "latest_checkpoint": "Arrived at regional hub",
                                    },
                                    "metadata": {"provider": "carrier-api"},
                                    "children": [
                                        {
                                            "id": f"{trace_id}:shipping:event",
                                            "type": "event",
                                            "name": "shipping-eta.result",
                                            "status": "completed",
                                            "latency_ms": 0,
                                            "output_data": {
                                                "checkpoint_count": 4,
                                                "expedite_available": False,
                                            },
                                            "metadata": {"checkpoint_count": 4},
                                        }
                                    ],
                                },
                                {
                                    "id": f"{trace_id}:respond",
                                    "type": "generation",
                                    "name": "compose-final-answer",
                                    "status": "completed",
                                    "latency_ms": 84,
                                    "input_tokens": 16,
                                    "output_tokens": 28,
                                    "model_name": "gpt-4o-mini",
                                    "output_data": {
                                        "answer": "The shipment is still on schedule for tomorrow morning and has already been escalated for enterprise handling."
                                    },
                                    "metadata": {"phase": "response"},
                                },
                            ],
                        }
                    ],
                }
            ]
        },
    )
    trace_payload = api_get_json(
        args.backend_url,
        api_key,
        f"/api/v1beta/observability/traces/{quote(trace_id, safe='')}",
    )
    observations_payload = api_get_json(
        args.backend_url,
        api_key,
        f"/api/v1beta/observability/traces/{quote(trace_id, safe='')}/observations",
    )
    session_payload = api_get_json(
        args.backend_url,
        api_key,
        f"/api/v1beta/observability/sessions/{quote(session_id, safe='')}",
    )
    user_payload = api_get_json(
        args.backend_url,
        api_key,
        f"/api/v1beta/observability/users/{quote(user_id, safe='')}",
    )

    write_json(output_dir / "tool-calling-multistep-ingest.json", ingest_payload)
    write_json(output_dir / "tool-calling-multistep-trace.json", trace_payload)
    write_json(output_dir / "tool-calling-multistep-observations.json", observations_payload)
    write_json(output_dir / "tool-calling-multistep-session.json", session_payload)
    write_json(output_dir / "tool-calling-multistep-user.json", user_payload)

    return {
        "trace_id": trace_id,
        "trace_name": trace_name,
        "session_id": session_id,
        "user_id": user_id,
        "tags": tags,
        "tool_names": tool_names,
        "observation_count": 7,
        "custom_trace_id": custom_trace_id,
    }


def seed_prompt_fixture(
    *,
    args: argparse.Namespace,
    api_key: str,
    run_id: str,
    output_dir: Path,
    run_compare: bool = False,
) -> dict[str, Any]:
    prompt_name = f"langfuse.e2e.{run_id.replace(':', '-').replace('_', '-').replace('/', '-')}"
    encoded_prompt_name = quote(prompt_name, safe="")

    create_payload = api_post_json(
        args.backend_url,
        api_key,
        "/api/v1beta/prompts",
        {
            "name": prompt_name,
            "prompt_type": "text",
            "prompt_text": "Hello {{name}} from version one",
            "description": f"Langfuse E2E prompt fixture for {run_id}",
            "labels": ["production"],
            "tags": ["langfuse-core", run_id],
            "config": {"model": "mock-model", "provider": "mock"},
            "commit_message": "Seed v1",
        },
    )
    version_payload = api_post_json(
        args.backend_url,
        api_key,
        f"/api/v1beta/prompts/{encoded_prompt_name}/versions",
        {
            "prompt_text": "Hi {{name}} from version two",
            "config": {"model": "mock-model", "provider": "mock"},
            "commit_message": "Seed v2",
            "labels": ["candidate"],
        },
    )
    labels_payload = api_patch_json(
        args.backend_url,
        api_key,
        f"/api/v1beta/prompts/{encoded_prompt_name}/labels",
        {"labels": {"latest": 2, "production": 2}},
    )
    prompt_detail = api_get_json(args.backend_url, api_key, f"/api/v1beta/prompts/{encoded_prompt_name}")

    analytics_payload = api_get_json(
        args.backend_url,
        api_key,
        f"/api/v1beta/prompts/{encoded_prompt_name}/analytics",
    )

    compare_trace_ids: list[str] = []
    compare_run_payloads: list[Any] = []
    if run_compare:
        for version in (1, 2):
            run_payload = api_post_json(
                args.backend_url,
                api_key,
                f"/api/v1beta/prompts/{encoded_prompt_name}/playground/run",
                {
                    "version": version,
                    "variables": {"name": "Langfuse"},
                    "provider": "mock",
                    "model": "mock-model",
                    "dry_run": False,
                },
            )
            compare_run_payloads.append(run_payload)
            trace_id = extract_payload(run_payload).get("trace_id")
            if trace_id:
                compare_trace_ids.append(str(trace_id))

        analytics_payload = api_get_json(
            args.backend_url,
            api_key,
            f"/api/v1beta/prompts/{encoded_prompt_name}/analytics",
        )

    write_json(output_dir / "prompt-create.json", create_payload)
    write_json(output_dir / "prompt-version.json", version_payload)
    write_json(output_dir / "prompt-labels.json", labels_payload)
    write_json(output_dir / "prompt-detail.json", prompt_detail)
    write_json(output_dir / "prompt-analytics.json", analytics_payload)
    if compare_run_payloads:
        write_json(output_dir / "prompt-playground-runs.json", compare_run_payloads)

    return {
        "prompt_name": prompt_name,
        "prompt_version_ids": [
            extract_payload(create_payload).get("versions", [{}])[0].get("id"),
            extract_payload(version_payload).get("versions", [{}])[0].get("id"),
        ],
        "compare_trace_ids": compare_trace_ids,
        "prompt_labels": extract_payload(labels_payload).get("labels", {}),
    }


def _first_value(payload: Any, *keys: str) -> str | None:
    data = extract_payload(payload)
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    return None


def seed_access_control_fixture(
    *,
    args: argparse.Namespace,
    api_key: str,
    output_dir: Path,
) -> dict[str, Any]:
    agent_types_payload = api_get_json(args.backend_url, api_key, "/api/v1/agents/agent-types")
    write_json(output_dir / "access-control-agent-types.json", agent_types_payload)

    agent_types = extract_items(agent_types_payload)
    agent_type_id = None
    for item in agent_types:
        if isinstance(item, dict):
            candidate = item.get("id") or item.get("value") or item.get("label")
            if candidate:
                agent_type_id = str(candidate)
                break
    if not agent_type_id:
        raise RuntimeError("Unable to resolve an agent type for access-control-isolation")

    agent_name = f"Access Control Agent {args.run_id}"
    agent_payload = api_post_json(
        args.backend_url,
        api_key,
        "/api/v1/agents",
        {
            "name": agent_name,
            "agent_type_id": agent_type_id,
            "prompt_template": "You are a deterministic access-control test agent.",
        },
    )
    write_json(output_dir / "access-control-agent.json", agent_payload)
    agent_id = _first_value(agent_payload, "agent_id", "id")
    if not agent_id:
        raise RuntimeError("Unable to resolve created agent_id for access-control-isolation")

    dataset_name = f"Access Control Dataset {args.run_id}"
    dataset_payload = api_post_json(
        args.backend_url,
        api_key,
        "/api/v1/datasets",
        {
            "name": dataset_name,
            "label": dataset_name,
            "type": "input-output",
            "use_case": "question-answering",
            "agent_type_id": agent_type_id,
            "description": "Dataset owned by the primary user for access control isolation.",
        },
    )
    write_json(output_dir / "access-control-dataset.json", dataset_payload)
    dataset_id = _first_value(dataset_payload, "dataset_id", "benchmark_id", "id")
    if not dataset_id:
        raise RuntimeError("Unable to resolve created dataset_id for access-control-isolation")

    dataset_example_payload = api_post_json(
        args.backend_url,
        api_key,
        f"/api/v1/datasets/{dataset_id}/examples",
        {
            "input_text": f"What is protected in run {args.run_id}?",
            "expected_output": "Owned datasets and experiments stay private.",
            "metadata": {"scenario": args.scenario, "run_id": args.run_id},
        },
    )
    write_json(output_dir / "access-control-dataset-example.json", dataset_example_payload)

    experiment_name = f"Access Control Experiment {args.run_id}"
    experiment_payload = api_post_json(
        args.backend_url,
        api_key,
        "/api/v1/experiments",
        {
            "name": experiment_name,
            "description": "Experiment owned by the primary user for access control isolation.",
            "status": "NOT_STARTED",
            "dataset_id": dataset_id,
            "agent_id": agent_id,
            "measures": [{"measure_id": "faithfulness"}],
            "configurations": {"model": "gpt-4o"},
        },
    )
    write_json(output_dir / "access-control-experiment.json", experiment_payload)
    experiment_id = _first_value(experiment_payload, "experiment_id", "id")
    if not experiment_id:
        raise RuntimeError("Unable to resolve created experiment_id for access-control-isolation")

    return {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "agent_type_id": agent_type_id,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
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
    ensure_bearer_token(args.backend_url)

    if args.scenario == "access-control-isolation":
        access_fixture = seed_access_control_fixture(
            args=args,
            api_key=api_key,
            output_dir=output_path.parent,
        )

        dataset_id = access_fixture["dataset_id"]
        dataset_name = access_fixture["dataset_name"]
        experiment_id = access_fixture["experiment_id"]
        agent_id = access_fixture["agent_id"]
        agent_name = access_fixture["agent_name"]
        scenario_id = f"T-E2E-{args.scenario}"

        frontend_urls = build_frontend_urls(args.frontend_url, experiment_id, None, dataset_id)
        backend_urls = build_backend_urls(args.backend_url, experiment_id, None, dataset_id)

        manifest = {
            "scenario_id": scenario_id,
            "scenario_type": args.scenario,
            "scenario": "langfuse-core",
            "run_id": args.run_id,
            "mode": args.mode,
            "scale": args.scale,
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "experiment_id": experiment_id,
            "experiment_run_id": None,
            "configuration_run_ids": [],
            "trace_ids": [],
            "baseline_trace_ids": [],
            "post_trace_ids": [],
            "score_ids": [],
            "feedback_ids": [],
            "session_ids": [],
            "user_ids": [],
            "tag_values": [],
            "prompt_version_ids": [],
            "best_config": None,
            "best_metrics": None,
            "frontend_urls": frontend_urls,
            "backend_urls": backend_urls,
            "assertions": {
                "protected_dataset_id": dataset_id,
                "protected_experiment_id": experiment_id,
                "protected_agent_id": agent_id,
            },
        }
        write_json(output_path, manifest)
        return 0

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

    experiment_optional_scenarios = {
        "tool-calling-multistep-trace",
        "trace-session-user-browse",
        "trace-feedback-collaboration",
        "prompt-version-lineage",
        "playground-run-and-compare",
        "trace-to-prompt-lineage",
    }
    requires_experiment_linkage = args.scenario not in experiment_optional_scenarios

    if requires_experiment_linkage and (not experiment_id or not experiment_run_id):
        raise RuntimeError("Unable to resolve experiment_id / experiment_run_id from guided summaries")

    experiment_payload: dict[str, Any] | None = None
    run_results_payload: dict[str, Any] | None = None
    runs_payload: dict[str, Any] | None = None
    if experiment_id:
        experiment_payload = api_get_json(
            args.backend_url,
            api_key,
            f"/api/v1/experiments/{experiment_id}",
        )
        write_json(output_path.parent / "backend-experiment.json", experiment_payload)
    if experiment_id and experiment_run_id:
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
        write_json(output_path.parent / "backend-run-results.json", run_results_payload)
        write_json(output_path.parent / "backend-runs.json", runs_payload)

    experiment = extract_payload(experiment_payload) if experiment_payload is not None else {}
    run_results = extract_payload(run_results_payload) if run_results_payload is not None else {}
    configuration_run_ids = (
        collect_configuration_run_ids(run_results_payload)
        if run_results_payload is not None
        else []
    )

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
    prompt_version_ids: list[str] = []
    prompt_fixture: dict[str, Any] | None = None
    observability_fixture: dict[str, Any] | None = None
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
    elif args.scenario == "trace-session-user-browse":
        observability_fixture = seed_observability_fixture(
            args=args,
            api_key=api_key,
            run_id=args.run_id,
            output_dir=output_path.parent,
        )
        assertions = {
            "browse_trace_id": observability_fixture["trace_id"],
            "browse_session_id": observability_fixture["session_id"],
            "browse_user_id": observability_fixture["user_id"],
        }
    elif args.scenario == "trace-feedback-collaboration":
        observability_fixture = seed_observability_fixture(
            args=args,
            api_key=api_key,
            run_id=args.run_id,
            output_dir=output_path.parent,
        )
        assertions = {
            "collaboration_trace_id": observability_fixture["trace_id"],
            "collaboration_session_id": observability_fixture["session_id"],
            "collaboration_user_id": observability_fixture["user_id"],
        }
    elif args.scenario == "tool-calling-multistep-trace":
        observability_fixture = seed_tool_calling_multistep_trace_fixture(
            args=args,
            api_key=api_key,
            run_id=args.run_id,
            output_dir=output_path.parent,
        )
        assertions = {
            "multistep_trace_id": observability_fixture["trace_id"],
            "multistep_trace_name": observability_fixture["trace_name"],
            "multistep_session_id": observability_fixture["session_id"],
            "multistep_user_id": observability_fixture["user_id"],
            "multistep_tool_names": observability_fixture["tool_names"],
            "multistep_observation_count": observability_fixture["observation_count"],
            "multistep_custom_trace_id": observability_fixture["custom_trace_id"],
        }
    elif args.scenario == "prompt-version-lineage":
        prompt_fixture = seed_prompt_fixture(
            args=args,
            api_key=api_key,
            run_id=args.run_id,
            output_dir=output_path.parent,
            run_compare=False,
        )
        prompt_version_ids = [value for value in prompt_fixture.get("prompt_version_ids", []) if value]
        assertions = {
            "prompt_name": prompt_fixture["prompt_name"],
            "prompt_labels": prompt_fixture["prompt_labels"],
        }
    elif args.scenario == "playground-run-and-compare":
        prompt_fixture = seed_prompt_fixture(
            args=args,
            api_key=api_key,
            run_id=args.run_id,
            output_dir=output_path.parent,
            run_compare=True,
        )
        prompt_version_ids = [value for value in prompt_fixture.get("prompt_version_ids", []) if value]
        assertions = {
            "prompt_name": prompt_fixture["prompt_name"],
            "compare_trace_ids": prompt_fixture["compare_trace_ids"],
        }
    elif args.scenario == "trace-to-prompt-lineage":
        prompt_fixture = seed_prompt_fixture(
            args=args,
            api_key=api_key,
            run_id=args.run_id,
            output_dir=output_path.parent,
            run_compare=True,
        )
        prompt_version_ids = [value for value in prompt_fixture.get("prompt_version_ids", []) if value]
        compare_trace_ids = [value for value in prompt_fixture.get("compare_trace_ids", []) if value]
        trace_for_prompt = compare_trace_ids[-1] if compare_trace_ids else None
        if trace_for_prompt:
            observations_payload = api_get_json(
                args.backend_url,
                api_key,
                f"/api/v1beta/observability/traces/{quote(trace_for_prompt, safe='')}/observations",
            )
            write_json(output_path.parent / "trace-to-prompt-observations.json", observations_payload)
        assertions = {
            "prompt_name": prompt_fixture["prompt_name"],
            "prompt_trace_id": trace_for_prompt,
        }
    elif args.scenario == "access-control-isolation":
        assertions = {
            "protected_dataset_id": dataset_id,
            "protected_experiment_id": experiment_id,
        }
    else:
        assertions = {"guided_flow": True}

    frontend_urls = build_frontend_urls(args.frontend_url, experiment_id, experiment_run_id, dataset_id)
    backend_urls = build_backend_urls(args.backend_url, experiment_id, experiment_run_id, dataset_id)

    if observability_fixture:
        trace_id = observability_fixture["trace_id"]
        session_id = observability_fixture["session_id"]
        user_id = observability_fixture["user_id"]
        trace_ids.append(trace_id)
        session_ids.append(session_id)
        frontend_urls.update(
            {
                "observability": f"{args.frontend_url.rstrip('/')}/observability",
                "trace": f"{args.frontend_url.rstrip('/')}/observability/traces/{quote(trace_id, safe='')}",
                "session": f"{args.frontend_url.rstrip('/')}/observability/sessions/{quote(session_id, safe='')}",
                "user": f"{args.frontend_url.rstrip('/')}/observability/users/{quote(user_id, safe='')}",
            }
        )
        backend_urls.update(
            {
                "trace": f"{args.backend_url.rstrip('/')}/api/v1beta/observability/traces/{quote(trace_id, safe='')}",
                "session": f"{args.backend_url.rstrip('/')}/api/v1beta/observability/sessions/{quote(session_id, safe='')}",
                "user": f"{args.backend_url.rstrip('/')}/api/v1beta/observability/users/{quote(user_id, safe='')}",
            }
        )
        tag_values = sorted(set(tag_values + list(observability_fixture.get("tags", []))))

    if prompt_fixture:
        prompt_name = prompt_fixture["prompt_name"]
        encoded_prompt_name = quote(prompt_name, safe="")
        compare_trace_ids = [value for value in prompt_fixture.get("compare_trace_ids", []) if value]
        trace_ids.extend(compare_trace_ids)
        prompt_version_ids = [value for value in prompt_fixture.get("prompt_version_ids", []) if value]
        frontend_urls.update(
            {
                "prompts": f"{args.frontend_url.rstrip('/')}/prompts",
                "prompt": f"{args.frontend_url.rstrip('/')}/prompts/{encoded_prompt_name}",
                "playground": f"{args.frontend_url.rstrip('/')}/playground?prompt={encoded_prompt_name}&playground=1&lhs=1&rhs=2",
            }
        )
        backend_urls.update(
            {
                "prompt": f"{args.backend_url.rstrip('/')}/api/v1beta/prompts/{encoded_prompt_name}",
                "prompt_analytics": f"{args.backend_url.rstrip('/')}/api/v1beta/prompts/{encoded_prompt_name}/analytics",
            }
        )
        if compare_trace_ids:
            assertions.setdefault("prompt_trace_id", compare_trace_ids[-1])
            frontend_urls["prompt_trace"] = (
                f"{args.frontend_url.rstrip('/')}/observability/traces/{quote(compare_trace_ids[-1], safe='')}"
            )
            backend_urls["prompt_trace"] = (
                f"{args.backend_url.rstrip('/')}/api/v1beta/observability/traces/{quote(compare_trace_ids[-1], safe='')}"
            )

    user_ids = {"guided-demo-user"}
    if observability_fixture:
        user_ids.add(observability_fixture["user_id"])

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
        "user_ids": sorted(user_ids),
        "tag_values": tag_values,
        "prompt_version_ids": prompt_version_ids,
        "best_config": post_summary.get("best_config") or optimize_summary.get("best_config"),
        "best_metrics": post_summary.get("best_metrics") or optimize_summary.get("best_metrics"),
        "frontend_urls": frontend_urls,
        "backend_urls": backend_urls,
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
