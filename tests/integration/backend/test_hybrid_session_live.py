"""Live backend smoke test for the interactive optimization session API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

import pytest

from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.models import SessionCreationRequest


def _resolve_backend_url() -> str | None:
    return os.getenv("TRAIGENT_API_URL") or os.getenv("TRAIGENT_BACKEND_URL")


def _score_config(config: dict[str, object]) -> dict[str, float | int]:
    model = str(config["model"])
    temperature = float(config["temperature"])

    accuracy = (
        0.9 - abs(temperature - 0.4) * 0.05
        if model == "gpt-4o"
        else 0.82 - abs(temperature - 0.2) * 0.04
    )
    cost = 0.35 if model == "gpt-4o" else 0.08

    return {"accuracy": accuracy, "cost": cost}


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("TRAIGENT_HYBRID_LIVE"),
    reason=(
        "Set TRAIGENT_HYBRID_LIVE=1 plus TRAIGENT_API_KEY and "
        "TRAIGENT_API_URL (preferred) or TRAIGENT_BACKEND_URL to run live hybrid session tests"
    ),
)
async def test_live_hybrid_session_round_trip() -> None:
    """Exercise create/next/result/finalize against the real backend."""
    api_key = os.getenv("TRAIGENT_API_KEY")
    backend_url = _resolve_backend_url()
    evaluator_id = os.getenv("TRAIGENT_HYBRID_LIVE_EVALUATOR_ID")

    if not api_key:
        pytest.skip("TRAIGENT_API_KEY must be set for the live hybrid session test")

    if not backend_url:
        pytest.skip(
            "Set TRAIGENT_API_URL (preferred) or TRAIGENT_BACKEND_URL for the live hybrid session test"
        )

    if not evaluator_id:
        pytest.skip(
            "TRAIGENT_HYBRID_LIVE_EVALUATOR_ID must identify a registered evaluator version"
        )

    retain_receipt = os.getenv("TRAIGENT_HYBRID_LIVE_RETAIN_RECEIPT") == "1"
    receipt_path_value = os.getenv("TRAIGENT_HYBRID_LIVE_RECEIPT")
    if retain_receipt and not receipt_path_value:
        pytest.skip(
            "TRAIGENT_HYBRID_LIVE_RECEIPT is required when retaining the readiness witness"
        )
    receipt_path = Path(receipt_path_value) if receipt_path_value else None

    session_id: str | None = None
    receipt_written = False

    async with TraigentCloudClient(
        api_key=api_key,
        base_url=backend_url,
        enable_fallback=False,
    ) as client:
        try:
            created = await client.create_optimization_session(
                SessionCreationRequest(
                    function_name="python_hybrid_live_smoke",
                    agent_key="python-hybrid-live-readiness-agent",
                    dataset_id="python-hybrid-live-readiness-dataset",
                    configuration_space={
                        "model": {
                            "type": "categorical",
                            "choices": ["gpt-4o-mini", "gpt-4o"],
                        },
                        "temperature": {
                            "type": "float",
                            "low": 0.0,
                            "high": 1.0,
                            "step": 0.2,
                        },
                    },
                    objectives=["accuracy", "cost"],
                    dataset_metadata={"size": 4, "suite": "python-hybrid-live-smoke"},
                    max_trials=4,
                    optimization_strategy={"algorithm": "optuna"},
                    evaluator_id=evaluator_id,
                )
            )

            session_id = created.session_id
            assert session_id
            assert created.status.value in {"created", "active"}

            next_trial = await client.get_next_trial(session_id)
            assert next_trial.should_continue is True
            assert next_trial.suggestion is not None

            suggestion = next_trial.suggestion
            assert suggestion.session_id == session_id
            assert suggestion.trial_number == 1
            assert suggestion.dataset_subset.indices
            assert all(0 <= index < 4 for index in suggestion.dataset_subset.indices)

            metrics = {**_score_config(suggestion.config), "total_examples": 4}

            await client.submit_trial_result(
                session_id=session_id,
                trial_id=suggestion.trial_id,
                metrics=metrics,
                duration=0.01,
                status="completed",
                metadata={"suite": "python-hybrid-live-smoke"},
            )

            finalized = await client.finalize_optimization(session_id)
            assert finalized.session_id == session_id
            assert finalized.total_trials >= 1
            assert finalized.successful_trials >= 1
            assert isinstance(finalized.best_metrics, dict)
            assert "accuracy" in finalized.best_metrics
            assert finalized.stop_reason in {
                "max_trials_reached",
                "search_complete",
                "finalized",
            }
            if retain_receipt and receipt_path is not None:
                metadata = created.metadata if isinstance(created.metadata, dict) else {}
                project_id = metadata.get("owner_project_id") or metadata.get("project_id")
                experiment_id = metadata.get("experiment_id")
                experiment_run_id = metadata.get("experiment_run_id")
                assert isinstance(project_id, str) and project_id
                assert isinstance(experiment_id, str) and experiment_id
                assert isinstance(experiment_run_id, str) and experiment_run_id
                for key, value in (
                    ("owner_project_id/project_id", project_id),
                    ("experiment_id", experiment_id),
                    ("experiment_run_id", experiment_run_id),
                ):
                    assert isinstance(value, str) and value, (
                        f"retained readiness witness must return {key} in session metadata"
                    )

                await client._ensure_session()
                assert client._aio_session is not None
                readiness_url = (
                    f"{backend_url.rstrip('/')}/api/v1beta/projects/"
                    f"{quote(project_id, safe='')}/agent-readiness"
                )
                matching_items: list[dict[str, object]] = []
                max_pages = 10
                for page in range(1, max_pages + 1):
                    async with client._aio_session.get(
                        readiness_url,
                        headers=await client._get_headers(),
                        params={"page": page, "per_page": 100},
                    ) as response:
                        body = await response.text()
                        assert response.status == 200, (
                            "readiness portfolio lookup failed: "
                            f"HTTP {response.status}: {body[:200]}"
                        )
                        payload = json.loads(body)

                    assert isinstance(payload, dict)
                    page_items = payload.get("items")
                    assert isinstance(page_items, list)
                    matching_items.extend(
                        item
                        for item in page_items
                        if isinstance(item, dict)
                        and item.get("anchor_run_id") == experiment_run_id
                    )
                    pagination = payload.get("pagination")
                    assert isinstance(pagination, dict)
                    if pagination.get("has_next") is not True:
                        break
                else:
                    pytest.fail(
                        f"readiness portfolio exceeded bounded pagination ({max_pages} pages)"
                    )

                assert len(matching_items) == 1, (
                    "retained readiness witness must find exactly one portfolio item "
                    f"for anchor run {experiment_run_id!r}; found {len(matching_items)}"
                )
                readiness_item = matching_items[0]
                assert readiness_item.get("anchor_experiment_id") == experiment_id
                readiness_agent = readiness_item.get("agent")
                assert isinstance(readiness_agent, dict)
                agent_id = readiness_agent.get("id")
                assert isinstance(agent_id, str) and agent_id, (
                    "retained readiness witness must return the canonical agent id "
                    "from the project-scoped readiness portfolio"
                )
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                receipt_path.write_text(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "project_id": project_id,
                            "agent_id": agent_id,
                            "experiment_id": experiment_id,
                            "experiment_run_id": experiment_run_id,
                            "declared_agent_id": "python-hybrid-live-readiness-agent",
                            "declared_dataset_id": "python-hybrid-live-readiness-dataset",
                            "expected": {
                                "dataset_check": "SUPPORTED",
                                "evaluator_version_snapshot_recorded": "SUPPORTED",
                                "evaluated_example_count": 4,
                                "unsupported_checks": "UNKNOWN",
                            },
                        },
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                receipt_written = True
        finally:
            if session_id and not receipt_written:
                try:
                    await client.delete_session(session_id, cascade=True)
                except Exception:
                    pass
