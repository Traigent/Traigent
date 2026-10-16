"""Exercise cloud entry points that a default ``optimize()`` run does not reach.

Each exerciser calls a public/semi-public SDK function with canaries in every
customer-controlled argument, under the same egress capture.  Failures of an
exerciser are recorded, never swallowed silently: the report lists them so
uncovered routes are visible rather than assumed clean.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .canaries import CANARIES

BACKEND_URL = "https://backend.example.test"


def _client() -> Any:
    from traigent.cloud.backend_client import (
        BackendClientConfig,
        BackendIntegratedClient,
    )

    return BackendIntegratedClient(
        api_key="uk_" + ("a" * 43),
        backend_config=BackendClientConfig(backend_base_url=BACKEND_URL),
    )


async def _trial_result_failed() -> None:
    c = CANARIES
    client = _client()
    await client._submit_trial_result_via_session(
        "sess-canary-direct",
        "trial-canary-direct",
        {c["knob_name"]: c["choice_value"]},
        {c["metric_name"]: 0.5},
        "failed",
        error_message=f"boom {c['exception_message']}",
        metadata={c["metadata_key"]: c["metadata_value"]},
    )


async def _config_run_updates() -> None:
    c = CANARIES
    client = _client()
    await client._update_config_run_status(
        "cfg-canary-direct", "failed", error_message=f"boom {c['exception_message']}"
    )
    await client._update_config_run_measures(
        "cfg-canary-direct", {c["metric_name"]: 0.5}, execution_time=0.1
    )


async def _intermediate_report() -> None:
    c = CANARIES
    client = _client()
    await client._report_intermediate_progress(
        {
            "session_id": "sess-canary-direct",
            "trial_id": "trial-canary-direct",
            "running_score": 0.5,
            "examples_attempted": 1,
            "objective_name": c["metric_name"],
        }
    )


async def _hybrid_session_and_optimizer() -> None:
    from traigent.cloud.optimizer_client import OptimizerDirectClient

    c = CANARIES
    client = _client()
    session_id, token, endpoint = await client.create_hybrid_session(
        problem_statement=c["agent_name"],
        search_space={c["knob_name"]: [c["choice_value"], c["prompt_template"]]},
        optimization_config={"objectives": [c["metric_name"]], "max_trials": 2},
        metadata={c["metadata_key"]: c["metadata_value"], "title": c["run_title"]},
    )
    await client.get_hybrid_session_status(session_id)
    async with OptimizerDirectClient(endpoint, token) as direct:
        await direct.get_next_configuration(session_id)
        await direct.get_session_status(session_id)
        await direct.submit_metrics(
            session_id,
            "trial-canary-direct",
            {c["metric_name"]: 0.5},
            0.1,
            metadata={c["metadata_key"]: c["metadata_value"]},
        )
        direct.set_batch_size(1)
        await direct.submit_metrics(
            session_id,
            "trial-canary-direct-2",
            {c["metric_name"]: 0.6},
            0.1,
            metadata={c["metadata_key"]: c["metadata_value"]},
        )
        await direct.flush()
    await client.finalize_hybrid_session(session_id)


async def _direct_serializer() -> None:
    """``client.py`` direct serializer sends ``outputs_sample``/``error_message``."""

    import aiohttp

    from traigent.cloud.client import TraigentCloudClient
    from traigent.cloud.models import TrialResultSubmission, TrialStatus

    c = CANARIES
    cloud = TraigentCloudClient.__new__(TraigentCloudClient)
    wire = cloud._serialize_trial_result(
        TrialResultSubmission(
            session_id="sess-canary-direct",
            trial_id="trial-canary-direct",
            metrics={c["metric_name"]: 0.5},
            duration=0.1,
            status=TrialStatus.FAILED,
            outputs_sample=[c["model_output"]],
            error_message=f"boom {c['exception_message']}",
            metadata={c["metadata_key"]: c["metadata_value"]},
        )
    )
    # Send the serialized body through the real aiohttp stack (patched below it).
    async with (
        aiohttp.ClientSession() as session,
        session.post(
            f"{BACKEND_URL}/api/v1/sessions/sess-canary-direct/results", json=wire
        ) as resp,
    ):
        await resp.read()


def _best_config_sync() -> None:
    c = CANARIES
    client = _client()
    client.publish_best_config_sync(
        {c["knob_name"]: c["choice_value"], "metric": c["metric_name"]},
        environment=c["agent_name"],
    )
    client.fetch_best_config_sync(
        "best-canary-direct", environment=c["agent_name"], function_ref=c["agent_name"]
    )


async def _example_set_upload() -> None:
    from traigent.cloud.dataset_converter import DatasetConverter
    from traigent.evaluators.base import Dataset, EvaluationExample

    c = CANARIES
    dataset = Dataset(
        name=c["evaluation_set"],
        examples=[
            EvaluationExample(
                input_data={"question": c["input_text"]},
                expected_output=c["output_text"],
                metadata={
                    "example_id": c["example_id"],
                    c["metadata_key"]: c["metadata_value"],
                },
            )
        ],
    )
    async with DatasetConverter(BACKEND_URL) as converter:
        await converter.upload_sdk_dataset_to_backend(
            dataset, agent_id=c["agent_name"], example_set_name=c["dataset_file_name"]
        )


EXERCISERS: dict[str, Callable[[], Awaitable[None] | None]] = {
    "trial_result_failed": _trial_result_failed,
    "config_run_updates": _config_run_updates,
    "intermediate_report": _intermediate_report,
    "hybrid_session_and_optimizer": _hybrid_session_and_optimizer,
    "direct_serializer": _direct_serializer,
    "best_config_sync": _best_config_sync,
    "example_set_upload": _example_set_upload,
}


def run_exercisers() -> dict[str, str | None]:
    """Run every exerciser; return ``name -> error`` (None when it ran clean)."""

    outcomes: dict[str, str | None] = {}
    for name, fn in EXERCISERS.items():
        try:
            out = fn()
            if asyncio.iscoroutine(out):
                asyncio.run(asyncio.wait_for(out, timeout=60))
            outcomes[name] = None
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            outcomes[name] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return outcomes
