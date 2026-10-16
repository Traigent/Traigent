"""Exercise cloud entry points that a default ``optimize()`` run does not reach.

Each exerciser calls a public/semi-public SDK function with canaries in every
customer-controlled argument, under the same egress capture.  Failures of an
exerciser are recorded, never swallowed silently: the report lists them so
uncovered routes are visible rather than assumed clean.
"""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Awaitable, Callable
from typing import Any

from .canaries import CANARIES
from .harness import EgressCapture

BACKEND_URL = "https://backend.example.test"

#: Capture the running exerciser reports faults to (set by ``run_exercisers``).
_ACTIVE: EgressCapture | None = None


def _inject(
    method: str, path_regex: str, outcomes: list[int | str], *, expects_retry: bool
) -> None:
    assert _ACTIVE is not None, "exercisers run under run_exercisers()"
    _ACTIVE.backend.inject(method, path_regex, outcomes, expects_retry=expects_retry)


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


def build_offline_session(storage_path: str) -> str:
    """A finished local session whose every field carries a canary."""

    from traigent.cloud.models import DECLARED_DATASET_IDENTITY_METADATA_KEY
    from traigent.storage.local_storage import LocalStorageManager

    c = CANARIES
    storage = LocalStorageManager(storage_path)
    sid = storage.create_session(
        function_name=c["function_name"],
        optimization_config={
            "search_space": {c["knob_name"]: [c["choice_value"]]},
            "configuration_space": {c["knob_name"]: [c["choice_value"]]},
            "default_config": {c["knob_name"]: c["default_config_value"]},
            "objectives": ["accuracy"],
            "evaluation_set": c["evaluation_set"],
            "constraints": [c["constraint_name"]],
            "max_trials": 2,
            "agent_name": c["agent_name"],
            "run_title": c["run_title"],
            "evaluator_id": c["evaluator_id"],
            "dataset_id": c["dataset_identity"],
        },
        metadata={
            c["metadata_key"]: c["metadata_value"],
            "offline_replay": c["offline_replay"],
            "dataset_name": c["evaluation_set"],
            "portal_name": c["agent_name"],
            "function_display_name": c["function_name"],
            "agent_key": c["agent_name"],
            DECLARED_DATASET_IDENTITY_METADATA_KEY: c["dataset_identity"],
        },
    )
    for i in range(2):
        storage.add_trial_result(
            sid,
            config={c["knob_name"]: c["choice_value"], "prompt": c["prompt_template"]},
            score=0.5 + i / 10,
            metadata={
                c["metadata_key"]: c["metadata_value"],
                "outputs_sample": [c["model_output"]],
                "offline_replay": c["offline_replay"],
            },
            error=f"boom {c['exception_message']}" if i == 0 else None,
        )
    storage.finalize_session(sid)
    return sid


def sync_offline_sessions(storage_path: str, *, only: str | None = None) -> None:
    """Replay local sessions through ``SyncManager`` (``traigent sync``)."""

    from traigent.cloud.sync_manager import SyncManager
    from traigent.config.types import TraigentConfig

    cfg = TraigentConfig(local_storage_path=storage_path)
    manager = SyncManager(cfg, api_key="uk_" + ("a" * 43))
    if only:
        manager.sync_session_to_cloud(only, force=True)
    else:
        manager.sync_all_sessions()


def _sync_manager_replay() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sid = build_offline_session(tmp)
        sync_offline_sessions(tmp, only=sid)


def _sync_manager_replay_with_faults() -> None:
    """Replay while the backend fails the create, then the first results."""

    _inject("POST", r"/sessions$", [503], expects_retry=False)
    with tempfile.TemporaryDirectory() as tmp:
        sid = build_offline_session(tmp)
        try:
            sync_offline_sessions(tmp, only=sid)  # create fails, sync aborts
        except Exception:  # noqa: BLE001 - the failed body is already captured
            pass
        # SyncManager does not resend a failed results POST (it resumes on the
        # next sync); the failed bodies are still on the wire.
        _inject("POST", r"/sessions/[^/]+/results$", [429, 503], expects_retry=False)
        try:
            sync_offline_sessions(tmp, only=sid)
        except Exception:  # noqa: BLE001
            pass


def _best_config_retry() -> None:
    """requests-side retry (``_run_sync_backend_request_with_retry``)."""

    c = CANARIES
    _inject("POST", r"/best-configs$", [503, "connection"], expects_retry=True)
    _inject("GET", r"/best-configs/", [429, 500], expects_retry=True)
    client = _client()
    client.publish_best_config_sync(
        {c["knob_name"]: c["choice_value"], "metric": c["metric_name"]},
        environment=c["agent_name"],
    )
    client.fetch_best_config_sync(
        "best-canary-direct", environment=c["agent_name"], function_ref=c["agent_name"]
    )


async def _aiohttp_retry_optimize() -> None:
    """aiohttp retry decorator (``retry_http_request``) on ``/optimize``."""

    from traigent.cloud.client import TraigentCloudClient

    c = CANARIES
    _inject("POST", r"/optimize$", [429, "connection"], expects_retry=True)
    cloud = TraigentCloudClient(api_key="uk_" + ("a" * 43), base_url=BACKEND_URL)
    try:
        await cloud._submit_optimization(
            {
                "function_name": c["function_name"],
                "configuration_space": {c["knob_name"]: [c["choice_value"]]},
                "objectives": [c["metric_name"]],
                "metadata": {c["metadata_key"]: c["metadata_value"]},
            }
        )
    finally:
        await cloud.close() if hasattr(cloud, "close") else None


async def _aiohttp_failed_submissions() -> None:
    """Non-retrying aiohttp writes: the failed body is still on the wire."""

    c = CANARIES
    _inject(
        "POST", r"/sessions/[^/]+/results$", [500, "connection"], expects_retry=False
    )
    _inject(
        "PUT",
        r"/configuration-runs/[^/]+/status$",
        [408, "connection"],
        expects_retry=False,
    )
    client = _client()
    for _ in range(2):
        try:
            await client._submit_trial_result_via_session(
                "sess-canary-fail",
                "trial-canary-fail",
                {c["knob_name"]: c["choice_value"]},
                {c["metric_name"]: 0.5},
                "failed",
                error_message=f"boom {c['exception_message']}",
                metadata={c["metadata_key"]: c["metadata_value"]},
            )
        except Exception:  # noqa: BLE001 - the body was sent; the failure is the point
            pass
        try:
            await client._update_config_run_status(
                "cfg-canary-fail",
                "failed",
                error_message=f"boom {c['exception_message']}",
            )
        except Exception:  # noqa: BLE001
            pass


async def _optimizer_batch_retry() -> None:
    """Failed batches stay buffered and are resent on the next flush."""

    from traigent.cloud.optimizer_client import OptimizerDirectClient

    c = CANARIES
    _inject(
        "POST",
        r"/hybrid/optimizer/[^/]+/batch$",
        [503, "connection", 408],
        expects_retry=True,
    )
    async with OptimizerDirectClient(
        f"{BACKEND_URL}/api/v1/hybrid/optimizer", "optimizer-token"
    ) as direct:
        direct.set_batch_size(2)
        for i in range(2):
            await direct.submit_metrics(
                "hybrid-canary-retry",
                f"trial-canary-retry-{i}",
                {c["metric_name"]: 0.5},
                0.1,
                metadata={c["metadata_key"]: c["metadata_value"]},
            )
        for _ in range(4):
            try:
                await direct.flush()
            except Exception:  # noqa: BLE001
                pass


EXERCISERS: dict[str, Callable[[], Awaitable[None] | None]] = {
    "trial_result_failed": _trial_result_failed,
    "config_run_updates": _config_run_updates,
    "intermediate_report": _intermediate_report,
    "hybrid_session_and_optimizer": _hybrid_session_and_optimizer,
    "direct_serializer": _direct_serializer,
    "best_config_sync": _best_config_sync,
    "example_set_upload": _example_set_upload,
    "sync_manager_replay": _sync_manager_replay,
    "sync_manager_replay_with_faults": _sync_manager_replay_with_faults,
    "best_config_retry": _best_config_retry,
    "aiohttp_retry_optimize": _aiohttp_retry_optimize,
    "aiohttp_failed_submissions": _aiohttp_failed_submissions,
    "optimizer_batch_retry": _optimizer_batch_retry,
}


def run_exercisers(capture: EgressCapture | None = None) -> dict[str, str | None]:
    """Run every exerciser; return ``name -> error`` (None when it ran clean).

    ``capture`` receives the fault injections; without one, the failure-path
    exercisers cannot arrange their faults and are reported as failed.
    """

    global _ACTIVE  # noqa: PLW0603
    _ACTIVE = capture
    outcomes: dict[str, str | None] = {}
    for name, fn in EXERCISERS.items():
        try:
            out = fn()
            if asyncio.iscoroutine(out):
                asyncio.run(asyncio.wait_for(out, timeout=60))
            outcomes[name] = None
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            outcomes[name] = f"{type(exc).__name__}: {str(exc)[:160]}"
        finally:
            if capture is not None:
                capture.backend.retire_faults(name)
    return outcomes
