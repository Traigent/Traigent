"""Deterministic snapshot of what an SDK run sends when NO purpose-key grant exists.

Used by ``test_no_grant_payloads_match_develop.py``. The SAME module is run
against the base develop commit to produce
``fixtures/no_grant_payloads_develop.json`` (see that test's docstring), so it
only uses APIs that exist on develop. Volatile values (timings, generated ids,
the SDK version string, object reprs) are normalized; everything else --
every key, every value -- must match byte for byte.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

_VOLATILE_KEY = re.compile(
    r"(time|duration|timestamp|_ms$|_at$|elapsed|latency|trial_id|session_id|"
    r"optimization_id|run_id|sdk_version|tokens_per_second)",
    re.IGNORECASE,
)
#: Keyed per-example signal digests (example_digest / output_digest /
#: signal_key_id) depend on the environment's project key material; their
#: presence is compared, their value is not.
_HEX_DIGEST = re.compile(r"[0-9a-f]{12,64}")
_UUIDISH = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _normalize(value: Any, key: str = "") -> Any:
    if key and _VOLATILE_KEY.search(key):
        return "<volatile>"
    if isinstance(value, dict):
        return {
            str(k): _normalize(v, str(k))
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, str):
        if _HEX_DIGEST.fullmatch(value):
            return "<hex>"
        return _UUIDISH.sub("<uuid>", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return f"<{type(value).__name__}>"


def snapshot_agent(text: str) -> str:
    return text.upper()


def _dataset() -> Any:
    from traigent.evaluators.base import Dataset, EvaluationExample

    rows = [("a", "A", "row-a"), ("b", "B", "row-b"), ("a", "A", "row-a2")]
    return Dataset(
        examples=[
            EvaluationExample(
                input_data={"text": text},
                expected_output=expected,
                metadata={"example_id": user_id},
            )
            for text, expected, user_id in rows
        ],
        name="no_grant_dataset",
    )


async def _run_optimize() -> tuple[dict[str, Any], Any]:
    from tests.shared.mocks.optimizers import MockOptimizer
    from traigent.config.types import TraigentConfig
    from traigent.core.backend_session_manager import BackendSessionManager
    from traigent.core.orchestrator import OptimizationOrchestrator
    from traigent.evaluators.local import LocalEvaluator

    captured: dict[str, Any] = {}
    original = BackendSessionManager.create_session

    def capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return original(self, *args, **kwargs)

    BackendSessionManager.create_session = capture  # type: ignore[method-assign]
    try:
        optimizer = MockOptimizer(
            config_space={"alpha": [0, 1]}, objectives=["accuracy"]
        )
        optimizer.set_max_suggestions(2)
        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=LocalEvaluator(metrics=["accuracy"], detailed=True),
            max_trials=2,
            config=TraigentConfig(offline=True, algorithm="grid"),
            agent_key="agent_snapshot",
        )
        result = await orchestrator.optimize(snapshot_agent, _dataset())
    finally:
        BackendSessionManager.create_session = original  # type: ignore[method-assign]
    return captured, result


def _session_bodies() -> dict[str, Any]:
    """Both typed session-create serializers' bodies for a no-grant request."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from traigent.cloud.api_operations import ApiOperations
    from traigent.cloud.client import TraigentCloudClient
    from traigent.cloud.models import SessionCreationRequest

    request = SessionCreationRequest(
        function_name="snapshot_agent",
        configuration_space={"alpha": [0, 1]},
        objectives=[{"name": "accuracy", "orientation": "maximize", "weight": 1.0}],
        dataset_metadata={"size": 3, "name": "no_grant"},
        max_trials=2,
        agent_key="agent_snapshot",
    )
    stub = SimpleNamespace(_ensure_owner_metadata=lambda metadata: metadata or {})
    return {
        "typed": ApiOperations(Mock())._build_session_payload(request, max_trials=2),
        "cloud": TraigentCloudClient._serialize_session_request(stub, request),
    }


def build_snapshot() -> dict[str, Any]:
    """Everything a no-grant run hands the Backend, normalized."""
    from traigent.config.types import TraigentConfig
    from traigent.core.metadata_helpers import build_backend_metadata

    os.environ.setdefault("TRAIGENT_COST_APPROVED", "true")
    os.environ.setdefault("TRAIGENT_RUN_COST_LIMIT", "100.0")
    captured, result = asyncio.run(_run_optimize())
    trials = []
    for trial in result.trials:
        trials.append(
            {
                "backend_metadata": build_backend_metadata(
                    trial, "accuracy", TraigentConfig(offline=True), "no_grant_dataset"
                ),
                "example_results": trial.metadata.get("example_results"),
                "metadata_keys": sorted(trial.metadata),
            }
        )
    snapshot = {
        "session_create_call": {
            "kwarg_names": sorted(captured),
            "values": {
                k: v
                for k, v in captured.items()
                if k not in ("func", "dataset", "function_descriptor", "start_time")
            },
        },
        "session_create_bodies": _session_bodies(),
        "trials": trials,
    }
    return json.loads(json.dumps(_normalize(snapshot), sort_keys=True))
