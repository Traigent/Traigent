"""End to end: an optimization run records per-trial content identity.

Drives a real ``OptimizationOrchestrator`` + ``LocalEvaluator`` run and checks
what each trial records and what the Backend submission carries:

* with a purpose-key grant: every ``ExampleResult`` carries the keyed ``ex1``
  id (the positional ``example_<i>`` fallback is gone) and ``exv1`` version,
  the user's own id moves to ``external_id``, and every trial records
  ``dataset_root`` plus its ``EvaluatedSetV1`` (root recomputable from the
  members) and a candidate agent version whose ``applied_config_digest``
  follows the trial's configuration;
* without a grant: no content id anywhere (fail closed) and the legacy per-row
  handle is unchanged;
* privacy mode withholds the whole block from the Backend payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.shared.mocks.optimizers import MockOptimizer
from traigent.config.types import TraigentConfig
from traigent.core.metadata_helpers import build_backend_metadata
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.evaluators.local import LocalEvaluator
from traigent.identity import content_identity as ci
from traigent.identity.keys import (
    ContentIdentityKeys,
    clear_content_identity_keys,
    set_content_identity_keys,
)

VECTORS = json.loads(
    (Path(__file__).parent / "fixtures" / "content_identity_v1_vectors.json").read_text(
        encoding="utf-8"
    )
)


@pytest.fixture(autouse=True)
def _approve_cost(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    monkeypatch.setenv("TRAIGENT_RUN_COST_LIMIT", "100.0")
    clear_content_identity_keys()
    yield
    clear_content_identity_keys()


def _grant() -> ContentIdentityKeys:
    row = VECTORS["key_derivation"][0]
    return ContentIdentityKeys.from_grant(
        {
            "tenant_id": row["tenant_id"],
            "kid": row["key_id"],
            "example_id_key": row["example_id_key_hex"],
            "example_version_key": row["example_version_key_hex"],
        }
    )


def identity_agent(text: str) -> str:
    return text.upper()


def _dataset() -> Dataset:
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
        name="identity_dataset",
    )


async def _run(agent_key: str | None = "agent_identity_test") -> Any:
    optimizer = MockOptimizer(config_space={"alpha": [0, 1]}, objectives=["accuracy"])
    optimizer.set_max_suggestions(2)
    orchestrator = OptimizationOrchestrator(
        optimizer=optimizer,
        evaluator=LocalEvaluator(metrics=["accuracy"], detailed=True),
        max_trials=2,
        config=TraigentConfig(offline=True, algorithm="grid"),
        agent_key=agent_key,
    )
    return await orchestrator.optimize(identity_agent, _dataset())


def _example_results(trial: Any) -> list[dict[str, Any]]:
    return list(trial.metadata.get("example_results") or [])


@pytest.mark.asyncio
async def test_keyed_run_records_identity_per_trial() -> None:
    set_content_identity_keys(_grant())
    result = await _run()
    assert len(result.trials) == 2

    configs_seen = set()
    for trial in result.trials:
        examples = _example_results(trial)
        assert len(examples) == 3
        for example in examples:
            assert example["example_id"].startswith("ex1:")
            assert example["example_version"].startswith("exv1:")
            assert not example["example_id"].startswith("example_")
        assert {e["external_id"] for e in examples} == {"row-a", "row-b", "row-a2"}
        # Rows 0 and 2 have identical content: one example_id, counted twice.
        assert examples[0]["example_id"] == examples[2]["example_id"]

        block = trial.metadata["content_identity"]
        assert block["scheme"] == ci.SCHEME
        assert block["provenance"] == "declared"
        assert block["trial_id"] == trial.trial_id
        assert block["unavailable"] == {}
        evaluated = block["evaluated"]
        dataset_root = evaluated["dataset_root"]
        assert evaluated["evaluated_root"] == dataset_root  # whole dataset, once
        assert (evaluated["distinct_count"], evaluated["total_count"]) == (2, 3)
        recomputed = ci.compute_multiset_root(
            [
                (m["example_id"], m["example_version"], m["count"])
                for m in evaluated["members"]
            ]
        )
        assert recomputed.root == evaluated["evaluated_root"]

        candidate = block["candidate"]
        assert candidate["agent_id"] == "agent_identity_test"
        assert (
            ci.compute_agent_build_digest(candidate["manifest"])
            == (candidate["build_digest"])
        )
        configs_seen.add(candidate["manifest"]["applied_config_digest"])
        assert block["observed_provider_versions"] == []  # no provider was called

        wire = build_backend_metadata(
            trial, "accuracy", TraigentConfig(offline=True), "identity_dataset"
        )
        assert wire["content_identity"] == block
        # The pattern-constrained measures[].example_id wire field never carries
        # a content id (configuration_run_schema: ^ex_[a-f0-9]{8,12}_[0-9]+$).
        for measure in wire.get("measures", []):
            assert not measure["example_id"].startswith("ex1:")
    assert len(configs_seen) == 2  # each trial's configuration is its own candidate


@pytest.mark.asyncio
async def test_unkeyed_run_mints_no_content_ids() -> None:
    result = await _run()
    for trial in result.trials:
        examples = _example_results(trial)
        assert [e["example_id"] for e in examples] == ["row-a", "row-b", "row-a2"]
        assert all("example_version" not in e for e in examples)
        block = trial.metadata["content_identity"]
        assert block["evaluated"] is None
        assert block["unavailable"]["evaluated"] == "purpose_keys_unavailable"
        assert block["candidate"] is not None  # the build version needs no key
        assert "ex1:" not in json.dumps(trial.metadata, default=str)


@pytest.mark.asyncio
async def test_privacy_mode_withholds_the_block_from_the_backend_payload() -> None:
    set_content_identity_keys(_grant())
    result = await _run()
    trial = result.trials[0]
    assert "content_identity" in trial.metadata
    private = TraigentConfig(offline=True)
    with pytest.warns(DeprecationWarning):
        private.privacy_enabled = True  # still the flag build_backend_metadata reads
    wire = build_backend_metadata(trial, "accuracy", private, "identity_dataset")
    assert "content_identity" not in wire


@pytest.mark.asyncio
async def test_no_agent_key_falls_back_to_the_function_name() -> None:
    set_content_identity_keys(_grant())
    result = await _run(agent_key=None)
    block = result.trials[0].metadata["content_identity"]
    # No declared agent name: the function name is the (flagged) fallback id.
    assert block["candidate"]["agent_id"] == "identity_agent"
    assert block["evaluated"] is not None


def _session_request(content_identity: dict[str, Any] | None) -> Any:
    from traigent.cloud.models import SessionCreationRequest

    return SessionCreationRequest(
        function_name="identity_agent",
        configuration_space={"alpha": [0, 1]},
        objectives=[{"name": "accuracy", "orientation": "maximize", "weight": 1.0}],
        dataset_metadata={"size": 3},
        content_identity=content_identity,
    )


def test_both_session_serializers_carry_the_top_level_object() -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    from traigent.cloud.api_operations import ApiOperations
    from traigent.cloud.client import TraigentCloudClient
    from traigent.identity.run import prepare_content_identity_run

    set_content_identity_keys(_grant())
    wire = prepare_content_identity_run(
        identity_agent, _dataset(), agent_key="agent_identity_test"
    ).session_wire({})
    request = _session_request(wire)
    typed = ApiOperations(Mock())._build_session_payload(request, max_trials=2)
    stub = SimpleNamespace(_ensure_owner_metadata=lambda metadata: metadata or {})
    cloud = TraigentCloudClient._serialize_session_request(stub, request)
    for payload in (typed, cloud):
        assert payload["content_identity"] == wire
        assert payload["content_identity"]["dataset"]["record_state"] == "draft"
    # Absent object: the body is unchanged (no key at all).
    plain = ApiOperations(Mock())._build_session_payload(
        _session_request(None), max_trials=2
    )
    assert "content_identity" not in plain


def test_session_manager_builds_the_object_and_withholds_it_in_privacy_mode() -> None:
    from traigent.core.backend_session_manager import BackendSessionManager

    set_content_identity_keys(_grant())
    manager = BackendSessionManager.__new__(BackendSessionManager)
    manager._traigent_config = TraigentConfig(offline=True)
    wire = manager._session_content_identity(
        identity_agent, _dataset(), "agent_identity_test", None
    )
    assert wire is not None and wire["key_status"] == "available"
    private = TraigentConfig(offline=True)
    with pytest.warns(DeprecationWarning):
        private.privacy_enabled = True
    manager._traigent_config = private
    assert (
        manager._session_content_identity(
            identity_agent, _dataset(), "agent_identity_test", None
        )
        is None
    )


@pytest.mark.backend_online  # SDK #2033: exercise the connected create path
def test_session_operations_threads_the_object_to_the_request() -> None:
    from typing import cast

    from tests.unit.cloud.test_dataset_declared_identity import CapturingFakeClient
    from traigent.cloud.session_operations import SessionOperations

    fake = CapturingFakeClient()
    wire = {"scheme": ci.SCHEME, "provenance": "declared", "key_status": "unavailable"}
    SessionOperations(cast(Any, fake)).create_session(
        "my_func",
        {"model": ["a", "b"]},
        metadata={"max_trials": 5, "dataset_size": 1, "evaluation_set": "lbl"},
        content_identity=wire,
    )
    assert fake.captured_session_request is not None
    assert fake.captured_session_request.content_identity == wire
