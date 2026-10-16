"""D4: privacy-mode session-create must not put the customer-chosen dataset
label (``dataset_metadata["name"]``) on the wire.

Scope: the ``dataset_metadata`` object itself. The label-derived
``dataset_id`` and ``metadata.evaluation_set`` are separate, owner-blessed
declared-identity fields and are covered by test_dataset_declared_identity.py.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.backend_bridges import SDKBackendBridge
from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.models import SessionCreationRequest

pytestmark = pytest.mark.backend_online

CANARY = "opt_ds_CANARYPATH5523"


def _request(privacy: bool = True, **extra: Any) -> SessionCreationRequest:
    dataset_metadata: dict[str, Any] = {
        "size": 8,
        "name": CANARY,
        "description": CANARY,
    }
    if privacy:
        dataset_metadata["privacy_mode"] = True
    return SessionCreationRequest(
        function_name="mcq_agent",
        configuration_space={"temperature": [0.0, 0.7]},
        objectives=["accuracy"],
        dataset_metadata=dataset_metadata,
        metadata={"evaluation_set": CANARY},
        **extra,
    )


def _typed(req: SessionCreationRequest) -> dict[str, Any]:
    return ApiOperations(MagicMock())._build_typed_session_payload(req, max_trials=5)


def _legacy(req: SessionCreationRequest) -> dict[str, Any]:
    return ApiOperations(MagicMock())._build_legacy_session_payload(req, max_trials=5)


def _direct(req: SessionCreationRequest) -> dict[str, Any]:
    client = object.__new__(TraigentCloudClient)
    client._ensure_owner_metadata = lambda metadata: metadata or {}
    return TraigentCloudClient._serialize_session_request(client, req)


def _bridge(req: SessionCreationRequest) -> dict[str, Any]:
    run = SDKBackendBridge().session_creation_to_backend_run(req, "exp-1")
    return dict(run.experiment_data)


def _wire_dataset_metadata(kind: str, req: SessionCreationRequest) -> Any:
    if kind == "typed":
        return _typed(req)["dataset_metadata"]
    if kind == "legacy":
        return _legacy(req)["dataset"]["metadata"]
    if kind == "direct":
        return _direct(req)["dataset_metadata"]
    return _bridge(req)["dataset_metadata"]


@pytest.mark.parametrize("kind", ["typed", "legacy", "direct", "bridge"])
def test_privacy_mode_dataset_metadata_has_no_label(kind: str) -> None:
    wire = _wire_dataset_metadata(kind, _request(privacy=True))
    assert CANARY not in json.dumps(wire, default=str)
    assert "name" not in wire
    assert wire.get("size") == 8


@pytest.mark.parametrize("kind", ["typed", "legacy", "direct", "bridge"])
def test_privacy_billing_tier_dataset_metadata_has_no_label(kind: str) -> None:
    wire = _wire_dataset_metadata(kind, _request(privacy=False, billing_tier="privacy"))
    assert CANARY not in json.dumps(wire, default=str)


@pytest.mark.parametrize("kind", ["typed", "legacy", "direct", "bridge"])
def test_non_privacy_mode_keeps_name(kind: str) -> None:
    wire = _wire_dataset_metadata(kind, _request(privacy=False))
    assert wire["name"] == CANARY
