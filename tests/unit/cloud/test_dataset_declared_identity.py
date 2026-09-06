"""Declared dataset identity on the typed session-create contract.

Correction to d507ccfa: a content fingerprint is the WRONG identity for
portal grouping. Identity must be DECLARED and STABLE -- changing a
dataset's content (adding an example, fixing a label) must NOT change its
identity, exactly as changing an agent's code does not change the agent's
identity. The content fingerprint keeps shipping as
``artifact_fingerprints["dataset"]`` -- it is valuable PROVENANCE (did the
content drift?) but is no longer identity.

The pivotal test below exercises `PrivacyOperations.create_privacy_
optimization_session` end to end into the REAL wire-payload builder
(`ApiOperations._build_session_payload`), same harness as
tests/unit/cloud/test_privacy_session_fingerprint.py, so a passing test
proves both halves of the correction actually reach the JSON sent to the
backend. The remaining cases exercise the wire builder directly against a
`SessionCreationRequest` (same style as test_typed_session_contract.py),
since the label-derivation rule only depends on
`dataset_metadata`/`metadata`, not on how the request was constructed.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.models import OptimizationSession, SessionCreationRequest
from traigent.cloud.privacy_operations import PrivacyOperations
from traigent.evaluators.base import Dataset, EvaluationExample

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online


def _example(input_data, expected_output):
    return EvaluationExample(input_data=input_data, expected_output=expected_output)


def _request(**overrides) -> SessionCreationRequest:
    base = {
        "function_name": "qa_agent",
        "configuration_space": {"temperature": [0.0, 1.0]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 1},
        "max_trials": 5,
    }
    base.update(overrides)
    return SessionCreationRequest(**base)


def _wire(request: SessionCreationRequest) -> dict:
    return ApiOperations(Mock())._build_session_payload(request, max_trials=5)


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _CapturingClient:
    """Minimal backend client stub that records the SessionCreationRequest
    handed to session creation, so the test can feed it through the real
    wire-payload builder afterwards (mirrors
    tests/unit/cloud/test_privacy_session_fingerprint.py)."""

    def __init__(self) -> None:
        self._active_sessions: dict[str, OptimizationSession] = {}
        self._max_active_sessions = 5
        self._active_sessions_lock = _NullLock()
        self.captured_request: SessionCreationRequest | None = None
        from types import SimpleNamespace

        self.session_bridge = SimpleNamespace(
            create_session_mapping=lambda **kwargs: None,
        )

    async def _check_rate_limit(self) -> None:
        return None

    async def _create_traigent_session_via_api(self, request):
        self.captured_request = request
        return ("session-1", "experiment-1", "run-1")

    def _register_security_session(self, *args, **kwargs):
        return None


async def _create_via_privacy_ops(*, dataset, dataset_id) -> dict:
    client = _CapturingClient()
    ops = PrivacyOperations(client)
    await ops.create_privacy_optimization_session(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
        dataset_metadata={"size": len(dataset.examples)},
        max_trials=5,
        dataset=dataset,
        dataset_id=dataset_id,
    )
    assert client.captured_request is not None
    return _wire(client.captured_request)


@pytest.mark.asyncio
async def test_same_declared_id_survives_a_content_change():
    """THE pivotal correction, in one assertion: the SAME declared id must
    reach the wire for two datasets that differ only in content, while the
    content fingerprint legitimately differs between them."""
    v1 = Dataset(examples=[_example({"question": "a"}, "answer-a")])
    v1_plus_one_example = Dataset(
        examples=[
            _example({"question": "a"}, "answer-a"),
            _example({"question": "b"}, "answer-b"),
        ]
    )

    payload_v1 = await _create_via_privacy_ops(
        dataset=v1, dataset_id="stable-dataset-42"
    )
    payload_v2 = await _create_via_privacy_ops(
        dataset=v1_plus_one_example, dataset_id="stable-dataset-42"
    )

    # Identity half: unchanged by the content edit.
    assert payload_v1["dataset_id_source"] == "declared"
    assert payload_v2["dataset_id_source"] == "declared"
    assert payload_v1["dataset_id"] == "stable-dataset-42"
    assert payload_v2["dataset_id"] == "stable-dataset-42"
    assert payload_v1["dataset_id"] == payload_v2["dataset_id"]

    # Provenance half: legitimately different -- the fingerprint still
    # detects that the content drifted.
    fp_v1 = payload_v1["artifact_fingerprints"]["dataset"]
    fp_v2 = payload_v2["artifact_fingerprints"]["dataset"]
    assert fp_v1 is not None
    assert fp_v2 is not None
    assert fp_v1 != fp_v2


def test_explicit_dataset_id_wins_over_label_derived_default():
    payload = _wire(
        _request(
            dataset_id="explicit-id",
            dataset_metadata={"size": 1, "name": "label-derived-name"},
        )
    )
    assert payload["dataset_id_source"] == "declared"
    assert payload["dataset_id"] == "explicit-id"


def test_label_derived_default_used_when_no_explicit_id_given():
    payload = _wire(_request(dataset_metadata={"size": 1, "name": "my-eval-set-label"}))
    assert payload["dataset_id_source"] == "declared"
    assert payload["dataset_id"] == "my-eval-set-label"


def test_evaluation_set_metadata_used_when_no_name_present():
    payload = _wire(
        _request(
            dataset_metadata={"size": 1},
            metadata={"evaluation_set": "regression-suite-v3"},
        )
    )
    assert payload["dataset_id_source"] == "declared"
    assert payload["dataset_id"] == "regression-suite-v3"


def test_no_label_and_no_explicit_id_sends_no_declared_identity():
    """No dataset_metadata['name'], no metadata['evaluation_set'], no
    explicit dataset_id -- absence must stay absence. No identity is
    invented from example content or anything else."""
    payload = _wire(_request(dataset_metadata={"size": 1}))
    assert "dataset_id" not in payload
    assert "dataset_id_source" not in payload


def test_sdk_default_dataset_name_sentinel_is_not_promoted_to_identity():
    """`Dataset.name` defaults to the literal "dataset" when the caller
    never names it (traigent/evaluators/base.py). Promoting that sentinel to
    a declared id would collide every unnamed dataset onto one shared
    identity -- exactly the bug this feature exists to prevent."""
    payload = _wire(_request(dataset_metadata={"size": 1, "name": "dataset"}))
    assert "dataset_id" not in payload
    assert "dataset_id_source" not in payload


def test_default_evaluation_set_sentinel_is_not_promoted_to_identity():
    payload = _wire(
        _request(dataset_metadata={"size": 1}, metadata={"evaluation_set": "default"})
    )
    assert "dataset_id" not in payload
    assert "dataset_id_source" not in payload


def test_dataset_id_present_only_with_declared_source_contract_valid():
    """Every payload emitted by this path either omits both fields, or
    carries them in the only combination the backend accepts for a
    client-declared id: dataset_id present iff dataset_id_source=="declared"."""
    cases = [
        _wire(_request(dataset_metadata={"size": 1})),  # nothing declared
        _wire(_request(dataset_id="x", dataset_metadata={"size": 1})),  # explicit
        _wire(_request(dataset_metadata={"size": 1, "name": "lbl"})),  # derived
    ]
    for payload in cases:
        has_id = "dataset_id" in payload
        has_source = "dataset_id_source" in payload
        assert has_id == has_source
        if has_id:
            assert payload["dataset_id_source"] == "declared"
            assert isinstance(payload["dataset_id"], str) and payload["dataset_id"]


def test_session_creation_request_rejects_blank_dataset_id():
    with pytest.raises(ValueError, match="dataset_id must be a non-blank string"):
        SessionCreationRequest(function_name="f", dataset_id="   ")


def test_session_creation_request_rejects_oversized_dataset_id():
    with pytest.raises(ValueError, match="dataset_id must be at most 255 characters"):
        SessionCreationRequest(function_name="f", dataset_id="x" * 256)


def test_session_creation_request_strips_dataset_id():
    request = SessionCreationRequest(function_name="f", dataset_id="  padded-id  ")
    assert request.dataset_id == "padded-id"
