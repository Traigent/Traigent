"""Dataset content fingerprint on the hybrid/privacy create_session path.

Production symptom (see issue background): typed/interactive sessions create
backend Experiment rows with no dataset identity because the dataset is
deliberately never uploaded on this path. The fix: SDK computes a
content-only sha256 fingerprint locally and sends *that* -- never the raw
dataset, never its name/label.

These tests exercise `PrivacyOperations.create_privacy_optimization_session`
end to end into the REAL wire-payload builder
(`ApiOperations._build_session_payload`, the same one covered by
tests/unit/cloud/test_typed_session_contract.py) so a passing test proves
the fingerprint actually reaches the JSON sent to the backend, not just an
intermediate dataclass field.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from unittest.mock import Mock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.models import OptimizationSession
from traigent.cloud.privacy_operations import PrivacyOperations
from traigent.evaluators.base import Dataset, EvaluationExample

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online


def _example(input_data, expected_output):
    return EvaluationExample(input_data=input_data, expected_output=expected_output)


class _CapturingClient:
    """Minimal backend client stub that records the SessionCreationRequest
    handed to session creation, so the test can feed it through the real
    wire-payload builder afterwards."""

    def __init__(self) -> None:
        self._active_sessions: dict[str, OptimizationSession] = {}
        self._max_active_sessions = 5
        self._active_sessions_lock = _NullLock()
        self.captured_request = None
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


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _wire_payload(session_request):
    """Run the request through the actual serializer used by the create-
    session HTTP path (traigent/cloud/api_operations.py)."""
    ops = ApiOperations(Mock())
    return ops._build_session_payload(session_request, max_trials=5)


@pytest.mark.asyncio
async def test_materialized_dataset_puts_nonnull_fingerprint_on_the_wire():
    client = _CapturingClient()
    ops = PrivacyOperations(client)
    dataset = Dataset(
        examples=[
            _example({"question": "a"}, "answer-a"),
            _example({"question": "b"}, "answer-b"),
        ],
        name="my-unique-portal-dataset-label-xyz123",
    )

    await ops.create_privacy_optimization_session(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
        dataset_metadata={"size": 2},
        max_trials=5,
        dataset=dataset,
    )

    assert client.captured_request is not None
    payload = _wire_payload(client.captured_request)

    fingerprint = payload["artifact_fingerprints"]["dataset"]
    assert fingerprint is not None
    assert fingerprint.startswith("fp1:")

    # Privacy posture: no raw example content or the dataset's own name
    # reaches the wire anywhere in the payload.
    blob = json.dumps(payload)
    assert "answer-a" not in blob
    assert "answer-b" not in blob
    assert dataset.name not in blob


@pytest.mark.asyncio
async def test_no_materialized_dataset_leaves_fingerprint_absent():
    """No materialized content available (only metadata) -> the dataset
    fingerprint stays None. It must never be invented from
    dataset_metadata (e.g. hashing the size/name)."""
    client = _CapturingClient()
    ops = PrivacyOperations(client)

    await ops.create_privacy_optimization_session(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
        dataset_metadata={"size": 10000, "type": "question_answering"},
        max_trials=5,
        # dataset intentionally omitted -- caller has no materialized content
    )

    assert client.captured_request is not None
    assert client.captured_request.artifact_fingerprints is None

    payload = _wire_payload(client.captured_request)
    # `artifact_fingerprints_to_wire(None)` returns None for a non-mapping
    # input, so `_attach_artifact_fingerprint_payload` omits the key
    # entirely rather than sending an all-None dict: missing stays missing.
    assert "artifact_fingerprints" not in payload


@pytest.mark.asyncio
async def test_same_content_different_names_same_fingerprint_on_the_wire():
    """The fingerprint is content-derived PROVENANCE: same content, same digest,
    regardless of what the dataset is called. Identity is the declared
    `dataset_id` (see test_dataset_declared_identity.py), never this digest."""
    examples = [_example({"question": "a"}, "answer-a")]

    client_a = _CapturingClient()
    await PrivacyOperations(client_a).create_privacy_optimization_session(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
        dataset_metadata={"size": 1},
        max_trials=5,
        dataset=Dataset(examples=list(examples), name="dataset-one"),
    )

    client_b = _CapturingClient()
    await PrivacyOperations(client_b).create_privacy_optimization_session(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
        dataset_metadata={"size": 1},
        max_trials=5,
        dataset=Dataset(examples=list(examples), name="a-completely-different-name"),
    )

    fp_a = _wire_payload(client_a.captured_request)["artifact_fingerprints"]["dataset"]
    fp_b = _wire_payload(client_b.captured_request)["artifact_fingerprints"]["dataset"]

    assert fp_a is not None
    assert fp_a == fp_b


@pytest.mark.asyncio
async def test_different_content_different_fingerprint_on_the_wire():
    client_a = _CapturingClient()
    await PrivacyOperations(client_a).create_privacy_optimization_session(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
        dataset_metadata={"size": 1},
        max_trials=5,
        dataset=Dataset(examples=[_example({"question": "a"}, "answer-a")]),
    )

    client_b = _CapturingClient()
    await PrivacyOperations(client_b).create_privacy_optimization_session(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
        dataset_metadata={"size": 1},
        max_trials=5,
        dataset=Dataset(examples=[_example({"question": "a"}, "DIFFERENT")]),
    )

    fp_a = _wire_payload(client_a.captured_request)["artifact_fingerprints"]["dataset"]
    fp_b = _wire_payload(client_b.captured_request)["artifact_fingerprints"]["dataset"]

    assert fp_a is not None
    assert fp_b is not None
    assert fp_a != fp_b


@pytest.mark.asyncio
async def test_generator_dataset_is_not_consumed_and_yields_no_fingerprint():
    """A single-use iterator the caller still needs must never be drained
    by fingerprinting. Since it cannot be safely fingerprinted, the
    fingerprint must stay absent (never invented) -- and the caller must
    still be able to read every item afterwards."""

    def gen() -> Iterator[EvaluationExample]:
        yield _example({"question": "a"}, "answer-a")
        yield _example({"question": "b"}, "answer-b")

    generator = gen()

    client = _CapturingClient()
    ops = PrivacyOperations(client)

    await ops.create_privacy_optimization_session(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
        dataset_metadata={"size": 2},
        max_trials=5,
        dataset=generator,
    )

    assert client.captured_request is not None
    assert client.captured_request.artifact_fingerprints is None

    # The caller can still read every item -- the generator was not touched.
    remaining = list(generator)
    assert len(remaining) == 2
    assert remaining[0].expected_output == "answer-a"
    assert remaining[1].expected_output == "answer-b"


@pytest.mark.asyncio
async def test_dataset_id_kwarg_threads_into_captured_request():
    """Correction (see tests/unit/cloud/test_dataset_declared_identity.py):
    declared identity is a SEPARATE keyword from the content fingerprint --
    both may be set on the same call."""
    client = _CapturingClient()
    ops = PrivacyOperations(client)
    dataset = Dataset(examples=[_example({"question": "a"}, "answer-a")])

    await ops.create_privacy_optimization_session(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=["accuracy"],
        dataset_metadata={"size": 1},
        max_trials=5,
        dataset=dataset,
        dataset_id="my-stable-dataset-id",
    )

    assert client.captured_request is not None
    assert client.captured_request.dataset_id == "my-stable-dataset-id"
    assert client.captured_request.artifact_fingerprints["dataset"] is not None
