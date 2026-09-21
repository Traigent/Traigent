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

from types import SimpleNamespace

from unittest.mock import AsyncMock, Mock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.models import (
    DATASET_ID_MAX_LENGTH,
    normalize_declared_dataset_id,
    OptimizationSession,
    SessionCreationRequest,
    session_dataset_identity_to_wire,
)
from traigent.cloud.privacy_operations import PrivacyOperations
from traigent.evaluators.base import Dataset, EvaluationExample
import logging
import re
import unicodedata
import time
from typing import Any, cast

from tests.unit.cloud.test_session_creation_warm_start import (
    CapturingFakeClient,
)
from traigent.api.decorators import EvaluationOptions, optimize
from traigent.api.types import OptimizationStatus
from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.models import (
    DECLARED_DATASET_IDENTITY_METADATA_KEY,
    SessionCreationResponse,
)
from traigent.cloud.session_operations import SessionOperations
from traigent.cloud.session_types import (
    SessionCreationFailureReason,
    SessionCreationResult,
)
from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.evaluators.base import load_inline_dataset
from traigent.optimizers.interactive_optimizer import (
    InteractiveOptimizer,
    RemoteGuidanceService,
)
from traigent.storage.local_storage import (
    LocalStorageManager,
)
from traigent.storage.local_storage import (
    OptimizationSession as LocalSession,
)
from traigent.utils.function_identity import resolve_function_descriptor

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
    assert payload["dataset_id"] is None
    assert payload["dataset_id_source"] == "unknown"


def test_sdk_default_dataset_name_sentinel_is_not_promoted_to_identity():
    """`Dataset.name` defaults to the literal "dataset" when the caller
    never names it (traigent/evaluators/base.py). Promoting that sentinel to
    a declared id would collide every unnamed dataset onto one shared
    identity -- exactly the bug this feature exists to prevent."""
    payload = _wire(_request(dataset_metadata={"size": 1, "name": "dataset"}))
    assert payload["dataset_id"] is None
    assert payload["dataset_id_source"] == "unknown"


def test_default_evaluation_set_sentinel_is_not_promoted_to_identity():
    payload = _wire(
        _request(dataset_metadata={"size": 1}, metadata={"evaluation_set": "default"})
    )
    assert payload["dataset_id"] is None
    assert payload["dataset_id_source"] == "unknown"


def test_dataset_id_present_only_with_declared_source_contract_valid():
    """Every typed payload carries a correlated identity source and value."""
    cases = [
        _wire(_request(dataset_metadata={"size": 1})),  # nothing declared
        _wire(_request(dataset_id="x", dataset_metadata={"size": 1})),  # explicit
        _wire(_request(dataset_metadata={"size": 1, "name": "lbl"})),  # derived
    ]
    for payload in cases:
        assert "dataset_id_source" in payload
        if payload["dataset_id_source"] == "declared":
            assert payload["dataset_id_source"] == "declared"
            assert isinstance(payload["dataset_id"], str) and payload["dataset_id"]
        else:
            assert payload["dataset_id_source"] == "unknown"
            assert payload["dataset_id"] is None


def test_session_creation_request_rejects_blank_dataset_id():
    with pytest.raises(ValueError, match="dataset_id must be a non-blank string"):
        SessionCreationRequest(function_name="f", dataset_id="   ")


def test_session_creation_request_rejects_oversized_dataset_id():
    with pytest.raises(ValueError, match="dataset_id must be at most 255 characters"):
        SessionCreationRequest(function_name="f", dataset_id="x" * 256)


def test_session_creation_request_strips_dataset_id():
    request = SessionCreationRequest(function_name="f", dataset_id="  padded-id  ")
    assert request.dataset_id == "padded-id"


def test_cloud_brain_serializer_also_emits_the_declared_identity():
    """The OTHER typed serializer must not drop half the identity.

    There are two typed session-create serializers: ``ApiOperations.
    _build_typed_session_payload`` (local/hybrid execution) and
    ``TraigentCloudClient._serialize_session_request`` (the cloud-brain /
    smart-algorithm path). History is grouped by (agent, dataset), so a
    serializer that emits the agent half and drops the dataset half fragments
    the cohort exactly as dropping ``agent_key`` once did -- the very drift the
    comment above that call site warns about. Both must emit the same fields.
    """
    from traigent.cloud.client import TraigentCloudClient

    request = SessionCreationRequest(
        function_name="qa_agent",
        configuration_space={"temperature": [0.0, 1.0]},
        objectives=[{"name": "accuracy", "orientation": "maximize", "weight": 1.0}],
        dataset_metadata={"name": "my-declared-eval", "size": 3},
    )
    # The serializer only needs owner-metadata passthrough from `self`; a stand-in
    # keeps this a unit test of the wire shape rather than of client construction.
    stub = SimpleNamespace(_ensure_owner_metadata=lambda metadata: metadata or {})
    payload = TraigentCloudClient._serialize_session_request(stub, request)

    assert payload.get("dataset_id_source") == "declared"
    assert payload.get("dataset_id") == "my-declared-eval", (
        "the cloud-brain serializer dropped the declared dataset identity; rows "
        "created through this path would still render 'No dataset'"
    )


# ---------------------------------------------------------------------------
# Over-long derived labels must not collide (codex review, 2026-09-07)
# ---------------------------------------------------------------------------


def _long_label(suffix: str) -> str:
    """A label whose first 255 characters are identical to its sibling's."""
    return "x" * DATASET_ID_MAX_LENGTH + suffix


def test_two_labels_sharing_a_255_char_prefix_get_distinct_identities():
    """Cutting a label to fit the backend cap merged unrelated datasets.

    The backend rejects a ``dataset_id`` over 255 characters, and this
    serializer used to cut the label down to fit. Two labels agreeing on their
    first 255 characters therefore produced the SAME declared identity, so two
    unrelated datasets converged on one Benchmark row and their optimization
    histories merged -- the precise failure this feature exists to prevent,
    inverted. Sibling paths are the realistic shape: a path is a blessed
    identity, and siblings differ only in their final segment.
    """
    first = session_dataset_identity_to_wire(
        SimpleNamespace(dataset_id=None, dataset_metadata={"name": _long_label("A")})
    )
    second = session_dataset_identity_to_wire(
        SimpleNamespace(dataset_id=None, dataset_metadata={"name": _long_label("B")})
    )

    assert first["dataset_id"] != second["dataset_id"], (
        "two datasets whose labels share a 255-character prefix collapsed onto "
        "one identity; their histories would silently merge"
    )
    for emitted in (first, second):
        assert len(emitted["dataset_id"]) <= DATASET_ID_MAX_LENGTH, (
            "the emitted identity exceeds the backend's cap and the session "
            "create would be rejected outright"
        )


def test_over_long_label_identity_is_deterministic_across_calls():
    """Identity is computed from the label, so it must not drift per call."""
    label = _long_label("stable")
    first = session_dataset_identity_to_wire(
        SimpleNamespace(dataset_id=None, dataset_metadata={"name": label})
    )
    second = session_dataset_identity_to_wire(
        SimpleNamespace(dataset_id=None, dataset_metadata={"name": label})
    )
    assert first == second


def test_labels_within_the_cap_keep_their_plain_readable_identity():
    """Only the over-long case is folded.

    Folding every label would remint the identity of every dataset already
    linked with a plain label and split its history at the upgrade.
    """
    emitted = session_dataset_identity_to_wire(
        SimpleNamespace(dataset_id=None, dataset_metadata={"name": "my-eval-set"})
    )
    assert emitted["dataset_id"] == "my-eval-set"


def test_cloud_brain_serializer_also_bounds_the_over_long_label():
    """The second typed serializer must carry the fix too.

    Both typed serializers share ``session_dataset_identity_to_wire``, and this
    asserts that sharing rather than assuming it: a future refactor that
    re-implements the wire shape on one path would reintroduce the collision on
    exactly one of the two lanes.
    """
    from traigent.cloud.client import TraigentCloudClient

    stub = SimpleNamespace(_ensure_owner_metadata=lambda metadata: metadata or {})

    def _emit(suffix: str) -> str:
        request = SessionCreationRequest(
            function_name="qa_agent",
            configuration_space={"temperature": [0.0, 1.0]},
            objectives=[{"name": "accuracy", "orientation": "maximize", "weight": 1.0}],
            dataset_metadata={"name": _long_label(suffix), "size": 3},
        )
        return TraigentCloudClient._serialize_session_request(stub, request)[
            "dataset_id"
        ]

    first, second = _emit("A"), _emit("B")
    assert first != second
    assert len(first) <= DATASET_ID_MAX_LENGTH
    assert len(second) <= DATASET_ID_MAX_LENGTH


def test_an_explicit_over_long_id_is_rejected_rather_than_quietly_folded():
    """An explicit id is the caller's assertion; it is never rewritten.

    Substituting a digest for an id the caller chose would make their identity
    unpredictable, so the length rule stays a loud construction-time error.
    """
    with pytest.raises(ValueError, match="at most 255 characters"):
        SessionCreationRequest(
            function_name="qa_agent",
            configuration_space={"temperature": [0.0, 1.0]},
            objectives=[{"name": "accuracy", "orientation": "maximize", "weight": 1.0}],
            dataset_id="y" * (DATASET_ID_MAX_LENGTH + 1),
        )


# ---------------------------------------------------------------------------
# Public EvaluationOptions.dataset_id, inline-name sentinel, offline sync
# ---------------------------------------------------------------------------

_SECRET_EXAMPLE = "SECRET-EXAMPLE-CONTENT-7f3a"
_INLINE = [{"input": {"text": _SECRET_EXAMPLE}, "output": "ok"}]


def _typed_payload_via_session_ops(create_kwargs: dict[str, Any]) -> dict:
    """Replay captured backend_client.create_session kwargs through the REAL
    BackendIntegratedClient -> SessionOperations -> SessionCreationRequest ->
    ApiOperations typed builder, returning the actual request body."""
    fake = CapturingFakeClient()
    client = object.__new__(BackendIntegratedClient)
    client._session_ops = SessionOperations(cast(Any, fake))
    kwargs = dict(create_kwargs)
    client.create_session(
        kwargs.pop("function_name"),
        kwargs.pop("search_space"),
        kwargs.pop("optimization_goal"),
        kwargs.pop("metadata"),
        **kwargs,
    )
    assert fake.captured_session_request is not None
    return ApiOperations(Mock())._build_typed_session_payload(
        fake.captured_session_request, max_trials=5
    )


class _RecordingBackendClient:
    """Backend-client stub for a public connected grid run: records the
    session-create kwargs and reports a local fallback so no further network
    interaction is attempted."""

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.no_egress = False
        self.cloud_egress_intent = False
        self.enable_fallback = True
        self.local_storage = None
        auth = Mock()
        auth.has_api_key = Mock(return_value=True)
        self.auth_manager = auth
        self.auth = auth

    def create_session(self, *args: Any, **kwargs: Any) -> SessionCreationResult:
        assert not args, "create_session must be called with keywords"
        self.create_calls.append(kwargs)
        return SessionCreationResult.fallback(
            session_id="local-fallback-1",
            reason=SessionCreationFailureReason.SESSION_FAILED,
            detail="recorded by test stub",
        )

    def __getattr__(self, name: str) -> Any:  # any other client call is inert
        return Mock(return_value=None)


async def _public_connected_grid_run(monkeypatch, tmp_path, evaluation) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
    monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
    stub = _RecordingBackendClient()
    monkeypatch.setattr(
        BackendSessionManager,
        "create_backend_client",
        staticmethod(lambda _config: stub),
    )

    @optimize(
        evaluation=evaluation,
        objectives=["accuracy"],
        configuration_space={"x": ["a", "b"]},
        injection_mode="parameter",
    )
    def answer(text: str, config) -> str:
        return "ok"

    await answer.optimize(algorithm="grid")
    assert len(stub.create_calls) == 1
    return _typed_payload_via_session_ops(stub.create_calls[0])


class TestInlineDatasetSentinel:
    def test_generated_inline_name_declares_no_identity(self):
        dataset = load_inline_dataset(_INLINE)
        assert dataset.name == "inline_dataset"  # display name unchanged
        payload = _wire(
            _request(
                dataset_metadata={"size": 1, "name": dataset.name},
                metadata={"evaluation_set": dataset.name},
            )
        )
        assert payload["dataset_id"] is None
        assert payload["dataset_id_source"] == "unknown"

    def test_explicit_id_spelled_inline_dataset_is_sent_verbatim(self):
        payload = _wire(
            _request(
                dataset_id="inline_dataset",
                dataset_metadata={"size": 1, "name": "inline_dataset"},
            )
        )
        assert payload["dataset_id_source"] == "declared"
        assert payload["dataset_id"] == "inline_dataset"

    def test_real_named_dataset_identity_is_byte_identical_to_the_label(self):
        payload = _wire(
            _request(
                dataset_metadata={"size": 1, "name": "support-v1"},
                metadata={"evaluation_set": "support-v1"},
            )
        )
        assert payload["dataset_id"] == "support-v1"


class TestEvaluationOptionsDatasetId:
    def test_strips_and_keeps_the_value(self):
        assert EvaluationOptions(dataset_id="  support-v1 ").dataset_id == "support-v1"

    def test_default_is_none(self):
        assert EvaluationOptions().dataset_id is None

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_blank_is_rejected(self, bad):
        with pytest.raises(Exception, match="dataset_id must be a non-blank string"):
            EvaluationOptions(dataset_id=bad)

    def test_over_255_is_rejected_never_truncated(self):
        with pytest.raises(Exception, match="at most 255 characters"):
            EvaluationOptions(dataset_id="z" * (DATASET_ID_MAX_LENGTH + 1))
        assert (
            EvaluationOptions(dataset_id="z" * DATASET_ID_MAX_LENGTH).dataset_id
            == "z" * DATASET_ID_MAX_LENGTH
        )

    def test_over_255_through_the_decorator_is_rejected(self):
        with pytest.raises(Exception, match="at most 255 characters"):

            @optimize(
                evaluation={"dataset_id": "z" * (DATASET_ID_MAX_LENGTH + 1)},
                configuration_space={"x": ["a", "b"]},
            )
            def answer(text: str) -> str:
                return text

    def test_decorator_threads_it_to_the_optimized_function(self):
        @optimize(
            evaluation={"dataset_id": " support-v1 "},
            configuration_space={"x": ["a", "b"]},
        )
        def answer(text: str) -> str:
            return text

        assert answer.dataset_id == "support-v1"


class TestConnectedGridDecoratorToWire:
    """Public decorator -> orchestrator -> BackendSessionManager -> backend
    client -> SessionOperations -> ApiOperations typed request body."""

    @pytest.mark.asyncio
    async def test_explicit_id_reaches_the_request_body(self, monkeypatch, tmp_path):
        payload = await _public_connected_grid_run(
            monkeypatch,
            tmp_path,
            {"eval_dataset": _INLINE, "dataset_id": "support-v1"},
        )
        assert payload["dataset_id_source"] == "declared"
        assert payload["dataset_id"] == "support-v1"
        # Control plane carries a label/identity, never example content.
        import json

        assert _SECRET_EXAMPLE not in json.dumps(payload, default=str)

    @pytest.mark.asyncio
    async def test_explicit_id_survives_content_change_and_rename(
        self, monkeypatch, tmp_path
    ):
        v1 = Dataset(examples=[_example({"text": "a"}, "A")], name="support")
        v2 = Dataset(
            examples=[_example({"text": "a"}, "A"), _example({"text": "b"}, "B")],
            name="support-renamed",
        )
        (tmp_path / "1").mkdir()
        (tmp_path / "2").mkdir()
        p1 = await _public_connected_grid_run(
            monkeypatch, tmp_path / "1", {"eval_dataset": v1, "dataset_id": "ds-42"}
        )
        p2 = await _public_connected_grid_run(
            monkeypatch, tmp_path / "2", {"eval_dataset": v2, "dataset_id": "ds-42"}
        )
        assert p1["dataset_id"] == p2["dataset_id"] == "ds-42"
        assert p1["dataset_metadata"]["name"] == "support"
        assert p2["dataset_metadata"]["name"] == "support-renamed"

    @pytest.mark.asyncio
    async def test_anonymous_inline_data_sends_no_identity_and_warns_once(
        self, monkeypatch, tmp_path, caplog
    ):
        with caplog.at_level(logging.WARNING):
            payload = await _public_connected_grid_run(
                monkeypatch, tmp_path, {"eval_dataset": _INLINE}
            )
        assert payload["dataset_id"] is None
        assert payload["dataset_id_source"] == "unknown"
        hits = [r for r in caplog.records if "Dataset not linked" in r.getMessage()]
        assert len(hits) == 1
        assert "EvaluationOptions(dataset_id=" in hits[0].getMessage()


def test_two_explicit_ids_stay_distinct_even_with_equal_names():
    a = _wire(_request(dataset_id="ds-a", dataset_metadata={"size": 1, "name": "x"}))
    b = _wire(_request(dataset_id="ds-b", dataset_metadata={"size": 1, "name": "x"}))
    assert a["dataset_id"] == "ds-a"
    assert b["dataset_id"] == "ds-b"


def test_backend_client_submits_declared_evaluator_source_verbatim():
    payload = _typed_payload_via_session_ops(
        {
            "function_name": "qa_agent",
            "search_space": {"model": ["a", "b"]},
            "optimization_goal": "maximize",
            "metadata": {"max_trials": 5, "dataset_size": 1},
            "evaluator_id": "logical-evaluator",
            "evaluator_id_source": "declared",
        }
    )

    assert payload["evaluator_id"] == "logical-evaluator"
    assert payload["evaluator_id_source"] == "declared"


def test_session_operations_threads_dataset_id_to_the_request():
    fake = CapturingFakeClient()
    SessionOperations(cast(Any, fake)).create_session(
        "my_func",
        {"model": ["a", "b"]},
        metadata={"max_trials": 5, "dataset_size": 1, "evaluation_set": "lbl"},
        dataset_id="ds-explicit",
    )
    assert fake.captured_session_request is not None
    assert fake.captured_session_request.dataset_id == "ds-explicit"


class TestManagedPaths:
    @pytest.mark.asyncio
    async def test_interactive_optimizer_request_serializes_the_explicit_id(self):
        from traigent.cloud.client import TraigentCloudClient

        service = Mock(spec=RemoteGuidanceService)
        service.create_session = AsyncMock(
            return_value=SessionCreationResponse(
                session_id="s-1", status="active", optimization_strategy={}
            )
        )
        optimizer = InteractiveOptimizer(
            config_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            remote_service=service,
            dataset_metadata={"size": 1, "name": "inline_dataset"},
            dataset_id="support-v1",
        )
        await optimizer.initialize_session(function_name="qa", max_trials=2)
        request = service.create_session.call_args[0][0]
        stub = SimpleNamespace(_ensure_owner_metadata=lambda m: m or {})
        payload = TraigentCloudClient._serialize_session_request(stub, request)
        assert payload["dataset_id_source"] == "declared"
        assert payload["dataset_id"] == "support-v1"

    @pytest.mark.asyncio
    async def test_interactive_optimizer_inline_name_sends_no_identity(self):
        from traigent.cloud.client import TraigentCloudClient

        service = Mock(spec=RemoteGuidanceService)
        service.create_session = AsyncMock(
            return_value=SessionCreationResponse(
                session_id="s-1", status="active", optimization_strategy={}
            )
        )
        optimizer = InteractiveOptimizer(
            config_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            remote_service=service,
            dataset_metadata={"size": 1, "name": "inline_dataset"},
        )
        await optimizer.initialize_session(function_name="qa", max_trials=2)
        request = service.create_session.call_args[0][0]
        stub = SimpleNamespace(_ensure_owner_metadata=lambda m: m or {})
        payload = TraigentCloudClient._serialize_session_request(stub, request)
        assert payload["dataset_id"] is None
        assert payload["dataset_id_source"] == "unknown"


class TestOfflineSyncIdentity:
    def _offline_manager(self, monkeypatch, tmp_path) -> BackendSessionManager:
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
        monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path / "results"))
        from traigent.optimizers.grid import GridSearchOptimizer

        return BackendSessionManager(
            backend_client=None,
            traigent_config=TraigentConfig(),
            objectives=["accuracy"],
            objective_schema=None,
            optimizer=GridSearchOptimizer({"x": ["a"]}, ["accuracy"]),
            optimization_id="opt-ds",
            optimization_status=OptimizationStatus.RUNNING,
        )

    def _create(self, manager, dataset, dataset_id=None) -> LocalSession:
        def func(text):
            return text

        ctx = manager.create_session(
            func=func,
            dataset=dataset,
            function_descriptor=resolve_function_descriptor(func),
            max_trials=1,
            start_time=time.time(),
            dataset_id=dataset_id,
        )
        storage = LocalStorageManager(manager._traigent_config.get_local_storage_path())
        session = storage.load_session(ctx.session_id)
        assert session is not None
        return session

    def _sync_payload(self, session: LocalSession) -> dict:
        sync = object.__new__(SyncManager)
        return SyncManager.convert_session_to_traigent_format(sync, session)[
            "session_create"
        ]

    def test_explicit_id_is_recorded_and_sent_on_sync(self, monkeypatch, tmp_path):
        manager = self._offline_manager(monkeypatch, tmp_path)
        session = self._create(manager, load_inline_dataset(_INLINE), "support-v1")
        assert session.metadata[DECLARED_DATASET_IDENTITY_METADATA_KEY] == {
            "dataset_id_source": "declared",
            "dataset_id": "support-v1",
        }
        payload = self._sync_payload(session)
        assert payload["dataset_id_source"] == "declared"
        assert payload["dataset_id"] == "support-v1"
        import json

        assert _SECRET_EXAMPLE not in json.dumps(payload, default=str)

    def test_named_dataset_label_is_recorded_like_the_live_path(
        self, monkeypatch, tmp_path
    ):
        manager = self._offline_manager(monkeypatch, tmp_path)
        session = self._create(
            manager, Dataset(examples=[_example({"text": "a"}, "A")], name="support-v1")
        )
        assert self._sync_payload(session)["dataset_id"] == "support-v1"

    def test_anonymous_inline_records_and_sends_no_identity(
        self, monkeypatch, tmp_path
    ):
        manager = self._offline_manager(monkeypatch, tmp_path)
        session = self._create(manager, load_inline_dataset(_INLINE))
        assert DECLARED_DATASET_IDENTITY_METADATA_KEY not in session.metadata
        payload = self._sync_payload(session)
        assert "dataset_id" not in payload
        assert "dataset_id_source" not in payload

    def test_legacy_record_without_identity_sends_none(self):
        legacy = LocalSession(
            session_id="legacy-1",
            function_name="qa",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            status="completed",
            total_trials=0,
            completed_trials=0,
            optimization_config={"search_space": {"x": ["a"]}},
            metadata={"evaluation_set": "support-v1"},
        )
        payload = self._sync_payload(legacy)
        assert "dataset_id" not in payload
        assert "dataset_id_source" not in payload


def test_no_key_fallback_local_record_persists_identity(tmp_path):
    """The no-API-key fallback record carries the identity the live create
    would have declared, so a later sync can send it."""
    storage = LocalStorageManager(str(tmp_path))
    fake = CapturingFakeClient()
    fake.local_storage = storage
    fake.auth_manager.has_api_key = lambda: False
    result = SessionOperations(cast(Any, fake)).create_session(
        "my_func",
        {"model": ["a", "b"]},
        metadata={"max_trials": 5, "dataset_size": 1, "evaluation_set": "lbl"},
        dataset_id="ds-explicit",
    )
    session = storage.load_session(result.session_id)
    assert session is not None
    assert session.metadata[DECLARED_DATASET_IDENTITY_METADATA_KEY] == {
        "dataset_id_source": "declared",
        "dataset_id": "ds-explicit",
    }


# ---------------------------------------------------------------------------
# Input hygiene: two visually identical ids must never be two identities
# ---------------------------------------------------------------------------

_NFC_CAFE = "caf\u00e9"  # composed: e-acute as one code point
_NFD_CAFE = "cafe\u0301"  # decomposed: "e" + combining acute
_ZWSP = "\u200b"


class TestUnicodeNormalization:
    """NFC and NFD spellings of one name are ONE identity, on both routes."""

    def test_explicit_id_nfc_and_nfd_normalize_to_the_same_id(self):
        assert normalize_declared_dataset_id(_NFD_CAFE) == _NFC_CAFE
        assert normalize_declared_dataset_id(_NFC_CAFE) == _NFC_CAFE

    def test_explicit_id_nfc_and_nfd_reach_the_wire_as_one_identity(self):
        composed = _wire(_request(dataset_id=_NFC_CAFE, dataset_metadata={"size": 1}))
        decomposed = _wire(_request(dataset_id=_NFD_CAFE, dataset_metadata={"size": 1}))
        assert composed["dataset_id"] == decomposed["dataset_id"] == _NFC_CAFE

    def test_derived_label_nfc_and_nfd_are_one_identity(self):
        composed = _wire(_request(dataset_metadata={"size": 1, "name": _NFC_CAFE}))
        decomposed = _wire(_request(dataset_metadata={"size": 1, "name": _NFD_CAFE}))
        assert composed["dataset_id"] == decomposed["dataset_id"] == _NFC_CAFE

    def test_explicit_and_derived_routes_share_one_canonical_space(self):
        """Two clients naming one dataset -- one explicitly, one by label --
        must not end up in two cohorts."""
        explicit = _wire(_request(dataset_id=_NFD_CAFE, dataset_metadata={"size": 1}))
        derived = _wire(_request(dataset_metadata={"size": 1, "name": _NFD_CAFE}))
        assert explicit["dataset_id"] == derived["dataset_id"]


class TestInvisibleCharacters:
    """An explicit id is rejected; a user's dataset NAME is cleaned, never fatal."""

    @pytest.mark.parametrize(
        ("bad", "codepoint"),
        [
            ("a\u200bb", "U+200B"),  # zero-width space
            ("a\nb\tc", "U+000A"),  # C0 controls
            ("a\ufeffb", "U+FEFF"),  # BOM / zero-width no-break space
            ("a\u200eb", "U+200E"),  # BiDi mark
            ("a\u0085b", "U+0085"),  # C1 control
        ],
    )
    def test_explicit_id_with_an_invisible_character_is_rejected(self, bad, codepoint):
        with pytest.raises(ValueError, match="control or zero-width characters"):
            normalize_declared_dataset_id(bad)
        with pytest.raises(ValueError, match=re.escape(codepoint)):
            normalize_declared_dataset_id(bad)

    def test_a_zero_width_only_explicit_id_is_rejected_not_kept_whole(self):
        with pytest.raises(ValueError, match="control or zero-width characters"):
            SessionCreationRequest(function_name="f", dataset_id=_ZWSP)

    def test_explicit_id_rejection_reaches_the_public_option(self):
        with pytest.raises(Exception, match="control or zero-width characters"):
            EvaluationOptions(dataset_id="support" + _ZWSP + "-v1")

    def test_derived_label_with_a_zero_width_char_is_cleaned_not_rejected(self):
        payload = _wire(
            _request(dataset_metadata={"size": 1, "name": "sup" + _ZWSP + "port-v1"})
        )
        assert payload["dataset_id"] == "support-v1"

    def test_derived_label_of_only_invisible_chars_declares_no_identity(self):
        payload = _wire(
            _request(dataset_metadata={"size": 1, "name": _ZWSP + "\ufeff"})
        )
        assert payload["dataset_id"] is None
        assert payload["dataset_id_source"] == "unknown"

    def test_cleaning_happens_before_the_sentinel_check(self):
        """An invisibly-decorated placeholder is still the placeholder."""
        payload = _wire(
            _request(
                dataset_metadata={"size": 1, "name": "inline" + _ZWSP + "_dataset"}
            )
        )
        assert payload["dataset_id"] is None
        assert payload["dataset_id_source"] == "unknown"

    def test_a_cleaned_label_groups_with_its_plain_spelling(self):
        decorated = _wire(
            _request(dataset_metadata={"size": 1, "name": "supp" + _ZWSP + "ort"})
        )
        plain = _wire(_request(dataset_metadata={"size": 1, "name": "support"}))
        assert decorated["dataset_id"] == plain["dataset_id"] == "support"


class TestCapIsAppliedAfterNormalization:
    def test_255_code_points_after_normalization_is_accepted(self):
        """The decomposed spelling is 510 code points and composes to 255, so
        measuring before normalizing would reject a legal id."""
        composed = _NFC_CAFE[-1] * DATASET_ID_MAX_LENGTH
        decomposed = unicodedata.normalize("NFD", composed)
        assert len(decomposed) > DATASET_ID_MAX_LENGTH
        normalized = normalize_declared_dataset_id(decomposed)
        assert normalized == composed
        assert len(normalized) == DATASET_ID_MAX_LENGTH

    def test_256_code_points_after_normalization_is_rejected(self):
        with pytest.raises(ValueError, match="at most 255 characters"):
            normalize_declared_dataset_id(_NFC_CAFE[-1] * (DATASET_ID_MAX_LENGTH + 1))


class TestPlainAsciiIsUnchanged:
    """No reminting: an existing plain label keeps byte-identical identity."""

    def test_explicit_plain_id_is_untouched(self):
        assert normalize_declared_dataset_id("support-v1") == "support-v1"

    def test_derived_plain_label_is_untouched(self):
        payload = _wire(_request(dataset_metadata={"size": 1, "name": "support-v1"}))
        assert payload["dataset_id"] == "support-v1"

    def test_explicit_still_wins_and_is_never_folded(self):
        payload = _wire(
            _request(
                dataset_id="explicit-id",
                dataset_metadata={"size": 1, "name": "label-derived-name"},
            )
        )
        assert payload["dataset_id"] == "explicit-id"

    def test_sentinels_remain_case_insensitive(self):
        payload = _wire(
            _request(dataset_metadata={"size": 1, "name": "Inline_Dataset"})
        )
        assert payload["dataset_id"] is None
        assert payload["dataset_id_source"] == "unknown"
