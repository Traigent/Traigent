"""Unit tests for warm_start_from threading through session creation."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.models import OptimizationSession, SessionCreationRequest
from traigent.cloud.session_operations import SessionOperations
from traigent.core.backend_session_manager import BackendSessionManager

# ---------------------------------------------------------------------------
# Shared FakeClient infrastructure (mirrors test_session_operations_validation.py)
# ---------------------------------------------------------------------------


class TrackingLock:
    def __init__(self) -> None:
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeSessionBridge:
    def create_session_mapping(self, **kwargs) -> None:
        return None

    def get_session_mapping(self, session_id: str):
        return SimpleNamespace(experiment_run_id="run-1")

    _session_mappings: dict[str, Any] = {}


class FakeAuth:
    async def get_headers(self) -> dict[str, str]:
        return {}


class FakeAuthManager:
    def __init__(self) -> None:
        self.auth = FakeAuth()

    def has_api_key(self) -> bool:
        return True


class CapturingFakeClient:
    """FakeClient that records the SessionCreationRequest passed to the API."""

    def __init__(self) -> None:
        self._active_sessions_lock = TrackingLock()
        self._active_sessions: dict[str, OptimizationSession] = {}
        self._max_active_sessions = 5
        self.session_bridge = FakeSessionBridge()
        self.backend_config = SimpleNamespace(api_base_url=None, backend_base_url=None)
        self.auth_manager = FakeAuthManager()
        self._register_security_session = MagicMock()
        self.local_storage = None
        self.captured_session_request: SessionCreationRequest | None = None

    async def _ensure_session(self):
        return SimpleNamespace(post=AsyncMock(), get=AsyncMock())

    async def _create_traigent_session_via_api(
        self, session_request: SessionCreationRequest
    ):
        self.captured_session_request = session_request
        return ("session-001", "exp-001", "run-001")

    def _revoke_security_session(self, *args, **kwargs) -> None:
        return None


@pytest.fixture(autouse=True)
def _offline_disabled(monkeypatch):
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_OFFLINE", "false")


# ---------------------------------------------------------------------------
# Tests: _build_typed_session_payload includes / omits warm_start_from
# ---------------------------------------------------------------------------


def _make_api_ops() -> ApiOperations:
    return ApiOperations(MagicMock())


def _typed_request(**kwargs: Any) -> SessionCreationRequest:
    """Minimal valid SessionCreationRequest for payload-builder tests."""
    defaults: dict[str, Any] = {
        "function_name": "test_func",
        "configuration_space": {"param": [1, 2, 3]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 1},
        "promotion_policy": None,
        "tvl_governance": None,
    }
    defaults.update(kwargs)
    return SessionCreationRequest(**defaults)


class TestBuildTypedSessionPayloadWarmStart:
    """_build_typed_session_payload warm_start_from contract."""

    def test_warm_start_from_included_when_set(self):
        ops = _make_api_ops()
        request = _typed_request(warm_start_from="exp_abc123")
        payload = ops._build_typed_session_payload(request, max_trials=5)
        assert payload["warm_start_from"] == "exp_abc123"

    def test_warm_start_from_omitted_when_none(self):
        ops = _make_api_ops()
        request = _typed_request(warm_start_from=None)
        payload = ops._build_typed_session_payload(request, max_trials=5)
        assert "warm_start_from" not in payload

    def test_warm_start_from_omitted_when_empty_string(self):
        """Empty string must not emit the key (falsy guard in payload builder)."""
        ops = _make_api_ops()
        request = _typed_request()
        request.warm_start_from = ""
        payload = ops._build_typed_session_payload(request, max_trials=5)
        assert "warm_start_from" not in payload

    def test_no_regression_baseline_payload_omits_warm_start_key(self):
        """Normal session without warm_start_from must not include the key."""
        ops = _make_api_ops()
        request = _typed_request()
        payload = ops._build_typed_session_payload(request, max_trials=10)
        assert "warm_start_from" not in payload
        # Confirm required typed-contract keys are still present
        assert "function_name" in payload
        assert "configuration_space" in payload
        assert "objectives" in payload
        assert "max_trials" in payload


# ---------------------------------------------------------------------------
# Tests: SessionCreationRequest.warm_start_from field
# ---------------------------------------------------------------------------


class TestSessionCreationRequestField:
    def test_default_is_none(self):
        req = SessionCreationRequest(function_name="f")
        assert req.warm_start_from is None

    def test_field_round_trips(self):
        req = SessionCreationRequest(function_name="f", warm_start_from="exp_xyz")
        assert req.warm_start_from == "exp_xyz"


# ---------------------------------------------------------------------------
# Tests: SessionOperations.create_session threads warm_start_from
# ---------------------------------------------------------------------------


class TestSessionOperationsWarmStartThreading:
    """create_session passes warm_start_from into the SessionCreationRequest."""

    def _make_ops(self) -> tuple[SessionOperations, CapturingFakeClient]:
        client = CapturingFakeClient()
        return SessionOperations(client), client

    def test_warm_start_from_threads_to_session_request(self):
        ops, client = self._make_ops()
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
            warm_start_from="exp_prior_123",
        )
        assert client.captured_session_request is not None
        assert client.captured_session_request.warm_start_from == "exp_prior_123"

    def test_no_warm_start_leaves_field_none(self):
        ops, client = self._make_ops()
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
        )
        assert client.captured_session_request is not None
        assert client.captured_session_request.warm_start_from is None

    def test_empty_string_warm_start_treated_as_none(self):
        """Empty string must normalise to None before reaching SessionCreationRequest."""
        ops, client = self._make_ops()
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 3, "dataset_size": 5, "evaluation_set": "test"},
            warm_start_from="",
        )
        assert client.captured_session_request is not None
        assert client.captured_session_request.warm_start_from is None


# ---------------------------------------------------------------------------
# Tests: LegacyOptimizeArgs recognises warm_start_from (no TypeError)
# ---------------------------------------------------------------------------


class TestLegacyOptimizeArgsWarmStart:
    """warm_start_from is a known direct option, so the legacy bundle accepts it."""

    def test_from_mapping_round_trips_warm_start(self):
        from traigent.api.decorators import LegacyOptimizeArgs

        # Must not raise TypeError: warm_start_from is in _DIRECT_OPTION_KEYS
        # (via _OPTIMIZE_DEFAULTS) so from_mapping routes it to a real field,
        # not to ``extra``.
        args = LegacyOptimizeArgs.from_mapping({"warm_start_from": "exp_1"})
        assert args.warm_start_from == "exp_1"
        assert "warm_start_from" not in args.extra

    def test_default_is_none(self):
        from traigent.api.decorators import LegacyOptimizeArgs

        assert LegacyOptimizeArgs().warm_start_from is None

    def test_iter_known_values_includes_warm_start(self):
        from traigent.api.decorators import LegacyOptimizeArgs

        args = LegacyOptimizeArgs(warm_start_from="exp_2")
        known = dict(args.iter_known_values())
        assert "warm_start_from" in known
        assert known["warm_start_from"] == "exp_2"


# ---------------------------------------------------------------------------
# Tests: orchestrator result-metadata assembly surfaces warm_start_from
# ---------------------------------------------------------------------------


class TestResultMetadataWarmStartProvenance:
    """_build_result_metadata records warm_start_from on the returned result."""

    def _call_build(self, monkeypatch, warm_start_from):
        import traigent.core.orchestrator as orch_mod
        from traigent.core.orchestrator import OptimizationOrchestrator

        # Neutralise offline detection so traigent_config value is irrelevant.
        monkeypatch.setattr(orch_mod, "policy_from_config", lambda _cfg: None)
        monkeypatch.setattr(orch_mod, "is_offline_requested", lambda _policy: False)

        fake_self = SimpleNamespace(
            _warm_start_from=warm_start_from,
            _function_descriptor=None,
            traigent_config=None,
        )
        return OptimizationOrchestrator._build_result_metadata(
            fake_self,
            session_summary=None,
            safeguards_telemetry={},
        )

    def test_metadata_includes_warm_start_when_set(self, monkeypatch):
        metadata = self._call_build(monkeypatch, "exp_prior_99")
        assert metadata["warm_start_from"] == "exp_prior_99"

    def test_metadata_omits_warm_start_when_unset(self, monkeypatch):
        metadata = self._call_build(monkeypatch, None)
        assert "warm_start_from" not in metadata

    def test_metadata_omits_warm_start_when_empty(self, monkeypatch):
        metadata = self._call_build(monkeypatch, "")
        assert "warm_start_from" not in metadata


class TestWarmStartTransferMetadataPassthrough:
    def _manager(self):
        manager = BackendSessionManager.__new__(BackendSessionManager)
        manager._backend_client = None
        manager._egress_disabled = lambda: False
        manager._session_owning_context = {}
        return manager

    def test_backend_warm_start_transfer_metadata_copied_verbatim(self):
        transfer_metadata = {
            "transfer_mode": "accepted",
            "final_warm_start_weight": "medium",
            "search_space_overlap": "partial",
            "n_seed_configs_applied": 2,
            "refused_reason": None,
        }
        result = SimpleNamespace(metadata={"warm_start_from": "exp_prior_99"})

        self._manager().attach_session_metadata(
            result=result,
            session_id="session-123",
            session_summary={"metadata": {"warm_start_transfer": transfer_metadata}},
        )

        assert result.metadata["warm_start_transfer"] == transfer_metadata
        assert result.metadata["warm_start_transfer"] is not transfer_metadata
        assert result.metadata["warm_start_from"] == "exp_prior_99"
        field_names = json.dumps(
            sorted(result.metadata["warm_start_transfer"].keys())
        ).lower()
        assert "score" not in field_names
        assert "signal" not in field_names

    @pytest.mark.parametrize(
        "session_summary",
        [
            {},
            {"metadata": {}},
            {"metadata": {"warm_start_transfer": None}},
            {"metadata": {"warm_start_transfer": "opaque"}},
        ],
    )
    def test_warm_start_transfer_absent_or_non_dict_adds_no_key(self, session_summary):
        result = SimpleNamespace(metadata={})

        self._manager().attach_session_metadata(
            result=result,
            session_id="session-123",
            session_summary=session_summary,
        )

        assert "warm_start_transfer" not in result.metadata

    def test_warm_start_transfer_extracted_from_finalization_response_object(self):
        # Real cloud path: finalize_session returns an OptimizationFinalizationResponse
        # dataclass that exposes ``.metadata`` as an ATTRIBUTE and has no ``.get()``.
        # Regression: calling ``.get()`` on it raised AttributeError, which the
        # persistence try/except swallowed as "persistence failed" on every cloud run
        # (also nulling experiment_id). Must read ``.metadata`` via attribute instead.
        transfer_metadata = {
            "transfer_mode": "replay_only",
            "final_warm_start_weight": "low",
            "search_space_overlap": "high",
            "n_seed_configs_applied": 3,
            "refused_reason": None,
        }
        finalization_response = SimpleNamespace(
            metadata={"warm_start_transfer": transfer_metadata}
        )
        assert not hasattr(finalization_response, "get")  # like the real dataclass
        result = SimpleNamespace(metadata={})

        self._manager().attach_session_metadata(
            result=result,
            session_id="session-123",
            session_summary=finalization_response,
        )

        assert result.metadata["warm_start_transfer"] == transfer_metadata
        assert result.metadata["warm_start_transfer"] is not transfer_metadata

    def test_session_summary_object_without_metadata_attr_does_not_crash(self):
        no_meta = SimpleNamespace()  # neither .get nor .metadata
        result = SimpleNamespace(metadata={})

        self._manager().attach_session_metadata(
            result=result,
            session_id="session-123",
            session_summary=no_meta,
        )

        assert "warm_start_transfer" not in result.metadata


# ---------------------------------------------------------------------------
# Tests: EVERY SessionCreationRequest -> /sessions payload serializer must
# emit warm_start_from. There are THREE live serializers (all asserted here):
#   1. ApiOperations._build_typed_session_payload   (typed contract, default)
#   2. ApiOperations._build_legacy_session_payload  (TRAIGENT_SESSION_CONTRACT=
#      legacy + auto-contract retry for non-governed sessions)
#   3. TraigentCloudClient._serialize_session_request (cloud-client path)
# A field set by the user must not silently die on whichever path a run takes.
# ---------------------------------------------------------------------------


def _serialize_typed(warm_start_from):
    ops = ApiOperations(MagicMock())
    request = SessionCreationRequest(
        function_name="f",
        configuration_space={"p": [1, 2]},
        objectives=["accuracy"],
        dataset_metadata={"size": 1},
        warm_start_from=warm_start_from,
    )
    return ops._build_typed_session_payload(request, max_trials=5)


def _serialize_legacy(warm_start_from):
    ops = ApiOperations(MagicMock())
    request = SessionCreationRequest(
        function_name="f",
        configuration_space={"p": [1, 2]},
        objectives=["accuracy"],
        dataset_metadata={"size": 1},
        warm_start_from=warm_start_from,
    )
    return ops._build_legacy_session_payload(request, max_trials=5)


def _serialize_cloud_client(warm_start_from):
    from traigent.cloud.client import TraigentCloudClient

    # Lightweight fake self: stub the two helpers the serializer calls so we
    # exercise only the payload-assembly logic (no live client construction).
    fake_self = SimpleNamespace(
        _ensure_owner_metadata=lambda metadata: metadata or {},
        _serialize_session_objective=lambda objective: objective,
    )
    request = SessionCreationRequest(
        function_name="f",
        configuration_space={"p": [1, 2]},
        objectives=["accuracy"],
        dataset_metadata={"size": 1},
        warm_start_from=warm_start_from,
    )
    return TraigentCloudClient._serialize_session_request(fake_self, request)


_ALL_SERIALIZERS = [
    pytest.param(_serialize_typed, id="typed"),
    pytest.param(_serialize_legacy, id="legacy"),
    pytest.param(_serialize_cloud_client, id="cloud_client"),
]


class TestAllSerializerPathsWarmStart:
    """Every live /sessions serializer emits/omits warm_start_from consistently."""

    @pytest.mark.parametrize("serialize", _ALL_SERIALIZERS)
    def test_includes_warm_start_when_set(self, serialize):
        payload = serialize("exp_drop")
        assert payload["warm_start_from"] == "exp_drop"

    @pytest.mark.parametrize("serialize", _ALL_SERIALIZERS)
    def test_omits_warm_start_when_none(self, serialize):
        payload = serialize(None)
        assert "warm_start_from" not in payload

    @pytest.mark.parametrize("serialize", _ALL_SERIALIZERS)
    def test_omits_warm_start_when_empty_string(self, serialize):
        payload = serialize("")
        assert "warm_start_from" not in payload
