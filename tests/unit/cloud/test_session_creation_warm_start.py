"""Unit tests for warm_start_from threading through session creation."""

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from traigent.cloud.api_operations import ApiOperations, TraigentSessionApiResult
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

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
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
# Tests: smart_pruning threading through session creation
# ---------------------------------------------------------------------------


SMART_PRUNING = {
    "label": "balanced",
    "warmup_steps": 1,
    "confidence": 0.8,
}


class TestSessionCreationSmartPruning:
    def test_request_field_round_trips(self):
        req = SessionCreationRequest(function_name="f", smart_pruning=SMART_PRUNING)
        assert req.smart_pruning == SMART_PRUNING

    def test_request_rejects_extra_smart_pruning_field(self):
        with pytest.raises(ValueError, match="smart_pruning.*unexpected"):
            SessionCreationRequest(
                function_name="f",
                smart_pruning={**SMART_PRUNING, "unexpected": "value"},
            )

    def test_typed_payload_includes_top_level_smart_pruning(self):
        ops = _make_api_ops()
        request = _typed_request(smart_pruning=SMART_PRUNING)
        payload = ops._build_typed_session_payload(request, max_trials=5)
        assert payload["smart_pruning"] == SMART_PRUNING

    def test_typed_payload_rejects_mutated_extra_smart_pruning_field(self):
        ops = _make_api_ops()
        request = _typed_request(smart_pruning=SMART_PRUNING)
        request.smart_pruning = {**SMART_PRUNING, "unexpected": "value"}

        with pytest.raises(ValueError, match="smart_pruning.*unexpected"):
            ops._build_typed_session_payload(request, max_trials=5)

    def test_legacy_payload_includes_top_level_smart_pruning(self):
        ops = _make_api_ops()
        request = _typed_request(smart_pruning=SMART_PRUNING)
        payload = ops._build_legacy_session_payload(request, max_trials=5)
        assert payload["smart_pruning"] == SMART_PRUNING

    def test_legacy_payload_rejects_mutated_extra_smart_pruning_field(self):
        ops = _make_api_ops()
        request = _typed_request(smart_pruning=SMART_PRUNING)
        request.smart_pruning = {**SMART_PRUNING, "unexpected": "value"}

        with pytest.raises(ValueError, match="smart_pruning.*unexpected"):
            ops._build_legacy_session_payload(request, max_trials=5)

    def test_cloud_client_serializer_includes_top_level_smart_pruning(self):
        from traigent.cloud.client import TraigentCloudClient

        fake_self = SimpleNamespace(
            _ensure_owner_metadata=lambda metadata: metadata or {},
            _serialize_session_objective=lambda objective: objective,
        )
        request = SessionCreationRequest(
            function_name="f",
            configuration_space={"p": [1, 2]},
            objectives=["accuracy"],
            dataset_metadata={"size": 1},
            smart_pruning=SMART_PRUNING,
        )

        payload = TraigentCloudClient._serialize_session_request(fake_self, request)

        assert payload["smart_pruning"] == SMART_PRUNING

    def test_cloud_client_serializer_rejects_mutated_extra_smart_pruning_field(self):
        from traigent.cloud.client import TraigentCloudClient

        fake_self = SimpleNamespace(
            _ensure_owner_metadata=lambda metadata: metadata or {},
            _serialize_session_objective=lambda objective: objective,
        )
        request = SessionCreationRequest(
            function_name="f",
            configuration_space={"p": [1, 2]},
            objectives=["accuracy"],
            dataset_metadata={"size": 1},
            smart_pruning=SMART_PRUNING,
        )
        request.smart_pruning = {**SMART_PRUNING, "unexpected": "value"}

        with pytest.raises(ValueError, match="smart_pruning.*unexpected"):
            TraigentCloudClient._serialize_session_request(fake_self, request)


# ---------------------------------------------------------------------------
# Tests: SessionOperations.create_session threads warm_start_from
# ---------------------------------------------------------------------------


class TestSessionOperationsWarmStartThreading:
    """create_session passes warm_start_from into the SessionCreationRequest."""

    def _make_ops(self) -> tuple[SessionOperations, CapturingFakeClient]:
        client = CapturingFakeClient()
        return SessionOperations(cast(Any, client)), client

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

    def test_smart_pruning_threads_to_session_request(self):
        ops, client = self._make_ops()
        ops.create_session(
            "my_func",
            {"model": ["a", "b"]},
            metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
            smart_pruning=SMART_PRUNING,
        )
        assert client.captured_session_request is not None
        assert client.captured_session_request.smart_pruning == SMART_PRUNING

    def test_smart_pruning_rejects_extra_field(self):
        ops, _client = self._make_ops()

        with pytest.raises(ValueError, match="smart_pruning.*unexpected"):
            ops.create_session(
                "my_func",
                {"model": ["a", "b"]},
                metadata={
                    "max_trials": 5,
                    "dataset_size": 10,
                    "evaluation_set": "test",
                },
                smart_pruning={**SMART_PRUNING, "unexpected": "value"},
            )


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


# ---------------------------------------------------------------------------
# Tests (#1683 Bug B): the backend communicates the warm-start decision in the
# session-CREATE response; the SDK must retain it and surface it in result
# metadata instead of reporting cold-start defaults. Finalize-time block wins;
# the CREATE-time block fills the gap when finalize carries none.
# ---------------------------------------------------------------------------


CREATE_TRANSFER = {
    "transfer_mode": "accepted",
    "final_warm_start_weight": "medium",
    "search_space_overlap": "identical",
    "n_seed_configs_applied": 3,
    "refused_reason": None,
}

FINALIZE_TRANSFER = {
    "transfer_mode": "replay_only",
    "final_warm_start_weight": "low",
    "search_space_overlap": "partial",
    "n_seed_configs_applied": 1,
    "refused_reason": None,
}


class FakeHttpResponse:
    """Minimal aiohttp-response stand-in exposing async ``json()``."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def json(self) -> dict[str, Any]:
        return self._payload


class TestParseSessionResponseWarmStartTransfer:
    """_parse_session_response retains the CREATE-time warm-start block."""

    def _parse(self, payload: dict[str, Any]) -> TraigentSessionApiResult:
        ops = _make_api_ops()
        return asyncio.run(ops._parse_session_response(FakeHttpResponse(payload)))

    def test_top_level_block_retained_verbatim(self):
        result = self._parse(
            {"session_id": "s1", "warm_start_transfer": CREATE_TRANSFER}
        )
        assert result.warm_start_transfer == CREATE_TRANSFER

    def test_metadata_block_retained_verbatim(self):
        result = self._parse(
            {
                "session_id": "s1",
                "metadata": {
                    "experiment_id": "e1",
                    "experiment_run_id": "r1",
                    "warm_start_transfer": CREATE_TRANSFER,
                },
            }
        )
        assert result.warm_start_transfer == CREATE_TRANSFER

    def test_top_level_wins_over_metadata(self):
        result = self._parse(
            {
                "session_id": "s1",
                "warm_start_transfer": CREATE_TRANSFER,
                "metadata": {"warm_start_transfer": FINALIZE_TRANSFER},
            }
        )
        assert result.warm_start_transfer == CREATE_TRANSFER

    @pytest.mark.parametrize("bad_value", [None, "opaque", 3, ["x"]])
    def test_absent_or_non_dict_block_is_none(self, bad_value):
        payload: dict[str, Any] = {"session_id": "s1"}
        if bad_value is not None:
            payload["warm_start_transfer"] = bad_value
        result = self._parse(payload)
        assert result.warm_start_transfer is None

    def test_result_still_unpacks_as_three_tuple(self):
        result = self._parse(
            {"session_id": "s1", "warm_start_transfer": CREATE_TRANSFER}
        )
        session_id, experiment_id, experiment_run_id = result
        assert session_id == "s1"
        assert experiment_id == "s1"  # metadata fallback
        assert experiment_run_id == "s1"


class WarmStartCreateFakeClient(CapturingFakeClient):
    """FakeClient whose CREATE response carries a warm_start_transfer block."""

    def __init__(self, warm_start_transfer: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._warm_start_transfer = warm_start_transfer

    async def _create_traigent_session_via_api(
        self, session_request: SessionCreationRequest
    ):
        self.captured_session_request = session_request
        return TraigentSessionApiResult(
            "session-001",
            "exp-001",
            "run-001",
            warm_start_transfer=self._warm_start_transfer,
        )


def _create_session(
    client: CapturingFakeClient,
) -> SessionOperations:
    ops = SessionOperations(cast(Any, client))
    ops.create_session(
        "my_func",
        {"model": ["a", "b"]},
        metadata={"max_trials": 5, "dataset_size": 10, "evaluation_set": "test"},
        warm_start_from="exp_prior_123",
    )
    return ops


class TestCreateResponseWarmStartTransferRetention:
    """create_session retains the CREATE-time block in session metadata."""

    def test_block_stored_verbatim_copy_in_active_session_metadata(self):
        client = WarmStartCreateFakeClient(warm_start_transfer=CREATE_TRANSFER)
        _create_session(client)
        stored = client._active_sessions["session-001"].metadata[
            "warm_start_transfer"
        ]
        assert stored == CREATE_TRANSFER
        assert stored is not CREATE_TRANSFER  # defensive copy, still verbatim

    def test_no_block_adds_no_key(self):
        client = WarmStartCreateFakeClient(warm_start_transfer=None)
        _create_session(client)
        assert (
            "warm_start_transfer"
            not in client._active_sessions["session-001"].metadata
        )

    def test_plain_tuple_create_result_still_supported(self):
        """Fakes/legacy clients returning a bare 3-tuple must keep working."""
        client = CapturingFakeClient()
        _create_session(client)
        assert (
            "warm_start_transfer"
            not in client._active_sessions["session-001"].metadata
        )


class TestFinalizeMergesCreateTimeWarmStartTransfer:
    """Finalize surfaces the CREATE-time block; a finalize-time block wins."""

    def _finalize(
        self,
        create_block: dict[str, Any] | None,
        finalize_payload: dict[str, Any] | None,
    ):
        client = WarmStartCreateFakeClient(warm_start_transfer=create_block)
        ops = _create_session(client)
        ops._finalize_session_via_api = AsyncMock(return_value=finalize_payload)  # type: ignore[method-assign]
        return asyncio.run(ops.finalize_session("session-001"))

    def test_create_time_block_fills_gap_when_finalize_has_none(self):
        response = self._finalize(CREATE_TRANSFER, {})
        assert response.metadata["warm_start_transfer"] == CREATE_TRANSFER

    def test_finalize_time_block_wins_over_create_time_block(self):
        response = self._finalize(
            CREATE_TRANSFER,
            {"metadata": {"warm_start_transfer": FINALIZE_TRANSFER}},
        )
        assert response.metadata["warm_start_transfer"] == FINALIZE_TRANSFER

    def test_finalize_only_block_still_surfaced(self):
        response = self._finalize(
            None, {"metadata": {"warm_start_transfer": FINALIZE_TRANSFER}}
        )
        assert response.metadata["warm_start_transfer"] == FINALIZE_TRANSFER

    def test_no_block_anywhere_adds_no_key(self):
        response = self._finalize(None, {})
        assert "warm_start_transfer" not in response.metadata

    def test_create_time_block_propagates_to_result_metadata_verbatim(self):
        """Full chain: CREATE response -> finalize response ->
        attach_session_metadata -> result.metadata (verbatim contract)."""
        response = self._finalize(CREATE_TRANSFER, {})

        manager = BackendSessionManager.__new__(BackendSessionManager)
        manager._backend_client = None
        manager._egress_disabled = lambda: False
        manager._session_owning_context = {}
        result = SimpleNamespace(metadata={"warm_start_from": "exp_prior_123"})

        manager.attach_session_metadata(
            result=cast(Any, result),
            session_id="session-001",
            session_summary=cast(Any, response),
        )

        assert result.metadata["warm_start_transfer"] == CREATE_TRANSFER
        assert result.metadata["warm_start_transfer"] is not CREATE_TRANSFER


# ---------------------------------------------------------------------------
# Tests (#1683 task 3): loud refusal — explicit warm_start_from + 0 seeds
# applied (or refused_reason set) must emit a logging.warning with aggregate
# info only; silent otherwise.
# ---------------------------------------------------------------------------


class TestWarmStartRefusalWarning:
    def _manager(self):
        manager = BackendSessionManager.__new__(BackendSessionManager)
        manager._backend_client = None
        manager._egress_disabled = lambda: False
        manager._session_owning_context = {}
        return manager

    def _attach(self, caplog, *, warm_start_from, transfer):
        result_metadata: dict[str, Any] = {}
        if warm_start_from is not None:
            result_metadata["warm_start_from"] = warm_start_from
        result = SimpleNamespace(metadata=result_metadata)
        summary = (
            {"metadata": {"warm_start_transfer": transfer}}
            if transfer is not None
            else {"metadata": {}}
        )
        with caplog.at_level(logging.WARNING):
            self._manager().attach_session_metadata(
                result=cast(Any, result),
                session_id="session-123",
                session_summary=summary,
            )
        return [
            record
            for record in caplog.records
            if "warm_start_from" in record.getMessage()
        ]

    def test_warns_on_refusal_with_reason_and_prior_experiment_id(self, caplog):
        refused = {
            "transfer_mode": "refused",
            "search_space_overlap": "unknown",
            "n_seed_configs_applied": 0,
            "refused_reason": "no_seed_configs",
        }
        records = self._attach(
            caplog, warm_start_from="exp_prior_99", transfer=refused
        )
        assert len(records) == 1
        message = records[0].getMessage()
        assert "no_seed_configs" in message
        assert "exp_prior_99" in message
        # Aggregate info only: never seed contents/scores/signals.
        assert "score" not in message.lower().replace("seed configs", "")

    def test_warns_on_zero_seeds_even_without_refused_reason(self, caplog):
        transfer = {"n_seed_configs_applied": 0, "refused_reason": None}
        records = self._attach(
            caplog, warm_start_from="exp_prior_99", transfer=transfer
        )
        assert len(records) == 1

    def test_silent_when_seeds_applied(self, caplog):
        transfer = {"n_seed_configs_applied": 2, "refused_reason": None}
        records = self._attach(
            caplog, warm_start_from="exp_prior_99", transfer=transfer
        )
        assert records == []

    def test_silent_when_warm_start_not_requested(self, caplog):
        refused = {"n_seed_configs_applied": 0, "refused_reason": "no_seed_configs"}
        records = self._attach(caplog, warm_start_from=None, transfer=refused)
        assert records == []

    def test_silent_when_no_transfer_block(self, caplog):
        records = self._attach(
            caplog, warm_start_from="exp_prior_99", transfer=None
        )
        assert records == []
