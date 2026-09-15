"""Tests for G1 §1: nullable run identity + require_run_id early failure.

Covers the acceptance items from the T-SDK brief:
1. A missing/null/empty experiment_run_id / experiment_id stays None,
   independently, with exactly one WARNING per create response and no id
   values in the message.
2. Finalize is session-addressed and omits experiment_run_id when None.
3. Resume never persists a substituted id (display label only).
4. require_run_id=True (option or env) fails closed at session-create time,
   before any trial.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from traigent.api.types import OptimizationStatus
from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.backend_client import BackendIntegratedClient
import traigent.cloud.client as _client_module
from traigent.cloud.session_types import (
    SessionCreationFailureReason,
    SessionCreationResult,
)
from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig
from traigent.core.backend_session_manager import BackendSessionManager
from traigent.core.objectives import create_default_objectives
from traigent.core.orchestrator import OptimizationOrchestrator
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
from traigent.optimizers.base import BaseOptimizer
from traigent.utils.function_identity import resolve_function_descriptor

# SDK #2033: opt into the connected/backend code paths (see pyproject markers).
pytestmark = pytest.mark.backend_online


class _RunIdMissingErrorNotYetDefined(Exception):
    """Sentinel substituted for RunIdMissingError when it does not exist yet.

    Resolved lazily (not imported at module scope) so a pre-implementation
    run of this file shows a real `pytest.raises` failure ("DID NOT RAISE")
    for every require_run_id assertion, instead of an ImportError that would
    crash collection for the whole module and hide the item-1/2/3 evidence.
    """


def _run_id_missing_error_cls() -> type[Exception]:
    return getattr(_client_module, "RunIdMissingError", _RunIdMissingErrorNotYetDefined)


def _backend_response(payload: dict[str, Any], status_code: int = 201):
    response = Mock(status_code=status_code, text="ok")
    response.json.return_value = payload
    return response


# ---------------------------------------------------------------------------
# Item 1: ingress normalization + single warning (api_operations)
# ---------------------------------------------------------------------------


class TestIngressNormalizationApiOperations:
    def setup_method(self):
        self.ops = ApiOperations(Mock())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw_run_id", [None, "", "   "])
    async def test_missing_run_id_variants_stay_none(self, raw_run_id, caplog):
        """Missing/null/empty/whitespace experiment_run_id -> None, one warning."""
        metadata: dict[str, Any] = {"experiment_id": "exp-1"}
        if raw_run_id is not None:
            metadata["experiment_run_id"] = raw_run_id
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(
            return_value={"session_id": "sess-1", "metadata": metadata}
        )

        with caplog.at_level(logging.WARNING, logger="traigent.cloud.api_operations"):
            result = await self.ops._parse_session_response(mock_response)

        session_id, exp_id, run_id = result
        assert session_id == "sess-1"
        assert exp_id == "exp-1"
        assert run_id is None

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "sess-1" not in message
        assert "exp-1" not in message

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw_exp_id", [None, "", "   "])
    async def test_missing_experiment_id_keeps_present_run_id(self, raw_exp_id, caplog):
        """A present run id survives even when experiment_id is absent."""
        metadata: dict[str, Any] = {"experiment_run_id": "run-1"}
        if raw_exp_id is not None:
            metadata["experiment_id"] = raw_exp_id
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(
            return_value={"session_id": "sess-2", "metadata": metadata}
        )

        with caplog.at_level(logging.WARNING, logger="traigent.cloud.api_operations"):
            result = await self.ops._parse_session_response(mock_response)

        session_id, exp_id, run_id = result
        assert exp_id is None
        assert run_id == "run-1"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_both_ids_present_no_warning(self, caplog):
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "sess-3",
                "metadata": {"experiment_id": "exp-3", "experiment_run_id": "run-3"},
            }
        )

        with caplog.at_level(logging.WARNING, logger="traigent.cloud.api_operations"):
            result = await self.ops._parse_session_response(mock_response)

        assert tuple(result) == ("sess-3", "exp-3", "run-3")
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# ---------------------------------------------------------------------------
# Item 1: ingress normalization + single warning (sync_manager historical
# import) + the require_run_id env gate on the historical-import path.
# ---------------------------------------------------------------------------


class TestIngressNormalizationSyncManager:
    @pytest.fixture
    def sync_manager(self, tmp_path: Path) -> SyncManager:
        config = MagicMock(spec=TraigentConfig)
        config.get_local_storage_path.return_value = str(tmp_path / "storage")
        config.custom_params = {}
        with patch("traigent.cloud.sync_manager.LocalStorageManager"):
            manager = SyncManager(config=config, api_key="tg_" + "a" * 61)
        manager._session = MagicMock()
        return manager

    @pytest.mark.parametrize("raw_run_id", [None, "", "   "])
    def test_missing_run_id_variants_stay_none(
        self, sync_manager: SyncManager, raw_run_id, caplog
    ) -> None:
        metadata: dict[str, Any] = {"experiment_id": "exp-1"}
        if raw_run_id is not None:
            metadata["experiment_run_id"] = raw_run_id
        sync_manager._session.post.return_value = _backend_response(
            {"session_id": "sess-1", "metadata": metadata}
        )

        with caplog.at_level(logging.WARNING, logger="traigent.cloud.sync_manager"):
            result = sync_manager._sync_create_session({})

        assert result["success"] is True
        assert result["experiment_id"] == "exp-1"
        assert result["experiment_run_id"] is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "sess-1" not in warnings[0].getMessage()

    def test_both_ids_present_no_warning(
        self, sync_manager: SyncManager, caplog
    ) -> None:
        sync_manager._session.post.return_value = _backend_response(
            {
                "session_id": "sess-2",
                "metadata": {"experiment_id": "exp-2", "experiment_run_id": "run-2"},
            }
        )

        with caplog.at_level(logging.WARNING, logger="traigent.cloud.sync_manager"):
            result = sync_manager._sync_create_session({})

        assert result["experiment_id"] == "exp-2"
        assert result["experiment_run_id"] == "run-2"
        assert not [r for r in caplog.records if r.levelno == logging.WARNING]

    def test_require_run_id_env_true_fails_historical_import(
        self, sync_manager: SyncManager, monkeypatch
    ) -> None:
        """A missing run id under TRAIGENT_REQUIRE_RUN_ID=true fails the import
        (a historical import is not a trial path -- a structured failure, not
        an exception)."""
        monkeypatch.setenv("TRAIGENT_REQUIRE_RUN_ID", "true")
        sync_manager._session.post.return_value = _backend_response(
            {"session_id": "sess-3", "metadata": {}}
        )

        result = sync_manager._sync_create_session({})

        assert result["success"] is False
        assert "experiment_run_id" in result["error"]

    def test_require_run_id_env_false_default_unaffected(
        self, sync_manager: SyncManager, monkeypatch
    ) -> None:
        monkeypatch.delenv("TRAIGENT_REQUIRE_RUN_ID", raising=False)
        sync_manager._session.post.return_value = _backend_response(
            {"session_id": "sess-4", "metadata": {}}
        )

        result = sync_manager._sync_create_session({})

        assert result["success"] is True
        assert result["experiment_run_id"] is None


# ---------------------------------------------------------------------------
# Item 2: finalize is session-addressed and omits experiment_run_id when None
# ---------------------------------------------------------------------------


class TestFinalizeSessionAddressed:
    @pytest.fixture
    def client(self) -> BackendIntegratedClient:
        with patch("traigent.cloud.backend_client.AIOHTTP_AVAILABLE", True):
            client = BackendIntegratedClient(
                api_key="test-api-key", base_url="http://localhost:5000"
            )
            client.auth_manager = Mock()
            client.auth_manager.augment_headers = AsyncMock(
                return_value={"Authorization": "Bearer test-key"}
            )
            return client

    @staticmethod
    def _mock_aiohttp_session():
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        return mock_session

    @pytest.mark.asyncio
    async def test_finalize_with_none_run_id_omits_key(
        self, client: BackendIntegratedClient
    ) -> None:
        session_id = "sess-none-run"
        client.session_bridge.create_session_mapping(
            session_id=session_id,
            experiment_id="exp-1",
            experiment_run_id=None,
            function_name="f",
            configuration_space={},
            objectives=["accuracy"],
        )
        mock_session = self._mock_aiohttp_session()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await client._session_ops.finalize_session(session_id)

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert f"/sessions/{session_id}/finalize" in call_args[0][0]
        assert "experiment_run_id" not in call_args[1]["json"]

    @pytest.mark.asyncio
    async def test_finalize_with_run_id_includes_key(
        self, client: BackendIntegratedClient
    ) -> None:
        session_id = "sess-with-run"
        client.session_bridge.create_session_mapping(
            session_id=session_id,
            experiment_id="exp-2",
            experiment_run_id="run-2",
            function_name="f",
            configuration_space={},
            objectives=["accuracy"],
        )
        mock_session = self._mock_aiohttp_session()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await client._session_ops.finalize_session(session_id)

        call_args = mock_session.post.call_args
        assert call_args[1]["json"]["experiment_run_id"] == "run-2"

    @pytest.mark.asyncio
    async def test_recovers_mapping_with_nullable_ids_from_active_session(
        self, client: BackendIntegratedClient
    ) -> None:
        """An active session with no mapping recovers one with nullable ids
        and still finalizes (addendum F6)."""
        session_id = "sess-recover"
        assert client.session_bridge.get_session_mapping(session_id) is None
        active_session = SimpleNamespace(
            metadata={"experiment_id": "exp-3"},  # no experiment_run_id
            function_name="f",
            configuration_space={},
            objectives=["accuracy"],
        )
        with client._active_sessions_lock:
            client._active_sessions[session_id] = active_session
        mock_session = self._mock_aiohttp_session()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await client._session_ops.finalize_session(session_id)

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "experiment_run_id" not in call_args[1]["json"]
        recovered = client.session_bridge.get_session_mapping(session_id)
        assert recovered is not None
        assert recovered.experiment_id == "exp-3"
        assert recovered.experiment_run_id is None

    @pytest.mark.asyncio
    async def test_local_fallback_session_makes_no_finalize_call(
        self, client: BackendIntegratedClient
    ) -> None:
        """No mapping, no active session (local fallback) -> zero finalize calls."""
        session_id = "sess-local-fallback"
        assert client.session_bridge.get_session_mapping(session_id) is None
        with client._active_sessions_lock:
            assert session_id not in client._active_sessions
        mock_session = self._mock_aiohttp_session()

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await client._session_ops.finalize_session(session_id)

        mock_session.post.assert_not_called()


class TestSyncFinalizeWireOmitsKey:
    @pytest.fixture
    def sync_manager(self, tmp_path: Path) -> SyncManager:
        config = MagicMock(spec=TraigentConfig)
        config.get_local_storage_path.return_value = str(tmp_path / "storage")
        config.custom_params = {}
        with patch("traigent.cloud.sync_manager.LocalStorageManager"):
            manager = SyncManager(config=config, api_key="tg_" + "a" * 61)
        manager._session = MagicMock()
        return manager

    def test_finalize_with_none_run_id_omits_key(
        self, sync_manager: SyncManager
    ) -> None:
        sync_manager._session.post.return_value = _backend_response({}, status_code=200)

        sync_manager._sync_finalize_session("sess-1", None)

        call_kwargs = sync_manager._session.post.call_args[1]
        assert "experiment_run_id" not in call_kwargs["json"]

    def test_finalize_with_run_id_includes_key(self, sync_manager: SyncManager) -> None:
        sync_manager._session.post.return_value = _backend_response({}, status_code=200)

        sync_manager._sync_finalize_session("sess-1", "run-1")

        call_kwargs = sync_manager._session.post.call_args[1]
        assert call_kwargs["json"]["experiment_run_id"] == "run-1"


# ---------------------------------------------------------------------------
# Item 3: resume never persists a substituted id (display label only)
# ---------------------------------------------------------------------------


class TestResumeNeverPersistsSubstitutedId:
    def test_experiment_id_for_session_is_a_pure_label_helper(self) -> None:
        """The fallback helper used only at URL-build time never mutates a
        persisted id -- it is a pure function from (maybe-None, session_id)
        to a label."""
        assert SyncManager._experiment_id_for_session(None, "sess-1") == "sess-1"
        assert SyncManager._experiment_id_for_session("exp-1", "sess-1") == "exp-1"
        assert SyncManager._experiment_id_for_session("", "sess-1") == "sess-1"

    # The full resume-without-experiment-id -> cloud_experiment_id is None,
    # and the portal URL uses the session id with no "/None", is covered end
    # to end by tests/unit/cloud/test_sync_manager.py's legacy-resume-state
    # regression test (updated by this change to assert None instead of the
    # substituted session id).


# ---------------------------------------------------------------------------
# Item 4: require_run_id=True (option or env) fails closed at create time,
# before any trial -- driven directly against BackendSessionManager.create_session
# with a stubbed backend client, and end to end through the real orchestrator.
# ---------------------------------------------------------------------------


def _make_optimizer() -> Mock:
    optimizer = Mock()
    optimizer.objectives = ["accuracy"]
    optimizer.config_space = {"param1": [1, 2, 3]}
    return optimizer


def _make_objective_schema():
    return create_default_objectives(
        objective_names=["accuracy"],
        orientations={"accuracy": "maximize"},
        weights={"accuracy": 1.0},
    )


def _make_dataset() -> Mock:
    dataset = Mock(spec=Dataset)
    dataset.name = "test_dataset"
    dataset.examples = [EvaluationExample({"query": "hi"}, "hi")]
    dataset.__len__ = Mock(return_value=1)
    return dataset


def _make_descriptor():
    def func(x):
        return x

    func.__name__ = "test_func"
    return func, resolve_function_descriptor(func)


def _make_manager(
    backend_client, traigent_config, *, require_run_id: bool | None = None
) -> BackendSessionManager:
    return BackendSessionManager(
        backend_client=backend_client,
        traigent_config=traigent_config,
        objectives=["accuracy"],
        objective_schema=_make_objective_schema(),
        optimizer=_make_optimizer(),
        optimization_id="opt-id",
        optimization_status=OptimizationStatus.RUNNING,
        require_run_id=require_run_id,
    )


class TestRequireRunIdBackendSessionManager:
    @pytest.fixture(autouse=True)
    def _online(self, monkeypatch):
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        monkeypatch.setenv("TRAIGENT_OFFLINE", "false")
        monkeypatch.delenv("TRAIGENT_REQUIRE_RUN_ID", raising=False)

    @pytest.fixture
    def traigent_config(self) -> TraigentConfig:
        config = TraigentConfig()
        config.execution_mode = "local"
        return config

    @pytest.mark.parametrize("mapping_run_id", [None])
    def test_connected_missing_run_id_in_mapping_raises(
        self, traigent_config, mapping_run_id
    ) -> None:
        """Connected create, mapping exists but experiment_run_id is None."""
        client = Mock()
        client.create_session = Mock(
            return_value=SessionCreationResult.connected(session_id="sess-1")
        )
        client.get_session_mapping = Mock(
            return_value=MagicMock(experiment_run_id=mapping_run_id)
        )
        client.upload_example_features = Mock(return_value=True)
        manager = _make_manager(client, traigent_config, require_run_id=True)
        func, descriptor = _make_descriptor()

        with pytest.raises(_run_id_missing_error_cls()):
            manager.create_session(
                func=func,
                dataset=_make_dataset(),
                function_descriptor=descriptor,
                max_trials=10,
                start_time=0.0,
            )

        client.upload_example_features.assert_not_called()

    def test_connected_no_mapping_at_all_raises(self, traigent_config) -> None:
        client = Mock()
        client.create_session = Mock(
            return_value=SessionCreationResult.connected(session_id="sess-2")
        )
        client.get_session_mapping = Mock(return_value=None)
        client.upload_example_features = Mock(return_value=True)
        manager = _make_manager(client, traigent_config, require_run_id=True)
        func, descriptor = _make_descriptor()

        with pytest.raises(_run_id_missing_error_cls()):
            manager.create_session(
                func=func,
                dataset=_make_dataset(),
                function_descriptor=descriptor,
                max_trials=10,
                start_time=0.0,
            )

        client.upload_example_features.assert_not_called()

    @pytest.mark.parametrize(
        "reason",
        [
            SessionCreationFailureReason.NO_API_KEY,
            SessionCreationFailureReason.SESSION_FAILED,
            SessionCreationFailureReason.AUTH,
        ],
    )
    def test_fallback_reasons_raise_before_any_local_session_work(
        self, traigent_config, reason
    ) -> None:
        """Any local-fallback create yields no authoritative run id -> raise,
        without contacting the network again (the fallback result is already
        in hand; no further backend_client calls are made)."""
        client = Mock()
        client.create_session = Mock(
            return_value=SessionCreationResult.fallback(
                session_id="sess-fallback", reason=reason
            )
        )
        manager = _make_manager(client, traigent_config, require_run_id=True)
        func, descriptor = _make_descriptor()

        with pytest.raises(_run_id_missing_error_cls()):
            manager.create_session(
                func=func,
                dataset=_make_dataset(),
                function_descriptor=descriptor,
                max_trials=10,
                start_time=0.0,
            )

        client.get_session_mapping.assert_not_called()

    def test_offline_no_egress_raises_without_creating_local_session(self) -> None:
        """A no-egress/offline run never contacts the backend and can never
        mint a run id -- fail closed before any local session is created."""
        config = TraigentConfig(no_egress=True, enable_usage_analytics=False)
        manager = _make_manager(None, config, require_run_id=True)
        func, descriptor = _make_descriptor()

        with pytest.raises(_run_id_missing_error_cls()):
            manager.create_session(
                func=func,
                dataset=_make_dataset(),
                function_descriptor=descriptor,
                max_trials=10,
                start_time=0.0,
            )

    def test_kwarg_false_overrides_env_true(self, traigent_config, monkeypatch) -> None:
        """Precedence: an explicit False option beats TRAIGENT_REQUIRE_RUN_ID=true."""
        monkeypatch.setenv("TRAIGENT_REQUIRE_RUN_ID", "true")
        client = Mock()
        client.create_session = Mock(
            return_value=SessionCreationResult.connected(session_id="sess-ok")
        )
        client.get_session_mapping = Mock(
            return_value=MagicMock(experiment_run_id=None)
        )
        client.upload_example_features = Mock(return_value=True)
        manager = _make_manager(client, traigent_config, require_run_id=False)
        func, descriptor = _make_descriptor()

        session_ctx = manager.create_session(
            func=func,
            dataset=_make_dataset(),
            function_descriptor=descriptor,
            max_trials=10,
            start_time=0.0,
        )

        assert session_ctx.session_id == "sess-ok"

    def test_option_unset_defers_to_env_true(
        self, traigent_config, monkeypatch
    ) -> None:
        """require_run_id=None (not supplied) falls back to the env flag."""
        monkeypatch.setenv("TRAIGENT_REQUIRE_RUN_ID", "true")
        client = Mock()
        client.create_session = Mock(
            return_value=SessionCreationResult.connected(session_id="sess-3")
        )
        client.get_session_mapping = Mock(
            return_value=MagicMock(experiment_run_id=None)
        )
        manager = _make_manager(client, traigent_config, require_run_id=None)
        func, descriptor = _make_descriptor()

        with pytest.raises(_run_id_missing_error_cls()):
            manager.create_session(
                func=func,
                dataset=_make_dataset(),
                function_descriptor=descriptor,
                max_trials=10,
                start_time=0.0,
            )

    def test_default_unset_flag_false_existing_behavior_unchanged(
        self, traigent_config
    ) -> None:
        """No option, no env: a missing run id never raises (unchanged default)."""
        client = Mock()
        client.create_session = Mock(
            return_value=SessionCreationResult.connected(session_id="sess-4")
        )
        client.get_session_mapping = Mock(
            return_value=MagicMock(experiment_run_id=None)
        )
        client.upload_example_features = Mock(return_value=True)
        manager = _make_manager(client, traigent_config)
        func, descriptor = _make_descriptor()

        session_ctx = manager.create_session(
            func=func,
            dataset=_make_dataset(),
            function_descriptor=descriptor,
            max_trials=10,
            start_time=0.0,
        )

        assert session_ctx.session_id == "sess-4"


class _CountingOptimizer(BaseOptimizer):
    """Deterministic optimizer: suggests param1=0,1,2 then stops."""

    def __init__(self, config_space: dict[str, Any], objectives: list[str], **kwargs):
        super().__init__(config_space, objectives, **kwargs)
        self._suggest_count = 0

    def suggest_next_trial(self, history):
        config = {"param1": self._suggest_count}
        self._suggest_count += 1
        return config

    def should_stop(self, history) -> bool:
        return self._suggest_count >= 3

    def tell(self, config, result) -> None:
        return None

    def is_finished(self) -> bool:
        return self._suggest_count >= 3


class _CountingEvaluator(BaseEvaluator):
    """Evaluator that counts calls; used to assert zero trials ran."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluation_count = 0

    async def evaluate(
        self,
        func,
        config,
        dataset,
        *,
        sample_lease=None,
        progress_callback=None,
        **_kwargs,
    ) -> EvaluationResult:
        self.evaluation_count += 1
        result = EvaluationResult(
            config=config,
            aggregated_metrics={"accuracy": 0.5},
            total_examples=1,
            successful_examples=1,
            duration=0.01,
            metrics={"accuracy": 0.5},
            outputs=["x"],
            errors=[None],
        )
        result.sample_budget_exhausted = False
        result.examples_consumed = 1
        return result


class TestRequireRunIdOrchestratorEndToEnd:
    """Drives the real orchestrator -> BackendSessionManager.create_session
    path with a stubbed backend client and a spy on the evaluator, per the
    sharpest form of item 4 (addendum F13)."""

    @pytest.fixture(autouse=True)
    def isolated_logs(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("TRAIGENT_OPTIMIZATION_LOG_DIR", str(tmp_path / "logs"))

    @staticmethod
    def _build(backend_client, config: TraigentConfig, **extra_kwargs):
        evaluator = _CountingEvaluator()
        optimizer = _CountingOptimizer({"param1": [0, 1, 2]}, ["accuracy"])
        orchestrator = OptimizationOrchestrator(
            optimizer=optimizer,
            evaluator=evaluator,
            max_trials=3,
            config=config,
            **extra_kwargs,
        )
        orchestrator.backend_session_manager._backend_client = backend_client
        return orchestrator, evaluator

    @staticmethod
    def _dataset() -> Dataset:
        return Dataset([EvaluationExample({"query": "hi"}, "hi")], name="d")

    @staticmethod
    async def _func(input_data, **config):
        return input_data.get("query", "default")

    @pytest.mark.asyncio
    async def test_require_run_id_kwarg_raises_before_any_trial(self) -> None:
        client = Mock()
        client.create_session = Mock(
            return_value=SessionCreationResult.connected(session_id="sess-e2e-1")
        )
        client.get_session_mapping = Mock(return_value=None)
        orchestrator, evaluator = self._build(
            client,
            TraigentConfig(enable_usage_analytics=False),
            require_run_id=True,
        )

        with pytest.raises(_run_id_missing_error_cls()):
            await orchestrator.optimize(self._func, self._dataset())

        assert evaluator.evaluation_count == 0

    @pytest.mark.asyncio
    async def test_require_run_id_env_true_raises_before_any_trial(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("TRAIGENT_REQUIRE_RUN_ID", "true")
        client = Mock()
        client.create_session = Mock(
            return_value=SessionCreationResult.connected(session_id="sess-e2e-2")
        )
        client.get_session_mapping = Mock(return_value=None)
        orchestrator, evaluator = self._build(
            client, TraigentConfig(enable_usage_analytics=False)
        )

        with pytest.raises(_run_id_missing_error_cls()):
            await orchestrator.optimize(self._func, self._dataset())

        assert evaluator.evaluation_count == 0

    @pytest.mark.asyncio
    async def test_kwarg_false_beats_env_true_offline_run_completes(
        self, monkeypatch
    ) -> None:
        """Precedence end to end: kwarg False overrides env True, and an
        offline run (which would otherwise fail closed under the flag)
        completes normally."""
        monkeypatch.setenv("TRAIGENT_REQUIRE_RUN_ID", "true")
        orchestrator, evaluator = self._build(
            None,
            TraigentConfig(no_egress=True, enable_usage_analytics=False),
            require_run_id=False,
        )

        result = await orchestrator.optimize(self._func, self._dataset())

        assert result.status == OptimizationStatus.COMPLETED
        assert evaluator.evaluation_count == 3
