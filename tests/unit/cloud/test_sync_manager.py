"""Unit tests for traigent/cloud/sync_manager.py.

Tests for local-to-cloud sync manager functionality. Offline ``traigent sync``
imports historical runs through the content-free typed-session endpoints
(``POST /sessions`` -> per-trial ``POST /sessions/{id}/results`` ->
``POST /sessions/{id}/finalize``). No agent/benchmark/experiment/experiment-run
resources are created and no prompt/output content egresses. Binding no
benchmark makes the backend take its no-dataset pass-through, so a run whose
server-side dataset would have zero examples imports cleanly (empty-dataset
sync fix).
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from traigent.cloud.sync_manager import (
    SyncManager,
    build_experiment_url,
)
from traigent.config.types import TraigentConfig
from traigent.storage.local_storage import OptimizationSession, TrialResult
from traigent.utils.exceptions import TraigentStorageError


def backend_response(status_code=201, payload=None, text="Created"):
    response = Mock(status_code=status_code, text=text)
    response.json.return_value = (
        payload if payload is not None else {"id": "created-id"}
    )
    return response


def create_response(
    *,
    session_id="session-id",
    experiment_id="experiment-id",
    experiment_run_id="experiment-run-id",
    project_id=None,
    tenant_id=None,
    status_code=201,
):
    """Build a ``POST /sessions`` response like the backend returns.

    Mirrors ``_parse_session_response``: session_id at top level plus an
    ``experiment_id``/``experiment_run_id`` in ``metadata``. project/tenant may
    ride either at the top level or under metadata; here they ride at the top
    level to exercise the context-threading path.
    """
    payload: dict = {
        "session_id": session_id,
        "metadata": {
            "experiment_id": experiment_id,
            "experiment_run_id": experiment_run_id,
        },
    }
    if project_id is not None:
        payload["project_id"] = project_id
    if tenant_id is not None:
        payload["tenant_id"] = tenant_id
    return backend_response(status_code=status_code, payload=payload)


def test_build_experiment_url_uses_portal_view_route() -> None:
    """Experiment browser links target the route mounted by the portal SPA."""
    assert (
        build_experiment_url("https://portal.traigent.ai/", "exp_123")
        == "https://portal.traigent.ai/experiments/view/exp_123"
    )


def test_build_experiment_url_adds_encoded_context_query() -> None:
    """Experiment links include owning context when provided."""
    assert (
        build_experiment_url(
            "https://portal.traigent.ai/",
            "exp/123",
            run_id="run/456",
            project_id="project/alpha",
            tenant_id="tenant acme",
        )
        == "https://portal.traigent.ai/experiments/view/exp%2F123"
        "?run_id=run%2F456&project_id=project%2Falpha&tenant_id=tenant%20acme"
    )


def test_build_experiment_url_omits_empty_context_query() -> None:
    """Empty owning context keeps the backward-compatible bare URL."""
    assert (
        build_experiment_url(
            "https://portal.traigent.ai/",
            "exp_123",
            project_id=" ",
            tenant_id=None,
        )
        == "https://portal.traigent.ai/experiments/view/exp_123"
    )


class TestSyncManager:
    """Tests for SyncManager class."""

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> TraigentConfig:
        """Create mock TraigentConfig for testing."""
        config = MagicMock(spec=TraigentConfig)
        config.get_local_storage_path.return_value = str(tmp_path / "storage")
        config.custom_params = {}
        return config

    @pytest.fixture
    def sync_manager(self, mock_config: TraigentConfig) -> SyncManager:
        """Create SyncManager instance with mock config."""
        with patch("traigent.cloud.sync_manager.LocalStorageManager"):
            manager = SyncManager(
                config=mock_config,
                api_key="tg_" + "a" * 61,  # pragma: allowlist secret
            )
            # Replace the real session with a mock
            manager._session = MagicMock(spec=requests.Session)
            return manager

    @pytest.fixture
    def sync_manager_no_key(self, mock_config: TraigentConfig) -> SyncManager:
        """Create SyncManager instance without API key."""
        with patch("traigent.cloud.sync_manager.LocalStorageManager"):
            manager = SyncManager(config=mock_config, api_key=None)
            # Replace the real session with a mock
            manager._session = MagicMock(spec=requests.Session)
            return manager

    @pytest.fixture
    def sample_session(self) -> OptimizationSession:
        """Create sample OptimizationSession for testing."""
        trials = [
            TrialResult(
                trial_id=1,
                config={"model": "gpt-3.5-turbo", "temperature": 0.7},
                score=0.85,
                timestamp="2025-01-01T12:00:00Z",
                metadata={"latency": 120},
                error=None,
            ),
            TrialResult(
                trial_id=2,
                config={"model": "gpt-4", "temperature": 0.5},
                score=0.92,
                timestamp="2025-01-01T12:05:00Z",
                metadata={"latency": 180},
                error=None,
            ),
            TrialResult(
                trial_id=3,
                config={"model": "gpt-3.5-turbo", "temperature": 0.9},
                score=0.78,
                timestamp="2025-01-01T12:10:00Z",
                metadata=None,
                error="Timeout error",
            ),
        ]

        return OptimizationSession(
            session_id="test_session_123",
            function_name="test_function",
            created_at="2025-01-01T12:00:00Z",
            updated_at="2025-01-01T12:10:00Z",
            status="completed",
            total_trials=10,
            completed_trials=3,
            best_config={"model": "gpt-4", "temperature": 0.5},
            best_score=0.92,
            baseline_score=0.75,
            trials=trials,
            optimization_config={"search_space": {"model": ["gpt-3.5-turbo", "gpt-4"]}},
            metadata={"optimizer": "bayesian"},
        )

    def test_convert_trials_to_results_includes_local_cost_measures(
        self, sync_manager: SyncManager
    ) -> None:
        """Synced configuration_runs include local edge_analytics cost measures."""
        trial = TrialResult(
            trial_id=1,
            config={"model": "gpt-4", "temperature": 0.2},
            score=0.91,
            timestamp="2026-06-09T10:00:00Z",
            metadata={"provider": "mock"},
            cost=0.03,
            total_cost=0.03,
            input_cost=0.01,
            output_cost=0.02,
        )

        results = sync_manager._convert_trials_to_results([trial])

        measures = results[0]["measures"]
        assert measures["score"] == 0.91
        # No "accuracy" alias when no all_metrics in metadata (bug #1319 fix)
        assert "accuracy" not in measures
        assert measures["cost"] == 0.03
        assert measures["total_cost"] == 0.03
        assert measures["input_cost"] == 0.01
        assert measures["output_cost"] == 0.02

    def test_convert_trials_to_results_includes_metadata_cost_measures(
        self, sync_manager: SyncManager
    ) -> None:
        """Legacy local trials with metadata-only costs still sync cost measures."""
        trial = TrialResult(
            trial_id=1,
            config={"model": "gpt-4", "temperature": 0.2},
            score=0.91,
            timestamp="2026-06-09T10:00:00Z",
            metadata={
                "total_example_cost": 0.04,
                "all_metrics": {
                    "input_cost": 0.015,
                    "output_cost": 0.025,
                },
            },
        )

        results = sync_manager._convert_trials_to_results([trial])

        measures = results[0]["measures"]
        assert measures["cost"] == 0.04
        assert measures["total_cost"] == 0.04
        assert measures["input_cost"] == 0.015
        assert measures["output_cost"] == 0.025

    def test_convert_trials_real_per_objective_metrics_not_clobbered(
        self, sync_manager: SyncManager
    ) -> None:
        """Real per-objective metrics (accuracy, latency) survive into measures.

        Regression test for bug #1319: previously ``accuracy`` was aliased to the
        composite ``score``, clobbering the actual accuracy metric recorded by the
        evaluator.  Per-objective metrics are stored under
        ``trial.metadata["all_metrics"]``; they must appear in the synced measures
        at their real values and must not be overwritten by the composite score.
        """
        trial = TrialResult(
            trial_id=1,
            config={"model": "gpt-4", "temperature": 0.3},
            score=0.7,
            timestamp="2026-06-18T10:00:00Z",
            metadata={
                "all_metrics": {
                    "accuracy": 0.85,
                    "latency": 120.0,
                }
            },
        )

        results = sync_manager._convert_trials_to_results([trial])

        measures = results[0]["measures"]
        # Real accuracy from evaluator — must NOT be clobbered by composite score
        assert measures["accuracy"] == 0.85, (
            f"accuracy should be 0.85 (real metric), got {measures['accuracy']}"
        )
        # Latency should be synced as a first-class measure
        assert measures["latency"] == 120.0, (
            f"latency should be 120.0, got {measures.get('latency')}"
        )
        # Composite score is preserved for backward compatibility
        assert measures["score"] == 0.7, (
            f"composite score should be 0.7, got {measures['score']}"
        )

    # Initialization Tests

    def test_init_with_api_key(self, mock_config: TraigentConfig) -> None:
        """Test SyncManager initialization with API key."""
        with patch("traigent.cloud.sync_manager.LocalStorageManager") as mock_storage:
            manager = SyncManager(config=mock_config, api_key="tg_" + "a" * 61)

            assert manager.config == mock_config
            assert manager.api_key == "tg_" + "a" * 61  # pragma: allowlist secret
            # API key may be in X-API-Key or Authorization header
            assert "X-API-Key" in manager.headers or "Authorization" in manager.headers
            mock_storage.assert_called_once()

    def test_init_without_api_key(self, mock_config: TraigentConfig) -> None:
        """Test SyncManager initialization without API key."""
        with patch("traigent.cloud.sync_manager.LocalStorageManager"):
            manager = SyncManager(config=mock_config, api_key=None)

            assert manager.config == mock_config
            assert manager.api_key is None
            assert manager.headers == {}

    def test_session_property(self, sync_manager: SyncManager) -> None:
        """Test session property returns HTTP session."""
        session = sync_manager.session
        assert isinstance(session, requests.Session)

    # Timeout Resolution Tests

    def test_resolve_request_timeout_default(self, mock_config: TraigentConfig) -> None:
        """Test timeout resolution uses default when no custom value set."""
        with patch("traigent.cloud.sync_manager.LocalStorageManager"):
            manager = SyncManager(config=mock_config, api_key="test_key")
            assert manager._request_timeout == 15.0

    def test_resolve_request_timeout_from_env(
        self, mock_config: TraigentConfig
    ) -> None:
        """Test timeout resolution from environment variable."""
        with (
            patch("traigent.cloud.sync_manager.LocalStorageManager"),
            patch.dict("os.environ", {"TRAIGENT_SYNC_HTTP_TIMEOUT": "30.0"}),
        ):
            manager = SyncManager(config=mock_config, api_key="test_key")
            assert manager._request_timeout == 30.0

    def test_resolve_request_timeout_from_custom_params(
        self, mock_config: TraigentConfig
    ) -> None:
        """Test timeout resolution from custom_params."""
        mock_config.custom_params = {"sync_request_timeout": 25.0}
        with patch("traigent.cloud.sync_manager.LocalStorageManager"):
            manager = SyncManager(config=mock_config, api_key="test_key")
            assert manager._request_timeout == 25.0

    def test_resolve_request_timeout_invalid_env_value(
        self, mock_config: TraigentConfig
    ) -> None:
        """Test timeout resolution handles invalid environment value."""
        with (
            patch("traigent.cloud.sync_manager.LocalStorageManager"),
            patch.dict("os.environ", {"TRAIGENT_SYNC_HTTP_TIMEOUT": "invalid"}),
        ):
            manager = SyncManager(config=mock_config, api_key="test_key")
            assert manager._request_timeout == 15.0  # Falls back to default

    def test_resolve_request_timeout_negative_value(
        self, mock_config: TraigentConfig
    ) -> None:
        """Test timeout resolution rejects negative values."""
        mock_config.custom_params = {"sync_request_timeout": -5.0}
        with patch("traigent.cloud.sync_manager.LocalStorageManager"):
            manager = SyncManager(config=mock_config, api_key="test_key")
            assert manager._request_timeout == 15.0  # Falls back to default

    def test_resolve_request_timeout_zero_value(
        self, mock_config: TraigentConfig
    ) -> None:
        """Test timeout resolution rejects zero values."""
        mock_config.custom_params = {"sync_request_timeout": 0.0}
        with patch("traigent.cloud.sync_manager.LocalStorageManager"):
            manager = SyncManager(config=mock_config, api_key="test_key")
            assert manager._request_timeout == 15.0  # Falls back to default

    def test_resolve_request_timeout_precedence(
        self, mock_config: TraigentConfig
    ) -> None:
        """Test timeout resolution precedence (env > custom_params)."""
        mock_config.custom_params = {"sync_request_timeout": 20.0}
        with (
            patch("traigent.cloud.sync_manager.LocalStorageManager"),
            patch.dict("os.environ", {"TRAIGENT_SYNC_HTTP_TIMEOUT": "35.0"}),
        ):
            manager = SyncManager(config=mock_config, api_key="test_key")
            assert manager._request_timeout == 35.0  # Env takes precedence

    # Sync Status Tests

    def test_get_sync_status_with_sessions(self, sync_manager: SyncManager) -> None:
        """Test get_sync_status returns correct status for existing sessions."""
        mock_sessions = [
            Mock(
                status="completed",
                completed_trials=5,
                function_name="func1",
                session_id="s1",
            ),
            Mock(
                status="completed",
                completed_trials=10,
                function_name="func2",
                session_id="s2",
            ),
            Mock(
                status="running",
                completed_trials=3,
                function_name="func3",
                session_id="s3",
            ),
        ]

        sync_manager.storage.list_sessions.return_value = mock_sessions
        sync_manager.storage.get_storage_info.return_value = {
            "total_size": 1024,
            "session_count": 3,
        }

        status = sync_manager.get_sync_status()

        assert status["total_sessions"] == 3
        assert status["completed_sessions"] == 2
        assert status["total_trials"] == 18
        assert status["sync_eligible"] == 2
        assert "estimated_cloud_value" in status
        assert "storage_info" in status

    def test_get_sync_status_no_sessions(self, sync_manager: SyncManager) -> None:
        """Test get_sync_status with no sessions."""
        sync_manager.storage.list_sessions.return_value = []
        sync_manager.storage.get_storage_info.return_value = {
            "total_size": 0,
            "session_count": 0,
        }

        status = sync_manager.get_sync_status()

        assert status["total_sessions"] == 0
        assert status["completed_sessions"] == 0
        assert status["total_trials"] == 0
        assert status["sync_eligible"] == 0

    # Cloud Value Estimation Tests

    def test_estimate_cloud_value_with_sessions(
        self, sync_manager: SyncManager
    ) -> None:
        """Test cloud value estimation with multiple sessions."""
        sessions = [
            Mock(completed_trials=5),
            Mock(completed_trials=10),
            Mock(completed_trials=15),
        ]

        result = sync_manager._estimate_cloud_value(sessions)

        assert result["sessions_to_sync"] == 3
        assert result["average_trials_per_session"] == 10.0
        assert result["estimated_time_invested_hours"] == 0.75  # 3 * 15 / 60
        assert "cloud_benefits" in result
        assert "unlimited_trials" in result["cloud_benefits"]

    def test_estimate_cloud_value_empty_sessions(
        self, sync_manager: SyncManager
    ) -> None:
        """Test cloud value estimation with empty session list."""
        result = sync_manager._estimate_cloud_value([])
        assert result == {}

    # Session Conversion Tests

    def test_convert_session_to_traigent_format(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Conversion emits the content-free typed-session-create payload."""
        result = sync_manager.convert_session_to_traigent_format(sample_session)

        # Only the two typed-session keys — no agent/benchmark/experiment/run.
        assert set(result) == {"session_create", "configuration_runs"}
        for legacy_key in ("agent", "benchmark", "experiment", "experiment_run"):
            assert legacy_key not in result

        session_create = result["session_create"]
        assert session_create["function_name"] == "test_function"
        # Shorthand search space is normalized to the typed wire contract.
        assert session_create["configuration_space"] == {
            "model": {"type": "categorical", "choices": ["gpt-3.5-turbo", "gpt-4"]}
        }
        # Objectives are derived from trial measure names (content-free).
        assert session_create["objectives"] == ["latency", "score"]
        # Content-free dataset label: name + size + privacy flag only.
        # Unknown local size is coerced to 1 (the typed path validates
        # dataset_metadata.size as a positive int whenever the key is present).
        assert session_create["dataset_metadata"] == {
            "size": 1,
            "name": "Local Dataset test_function",
            "privacy_mode": True,
        }
        assert session_create["max_trials"] == 3
        # native_local binds no benchmark -> the EMPTY_DATASET guard never fires.
        assert session_create["optimization_strategy"] == {
            "algorithm": "optuna",
            "tracking_mode": "native_local",
        }
        assert session_create["metadata"] == {
            "function_name": "test_function",
            "evaluation_set": "default",
            "source": "offline_sync",
        }

        configuration_runs = result["configuration_runs"]
        assert len(configuration_runs) == 3
        first = configuration_runs[0]
        assert set(first) == {
            "trial_id",
            "experiment_parameters",
            "measures",
            "status",
        }
        assert first["trial_id"] == 1
        assert first["experiment_parameters"] == {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
        }
        assert first["status"] == "COMPLETED"
        assert first["measures"]["score"] == 0.85

    def test_convert_session_to_traigent_format_minimal_data(
        self, sync_manager: SyncManager
    ) -> None:
        """Test conversion with minimal session data."""
        minimal_session = OptimizationSession(
            session_id="minimal_123",
            function_name="minimal_func",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-01T00:00:00Z",
            status="completed",
            total_trials=0,
            completed_trials=0,
            trials=[],
        )

        result = sync_manager.convert_session_to_traigent_format(minimal_session)

        session_create = result["session_create"]
        assert session_create["function_name"] == "minimal_func"
        # No search space -> empty typed configuration space.
        assert session_create["configuration_space"] == {}
        # No trials -> the default ["score"] objective.
        assert session_create["objectives"] == ["score"]
        assert session_create["dataset_metadata"] == {
            "size": 1,
            "name": "Local Dataset minimal_func",
            "privacy_mode": True,
        }
        # max(len(runs), 1) keeps max_trials >= 1 even with no trials.
        assert session_create["max_trials"] == 1
        assert (
            session_create["optimization_strategy"]["tracking_mode"] == "native_local"
        )
        assert result["configuration_runs"] == []

    def test_convert_session_to_traigent_format_with_cost(
        self, sync_manager: SyncManager
    ) -> None:
        """Cost fields ride into content-free measures as floats."""
        session = OptimizationSession(
            session_id="cost_session",
            function_name="cost/fn?demo",
            created_at="2026-06-09T10:00:00Z",
            updated_at="2026-06-09T10:01:00Z",
            status="completed",
            total_trials=2,
            completed_trials=2,
            trials=[
                TrialResult(
                    trial_id=1,
                    config={"temperature": 0.1},
                    score=0.9,
                    timestamp="2026-06-09T10:00:00Z",
                    metadata={"total_cost": 0.11, "latency_ms": 120},
                ),
                TrialResult(
                    trial_id=2,
                    config={"temperature": 0.2},
                    score=0.91,
                    timestamp="2026-06-09T10:01:00Z",
                    cost=0.12,
                    total_cost=0.12,
                    input_cost=0.05,
                    output_cost=0.07,
                ),
            ],
            optimization_config={"search_space": {"temperature": [0.1, 0.2]}},
        )

        result = sync_manager.convert_session_to_traigent_format(session)

        assert set(result) == {"session_create", "configuration_runs"}
        session_create = result["session_create"]
        # Name sanitized to the backend name pattern (slash/question -> space).
        assert (
            session_create["dataset_metadata"]["name"] == "Local Dataset cost fn demo"
        )
        assert session_create["dataset_metadata"]["size"] == 1
        assert session_create["dataset_metadata"]["privacy_mode"] is True
        assert (
            session_create["optimization_strategy"]["tracking_mode"] == "native_local"
        )
        assert session_create["configuration_space"] == {
            "temperature": {"type": "categorical", "choices": [0.1, 0.2]}
        }
        assert session_create["objectives"]  # non-empty derived measure names

        configuration_runs = result["configuration_runs"]
        assert len(configuration_runs) == 2
        for configuration_run in configuration_runs:
            assert set(configuration_run) == {
                "trial_id",
                "experiment_parameters",
                "measures",
                "status",
            }
            assert configuration_run["status"] == "COMPLETED"
        # Cost measure is a plain float (never a bool).
        cost = configuration_runs[1]["measures"]["cost"]
        assert isinstance(cost, float)
        assert not isinstance(cost, bool)

    # Trial Conversion Tests

    def test_convert_trials_to_results(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test conversion of trials to Traigent results format."""
        results = sync_manager._convert_trials_to_results(sample_session.trials)

        assert len(results) == 3

        # Check successful trial
        assert results[0]["trial_id"] == 1
        assert results[0]["experiment_parameters"]["model"] == "gpt-3.5-turbo"
        assert results[0]["measures"]["score"] == 0.85
        # "accuracy" is no longer aliased from score (bug #1319 fix)
        assert "accuracy" not in results[0]["measures"]
        assert results[0]["status"] == "completed"
        assert "error" not in results[0]

        # Check failed trial
        assert results[2]["trial_id"] == 3
        assert results[2]["status"] == "failed"
        assert results[2]["error"] == "Timeout error"

    def test_convert_trials_to_results_empty_list(
        self, sync_manager: SyncManager
    ) -> None:
        """Test conversion of empty trial list."""
        results = sync_manager._convert_trials_to_results([])
        assert results == []

    # Sync Session to Cloud Tests

    def test_sync_session_to_cloud_dry_run(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test sync session with dry_run mode."""
        sync_manager.storage.load_session.return_value = sample_session

        result = sync_manager.sync_session_to_cloud("test_session_123", dry_run=True)

        assert result["status"] == "success"
        assert result["dry_run"] is True
        assert result["data_converted"] is True
        assert result["trials_converted"] == 3
        assert "preview" in result
        preview = result["preview"]
        assert set(preview) == {
            "function_name",
            "dataset_name",
            "dataset_size",
            "trial_count",
            "best_score",
            "already_synced",
        }
        assert preview["function_name"] == "test_function"
        assert preview["dataset_name"] == "Local Dataset test_function"
        assert preview["dataset_size"] == 1
        assert preview["trial_count"] == 3
        assert preview["best_score"] == 0.92
        assert preview["already_synced"] is False
        # Dry-run uploads nothing.
        sync_manager._session.post.assert_not_called()

    def test_sync_session_to_cloud_session_not_found(
        self, sync_manager: SyncManager
    ) -> None:
        """Test sync session when session not found."""
        sync_manager.storage.load_session.return_value = None

        with pytest.raises(TraigentStorageError, match="Session .* not found"):
            sync_manager.sync_session_to_cloud("nonexistent_session")

    def test_sync_session_to_cloud_no_api_key(
        self, sync_manager_no_key: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test sync session without API key."""
        sync_manager_no_key.storage.load_session.return_value = sample_session

        result = sync_manager_no_key.sync_session_to_cloud("test_session_123")

        assert result["status"] == "error"
        assert "No API key provided" in result["errors"][0]
        # Verify signup URL is included in the error message
        from traigent.config.backend_config import SIGNUP_URL

        assert SIGNUP_URL in result["errors"][0]

    def test_sync_session_to_cloud_success(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test successful sync of session to cloud."""
        sync_manager.storage.load_session.return_value = sample_session

        # create -> 3 results -> finalize; new sync uses only .post.
        sync_manager._session.post.side_effect = [
            create_response(),
            backend_response(payload={"id": "result-1"}),
            backend_response(payload={"id": "result-2"}),
            backend_response(payload={"id": "result-3"}),
            backend_response(status_code=200, payload={"status": "finalized"}),
        ]

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "success"
        assert result["dry_run"] is False
        assert result["data_converted"] is True
        assert "cloud_url" in result
        assert len(result["errors"]) == 0
        # New sync never touches the classic .get/.put endpoints.
        sync_manager._session.get.assert_not_called()
        sync_manager._session.put.assert_not_called()

    def test_sync_session_to_cloud_posts_session_endpoints_and_threads_ids(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Full mocked flow posts the content-free session endpoints in order.

        Contract (empty-dataset sync fix): a completed local session with N
        trials makes exactly these POSTs, in order — ``POST /sessions``, then
        ``POST /sessions/{id}/results`` N times, then
        ``POST /sessions/{id}/finalize`` — and threads the create response's
        ids (session/experiment/experiment_run/project/tenant) into the result
        and cloud_url. No ``.get`` and no ``.put``.
        """
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.post.side_effect = [
            create_response(project_id="project/alpha", tenant_id="tenant acme"),
            backend_response(payload={"id": "result-1"}),
            backend_response(payload={"id": "result-2"}),
            backend_response(payload={"id": "result-3"}),
            backend_response(status_code=200, payload={"status": "finalized"}),
        ]

        with patch(
            "traigent.cloud.sync_manager.BackendConfig.get_cloud_web_url",
            return_value="https://portal.traigent.ai/",
        ):
            result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "success"
        assert result["trials_converted"] == 3
        assert result["cloud_session_id"] == "session-id"
        assert result["cloud_experiment_id"] == "experiment-id"
        assert result["cloud_experiment_run_id"] == "experiment-run-id"
        assert result["project_id"] == "project/alpha"
        assert result["tenant_id"] == "tenant acme"
        assert (
            result["cloud_url"]
            == "https://portal.traigent.ai/experiments/view/experiment-id"
            "?run_id=experiment-run-id&project_id=project%2Falpha&tenant_id=tenant%20acme"
        )

        base = sync_manager.base_url
        post_calls = sync_manager._session.post.call_args_list
        assert [call.args[0] for call in post_calls] == [
            f"{base}/sessions",
            f"{base}/sessions/session-id/results",
            f"{base}/sessions/session-id/results",
            f"{base}/sessions/session-id/results",
            f"{base}/sessions/session-id/finalize",
        ]
        # Every POST carries the resolved request timeout.
        for call in post_calls:
            assert call.kwargs["timeout"] == sync_manager._request_timeout

        # The create body is the typed native_local contract with no benchmark.
        create_body = post_calls[0].kwargs["json"]
        assert create_body["function_name"] == "test_function"
        assert create_body["optimization_strategy"]["tracking_mode"] == "native_local"
        assert create_body["dataset_metadata"]["privacy_mode"] is True
        assert "benchmark" not in create_body
        assert "benchmark_id" not in create_body

        # Each result body is content-free: config + numeric metrics only.
        result_body = post_calls[1].kwargs["json"]
        assert set(result_body) == {"trial_id", "config", "status", "metrics"}
        assert result_body["status"] == "COMPLETED"
        assert result_body["config"] == {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
        }

        # The finalize body carries the content-free reason + experiment_run_id.
        finalize_body = post_calls[4].kwargs["json"]
        assert finalize_body == {
            "reason": "offline_sync_finalization",
            "experiment_run_id": "experiment-run-id",
        }

        # Never the classic agent/dataset/experiment endpoints.
        sync_manager._session.get.assert_not_called()
        sync_manager._session.put.assert_not_called()

    def test_sync_session_to_cloud_create_failure_is_error(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """A failed session-create POST surfaces as a loud error, no results."""
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.post.return_value = backend_response(
            status_code=500, text="Internal Server Error"
        )

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "error"
        assert len(result["errors"]) == 1
        assert "Session create failed" in result["errors"][0]
        # Only the create POST was attempted — no results/finalize after a failed
        # create.
        assert sync_manager._session.post.call_count == 1

    def test_sync_session_to_cloud_result_failure_is_partial(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """A failed per-trial result POST prevents success (stays partial)."""
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.post.side_effect = [
            create_response(),
            backend_response(payload={"id": "result-1"}),
            backend_response(status_code=500, text="result failed"),
            backend_response(payload={"id": "result-3"}),
        ]

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "partial"
        assert any("Result sync failed" in err for err in result["errors"])
        # A partial upload is NOT finalized (so a retry can resume and finish it).
        finalize_posts = [
            call
            for call in sync_manager._session.post.call_args_list
            if call.args[0].endswith("/finalize")
        ]
        assert finalize_posts == []

    def test_sync_session_to_cloud_all_results_fail_is_partial(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test partial status when the session is created but every result fails."""
        sync_manager.storage.load_session.return_value = sample_session

        def mock_post(url: str, *args, **kwargs):
            if url.endswith("/sessions"):
                return create_response()
            return backend_response(status_code=500, text="Internal Server Error")

        sync_manager._session.post.side_effect = mock_post

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "partial"
        assert len(result["errors"]) > 0

    def test_sync_session_to_cloud_exception(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test sync handles exceptions gracefully."""
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.post.side_effect = Exception("Network error")

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "error"
        # The .post exception is caught inside _sync_create_session and surfaced
        # as a create failure carrying the exception text.
        assert any("network error" in err.lower() for err in result["errors"])

    # New content-free session endpoint methods

    def test_sync_create_session_success(self, sync_manager: SyncManager) -> None:
        """_sync_create_session parses ids from the /sessions response."""
        sync_manager._session.post.return_value = create_response()

        result = sync_manager._sync_create_session({"function_name": "fn"})

        assert result == {
            "success": True,
            "session_id": "session-id",
            "experiment_id": "experiment-id",
            "experiment_run_id": "experiment-run-id",
            "project_id": None,
            "tenant_id": None,
        }
        sync_manager._session.post.assert_called_once_with(
            f"{sync_manager.base_url}/sessions",
            json={"function_name": "fn"},
            timeout=sync_manager._request_timeout,
        )

    def test_sync_create_session_threads_stripped_context(
        self, sync_manager: SyncManager
    ) -> None:
        """project_id/tenant_id are parsed and whitespace-stripped."""
        sync_manager._session.post.return_value = create_response(
            project_id="project-9", tenant_id="  tenant-7  "
        )

        result = sync_manager._sync_create_session({})

        assert result["success"] is True
        assert result["project_id"] == "project-9"
        assert result["tenant_id"] == "tenant-7"

    def test_sync_create_session_falls_back_to_session_id(
        self, sync_manager: SyncManager
    ) -> None:
        """Missing experiment ids fall back to the session_id."""
        sync_manager._session.post.return_value = backend_response(
            status_code=201, payload={"session_id": "sess-only"}
        )

        result = sync_manager._sync_create_session({})

        assert result["success"] is True
        assert result["session_id"] == "sess-only"
        assert result["experiment_id"] == "sess-only"
        assert result["experiment_run_id"] == "sess-only"

    def test_sync_create_session_http_failure(self, sync_manager: SyncManager) -> None:
        """A non-2xx create response is a structured failure."""
        sync_manager._session.post.return_value = backend_response(
            status_code=500, text="boom"
        )

        result = sync_manager._sync_create_session({})

        assert result == {"success": False, "error": "HTTP 500: boom"}

    def test_sync_create_session_missing_session_id(
        self, sync_manager: SyncManager
    ) -> None:
        """A 2xx response without session_id is a structured failure."""
        sync_manager._session.post.return_value = backend_response(
            status_code=201, payload={"metadata": {}}
        )

        result = sync_manager._sync_create_session({})

        assert result["success"] is False
        assert "session_id" in result["error"]

    def test_sync_create_session_exception(self, sync_manager: SyncManager) -> None:
        """A transport exception is caught and returned as a failure."""
        sync_manager._session.post.side_effect = Exception("network down")

        result = sync_manager._sync_create_session({})

        assert result == {"success": False, "error": "network down"}

    def test_sync_session_results_success(self, sync_manager: SyncManager) -> None:
        """Each configuration run POSTs a content-free result and reports synced."""
        sync_manager._session.post.return_value = backend_response(
            payload={"id": "result-99"}
        )
        configuration_runs = [
            {
                "trial_id": 7,
                "experiment_parameters": {"model": "gpt-4"},
                "measures": {"score": 0.5},
                "status": "COMPLETED",
            }
        ]
        recorded: list[tuple[str, str | None]] = []

        result = sync_manager._sync_session_results(
            "sess-1",
            configuration_runs,
            on_synced=lambda key, result_id: recorded.append((key, result_id)),
        )

        assert result == {
            "success": True,
            "synced": 1,
            "skipped": 0,
            "errors": [],
            "configuration_run_ids": ["result-99"],
        }
        assert recorded == [("cfg_0", "result-99")]
        sync_manager._session.post.assert_called_once_with(
            f"{sync_manager.base_url}/sessions/sess-1/results",
            json={
                "trial_id": 7,
                "config": {"model": "gpt-4"},
                "status": "COMPLETED",
                "metrics": {"score": 0.5},
            },
            timeout=sync_manager._request_timeout,
        )

    def test_sync_session_results_skips_already_synced(
        self, sync_manager: SyncManager
    ) -> None:
        """Runs whose stable key is already synced are skipped, never re-POSTed."""
        sync_manager._session.post.return_value = backend_response(
            payload={"id": "result-1"}
        )
        configuration_runs = [
            {
                "trial_id": 1,
                "experiment_parameters": {"a": 1},
                "measures": {"score": 0.1},
                "status": "COMPLETED",
            },
            {
                "trial_id": 2,
                "experiment_parameters": {"a": 2},
                "measures": {"score": 0.2},
                "status": "COMPLETED",
            },
        ]

        result = sync_manager._sync_session_results(
            "sess-1",
            configuration_runs,
            already_synced_keys={"cfg_0"},
        )

        assert result["synced"] == 1
        assert result["skipped"] == 1
        assert result["success"] is True
        # Only the not-yet-synced cfg_1 was POSTed.
        sync_manager._session.post.assert_called_once()
        assert sync_manager._session.post.call_args.args[0] == (
            f"{sync_manager.base_url}/sessions/sess-1/results"
        )
        assert sync_manager._session.post.call_args.kwargs["json"]["trial_id"] == 2

    def test_sync_session_results_http_failure(self, sync_manager: SyncManager) -> None:
        """A non-2xx result POST is reported as a per-trial error, not synced."""
        sync_manager._session.post.return_value = backend_response(
            status_code=500, text="bad"
        )
        configuration_runs = [
            {
                "trial_id": 1,
                "experiment_parameters": {"a": 1},
                "measures": {"score": 0.1},
                "status": "COMPLETED",
            }
        ]

        result = sync_manager._sync_session_results("sess-1", configuration_runs)

        assert result["success"] is False
        assert result["synced"] == 0
        assert result["errors"] == ["trial 1: HTTP 500: bad"]

    def test_sync_finalize_session_success(self, sync_manager: SyncManager) -> None:
        """Finalize POSTs the content-free reason and classifies as completed."""
        sync_manager._session.post.return_value = backend_response(
            status_code=200, payload={"status": "finalized"}
        )

        result = sync_manager._sync_finalize_session("sess-1", "run-1")

        assert result == {"success": True, "classification": "completed"}
        sync_manager._session.post.assert_called_once_with(
            f"{sync_manager.base_url}/sessions/sess-1/finalize",
            json={
                "reason": "offline_sync_finalization",
                "experiment_run_id": "run-1",
            },
            timeout=sync_manager._request_timeout,
        )

    def test_sync_finalize_session_http_failure(
        self, sync_manager: SyncManager
    ) -> None:
        """A non-2xx finalize response is a structured failure."""
        sync_manager._session.post.return_value = backend_response(
            status_code=409, text="conflict"
        )

        result = sync_manager._sync_finalize_session("sess-1", "run-1")

        assert result == {"success": False, "error": "HTTP 409: conflict"}

    # Resume idempotency (no duplicate result rows on retry)

    @staticmethod
    def _wire_persisting_sync_state(
        sync_manager: SyncManager, session: OptimizationSession
    ) -> None:
        """Make the mocked storage actually mutate ``session.sync_state``.

        The default MagicMock ``update_sync_state`` is a no-op, so a resume
        would never see the incremental per-config-run progress that the SUT
        persists. This side_effect mirrors the real LocalStorageManager merge
        (shallow top-level patch + per-trial merge under ``sync_state["trials"]``)
        so the no-duplicate-on-resume behavior is exercised end-to-end instead
        of being mocked away.
        """

        def _persist(session_id, patch, *, trial_updates=None):
            state = dict(session.sync_state or {})
            state.update(patch)
            if trial_updates:
                trials_state = dict(state.get("trials") or {})
                for trial_key, trial_patch in trial_updates.items():
                    merged = dict(trials_state.get(trial_key) or {})
                    merged.update(trial_patch)
                    trials_state[trial_key] = merged
                state["trials"] = trials_state
            session.sync_state = state
            return session

        sync_manager.storage.update_sync_state.side_effect = _persist

    def test_sync_session_to_cloud_resume_partial_uses_persisted_context_url(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """A partial-sync retry reuses the session AND skips already-synced runs.

        Regression guard for the data-integrity BLOCKER: the backend does NOT
        dedup result creates, so the SDK must be idempotent itself. A resume
        must NOT re-POST ``POST /sessions`` (the session is reused) and must
        re-POST only the result rows not yet recorded as synced.
        """
        payload_hash = sync_manager._compute_payload_hash(
            sync_manager.convert_session_to_traigent_format(sample_session)
        )
        # Prior attempt already uploaded results cfg_0 and cfg_1 (2 of 3);
        # cfg_2 is the only one left to upload on resume.
        sample_session.sync_state = {
            "status": "partial",
            "payload_hash": payload_hash,
            "cloud_session_id": "session-id",
            "cloud_experiment_id": "experiment-id",
            "cloud_experiment_run_id": "experiment-run-id",
            "project_id": "project/alpha",
            "tenant_id": "tenant acme",
            "attempts": 1,
            "trials": {
                "cfg_0": {
                    "status": "synced",
                    "cloud_configuration_run_id": "result-0",
                },
                "cfg_1": {
                    "status": "synced",
                    "cloud_configuration_run_id": "result-1",
                },
            },
        }
        sync_manager.storage.load_session.return_value = sample_session
        self._wire_persisting_sync_state(sync_manager, sample_session)
        # Only the single remaining result (cfg_2) + finalize should be POSTed.
        sync_manager._session.post.side_effect = [
            backend_response(payload={"id": "result-2"}),
            backend_response(status_code=200, payload={"status": "finalized"}),
        ]

        with patch(
            "traigent.cloud.sync_manager.BackendConfig.get_cloud_web_url",
            return_value="https://portal.traigent.ai/",
        ):
            result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "success"
        assert result["project_id"] == "project/alpha"
        assert result["tenant_id"] == "tenant acme"
        assert (
            result["cloud_url"]
            == "https://portal.traigent.ai/experiments/view/experiment-id"
            "?run_id=experiment-run-id&project_id=project%2Falpha&tenant_id=tenant%20acme"
        )

        base = sync_manager.base_url
        post_urls = [call.args[0] for call in sync_manager._session.post.call_args_list]
        # The session is REUSED: create is never re-POSTed on resume.
        assert f"{base}/sessions" not in post_urls
        # No-duplicate guard: exactly ONE result POST (the remaining cfg_2).
        result_posts = [url for url in post_urls if url.endswith("/results")]
        assert result_posts == [f"{base}/sessions/session-id/results"], (
            "resume must POST only the not-yet-synced result, never re-post synced ones"
        )
        # Finalized exactly once.
        finalize_posts = [url for url in post_urls if url.endswith("/finalize")]
        assert finalize_posts == [f"{base}/sessions/session-id/finalize"]

        # cfg_2 now recorded as synced so a further resume is a clean no-op.
        assert sample_session.sync_state["trials"]["cfg_2"] == {
            "status": "synced",
            "cloud_configuration_run_id": "result-2",
        }

        # Top-level outcome patch still carries context + attempt bookkeeping.
        outcome_calls = [
            call
            for call in sync_manager.storage.update_sync_state.call_args_list
            if call.kwargs.get("trial_updates") is None
        ]
        sync_state_patch = outcome_calls[-1].args[1]
        assert sync_state_patch["project_id"] == "project/alpha"
        assert sync_state_patch["tenant_id"] == "tenant acme"
        assert sync_state_patch["cloud_session_id"] == "session-id"
        assert sync_state_patch["cloud_url"] == result["cloud_url"]
        assert sync_state_patch["attempts"] == 2
        # No legacy agent/benchmark ids are persisted anymore.
        assert "cloud_agent_id" not in sync_state_patch
        assert "cloud_benchmark_id" not in sync_state_patch

    def test_sync_resume_does_not_duplicate_config_runs_after_midbatch_failure(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """End-to-end no-duplicate-on-retry guard for the data-integrity fix.

        Scenario:
          - sample_session has 3 result rows (cfg_0, cfg_1, cfg_2).
          - Attempt 1: result #2 (cfg_1) FAILS with a 500; cfg_0 and cfg_2
            succeed. The attempt records itself as ``partial`` and is NOT
            finalized.
          - Attempt 2 (resume, same payload_hash): cfg_0 and cfg_2 are recorded
            as synced and MUST NOT be re-POSTed (the backend does not dedup).
            Only cfg_1 is re-POSTed, then the session is finalized once.
        """
        sample_session.sync_state = None
        sync_manager.storage.load_session.return_value = sample_session
        self._wire_persisting_sync_state(sync_manager, sample_session)

        # --- Attempt 1: cfg_1 (the 2nd result POST) fails with 500. ---
        # Order of POSTs: create session, then results cfg_0, cfg_1, cfg_2.
        sync_manager._session.post.side_effect = [
            create_response(),
            backend_response(payload={"id": "result-0"}),  # cfg_0 OK
            backend_response(status_code=500, text="result 2 failed"),  # cfg_1 FAIL
            backend_response(payload={"id": "result-2"}),  # cfg_2 OK
        ]

        with patch(
            "traigent.cloud.sync_manager.BackendConfig.get_cloud_web_url",
            return_value="https://portal.traigent.ai/",
        ):
            first = sync_manager.sync_session_to_cloud("test_session_123")

        assert first["status"] == "partial"
        # cfg_0 and cfg_2 persisted as synced; cfg_1 NOT (it failed).
        trials_state = sample_session.sync_state["trials"]
        assert trials_state["cfg_0"]["status"] == "synced"
        assert trials_state["cfg_2"]["status"] == "synced"
        assert "cfg_1" not in trials_state
        # No finalize on a partial upload — the session stays open so the resume
        # can finish it.
        first_finalize_posts = [
            call
            for call in sync_manager._session.post.call_args_list
            if call.args[0].endswith("/finalize")
        ]
        assert first_finalize_posts == []

        # --- Attempt 2: resume. Only cfg_1 must be re-POSTed, then finalize. ---
        sync_manager._session.post.reset_mock(side_effect=True)
        # Resume reuses the saved session id, so the only result POST in attempt 2
        # is the single retried cfg_1, followed by finalize.
        sync_manager._session.post.side_effect = [
            backend_response(payload={"id": "result-1"}),  # cfg_1 retry
            backend_response(status_code=200, payload={"status": "finalized"}),
        ]

        with patch(
            "traigent.cloud.sync_manager.BackendConfig.get_cloud_web_url",
            return_value="https://portal.traigent.ai/",
        ):
            second = sync_manager.sync_session_to_cloud("test_session_123")

        assert second["status"] == "success"

        base = sync_manager.base_url
        post_urls = [call.args[0] for call in sync_manager._session.post.call_args_list]
        # The session is REUSED: no create POST on resume.
        assert f"{base}/sessions" not in post_urls
        # CORE ASSERTION — exactly one result POST on resume: the failed cfg_1.
        result_posts = [
            call
            for call in sync_manager._session.post.call_args_list
            if call.args[0].endswith("/results")
        ]
        assert len(result_posts) == 1, (
            "resume must NOT re-post the already-synced results (cfg_0/cfg_2); "
            f"got {len(result_posts)} result POSTs"
        )
        # And it carried cfg_1's content-free payload. Derive the expected body
        # from the SUT's own conversion so the assertion is grounded in real
        # behavior, not a hand-mirrored copy.
        expected_runs = sync_manager.convert_session_to_traigent_format(sample_session)[
            "configuration_runs"
        ]
        cfg_1 = expected_runs[1]
        assert result_posts[0].kwargs["json"] == {
            "trial_id": cfg_1["trial_id"],
            "config": cfg_1["experiment_parameters"],
            "status": "COMPLETED",
            "metrics": cfg_1["measures"],
        }

        # Every result uploaded exactly once across the two attempts.
        assert set(sample_session.sync_state["trials"]) == {"cfg_0", "cfg_1", "cfg_2"}
        for key in ("cfg_0", "cfg_1", "cfg_2"):
            assert sample_session.sync_state["trials"][key]["status"] == "synced"

        # Session finalized exactly once (only after the clean resume).
        finalize_posts = [url for url in post_urls if url.endswith("/finalize")]
        assert finalize_posts == [f"{base}/sessions/session-id/finalize"]

    # Sync All Sessions Tests

    def test_sync_all_sessions_dry_run(self, sync_manager: SyncManager) -> None:
        """Test sync all sessions in dry run mode."""
        mock_sessions = [
            Mock(status="completed", session_id="s1", completed_trials=5),
            Mock(status="completed", session_id="s2", completed_trials=10),
            Mock(status="running", session_id="s3", completed_trials=3),
        ]

        sync_manager.storage.list_sessions.return_value = mock_sessions

        # Mock dry run results
        def mock_sync(session_id, dry_run=False, force=False):
            return {
                "session_id": session_id,
                "status": "success",
                "dry_run": dry_run,
            }

        with patch.object(sync_manager, "sync_session_to_cloud", side_effect=mock_sync):
            result = sync_manager.sync_all_sessions(dry_run=True)

        assert result["total_sessions"] == 3
        assert result["eligible_sessions"] == 2
        assert result["synced_successfully"] == 2
        assert result["sync_errors"] == 0
        assert result["overall_status"] == "success"

    def test_sync_all_sessions_no_completed(self, sync_manager: SyncManager) -> None:
        """Test sync all sessions when no completed sessions exist."""
        mock_sessions = [
            Mock(status="running", session_id="s1", completed_trials=0, metadata={}),
            Mock(status="pending", session_id="s2", completed_trials=0, metadata={}),
        ]

        sync_manager.storage.list_sessions.return_value = mock_sessions

        result = sync_manager.sync_all_sessions()

        assert result["total_sessions"] == 2
        assert result["eligible_sessions"] == 0
        assert result["synced_successfully"] == 0
        assert result["overall_status"] == "success"

    def test_sync_all_sessions_partial_success(self, sync_manager: SyncManager) -> None:
        """Test sync all sessions with partial success."""
        mock_sessions = [
            Mock(status="completed", session_id="s1", completed_trials=5),
            Mock(status="completed", session_id="s2", completed_trials=10),
        ]

        sync_manager.storage.list_sessions.return_value = mock_sessions

        def mock_sync(session_id, dry_run=False, force=False):
            if session_id == "s1":
                return {"session_id": session_id, "status": "success"}
            else:
                return {"session_id": session_id, "status": "error"}

        with patch.object(sync_manager, "sync_session_to_cloud", side_effect=mock_sync):
            result = sync_manager.sync_all_sessions()

        assert result["synced_successfully"] == 1
        assert result["sync_errors"] == 1
        assert result["overall_status"] == "partial"

    def test_sync_all_sessions_all_failures(self, sync_manager: SyncManager) -> None:
        """Test sync all sessions when all fail."""
        mock_sessions = [
            Mock(status="completed", session_id="s1"),
            Mock(status="completed", session_id="s2"),
        ]

        sync_manager.storage.list_sessions.return_value = mock_sessions

        with patch.object(
            sync_manager,
            "sync_session_to_cloud",
            side_effect=Exception("Network error"),
        ):
            result = sync_manager.sync_all_sessions()

        assert result["synced_successfully"] == 0
        assert result["sync_errors"] == 2
        assert result["overall_status"] == "failed"

    # Analytics Preview Tests

    def test_get_cloud_analytics_preview_with_data(
        self, sync_manager: SyncManager
    ) -> None:
        """Test cloud analytics preview with completed sessions."""
        mock_sessions = [
            Mock(
                status="completed",
                session_id="s1",
                completed_trials=5,
                function_name="func1",
            ),
            Mock(
                status="completed",
                session_id="s2",
                completed_trials=10,
                function_name="func2",
            ),
        ]

        sync_manager.storage.list_sessions.return_value = mock_sessions
        sync_manager.storage.get_session_summary.side_effect = [
            {"improvement": 15.5},
            {"improvement": 20.3},
        ]

        result = sync_manager.get_cloud_analytics_preview()

        assert result["sessions_ready_for_sync"] == 2
        assert result["total_optimization_trials"] == 15
        assert result["functions_optimized"] == 2
        assert result["average_improvement"] == pytest.approx(17.9)
        assert "cloud_analytics_available" in result
        assert "estimated_dashboard_value" in result

    def test_get_cloud_analytics_preview_no_sessions(
        self, sync_manager: SyncManager
    ) -> None:
        """Test cloud analytics preview with no completed sessions."""
        sync_manager.storage.list_sessions.return_value = []

        result = sync_manager.get_cloud_analytics_preview()

        assert "message" in result
        assert result["message"] == "No completed sessions to analyze"

    def test_get_cloud_analytics_preview_no_improvement_data(
        self, sync_manager: SyncManager
    ) -> None:
        """Test cloud analytics preview when improvement data is unavailable."""
        mock_sessions = [
            Mock(status="completed", session_id="s1", completed_trials=5),
        ]

        sync_manager.storage.list_sessions.return_value = mock_sessions
        sync_manager.storage.get_session_summary.return_value = None

        result = sync_manager.get_cloud_analytics_preview()

        assert result["average_improvement"] is None

    # Cleanup Tests

    def test_cleanup_after_sync_with_backup(
        self, sync_manager: SyncManager, tmp_path: Path
    ) -> None:
        """Test cleanup with backup enabled."""
        sync_manager.storage.storage_path = tmp_path
        sync_manager.storage.export_session.return_value = True
        sync_manager.storage.delete_session.return_value = True

        result = sync_manager.cleanup_after_sync(["s1", "s2"], keep_backup=True)

        assert result["sessions_processed"] == 2
        assert result["sessions_backed_up"] == 2
        assert result["sessions_deleted"] == 2
        assert len(result["errors"]) == 0

    def test_cleanup_after_sync_without_backup(self, sync_manager: SyncManager) -> None:
        """Test cleanup without backup."""
        sync_manager.storage.delete_session.return_value = True

        result = sync_manager.cleanup_after_sync(["s1", "s2"], keep_backup=False)

        assert result["sessions_processed"] == 2
        assert result["sessions_backed_up"] == 0
        assert result["sessions_deleted"] == 2

    def test_cleanup_after_sync_backup_failure(
        self, sync_manager: SyncManager, tmp_path: Path
    ) -> None:
        """Test cleanup when backup fails."""
        sync_manager.storage.storage_path = tmp_path
        sync_manager.storage.export_session.side_effect = Exception("Backup failed")
        sync_manager.storage.delete_session.return_value = True

        result = sync_manager.cleanup_after_sync(["s1"], keep_backup=True)

        assert result["sessions_backed_up"] == 0
        assert len(result["errors"]) > 0
        assert any("Backup failed" in err for err in result["errors"])

    def test_cleanup_after_sync_deletion_failure(
        self, sync_manager: SyncManager
    ) -> None:
        """Test cleanup when deletion fails."""
        sync_manager.storage.delete_session.side_effect = Exception("Delete failed")

        result = sync_manager.cleanup_after_sync(["s1"], keep_backup=False)

        assert result["sessions_deleted"] == 0
        assert len(result["errors"]) > 0
        assert any("Deletion failed" in err for err in result["errors"])

    def test_cleanup_after_sync_empty_list(self, sync_manager: SyncManager) -> None:
        """Test cleanup with empty session list."""
        result = sync_manager.cleanup_after_sync([], keep_backup=True)

        assert result["sessions_processed"] == 0
        assert result["sessions_backed_up"] == 0
        assert result["sessions_deleted"] == 0

    # Edge Cases and Error Handling

    def test_convert_trials_with_none_metadata(self, sync_manager: SyncManager) -> None:
        """Test trial conversion when metadata is None."""
        trial = TrialResult(
            trial_id=1,
            config={"param": "value"},
            score=0.8,
            timestamp="2025-01-01T00:00:00Z",
            metadata=None,
            error=None,
        )

        results = sync_manager._convert_trials_to_results([trial])

        assert len(results) == 1
        assert "metadata" not in results[0]

    def test_resolve_request_timeout_multiple_custom_params(
        self, mock_config: TraigentConfig
    ) -> None:
        """Test timeout resolution checks multiple custom param keys."""
        mock_config.custom_params = {
            "sync_request_timeout_seconds": 40.0,
            "request_timeout": 50.0,
        }
        with patch("traigent.cloud.sync_manager.LocalStorageManager"):
            manager = SyncManager(config=mock_config, api_key="test_key")
            # Should use first matching key
            assert manager._request_timeout == 40.0

    def test_base_url_initialization(self, sync_manager: SyncManager) -> None:
        """Test that base URL is properly initialized and trailing slash removed."""
        # The base_url should have trailing slash removed
        assert not sync_manager.base_url.endswith("/")
        assert "http" in sync_manager.base_url
