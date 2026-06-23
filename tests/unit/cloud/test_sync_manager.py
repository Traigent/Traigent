"""Unit tests for traigent/cloud/sync_manager.py.

Tests for local-to-cloud sync manager functionality including session synchronization,
data conversion, and cloud upload operations.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Reliability FUNC-CLOUD-HYBRID FUNC-AGENTS REQ-CLOUD-009 REQ-AGNT-013

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from traigent.cloud.sync_manager import (
    DEFAULT_SYNC_AGENT_TYPE_ID,
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
            project_id="project/alpha",
            tenant_id="tenant acme",
        )
        == "https://portal.traigent.ai/experiments/view/exp%2F123"
        "?project_id=project%2Falpha&tenant_id=tenant%20acme"
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
        """Test conversion of local session to Traigent format."""
        result = sync_manager.convert_session_to_traigent_format(sample_session)

        # Check structure
        assert "agent" in result
        assert "benchmark" in result
        assert "experiment" in result
        assert "experiment_run" in result
        assert "configuration_runs" in result
        assert "model_parameters" not in result

        # Check agent data
        assert result["agent"] == {
            "name": "Local Agent: test_function",
            "agent_type_id": DEFAULT_SYNC_AGENT_TYPE_ID,
        }

        # Check benchmark payload
        assert result["benchmark"]["name"] == "Local Dataset test_function"
        assert result["benchmark"]["type"] == "input-output"
        assert result["benchmark"]["label"]

        # Check experiment template
        assert result["experiment"]["name"] == "Local Import: test_function"
        # Experiment is created with PENDING (non-run-requiring) so the backend
        # accepts it before runs exist.  Both RUNNING and COMPLETED are rejected at
        # create time with 409 EXPERIMENT_HAS_NO_RUNS (#1420).
        # _finalize_experiment transitions it to COMPLETED via PUT after all runs.
        assert result["experiment"]["status"] == "PENDING"
        assert result["experiment"]["measures"]
        assert "agent_id" not in result["experiment"]
        assert "dataset_id" not in result["experiment"]
        assert "model_parameters_id" not in result["experiment"]

        experiment_payload = sync_manager._build_experiment_payload(
            result["experiment"], "agent-id", "benchmark-id"
        )
        assert experiment_payload["agent_id"] == "agent-id"
        assert experiment_payload["dataset_id"] == "benchmark-id"

        experiment_run_payload = sync_manager._build_experiment_run_payload(
            result["experiment_run"], "agent-id", "benchmark-id"
        )
        assert set(experiment_run_payload) == {"experiment_data"}
        assert set(experiment_run_payload["experiment_data"]) == {
            "agent_id",
            "benchmark_id",
            "measures",
            "configurations",
        }
        assert len(result["configuration_runs"]) == 3
        assert result["configuration_runs"][0]["status"] == "COMPLETED"

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

        assert result["agent"]["name"] == "Local Agent: minimal_func"
        assert len(result["configuration_runs"]) == 0
        assert result["experiment_run"]["measures"] == ["score"]

    def test_payload_builder_matches_current_backend_contract_with_cost(
        self, sync_manager: SyncManager
    ) -> None:
        """Build current-contract payloads without legacy fields."""
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

        assert result["agent"] == {
            "name": "Local Agent: cost/fn?demo",
            "agent_type_id": "completion",
        }
        assert result["benchmark"]["name"] == "Local Dataset cost fn demo"
        assert result["benchmark"]["type"] == "input-output"
        assert result["benchmark"]["label"]

        experiment_payload = sync_manager._build_experiment_payload(
            result["experiment"], "agent-id", "benchmark-id"
        )
        assert set(experiment_payload) == {
            "name",
            "agent_id",
            "dataset_id",
            "measures",
            "configurations",
            "status",
        }
        assert experiment_payload["agent_id"] == "agent-id"
        assert experiment_payload["dataset_id"] == "benchmark-id"
        assert experiment_payload["measures"]
        # Experiment is POSTed with PENDING; _finalize_experiment PUTs to COMPLETED.
        # RUNNING and COMPLETED are both rejected at create time (#1420).
        assert experiment_payload["status"] == "PENDING"

        experiment_run_payload = sync_manager._build_experiment_run_payload(
            result["experiment_run"], "agent-id", "benchmark-id"
        )
        assert experiment_run_payload["experiment_data"] == {
            "agent_id": "agent-id",
            "benchmark_id": "benchmark-id",
            "measures": result["experiment_run"]["measures"],
            "configurations": result["experiment_run"]["configurations"],
        }

        for configuration_payload in result["configuration_runs"]:
            assert set(configuration_payload) == {
                "experiment_parameters",
                "measures",
                "status",
            }
            assert configuration_payload["status"] == "COMPLETED"
            assert isinstance(configuration_payload["measures"]["cost"], (int, float))
            assert not isinstance(configuration_payload["measures"]["cost"], bool)

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
        assert "preview" in result
        assert result["preview"]["experiment_name"] == "Local Import: test_function"
        assert result["preview"]["trial_count"] == 3
        assert result["preview"]["best_score"] == 0.92

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

        # Mock successful responses; PUT is used for experiment finalization.
        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.return_value = backend_response()
        sync_manager._session.put.return_value = backend_response(status_code=200)

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "success"
        assert result["dry_run"] is False
        assert result["data_converted"] is True
        assert "cloud_url" in result
        assert len(result["errors"]) == 0

    def test_sync_session_to_cloud_posts_current_sequence_and_threads_ids(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Full mocked flow posts current endpoints and returned IDs.

        Verifies the ordering contract required by issue #1420:
          1. POST /experiments with status=PENDING (non-run-requiring)
          2. POST /experiment-runs/{id}/runs
          3. POST /configuration-runs/... (N times)
          4. PUT  /experiments/{id} with status=COMPLETED

        The backend rejects both RUNNING and COMPLETED at create time with
        409 EXPERIMENT_HAS_NO_RUNS.  The finalization route is PUT, not PATCH
        (TraigentBackend src/routes/experiment_routes.py:1058).
        """
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.side_effect = [
            backend_response(payload={"id": "agent-id"}),
            backend_response(payload={"id": "benchmark-id"}),
            backend_response(
                payload={
                    "id": "experiment-id",
                    "project_id": "project/alpha",
                    "tenant_id": "tenant acme",
                }
            ),
            backend_response(payload={"id": "experiment-run-id"}),
            backend_response(payload={"id": "configuration-run-1"}),
            backend_response(payload={"id": "configuration-run-2"}),
            backend_response(payload={"id": "configuration-run-3"}),
        ]
        # PUT /experiments/{id} finalizes to COMPLETED after all runs are uploaded.
        sync_manager._session.put.return_value = backend_response(status_code=200)

        with patch(
            "traigent.cloud.sync_manager.BackendConfig.get_cloud_web_url",
            return_value="https://portal.traigent.ai/",
        ):
            result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "success"
        assert result["cloud_experiment_id"] == "experiment-id"
        assert (
            result["cloud_url"]
            == "https://portal.traigent.ai/experiments/view/experiment-id"
            "?project_id=project%2Falpha&tenant_id=tenant%20acme"
        )
        post_calls = sync_manager._session.post.call_args_list
        assert [call.args[0] for call in post_calls] == [
            f"{sync_manager.base_url}/agents",
            f"{sync_manager.base_url}/datasets",
            f"{sync_manager.base_url}/experiments",
            f"{sync_manager.base_url}/experiment-runs/experiment-id/runs",
            f"{sync_manager.base_url}/configuration-runs/runs/"
            "experiment-run-id/configurations",
            f"{sync_manager.base_url}/configuration-runs/runs/"
            "experiment-run-id/configurations",
            f"{sync_manager.base_url}/configuration-runs/runs/"
            "experiment-run-id/configurations",
        ]

        # Verify experiment is created with PENDING (non-run-requiring) — the
        # fix for issue #1420: both RUNNING and COMPLETED before runs exist
        # cause 409 EXPERIMENT_HAS_NO_RUNS on the real backend.
        experiment_payload = post_calls[2].kwargs["json"]
        assert experiment_payload["agent_id"] == "agent-id"
        assert experiment_payload["dataset_id"] == "benchmark-id"
        assert experiment_payload["measures"]
        assert experiment_payload["status"] == "PENDING", (
            "Experiment must be created with status=PENDING (non-run-requiring) "
            "so the backend accepts it before any experiment_run exists (#1420). "
            "Both RUNNING and COMPLETED are rejected at create time."
        )

        experiment_run_payload = post_calls[3].kwargs["json"]
        assert set(experiment_run_payload) == {"experiment_data"}
        assert experiment_run_payload["experiment_data"] == {
            "agent_id": "agent-id",
            "benchmark_id": "benchmark-id",
            "measures": experiment_payload["measures"],
            "configurations": experiment_payload["configurations"],
        }

        # Verify PUT /experiments/{id} with status=COMPLETED was called last.
        # The backend exposes only PUT (not PATCH) for experiment updates
        # (TraigentBackend src/routes/experiment_routes.py:1058).
        put_calls = sync_manager._session.put.call_args_list
        assert len(put_calls) == 1, "Expected exactly one PUT to finalize experiment"
        assert put_calls[0].args[0] == (
            f"{sync_manager.base_url}/experiments/experiment-id"
        )
        assert put_calls[0].kwargs["json"] == {"status": "COMPLETED"}

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
        """A partial-sync retry preserves context AND skips already-synced runs.

        Regression guard for the data-integrity BLOCKER (#1420): the prior weak
        version of this test asserted ALL three config-runs were re-POSTed on
        resume — which is exactly the duplicate-on-retry bug, because the
        backend does NOT dedup config-run creates (backend issue #1330). The
        SDK must be idempotent itself: a resume re-POSTs only the config-runs
        not yet recorded as synced.
        """
        payload_hash = sync_manager._compute_payload_hash(
            sync_manager.convert_session_to_traigent_format(sample_session)
        )
        # Prior attempt already uploaded config-runs cfg_0 and cfg_1 (2 of 3);
        # cfg_2 is the only one left to upload on resume.
        sample_session.sync_state = {
            "status": "partial",
            "payload_hash": payload_hash,
            "cloud_agent_id": "agent-id",
            "cloud_benchmark_id": "benchmark-id",
            "cloud_experiment_id": "experiment-id",
            "cloud_experiment_run_id": "experiment-run-id",
            "project_id": "project/alpha",
            "tenant_id": "tenant acme",
            "attempts": 1,
            "trials": {
                "cfg_0": {
                    "status": "synced",
                    "cloud_configuration_run_id": "configuration-run-0",
                },
                "cfg_1": {
                    "status": "synced",
                    "cloud_configuration_run_id": "configuration-run-1",
                },
            },
        }
        sync_manager.storage.load_session.return_value = sample_session
        self._wire_persisting_sync_state(sync_manager, sample_session)
        # Only the single remaining config-run (cfg_2) should be POSTed.
        sync_manager._session.post.side_effect = [
            backend_response(payload={"id": "configuration-run-2"}),
        ]
        # PUT finalizes the experiment to COMPLETED after all runs upload.
        sync_manager._session.put.return_value = backend_response(status_code=200)

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
            "?project_id=project%2Falpha&tenant_id=tenant%20acme"
        )
        # No-duplicate guard: exactly ONE config-run POST (the remaining cfg_2),
        # NOT three. The already-synced cfg_0 / cfg_1 are skipped.
        config_post_urls = [
            call.args[0]
            for call in sync_manager._session.post.call_args_list
            if "configurations" in call.args[0]
        ]
        assert config_post_urls == [
            f"{sync_manager.base_url}/configuration-runs/runs/"
            "experiment-run-id/configurations",
        ], (
            "resume must POST only the not-yet-synced config-run, never re-post synced ones"
        )

        # Finalized to COMPLETED exactly once.
        put_calls = sync_manager._session.put.call_args_list
        assert len(put_calls) == 1
        assert put_calls[0].kwargs["json"] == {"status": "COMPLETED"}

        # cfg_2 now recorded as synced so a further resume is a clean no-op.
        assert sample_session.sync_state["trials"]["cfg_2"] == {
            "status": "synced",
            "cloud_configuration_run_id": "configuration-run-2",
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
        assert sync_state_patch["cloud_url"] == result["cloud_url"]
        assert sync_state_patch["attempts"] == 2

    def test_sync_resume_does_not_duplicate_config_runs_after_midbatch_failure(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """End-to-end no-duplicate-on-retry guard for the #1420 data-integrity fix.

        Scenario (matches the codex BLOCKER reproduction):
          - sample_session has 3 config-runs (cfg_0, cfg_1, cfg_2).
          - Attempt 1: config-run #2 (cfg_1) FAILS with a 500; cfg_0 and cfg_2
            succeed. The attempt records itself as ``partial``.
          - Attempt 2 (resume, same payload_hash): cfg_0 and cfg_2 are recorded
            as synced and MUST NOT be re-POSTed (the backend does not dedup —
            backend issue #1330). Only cfg_1 is re-POSTed.
          - After the resume the experiment is finalized to COMPLETED exactly
            once and every config-run is POSTed exactly once across the two
            attempts (no duplicates).
        """
        sample_session.sync_state = None
        sync_manager.storage.load_session.return_value = sample_session
        self._wire_persisting_sync_state(sync_manager, sample_session)

        # --- Attempt 1: cfg_1 (the 2nd config-run POST) fails with 500. ---
        # Order of POSTs: agent, benchmark, experiment, experiment-run,
        # then config-runs cfg_0, cfg_1, cfg_2.
        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.side_effect = [
            backend_response(payload={"id": "agent-id"}),
            backend_response(payload={"id": "benchmark-id"}),
            backend_response(payload={"id": "experiment-id"}),
            backend_response(payload={"id": "experiment-run-id"}),
            backend_response(payload={"id": "configuration-run-0"}),  # cfg_0 OK
            backend_response(status_code=500, text="config 2 failed"),  # cfg_1 FAIL
            backend_response(payload={"id": "configuration-run-2"}),  # cfg_2 OK
        ]
        # Finalization is gated on a clean upload, so it is not reached here.
        sync_manager._session.put.return_value = backend_response(status_code=200)

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
        # No finalize on a partial upload — the experiment stays un-completed so
        # the resume can finish it.
        assert sync_manager._session.put.call_count == 0

        # --- Attempt 2: resume. Only cfg_1 must be re-POSTed. ---
        sync_manager._session.post.reset_mock()
        sync_manager._session.put.reset_mock()
        sync_manager._session.get.reset_mock()
        # Resume reuses saved agent/benchmark/experiment/run ids, so the ONLY
        # POST in attempt 2 is the single retried config-run cfg_1.
        sync_manager._session.post.side_effect = [
            backend_response(payload={"id": "configuration-run-1"}),  # cfg_1 retry
        ]
        sync_manager._session.put.return_value = backend_response(status_code=200)

        with patch(
            "traigent.cloud.sync_manager.BackendConfig.get_cloud_web_url",
            return_value="https://portal.traigent.ai/",
        ):
            second = sync_manager.sync_session_to_cloud("test_session_123")

        assert second["status"] == "success"

        # CORE ASSERTION — no duplicate config-run POSTs on resume.
        config_posts = [
            call
            for call in sync_manager._session.post.call_args_list
            if "configurations" in call.args[0]
        ]
        # Exactly one config-run POST on resume: the previously-failed cfg_1.
        assert len(config_posts) == 1, (
            "resume must NOT re-post the already-synced config-runs (cfg_0/cfg_2); "
            f"got {len(config_posts)} config-run POSTs"
        )
        # And it carried cfg_1's payload — the trial that actually failed before.
        # Derive the expected per-config-run payloads from the SUT's own
        # conversion so the assertion is grounded in real behavior, not a
        # hand-mirrored copy.
        expected_config_runs = sync_manager.convert_session_to_traigent_format(
            sample_session
        )["configuration_runs"]
        retried_payload = config_posts[0].kwargs["json"]
        assert retried_payload == expected_config_runs[1]  # cfg_1 == index 1

        # Every config-run uploaded exactly once across the two attempts.
        assert set(sample_session.sync_state["trials"]) == {"cfg_0", "cfg_1", "cfg_2"}
        for key in ("cfg_0", "cfg_1", "cfg_2"):
            assert sample_session.sync_state["trials"][key]["status"] == "synced"

        # Experiment finalized to COMPLETED exactly once (only after the clean
        # resume), never on the failed first attempt.
        put_calls = sync_manager._session.put.call_args_list
        assert len(put_calls) == 1, "experiment must be finalized exactly once"
        assert put_calls[0].args[0] == (
            f"{sync_manager.base_url}/experiments/experiment-id"
        )
        assert put_calls[0].kwargs["json"] == {"status": "COMPLETED"}

    def test_sync_session_to_cloud_reuses_agent_and_benchmark_for_idempotency(
        self, sync_manager: SyncManager
    ) -> None:
        """Quota and duplicate-name failures fall back to existing resources."""
        session = OptimizationSession(
            session_id="idem-session",
            function_name="idem_fn",
            created_at="2026-06-09T10:00:00Z",
            updated_at="2026-06-09T10:00:00Z",
            status="completed",
            total_trials=1,
            completed_trials=1,
            trials=[
                TrialResult(
                    trial_id=1,
                    config={"model": "gpt-test"},
                    score=1.0,
                    timestamp="2026-06-09T10:00:00Z",
                    metadata={"total_cost": 0.001},
                )
            ],
            optimization_config={"search_space": {"model": ["gpt-test"]}},
        )
        sync_manager.storage.load_session.return_value = session
        sync_manager._session.get.side_effect = [
            backend_response(404),
            backend_response(
                200,
                payload={
                    "agents": [
                        {"id": "existing-agent-id", "name": "Local Agent: idem_fn"}
                    ]
                },
            ),
            backend_response(404),
            backend_response(
                200,
                payload={
                    "datasets": [
                        {
                            "id": "existing-benchmark-id",
                            "name": "Local Dataset idem_fn",
                        }
                    ]
                },
            ),
        ]
        sync_manager._session.post.side_effect = [
            backend_response(status_code=429, text="quota exceeded"),
            backend_response(status_code=500, text="duplicate benchmark"),
            backend_response(payload={"id": "experiment-id"}),
            backend_response(payload={"id": "experiment-run-id"}),
            backend_response(payload={"id": "configuration-run-id"}),
        ]
        # PUT finalizes experiment after all config runs are uploaded.
        sync_manager._session.put.return_value = backend_response(status_code=200)

        result = sync_manager.sync_session_to_cloud("idem-session")

        assert result["status"] == "success"
        post_calls = sync_manager._session.post.call_args_list
        assert post_calls[0].args[0] == f"{sync_manager.base_url}/agents"
        assert post_calls[1].args[0] == f"{sync_manager.base_url}/datasets"
        experiment_payload = post_calls[2].kwargs["json"]
        assert experiment_payload["agent_id"] == "existing-agent-id"
        assert experiment_payload["dataset_id"] == "existing-benchmark-id"

    def test_sync_session_to_cloud_configuration_failure_is_partial(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """A failed configuration-row POST prevents success status."""
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.side_effect = [
            backend_response(payload={"id": "agent-id"}),
            backend_response(payload={"id": "benchmark-id"}),
            backend_response(payload={"id": "experiment-id"}),
            backend_response(payload={"id": "experiment-run-id"}),
            backend_response(payload={"id": "configuration-run-1"}),
            backend_response(status_code=500, text="config failed"),
            backend_response(payload={"id": "configuration-run-3"}),
        ]

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "partial"
        assert any("Configuration run sync failed" in err for err in result["errors"])

    def test_sync_session_to_cloud_partial_success(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test partial success when some steps fail."""
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.get.return_value = backend_response(404)

        # Mock mixed responses (some succeed, some fail)
        def mock_post(url: str, *args, **kwargs):
            if "agents" in url or "datasets" in url:
                return backend_response()
            return backend_response(status_code=500, text="Internal Server Error")

        sync_manager._session.post.side_effect = mock_post

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "partial"
        assert len(result["errors"]) > 0

    def test_sync_session_to_cloud_all_failures(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test sync when all steps fail."""
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.return_value = backend_response(
            status_code=500, text="Internal Server Error"
        )

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "error"
        assert len(result["errors"]) == 1
        assert "Agent sync failed" in result["errors"][0]

    def test_sync_session_to_cloud_exception(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test sync handles exceptions gracefully."""
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.side_effect = Exception("Network error")

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "error"
        # When exceptions occur in sync methods, they get wrapped in error messages
        assert any(
            "sync failed" in err.lower() or "network error" in err.lower()
            for err in result["errors"]
        )

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

    # Individual Sync Methods Tests

    def test_sync_agent_success(self, sync_manager: SyncManager) -> None:
        """Test successful agent sync."""
        agent_data = {"name": "Test Agent", "agent_type_id": "completion"}

        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.return_value = backend_response(
            payload={"id": "agent_123"}
        )

        result = sync_manager._sync_agent(agent_data)

        assert result["success"] is True
        assert result["agent_id"] == "agent_123"

    def test_sync_agent_already_exists(self, sync_manager: SyncManager) -> None:
        """Test agent sync reuses an existing matching agent."""
        agent_data = {"name": "Test Agent", "agent_type_id": "completion"}

        sync_manager._session.get.return_value = backend_response(
            200, payload={"agents": [{"id": "agent_123", "name": "Test Agent"}]}
        )

        result = sync_manager._sync_agent(agent_data)

        assert result["success"] is True
        assert result["agent_id"] == "agent_123"
        assert result["reused"] is True
        sync_manager._session.post.assert_not_called()

    def test_sync_agent_failure(self, sync_manager: SyncManager) -> None:
        """Test agent sync failure."""
        agent_data = {"name": "Test Agent", "agent_type_id": "completion"}

        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.return_value = backend_response(
            status_code=500, text="Internal Server Error"
        )

        result = sync_manager._sync_agent(agent_data)

        assert result["success"] is False
        assert "HTTP 500" in result["error"]

    def test_sync_agent_exception(self, sync_manager: SyncManager) -> None:
        """Test agent sync handles exceptions."""
        agent_data = {"name": "Test Agent", "agent_type_id": "completion"}
        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.side_effect = Exception("Network error")

        result = sync_manager._sync_agent(agent_data)

        assert result["success"] is False
        assert "Network error" in result["error"]

    def test_sync_benchmark_success(self, sync_manager: SyncManager) -> None:
        """Test successful benchmark sync."""
        benchmark_data = {
            "name": "Test Benchmark",
            "type": "input-output",
            "label": "eval",
        }

        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.return_value = backend_response(
            payload={"id": "bench_123"}
        )

        result = sync_manager._sync_benchmark(benchmark_data)

        assert result["success"] is True
        assert result["benchmark_id"] == "bench_123"
        sync_manager._session.get.assert_called_once_with(
            f"{sync_manager.base_url}/datasets",
            params={"name": "Test Benchmark"},
            timeout=sync_manager._request_timeout,
        )
        sync_manager._session.post.assert_called_once_with(
            f"{sync_manager.base_url}/datasets",
            json=benchmark_data,
            timeout=sync_manager._request_timeout,
        )

    def test_sync_benchmark_failure(self, sync_manager: SyncManager) -> None:
        """Test benchmark sync failure."""
        benchmark_data = {
            "name": "Test Benchmark",
            "type": "input-output",
            "label": "eval",
        }

        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.return_value = backend_response(
            status_code=400, text="Bad Request"
        )

        result = sync_manager._sync_benchmark(benchmark_data)

        assert result["success"] is False
        assert "HTTP 400" in result["error"]

    def test_sync_benchmark_reuses_existing_by_name(
        self, sync_manager: SyncManager
    ) -> None:
        """Benchmark sync reuses an existing exact-name benchmark."""
        benchmark_data = {
            "name": "Test Benchmark",
            "type": "input-output",
            "label": "eval",
        }

        sync_manager._session.get.return_value = backend_response(
            200,
            payload={"datasets": [{"id": "bench_123", "name": "Test Benchmark"}]},
        )

        result = sync_manager._sync_benchmark(benchmark_data)

        assert result["success"] is True
        assert result["benchmark_id"] == "bench_123"
        assert result["reused"] is True
        sync_manager._session.get.assert_called_once_with(
            f"{sync_manager.base_url}/datasets",
            params={"name": "Test Benchmark"},
            timeout=sync_manager._request_timeout,
        )
        sync_manager._session.post.assert_not_called()

    def test_sync_experiment_success(self, sync_manager: SyncManager) -> None:
        """Test successful experiment sync."""
        experiment_data = {
            "name": "Test Experiment",
            "agent_id": "agent-123",
            "dataset_id": "benchmark-123",
            "measures": ["score"],
            "configurations": {},
            "status": "COMPLETED",
        }

        sync_manager._session.post.return_value = backend_response(
            payload={"id": "exp_123"}
        )

        result = sync_manager._sync_experiment(experiment_data)

        assert result["success"] is True
        assert result["experiment_id"] == "exp_123"

    def test_sync_experiment_failure(self, sync_manager: SyncManager) -> None:
        """Test experiment sync failure."""
        experiment_data = {
            "name": "Test Experiment",
            "agent_id": "agent-123",
            "dataset_id": "benchmark-123",
            "measures": ["score"],
            "configurations": {},
            "status": "COMPLETED",
        }

        sync_manager._session.post.return_value = backend_response(
            status_code=403, text="Forbidden"
        )

        result = sync_manager._sync_experiment(experiment_data)

        assert result["success"] is False
        assert "HTTP 403" in result["error"]

    def test_sync_experiment_run_success(self, sync_manager: SyncManager) -> None:
        """Test successful experiment run sync."""
        run_data = {
            "experiment_data": {
                "agent_id": "agent-123",
                "benchmark_id": "benchmark-123",
                "measures": ["score"],
                "configurations": {},
            }
        }

        sync_manager._session.post.return_value = backend_response(
            payload={"id": "run_123"}
        )

        result = sync_manager._sync_experiment_run("exp_123", run_data)

        assert result["success"] is True
        assert result["run_id"] == "run_123"

    def test_sync_experiment_run_failure(self, sync_manager: SyncManager) -> None:
        """Test experiment run sync failure."""
        run_data = {
            "experiment_data": {
                "agent_id": "agent-123",
                "benchmark_id": "benchmark-123",
                "measures": ["score"],
                "configurations": {},
            }
        }

        sync_manager._session.post.return_value = backend_response(
            status_code=404, text="Not Found"
        )

        result = sync_manager._sync_experiment_run("exp_123", run_data)

        assert result["success"] is False
        assert "HTTP 404" in result["error"]

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

    # Regression tests for issue #1420 — sync ordering and exit code

    def test_sync_run_ordering_experiment_created_pending_then_finalized(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Completed local session syncs without 409: experiment created PENDING,
        runs uploaded, then experiment transitioned to COMPLETED via PUT.

        Regression test for issue #1420: the backend rejects
        POST /experiments with status=RUNNING or COMPLETED when no experiment_run
        exists (HTTP 409 EXPERIMENT_HAS_NO_RUNS, _RUN_REQUIRING_EXPERIMENT_STATUSES
        = {RUNNING, COMPLETED}, TraigentBackend src/models/status_enums.py:333).
        The fix: create with PENDING (non-run-requiring) and PUT to COMPLETED after
        all runs are uploaded (PUT /experiments/{id}, not PATCH).
        """
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.side_effect = [
            backend_response(payload={"id": "agent-id"}),
            backend_response(payload={"id": "benchmark-id"}),
            backend_response(payload={"id": "experiment-id"}),
            backend_response(payload={"id": "experiment-run-id"}),
            backend_response(payload={"id": "cfg-run-1"}),
            backend_response(payload={"id": "cfg-run-2"}),
            backend_response(payload={"id": "cfg-run-3"}),
        ]
        sync_manager._session.put.return_value = backend_response(status_code=200)

        with patch(
            "traigent.cloud.sync_manager.BackendConfig.get_cloud_web_url",
            return_value="https://portal.traigent.ai/",
        ):
            result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "success", (
            f"Expected success but got {result['status']}: {result.get('errors')}"
        )

        # ASSERT ORDERING: experiment POSTed with status=PENDING (non-run-requiring).
        # Both RUNNING and COMPLETED are rejected at create time with
        # 409 EXPERIMENT_HAS_NO_RUNS (#1420).
        post_calls = sync_manager._session.post.call_args_list
        experiment_post_payload = post_calls[2].kwargs["json"]
        assert experiment_post_payload["status"] == "PENDING", (
            "Experiment must be created with status=PENDING (non-run-requiring); "
            "both RUNNING and COMPLETED before runs exist cause HTTP 409 "
            "EXPERIMENT_HAS_NO_RUNS (#1420)"
        )

        # ASSERT ORDERING: runs uploaded BEFORE finalization.
        # The 4th POST is experiment_run, 5th-7th are configuration_runs.
        assert post_calls[3].args[0].endswith("/experiment-runs/experiment-id/runs")
        for cfg_call in post_calls[4:]:
            assert "configuration-runs" in cfg_call.args[0]

        # ASSERT ORDERING: PUT to COMPLETED is the LAST call, after all runs.
        # The backend exposes PUT (not PATCH) for experiment updates (#1420 fix).
        put_calls = sync_manager._session.put.call_args_list
        assert len(put_calls) == 1, "Expected exactly one PUT to finalize"
        assert put_calls[0].args[0].endswith("/experiments/experiment-id")
        assert put_calls[0].kwargs["json"] == {"status": "COMPLETED"}

        # Confirm the final experiment state is COMPLETED in the result.
        assert result["cloud_experiment_id"] == "experiment-id"

    def test_dry_run_outcome_matches_real_sync_outcome(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """--dry-run predicts the same outcome as the real sync.

        For a session with correct ordering (RUNNING create status), dry-run
        must report success.  If the ordering regresses to COMPLETED on create,
        dry-run must report error — not silently claim success while the real
        sync would 409.
        """
        sync_manager.storage.load_session.return_value = sample_session

        # Dry-run on a valid session (RUNNING create status) -> success.
        dry_result = sync_manager.sync_session_to_cloud(
            "test_session_123", dry_run=True
        )
        assert dry_result["status"] == "success", (
            f"Dry-run reported {dry_result['status']} for a valid session"
        )
        assert dry_result["dry_run"] is True
        assert "preview" in dry_result
        assert dry_result["preview"]["ordering_valid"] is True
        assert dry_result["preview"]["experiment_create_status"] == "PENDING"

        # Simulate the regression: temporarily make the payload use COMPLETED.
        # Dry-run must detect this and report error, not success.
        with patch.object(
            sync_manager,
            "convert_session_to_traigent_format",
            return_value={
                "agent": {"name": "agent", "agent_type_id": "completion"},
                "benchmark": {"name": "bench", "type": "input-output", "label": "l"},
                "experiment": {
                    "name": "exp",
                    "measures": ["score"],
                    "configurations": {},
                    "status": "COMPLETED",  # regression: run-requiring status at create
                },
                "experiment_run": {"measures": ["score"], "configurations": {}},
                "configuration_runs": [],
            },
        ):
            regressed_result = sync_manager.sync_session_to_cloud(
                "test_session_123", dry_run=True
            )

        assert regressed_result["status"] == "error", (
            "Dry-run must report error when experiment would be created with "
            "status=COMPLETED before any runs (would 409 on real sync)"
        )
        assert any(
            "409" in err or "COMPLETED" in err or "run-requiring" in err
            for err in regressed_result.get("errors", [])
        ), (
            f"Expected a 409/COMPLETED/run-requiring error, got {regressed_result.get('errors')}"
        )

    def test_dry_run_rejects_running_create_status(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Dry-run must also flag RUNNING as a run-requiring create-status error.

        The backend rejects both RUNNING and COMPLETED at create time with
        409 EXPERIMENT_HAS_NO_RUNS.  Dry-run must predict failure for either.
        """
        sync_manager.storage.load_session.return_value = sample_session

        with patch.object(
            sync_manager,
            "convert_session_to_traigent_format",
            return_value={
                "agent": {"name": "agent", "agent_type_id": "completion"},
                "benchmark": {"name": "bench", "type": "input-output", "label": "l"},
                "experiment": {
                    "name": "exp",
                    "measures": ["score"],
                    "configurations": {},
                    "status": "RUNNING",  # run-requiring: backend rejects this too
                },
                "experiment_run": {"measures": ["score"], "configurations": {}},
                "configuration_runs": [],
            },
        ):
            result = sync_manager.sync_session_to_cloud(
                "test_session_123", dry_run=True
            )

        assert result["status"] == "error", (
            "Dry-run must report error when experiment would be created with "
            "status=RUNNING (also run-requiring, also causes 409 on real sync)"
        )
        assert result["preview"]["ordering_valid"] is False
        assert any(
            "run-requiring" in err or "RUNNING" in err or "409" in err
            for err in result.get("errors", [])
        ), f"Expected a run-requiring/RUNNING/409 error, got {result.get('errors')}"

    def test_sync_experiment_failure_sets_error_status_in_result(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """A failed experiment POST leaves status as 'partial' (error before runs)."""
        sync_manager.storage.load_session.return_value = sample_session
        sync_manager._session.get.return_value = backend_response(404)
        sync_manager._session.post.side_effect = [
            backend_response(payload={"id": "agent-id"}),
            backend_response(payload={"id": "benchmark-id"}),
            # Experiment POST fails — simulates the 409 that existed before fix.
            backend_response(
                status_code=409,
                text='{"error": "EXPERIMENT_HAS_NO_RUNS"}',
            ),
        ]

        result = sync_manager.sync_session_to_cloud("test_session_123")

        # Verify failure is surfaced correctly.
        assert result["status"] in {"partial", "error"}, (
            f"Expected partial/error, got {result['status']}"
        )
        assert any("Experiment sync failed" in err for err in result.get("errors", []))
        # _finalize_experiment must NOT be called when experiment creation fails.
        sync_manager._session.put.assert_not_called()
