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

from traigent.cloud.sync_manager import SyncManager
from traigent.config.types import TraigentConfig
from traigent.storage.local_storage import OptimizationSession, TrialResult
from traigent.utils.exceptions import TraigentStorageError


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
            manager = SyncManager(config=mock_config, api_key="tg_" + "a" * 61)
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

    # Initialization Tests

    def test_init_with_api_key(self, mock_config: TraigentConfig) -> None:
        """Test SyncManager initialization with API key."""
        with patch("traigent.cloud.sync_manager.LocalStorageManager") as mock_storage:
            manager = SyncManager(config=mock_config, api_key="tg_" + "a" * 61)

            assert manager.config == mock_config
            assert manager.api_key == "tg_" + "a" * 61
            assert "Authorization" in manager.headers
            assert manager.headers["Authorization"] == "Bearer " + "tg_" + "a" * 61
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

    def test_convert_session_to_optigen_format(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test conversion of local session to OptiGen format."""
        result = sync_manager.convert_session_to_optigen_format(sample_session)

        # Check structure
        assert "agent" in result
        assert "benchmark" in result
        assert "model_parameters" in result
        assert "experiment" in result
        assert "experiment_run" in result

        # Check agent data
        assert result["agent"]["name"] == "Local Agent: test_function"
        assert result["agent"]["agent_type"] == "custom"
        assert result["agent"]["source"] == "local_import"

        # Check benchmark data
        assert result["benchmark"]["name"] == "Local Benchmark: test_function"
        assert result["benchmark"]["examples_count"] == 3
        assert result["benchmark"]["type"] == "custom"

        # Check experiment data
        assert result["experiment"]["name"] == "Local Import: test_function"
        assert result["experiment"]["status"] == "completed"
        assert result["experiment"]["source"] == "local_import"
        assert (
            result["experiment"]["metadata"]["original_session_id"]
            == "test_session_123"
        )

        # Check experiment run
        assert result["experiment_run"]["status"] == "completed"
        assert len(result["experiment_run"]["results"]) == 3
        assert result["experiment_run"]["metadata"]["total_trials"] == 3
        assert result["experiment_run"]["metadata"]["best_score"] == 0.92

    def test_convert_session_to_optigen_format_minimal_data(
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

        result = sync_manager.convert_session_to_optigen_format(minimal_session)

        assert result["agent"]["name"] == "Local Agent: minimal_func"
        assert len(result["experiment_run"]["results"]) == 0
        assert result["experiment_run"]["metadata"]["total_trials"] == 0

    # Trial Conversion Tests

    def test_convert_trials_to_results(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test conversion of trials to OptiGen results format."""
        results = sync_manager._convert_trials_to_results(sample_session.trials)

        assert len(results) == 3

        # Check successful trial
        assert results[0]["trial_id"] == 1
        assert results[0]["experiment_parameters"]["model"] == "gpt-3.5-turbo"
        assert results[0]["measures"]["score"] == 0.85
        assert results[0]["measures"]["accuracy"] == 0.85
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

    def test_sync_session_to_cloud_success(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test successful sync of session to cloud."""
        sync_manager.storage.load_session.return_value = sample_session

        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.text = "OK"

        sync_manager._session.post.return_value = mock_response

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "success"
        assert result["dry_run"] is False
        assert result["data_converted"] is True
        assert "cloud_url" in result
        assert len(result["errors"]) == 0

    def test_sync_session_to_cloud_partial_success(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test partial success when some steps fail."""
        sync_manager.storage.load_session.return_value = sample_session

        # Mock mixed responses (some succeed, some fail)
        def mock_post(url: str, *args, **kwargs):
            response = Mock()
            if "agents" in url or "benchmarks" in url:
                response.status_code = 201
            else:
                response.status_code = 500
                response.text = "Internal Server Error"
            return response

        sync_manager._session.post.side_effect = mock_post

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "partial"
        assert len(result["errors"]) > 0

    def test_sync_session_to_cloud_all_failures(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test sync when all steps fail."""
        sync_manager.storage.load_session.return_value = sample_session

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        sync_manager._session.post.return_value = mock_response

        result = sync_manager.sync_session_to_cloud("test_session_123")

        assert result["status"] == "error"
        assert len(result["errors"]) == 4  # All 4 steps failed

    def test_sync_session_to_cloud_exception(
        self, sync_manager: SyncManager, sample_session: OptimizationSession
    ) -> None:
        """Test sync handles exceptions gracefully."""
        sync_manager.storage.load_session.return_value = sample_session
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
        def mock_sync(session_id, dry_run=False):
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
            Mock(status="running", session_id="s1"),
            Mock(status="pending", session_id="s2"),
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

        def mock_sync(session_id, dry_run=False):
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
        agent_data = {"id": "agent_123", "name": "Test Agent"}

        mock_response = Mock()
        mock_response.status_code = 201
        sync_manager._session.post.return_value = mock_response

        result = sync_manager._sync_agent(agent_data)

        assert result["success"] is True
        assert result["agent_id"] == "agent_123"

    def test_sync_agent_already_exists(self, sync_manager: SyncManager) -> None:
        """Test agent sync when agent already exists (409)."""
        agent_data = {"id": "agent_123", "name": "Test Agent"}

        mock_response = Mock()
        mock_response.status_code = 409  # Conflict
        sync_manager._session.post.return_value = mock_response

        result = sync_manager._sync_agent(agent_data)

        assert result["success"] is True
        assert result["agent_id"] == "agent_123"

    def test_sync_agent_failure(self, sync_manager: SyncManager) -> None:
        """Test agent sync failure."""
        agent_data = {"id": "agent_123", "name": "Test Agent"}

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        sync_manager._session.post.return_value = mock_response

        result = sync_manager._sync_agent(agent_data)

        assert result["success"] is False
        assert "HTTP 500" in result["error"]

    def test_sync_agent_exception(self, sync_manager: SyncManager) -> None:
        """Test agent sync handles exceptions."""
        agent_data = {"id": "agent_123", "name": "Test Agent"}
        sync_manager._session.post.side_effect = Exception("Network error")

        result = sync_manager._sync_agent(agent_data)

        assert result["success"] is False
        assert "Network error" in result["error"]

    def test_sync_benchmark_success(self, sync_manager: SyncManager) -> None:
        """Test successful benchmark sync."""
        benchmark_data = {"id": "bench_123", "name": "Test Benchmark"}

        mock_response = Mock()
        mock_response.status_code = 200
        sync_manager._session.post.return_value = mock_response

        result = sync_manager._sync_benchmark(benchmark_data)

        assert result["success"] is True
        assert result["benchmark_id"] == "bench_123"

    def test_sync_benchmark_failure(self, sync_manager: SyncManager) -> None:
        """Test benchmark sync failure."""
        benchmark_data = {"id": "bench_123", "name": "Test Benchmark"}

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        sync_manager._session.post.return_value = mock_response

        result = sync_manager._sync_benchmark(benchmark_data)

        assert result["success"] is False
        assert "HTTP 400" in result["error"]

    def test_sync_experiment_success(self, sync_manager: SyncManager) -> None:
        """Test successful experiment sync."""
        experiment_data = {"id": "exp_123", "name": "Test Experiment"}

        mock_response = Mock()
        mock_response.status_code = 201
        sync_manager._session.post.return_value = mock_response

        result = sync_manager._sync_experiment(experiment_data)

        assert result["success"] is True
        assert result["experiment_id"] == "exp_123"

    def test_sync_experiment_failure(self, sync_manager: SyncManager) -> None:
        """Test experiment sync failure."""
        experiment_data = {"id": "exp_123", "name": "Test Experiment"}

        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"
        sync_manager._session.post.return_value = mock_response

        result = sync_manager._sync_experiment(experiment_data)

        assert result["success"] is False
        assert "HTTP 403" in result["error"]

    def test_sync_experiment_run_success(self, sync_manager: SyncManager) -> None:
        """Test successful experiment run sync."""
        run_data = {"id": "run_123", "experiment_id": "exp_123"}

        mock_response = Mock()
        mock_response.status_code = 200
        sync_manager._session.post.return_value = mock_response

        result = sync_manager._sync_experiment_run(run_data)

        assert result["success"] is True
        assert result["run_id"] == "run_123"

    def test_sync_experiment_run_failure(self, sync_manager: SyncManager) -> None:
        """Test experiment run sync failure."""
        run_data = {"id": "run_123", "experiment_id": "exp_123"}

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        sync_manager._session.post.return_value = mock_response

        result = sync_manager._sync_experiment_run(run_data)

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
