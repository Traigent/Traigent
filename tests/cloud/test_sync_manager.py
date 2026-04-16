"""
Comprehensive test suite for SyncManager.
Tests data conversion, API interactions, and sync operations.
"""

import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from traigent.cloud.sync_manager import SyncManager
from traigent.config.backend_config import BackendConfig
from traigent.config.types import TraigentConfig
from traigent.storage.local_storage import LocalStorageManager, TrialResult
from traigent.utils.exceptions import TraigentStorageError


class TestSyncManager:
    """Test suite for SyncManager with comprehensive data conversion and sync tests."""

    def setup_method(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)

        # Create test config
        self.config = TraigentConfig(
            execution_mode="edge_analytics",
            local_storage_path=str(self.storage_path),
            minimal_logging=True,
        )

        self.api_key = "test_api_key_123"  # pragma: allowlist secret
        self.sync_manager = SyncManager(self.config, self.api_key)

        # Don't create test session by default - tests will create as needed
        self.test_session = None

    def teardown_method(self):
        """Clean up temporary storage."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_session(self):
        """Create a test optimization session with trials."""
        storage = LocalStorageManager(str(self.storage_path))

        session_id = storage.create_session(
            function_name="test_llm_function",
            optimization_config={
                "search_space": {
                    "model": ["gpt-4", "gpt-3.5-turbo"],
                    "temperature": {"min": 0.0, "max": 1.0},
                    "max_tokens": [100, 200, 500],
                },
                "optimization_goal": "maximize",
                "baseline_config": {"model": "gpt-3.5-turbo", "temperature": 0.7},
            },
            metadata={"user": "test_user", "version": "1.0.0"},
        )

        # Add trial results
        trials = [
            ({"model": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 200}, 0.75),
            ({"model": "gpt-4", "temperature": 0.5, "max_tokens": 100}, 0.82),
            ({"model": "gpt-4", "temperature": 0.9, "max_tokens": 500}, 0.78),
            ({"model": "gpt-3.5-turbo", "temperature": 0.3, "max_tokens": 200}, 0.85),
        ]

        for config, score in trials:
            storage.add_trial_result(
                session_id,
                config,
                score,
                metadata={"duration": 2.5, "tokens_used": 150},
            )

        storage.finalize_session(session_id, "completed")
        return storage.load_session(session_id)

    def test_initialization_with_api_key(self):
        """Test sync manager initialization with API key."""
        assert self.sync_manager.config == self.config
        assert self.sync_manager.api_key == self.api_key
        expected_base_url = BackendConfig.get_cloud_api_url().rstrip("/")
        assert self.sync_manager.base_url == expected_base_url
        assert self.sync_manager.storage is not None
        # API key may be in X-API-Key or Authorization header
        headers = self.sync_manager.session.headers
        assert "X-API-Key" in headers or "Authorization" in headers

    def test_initialization_without_api_key(self):
        """Test sync manager initialization without API key."""
        sync_manager = SyncManager(self.config)
        assert sync_manager.api_key is None
        headers = sync_manager.session.headers
        assert "Authorization" not in headers and "X-API-Key" not in headers

    def test_get_sync_status_empty(self):
        """Test getting sync status with no sessions."""
        status = self.sync_manager.get_sync_status()

        assert status["total_sessions"] == 0
        assert status["completed_sessions"] == 0
        assert status["total_trials"] == 0
        assert status["sync_eligible"] == 0
        assert "estimated_cloud_value" in status
        assert "storage_info" in status

    def test_get_sync_status_with_sessions(self):
        """Test getting sync status with actual sessions."""
        # Create test session for this test
        self.test_session = self._create_test_session()
        status = self.sync_manager.get_sync_status()

        assert status["total_sessions"] == 1
        assert status["completed_sessions"] == 1
        assert status["total_trials"] == 4
        assert status["sync_eligible"] == 1

        # Check estimated cloud value
        cloud_value = status["estimated_cloud_value"]
        assert cloud_value["sessions_to_sync"] == 1
        assert cloud_value["average_trials_per_session"] == 4.0
        assert "cloud_benefits" in cloud_value

    def test_estimate_cloud_value(self):
        """Test cloud value estimation."""
        self.test_session = self._create_test_session()
        sessions = [self.test_session]
        cloud_value = self.sync_manager._estimate_cloud_value(sessions)

        assert cloud_value["sessions_to_sync"] == 1
        assert cloud_value["average_trials_per_session"] == 4.0
        assert cloud_value["estimated_time_invested_hours"] == 0.25  # 15 min / 60

        benefits = cloud_value["cloud_benefits"]
        assert "advanced_algorithms" in benefits
        assert "web_dashboard" in benefits
        assert "team_collaboration" in benefits
        assert "vs. current 4 trial average" in benefits["unlimited_trials"]

    def test_convert_session_to_traigent_format(self):
        """Test converting local session to Traigent format."""
        self.test_session = self._create_test_session()
        traigent_data = self.sync_manager.convert_session_to_traigent_format(
            self.test_session
        )

        # Check structure
        assert "agent" in traigent_data
        assert "dataset" in traigent_data
        assert "benchmark" in traigent_data
        assert "model_parameters" in traigent_data
        assert "experiment" in traigent_data
        assert "experiment_run" in traigent_data

        # Check agent data
        agent = traigent_data["agent"]
        assert agent["name"] == "Local Agent: test_llm_function"
        assert agent["agent_type"] == "custom"
        assert agent["source"] == "local_import"

        # Check dataset data
        dataset = traigent_data["dataset"]
        assert dataset["name"] == "Local Dataset: test_llm_function"
        assert dataset["dataset_id"] == dataset["id"]
        assert dataset["benchmark_id"] == dataset["id"]
        assert dataset["type"] == "custom"
        assert dataset["examples_count"] == 4

        # Legacy alias remains available for older callers
        benchmark = traigent_data["benchmark"]
        assert benchmark == dataset

        # Check experiment data
        experiment = traigent_data["experiment"]
        assert experiment["name"] == "Local Import: test_llm_function"
        assert experiment["agent_id"] == agent["id"]
        assert experiment["dataset_id"] == dataset["id"]
        assert experiment["benchmark_id"] == dataset["id"]
        assert experiment["evaluation_set_id"] == dataset["id"]
        assert experiment["eval_dataset_id"] == dataset["id"]
        assert experiment["status"] == "completed"
        assert "original_session_id" in experiment["metadata"]

        # Check experiment run data
        experiment_run = traigent_data["experiment_run"]
        assert experiment_run["experiment_id"] == experiment["id"]
        assert experiment_run["status"] == "completed"
        assert len(experiment_run["results"]) == 4
        assert experiment_run["metadata"]["total_trials"] == 4
        assert experiment_run["metadata"]["best_score"] == 0.85

    @pytest.mark.parametrize(
        "method_name,resource_path,payload_key",
        [
            ("_sync_agent", "agents", "agent_id"),
            ("_sync_benchmark", "datasets", "dataset_id"),
            ("_sync_experiment", "experiments", "experiment_id"),
            ("_sync_experiment_run", "experiment-runs", "run_id"),
        ],
    )
    def test_sync_operations_use_timeout(self, method_name, resource_path, payload_key):
        """Ensure blocking HTTP calls include an explicit timeout."""
        sync_manager = self.sync_manager
        payload = {"id": "resource-123"}
        mock_response = Mock(status_code=201, text="created")
        original_post = sync_manager._session.post

        try:
            sync_manager._session.post = Mock(return_value=mock_response)
            method = getattr(sync_manager, method_name)
            result = method(payload)

            expected_url = f"{sync_manager.base_url}/{resource_path}"
            sync_manager._session.post.assert_called_once_with(
                expected_url,
                json=payload,
                timeout=sync_manager._request_timeout,
            )
            assert result["success"] is True
            assert payload_key in result
        finally:
            sync_manager._session.post = original_post

    def test_sync_benchmark_falls_back_to_legacy_route(self):
        """Test dataset sync falls back to legacy benchmark route on 404."""
        sync_manager = self.sync_manager
        payload = {"id": "dataset-123", "dataset_id": "dataset-123"}
        responses = [
            Mock(status_code=404, text="not found"),
            Mock(status_code=201, text="created"),
        ]
        original_post = sync_manager._session.post
        original_get = sync_manager._session.get

        try:
            sync_manager._session.post = Mock(side_effect=responses)
            sync_manager._session.get = Mock(return_value=Mock(status_code=404, text="missing"))
            result = sync_manager._sync_benchmark(payload)

            assert result["success"] is True
            assert result["dataset_id"] == "dataset-123"
            assert result["benchmark_id"] == "dataset-123"
            assert sync_manager._session.post.call_count == 2
            first_call = sync_manager._session.post.call_args_list[0]
            second_call = sync_manager._session.post.call_args_list[1]
            assert first_call.args[0] == f"{sync_manager.base_url}/datasets"
            assert second_call.args[0] == f"{sync_manager.base_url}/benchmarks"
        finally:
            sync_manager._session.post = original_post
            sync_manager._session.get = original_get

    def test_convert_trials_to_results(self):
        """Test converting local trials to Traigent configuration_run format."""
        self.test_session = self._create_test_session()
        trials = self.test_session.trials
        results = self.sync_manager._convert_trials_to_results(trials)

        assert len(results) == 4

        # Check first result
        result = results[0]
        assert result["trial_id"] == trials[0].trial_id
        assert result["experiment_parameters"] == trials[0].config
        assert result["measures"]["score"] == trials[0].score
        assert result["measures"]["accuracy"] == trials[0].score  # Mapped
        assert result["status"] == "completed"

        # Check result with error (if any)
        # Add a trial with error for testing
        error_trial = TrialResult(
            trial_id="error_trial",
            config={"model": "invalid"},
            score=0.0,
            timestamp=datetime.now().isoformat(),
            error="Configuration error",
        )

        error_results = self.sync_manager._convert_trials_to_results([error_trial])
        error_result = error_results[0]
        assert error_result["status"] == "failed"
        assert error_result["error"] == "Configuration error"

    def test_sync_session_to_cloud_dry_run(self):
        """Test syncing session in dry run mode."""
        self.test_session = self._create_test_session()
        result = self.sync_manager.sync_session_to_cloud(
            self.test_session.session_id, dry_run=True
        )

        assert result["status"] == "success"
        assert result["dry_run"] is True
        assert result["data_converted"] is True
        assert result["trials_converted"] == 4
        assert "preview" in result

        preview = result["preview"]
        assert preview["experiment_name"] == "Local Import: test_llm_function"
        assert preview["trial_count"] == 4
        assert preview["best_score"] == 0.85

    def test_sync_session_to_cloud_no_api_key(self):
        """Test syncing session without API key."""
        self.test_session = self._create_test_session()
        sync_manager = SyncManager(self.config)  # No API key

        result = sync_manager.sync_session_to_cloud(
            self.test_session.session_id, dry_run=False
        )

        assert result["status"] == "error"
        assert "No API key provided" in result["errors"][0]

    def test_sync_session_to_cloud_nonexistent(self):
        """Test syncing non-existent session."""
        with pytest.raises(TraigentStorageError, match="Session .* not found"):
            self.sync_manager.sync_session_to_cloud("fake_session_id")

    @patch("requests.Session.post")
    def test_sync_session_to_cloud_success(self, mock_post):
        """Test successful session sync to cloud."""
        # Mock successful API responses
        self.test_session = self._create_test_session()
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.text = "Created"
        mock_post.return_value = mock_response

        result = self.sync_manager.sync_session_to_cloud(
            self.test_session.session_id, dry_run=False
        )

        assert result["status"] == "success"
        assert len(result["errors"]) == 0
        assert "cloud_url" in result

        # Should have made 4 API calls (agent, benchmark, experiment, experiment_run)
        assert mock_post.call_count == 4

    @patch("requests.Session.post")
    def test_sync_session_to_cloud_partial_failure(self, mock_post):
        """Test session sync with partial failures."""
        # Mock mixed success/failure responses
        responses = [
            Mock(status_code=201, text="Created"),  # Agent success
            Mock(status_code=400, text="Bad Request"),  # Benchmark failure
            Mock(status_code=201, text="Created"),  # Experiment success
            Mock(status_code=201, text="Created"),  # Run success
        ]
        self.test_session = self._create_test_session()
        mock_post.side_effect = responses

        result = self.sync_manager.sync_session_to_cloud(
            self.test_session.session_id, dry_run=False
        )

        assert result["status"] == "partial"
        assert len(result["errors"]) == 1
        assert "Benchmark sync failed" in result["errors"][0]

    @patch("requests.Session.post")
    def test_sync_session_to_cloud_complete_failure(self, mock_post):
        """Test session sync with complete failure."""
        self.test_session = self._create_test_session()
        mock_post.side_effect = Exception("Network error")

        result = self.sync_manager.sync_session_to_cloud(
            self.test_session.session_id, dry_run=False
        )

        assert result["status"] == "error"
        assert "Network error" in result["errors"][0]

    def test_sync_all_sessions_empty(self):
        """Test syncing all sessions when none exist."""
        # Don't create any sessions for this test

        result = self.sync_manager.sync_all_sessions(dry_run=True)

        assert result["total_sessions"] == 0
        assert result["eligible_sessions"] == 0
        assert result["synced_successfully"] == 0
        assert result["sync_errors"] == 0
        assert result["overall_status"] == "success"

    @patch("requests.Session.post")
    def test_sync_all_sessions_success(self, mock_post):
        """Test successfully syncing all sessions."""
        # Create test session for this test
        self.test_session = self._create_test_session()
        # Mock successful API responses
        mock_response = Mock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        result = self.sync_manager.sync_all_sessions(dry_run=False)

        assert result["total_sessions"] == 1
        assert result["eligible_sessions"] == 1
        assert result["synced_successfully"] == 1
        assert result["sync_errors"] == 0
        assert result["overall_status"] == "success"
        assert len(result["session_results"]) == 1

    @patch("requests.Session.post")
    def test_sync_all_sessions_mixed_results(self, mock_post):
        """Test syncing all sessions with mixed results."""
        # Create first session that will succeed
        self.test_session = self._create_test_session()

        # Create another session that will fail
        storage = self.sync_manager.storage
        failed_session_id = storage.create_session("failed_function")
        storage.finalize_session(failed_session_id, "completed")

        # Mock responses - first session succeeds, second fails
        success_response = Mock(status_code=201, text="Created")
        failure_response = Mock(status_code=400, text="Bad Request")

        mock_post.side_effect = [
            success_response,
            success_response,
            success_response,
            success_response,  # First session
            failure_response,  # Second session fails on first call
        ]

        result = self.sync_manager.sync_all_sessions(dry_run=False)

        assert result["total_sessions"] == 2
        assert result["eligible_sessions"] == 2
        assert result["synced_successfully"] == 1
        assert result["sync_errors"] == 1
        assert result["overall_status"] == "partial"

    def test_sync_agent_success(self):
        """Test successful agent sync."""
        agent_data = {"id": "test_agent", "name": "Test Agent"}

        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_post.return_value = mock_response

            result = self.sync_manager._sync_agent(agent_data)

            assert result["success"] is True
            assert result["agent_id"] == "test_agent"
            expected_endpoint = (
                f"{BackendConfig.get_cloud_api_url().rstrip('/')}/agents"
            )
            mock_post.assert_called_with(
                expected_endpoint,
                json=agent_data,
                timeout=self.sync_manager._request_timeout,
            )

    def test_sync_agent_already_exists(self):
        """Test agent sync when agent already exists (409 conflict)."""
        agent_data = {"id": "existing_agent", "name": "Existing Agent"}

        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 409  # Conflict
            mock_post.return_value = mock_response

            result = self.sync_manager._sync_agent(agent_data)

            assert result["success"] is True  # 409 is treated as success

    def test_sync_agent_failure(self):
        """Test agent sync failure."""
        agent_data = {"id": "test_agent", "name": "Test Agent"}

        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Validation error"
            mock_post.return_value = mock_response

            result = self.sync_manager._sync_agent(agent_data)

            assert result["success"] is False
            assert "HTTP 400: Validation error" in result["error"]

    def test_sync_agent_exception(self):
        """Test agent sync with network exception."""
        agent_data = {"id": "test_agent", "name": "Test Agent"}

        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.side_effect = requests.ConnectionError("Network error")

            result = self.sync_manager._sync_agent(agent_data)

            assert result["success"] is False
            assert "Network error" in result["error"]

    def test_sync_benchmark_operations(self):
        """Test benchmark sync operations."""
        benchmark_data = {"id": "test_benchmark", "name": "Test Benchmark"}

        # Test success
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_response = Mock(status_code=201)
            mock_post.return_value = mock_response

            result = self.sync_manager._sync_benchmark(benchmark_data)
            assert result["success"] is True

        # Test failure
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_response = Mock(status_code=500, text="Server error")
            mock_post.return_value = mock_response

            result = self.sync_manager._sync_benchmark(benchmark_data)
            assert result["success"] is False

    def test_sync_experiment_operations(self):
        """Test experiment sync operations."""
        experiment_data = {"id": "test_experiment", "name": "Test Experiment"}

        # Test success
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_response = Mock(status_code=201)
            mock_post.return_value = mock_response

            result = self.sync_manager._sync_experiment(experiment_data)
            assert result["success"] is True

    def test_sync_experiment_run_operations(self):
        """Test experiment run sync operations."""
        run_data = {"id": "test_run", "experiment_id": "test_experiment"}

        # Test success
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_response = Mock(status_code=200)
            mock_post.return_value = mock_response

            result = self.sync_manager._sync_experiment_run(run_data)
            assert result["success"] is True

    def test_get_cloud_analytics_preview_empty(self):
        """Test getting analytics preview with no completed sessions."""
        # Don't create any sessions for this test

        preview = self.sync_manager.get_cloud_analytics_preview()
        assert preview["message"] == "No completed sessions to analyze"

    def test_get_cloud_analytics_preview_with_data(self):
        """Test getting analytics preview with actual data."""
        self.test_session = self._create_test_session()
        preview = self.sync_manager.get_cloud_analytics_preview()

        assert preview["sessions_ready_for_sync"] == 1
        assert preview["total_optimization_trials"] == 4
        assert preview["functions_optimized"] == 1

        # Check cloud analytics available
        analytics = preview["cloud_analytics_available"]
        assert "cross_function_insights" in analytics
        assert "trend_analysis" in analytics
        assert "parameter_impact_analysis" in analytics

        # Check estimated dashboard value
        dashboard_value = preview["estimated_dashboard_value"]
        assert "5 minutes" in dashboard_value["time_saved_reviewing_results"]
        assert "4 trials" in dashboard_value["insights_gained"]

    def test_cleanup_after_sync_with_backup(self):
        """Test cleanup after sync with backup creation."""
        self.test_session = self._create_test_session()
        session_ids = [self.test_session.session_id]

        result = self.sync_manager.cleanup_after_sync(session_ids, keep_backup=True)

        assert result["sessions_processed"] == 1
        assert result["sessions_backed_up"] == 1
        assert result["sessions_deleted"] == 1
        assert len(result["errors"]) == 0

        # Verify session was deleted
        assert (
            self.sync_manager.storage.load_session(self.test_session.session_id) is None
        )

        # Verify backup was created
        backup_dirs = list(
            (self.sync_manager.storage.storage_path / "backups").glob("sync_backup_*")
        )
        assert len(backup_dirs) == 1

        backup_file = backup_dirs[0] / f"{self.test_session.session_id}.json"
        assert backup_file.exists()

    def test_cleanup_after_sync_no_backup(self):
        """Test cleanup after sync without backup."""
        self.test_session = self._create_test_session()
        session_ids = [self.test_session.session_id]

        result = self.sync_manager.cleanup_after_sync(session_ids, keep_backup=False)

        assert result["sessions_processed"] == 1
        assert result["sessions_backed_up"] == 0
        assert result["sessions_deleted"] == 1

    def test_cleanup_after_sync_with_errors(self):
        """Test cleanup with errors in backup/deletion."""
        session_ids = ["nonexistent_session"]

        result = self.sync_manager.cleanup_after_sync(session_ids, keep_backup=True)

        assert result["sessions_processed"] == 1
        assert result["sessions_backed_up"] == 0
        assert result["sessions_deleted"] == 0
        assert len(result["errors"]) > 0

    def test_edge_case_empty_session(self):
        """Test handling of session with no trials."""
        storage = self.sync_manager.storage
        empty_session_id = storage.create_session("empty_function")
        storage.finalize_session(empty_session_id, "completed")

        empty_session = storage.load_session(empty_session_id)
        traigent_data = self.sync_manager.convert_session_to_traigent_format(
            empty_session
        )

        # Should handle empty trials gracefully
        assert len(traigent_data["experiment_run"]["results"]) == 0
        assert traigent_data["experiment_run"]["metadata"]["total_trials"] == 0

    def test_edge_case_large_session(self):
        """Test handling of session with many trials."""
        storage = self.sync_manager.storage
        large_session_id = storage.create_session("large_function")

        # Add many trials
        for i in range(100):
            storage.add_trial_result(
                large_session_id, {"param": i}, 0.5 + (i % 50) * 0.01
            )

        storage.finalize_session(large_session_id, "completed")

        large_session = storage.load_session(large_session_id)
        traigent_data = self.sync_manager.convert_session_to_traigent_format(
            large_session
        )

        # Should handle large number of trials
        assert len(traigent_data["experiment_run"]["results"]) == 100
        assert traigent_data["experiment_run"]["metadata"]["total_trials"] == 100

    def test_edge_case_special_characters(self):
        """Test handling of special characters in session data."""
        storage = self.sync_manager.storage
        special_session_id = storage.create_session(
            "special_function_测试",
            optimization_config={
                "search_space": {"unicode_param": ["测试", "🎯", 'special"chars']}
            },
        )

        storage.add_trial_result(special_session_id, {"unicode_param": "测试"}, 0.8)

        storage.finalize_session(special_session_id, "completed")

        special_session = storage.load_session(special_session_id)
        traigent_data = self.sync_manager.convert_session_to_traigent_format(
            special_session
        )

        # Should handle special characters without issues
        assert "测试" in traigent_data["agent"]["name"]
        assert (
            traigent_data["experiment_run"]["results"][0]["experiment_parameters"][
                "unicode_param"
            ]
            == "测试"
        )

    def test_session_request_handling(self):
        """Test request session management and connection pooling."""
        # Verify session is properly configured
        assert isinstance(self.sync_manager.session, requests.Session)
        # API key may be in X-API-Key or Authorization header
        headers = self.sync_manager.session.headers
        assert "X-API-Key" in headers or "Authorization" in headers

        # Test that session is reused for multiple requests
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_response = Mock(status_code=201)
            mock_post.return_value = mock_response

            # Make multiple sync calls
            self.sync_manager._sync_agent({"id": "agent1"})
            self.sync_manager._sync_benchmark({"id": "benchmark1"})

            # Should use same session instance
            assert mock_post.call_count == 2

    def test_concurrent_sync_operations(self):
        """Test handling of concurrent sync operations."""
        # Create original test session
        self.test_session = self._create_test_session()

        # Create multiple sessions
        storage = self.sync_manager.storage
        session_ids = []

        for i in range(3):
            session_id = storage.create_session(f"concurrent_func_{i}")
            storage.add_trial_result(session_id, {"param": i}, 0.5 + i * 0.1)
            storage.finalize_session(session_id, "completed")
            session_ids.append(session_id)

        # Test sync status calculation
        status = self.sync_manager.get_sync_status()
        assert status["total_sessions"] == 4  # Including original test session
        assert status["completed_sessions"] == 4
        assert status["sync_eligible"] == 4

    def test_error_resilience(self):
        """Test error resilience and recovery mechanisms."""
        # Test with malformed session data
        self.test_session = self._create_test_session()
        session = self.test_session
        session.optimization_config = None  # Malformed config

        # Should handle gracefully without crashing
        traigent_data = self.sync_manager.convert_session_to_traigent_format(session)
        assert traigent_data is not None
        assert "experiment" in traigent_data
