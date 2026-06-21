"""
Comprehensive test suite for SyncManager.
Tests data conversion, API interactions, and sync operations.
"""

import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from traigent.cloud.sync_manager import SyncManager, sanitize_backend_name
from traigent.config.backend_config import BackendConfig
from traigent.config.types import TraigentConfig
from traigent.storage.local_storage import LocalStorageManager, TrialResult
from traigent.utils.exceptions import TraigentStorageError

VALID_AGENT_TYPE_IDS = {
    "chat",
    "classification",
    "completion",
    "qa",
    "retrieval",
    "tool",
    "function",
}
BACKEND_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9 _-]+$")


def backend_response(status_code=201, payload=None, text="Created"):
    response = Mock(status_code=status_code, text=text)
    response.json.return_value = (
        payload if payload is not None else {"id": "created-id"}
    )
    return response


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
        assert "benchmark" in traigent_data
        assert "experiment" in traigent_data
        assert "experiment_run" in traigent_data
        assert "configuration_runs" in traigent_data
        assert "model_parameters" not in traigent_data

        # Check agent data
        agent = traigent_data["agent"]
        assert agent == {
            "name": "Local Agent: test_llm_function",
            "agent_type_id": "completion",
        }

        # Check benchmark data
        benchmark = traigent_data["benchmark"]
        assert benchmark["name"] == "Local Dataset test_llm_function"
        assert BACKEND_NAME_PATTERN.fullmatch(benchmark["name"])
        assert benchmark["type"] == "input-output"
        assert benchmark["label"]

        # Check experiment template data; returned backend IDs are filled at sync time.
        experiment = traigent_data["experiment"]
        assert experiment["name"] == "Local Import: test_llm_function"
        assert experiment["status"] == "COMPLETED"
        assert experiment["measures"]
        assert "agent_id" not in experiment
        assert "dataset_id" not in experiment
        assert "model_parameters_id" not in experiment

        experiment_payload = self.sync_manager._build_experiment_payload(
            experiment, "agent-123", "benchmark-456"
        )
        assert experiment_payload["agent_id"] == "agent-123"
        assert experiment_payload["dataset_id"] == "benchmark-456"

        # Check experiment run and configuration-run payloads
        experiment_run = self.sync_manager._build_experiment_run_payload(
            traigent_data["experiment_run"], "agent-123", "benchmark-456"
        )
        assert set(experiment_run) == {"experiment_data"}
        assert experiment_run["experiment_data"] == {
            "agent_id": "agent-123",
            "benchmark_id": "benchmark-456",
            "measures": traigent_data["experiment_run"]["measures"],
            "configurations": traigent_data["experiment_run"]["configurations"],
        }
        assert len(traigent_data["configuration_runs"]) == 4
        assert {
            "experiment_parameters",
            "measures",
            "status",
        } == set(traigent_data["configuration_runs"][0])
        assert traigent_data["configuration_runs"][0]["status"] == "COMPLETED"

    @pytest.mark.parametrize(
        "method_name,resource_path,payload_key",
        [
            ("_sync_agent", "agents", "agent_id"),
            ("_sync_benchmark", "datasets", "benchmark_id"),
            ("_sync_experiment", "experiments", "experiment_id"),
            ("_sync_experiment_run", "experiment-runs", "run_id"),
        ],
    )
    def test_sync_operations_use_timeout(self, method_name, resource_path, payload_key):
        """Ensure blocking HTTP calls include an explicit timeout."""
        sync_manager = self.sync_manager
        payload = {"name": "resource"}
        if method_name == "_sync_experiment_run":
            payload = {
                "experiment_data": {
                    "agent_id": "agent-123",
                    "benchmark_id": "benchmark-123",
                    "measures": ["score"],
                    "configurations": {},
                }
            }
        mock_response = backend_response(payload={"id": "resource-123"}, text="created")
        original_post = sync_manager._session.post
        original_get = sync_manager._session.get

        try:
            sync_manager._session.post = Mock(return_value=mock_response)
            sync_manager._session.get = Mock(return_value=backend_response(404))
            method = getattr(sync_manager, method_name)
            if method_name == "_sync_experiment_run":
                result = method("experiment-123", payload)
            else:
                result = method(payload)

            expected_url = (
                f"{sync_manager.base_url}/experiment-runs/experiment-123/runs"
                if method_name == "_sync_experiment_run"
                else f"{sync_manager.base_url}/{resource_path}"
            )
            sync_manager._session.post.assert_called_once_with(
                expected_url,
                json=payload,
                timeout=sync_manager._request_timeout,
            )
            assert result["success"] is True
            assert payload_key in result
        finally:
            sync_manager._session.post = original_post
            sync_manager._session.get = original_get

    def test_sync_benchmark_posts_to_canonical_route(self):
        """Benchmark sync uses the canonical current backend route."""
        sync_manager = self.sync_manager
        payload = {"name": "Dataset 123", "type": "input-output", "label": "eval"}
        original_post = sync_manager._session.post
        original_get = sync_manager._session.get

        try:
            sync_manager._session.post = Mock(
                return_value=backend_response(payload={"id": "benchmark-123"})
            )
            sync_manager._session.get = Mock(return_value=backend_response(404))
            result = sync_manager._sync_benchmark(payload)

            assert result["success"] is True
            assert result["dataset_id"] == "benchmark-123"
            assert result["benchmark_id"] == "benchmark-123"
            sync_manager._session.post.assert_called_once_with(
                f"{sync_manager.base_url}/datasets",
                json=payload,
                timeout=sync_manager._request_timeout,
            )
        finally:
            sync_manager._session.post = original_post
            sync_manager._session.get = original_get

    def test_convert_session_payloads_match_backend_contract(self):
        """Generated sync payloads satisfy current backend validation contracts."""
        sanitized_name = sanitize_backend_name("Local Dataset: fn/with?punctuation")
        assert sanitized_name == "Local Dataset fn with punctuation"
        assert BACKEND_NAME_PATTERN.fullmatch(sanitized_name)

        storage = self.sync_manager.storage
        session_id = storage.create_session("fn:with/slash?and*punctuation")
        storage.add_trial_result(
            session_id,
            {"param": 1},
            0.91,
            metadata={"total_cost": 0.17},
        )
        storage.finalize_session(session_id, "completed")
        session = storage.load_session(session_id)

        traigent_data = self.sync_manager.convert_session_to_traigent_format(session)

        benchmark_name = traigent_data["benchmark"]["name"]
        assert BACKEND_NAME_PATTERN.fullmatch(benchmark_name)
        assert ":" not in benchmark_name

        agent = traigent_data["agent"]
        assert "agent_type" not in agent
        assert agent["agent_type_id"] in VALID_AGENT_TYPE_IDS
        assert set(agent) == {"name", "agent_type_id"}

        benchmark = traigent_data["benchmark"]
        assert benchmark["type"] == "input-output"
        assert benchmark["label"]

        experiment_run = self.sync_manager._build_experiment_run_payload(
            traigent_data["experiment_run"], "agent-id", "benchmark-id"
        )
        assert set(experiment_run["experiment_data"]) == {
            "agent_id",
            "benchmark_id",
            "measures",
            "configurations",
        }

    def test_sync_experiment_run_posts_to_nested_backend_route(self):
        """Experiment run creation uses the route exposed by the backend."""
        run_data = {
            "experiment_data": {
                "agent_id": "agent-123",
                "benchmark_id": "benchmark-123",
                "measures": ["score"],
                "configurations": {},
            }
        }
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_response = backend_response(payload={"id": "run-123"})
            mock_post.return_value = mock_response

            result = self.sync_manager._sync_experiment_run("test_experiment", run_data)

            assert result["success"] is True
            mock_post.assert_called_once_with(
                f"{self.sync_manager.base_url}/experiment-runs/test_experiment/runs",
                json=run_data,
                timeout=self.sync_manager._request_timeout,
            )

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
        self.sync_manager._session.get = Mock(return_value=backend_response(404))
        mock_post.side_effect = [
            backend_response(payload={"id": "agent-id"}),
            backend_response(payload={"id": "benchmark-id"}),
            backend_response(payload={"id": "experiment-id"}),
            backend_response(payload={"id": "experiment-run-id"}),
            *[
                backend_response(payload={"id": f"configuration-run-{index}"})
                for index in range(4)
            ],
        ]

        result = self.sync_manager.sync_session_to_cloud(
            self.test_session.session_id, dry_run=False
        )

        assert result["status"] == "success"
        assert len(result["errors"]) == 0
        assert "cloud_url" in result

        # agent, benchmark, experiment, experiment_run, and one config row per trial.
        assert mock_post.call_count == 8

    def test_sync_session_to_cloud_contract_payloads_with_cost(self):
        """A full mocked sync emits corrected URLs and a cost-bearing run body."""
        storage = self.sync_manager.storage
        session_id = storage.create_session("contract:sync/fn")
        storage.add_trial_result(
            session_id,
            {"model": "gpt-test", "temperature": 0.2},
            0.88,
            metadata={"total_cost": 0.42, "input_cost": 0.2, "output_cost": 0.22},
        )
        storage.finalize_session(session_id, "completed")
        original_post = self.sync_manager._session.post
        original_get = self.sync_manager._session.get

        try:
            self.sync_manager._session.get = Mock(return_value=backend_response(404))
            self.sync_manager._session.post = Mock(
                side_effect=[
                    backend_response(payload={"id": "agent-id"}),
                    backend_response(payload={"id": "benchmark-id"}),
                    backend_response(payload={"id": "experiment-id"}),
                    backend_response(payload={"id": "experiment-run-id"}),
                    backend_response(payload={"id": "configuration-run-id"}),
                ]
            )
            result = self.sync_manager.sync_session_to_cloud(session_id, dry_run=False)
        finally:
            post_mock = self.sync_manager._session.post
            self.sync_manager._session.post = original_post
            self.sync_manager._session.get = original_get

        assert result["status"] == "success"
        assert post_mock.call_count == 5

        calls = post_mock.call_args_list
        assert [call.args[0] for call in calls] == [
            f"{self.sync_manager.base_url}/agents",
            f"{self.sync_manager.base_url}/datasets",
            f"{self.sync_manager.base_url}/experiments",
            f"{self.sync_manager.base_url}/experiment-runs/experiment-id/runs",
            f"{self.sync_manager.base_url}/configuration-runs/runs/"
            "experiment-run-id/configurations",
        ]

        agent_payload = calls[0].kwargs["json"]
        assert agent_payload == {
            "name": "Local Agent: contract:sync/fn",
            "agent_type_id": "completion",
        }

        benchmark_payload = calls[1].kwargs["json"]
        assert BACKEND_NAME_PATTERN.fullmatch(benchmark_payload["name"])
        assert benchmark_payload["type"] == "input-output"
        assert benchmark_payload["label"]

        experiment_payload = calls[2].kwargs["json"]
        assert experiment_payload["agent_id"] == "agent-id"
        assert experiment_payload["dataset_id"] == "benchmark-id"
        assert experiment_payload["status"] == "COMPLETED"
        assert experiment_payload["measures"]
        assert "model_parameters_id" not in experiment_payload

        run_payload = calls[3].kwargs["json"]
        assert set(run_payload) == {"experiment_data"}
        assert run_payload["experiment_data"] == {
            "agent_id": "agent-id",
            "benchmark_id": "benchmark-id",
            "measures": experiment_payload["measures"],
            "configurations": experiment_payload["configurations"],
        }

        configuration_payload = calls[4].kwargs["json"]
        assert configuration_payload["experiment_parameters"] == {
            "model": "gpt-test",
            "temperature": 0.2,
        }
        assert configuration_payload["status"] == "COMPLETED"
        assert isinstance(configuration_payload["measures"]["cost"], float)
        assert configuration_payload["measures"]["cost"] == 0.42
        assert configuration_payload["measures"]["total_cost"] == 0.42
        assert configuration_payload["measures"]["input_cost"] == 0.2
        assert configuration_payload["measures"]["output_cost"] == 0.22

    @patch("requests.Session.post")
    def test_sync_session_to_cloud_partial_failure(self, mock_post):
        """Test session sync with partial failures."""
        # Mock mixed success/failure responses
        responses = [
            backend_response(payload={"id": "agent-id"}),  # Agent success
            backend_response(status_code=400, text="Bad Request"),  # Benchmark failure
        ]
        self.test_session = self._create_test_session()
        self.sync_manager._session.get = Mock(return_value=backend_response(404))
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
        self.sync_manager._session.get = Mock(return_value=backend_response(404))
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
        self.sync_manager._session.get = Mock(return_value=backend_response(404))
        mock_post.return_value = backend_response()

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

        self.sync_manager._session.get = Mock(return_value=backend_response(404))
        success_response = backend_response()
        failure_response = backend_response(status_code=400, text="Bad Request")

        mock_post.side_effect = [
            *[success_response for _ in range(8)],  # First session
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
        agent_data = {"name": "Test Agent", "agent_type_id": "completion"}

        with (
            patch.object(self.sync_manager.session, "get") as mock_get,
            patch.object(self.sync_manager.session, "post") as mock_post,
        ):
            mock_get.return_value = backend_response(404)
            mock_post.return_value = backend_response(payload={"id": "test_agent"})

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
        """Test agent sync reuses an existing matching agent."""
        agent_data = {"name": "Existing Agent", "agent_type_id": "completion"}

        with (
            patch.object(self.sync_manager.session, "get") as mock_get,
            patch.object(self.sync_manager.session, "post") as mock_post,
        ):
            mock_get.return_value = backend_response(
                200,
                payload={
                    "agents": [{"id": "existing_agent", "name": "Existing Agent"}]
                },
            )

            result = self.sync_manager._sync_agent(agent_data)

            assert result["success"] is True
            assert result["agent_id"] == "existing_agent"
            assert result["reused"] is True
            mock_post.assert_not_called()

    def test_sync_agent_failure(self):
        """Test agent sync failure."""
        agent_data = {"name": "Test Agent", "agent_type_id": "completion"}

        with (
            patch.object(self.sync_manager.session, "get") as mock_get,
            patch.object(self.sync_manager.session, "post") as mock_post,
        ):
            mock_get.return_value = backend_response(404)
            mock_post.return_value = backend_response(
                status_code=400, text="Validation error"
            )

            result = self.sync_manager._sync_agent(agent_data)

            assert result["success"] is False
            assert "HTTP 400: Validation error" in result["error"]

    def test_sync_agent_exception(self):
        """Test agent sync with network exception."""
        agent_data = {"name": "Test Agent", "agent_type_id": "completion"}

        with (
            patch.object(self.sync_manager.session, "get") as mock_get,
            patch.object(self.sync_manager.session, "post") as mock_post,
        ):
            mock_get.return_value = backend_response(404)
            mock_post.side_effect = requests.ConnectionError("Network error")

            result = self.sync_manager._sync_agent(agent_data)

            assert result["success"] is False
            assert "Network error" in result["error"]

    def test_sync_benchmark_operations(self):
        """Test benchmark sync operations."""
        benchmark_data = {
            "name": "Test Benchmark",
            "type": "input-output",
            "label": "eval",
        }

        # Test success
        with (
            patch.object(self.sync_manager.session, "get") as mock_get,
            patch.object(self.sync_manager.session, "post") as mock_post,
        ):
            mock_get.return_value = backend_response(404)
            mock_post.return_value = backend_response(payload={"id": "test_benchmark"})

            result = self.sync_manager._sync_benchmark(benchmark_data)
            assert result["success"] is True

        # Test failure
        with (
            patch.object(self.sync_manager.session, "get") as mock_get,
            patch.object(self.sync_manager.session, "post") as mock_post,
        ):
            mock_get.return_value = backend_response(404)
            mock_post.return_value = backend_response(
                status_code=400, text="Bad request"
            )

            result = self.sync_manager._sync_benchmark(benchmark_data)
            assert result["success"] is False

    def test_sync_experiment_operations(self):
        """Test experiment sync operations."""
        experiment_data = {
            "name": "Test Experiment",
            "agent_id": "agent-123",
            "dataset_id": "benchmark-123",
            "measures": ["score"],
            "configurations": {},
            "status": "COMPLETED",
        }

        # Test success
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = backend_response(payload={"id": "test_experiment"})

            result = self.sync_manager._sync_experiment(experiment_data)
            assert result["success"] is True

    def test_sync_experiment_run_operations(self):
        """Test experiment run sync operations."""
        run_data = {
            "experiment_data": {
                "agent_id": "agent-123",
                "benchmark_id": "benchmark-123",
                "measures": ["score"],
                "configurations": {},
            }
        }

        # Test success
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = backend_response(payload={"id": "test_run"})

            result = self.sync_manager._sync_experiment_run("test_experiment", run_data)
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
        assert len(traigent_data["configuration_runs"]) == 0
        assert traigent_data["experiment_run"]["measures"] == ["score"]

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
        assert len(traigent_data["configuration_runs"]) == 100

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
            traigent_data["configuration_runs"][0]["experiment_parameters"][
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
        with (
            patch.object(self.sync_manager.session, "get") as mock_get,
            patch.object(self.sync_manager.session, "post") as mock_post,
        ):
            mock_get.return_value = backend_response(404)
            mock_post.return_value = backend_response()

            # Make multiple sync calls
            self.sync_manager._sync_agent(
                {"name": "agent1", "agent_type_id": "completion"}
            )
            self.sync_manager._sync_benchmark(
                {"name": "benchmark1", "type": "input-output", "label": "eval"}
            )

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

    # --- #1344 regression tests ---

    def test_is_finished_session_completed(self):
        """Completed sessions are always eligible for sync."""
        session = self._create_test_session()
        assert self.sync_manager._is_finished_session(session) is True

    def test_is_finished_session_pending_no_trials(self):
        """Pending sessions with no trials are not eligible."""
        storage = self.sync_manager.storage
        sid = storage.create_session("no_trials_func")
        s = storage.load_session(sid)
        assert s is not None
        assert self.sync_manager._is_finished_session(s) is False

    def test_is_finished_session_failed(self):
        """Failed sessions are not eligible (they may be retried separately)."""
        storage = self.sync_manager.storage
        sid = storage.create_session("failed_func")
        storage.add_trial_result(sid, {"p": 1}, 0.5)
        storage.finalize_session(sid, "failed")
        s = storage.load_session(sid)
        assert s is not None
        assert self.sync_manager._is_finished_session(s) is False

    def test_is_finished_session_pending_with_stop_reason(self):
        """Pending session with completed_trials + stop_reason is treated as finished (pre-fix compat)."""
        storage = self.sync_manager.storage
        sid = storage.create_session(
            "stuck_func",
            metadata={"stop_reason": "max_trials_reached"},
        )
        storage.add_trial_result(sid, {"p": 1}, 0.7)
        storage.add_trial_result(sid, {"p": 2}, 0.8)
        # Do NOT call finalize_session — simulate pre-fix SDK leaving status="pending"
        s = storage.load_session(sid)
        assert s is not None
        assert s.status == "pending"
        assert s.completed_trials == 2
        assert self.sync_manager._is_finished_session(s) is True

    def test_get_sync_status_includes_stuck_pending_sessions(self):
        """get_sync_status counts stuck-pending finished sessions in sync_eligible (#1344)."""
        storage = self.sync_manager.storage
        # Stuck session: pending status but has trials and stop_reason
        sid = storage.create_session(
            "stuck_session",
            metadata={"stop_reason": "max_trials_reached"},
        )
        storage.add_trial_result(sid, {"p": 1}, 0.9)

        status = self.sync_manager.get_sync_status()
        assert status["sync_eligible"] >= 1
        assert status["completed_sessions"] >= 1

    def test_sync_all_includes_stuck_pending_sessions(self):
        """sync_all_sessions picks up stuck-pending sessions as eligible (#1344)."""
        storage = self.sync_manager.storage
        sid = storage.create_session(
            "stuck_sync_func",
            metadata={"stop_reason": "budget_exhausted"},
        )
        storage.add_trial_result(sid, {"p": 1}, 0.75)
        # status remains "pending"

        result = self.sync_all_with_no_api_key()
        session_ids = [r["session_id"] for r in result["session_results"]]
        assert sid in session_ids

    def sync_all_with_no_api_key(self) -> dict:
        """Helper: run sync_all in dry_run to capture which sessions are picked up."""
        return self.sync_manager.sync_all_sessions(dry_run=True)
