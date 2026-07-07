"""
Comprehensive test suite for SyncManager.

Offline ``traigent sync`` uses the content-free typed-session endpoints:
``POST /sessions`` -> per-trial ``POST /sessions/{id}/results`` ->
``POST /sessions/{id}/finalize``. No agent/benchmark/experiment content
egresses, and a run whose server-side dataset would have zero examples now
imports cleanly. These tests exercise data conversion, the per-endpoint
helpers, and the end-to-end sync sequence against that contract.
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

BACKEND_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9 _-]+$")


def backend_response(status_code=201, payload=None, text="Created"):
    response = Mock(status_code=status_code, text=text)
    response.json.return_value = (
        payload if payload is not None else {"id": "created-id"}
    )
    return response


def create_session_response(
    session_id="sess-id",
    experiment_id="experiment-id",
    experiment_run_id="experiment-run-id",
    project_id=None,
    tenant_id=None,
    context_in_metadata=False,
):
    """Build a `POST /sessions` create response like the backend returns.

    session_id is top-level; experiment_id/experiment_run_id live under
    ``metadata``. Owning-context ids (project/tenant) can be threaded either at
    the top level or under ``metadata`` to exercise both parse paths.
    """
    metadata = {
        "experiment_id": experiment_id,
        "experiment_run_id": experiment_run_id,
    }
    payload = {"session_id": session_id, "metadata": metadata}
    target = metadata if context_in_metadata else payload
    if project_id is not None:
        target["project_id"] = project_id
    if tenant_id is not None:
        target["tenant_id"] = tenant_id
    return backend_response(payload=payload)


class TestSyncManager:
    """Test suite for SyncManager with comprehensive data conversion and sync tests."""

    @pytest.fixture(autouse=True)
    def _enable_mocked_backend_egress(self, monkeypatch):
        """Keep backend egress enabled for the mocked-transport sync tests.

        The shared conftest defaults every test outside ``tests/unit/cloud/`` to
        ``TRAIGENT_OFFLINE_MODE=true``. These tests exercise the sync transport
        boundary with a mocked HTTP session, so egress must stay enabled or the
        content-free session endpoints fail closed before any (mocked) POST.
        This function-scoped autouse fixture runs after the conftest one, so it
        wins. It touches only test-local env state, never the SDK's policy code.
        """
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        monkeypatch.setenv("TRAIGENT_OFFLINE", "false")

    def setup_method(self):
        """Set up test environment with temporary storage."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir)

        # Create test config
        self.config = TraigentConfig(
            offline=True,
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
        """Convert a local session to the content-free typed-session payload."""
        self.test_session = self._create_test_session()
        traigent_data = self.sync_manager.convert_session_to_traigent_format(
            self.test_session
        )

        # Only the two content-free keys are emitted; no agent/benchmark/
        # experiment/experiment_run content egresses anymore.
        assert set(traigent_data) == {"session_create", "configuration_runs"}
        for removed_key in ("agent", "benchmark", "experiment", "experiment_run"):
            assert removed_key not in traigent_data

        session_create = traigent_data["session_create"]
        assert session_create["function_name"] == "test_llm_function"

        # Configuration space is normalized to the typed wire contract.
        configuration_space = session_create["configuration_space"]
        assert configuration_space["model"] == {
            "type": "categorical",
            "choices": ["gpt-4", "gpt-3.5-turbo"],
        }
        assert configuration_space["max_tokens"] == {
            "type": "categorical",
            "choices": [100, 200, 500],
        }
        # A dict without type/low/high passes through unchanged.
        assert configuration_space["temperature"] == {"min": 0.0, "max": 1.0}

        # Objectives are derived measure names (content-free) with score present.
        objectives = session_create["objectives"]
        assert isinstance(objectives, list) and objectives
        assert "score" in objectives

        # Dataset metadata is a content-free label + size only.
        dataset_metadata = session_create["dataset_metadata"]
        assert dataset_metadata["privacy_mode"] is True
        assert dataset_metadata["name"] == "Local Dataset test_llm_function"
        assert BACKEND_NAME_PATTERN.fullmatch(dataset_metadata["name"])
        assert isinstance(dataset_metadata["size"], int)

        # native_local tracking is what makes the empty-dataset import work.
        assert session_create["optimization_strategy"] == {
            "algorithm": "optuna",
            "tracking_mode": "native_local",
        }
        assert session_create["metadata"]["source"] == "offline_sync"
        assert session_create["metadata"]["function_name"] == "test_llm_function"
        assert session_create["max_trials"] == 4

        # Configuration runs now carry a trial_id alongside the run payload.
        configuration_runs = traigent_data["configuration_runs"]
        assert len(configuration_runs) == 4
        assert set(configuration_runs[0]) == {
            "trial_id",
            "experiment_parameters",
            "measures",
            "status",
        }
        assert configuration_runs[0]["status"] == "COMPLETED"

    def test_convert_session_payloads_match_backend_contract(self):
        """Generated typed-session payloads satisfy the backend name contract."""
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

        # Content-free shape: only session_create + configuration_runs, no
        # agent/benchmark/experiment keys.
        assert set(traigent_data) == {"session_create", "configuration_runs"}
        assert "agent" not in traigent_data and "benchmark" not in traigent_data

        session_create = traigent_data["session_create"]
        dataset_name = session_create["dataset_metadata"]["name"]
        assert BACKEND_NAME_PATTERN.fullmatch(dataset_name)
        assert ":" not in dataset_name
        assert session_create["dataset_metadata"]["privacy_mode"] is True
        assert (
            session_create["optimization_strategy"]["tracking_mode"] == "native_local"
        )

        configuration_run = traigent_data["configuration_runs"][0]
        assert set(configuration_run) == {
            "trial_id",
            "experiment_parameters",
            "measures",
            "status",
        }
        assert configuration_run["measures"]["total_cost"] == 0.17

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
        # score is NOT aliased to "accuracy" — only real per-objective metrics
        # and cost fields land in measures.
        assert "accuracy" not in result["measures"]
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
        assert set(preview) == {
            "function_name",
            "dataset_name",
            "dataset_size",
            "trial_count",
            "best_score",
            "already_synced",
        }
        assert preview["function_name"] == "test_llm_function"
        assert preview["dataset_name"] == "Local Dataset test_llm_function"
        assert preview["trial_count"] == 4
        assert preview["best_score"] == 0.85
        assert preview["already_synced"] is False

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
        """A full sync creates the session, posts each result, then finalizes."""
        self.test_session = self._create_test_session()
        # The new sync path makes NO .get / .put calls — only .post.
        no_get = Mock()
        no_put = Mock()
        self.sync_manager._session.get = no_get
        self.sync_manager._session.put = no_put

        mock_post.side_effect = [
            create_session_response(),
            *[backend_response(payload={"id": f"result-{i}"}) for i in range(4)],
            backend_response(payload={}),
        ]

        result = self.sync_manager.sync_session_to_cloud(
            self.test_session.session_id, dry_run=False
        )

        assert result["status"] == "success"
        assert result["errors"] == []
        assert result["data_converted"] is True
        assert result["trials_converted"] == 4
        assert result["cloud_session_id"] == "sess-id"
        assert result["cloud_experiment_id"] == "experiment-id"
        assert result["cloud_experiment_run_id"] == "experiment-run-id"
        assert "cloud_url" in result

        # 1 create + 4 result posts + 1 finalize.
        assert mock_post.call_count == 6
        base = self.sync_manager.base_url
        assert [call.args[0] for call in mock_post.call_args_list] == [
            f"{base}/sessions",
            f"{base}/sessions/sess-id/results",
            f"{base}/sessions/sess-id/results",
            f"{base}/sessions/sess-id/results",
            f"{base}/sessions/sess-id/results",
            f"{base}/sessions/sess-id/finalize",
        ]
        no_get.assert_not_called()
        no_put.assert_not_called()

    @patch("requests.Session.post")
    def test_sync_session_to_cloud_threads_owner_context(self, mock_post):
        """Owning-context ids from the create response are recorded on the result."""
        self.test_session = self._create_test_session()
        mock_post.side_effect = [
            create_session_response(project_id="proj-1", tenant_id="tenant-1"),
            *[backend_response(payload={"id": f"result-{i}"}) for i in range(4)],
            backend_response(payload={}),
        ]

        result = self.sync_manager.sync_session_to_cloud(
            self.test_session.session_id, dry_run=False
        )

        assert result["status"] == "success"
        assert result["project_id"] == "proj-1"
        assert result["tenant_id"] == "tenant-1"

    def test_sync_session_to_cloud_contract_payloads_with_cost(self):
        """A full mocked sync emits the typed-session URLs and cost-bearing bodies."""
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

        try:
            self.sync_manager._session.post = Mock(
                side_effect=[
                    create_session_response(),
                    backend_response(payload={"id": "result-id"}),
                    backend_response(payload={}),
                ]
            )
            result = self.sync_manager.sync_session_to_cloud(session_id, dry_run=False)
        finally:
            post_mock = self.sync_manager._session.post
            self.sync_manager._session.post = original_post

        assert result["status"] == "success"
        assert post_mock.call_count == 3

        calls = post_mock.call_args_list
        base = self.sync_manager.base_url
        assert [call.args[0] for call in calls] == [
            f"{base}/sessions",
            f"{base}/sessions/sess-id/results",
            f"{base}/sessions/sess-id/finalize",
        ]

        # Create body: content-free typed session.
        create_payload = calls[0].kwargs["json"]
        assert create_payload["function_name"] == "contract:sync/fn"
        assert BACKEND_NAME_PATTERN.fullmatch(
            create_payload["dataset_metadata"]["name"]
        )
        assert ":" not in create_payload["dataset_metadata"]["name"]
        assert create_payload["dataset_metadata"]["privacy_mode"] is True
        assert (
            create_payload["optimization_strategy"]["tracking_mode"] == "native_local"
        )

        # Result body: {trial_id, config, status, metrics} with cost measures.
        result_payload = calls[1].kwargs["json"]
        assert result_payload["config"] == {"model": "gpt-test", "temperature": 0.2}
        assert result_payload["status"] == "COMPLETED"
        metrics = result_payload["metrics"]
        assert isinstance(metrics["cost"], float)
        assert metrics["cost"] == 0.42
        assert metrics["total_cost"] == 0.42
        assert metrics["input_cost"] == 0.2
        assert metrics["output_cost"] == 0.22
        assert metrics["score"] == 0.88

        # Finalize body: content-free reason + experiment_run_id.
        finalize_payload = calls[2].kwargs["json"]
        assert finalize_payload == {
            "reason": "offline_sync_finalization",
            "experiment_run_id": "experiment-run-id",
        }

    @patch("requests.Session.post")
    def test_sync_session_to_cloud_partial_failure(self, mock_post):
        """A failed result POST yields a partial sync and skips finalize."""
        self.test_session = self._create_test_session()
        mock_post.side_effect = [
            create_session_response(),
            backend_response(status_code=400, text="Bad Request"),
            backend_response(payload={"id": "result-1"}),
            backend_response(payload={"id": "result-2"}),
            backend_response(payload={"id": "result-3"}),
        ]

        result = self.sync_manager.sync_session_to_cloud(
            self.test_session.session_id, dry_run=False
        )

        assert result["status"] == "partial"
        assert any("Result sync failed" in error for error in result["errors"])
        assert any("HTTP 400" in error for error in result["errors"])
        # Create + 4 result posts, but NO finalize (a partial upload stays open).
        assert mock_post.call_count == 5

    @patch("requests.Session.post")
    def test_sync_session_to_cloud_create_failure(self, mock_post):
        """A failed session-create POST aborts before any results are posted."""
        self.test_session = self._create_test_session()
        mock_post.side_effect = [
            backend_response(status_code=400, text="Bad Request"),
        ]

        result = self.sync_manager.sync_session_to_cloud(
            self.test_session.session_id, dry_run=False
        )

        assert result["status"] == "error"
        assert any("Session create failed" in error for error in result["errors"])
        # Only the create POST is attempted.
        assert mock_post.call_count == 1

    @patch("requests.Session.post")
    def test_sync_session_to_cloud_complete_failure(self, mock_post):
        """A network exception on the first POST surfaces as an error result."""
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
        # Every POST (create + results + finalize) succeeds; the create response
        # carries the session_id used to route results/finalize.
        mock_post.return_value = create_session_response()

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

        # Create another session whose create POST will fail
        storage = self.sync_manager.storage
        storage.finalize_session(storage.create_session("failed_function"), "completed")

        def route(url, json=None, timeout=None):
            # Fail only the create POST for the "failed_function" session; every
            # other POST (creates, results, finalize) succeeds. Keyed off the
            # request body so it is independent of session iteration order.
            if url.endswith("/sessions"):
                if json.get("function_name") == "failed_function":
                    return backend_response(status_code=400, text="Bad Request")
                return create_session_response()
            return backend_response(payload={})

        mock_post.side_effect = route

        result = self.sync_manager.sync_all_sessions(dry_run=False)

        assert result["total_sessions"] == 2
        assert result["eligible_sessions"] == 2
        assert result["synced_successfully"] == 1
        assert result["sync_errors"] == 1
        assert result["overall_status"] == "partial"

    def test_sync_create_session_success(self):
        """_sync_create_session POSTs to /sessions and parses the response ids."""
        session_create = {"function_name": "fn"}
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = create_session_response(
                project_id=" proj-1 ", tenant_id="tenant-1"
            )

            result = self.sync_manager._sync_create_session(session_create)

        assert result["success"] is True
        assert result["session_id"] == "sess-id"
        assert result["experiment_id"] == "experiment-id"
        assert result["experiment_run_id"] == "experiment-run-id"
        # Context ids are stripped.
        assert result["project_id"] == "proj-1"
        assert result["tenant_id"] == "tenant-1"
        mock_post.assert_called_once_with(
            f"{self.sync_manager.base_url}/sessions",
            json=session_create,
            timeout=self.sync_manager._request_timeout,
        )

    def test_sync_create_session_context_in_metadata(self):
        """Owning-context ids under metadata are also parsed."""
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = create_session_response(
                project_id="proj-2",
                tenant_id="tenant-2",
                context_in_metadata=True,
            )

            result = self.sync_manager._sync_create_session({"function_name": "fn"})

        assert result["success"] is True
        assert result["project_id"] == "proj-2"
        assert result["tenant_id"] == "tenant-2"

    def test_sync_create_session_id_fallback(self):
        """experiment_id / experiment_run_id fall back to session_id when absent."""
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = backend_response(
                payload={"session_id": "only-session"}
            )

            result = self.sync_manager._sync_create_session({"function_name": "fn"})

        assert result["success"] is True
        assert result["session_id"] == "only-session"
        assert result["experiment_id"] == "only-session"
        assert result["experiment_run_id"] == "only-session"
        assert result["project_id"] is None
        assert result["tenant_id"] is None

    def test_sync_create_session_failure(self):
        """A non-2xx create response is an error with the HTTP status/text."""
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = backend_response(
                status_code=400, text="Validation error"
            )

            result = self.sync_manager._sync_create_session({"function_name": "fn"})

        assert result["success"] is False
        assert "HTTP 400: Validation error" in result["error"]

    def test_sync_create_session_exception(self):
        """A network exception during create is captured as an error."""
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.side_effect = requests.ConnectionError("Network error")

            result = self.sync_manager._sync_create_session({"function_name": "fn"})

        assert result["success"] is False
        assert "Network error" in result["error"]

    def test_sync_session_results_posts_each_run(self):
        """Each configuration run is POSTed to /sessions/{id}/results with a timeout."""
        runs = [
            {
                "trial_id": "t0",
                "experiment_parameters": {"a": 1},
                "measures": {"score": 0.5},
                "status": "COMPLETED",
            },
            {
                "trial_id": "t1",
                "experiment_parameters": {"a": 2},
                "measures": {"score": 0.6},
                "status": "COMPLETED",
            },
        ]
        recorded: list[tuple[str, str | None]] = []

        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = backend_response(payload={"id": "result-x"})

            result = self.sync_manager._sync_session_results(
                "sess-id",
                runs,
                on_synced=lambda key, result_id: recorded.append((key, result_id)),
            )

        assert result["success"] is True
        assert result["synced"] == 2
        assert result["skipped"] == 0
        assert result["errors"] == []
        assert mock_post.call_count == 2

        first_call = mock_post.call_args_list[0]
        assert (
            first_call.args[0]
            == f"{self.sync_manager.base_url}/sessions/sess-id/results"
        )
        assert first_call.kwargs["json"] == {
            "trial_id": "t0",
            "config": {"a": 1},
            "status": "COMPLETED",
            "metrics": {"score": 0.5},
        }
        assert first_call.kwargs["timeout"] == self.sync_manager._request_timeout
        # on_synced is invoked incrementally per successful POST.
        assert recorded == [("cfg_0", "result-x"), ("cfg_1", "result-x")]

    def test_sync_session_results_skips_already_synced(self):
        """Runs whose key is already synced are skipped, not re-POSTed."""
        runs = [
            {
                "trial_id": "t0",
                "experiment_parameters": {"a": 1},
                "measures": {"score": 0.5},
                "status": "COMPLETED",
            },
            {
                "trial_id": "t1",
                "experiment_parameters": {"a": 2},
                "measures": {"score": 0.6},
                "status": "COMPLETED",
            },
        ]

        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = backend_response(payload={"id": "result-x"})

            result = self.sync_manager._sync_session_results(
                "sess-id", runs, already_synced_keys={"cfg_0"}
            )

        assert result["success"] is True
        assert result["synced"] == 1
        assert result["skipped"] == 1
        assert mock_post.call_count == 1

    def test_sync_session_results_failure(self):
        """A non-2xx result POST is reported as a per-trial error."""
        runs = [
            {
                "trial_id": "t0",
                "experiment_parameters": {"a": 1},
                "measures": {"score": 0.5},
                "status": "COMPLETED",
            },
        ]

        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = backend_response(
                status_code=400, text="Bad Request"
            )

            result = self.sync_manager._sync_session_results("sess-id", runs)

        assert result["success"] is False
        assert result["synced"] == 0
        assert any("trial 1: HTTP 400" in error for error in result["errors"])

    def test_sync_finalize_session_success(self):
        """_sync_finalize_session POSTs to /finalize with a timeout and classifies."""
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = backend_response(payload={})

            result = self.sync_manager._sync_finalize_session(
                "sess-id", "experiment-run-id"
            )

        assert result["success"] is True
        assert result["classification"] == "completed"
        mock_post.assert_called_once_with(
            f"{self.sync_manager.base_url}/sessions/sess-id/finalize",
            json={
                "reason": "offline_sync_finalization",
                "experiment_run_id": "experiment-run-id",
            },
            timeout=self.sync_manager._request_timeout,
        )

    def test_sync_finalize_session_failure(self):
        """A non-2xx finalize response is an error with the HTTP status/text."""
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = backend_response(
                status_code=400, text="Bad Request"
            )

            result = self.sync_manager._sync_finalize_session("sess-id", "run-id")

        assert result["success"] is False
        assert "HTTP 400: Bad Request" in result["error"]

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

        # Should handle empty trials gracefully: no configuration runs, and the
        # objectives fall back to the default ["score"].
        assert len(traigent_data["configuration_runs"]) == 0
        assert traigent_data["session_create"]["objectives"] == ["score"]

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
        session_create = traigent_data["session_create"]
        assert session_create["function_name"] == "special_function_测试"
        assert session_create["metadata"]["function_name"] == "special_function_测试"
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

        # Test that the same session instance is reused across sync calls.
        with patch.object(self.sync_manager.session, "post") as mock_post:
            mock_post.return_value = create_session_response()

            self.sync_manager._sync_create_session({"function_name": "fn"})
            self.sync_manager._sync_finalize_session("sess-id", "experiment-run-id")

            # Should use same session instance for both requests.
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
        assert "session_create" in traigent_data

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
