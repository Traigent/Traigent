"""Unit tests for stateful cloud client operations."""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
import pytest_asyncio

from traigent.cloud.client import CloudServiceError, TraigentCloudClient
from traigent.cloud.models import (
    OptimizationSessionStatus,
    SessionObjectiveDefinition,
    TrialResultSubmission,
    TrialStatus,
)
from traigent.config.backend_config import BackendConfig


@pytest_asyncio.fixture
async def mock_session():
    """Create a mock aiohttp session."""
    # Use AsyncMock directly without spec to avoid issues with already-mocked aiohttp
    session = AsyncMock()

    # Create mock context manager for POST/GET requests
    mock_response_cm = AsyncMock()
    mock_response_cm.__aenter__ = AsyncMock()
    mock_response_cm.__aexit__ = AsyncMock(return_value=None)

    # Mock response objects
    session.post = Mock(return_value=mock_response_cm)
    session.get = Mock(return_value=mock_response_cm)
    session.close = AsyncMock()

    return session


@pytest_asyncio.fixture
async def cloud_client(mock_session):
    """Create a cloud client with mocked session and auth."""
    with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", True):
        with patch(
            "traigent.cloud.client.aiohttp.ClientSession", return_value=mock_session
        ):
            with patch("traigent.cloud.client.AuthManager") as mock_auth_mgr:
                # Setup mock auth manager
                mock_auth_instance = Mock()
                mock_auth_instance.get_headers = AsyncMock(
                    return_value={
                        "Authorization": "Bearer test-key",
                        "X-Traigent-Client": "test",
                        "Content-Type": "application/json",
                    }
                )
                mock_auth_instance.is_authenticated = AsyncMock(return_value=True)
                mock_auth_instance.get_owner_fingerprint = Mock(
                    return_value={
                        "owner_user_id": "user-123",
                        "owner_api_key_id": "key-789",
                        "created_by": "user-123",
                        "owner_scope": ["optimize"],
                        "credential_source": "test-suite",
                        "owner_api_key_preview": "tg_test_preview",  # pragma: allowlist secret
                    }
                )
                mock_auth_mgr.return_value = mock_auth_instance

                client = TraigentCloudClient(api_key="test-key")
                client._session = mock_session
                client.auth = mock_auth_instance
                yield client


class TestSessionCreation:
    """Test session creation functionality."""

    @pytest.mark.asyncio
    async def test_create_optimization_session_success(
        self, cloud_client, mock_session
    ):
        """Test successful session creation."""
        # Mock response
        mock_response = Mock()
        mock_response.status = 201
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "session-123",
                "status": "active",
                "optimization_strategy": {"exploration_ratio": 0.3},
                "estimated_duration": 3600.0,
                "billing_estimate": {"credits": 100},
            }
        )

        mock_session.post.return_value.__aenter__.return_value = mock_response

        # Create session
        response = await cloud_client.create_optimization_session(
            request_or_function_name="test_function",
            configuration_space={"temperature": (0.0, 1.0)},
            objectives=["accuracy"],
            dataset_metadata={"size": 1000, "type": "qa"},
            max_trials=50,
        )

        assert response.session_id == "session-123"
        assert response.status == OptimizationSessionStatus.ACTIVE
        assert response.optimization_strategy["exploration_ratio"] == 0.3
        assert response.estimated_duration == 3600.0

        # Verify API call
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        # The client uses api_base_url which includes the full API path
        expected_endpoint = (
            f"{BackendConfig.get_backend_api_url().rstrip('/')}/sessions"
        )
        assert call_args[0][0] == expected_endpoint

        submitted_payload = call_args.kwargs["json"]
        assert submitted_payload["metadata"]["owner_user_id"] == "user-123"
        assert submitted_payload["metadata"]["owner_api_key_id"] == "key-789"
        assert submitted_payload["metadata"]["created_by"] == "user-123"

    @pytest.mark.asyncio
    async def test_create_optimization_session_serializes_typed_objectives_and_conditionals(
        self, cloud_client, mock_session
    ):
        mock_response = Mock()
        mock_response.status = 201
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "session-typed-123",
                "status": "active",
                "optimization_strategy": {"algorithm": "optuna"},
            }
        )
        mock_session.post.return_value.__aenter__.return_value = mock_response

        await cloud_client.create_optimization_session(
            request_or_function_name="test_function",
            configuration_space={
                "model": {"type": "categorical", "choices": ["cheap", "accurate"]},
                "max_tokens": {
                    "type": "int",
                    "low": 64,
                    "high": 256,
                    "conditions": {"model": "accurate"},
                    "default": 64,
                },
            },
            objectives=[
                SessionObjectiveDefinition(
                    metric="accuracy",
                    direction="maximize",
                    weight=2.0,
                ),
                {"metric": "latency", "direction": "minimize", "weight": 1.0},
            ],
            dataset_metadata={"size": 100},
            max_trials=12,
            budget={"max_cost_usd": 2.0},
            constraints={
                "derived": [{"require": "metrics.accuracy >= 0.8"}],
            },
        )

        submitted_payload = mock_session.post.call_args.kwargs["json"]
        assert submitted_payload["objectives"] == [
            {"metric": "accuracy", "direction": "maximize", "weight": 2.0},
            {"metric": "latency", "direction": "minimize", "weight": 1.0},
        ]
        assert submitted_payload["budget"] == {"max_cost_usd": 2.0}
        assert submitted_payload["constraints"] == {
            "derived": [{"require": "metrics.accuracy >= 0.8"}]
        }
        assert submitted_payload["configuration_space"]["max_tokens"] == {
            "type": "int",
            "low": 64,
            "high": 256,
            "conditions": {"model": "accurate"},
            "default": 64,
        }

    @pytest.mark.asyncio
    async def test_create_optimization_session_serializes_banded_objectives_and_policy(
        self, cloud_client, mock_session
    ):
        mock_response = Mock()
        mock_response.status = 201
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "session-typed-456",
                "status": "active",
                "optimization_strategy": {"algorithm": "optuna"},
            }
        )
        mock_session.post.return_value.__aenter__.return_value = mock_response

        await cloud_client.create_optimization_session(
            request_or_function_name="test_function",
            configuration_space={
                "retrieval_pair": {
                    "type": "categorical",
                    "choices": ["choice_0", "choice_1"],
                    "value_map": {
                        "choice_0": ["dense", "rerank"],
                        "choice_1": ["bm25", "none"],
                    },
                },
            },
            objectives=[
                SessionObjectiveDefinition(
                    metric="response_length",
                    band={"low": 120, "high": 180},
                    test="TOST",
                    alpha=0.05,
                    weight=2.0,
                ),
            ],
            dataset_metadata={"size": 100},
            max_trials=4,
            default_config={"temperature": 0.7},
            promotion_policy={"dominance": "epsilon_pareto", "alpha": 0.05},
        )

        submitted_payload = mock_session.post.call_args.kwargs["json"]
        assert submitted_payload["objectives"] == [
            {
                "metric": "response_length",
                "band": {"low": 120, "high": 180},
                "test": "TOST",
                "alpha": 0.05,
                "weight": 2.0,
            }
        ]
        assert submitted_payload["default_config"] == {"temperature": 0.7}
        assert submitted_payload["promotion_policy"] == {
            "dominance": "epsilon_pareto",
            "alpha": 0.05,
        }

    @pytest.mark.asyncio
    async def test_create_optimization_session_omits_optional_session_fields_when_absent(
        self, cloud_client, mock_session
    ):
        mock_response = Mock()
        mock_response.status = 201
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "session-typed-789",
                "status": "active",
                "optimization_strategy": {"algorithm": "optuna"},
            }
        )
        mock_session.post.return_value.__aenter__.return_value = mock_response

        await cloud_client.create_optimization_session(
            request_or_function_name="test_function",
            configuration_space={
                "temperature": {"type": "float", "low": 0.0, "high": 1.0}
            },
            objectives=["accuracy"],
            dataset_metadata={"size": 100},
            max_trials=3,
        )

        submitted_payload = mock_session.post.call_args.kwargs["json"]
        assert "budget" not in submitted_payload
        assert "constraints" not in submitted_payload
        assert "default_config" not in submitted_payload
        assert "promotion_policy" not in submitted_payload

    @pytest.mark.asyncio
    async def test_create_session_no_client_session(self, cloud_client):
        """Test session creation without initialized client session."""
        cloud_client._session = None

        # Since AIOHTTP is available in the mocked environment,
        # _ensure_session will recreate the session, so we need to mock it differently
        # to actually test the error case
        with patch("traigent.cloud.client.AIOHTTP_AVAILABLE", False):
            cloud_client._session = None

            with pytest.raises(
                CloudServiceError, match="Client session not initialized"
            ):
                await cloud_client.create_optimization_session(
                    request_or_function_name="test",
                    configuration_space={},
                    objectives=["accuracy"],
                    dataset_metadata={},
                )

    @pytest.mark.asyncio
    async def test_create_session_http_error(self, cloud_client, mock_session):
        """Test session creation with HTTP error."""
        mock_response = Mock()
        mock_response.status = 400
        mock_response.text = AsyncMock(
            return_value="Bad request: Invalid configuration space"
        )

        mock_session.post.return_value.__aenter__.return_value = mock_response

        try:
            result = await cloud_client.create_optimization_session(
                request_or_function_name="test",
                configuration_space={},
                objectives=["accuracy"],
                dataset_metadata={},
            )
            # If no exception is raised, fail the test
            pytest.fail(f"Expected CloudServiceError but got result: {result}")
        except CloudServiceError as e:
            # Check that the error message matches
            assert "Failed to create session" in str(e)
            assert "400" in str(e)


class TestTrialOperations:
    """Test trial suggestion and result submission."""

    @pytest.mark.asyncio
    async def test_get_next_trial_success(self, cloud_client, mock_session):
        """Test successful trial suggestion retrieval."""
        # Mock response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "suggestion": {
                    "trial_id": "trial-001",
                    "session_id": "session-123",
                    "trial_number": 1,
                    "config": {"temperature": 0.7, "model": "GPT-4o"},
                    "dataset_subset": {
                        "indices": [0, 5, 10, 15, 20],
                        "selection_strategy": "diverse_sampling",
                        "confidence_level": 0.8,
                        "estimated_representativeness": 0.75,
                        "metadata": {},
                    },
                    "exploration_type": "exploration",
                    "priority": 1,
                    "estimated_duration": 45.0,
                },
                "should_continue": True,
                "session_status": "active",
            }
        )

        mock_session.post.return_value.__aenter__.return_value = mock_response

        # Get next trial
        response = await cloud_client.get_next_trial("session-123")

        assert response.suggestion is not None
        assert response.suggestion.trial_id == "trial-001"
        assert response.suggestion.config["temperature"] == 0.7
        assert len(response.suggestion.dataset_subset.indices) == 5
        assert response.should_continue is True

        # Verify API call
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "sessions/session-123/next-trial" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_next_trial_no_more_trials(self, cloud_client, mock_session):
        """Test getting next trial when optimization is complete."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "suggestion": None,
                "should_continue": False,
                "reason": "Max trials reached",
                "session_status": "completed",
            }
        )

        mock_session.post.return_value.__aenter__.return_value = mock_response

        response = await cloud_client.get_next_trial("session-123")

        assert response.suggestion is None
        assert response.should_continue is False
        assert response.reason == "Max trials reached"
        assert response.session_status == OptimizationSessionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_submit_trial_result_success(self, cloud_client, mock_session):
        """Test successful trial result submission."""
        mock_response = Mock()
        mock_response.status = 201

        mock_session.post.return_value.__aenter__.return_value = mock_response

        # Submit result
        await cloud_client.submit_trial_result(
            session_id="session-123",
            trial_id="trial-001",
            metrics={"accuracy": 0.85, "speed": 0.92},
            duration=45.2,
            status="completed",
        )

        # Verify API call
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "sessions/session-123/results" in call_args[0][0]

        # Check serialized data
        submitted_data = call_args[1]["json"]
        assert submitted_data["trial_id"] == "trial-001"
        assert submitted_data["metrics"]["accuracy"] == 0.85
        assert submitted_data["duration"] == 45.2
        assert submitted_data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_submit_trial_result_with_error(self, cloud_client, mock_session):
        """Test submitting failed trial result."""
        mock_response = Mock()
        mock_response.status = 204

        mock_session.post.return_value.__aenter__.return_value = mock_response

        await cloud_client.submit_trial_result(
            session_id="session-123",
            trial_id="trial-001",
            metrics={},
            duration=2.5,
            status="failed",
            error_message="Connection timeout",
        )

        # Check submission
        submitted_data = mock_session.post.call_args[1]["json"]
        assert submitted_data["status"] == "failed"
        assert submitted_data["error_message"] == "Connection timeout"


class TestFinalization:
    """Test session finalization."""

    @pytest.mark.asyncio
    async def test_finalize_optimization_success(self, cloud_client, mock_session):
        """Test successful optimization finalization."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "session-123",
                "best_config": {"temperature": 0.7, "model": "GPT-4o"},
                "best_metrics": {"accuracy": 0.92, "speed": 0.88},
                "total_trials": 45,
                "successful_trials": 43,
                "total_duration": 3600.0,
                "cost_savings": 0.68,
                "convergence_history": [
                    {"trial": 1, "best_score": 0.75},
                    {"trial": 10, "best_score": 0.85},
                    {"trial": 45, "best_score": 0.92},
                ],
            }
        )

        mock_session.post.return_value.__aenter__.return_value = mock_response

        # Finalize
        response = await cloud_client.finalize_optimization(
            session_id="session-123", include_full_history=False
        )

        assert response.session_id == "session-123"
        assert response.best_config["temperature"] == 0.7
        assert response.best_metrics["accuracy"] == 0.92
        assert response.successful_trials == 43
        assert response.cost_savings == 0.68
        assert len(response.convergence_history) == 3

        # Verify API call
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "sessions/session-123/finalize" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_finalize_with_full_history(self, cloud_client, mock_session):
        """Test finalization with full history."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "session_id": "session-123",
                "best_config": {"temperature": 0.7},
                "best_metrics": {"accuracy": 0.92},
                "total_trials": 3,
                "successful_trials": 3,
                "total_duration": 300.0,
                "cost_savings": 0.5,
                "full_history": [
                    {
                        "session_id": "session-123",
                        "trial_id": "trial-001",
                        "metrics": {"accuracy": 0.80},
                        "duration": 40.0,
                        "status": "completed",
                    },
                    {
                        "session_id": "session-123",
                        "trial_id": "trial-002",
                        "metrics": {"accuracy": 0.85},
                        "duration": 42.0,
                        "status": "completed",
                    },
                    {
                        "session_id": "session-123",
                        "trial_id": "trial-003",
                        "metrics": {"accuracy": 0.92},
                        "duration": 45.0,
                        "status": "completed",
                    },
                ],
            }
        )

        mock_session.post.return_value.__aenter__.return_value = mock_response

        response = await cloud_client.finalize_optimization(
            session_id="session-123", include_full_history=True
        )

        assert response.full_history is not None
        assert len(response.full_history) == 3
        assert response.full_history[0].trial_id == "trial-001"
        assert response.full_history[2].metrics["accuracy"] == 0.92


class TestSerialization:
    """Test serialization and deserialization methods."""

    def test_serialize_trial_result(self, cloud_client):
        """Test serializing trial result submission."""
        result = TrialResultSubmission(
            session_id="session-123",
            trial_id="trial-001",
            metrics={"accuracy": 0.85},
            duration=45.2,
            status=TrialStatus.COMPLETED,
            outputs_sample=["output1", "output2"],
            error_message=None,
            metadata={"key": "value"},
        )

        serialized = cloud_client._serialize_trial_result(result)

        assert serialized["session_id"] == "session-123"
        assert serialized["trial_id"] == "trial-001"
        assert serialized["metrics"]["accuracy"] == 0.85
        assert serialized["duration"] == 45.2
        assert serialized["status"] == "completed"
        assert serialized["outputs_sample"] == ["output1", "output2"]
        assert serialized["metadata"]["key"] == "value"

    def test_deserialize_next_trial_response(self, cloud_client):
        """Test deserializing next trial response."""
        data = {
            "suggestion": {
                "trial_id": "trial-001",
                "session_id": "session-123",
                "trial_number": 1,
                "config": {"temperature": 0.7},
                "dataset_subset": {
                    "indices": [0, 1, 2],
                    "selection_strategy": "random",
                    "confidence_level": 0.5,
                    "estimated_representativeness": 0.5,
                    "metadata": {},
                },
                "exploration_type": "exploration",
                "priority": 2,
                "metadata": {"test": True},
            },
            "should_continue": True,
            "stop_reason": None,
            "session_status": "active",
        }

        response = cloud_client._deserialize_next_trial_response(data)

        assert response.suggestion.trial_id == "trial-001"
        assert response.suggestion.config["temperature"] == 0.7
        assert len(response.suggestion.dataset_subset.indices) == 3
        assert response.suggestion.priority == 2
        assert response.should_continue is True
        assert response.stop_reason is None

    def test_deserialize_finalization_response_prefers_top_level_stop_reason(
        self, cloud_client
    ):
        """Test deserializing finalization response with explicit stop_reason."""
        data = {
            "session_id": "session-123",
            "best_config": {"temperature": 0.7},
            "best_metrics": {"accuracy": 0.91},
            "total_trials": 3,
            "successful_trials": 3,
            "total_duration": 1.2,
            "cost_savings": 0.0,
            "stop_reason": "max_trials_reached",
            "metadata": {"stop_reason": "finalized"},
        }

        response = cloud_client._deserialize_finalization_response(data)

        assert response.session_id == "session-123"
        assert response.stop_reason == "max_trials_reached"

    def test_deserialize_finalization_response_falls_back_to_metadata_stop_reason(
        self, cloud_client
    ):
        """Test deserializing finalization response with legacy metadata stop_reason."""
        data = {
            "session_id": "session-123",
            "best_config": {"temperature": 0.7},
            "best_metrics": {"accuracy": 0.91},
            "total_trials": 3,
            "successful_trials": 3,
            "total_duration": 1.2,
            "cost_savings": 0.0,
            "metadata": {"stop_reason": "search_complete"},
        }

        response = cloud_client._deserialize_finalization_response(data)

        assert response.session_id == "session-123"
        assert response.stop_reason == "search_complete"


class TestErrorHandling:
    """Test error handling in stateful operations."""

    @pytest.mark.asyncio
    async def test_network_error_handling(self, cloud_client, mock_session):
        """Test network error handling."""
        mock_session.post.side_effect = aiohttp.ClientError("Connection refused")

        with pytest.raises(CloudServiceError, match="Network error creating session"):
            await cloud_client.create_optimization_session(
                request_or_function_name="test",
                configuration_space={},
                objectives=["accuracy"],
                dataset_metadata={},
            )

    @pytest.mark.asyncio
    async def test_http_error_responses(self, cloud_client, mock_session):
        """Test various HTTP error responses."""
        # 404 error
        mock_response = Mock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Session not found")

        mock_session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(CloudServiceError, match="Failed to get next trial.*404"):
            await cloud_client.get_next_trial("nonexistent-session")

        # 500 error
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal server error")

        with pytest.raises(CloudServiceError, match="Failed to submit result.*500"):
            await cloud_client.submit_trial_result(
                session_id="session-123", trial_id="trial-001", metrics={}, duration=1.0
            )

    @pytest.mark.asyncio
    async def test_next_trial_forbidden_error(self, cloud_client, mock_session):
        """Ensure 403 errors return actionable ownership guidance."""

        cloud_client._session_owners["session-123"] = {
            "owner_user_id": "owner-007",
            "owner_api_key_preview": "tg_owner_preview",  # pragma: allowlist secret
        }

        mock_response = Mock()
        mock_response.status = 403
        mock_response.text = AsyncMock(return_value="Forbidden: session owner mismatch")

        mock_session.post.reset_mock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(CloudServiceError) as excinfo:
            await cloud_client.get_next_trial("session-123")

        message = str(excinfo.value)
        assert "Forbidden" in message
        assert "owner-007" in message
        assert "Re-authenticate" in message

    @pytest.mark.asyncio
    async def test_submit_trial_forbidden_error(self, cloud_client, mock_session):
        """403 on submit should not be retried and should surface guidance."""

        cloud_client._session_owners["session-123"] = {
            "owner_user_id": "owner-007",
            "owner_api_key_preview": "tg_owner_preview",  # pragma: allowlist secret
        }

        mock_response = Mock()
        mock_response.status = 403
        mock_response.text = AsyncMock(return_value="Forbidden: session owner mismatch")

        mock_session.post.reset_mock()
        mock_session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(CloudServiceError) as excinfo:
            await cloud_client.submit_trial_result(
                session_id="session-123",
                trial_id="trial-001",
                metrics={"accuracy": 0.9},
                duration=1.0,
            )

        message = str(excinfo.value)
        assert "Forbidden" in message
        assert "Re-authenticate" in message
