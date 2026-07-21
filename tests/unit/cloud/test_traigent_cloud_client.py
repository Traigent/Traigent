"""Tests for Traigent Cloud Client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from traigent.cloud.auth import AuthManager
from traigent.cloud.client import (
    CloudOptimizationResult,
    CloudServiceError,
    TraigentCloudClient,
)
from traigent.config.backend_config import BackendConfig
from traigent.evaluators.base import Dataset, EvaluationExample


async def _stub_validate(self, api_key):  # noqa: ARG001
    """Bypass backend API key validation for offline tests."""
    return None


def _patch_backend_validate():
    """Patch ``AuthManager._validate_api_key_with_backend`` for offline tests.

    B4 round 3 made ``get_auth_headers()`` fail closed when authentication
    fails, so any test that calls a method which builds auth headers must
    either authenticate against a working backend or stub the backend
    validation hook.
    """
    return patch.object(
        AuthManager, "_validate_api_key_with_backend", new=_stub_validate
    )


@pytest.fixture
def sample_dataset():
    """Create sample dataset for testing."""
    examples = [
        EvaluationExample(
            input_data={"text": "Test input 1"}, expected_output="output1"
        ),
        EvaluationExample(
            input_data={"text": "Test input 2"}, expected_output="output2"
        ),
        EvaluationExample(
            input_data={"text": "Test input 3"}, expected_output="output3"
        ),
    ]
    return Dataset(examples=examples, name="test_dataset")


@pytest.fixture
def mock_cloud_client():
    """Create mock cloud client for testing."""
    return TraigentCloudClient(
        api_key="tg_test_" + "x" * 56,  # pragma: allowlist secret
        base_url="http://localhost:8000",
        enable_fallback=True,
    )


class TestTraigentCloudClient:
    """Test cases for Traigent Cloud Client."""

    def test_client_initialization(self):
        """Test client initialization with different parameters."""
        client = TraigentCloudClient(
            api_key="tg_test_key",  # pragma: allowlist secret
            base_url="http://localhost:5000",
            enable_fallback=False,
            max_retries=5,
            timeout=60.0,
        )

        assert client.base_url == "http://localhost:5000"
        assert client.api_base_url == "http://localhost:5000/api/v1"
        assert client.enable_fallback is False
        assert client.max_retries == 5
        assert client.timeout == 60.0

    def test_client_default_initialization(self, monkeypatch):
        """Test client initialization with defaults."""
        # Ensure backend resolution does not depend on external environment configuration
        for var in [
            "TRAIGENT_BACKEND_URL",
            "TRAIGENT_API_URL",
            "TRAIGENT_DEFAULT_LOCAL_URL",
        ]:
            monkeypatch.delenv(var, raising=False)

        with patch(
            "traigent.cloud.credential_manager.CredentialManager.get_stored_backend_url",
            return_value=None,
        ):
            monkeypatch.setenv("TRAIGENT_ENV", "production")
            client = TraigentCloudClient()

        assert client.base_url == BackendConfig.DEFAULT_PROD_URL
        assert client.api_base_url == BackendConfig.get_cloud_api_url()
        assert client.enable_fallback is True
        assert client.max_retries == 3
        assert client.timeout == 30.0

    def test_context_manager(self, mock_cloud_client):
        """Test async context manager functionality."""

        async def run_test():
            with _patch_backend_validate():
                async with mock_cloud_client as client:
                    assert client._session is not None

                # Session should be closed after exit
                assert mock_cloud_client._session is None

        asyncio.run(run_test())

    def test_close_clears_shared_session(self, mock_cloud_client):
        """Public close() should release the shared HTTP session."""

        async def run_test():
            session = MagicMock()
            session.close = AsyncMock()
            mock_cloud_client._session = session

            await mock_cloud_client.close()

            session.close.assert_awaited_once()
            assert mock_cloud_client._session is None

        asyncio.run(run_test())

    def test_optimize_function_remote_execution_unavailable(
        self, mock_cloud_client, sample_dataset
    ):
        """Test cloud optimization fails closed until remote execution exists."""

        async def run_test():
            # Mock the auth and submission
            with (
                patch.object(
                    mock_cloud_client.auth, "is_authenticated", return_value=True
                ),
                patch.object(
                    mock_cloud_client.auth,
                    "get_headers",
                    return_value={
                        "Authorization": "Bearer test_token",
                        "Content-Type": "application/json",
                    },
                ),
                patch.object(mock_cloud_client, "_submit_optimization") as mock_submit,
            ):
                mock_submit.return_value = {
                    "best_config": {"param1": "value1", "param2": 0.5},
                    "best_metrics": {"accuracy": 0.85, "speed": 0.9},
                    "trials_count": 25,
                }

                async with mock_cloud_client as client:
                    with pytest.raises(CloudServiceError, match="use hybrid"):
                        await client.optimize_function(
                            function_name="test_function",
                            dataset=sample_dataset,
                            configuration_space={
                                "param1": ["a", "b"],
                                "param2": [0.1, 0.5, 0.9],
                            },
                            objectives=["accuracy", "speed"],
                            max_trials=50,
                        )

                mock_submit.assert_not_called()

        asyncio.run(run_test())

    def test_optimize_function_auth_failure(self, mock_cloud_client, sample_dataset):
        """Test optimization failure due to authentication."""

        async def run_test():
            from traigent.cloud.auth import AuthenticationError

            with (
                patch.object(
                    mock_cloud_client.auth, "is_authenticated", return_value=False
                ),
                patch.object(
                    mock_cloud_client.auth,
                    "get_headers",
                    side_effect=AuthenticationError("Not authenticated"),
                ),
            ):
                with pytest.raises(AuthenticationError, match="Not authenticated"):
                    async with mock_cloud_client:
                        pass  # The exception should be raised in __aenter__

        asyncio.run(run_test())

    def test_optimize_function_does_not_fallback(
        self, mock_cloud_client, sample_dataset
    ):
        """Test remote cloud optimization does not silently fallback."""

        async def local_function(text: str, param: int = 1) -> str:
            return f"{text}:{param}"

        async def run_test():
            with (
                patch.object(
                    mock_cloud_client.auth, "is_authenticated", return_value=True
                ),
                patch.object(
                    mock_cloud_client.auth,
                    "get_headers",
                    return_value={
                        "Authorization": "Bearer test_token",
                        "Content-Type": "application/json",
                    },
                ),
                patch.object(
                    mock_cloud_client,
                    "_submit_optimization",
                    side_effect=Exception("Network error"),
                ),
                patch(
                    "traigent.optimizers.registry.get_optimizer"
                ) as mock_get_optimizer,
                patch(
                    "traigent.core.orchestrator.OptimizationOrchestrator"
                ) as mock_orchestrator_class,
            ):
                mock_optimizer = MagicMock()
                mock_get_optimizer.return_value = mock_optimizer

                mock_orchestrator = mock_orchestrator_class.return_value
                mock_orchestrator.optimize = AsyncMock(
                    return_value=MagicMock(
                        best_config={"param": 1},
                        best_metrics={"accuracy": 0.8},
                        trials=[object(), object()],
                    )
                )

                async with mock_cloud_client as client:
                    with pytest.raises(CloudServiceError, match="use hybrid"):
                        await client.optimize_function(
                            function_name="test_function",
                            dataset=sample_dataset,
                            configuration_space={"param": [1, 2, 3]},
                            objectives=["accuracy"],
                            local_function=local_function,
                        )

                mock_get_optimizer.assert_not_called()
                mock_orchestrator_class.assert_not_called()

        asyncio.run(run_test())

    def test_optimize_function_no_fallback(self, sample_dataset):
        """Test optimization failure without fallback."""

        async def run_test():
            client = TraigentCloudClient(
                enable_fallback=False,
                api_key="tg_test_" + "x" * 56,  # pragma: allowlist secret
            )

            with (
                patch.object(client.auth, "is_authenticated", return_value=True),
                patch.object(
                    client.auth,
                    "get_headers",
                    return_value={
                        "Authorization": "Bearer test_token",
                        "Content-Type": "application/json",
                    },
                ),
                patch.object(
                    client,
                    "_submit_optimization",
                    side_effect=Exception("Network error"),
                ),
            ):
                async with client as c:
                    with pytest.raises(CloudServiceError, match="use hybrid"):
                        await c.optimize_function(
                            function_name="test_function",
                            dataset=sample_dataset,
                            configuration_space={},
                            objectives=["accuracy"],
                        )

        asyncio.run(run_test())

    def test_submit_optimization_success(self, mock_cloud_client):
        """Test successful optimization submission."""

        async def run_test():
            mock_response = MagicMock()
            mock_response.status = 200
            # The response.json() returns the actual dict that retry_http_request will return
            mock_response.json = AsyncMock(
                return_value={
                    "best_config": {"param": "value"},
                    "best_metrics": {"accuracy": 0.9},
                    "trials_count": 10,
                }
            )

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_cloud_client._session = mock_session

            with _patch_backend_validate():
                result = await mock_cloud_client._submit_optimization(
                    {
                        "function_name": "test",
                        "dataset": {},
                        "configuration_space": {},
                        "objectives": ["accuracy"],
                    }
                )

            assert result["best_config"] == {"param": "value"}
            assert result["best_metrics"] == {"accuracy": 0.9}
            assert result["trials_count"] == 10

        asyncio.run(run_test())

    def test_submit_optimization_rate_limited(self, mock_cloud_client):
        """Test optimization submission with rate limiting."""

        async def run_test():
            # First request: rate limited (429)
            mock_response_429 = MagicMock()
            mock_response_429.status = 429

            # Second request: success
            mock_response_200 = MagicMock()
            mock_response_200.status = 200
            mock_response_200.json = AsyncMock(
                return_value={
                    "best_config": {"param": "value"},
                    "best_metrics": {"accuracy": 0.9},
                    "trials_count": 10,
                }
            )

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__.side_effect = [
                mock_response_429,
                mock_response_200,
            ]
            mock_cloud_client._session = mock_session

            with (
                _patch_backend_validate(),
                patch("asyncio.sleep", new_callable=AsyncMock),
            ):
                result = await mock_cloud_client._submit_optimization({})

            assert result["trials_count"] == 10

        asyncio.run(run_test())

    def test_submit_optimization_max_retries(self, mock_cloud_client):
        """Test optimization submission exceeding max retries."""

        async def run_test():
            mock_response = MagicMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__.return_value = mock_response
            mock_cloud_client._session = mock_session

            with _patch_backend_validate():
                with pytest.raises(CloudServiceError, match="HTTP 500"):
                    await mock_cloud_client._submit_optimization({})

        asyncio.run(run_test())

    def test_submit_optimization_idempotency_key_stable_across_retries(
        self, mock_cloud_client
    ):
        """A retried /optimize POST reuses one Idempotency-Key.

        The backend dedupes non-idempotent POSTs only when every retry attempt
        carries the SAME Idempotency-Key. Regenerating the key per attempt would
        defeat the dedup and produce a duplicate optimization run, so this test
        pins the invariant: one transient 429 then success, and both POST
        attempts must send an identical Idempotency-Key header.
        """

        async def run_test():
            # First request: rate limited (429, retryable) -> retry.
            mock_response_429 = MagicMock()
            mock_response_429.status = 429

            # Second request: success.
            mock_response_200 = MagicMock()
            mock_response_200.status = 200
            mock_response_200.json = AsyncMock(
                return_value={"trials_count": 10}
            )

            mock_session = MagicMock()
            mock_session.post.return_value.__aenter__.side_effect = [
                mock_response_429,
                mock_response_200,
            ]
            mock_cloud_client._session = mock_session

            with (
                _patch_backend_validate(),
                patch("asyncio.sleep", new_callable=AsyncMock),
            ):
                await mock_cloud_client._submit_optimization({})

            # Two POST attempts were made (initial + one retry).
            assert mock_session.post.call_count == 2
            keys = [
                call.kwargs["headers"]["Idempotency-Key"]
                for call in mock_session.post.call_args_list
            ]
            # (a) the header is present on every POST, and
            # (b) it is IDENTICAL across the retry.
            assert all(keys), "Idempotency-Key header missing on a POST attempt"
            assert keys[0] == keys[1], (
                "Idempotency-Key changed between retries; backend dedup would fail"
            )

        asyncio.run(run_test())

    def test_submit_optimization_idempotency_key_differs_per_call(
        self, mock_cloud_client
    ):
        """A new logical /optimize call gets a fresh Idempotency-Key.

        The key must be stable within one logical call (across its retries) but
        distinct between separate calls, so two intentional submissions are not
        collapsed into one by the backend's dedup window.
        """

        async def run_test():
            def _fresh_success_session():
                mock_response = MagicMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={"trials_count": 1})
                mock_session = MagicMock()
                mock_session.post.return_value.__aenter__.return_value = (
                    mock_response
                )
                return mock_session

            with _patch_backend_validate():
                session_a = _fresh_success_session()
                mock_cloud_client._session = session_a
                await mock_cloud_client._submit_optimization({})

                session_b = _fresh_success_session()
                mock_cloud_client._session = session_b
                await mock_cloud_client._submit_optimization({})

            key_a = session_a.post.call_args.kwargs["headers"]["Idempotency-Key"]
            key_b = session_b.post.call_args.kwargs["headers"]["Idempotency-Key"]
            assert key_a and key_b
            assert key_a != key_b, (
                "Distinct logical calls must not share an Idempotency-Key"
            )

        asyncio.run(run_test())

    def test_serialize_dataset(self, mock_cloud_client, sample_dataset):
        """Test dataset serialization for cloud transmission."""
        serialized = mock_cloud_client._serialize_dataset(sample_dataset)

        assert serialized["name"] == "test_dataset"
        assert len(serialized["examples"]) == 3

        first_example = serialized["examples"][0]
        assert first_example["input_data"] == {"text": "Test input 1"}
        assert first_example["expected_output"] == "output1"
        assert "metadata" in first_example

    def test_get_usage_stats(self, mock_cloud_client):
        """Test getting usage statistics."""

        async def run_test():
            with patch.object(
                mock_cloud_client.usage_tracker, "get_usage_stats"
            ) as mock_stats:
                mock_stats.return_value = {
                    "total_optimizations": 5,
                    "total_credits": 12.5,
                    "total_time": 120.0,
                }

                stats = await mock_cloud_client.get_usage_stats()
                assert stats["total_optimizations"] == 5
                assert stats["total_credits"] == 12.5
                assert stats["total_time"] == 120.0

        asyncio.run(run_test())

    def test_check_service_status_healthy(self, mock_cloud_client):
        """Test service status check when healthy."""

        async def run_test():
            mock_response = MagicMock()
            mock_response.json = AsyncMock(
                return_value={"status": "healthy", "uptime": "24h", "version": "1.0.0"}
            )

            mock_session = MagicMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            mock_cloud_client._session = mock_session

            with _patch_backend_validate():
                status = await mock_cloud_client.check_service_status()
            assert status["status"] == "healthy"
            assert "uptime" in status

        asyncio.run(run_test())

    def test_check_service_status_unavailable(self, mock_cloud_client):
        """Test service status check when unavailable."""

        async def run_test():
            mock_session = MagicMock()
            mock_session.get.side_effect = Exception("Connection failed")
            mock_cloud_client._session = mock_session

            with _patch_backend_validate():
                status = await mock_cloud_client.check_service_status()
            assert status["status"] == "unavailable"
            assert "error" in status

        asyncio.run(run_test())


class TestCloudOptimizationResult:
    """Test cases for CloudOptimizationResult dataclass."""

    def test_cloud_optimization_result_creation(self):
        """Test creation of CloudOptimizationResult."""
        result = CloudOptimizationResult(
            best_config={"param": "value"},
            best_metrics={"accuracy": 0.9},
            trials_count=25,
            cost_reduction=0.65,
            optimization_time=120.5,
            subset_used=True,
            subset_size=150,
        )

        assert result.best_config == {"param": "value"}
        assert result.best_metrics == {"accuracy": 0.9}
        assert result.trials_count == 25
        assert result.cost_reduction == 0.65
        assert result.optimization_time == 120.5
        assert result.subset_used is True
        assert result.subset_size == 150

    def test_cloud_optimization_result_defaults(self):
        """Test CloudOptimizationResult with default values."""
        result = CloudOptimizationResult(
            best_config={},
            best_metrics={},
            trials_count=0,
            cost_reduction=0.0,
            optimization_time=0.0,
            subset_used=False,
        )

        assert result.subset_size is None


class TestCloudServiceError:
    """Test cases for CloudServiceError exception."""

    def test_cloud_service_error_creation(self):
        """Test CloudServiceError creation."""
        error = CloudServiceError("Test error message")
        assert str(error) == "Test error message"

    def test_cloud_service_error_inheritance(self):
        """Test CloudServiceError inheritance from Exception."""
        error = CloudServiceError("Test error")
        assert isinstance(error, Exception)


class TestFallbackOptimizationTrialsCount:
    """_fallback_optimization reports len(optimization_result.trials) directly.

    Regression (#1493): the former guard
    ``len(trials) if isinstance(trials, list) else 0`` would mask a
    producer-contract violation; the fix uses ``len(trials)`` directly so a
    non-list trials value raises instead of silently reporting 0.
    """

    def test_trials_count_equals_number_of_returned_trials(self) -> None:
        """trials_count == len(optimization_result.trials) for a valid result."""
        from datetime import UTC, datetime

        from traigent.api.types import (
            OptimizationResult,
            OptimizationStatus,
            TrialResult,
            TrialStatus,
        )

        def _trial(tid: str) -> TrialResult:
            return TrialResult(
                trial_id=tid,
                config={"k": "v"},
                metrics={"accuracy": 0.5},
                status=TrialStatus.COMPLETED,
                duration=1.0,
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
            )

        # 4-trial result — OptimizationResult.trials is always list[TrialResult]
        mock_result = OptimizationResult(
            trials=[_trial("t1"), _trial("t2"), _trial("t3"), _trial("t4")],
            best_config={"k": "v"},
            best_score=0.5,
            optimization_id="oid",
            duration=4.0,
            convergence_info={},
            status=OptimizationStatus.COMPLETED,
            objectives=["accuracy"],
            algorithm="random",
            timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        )

        client = TraigentCloudClient(
            api_key="tg_test_" + "x" * 56,  # pragma: allowlist secret
            base_url="http://localhost",
        )
        dataset = Dataset([EvaluationExample({"text": "x"}, "x")], name="d")

        async def run_test() -> CloudOptimizationResult:
            # Patch at the source modules — local imports in _fallback_optimization
            # pick up the patched versions from sys.modules.
            with (
                patch(
                    "traigent.core.orchestrator.OptimizationOrchestrator"
                ) as mock_orch_cls,
                patch("traigent.evaluators.local.LocalEvaluator"),
                patch("traigent.optimizers.registry.get_optimizer"),
            ):
                mock_orch = mock_orch_cls.return_value
                mock_orch.optimize = AsyncMock(return_value=mock_result)
                return await client._fallback_optimization(
                    function_name="dummy",
                    dataset=dataset,
                    configuration_space={"k": ["v"]},
                    objectives=["accuracy"],
                    max_trials=4,
                    local_function=lambda text: text,
                )

        result = asyncio.run(run_test())

        # Must be 4 (len(trials)), not 0 (former isinstance-guard fallback value)
        assert result.trials_count == 4

    def test_non_list_trials_raises_not_zero_count(self) -> None:
        """A non-list ``trials`` (contract violation) must raise, not report 0.

        Pre-fix: ``len(trials) if isinstance(trials, list) else 0`` returned 0
        for a non-list value, masking the producer-contract breach.
        Post-fix: ``len(trials)`` raises TypeError, surfacing it.

        FAILS on pre-fix code (which would return trials_count=0); PASSES on
        the fixed code (which raises).
        """
        # optimization_result whose `trials` is None — a producer-contract
        # violation that the removed `isinstance ... else 0` guard would mask.
        from types import SimpleNamespace

        bad_result = SimpleNamespace(
            trials=None,
            best_config={"k": "v"},
            best_metrics={"accuracy": 0.5},
            best_score=0.5,
        )

        client = TraigentCloudClient(
            api_key="tg_test_" + "x" * 56,  # pragma: allowlist secret
            base_url="http://localhost",
        )
        dataset = Dataset([EvaluationExample({"text": "x"}, "x")], name="d")

        async def run_test() -> CloudOptimizationResult:
            with (
                patch(
                    "traigent.core.orchestrator.OptimizationOrchestrator"
                ) as mock_orch_cls,
                patch("traigent.evaluators.local.LocalEvaluator"),
                patch("traigent.optimizers.registry.get_optimizer"),
            ):
                mock_orch = mock_orch_cls.return_value
                mock_orch.optimize = AsyncMock(return_value=bad_result)
                return await client._fallback_optimization(
                    function_name="dummy",
                    dataset=dataset,
                    configuration_space={"k": ["v"]},
                    objectives=["accuracy"],
                    max_trials=4,
                    local_function=lambda text: text,
                )

        with pytest.raises(TypeError):
            asyncio.run(run_test())
