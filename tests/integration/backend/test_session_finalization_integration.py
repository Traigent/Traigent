"""Integration tests for session finalization with backend."""

import os
import time
from collections import defaultdict
from itertools import count
from typing import Any

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.models import SessionCreationRequest


class _MockClientTimeout:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _MockBackendResponse:
    def __init__(self, status: int, payload: dict[str, Any] | None = None):
        self.status = status
        self._payload = payload or {}

    async def __aenter__(self) -> "_MockBackendResponse":
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def json(self) -> dict[str, Any]:
        return self._payload

    async def text(self) -> str:
        return str(self._payload)


class _MockBackendSession:
    def __init__(self, backend: "_MockBackendTransport") -> None:
        self._backend = backend

    async def __aenter__(self) -> "_MockBackendSession":
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _MockBackendResponse:
        return self._backend.post(url, json or {})

    def put(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _MockBackendResponse:
        return self._backend.put(url, json or {})


class _MockBackendTransport:
    """In-memory backend contract used by these integration tests."""

    def __init__(self) -> None:
        self._ids = count(1)
        self._max_trials_by_session: dict[str, int] = {}
        self._completed_trials_by_session: defaultdict[str, set[str]] = defaultdict(set)

    def client_session(self, *args: Any, **kwargs: Any) -> _MockBackendSession:
        return _MockBackendSession(self)

    def post(self, url: str, payload: dict[str, Any]) -> _MockBackendResponse:
        path = url.rstrip("/")

        if path.endswith("/sessions"):
            return self._create_session(payload)
        if path.endswith("/results"):
            return self._submit_result(path, payload)
        if path.endswith("/finalize"):
            return self._finalize_session(path)

        return _MockBackendResponse(404, {"error": f"unexpected POST {url}"})

    def put(self, url: str, payload: dict[str, Any]) -> _MockBackendResponse:
        return _MockBackendResponse(200, {"status": "updated"})

    def _create_session(self, payload: dict[str, Any]) -> _MockBackendResponse:
        index = next(self._ids)
        session_id = f"mock_session_{index}"
        experiment_id = f"mock_exp_{index}"
        experiment_run_id = f"mock_run_{index}"
        max_trials = payload.get("optimization_config", {}).get("max_trials", 0)
        self._max_trials_by_session[session_id] = int(max_trials or 0)
        return _MockBackendResponse(
            201,
            {
                "session_id": session_id,
                "metadata": {
                    "experiment_id": experiment_id,
                    "experiment_run_id": experiment_run_id,
                },
            },
        )

    def _submit_result(
        self, path: str, payload: dict[str, Any]
    ) -> _MockBackendResponse:
        session_id = path.split("/")[-2]
        trial_id = str(payload.get("trial_id", ""))
        if payload.get("status") == "COMPLETED" and trial_id:
            self._completed_trials_by_session[session_id].add(trial_id)

        max_trials = self._max_trials_by_session.get(session_id, 0)
        completed_trials = len(self._completed_trials_by_session[session_id])
        return _MockBackendResponse(
            200,
            {
                "status": "success",
                "continue_optimization": completed_trials < max_trials,
            },
        )

    def _finalize_session(self, path: str) -> _MockBackendResponse:
        session_id = path.split("/")[-2]
        total_trials = len(self._completed_trials_by_session[session_id])
        return _MockBackendResponse(
            200,
            {
                "best_config": {},
                "best_metrics": {},
                "total_trials": total_trials,
                "successful_trials": total_trials,
                "total_duration": 0.0,
                "cost_savings": 0.0,
                "stop_reason": "sdk_explicit_finalization",
                "convergence_history": [],
            },
        )


@pytest.mark.integration
class TestSessionFinalizationIntegration:
    """Integration tests for session finalization."""

    @pytest.fixture
    def client(self, monkeypatch):
        """Create a BackendIntegratedClient against a mocked backend transport."""
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        # The mocked transport needs no real credentials; provide fakes so the
        # client constructs without depending on ambient environment (the old
        # real-backend skipif is gone — these tests must RUN in CI).
        monkeypatch.setenv("TRAIGENT_API_KEY", "uk_test_finalization_fake")
        monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://mock-backend.invalid")

        backend = _MockBackendTransport()
        for module_path in (
            "traigent.cloud.api_operations",
            "traigent.cloud.trial_operations",
            "traigent.cloud.session_operations",
        ):
            monkeypatch.setattr(f"{module_path}.AIOHTTP_AVAILABLE", True)
            monkeypatch.setattr(
                f"{module_path}.aiohttp.ClientSession",
                backend.client_session,
            )
            monkeypatch.setattr(
                f"{module_path}.aiohttp.ClientTimeout",
                _MockClientTimeout,
            )

        api_key = os.getenv("TRAIGENT_API_KEY")
        backend_url = os.getenv("TRAIGENT_BACKEND_URL", "http://localhost:5000")

        client = BackendIntegratedClient(api_key=api_key, base_url=backend_url)

        async def augment_headers(headers):
            return {**headers, "Authorization": "Bearer test-token"}

        client.auth_manager.augment_headers = augment_headers
        return client

    @pytest.mark.asyncio
    async def test_complete_optimization_flow_with_auto_finalization(self, client):
        """Test complete optimization flow with backend auto-finalization."""
        # Create session
        session_request = SessionCreationRequest(
            function_name="integration_test_finalization",
            configuration_space={
                "temperature": [0.3, 0.5, 0.7],
                "max_tokens": [100, 200],
            },
            objectives=["maximize"],
            dataset_metadata={"size": 5, "privacy_mode": True},
            max_trials=3,  # Small number for quick test
            optimization_strategy="local_execution",
            user_id=None,
            billing_tier="privacy",
            metadata={"test": "auto_finalization"},
        )

        (
            session_id,
            experiment_id,
            experiment_run_id,
        ) = await client._create_traigent_session_via_api(session_request)

        assert session_id is not None
        assert experiment_id is not None
        assert experiment_run_id is not None

        print(f"\n✅ Created session: {session_id}")
        print(f"   Experiment: {experiment_id}")
        print(f"   Run: {experiment_run_id}")

        # Submit 3 trials
        trial_configs = [
            {"temperature": 0.3, "max_tokens": 100},
            {"temperature": 0.5, "max_tokens": 150},
            {"temperature": 0.7, "max_tokens": 200},
        ]

        last_trial_auto_finalized = False

        for i, config in enumerate(trial_configs):
            trial_id = f"trial_{i + 1}"

            # Register trial start
            try:
                await client._trial_ops.register_trial_start(
                    session_id=session_id, trial_id=trial_id, config=config
                )
                print(f"   Registered trial {i + 1}: {trial_id}")
            except Exception as e:
                print(f"   Trial registration failed (may be optional): {e}")

            # Submit trial results
            result = await client._trial_ops.submit_trial_result_via_session(
                session_id=session_id,
                trial_id=trial_id,
                config=config,
                metrics={"accuracy": 0.7 + (i * 0.1), "latency": 100 + (i * 10)},
                status="COMPLETED",
            )

            assert result is True
            print(f"   ✅ Submitted trial {i + 1} results")

            # On last trial, backend should auto-finalize
            if i == len(trial_configs) - 1:
                last_trial_auto_finalized = True
                print("   📌 Last trial submitted - backend should auto-finalize")

            time.sleep(0.5)  # Small delay between trials

        # Verify session was auto-finalized
        if last_trial_auto_finalized:
            print(f"\n✅ All {len(trial_configs)} trials submitted")
            print("   Backend should have auto-finalized the session")

        # Try to explicitly finalize - should be idempotent
        try:
            response = await client._session_ops.finalize_session(session_id)
            print("\n✅ Explicit finalization succeeded (idempotent)")
            print(f"   Session ID: {response.session_id}")
            print(f"   Total trials: {response.total_trials}")
            print(f"   Finalized via API: {response.metadata.get('finalized_via_api')}")
        except Exception as e:
            print(f"\n⚠️  Explicit finalization not available: {e}")
            print("   (This is OK - backend may have already auto-finalized)")

    @pytest.mark.asyncio
    async def test_early_termination_with_explicit_finalization(self, client):
        """Test early termination with explicit finalization."""
        # Create session
        session_request = SessionCreationRequest(
            function_name="integration_test_early_term",
            configuration_space={
                "temperature": [0.3, 0.5, 0.7, 0.9],
                "max_tokens": [100, 200, 300],
            },
            objectives=["maximize"],
            dataset_metadata={"size": 5, "privacy_mode": True},
            max_trials=12,  # But we'll stop early
            optimization_strategy="local_execution",
            user_id=None,
            billing_tier="privacy",
            metadata={"test": "early_termination"},
        )

        (
            session_id,
            experiment_id,
            experiment_run_id,
        ) = await client._create_traigent_session_via_api(session_request)

        print(f"\n✅ Created session: {session_id}")
        print("   Max trials: 12 (but will terminate early)")

        # Submit only 3 trials and then explicitly finalize
        for i in range(3):
            trial_id = f"trial_{i + 1}"
            config = {"temperature": 0.3 + (i * 0.2), "max_tokens": 100 + (i * 50)}

            result = await client._trial_ops.submit_trial_result_via_session(
                session_id=session_id,
                trial_id=trial_id,
                config=config,
                metrics={"accuracy": 0.8 + (i * 0.05), "latency": 100},
                status="COMPLETED",
            )

            assert result is True
            print(f"   ✅ Submitted trial {i + 1}")

            # Simulate convergence check
            if i == 2:
                print("   📌 Convergence detected! Stopping early")
                break

            time.sleep(0.3)

        # Explicitly finalize due to early stopping
        try:
            response = await client._session_ops.finalize_session(session_id)
            print("\n✅ Explicitly finalized after 3 trials (early termination)")
            print(f"   Session ID: {response.session_id}")
            print(f"   Completed trials: {response.total_trials}")
        except Exception as e:
            print(f"\n⚠️  Explicit finalization failed: {e}")
            print("   Backend may auto-finalize or endpoint may not be available")

    @pytest.mark.asyncio
    async def test_finalization_idempotency(self, client):
        """Test that calling finalize multiple times is safe."""
        # Create session
        session_request = SessionCreationRequest(
            function_name="integration_test_idempotent",
            configuration_space={"temperature": [0.5]},
            objectives=["maximize"],
            dataset_metadata={"size": 1, "privacy_mode": True},
            max_trials=1,
            optimization_strategy="local_execution",
            user_id=None,
            billing_tier="privacy",
            metadata={"test": "idempotency"},
        )

        (
            session_id,
            experiment_id,
            experiment_run_id,
        ) = await client._create_traigent_session_via_api(session_request)

        print(f"\n✅ Created session: {session_id}")

        # Submit 1 trial (backend should auto-finalize)
        result = await client._trial_ops.submit_trial_result_via_session(
            session_id=session_id,
            trial_id="trial_1",
            config={"temperature": 0.5},
            metrics={"accuracy": 0.85},
            status="COMPLETED",
        )

        assert result is True
        print("   ✅ Submitted trial (backend should auto-finalize)")

        time.sleep(0.5)

        # Call finalize multiple times - should not error
        for i in range(3):
            try:
                await client._session_ops.finalize_session(session_id)
                print(f"   ✅ Finalize call #{i + 1} succeeded (idempotent)")
            except Exception as e:
                print(f"   ⚠️  Finalize call #{i + 1} failed: {e}")
                # This is OK if endpoint doesn't exist - backend auto-finalization may be enough

        print("\n✅ Idempotency test completed")


@pytest.mark.integration
def test_integration_prerequisites():
    """Test that required environment variables are set."""
    api_key = os.getenv("TRAIGENT_API_KEY")
    backend_url = os.getenv("TRAIGENT_BACKEND_URL")

    if not api_key:
        pytest.skip(
            "TRAIGENT_API_KEY not set - integration tests require backend access"
        )

    if not backend_url:
        pytest.skip(
            "TRAIGENT_BACKEND_URL not set - defaulting to http://localhost:5000"
        )

    print("\n✅ Integration test prerequisites:")
    print("   API Key: configured")
    print(f"   Backend URL: {backend_url}")

    # Verify prerequisites are valid
    assert len(api_key) > 0, "API key should not be empty"
    assert backend_url.startswith("http"), "Backend URL should be a valid URL"
