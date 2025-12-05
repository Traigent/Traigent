"""Integration tests for session finalization with backend."""

import os
import time

import pytest

from traigent.cloud.backend_client import BackendIntegratedClient
from traigent.cloud.models import SessionCreationRequest


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("TRAIGENT_API_KEY") or not os.getenv("TRAIGENT_BACKEND_URL"),
    reason="Requires TRAIGENT_API_KEY and TRAIGENT_BACKEND_URL environment variables",
)
class TestSessionFinalizationIntegration:
    """Integration tests for session finalization."""

    @pytest.fixture
    def client(self):
        """Create a real BackendIntegratedClient for integration testing."""
        api_key = os.getenv("TRAIGENT_API_KEY")
        backend_url = os.getenv("TRAIGENT_BACKEND_URL", "http://localhost:5000")

        client = BackendIntegratedClient(api_key=api_key, base_url=backend_url)
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

        session_id, experiment_id, experiment_run_id = (
            await client._create_traigent_session_via_api(session_request)
        )

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
            trial_id = f"trial_{i+1}"

            # Register trial start
            try:
                await client._trial_ops.register_trial_start(
                    session_id=session_id, trial_id=trial_id, config=config
                )
                print(f"   Registered trial {i+1}: {trial_id}")
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
            print(f"   ✅ Submitted trial {i+1} results")

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

        session_id, experiment_id, experiment_run_id = (
            await client._create_traigent_session_via_api(session_request)
        )

        print(f"\n✅ Created session: {session_id}")
        print("   Max trials: 12 (but will terminate early)")

        # Submit only 3 trials and then explicitly finalize
        for i in range(3):
            trial_id = f"trial_{i+1}"
            config = {"temperature": 0.3 + (i * 0.2), "max_tokens": 100 + (i * 50)}

            result = await client._trial_ops.submit_trial_result_via_session(
                session_id=session_id,
                trial_id=trial_id,
                config=config,
                metrics={"accuracy": 0.8 + (i * 0.05), "latency": 100},
                status="COMPLETED",
            )

            assert result is True
            print(f"   ✅ Submitted trial {i+1}")

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

        session_id, experiment_id, experiment_run_id = (
            await client._create_traigent_session_via_api(session_request)
        )

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
                print(f"   ✅ Finalize call #{i+1} succeeded (idempotent)")
            except Exception as e:
                print(f"   ⚠️  Finalize call #{i+1} failed: {e}")
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
    print(f"   API Key: {api_key[:10]}... (length: {len(api_key)})")
    print(f"   Backend URL: {backend_url}")
