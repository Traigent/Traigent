"""Mock cloud service implementations for testing.

This module provides mock implementations of TraiGent cloud services
for testing without requiring actual cloud connectivity.
"""

from unittest.mock import Mock

from traigent.cloud.models import (
    DatasetSubsetIndices,
    NextTrialResponse,
    OptimizationFinalizationResponse,
    OptimizationSessionStatus,
    SessionCreationResponse,
    TrialSuggestion,
)


class MockTraiGentCloudClient:
    """Mock TraiGent cloud client for testing."""

    def __init__(
        self,
        api_key: str = "test-key",
        base_url: str = "https://api.test.com",
        enable_fallback: bool = True,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.enable_fallback = enable_fallback
        self.max_retries = max_retries
        self.timeout = timeout
        self._session_counter = 0
        self._trial_counter = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def create_optimization_session(self, **kwargs) -> SessionCreationResponse:
        """Create a mock optimization session."""
        self._session_counter += 1
        session_id = f"session-{self._session_counter}"

        return SessionCreationResponse(
            session_id=session_id,
            status=OptimizationSessionStatus.ACTIVE,
            optimization_strategy={
                "algorithm": "bayesian",
                "acquisition_function": "expected_improvement",
                "initial_random_samples": 5,
            },
            estimated_duration=120.0,
            billing_estimate=0.05,
            metadata={"mock": True},
        )

    async def get_next_trial(self, session_id: str) -> NextTrialResponse:
        """Get next trial suggestion."""
        self._trial_counter += 1
        trial_id = f"trial-{self._trial_counter}"

        # Mock configuration suggestions
        configs = [
            {"model": "gpt-3.5-turbo", "temperature": 0.1},
            {"model": "gpt-4", "temperature": 0.3},
            {"model": "gpt-4o-mini", "temperature": 0.5},
            {"model": "gpt-3.5-turbo", "temperature": 0.7},
            {"model": "gpt-4", "temperature": 0.9},
        ]
        config = configs[(self._trial_counter - 1) % len(configs)]

        suggestion = TrialSuggestion(
            trial_id=trial_id,
            session_id=session_id,
            trial_number=self._trial_counter,
            config=config,
            dataset_subset=DatasetSubsetIndices(
                indices=list(range(min(10, self._trial_counter * 2))),
                selection_strategy="adaptive",
                confidence_level=0.95,
                estimated_representativeness=0.9,
                metadata={},
            ),
            exploration_type=(
                "exploration" if self._trial_counter <= 3 else "exploitation"
            ),
            priority=1,
            estimated_duration=30.0,
            metadata={"mock_trial": True},
        )

        should_continue = self._trial_counter < 10  # Stop after 10 trials

        return NextTrialResponse(
            suggestion=suggestion,
            should_continue=should_continue,
            reason=(
                "Continue optimization" if should_continue else "Convergence achieved"
            ),
            session_status=(
                OptimizationSessionStatus.ACTIVE
                if should_continue
                else OptimizationSessionStatus.COMPLETED
            ),
            metadata={"trials_completed": self._trial_counter},
        )

    async def submit_trial_result(self, session_id: str, trial_id: str, **kwargs):
        """Submit trial result (mock implementation)."""
        return {"status": "accepted", "trial_id": trial_id}

    async def finalize_optimization(
        self, session_id: str
    ) -> OptimizationFinalizationResponse:
        """Finalize optimization session."""
        return OptimizationFinalizationResponse(
            session_id=session_id,
            best_config={"model": "gpt-4", "temperature": 0.3},
            best_metrics={"accuracy": 0.95, "cost": 0.08, "latency": 1.2},
            total_trials=self._trial_counter,
            successful_trials=self._trial_counter,
            total_duration=150.0,
            cost_savings=0.3,
            convergence_history=[],
            full_history=None,
            metadata={"optimization_completed": True},
        )

    async def optimize_function(self, **kwargs):
        """Direct function optimization (simplified mock)."""
        return Mock(
            best_config={"model": "gpt-4", "temperature": 0.1, "max_tokens": 500},
            best_metrics={"accuracy": 0.94, "cost": 0.08, "latency": 1.2},
            trials_count=15,
            cost_reduction=0.67,
            optimization_time=180.0,
            subset_used=True,
            subset_size=8,
        )


class MockTraiGentCloudClientWithAuth(MockTraiGentCloudClient):
    """Mock cloud client with authentication simulation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._authenticated = True

    async def authenticate(self) -> bool:
        """Mock authentication."""
        return self._authenticated

    def set_authenticated(self, status: bool):
        """Set authentication status for testing."""
        self._authenticated = status


class MockPrivacyCloudClient(MockTraiGentCloudClient):
    """Mock cloud client for privacy-first testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.privacy_mode = True
        self.data_retained = False

    async def create_optimization_session(self, **kwargs):
        """Create privacy-focused session."""
        response = await super().create_optimization_session(**kwargs)
        response.metadata.update(
            {"privacy_mode": True, "data_retention": "none", "encryption": "end_to_end"}
        )
        return response

    async def submit_trial_result(self, session_id: str, trial_id: str, **kwargs):
        """Submit trial result with privacy guarantees."""
        # Simulate data processing without retention
        processed_result = {
            "metrics_only": True,
            "raw_data_discarded": True,
            "privacy_preserved": True,
        }
        return {"status": "accepted", "trial_id": trial_id, "privacy": processed_result}


class MockHybridCloudClient:
    """Mock hybrid cloud client for local/cloud testing."""

    def __init__(self, prefer_local: bool = True):
        self.prefer_local = prefer_local
        self.local_fallback_count = 0
        self.cloud_success_count = 0

    async def optimize_function(self, **kwargs):
        """Optimize with hybrid local/cloud approach."""
        if self.prefer_local or self.should_use_local():
            self.local_fallback_count += 1
            return self._local_optimization(**kwargs)
        else:
            self.cloud_success_count += 1
            return self._cloud_optimization(**kwargs)

    def should_use_local(self) -> bool:
        """Determine if should use local optimization."""
        # Simulate conditions where local is preferred
        return self.prefer_local or self.cloud_success_count % 3 == 0

    def _local_optimization(self, **kwargs):
        """Mock local optimization."""
        return Mock(
            best_config={"model": "gpt-3.5-turbo", "temperature": 0.5},
            best_metrics={"accuracy": 0.88, "cost": 0.02, "latency": 0.8},
            trials_count=8,
            optimization_mode="edge_analytics",
            fallback_reason="privacy_preferred",
        )

    def _cloud_optimization(self, **kwargs):
        """Mock cloud optimization."""
        return Mock(
            best_config={"model": "gpt-4", "temperature": 0.2},
            best_metrics={"accuracy": 0.93, "cost": 0.06, "latency": 1.1},
            trials_count=12,
            optimization_mode="cloud",
            cloud_benefits=["advanced_algorithms", "larger_search_space"],
        )


class MockCloudServiceError(Exception):
    """Mock cloud service error for testing error handling."""

    pass


def create_mock_cloud_client(client_type: str = "standard", **kwargs):
    """Factory function to create mock cloud clients."""
    clients = {
        "standard": MockTraiGentCloudClient,
        "auth": MockTraiGentCloudClientWithAuth,
        "privacy": MockPrivacyCloudClient,
        "hybrid": MockHybridCloudClient,
    }

    if client_type not in clients:
        raise ValueError(f"Unknown client type: {client_type}")

    return clients[client_type](**kwargs)
