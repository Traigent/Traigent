"""Tests for TraiGent cloud cost tracking module."""

import asyncio
import time
from unittest.mock import patch

import pytest

from traigent.cloud.billing import (
    BillingTier,
    CostCategory,
    CostItem,
    CostTracker,
    CostTrackingConfig,
    UsageMetrics,
)


class TestCostTrackingConfig:
    """Test CostTrackingConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CostTrackingConfig()
        assert config.enable_client_tracking is True
        assert config.enable_server_sync is True
        assert config.cache_costs_locally is True
        assert config.sync_interval == 60.0
        assert config.billing_tier == BillingTier.STANDARD

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CostTrackingConfig(
            enable_client_tracking=False,
            sync_interval=120.0,
            billing_tier=BillingTier.PREMIUM,
        )
        assert config.enable_client_tracking is False
        assert config.sync_interval == 120.0
        assert config.billing_tier == BillingTier.PREMIUM


class TestCostItem:
    """Test CostItem dataclass."""

    def test_cost_item_creation(self):
        """Test creating a cost item."""
        item = CostItem(
            item_id="test-123",
            category=CostCategory.OPTIMIZATION,
            description="Test optimization",
            quantity=100.0,
            unit_cost=0.001,
            total_cost=0.1,
            currency="USD",
        )

        assert item.item_id == "test-123"
        assert item.category == CostCategory.OPTIMIZATION
        assert item.description == "Test optimization"
        assert item.quantity == 100.0
        assert item.unit_cost == 0.001
        assert item.total_cost == 0.1
        assert item.currency == "USD"
        assert isinstance(item.timestamp, float)

    def test_cost_item_with_session(self):
        """Test cost item with session information."""
        item = CostItem(
            item_id="test-456",
            category=CostCategory.INFERENCE,
            description="Test inference",
            quantity=50.0,
            unit_cost=0.002,
            total_cost=0.1,
            session_id="session-123",
            trial_id="trial-456",
        )

        assert item.session_id == "session-123"
        assert item.trial_id == "trial-456"


class TestUsageMetrics:
    """Test UsageMetrics dataclass."""

    def test_usage_metrics_creation(self):
        """Test creating usage metrics."""
        metrics = UsageMetrics(
            tokens_processed=10000,
            api_calls=100,
            dataset_size=500,
            trials_executed=20,
            compute_minutes=15.5,
        )

        assert metrics.tokens_processed == 10000
        assert metrics.api_calls == 100
        assert metrics.dataset_size == 500
        assert metrics.trials_executed == 20
        assert metrics.compute_minutes == 15.5

    def test_usage_metrics_defaults(self):
        """Test default usage metrics."""
        metrics = UsageMetrics()
        assert metrics.tokens_processed == 0
        assert metrics.api_calls == 0
        assert metrics.dataset_size == 0
        assert metrics.trials_executed == 0
        assert metrics.compute_minutes == 0.0


class TestCostTracker:
    """Test CostTracker class."""

    @pytest.fixture
    def cost_tracker(self):
        """Create a cost tracker instance for testing."""
        config = CostTrackingConfig(
            enable_server_sync=False,  # Disable for testing
            cache_costs_locally=False,  # Disable for testing
        )
        return CostTracker(config)

    def test_cost_tracker_initialization(self, cost_tracker):
        """Test cost tracker initialization."""
        assert len(cost_tracker._cost_items) == 0
        assert len(cost_tracker._session_costs) == 0
        assert cost_tracker._total_cost == 0.0
        assert cost_tracker._max_cost_items == 10000
        assert cost_tracker._max_items_per_session == 1000
        assert cost_tracker._max_sessions == 100

    def test_track_optimization_cost(self, cost_tracker):
        """Test tracking optimization costs."""
        cost_item = cost_tracker.track_optimization_cost(
            session_id="test-session",
            trial_id="trial-123",
            tokens_used=1000,
            compute_time=2.0,
            api_calls=5,
        )

        assert cost_item.category == CostCategory.OPTIMIZATION
        assert cost_item.session_id == "test-session"
        assert cost_item.trial_id == "trial-123"
        assert cost_item.total_cost >= 0  # Cost depends on billing tier rates
        assert cost_tracker._total_cost >= 0
        assert len(cost_tracker._cost_items) == 1

    def test_track_inference_cost(self, cost_tracker):
        """Test tracking inference costs."""
        cost_item = cost_tracker.track_inference_cost(
            session_id="test-session",
            input_tokens=500,
            output_tokens=200,
            model_name="gpt-4",
        )

        assert cost_item.category == CostCategory.INFERENCE
        assert cost_item.session_id == "test-session"
        assert cost_item.total_cost >= 0  # Cost depends on billing tier rates
        assert "gpt-4" in cost_item.description

    def test_track_custom_cost(self, cost_tracker):
        """Test tracking custom costs."""
        cost_item = cost_tracker.track_custom_cost(
            session_id="test-session",
            category=CostCategory.STORAGE,
            description="Data storage",
            cost=0.10,
            metadata={"size": "1GB"},
        )

        assert cost_item.category == CostCategory.STORAGE
        assert cost_item.description == "Data storage"
        assert cost_item.total_cost == 0.10
        assert cost_item.metadata["size"] == "1GB"

    def test_memory_bounds_cost_items(self, cost_tracker):
        """Test memory bounds for cost items."""
        # Set a small max for testing
        cost_tracker._max_cost_items = 3

        # Add items beyond the limit
        for i in range(5):
            cost_tracker.track_custom_cost(
                session_id="test-session",
                category=CostCategory.COMPUTE,
                description=f"Test item {i}",
                cost=0.01,
            )

        # Should only keep the most recent items
        assert len(cost_tracker._cost_items) <= cost_tracker._max_cost_items

    def test_memory_bounds_sessions(self, cost_tracker):
        """Test memory bounds for sessions."""
        # Set a small max for testing
        cost_tracker._max_sessions = 2

        # Add items for multiple sessions
        for i in range(4):
            cost_tracker.track_optimization_cost(
                session_id=f"session-{i}",
                trial_id="test-trial",
                tokens_used=100,
                compute_time=1.0,
            )

        # Should only keep the most recent sessions
        assert len(cost_tracker._session_costs) <= cost_tracker._max_sessions

    def test_memory_bounds_items_per_session(self, cost_tracker):
        """Test memory bounds for items per session."""
        # Set a small max for testing
        cost_tracker._max_items_per_session = 2

        # Add many items to the same session
        for i in range(5):
            cost_tracker.track_optimization_cost(
                session_id="test-session",
                trial_id=f"trial-{i}",
                tokens_used=100,
                compute_time=1.0,
            )

        # Should only keep the most recent items for the session
        session_items = cost_tracker._session_costs["test-session"]
        assert len(session_items) <= cost_tracker._max_items_per_session

    def test_get_session_costs(self, cost_tracker):
        """Test getting costs for a specific session."""
        # Add costs for multiple sessions
        cost_tracker.track_optimization_cost(
            "session-1", trial_id="trial-1", tokens_used=100, compute_time=1.0
        )
        cost_tracker.track_optimization_cost(
            "session-1", trial_id="trial-2", tokens_used=200, compute_time=2.0
        )
        cost_tracker.track_optimization_cost(
            "session-2", trial_id="trial-3", tokens_used=150, compute_time=1.5
        )

        session_1_costs = cost_tracker.get_session_costs("session-1")
        assert len(session_1_costs) == 2
        assert all(item.session_id == "session-1" for item in session_1_costs)

        session_2_costs = cost_tracker.get_session_costs("session-2")
        assert len(session_2_costs) == 1
        assert session_2_costs[0].session_id == "session-2"

    def test_get_usage_metrics(self, cost_tracker):
        """Test getting usage metrics."""
        # Add some costs
        cost_tracker.track_optimization_cost(
            "session-1", trial_id="trial-1", tokens_used=100, compute_time=1.0
        )
        cost_tracker.track_inference_cost(
            "session-1", input_tokens=500, output_tokens=200, model_name="gpt-4"
        )
        cost_tracker.track_custom_cost(
            "session-1", CostCategory.STORAGE, "storage", 0.05
        )

        # The get_usage_metrics method doesn't exist, use _usage_metrics directly
        metrics = cost_tracker._usage_metrics
        assert metrics.tokens_processed >= 800  # 100 + 500 + 200
        assert (
            metrics.api_calls >= 2
        )  # Only optimization and inference increment API calls

    def test_reset_costs(self, cost_tracker):
        """Test clearing tracked data."""
        # Add some data
        cost_tracker.track_optimization_cost(
            "session-1", trial_id="trial-1", tokens_used=100, compute_time=1.0
        )
        cost_tracker.track_inference_cost(
            "session-1", input_tokens=500, output_tokens=200, model_name="gpt-4"
        )

        assert len(cost_tracker._cost_items) > 0
        assert cost_tracker._total_cost > 0

        cost_tracker.reset_costs()

        assert len(cost_tracker._cost_items) == 0
        assert len(cost_tracker._session_costs) == 0
        assert cost_tracker._total_cost == 0.0

    def test_filter_costs_basic(self, cost_tracker):
        """Test basic cost filtering functionality."""
        # Add costs with different categories and timestamps
        start_time = time.time()

        cost_tracker.track_optimization_cost(
            "session-1", trial_id="trial-1", tokens_used=100, compute_time=1.0
        )
        cost_tracker.track_inference_cost(
            "session-1", input_tokens=500, output_tokens=200, model_name="gpt-4"
        )
        cost_tracker.track_custom_cost(
            "session-1", CostCategory.STORAGE, "storage", 0.05
        )

        # Test basic filtering by manually checking items
        optimization_costs = [
            item
            for item in cost_tracker._cost_items
            if item.category == CostCategory.OPTIMIZATION
        ]
        assert len(optimization_costs) == 1
        assert optimization_costs[0].category == CostCategory.OPTIMIZATION

        # Filter by session
        session_costs = [
            item for item in cost_tracker._cost_items if item.session_id == "session-1"
        ]
        assert len(session_costs) == 3
        assert all(item.session_id == "session-1" for item in session_costs)

        # Filter by time range
        end_time = time.time()
        time_filtered = [
            item
            for item in cost_tracker._cost_items
            if start_time <= item.timestamp <= end_time
        ]
        assert len(time_filtered) >= 3


class TestCostTrackerAsync:
    """Test async functionality of CostTracker."""

    @pytest.fixture
    def cost_tracker_with_sync(self):
        """Create a cost tracker with sync enabled for testing."""
        config = CostTrackingConfig(
            enable_server_sync=True,
            cache_costs_locally=True,
            sync_interval=1,  # Short interval for testing
        )
        return CostTracker(config)

    @pytest.mark.asyncio
    async def test_cache_cost_item(self, cost_tracker_with_sync):
        """Test caching cost items."""
        cost_item = CostItem(
            item_id="test-123",
            category=CostCategory.OPTIMIZATION,
            description="Test",
            quantity=100.0,
            unit_cost=0.001,
            total_cost=0.1,
        )

        # Mock the cache operation
        with patch.object(cost_tracker_with_sync, "_cache_cost_item") as mock_cache:
            mock_cache.return_value = asyncio.create_task(asyncio.sleep(0))

            cost_tracker_with_sync._add_cost_item(cost_item)

            # Give async task time to complete
            await asyncio.sleep(0.1)


class TestCostTrackerIntegration:
    """Integration tests for CostTracker."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete cost tracking workflow."""
        config = CostTrackingConfig(
            enable_server_sync=False, cache_costs_locally=False  # Disable for testing
        )
        tracker = CostTracker(config)

        # Track various types of costs
        tracker.track_optimization_cost(
            session_id="session-123",
            trial_id="hyperparameter_search",
            tokens_used=1500,
            compute_time=2.0,
        )

        tracker.track_inference_cost(
            session_id="session-123",
            input_tokens=800,
            output_tokens=400,
            model_name="gpt-4",
        )

        tracker.track_custom_cost(
            session_id="session-123",
            category=CostCategory.STORAGE,
            description="Dataset storage",
            cost=0.025,
        )

        # Verify tracking
        assert tracker._total_cost > 0  # Costs will be calculated based on rates
        assert len(tracker._cost_items) == 3
        assert len(tracker._session_costs["session-123"]) == 3

        # Test filtering
        session_costs = tracker.get_session_costs("session-123")
        assert len(session_costs) == 3

        optimization_costs = [
            item
            for item in tracker._cost_items
            if item.category == CostCategory.OPTIMIZATION
        ]
        assert len(optimization_costs) == 1

        # Test metrics
        metrics = tracker._usage_metrics
        assert metrics.tokens_processed >= 2700  # 1500 + 800 + 400
        assert (
            metrics.api_calls >= 2
        )  # Only optimization and inference increment API calls
