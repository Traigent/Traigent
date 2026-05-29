"""Tests for Traigent Cloud Service billing and usage tracking."""

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from traigent.cloud.billing import (
    BillingManager,
    BillingPlan,
    UsageRecord,
    UsageTracker,
    _read_json_file,
    _write_json_file,
)


class TestUsageRecord:
    """Test cases for UsageRecord dataclass."""

    def test_usage_record_creation(self):
        """Test UsageRecord creation."""
        timestamp = datetime.now(UTC)

        record = UsageRecord(
            timestamp=timestamp,
            function_name="test_function",
            trials_count=25,
            dataset_size=100,
            optimization_time=120.5,
            cost_credits=5.25,
            billing_tier="standard",
        )

        assert record.timestamp == timestamp
        assert record.function_name == "test_function"
        assert record.trials_count == 25
        assert record.dataset_size == 100
        assert record.optimization_time == 120.5
        assert record.cost_credits == 5.25
        assert record.billing_tier == "standard"

    def test_usage_record_to_dict(self):
        """Test UsageRecord serialization to dictionary."""
        timestamp = datetime.now(UTC)

        record = UsageRecord(
            timestamp=timestamp,
            function_name="test_function",
            trials_count=25,
            dataset_size=100,
            optimization_time=120.5,
            cost_credits=5.25,
        )

        record_dict = record.to_dict()

        assert record_dict["timestamp"] == timestamp.isoformat()
        assert record_dict["function_name"] == "test_function"
        assert record_dict["trials_count"] == 25
        assert record_dict["dataset_size"] == 100
        assert record_dict["optimization_time"] == 120.5
        assert record_dict["cost_credits"] == 5.25
        assert record_dict["billing_tier"] == "standard"

    def test_usage_record_from_dict(self):
        """Test UsageRecord deserialization from dictionary."""
        timestamp = datetime.now(UTC)

        record_dict = {
            "timestamp": timestamp.isoformat(),
            "function_name": "test_function",
            "trials_count": 25,
            "dataset_size": 100,
            "optimization_time": 120.5,
            "cost_credits": 5.25,
            "billing_tier": "professional",
        }

        record = UsageRecord.from_dict(record_dict)

        assert record.timestamp == timestamp
        assert record.function_name == "test_function"
        assert record.trials_count == 25
        assert record.dataset_size == 100
        assert record.optimization_time == 120.5
        assert record.cost_credits == 5.25
        assert record.billing_tier == "professional"

    def test_usage_record_from_dict_default_tier(self):
        """Test UsageRecord deserialization with default billing tier."""
        timestamp = datetime.now(UTC)

        record_dict = {
            "timestamp": timestamp.isoformat(),
            "function_name": "test_function",
            "trials_count": 25,
            "dataset_size": 100,
            "optimization_time": 120.5,
            "cost_credits": 5.25,
            # No billing_tier specified
        }

        record = UsageRecord.from_dict(record_dict)
        assert record.billing_tier == "standard"


class TestBillingPlan:
    """Test cases for BillingPlan dataclass."""

    def test_billing_plan_creation(self):
        """Test BillingPlan creation."""
        plan = BillingPlan(
            name="Professional",
            monthly_credits=10000,
            cost_per_credit=0.008,
            max_trials_per_optimization=500,
            max_dataset_size=10000,
            priority_support=True,
            advanced_algorithms=True,
        )

        assert plan.name == "Professional"
        assert plan.monthly_credits == 10000
        assert plan.cost_per_credit == 0.008
        assert plan.max_trials_per_optimization == 500
        assert plan.max_dataset_size == 10000
        assert plan.priority_support is True
        assert plan.advanced_algorithms is True

    def test_billing_plan_defaults(self):
        """Test BillingPlan with default values."""
        plan = BillingPlan(
            name="Basic",
            monthly_credits=1000,
            cost_per_credit=0.01,
            max_trials_per_optimization=100,
            max_dataset_size=1000,
        )

        assert plan.priority_support is False
        assert plan.advanced_algorithms is False

    def test_billing_plan_calculate_monthly_cost(self):
        """Test monthly cost calculation."""
        plan = BillingPlan(
            name="Test Plan",
            monthly_credits=1000,
            cost_per_credit=0.01,
            max_trials_per_optimization=100,
            max_dataset_size=1000,
        )

        monthly_cost = plan.calculate_monthly_cost()
        assert monthly_cost == 10.0  # 1000 * 0.01


class TestUsageTracker:
    """Test cases for UsageTracker."""

    def test_json_helpers_round_trip(self, tmp_path):
        """The async file helpers should preserve JSON payloads."""
        payload = {"usage_records": [{"function_name": "test_func"}]}
        storage_path = tmp_path / "usage.json"

        _write_json_file(storage_path, payload)

        assert _read_json_file(storage_path) == payload

    def test_usage_tracker_initialization_default_path(self):
        """Test UsageTracker initialization with default path."""
        tracker = UsageTracker()

        expected_path = Path.home() / ".traigent" / "usage.json"
        assert tracker.storage_path == expected_path
        assert (
            len(tracker.billing_plans) == 4
        )  # free, standard, professional, enterprise

    def test_usage_tracker_initialization_custom_path(self):
        """Test UsageTracker initialization with custom path."""
        custom_path = "/tmp/test_usage.json"
        tracker = UsageTracker(storage_path=custom_path)

        assert tracker.storage_path == Path(custom_path)

    def test_usage_tracker_billing_plans(self):
        """Test UsageTracker billing plans setup."""
        tracker = UsageTracker()

        # Check all plans exist
        assert "free" in tracker.billing_plans
        assert "standard" in tracker.billing_plans
        assert "professional" in tracker.billing_plans
        assert "enterprise" in tracker.billing_plans

        # Check plan properties
        free_plan = tracker.billing_plans["free"]
        assert free_plan.name == "Free"
        assert free_plan.monthly_credits == 100
        assert free_plan.cost_per_credit == 0.0

        enterprise_plan = tracker.billing_plans["enterprise"]
        assert enterprise_plan.name == "Enterprise"
        assert enterprise_plan.max_trials_per_optimization == -1  # Unlimited
        assert enterprise_plan.max_dataset_size == -1  # Unlimited

    def test_record_optimization(self):
        """Test recording optimization usage."""

        async def run_test():
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as f:
                tracker = UsageTracker(storage_path=f.name)

                record = await tracker.record_optimization(
                    function_name="test_function",
                    trials_count=25,
                    dataset_size=100,
                    optimization_time=120.5,
                    billing_tier="standard",
                )

                assert isinstance(record, UsageRecord)
                assert record.function_name == "test_function"
                assert record.trials_count == 25
                assert record.dataset_size == 100
                assert record.optimization_time == 120.5
                assert record.billing_tier == "standard"
                assert record.cost_credits > 0

                # Check that record was added to tracker
                assert len(tracker._usage_records) == 1
                assert tracker._usage_records[0] == record

        asyncio.run(run_test())

    def test_calculate_cost_credits_standard(self):
        """Test cost calculation for standard tier."""
        tracker = UsageTracker()

        cost = tracker._calculate_cost_credits(
            trials_count=50, dataset_size=200, billing_tier="standard"
        )

        # Base cost: 0.1, Trial cost: 50*0.01=0.5, Data cost: 200*0.001=0.2
        # Total: 0.8, Standard discount: 0.9
        expected_cost = (0.1 + 0.5 + 0.2) * 0.9
        assert cost == expected_cost

    def test_calculate_cost_credits_enterprise(self):
        """Test cost calculation for enterprise tier."""
        tracker = UsageTracker()

        cost = tracker._calculate_cost_credits(
            trials_count=100, dataset_size=500, billing_tier="enterprise"
        )

        # Base cost: 0.1, Trial cost: 100*0.01=1.0, Data cost: 500*0.001=0.5
        # Total: 1.6, Enterprise discount: 0.7
        expected_cost = (0.1 + 1.0 + 0.5) * 0.7
        assert cost == expected_cost

    def test_calculate_cost_credits_unknown_tier(self):
        """Test cost calculation for unknown tier."""
        tracker = UsageTracker()

        cost = tracker._calculate_cost_credits(
            trials_count=10, dataset_size=50, billing_tier="unknown"
        )

        # Should use default discount of 1.0 (no discount)
        expected_cost = 0.1 + 0.1 + 0.05
        assert cost == expected_cost

    def test_get_usage_stats_empty(self):
        """Test getting usage stats with no records."""

        async def run_test():
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as f:
                tracker = UsageTracker(storage_path=f.name)

                stats = await tracker.get_usage_stats()

                assert stats["total_optimizations"] == 0
                assert stats["total_trials"] == 0
                assert stats["total_credits"] == 0.0
                assert stats["total_time"] == 0.0
                assert stats["functions_optimized"] == []

        asyncio.run(run_test())

    def test_get_usage_stats_with_records(self):
        """Test getting usage stats with records."""

        async def run_test():
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as f:
                tracker = UsageTracker(storage_path=f.name)

                # Add some test records
                await tracker.record_optimization("func1", 25, 100, 60.0, "standard")
                await tracker.record_optimization("func2", 30, 150, 90.0, "standard")
                await tracker.record_optimization("func1", 20, 80, 45.0, "professional")

                stats = await tracker.get_usage_stats()

                assert stats["total_optimizations"] == 3
                assert stats["total_trials"] == 75  # 25 + 30 + 20
                assert stats["total_time"] == 195.0  # 60 + 90 + 45
                assert stats["avg_trials_per_optimization"] == 25.0
                assert set(stats["functions_optimized"]) == {"func1", "func2"}

                # Check function-specific stats
                func1_stats = stats["function_stats"]["func1"]
                assert func1_stats["optimizations"] == 2
                assert func1_stats["total_trials"] == 45  # 25 + 20

        asyncio.run(run_test())

    def test_get_usage_stats_date_range(self):
        """Test getting usage stats for specific date range."""

        async def run_test():
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as f:
                tracker = UsageTracker(storage_path=f.name)

                # Create records with specific timestamps
                now = datetime.now(UTC)
                yesterday = now - timedelta(days=1)

                # Add record from yesterday
                record_yesterday = UsageRecord(
                    timestamp=yesterday,
                    function_name="func1",
                    trials_count=25,
                    dataset_size=100,
                    optimization_time=60.0,
                    cost_credits=2.5,
                )
                tracker._usage_records.append(record_yesterday)

                # Add record from today
                await tracker.record_optimization("func2", 30, 150, 90.0)

                # Get stats for today only
                start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
                stats = await tracker.get_usage_stats(start_date=start_of_today)

                assert stats["total_optimizations"] == 1  # Only today's record
                assert stats["functions_optimized"] == ["func2"]

        asyncio.run(run_test())

    def test_get_billing_plan(self):
        """Test getting billing plan details."""
        tracker = UsageTracker()

        plan = tracker.get_billing_plan("professional")
        assert plan is not None
        assert plan.name == "Professional"
        assert plan.monthly_credits == 10000

        # Test case insensitive
        plan = tracker.get_billing_plan("PROFESSIONAL")
        assert plan is not None

        # Test non-existent plan
        plan = tracker.get_billing_plan("nonexistent")
        assert plan is None

    def test_list_billing_plans(self):
        """Test listing all billing plans."""
        tracker = UsageTracker()

        plans = tracker.list_billing_plans()
        assert len(plans) == 4
        assert "free" in plans
        assert "standard" in plans
        assert "professional" in plans
        assert "enterprise" in plans

    def test_estimate_monthly_cost(self):
        """Test monthly cost estimation."""

        async def run_test():
            tracker = UsageTracker()

            estimate = await tracker.estimate_monthly_cost(
                optimizations_per_month=100,
                avg_trials=50,
                avg_dataset_size=200,
                billing_tier="standard",
            )

            assert estimate["optimizations_per_month"] == 100
            assert estimate["billing_tier"] == "standard"
            assert estimate["credits_per_optimization"] > 0
            assert estimate["total_credits_per_month"] > 0
            assert estimate["monthly_cost_usd"] > 0

        asyncio.run(run_test())

    def test_estimate_monthly_cost_unknown_tier(self):
        """Test monthly cost estimation with unknown tier."""

        async def run_test():
            tracker = UsageTracker()

            with pytest.raises(ValueError, match="Unknown billing tier"):
                await tracker.estimate_monthly_cost(
                    optimizations_per_month=100, billing_tier="unknown_tier"
                )

        asyncio.run(run_test())

    def test_load_usage_data_file_not_exists(self):
        """Test loading usage data when file doesn't exist."""
        with tempfile.NamedTemporaryFile(delete=True) as f:
            # File gets deleted, so it won't exist
            tracker = UsageTracker(storage_path=f.name)

            assert len(tracker._usage_records) == 0

    def test_load_usage_data_invalid_json(self):
        """Test loading usage data with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write("invalid json content")
            f.flush()

            # Should handle error gracefully
            tracker = UsageTracker(storage_path=f.name)
            assert len(tracker._usage_records) == 0

    def test_save_and_load_usage_data(self):
        """Test saving and loading usage data."""

        async def run_test():
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as f:
                tracker = UsageTracker(storage_path=f.name)

                # Add a record
                await tracker.record_optimization("test_func", 25, 100, 60.0)

                # Create new tracker with same storage path
                tracker2 = UsageTracker(storage_path=f.name)

                # Should load the saved record
                assert len(tracker2._usage_records) == 1
                assert tracker2._usage_records[0].function_name == "test_func"

        asyncio.run(run_test())


class TestBillingManager:
    """Test cases for BillingManager."""

    def test_billing_manager_initialization(self):
        """Test BillingManager initialization."""
        tracker = UsageTracker()
        manager = BillingManager(tracker)

        assert manager.usage_tracker == tracker
        assert manager.current_plan == "free"

    def test_check_usage_limits_within_limits(self):
        """Test usage limits check when within limits."""

        async def run_test():
            tracker = UsageTracker()
            manager = BillingManager(tracker)
            manager.current_plan = "standard"

            result = await manager.check_usage_limits(
                trials_count=50,  # Within standard limit of 100
                dataset_size=500,  # Within standard limit of 1000
            )

            assert result["allowed"] is True
            assert "estimated_cost" in result
            assert "remaining_credits" in result

        asyncio.run(run_test())

    def test_check_usage_limits_trials_exceeded(self):
        """Test usage limits check when trials limit exceeded."""

        async def run_test():
            tracker = UsageTracker()
            manager = BillingManager(tracker)
            manager.current_plan = "standard"

            result = await manager.check_usage_limits(
                trials_count=200, dataset_size=500  # Exceeds standard limit of 100
            )

            assert result["allowed"] is False
            assert "Trials limit exceeded" in result["reason"]
            assert "suggested_action" in result

        asyncio.run(run_test())

    def test_check_usage_limits_dataset_exceeded(self):
        """Test usage limits check when dataset size limit exceeded."""

        async def run_test():
            tracker = UsageTracker()
            manager = BillingManager(tracker)
            manager.current_plan = "standard"

            result = await manager.check_usage_limits(
                trials_count=50, dataset_size=2000  # Exceeds standard limit of 1000
            )

            assert result["allowed"] is False
            assert "Dataset size limit exceeded" in result["reason"]

        asyncio.run(run_test())

    def test_check_usage_limits_enterprise_unlimited(self):
        """Test usage limits check for enterprise (unlimited)."""

        async def run_test():
            tracker = UsageTracker()
            manager = BillingManager(tracker)
            manager.current_plan = "enterprise"

            result = await manager.check_usage_limits(
                trials_count=1000,  # Would exceed other plans
                dataset_size=50000,  # Would exceed other plans
            )

            assert result["allowed"] is True

        asyncio.run(run_test())

    def test_check_usage_limits_credits_exceeded(self):
        """Test usage limits check when monthly credits would be exceeded."""

        async def run_test():
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as f:
                tracker = UsageTracker(storage_path=f.name)
                manager = BillingManager(tracker)
                manager.current_plan = (
                    "free"  # Only 100 credits per month, 20 trials max, 100 data max
                )

                # Create usage that consumes most of the 100 credits
                # Free plan calculation: base_cost=0.1 + trial_cost=trials*0.01 + data_cost=data*0.001
                # For 20 trials (max), 100 data (max): 0.1 + 0.2 + 0.1 = 0.4 credits each
                # Let's add 250 records: 250 * 0.4 = 100 credits exactly
                for i in range(250):
                    await tracker.record_optimization(f"func{i}", 20, 100, 60.0, "free")

                # Now try to add one more optimization that would exceed the 100 credit limit
                # Any additional optimization should exceed the limit
                result = await manager.check_usage_limits(
                    trials_count=1,  # Minimal but within limits
                    dataset_size=1,  # Minimal but within limits
                )

                assert result["allowed"] is False
                assert "Monthly credit limit would be exceeded" in result["reason"]

        asyncio.run(run_test())

    def test_upgrade_plan_rejects_local_upgrade_to_paid_tier(self):
        """SDK#924: post-fix, locally upgrading from `free` to a paid
        tier (`standard`/`professional`/`enterprise`) must raise
        ConfigurationError. The SDK has no authority to grant itself
        a higher subscription tier — that has to be a backend/billing-
        portal action.

        Pre-fix this returned True and silently set
        `self.current_plan = "professional"`, granting the SDK
        professional-tier quotas (10k credits + 500 max trials)
        without any server-side validation."""
        from traigent.utils.exceptions import ConfigurationError

        tracker = UsageTracker()
        manager = BillingManager(tracker)

        for upgrade_target in ("standard", "professional", "enterprise"):
            with pytest.raises(ConfigurationError, match="SDK#924"):
                manager.upgrade_plan(upgrade_target)
            assert manager.current_plan == "free", (
                f"upgrade_plan('{upgrade_target}') must not mutate "
                f"current_plan when it raises"
            )

    def test_upgrade_plan_allows_downgrade(self):
        """SDK#924: downgrades are allowed — a user can cap themselves
        to a lower tier (uses LESS quota, never more)."""
        tracker = UsageTracker()
        manager = BillingManager(tracker)
        # Simulate a user already on professional via the backend (set
        # the field directly to skip the upgrade-block).
        manager.current_plan = "professional"

        for downgrade_target in ("standard", "free"):
            result = manager.upgrade_plan(downgrade_target)
            assert result is True
            assert manager.current_plan == downgrade_target
            manager.current_plan = "professional"

    def test_upgrade_plan_no_op_same_tier_succeeds(self):
        """SDK#924: same-tier set must succeed (idempotent reconcile)."""
        tracker = UsageTracker()
        manager = BillingManager(tracker)
        result = manager.upgrade_plan("free")
        assert result is True
        assert manager.current_plan == "free"

    def test_upgrade_plan_unknown_target_in_billing_plans_but_missing_tier_order_raises(self):
        """Greptile P1 of PR #968: a plan registered in billing_plans
        but absent from _TIER_ORDER must raise ConfigurationError.
        Pre-fix this fell back to index 0 (free) and silently allowed
        the change — re-opening the upgrade bypass for any future
        tier added without an _TIER_ORDER update."""
        from traigent.cloud.billing import BillingPlan
        from traigent.utils.exceptions import ConfigurationError

        tracker = UsageTracker()
        # Inject a fictitious tier into billing_plans WITHOUT updating
        # _TIER_ORDER (simulating the maintenance gap Greptile flagged).
        tracker.billing_plans["super_secret_unlimited"] = BillingPlan(
            name="Super Secret",
            monthly_credits=999_999_999,
            cost_per_credit=0.0,
            max_trials_per_optimization=-1,
            max_dataset_size=-1,
        )
        manager = BillingManager(tracker)

        with pytest.raises(ConfigurationError, match="_TIER_ORDER"):
            manager.upgrade_plan("super_secret_unlimited")

    def test_upgrade_plan_current_plan_not_in_tier_order_raises(self):
        """Greptile P1 of PR #968 (symmetric case): if current_plan is
        somehow set to a value missing from _TIER_ORDER (e.g., a
        legitimate plan added to billing_plans but not yet ordered),
        upgrade_plan must raise instead of misclassifying as free."""
        from traigent.utils.exceptions import ConfigurationError

        tracker = UsageTracker()
        manager = BillingManager(tracker)
        # Force current_plan to a non-ordered value (skipping the
        # upgrade-block by direct attribute assignment).
        manager.current_plan = "unknown_legacy_tier"

        with pytest.raises(ConfigurationError, match="_TIER_ORDER"):
            manager.upgrade_plan("free")

    def test_upgrade_plan_invalid(self):
        """Test plan upgrade with invalid plan."""
        tracker = UsageTracker()
        manager = BillingManager(tracker)

        result = manager.upgrade_plan("nonexistent_plan")

        assert result is False
        assert manager.current_plan == "free"  # Should remain unchanged

    def test_get_current_plan_info(self):
        """Test getting current plan information."""
        tracker = UsageTracker()
        manager = BillingManager(tracker)
        manager.current_plan = "professional"

        info = manager.get_current_plan_info()

        assert info["name"] == "Professional"
        assert info["monthly_credits"] == 10000
        assert info["cost_per_credit"] == 0.008
        assert info["priority_support"] is True
        assert info["advanced_algorithms"] is True

    def test_get_current_plan_info_invalid(self):
        """Test getting plan info for invalid plan."""
        tracker = UsageTracker()
        manager = BillingManager(tracker)
        manager.current_plan = "invalid_plan"

        info = manager.get_current_plan_info()
        assert info == {}
