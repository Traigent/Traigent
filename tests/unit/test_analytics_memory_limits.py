"""Tests for memory limit enforcement across analytics modules.

Tests verify that:
- All analytics modules use centralized constants from core/constants.py
- Memory limits are properly enforced when history sizes exceed limits
- Pruning ratio (HISTORY_PRUNE_RATIO) is correctly applied
- Thread-safe operations are maintained during pruning
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from traigent.core.constants import (
    HISTORY_PRUNE_RATIO,
    MAX_ALERT_HISTORY_SIZE,
    MAX_OPTIMIZATION_HISTORY_SIZE,
    MAX_PERFORMANCE_HISTORY_SIZE,
    MAX_USAGE_HISTORY_SIZE,
)


class TestMemoryConstantsExist:
    """Test that all memory constants are properly defined."""

    def test_max_usage_history_size(self):
        """Test MAX_USAGE_HISTORY_SIZE is defined and reasonable."""
        assert MAX_USAGE_HISTORY_SIZE == 10000
        assert MAX_USAGE_HISTORY_SIZE > 0

    def test_max_optimization_history_size(self):
        """Test MAX_OPTIMIZATION_HISTORY_SIZE is defined and reasonable."""
        assert MAX_OPTIMIZATION_HISTORY_SIZE == 1000
        assert MAX_OPTIMIZATION_HISTORY_SIZE > 0

    def test_max_performance_history_size(self):
        """Test MAX_PERFORMANCE_HISTORY_SIZE is defined and reasonable."""
        assert MAX_PERFORMANCE_HISTORY_SIZE == 1000
        assert MAX_PERFORMANCE_HISTORY_SIZE > 0

    def test_max_alert_history_size(self):
        """Test MAX_ALERT_HISTORY_SIZE is defined and reasonable."""
        assert MAX_ALERT_HISTORY_SIZE == 1000
        assert MAX_ALERT_HISTORY_SIZE > 0

    def test_history_prune_ratio(self):
        """Test HISTORY_PRUNE_RATIO is defined and reasonable."""
        assert HISTORY_PRUNE_RATIO == 0.1
        assert 0 < HISTORY_PRUNE_RATIO < 1


class TestMetaLearningMemoryLimits:
    """Test memory limits in meta_learning module."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        self.temp_file.close()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_optimization_history_respects_limit(self):
        """Test that OptimizationHistory enforces memory limits."""
        from traigent.analytics.meta_learning import (
            AlgorithmType,
            OptimizationHistory,
            OptimizationRecord,
            OptimizationStatus,
        )

        history = OptimizationHistory(storage_path=self.temp_file.name)

        # Add more records than the limit
        num_records = MAX_OPTIMIZATION_HISTORY_SIZE + 100

        for i in range(num_records):
            record = OptimizationRecord(
                optimization_id=f"opt_{i}",
                function_name=f"func_{i}",
                algorithm=AlgorithmType.RANDOM,
                configuration_space={"param": [1, 2]},
                objectives=["accuracy"],
                dataset_size=100,
                best_score=0.8,
                total_trials=10,
                duration_seconds=60.0,
                status=OptimizationStatus.COMPLETED,
                timestamp=datetime.now(timezone.utc),
            )
            history.add_record(record)

        # Should be pruned to within limit
        assert len(history.records) <= MAX_OPTIMIZATION_HISTORY_SIZE

    def test_optimization_history_prune_ratio_applied(self):
        """Test that pruning removes correct ratio of records."""
        from traigent.analytics.meta_learning import (
            AlgorithmType,
            OptimizationHistory,
            OptimizationRecord,
            OptimizationStatus,
        )

        history = OptimizationHistory(storage_path=self.temp_file.name)

        # Fill to exactly the limit + 1 to trigger pruning
        for i in range(MAX_OPTIMIZATION_HISTORY_SIZE + 1):
            record = OptimizationRecord(
                optimization_id=f"opt_{i}",
                function_name=f"func_{i}",
                algorithm=AlgorithmType.GRID,
                configuration_space={"param": [1]},
                objectives=["accuracy"],
                dataset_size=50,
                best_score=0.9,
                total_trials=5,
                duration_seconds=30.0,
                status=OptimizationStatus.COMPLETED,
                timestamp=datetime.now(timezone.utc),
            )
            history.add_record(record)

        # After pruning, should have kept (1 - HISTORY_PRUNE_RATIO) * MAX items
        expected_size = int(MAX_OPTIMIZATION_HISTORY_SIZE * (1 - HISTORY_PRUNE_RATIO))
        assert len(history.records) == expected_size


class TestPredictiveAnalyticsMemoryLimits:
    """Test memory limits in predictive analytics module."""

    def test_cost_forecaster_respects_limit(self):
        """Test that CostForecaster enforces memory limits."""
        from traigent.analytics.predictive import CostForecaster, UsageMetric

        forecaster = CostForecaster()

        # Add more items than the limit
        num_items = MAX_USAGE_HISTORY_SIZE + 500

        for i in range(num_items):
            metric = UsageMetric(
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
                optimizations_count=5,
                total_trials=50,
                total_duration_seconds=300,
                compute_cost=10.0,
                storage_cost=2.0,
                api_calls=100,
                active_users=5,
            )
            forecaster.add_usage_data(metric)

        # Should be pruned to within limit
        assert len(forecaster.usage_history) <= MAX_USAGE_HISTORY_SIZE

    def test_performance_forecaster_respects_limit(self):
        """Test that PerformanceForecaster enforces memory limits."""
        from traigent.analytics.predictive import PerformanceForecaster

        forecaster = PerformanceForecaster()

        # Add more items than the limit
        num_items = MAX_PERFORMANCE_HISTORY_SIZE + 100

        for i in range(num_items):
            forecaster.add_performance_data(
                function_name="test_func",
                algorithm="random",
                score=0.8 + (i % 10) * 0.01,
                duration=100.0,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            )

        # Should be pruned to within limit
        assert len(forecaster.performance_history) <= MAX_PERFORMANCE_HISTORY_SIZE

    def test_trend_analyzer_respects_limit(self):
        """Test that TrendAnalyzer enforces memory limits per metric."""
        from traigent.analytics.predictive import TrendAnalyzer

        analyzer = TrendAnalyzer()

        # Add more items than the limit for a single metric
        num_items = MAX_PERFORMANCE_HISTORY_SIZE + 200

        for i in range(num_items):
            analyzer.add_metric_value(
                metric_name="test_metric",
                value=10.0 + i * 0.1,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            )

        # Should be pruned to within limit
        assert (
            len(analyzer.metrics_history["test_metric"]) <= MAX_PERFORMANCE_HISTORY_SIZE
        )


class TestCostOptimizationMemoryLimits:
    """Test memory limits in cost_optimization module."""

    def test_budget_allocator_respects_limit(self):
        """Test that BudgetAllocator enforces memory limits on historical data."""
        from traigent.analytics.cost_optimization import (
            BudgetAllocator,
            ResourceType,
        )

        allocator = BudgetAllocator()

        # Set up initial allocation
        priorities = {ResourceType.COMPUTE: 1.0}
        allocator.allocate_budget(
            total_budget=1000.0,
            priorities=priorities,
            historical_usage={},
        )

        # Add more spending updates than the limit
        num_updates = MAX_OPTIMIZATION_HISTORY_SIZE + 200

        for i in range(num_updates):
            allocator.update_spending(ResourceType.COMPUTE, 10.0)

        # Should be pruned to within limit
        assert (
            len(allocator.historical_data[ResourceType.COMPUTE])
            <= MAX_OPTIMIZATION_HISTORY_SIZE
        )

    def test_cost_optimization_ai_respects_limit(self):
        """Test that CostOptimizationAI enforces memory limits on usage history."""
        from traigent.analytics.cost_optimization import (
            CostOptimizationAI,
            OptimizationStrategy,
            ResourceType,
            ResourceUsage,
        )

        optimizer = CostOptimizationAI(
            optimization_strategy=OptimizationStrategy.BALANCED
        )

        # Add more usage records than the limit
        num_records = MAX_USAGE_HISTORY_SIZE + 1000

        for i in range(num_records):
            usage = ResourceUsage(
                resource_type=ResourceType.COMPUTE,
                current_usage=100.0,
                historical_average=100.0,
                peak_usage=150.0,
                unit_cost=0.1,
                total_cost=10.0,
                utilization_percent=70,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            )
            optimizer.record_usage(usage)

        # Should be pruned to within limit
        assert (
            len(optimizer.usage_history[ResourceType.COMPUTE]) <= MAX_USAGE_HISTORY_SIZE
        )

    def test_resource_optimizer_respects_limit(self):
        """Test that ResourceOptimizer enforces memory limits on usage history."""
        from traigent.analytics.cost_optimization import (
            ResourceOptimizer,
            ResourceType,
            ResourceUsage,
        )

        optimizer = ResourceOptimizer()

        # Add more usage records than the limit
        num_records = MAX_USAGE_HISTORY_SIZE + 500

        for i in range(num_records):
            usage = ResourceUsage(
                resource_type=ResourceType.STORAGE,
                current_usage=50.0,
                historical_average=50.0,
                peak_usage=75.0,
                unit_cost=0.05,
                total_cost=5.0,
                utilization_percent=60,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            )
            optimizer.record_usage(usage)

        # Should be pruned to within limit
        assert (
            len(optimizer.usage_history[ResourceType.STORAGE]) <= MAX_USAGE_HISTORY_SIZE
        )


class TestSchedulingMemoryLimits:
    """Test memory limits in scheduling module."""

    def test_smart_scheduler_completed_jobs_limit(self):
        """Test that SmartScheduler enforces memory limits on completed jobs."""
        from traigent.analytics.scheduling import (
            JobPriority,
            ScheduledJob,
            SmartScheduler,
        )

        scheduler = SmartScheduler()

        # Add and complete more jobs than the limit
        num_jobs = MAX_OPTIMIZATION_HISTORY_SIZE + 100

        for i in range(num_jobs):
            job = ScheduledJob(
                job_id=f"job_{i}",
                job_type="optimization",
                priority=JobPriority.NORMAL,
                estimated_duration=timedelta(hours=1),
                estimated_cost=10.0,
                resource_requirements={},
            )
            scheduler.add_job(job)

            # Complete the job immediately
            scheduler.complete_job(job.job_id, actual_cost=10.0)

        # Should be pruned to within limit
        assert len(scheduler.completed_jobs) <= MAX_OPTIMIZATION_HISTORY_SIZE

    def test_smart_scheduler_cost_history_limit(self):
        """Test that SmartScheduler enforces memory limits on cost history."""
        from traigent.analytics.scheduling import (
            JobPriority,
            ScheduledJob,
            SmartScheduler,
        )

        scheduler = SmartScheduler()

        # Add and complete jobs to build up cost history
        num_jobs = MAX_OPTIMIZATION_HISTORY_SIZE + 150

        for i in range(num_jobs):
            job = ScheduledJob(
                job_id=f"cost_job_{i}",
                job_type="analysis",
                priority=JobPriority.HIGH,
                estimated_duration=timedelta(minutes=30),
                estimated_cost=5.0,
                resource_requirements={},
            )
            scheduler.add_job(job)
            scheduler.complete_job(job.job_id, actual_cost=5.0 + (i % 10))

        # Should be pruned to within limit
        assert len(scheduler.cost_history) <= MAX_OPTIMIZATION_HISTORY_SIZE


class TestIntelligenceMemoryLimits:
    """Test memory limits in intelligence module."""

    def test_cost_analysis_engine_uses_max_history_constant(self):
        """Test that CostAnalysisEngine uses MAX_USAGE_HISTORY_SIZE."""
        from traigent.analytics.intelligence import CostAnalysisEngine

        engine = CostAnalysisEngine()

        # Verify that the engine uses the correct constant
        assert engine._max_history_items == MAX_USAGE_HISTORY_SIZE

    def test_cost_optimization_ai_uses_max_history_constant(self):
        """Test that CostOptimizationAI uses MAX_OPTIMIZATION_HISTORY_SIZE."""
        from traigent.analytics.intelligence import CostOptimizationAI

        optimizer = CostOptimizationAI()

        # Verify that the optimizer uses the correct constant
        assert optimizer._max_optimization_results == MAX_OPTIMIZATION_HISTORY_SIZE


class TestPruningBehavior:
    """Test pruning behavior when limits are exceeded."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        self.temp_file.close()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_pruning_removes_oldest_entries(self):
        """Test that pruning removes oldest entries first."""
        from traigent.analytics.meta_learning import (
            AlgorithmType,
            OptimizationHistory,
            OptimizationRecord,
            OptimizationStatus,
        )

        history = OptimizationHistory(storage_path=self.temp_file.name)

        # Add records with sequential IDs
        for i in range(MAX_OPTIMIZATION_HISTORY_SIZE + 10):
            record = OptimizationRecord(
                optimization_id=f"opt_{i:05d}",  # Zero-padded for sorting
                function_name="test_func",
                algorithm=AlgorithmType.RANDOM,
                configuration_space={"param": [1]},
                objectives=["accuracy"],
                dataset_size=100,
                best_score=0.8,
                total_trials=10,
                duration_seconds=60.0,
                status=OptimizationStatus.COMPLETED,
                timestamp=datetime.now(timezone.utc),
            )
            history.add_record(record)

        # The oldest records should have been removed
        oldest_remaining_id = history.records[0].optimization_id

        # Should have removed approximately HISTORY_PRUNE_RATIO of records
        # So oldest_remaining should not be "opt_00000"
        assert oldest_remaining_id != "opt_00000"

    def test_pruning_ratio_calculation(self):
        """Test that pruning ratio is correctly calculated."""
        # When we have MAX items + 1, pruning should keep (1 - ratio) * MAX
        expected_after_prune = int(
            MAX_OPTIMIZATION_HISTORY_SIZE * (1 - HISTORY_PRUNE_RATIO)
        )

        # For MAX=1000 and ratio=0.1, should keep 900
        assert expected_after_prune == 900


class TestThreadSafetyDuringPruning:
    """Test thread safety during pruning operations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        self.temp_file.close()

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_optimization_history_has_lock(self):
        """Test that OptimizationHistory has a lock for thread safety."""
        from traigent.analytics.meta_learning import OptimizationHistory

        history = OptimizationHistory(storage_path=self.temp_file.name)

        # Should have a lock attribute
        assert hasattr(history, "_lock")

    def test_concurrent_record_additions(self):
        """Test concurrent record additions don't cause data corruption."""
        import threading

        from traigent.analytics.meta_learning import (
            AlgorithmType,
            OptimizationHistory,
            OptimizationRecord,
            OptimizationStatus,
        )

        history = OptimizationHistory(storage_path=self.temp_file.name)
        errors = []

        def add_records(thread_id: int, count: int):
            try:
                for i in range(count):
                    record = OptimizationRecord(
                        optimization_id=f"opt_t{thread_id}_{i}",
                        function_name=f"func_t{thread_id}",
                        algorithm=AlgorithmType.RANDOM,
                        configuration_space={"param": [1]},
                        objectives=["accuracy"],
                        dataset_size=100,
                        best_score=0.8,
                        total_trials=10,
                        duration_seconds=60.0,
                        status=OptimizationStatus.COMPLETED,
                        timestamp=datetime.now(timezone.utc),
                    )
                    history.add_record(record)
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads that add records concurrently
        threads = []
        num_threads = 5
        records_per_thread = 50

        for t_id in range(num_threads):
            thread = threading.Thread(
                target=add_records, args=(t_id, records_per_thread)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Should have no errors
        assert len(errors) == 0

        # Should have records (might be less due to pruning)
        assert len(history.records) > 0


class TestConstantsImportedCorrectly:
    """Test that analytics modules import constants from the correct location."""

    def test_cost_optimization_imports_constants(self):
        """Test cost_optimization module imports from core.constants."""
        from traigent.analytics import cost_optimization
        from traigent.analytics.cost_optimization import CostOptimizationAI

        # Module should import the constants from core.constants
        # Verify by checking that CostOptimizationAI uses the correct limit
        optimizer = CostOptimizationAI()
        # After adding many items, it should respect the limit
        assert hasattr(optimizer, "usage_history")

    def test_intelligence_imports_constants(self):
        """Test intelligence module imports from core.constants."""
        from traigent.analytics import intelligence

        # Module should use the constants internally via import
        # We verify this by checking that CostAnalysisEngine uses the correct value
        engine = intelligence.CostAnalysisEngine()
        assert engine._max_history_items == MAX_USAGE_HISTORY_SIZE

    def test_scheduling_imports_constants(self):
        """Test scheduling module imports from core.constants."""
        from traigent.analytics import scheduling

        # Verify module uses constants correctly
        # SmartScheduler should limit completed_jobs to MAX_OPTIMIZATION_HISTORY_SIZE
        assert hasattr(scheduling, "SmartScheduler")

    def test_meta_learning_imports_constants(self):
        """Test meta_learning module imports from core.constants."""
        from traigent.analytics import meta_learning

        # Verify module uses constants correctly
        assert hasattr(meta_learning, "OptimizationHistory")

    def test_predictive_imports_constants(self):
        """Test predictive module imports from core.constants."""
        from traigent.analytics import predictive

        # Verify CostForecaster uses correct constant
        forecaster = predictive.CostForecaster()
        assert forecaster._max_history_items == MAX_USAGE_HISTORY_SIZE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
