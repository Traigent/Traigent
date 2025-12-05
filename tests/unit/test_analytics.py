"""Tests for analytics modules in Sprint 8."""

import os
import tempfile
from datetime import datetime, timedelta, timezone

import pytest

from traigent.analytics.anomaly import (
    AlertManager,
    AlertSeverity,
    AnomalyDetector,
    AnomalyEvent,
    AnomalyType,
    PerformanceMonitor,
    StatisticalDetector,
    ThresholdDetector,
)
from traigent.analytics.cost_optimization import (
    BudgetAllocator,
    CostOptimizationAction,
    CostOptimizationAI,
    OptimizationStrategy,
    Priority,
    ResourceOptimizer,
    ResourceType,
    ResourceUsage,
)
from traigent.analytics.meta_learning import (
    AlgorithmSelector,
    AlgorithmType,
    MetaLearningEngine,
    OptimizationHistory,
    OptimizationRecord,
    OptimizationStatus,
    PerformancePredictor,
    ProblemCharacteristics,
)
from traigent.analytics.predictive import (
    CostForecaster,
    ForecastPeriod,
    ForecastResult,
    PerformanceForecaster,
    PredictiveAnalytics,
    TrendAnalyzer,
    UsageMetric,
)
from traigent.analytics.scheduling import (
    JobPriority,
    ScheduledJob,
    ScheduleWindow,
    SmartScheduler,
)


class TestMetaLearningEngine:
    """Test meta-learning engine functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        self.temp_file.close()
        self.engine = MetaLearningEngine(storage_path=self.temp_file.name)

    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_record_optimization(self):
        """Test recording optimization results."""
        self.engine.record_optimization(
            optimization_id="test_opt_1",
            function_name="test_function",
            algorithm="grid",
            configuration_space={"param1": [1, 2, 3], "param2": (0.1, 1.0)},
            objectives=["accuracy"],
            dataset_size=100,
            best_score=0.85,
            total_trials=10,
            duration_seconds=120.5,
        )

        assert len(self.engine.history.records) == 1
        record = self.engine.history.records[0]
        assert record.function_name == "test_function"
        assert record.best_score == 0.85
        assert record.algorithm == AlgorithmType.GRID

    def test_get_optimization_recommendations(self):
        """Test getting optimization recommendations."""
        # Add some historical data
        for i in range(5):
            self.engine.record_optimization(
                optimization_id=f"opt_{i}",
                function_name="test_function",
                algorithm="random",
                configuration_space={"param1": [1, 2, 3]},
                objectives=["accuracy"],
                dataset_size=100,
                best_score=0.8 + i * 0.02,
                total_trials=20,
                duration_seconds=100 + i * 10,
            )

        recommendations = self.engine.get_optimization_recommendations(
            function_name="test_function",
            configuration_space={"param1": [1, 2, 3]},
            objectives=["accuracy"],
        )

        assert "recommended_algorithm" in recommendations
        assert "recommendation_confidence" in recommendations
        assert "algorithm_rankings" in recommendations
        assert "performance_prediction" in recommendations

    def test_get_insights(self):
        """Test getting optimization insights."""
        # Add test data
        self.engine.record_optimization(
            optimization_id="test_opt",
            function_name="test_function",
            algorithm="grid",
            configuration_space={"param1": [1, 2]},
            objectives=["accuracy"],
            dataset_size=50,
            best_score=0.9,
            total_trials=5,
            duration_seconds=60,
        )

        insights = self.engine.get_insights()

        assert "total_optimizations" in insights
        assert "algorithm_performance" in insights
        assert insights["total_optimizations"] == 1

    def test_algorithm_selector(self):
        """Test algorithm selection functionality."""
        history = OptimizationHistory(storage_path=self.temp_file.name)
        selector = AlgorithmSelector(history)

        # Add some test data
        record = OptimizationRecord(
            optimization_id="test_1",
            function_name="test_func",
            algorithm=AlgorithmType.GRID,
            configuration_space={"param1": [1, 2, 3]},
            objectives=["accuracy"],
            dataset_size=100,
            best_score=0.85,
            total_trials=10,
            duration_seconds=120,
            status=OptimizationStatus.COMPLETED,
            timestamp=datetime.utcnow(),
        )
        history.add_record(record)

        algorithm, confidence = selector.recommend_algorithm(
            function_name="test_func",
            configuration_space={"param1": [1, 2, 3]},
            objectives=["accuracy"],
        )

        assert isinstance(algorithm, AlgorithmType)
        assert 0 <= confidence <= 1

    def test_performance_predictor(self):
        """Test performance prediction functionality."""
        history = OptimizationHistory(storage_path=self.temp_file.name)
        predictor = PerformancePredictor(history)

        # Add test data
        record = OptimizationRecord(
            optimization_id="test_1",
            function_name="test_func",
            algorithm=AlgorithmType.RANDOM,
            configuration_space={"param1": [1, 2, 3]},
            objectives=["accuracy"],
            dataset_size=100,
            best_score=0.8,
            total_trials=20,
            duration_seconds=200,
            status=OptimizationStatus.COMPLETED,
            timestamp=datetime.utcnow(),
        )
        history.add_record(record)

        prediction = predictor.predict_optimization_outcome(
            function_name="test_func",
            algorithm=AlgorithmType.RANDOM,
            configuration_space={"param1": [1, 2, 3]},
            max_trials=30,
        )

        assert "predicted_best_score" in prediction
        assert "predicted_duration_seconds" in prediction
        assert "success_probability" in prediction

    def test_convergence_curve_with_zero_trials(self):
        """Convergence curve estimation should handle zero-trial history."""
        history = OptimizationHistory(storage_path=self.temp_file.name)
        predictor = PerformancePredictor(history)

        record = OptimizationRecord(
            optimization_id="test_zero_trials",
            function_name="test_func_zero",
            algorithm=AlgorithmType.GRID,
            configuration_space={"param": [1]},
            objectives=["accuracy"],
            dataset_size=50,
            best_score=0.75,
            total_trials=0,
            duration_seconds=0,
            status=OptimizationStatus.COMPLETED,
            timestamp=datetime.utcnow(),
        )
        history.add_record(record)

        curve = predictor.estimate_convergence_curve(
            function_name="test_func_zero",
            algorithm=AlgorithmType.GRID,
            max_trials=5,
        )

        assert len(curve) == 5
        assert all(isinstance(point, tuple) and len(point) == 2 for point in curve)

    def test_problem_characteristics_similarity(self):
        """Test problem characteristics similarity calculation."""
        char1 = ProblemCharacteristics(
            config_space_size=5,
            param_types={"param1": "categorical", "param2": "continuous"},
            num_objectives=1,
            problem_complexity=0.5,
            estimated_evaluation_time=1.0,
        )

        char2 = ProblemCharacteristics(
            config_space_size=4,
            param_types={"param1": "categorical", "param2": "continuous"},
            num_objectives=1,
            problem_complexity=0.6,
            estimated_evaluation_time=1.2,
        )

        similarity = char1.similarity(char2)
        assert 0 <= similarity <= 1
        assert similarity > 0.5  # Should be reasonably similar


class TestPredictiveAnalytics:
    """Test predictive analytics functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analytics = PredictiveAnalytics()

    def test_cost_forecaster(self):
        """Test cost forecasting functionality."""
        forecaster = CostForecaster()

        # Add sample usage data
        for i in range(10):
            metric = UsageMetric(
                timestamp=datetime.utcnow() - timedelta(days=i),
                optimizations_count=5 + i,
                total_trials=50 + i * 5,
                total_duration_seconds=300 + i * 20,
                compute_cost=10.0 + i * 0.5,
                storage_cost=2.0 + i * 0.1,
                api_calls=100 + i * 10,
                active_users=5 + i,
            )
            forecaster.add_usage_data(metric)

        # Use latest API: provide historical data list
        historical_costs = [
            {"timestamp": m.timestamp, "cost": m.compute_cost + m.storage_cost}
            for m in forecaster.usage_history
        ]
        forecast = forecaster.forecast_costs(
            historical_costs, forecast_period=ForecastPeriod.MONTHLY
        )

        assert isinstance(forecast, ForecastResult)
        assert len(forecast.predicted_values) == 30

    def test_performance_forecaster(self):
        """Test performance forecasting functionality."""
        forecaster = PerformanceForecaster()

        # Add sample performance data
        for i in range(10):
            forecaster.add_performance_data(
                function_name="test_function",
                algorithm="random",
                score=0.8 + i * 0.01,
                duration=100 + i * 5,
                timestamp=datetime.now(timezone.utc) - timedelta(days=i),
            )

        forecast = forecaster.forecast_performance_trends(
            function_name="test_function", algorithm="random", forecast_days=30
        )

        assert "score_forecast" in forecast
        assert "duration_forecast" in forecast
        assert "performance_insights" in forecast

    def test_trend_analyzer(self):
        """Test trend analysis functionality."""
        analyzer = TrendAnalyzer()

        # Add sample data with an upward trend
        for i in range(15):
            analyzer.add_metric_value(
                metric_name="optimization_count",
                value=10 + i * 0.5,
                timestamp=datetime.utcnow() - timedelta(days=14 - i),
            )

        analysis = analyzer.analyze_trends("optimization_count", period_days=14)

        assert "trend" in analysis
        assert "statistics" in analysis
        assert "insights" in analysis
        assert analysis["trend"]["direction"] in [
            "increasing",
            "decreasing",
            "stable",
            "volatile",
        ]

    def test_degradation_detection(self):
        """Test performance degradation detection."""
        forecaster = PerformanceForecaster()

        # Add data showing degradation
        for i in range(20):
            score = 0.9 - (i * 0.02) if i > 10 else 0.9  # Degradation after day 10
            forecaster.add_performance_data(
                function_name="test_function",
                algorithm="random",
                score=score,
                duration=100,
                timestamp=datetime.now(timezone.utc) - timedelta(days=19 - i),
            )

        degradation = forecaster.detect_performance_degradation(
            function_name="test_function", lookback_days=10
        )

        assert "degradation_detected" in degradation
        if degradation["degradation_detected"]:
            assert "issues" in degradation
            assert "recommendation" in degradation


class TestAnomalyDetector:
    """Test anomaly detection functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.detector = AnomalyDetector()

    def test_statistical_detector(self):
        """Test statistical anomaly detection."""
        detector = StatisticalDetector()

        # Add normal data
        for i in range(50):
            detector.add_data_point("test_metric", 10.0 + i * 0.1)

        # Test normal value
        normal_anomalies = detector.detect_anomalies("test_metric", 12.0)
        assert len(normal_anomalies) == 0

        # Test anomalous value
        anomalous_value = 25.0  # Much higher than normal
        anomalies = detector.detect_anomalies("test_metric", anomalous_value)
        assert len(anomalies) > 0
        assert anomalies[0].anomaly_type in [
            AnomalyType.PATTERN_DEVIATION,
            AnomalyType.THRESHOLD_VIOLATION,
        ]

    def test_threshold_detector(self):
        """Test threshold-based anomaly detection."""
        detector = ThresholdDetector()

        # Set thresholds
        detector.set_threshold("test_metric", min_threshold=5.0, max_threshold=15.0)

        # Test values within threshold
        normal_anomalies = detector.detect_anomalies("test_metric", 10.0)
        assert len(normal_anomalies) == 0

        # Test values outside threshold
        high_anomalies = detector.detect_anomalies("test_metric", 20.0)
        assert len(high_anomalies) == 1
        assert high_anomalies[0].anomaly_type == AnomalyType.THRESHOLD_VIOLATION

        low_anomalies = detector.detect_anomalies("test_metric", 2.0)
        assert len(low_anomalies) == 1
        assert low_anomalies[0].anomaly_type == AnomalyType.THRESHOLD_VIOLATION

    def test_threshold_detector_handles_zero_thresholds(self):
        """Zero thresholds should not cause divide-by-zero errors."""
        detector = ThresholdDetector()
        detector.set_threshold("zero_metric", min_threshold=0.0, max_threshold=0.0)

        below_zero = detector.detect_anomalies("zero_metric", -1.0)
        assert below_zero
        assert below_zero[0].deviation_score == pytest.approx(1.0)

        above_zero = detector.detect_anomalies("zero_metric", 2.0)
        assert above_zero
        assert above_zero[0].deviation_score == pytest.approx(2.0)

    def test_monitor_metric_uses_timezone_aware_defaults(self):
        """Alerts generated with default timestamps remain timezone-aware."""
        detector = AnomalyDetector()
        detector.threshold_detector.set_threshold("metric_default", max_threshold=0.0)

        anomalies = detector.monitor_metric("metric_default", 1.0)
        assert anomalies

        summary = detector.alert_manager.get_alert_summary(hours=1)
        assert summary["total_alerts"] >= 1
        assert all(
            alert.timestamp.tzinfo is not None
            for alert in detector.alert_manager.alert_history
        )

    def test_monitor_optimization_default_timestamp(self):
        """Optimization monitoring stores UTC-aware timestamps without input."""
        detector = AnomalyDetector()
        detector.threshold_detector.set_threshold(
            "function:algo:score", max_threshold=0.0
        )

        detector.monitor_optimization(
            optimization_id="opt-1",
            function_name="function",
            algorithm="algo",
            score=1.0,
            duration=5.0,
        )

        alerts = detector.alert_manager.get_recent_alerts(hours=1)
        assert alerts
        assert all(alert.timestamp.tzinfo is not None for alert in alerts)

        history_key = "function:algo"
        stored_timestamp = detector.regression_detector.metric_histories[history_key][
            0
        ][0]
        assert stored_timestamp.tzinfo is not None

    def test_performance_monitor(self):
        """Test performance monitoring and regression detection."""
        monitor = PerformanceMonitor()

        # Record normal performance
        for i in range(20):
            monitor.record_performance(
                function_name="test_function",
                algorithm="grid",
                score=0.85 + i * 0.001,  # Slight improvement
                duration=100 - i * 0.5,  # Slight speedup
                timestamp=datetime.now(timezone.utc) - timedelta(hours=20 - i),
            )

        # Record degraded performance
        for i in range(5):
            monitor.record_performance(
                function_name="test_function",
                algorithm="grid",
                score=0.7 - i * 0.02,  # Significant degradation
                duration=150 + i * 10,  # Significant slowdown
                timestamp=datetime.now(timezone.utc) - timedelta(hours=5 - i),
            )

        regressions = monitor.detect_performance_regression(
            function_name="test_function", algorithm="grid", lookback_hours=6
        )

        assert len(regressions) > 0
        assert any(
            r.anomaly_type == AnomalyType.PERFORMANCE_DEGRADATION for r in regressions
        )

    def test_alert_manager(self):
        """Test alert management functionality."""
        alert_manager = AlertManager()

        # Create test anomaly
        anomaly = AnomalyEvent(
            anomaly_id="test_anomaly_1",
            anomaly_type=AnomalyType.COST_SPIKE,
            severity=AlertSeverity.HIGH,
            metric_name="compute_cost",
            actual_value=100.0,
            expected_value=50.0,
            deviation_score=2.0,
            timestamp=datetime.utcnow(),
            description="Test cost spike",
        )

        # Process anomaly
        processed = alert_manager.process_anomaly(anomaly)
        assert processed

        # Get recent alerts
        recent_alerts = alert_manager.get_recent_alerts(hours=1)
        assert len(recent_alerts) == 1
        assert recent_alerts[0].anomaly_id == "test_anomaly_1"

        # Get alert summary
        summary = alert_manager.get_alert_summary(hours=1)
        assert summary["total_alerts"] == 1
        assert "severity_breakdown" in summary

    def test_system_health(self):
        """Test system health monitoring."""
        health = self.detector.get_system_health()

        assert "health_status" in health
        assert "recent_alerts_1h" in health
        assert "monitoring_active" in health
        assert health["monitoring_active"]

    def test_monitor_optimization(self):
        """Test monitoring optimization performance."""
        anomalies = self.detector.monitor_optimization(
            optimization_id="test_opt_1",
            function_name="test_function",
            algorithm="grid",
            score=0.85,
            duration=120.0,
            cost=10.0,
            additional_metrics={"memory_usage": 512.0},
        )

        # Should not detect anomalies for first optimization
        assert isinstance(anomalies, list)


class TestCostOptimizationAI:
    """Test AI-powered cost optimization functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = CostOptimizationAI(
            optimization_strategy=OptimizationStrategy.BALANCED
        )

    def test_analyze_cost_patterns(self):
        """Test cost pattern analysis."""
        # Add sample usage data
        for resource_type in [ResourceType.COMPUTE, ResourceType.STORAGE]:
            usage_list = []
            for i in range(30):
                usage = ResourceUsage(
                    resource_type=resource_type,
                    current_usage=100.0 + i * 2,
                    historical_average=100.0,
                    peak_usage=150.0,
                    unit_cost=0.1,
                    total_cost=10.0 + i * 0.2,
                    utilization_percent=60 + i,
                    timestamp=datetime.now(timezone.utc) - timedelta(days=30 - i),
                )
                usage_list.append(usage)

            self.optimizer.usage_history[resource_type] = usage_list

        analysis = self.optimizer.analyze_cost_patterns(days_back=30)

        assert "resource_analysis" in analysis
        assert "cost_trends" in analysis
        assert "optimization_opportunities" in analysis
        assert len(analysis["optimization_opportunities"]) >= 0

    def test_generate_optimization_recommendations(self):
        """Test generating optimization recommendations."""
        # Add sample usage data for compute
        usage_list = []
        for i in range(20):
            usage = ResourceUsage(
                resource_type=ResourceType.COMPUTE,
                current_usage=50.0,  # Low utilization
                historical_average=50.0,
                peak_usage=80.0,
                unit_cost=0.1,
                total_cost=15.0,
                utilization_percent=40,  # Low utilization for rightsizing opportunity
                timestamp=datetime.utcnow() - timedelta(days=i),
            )
            usage_list.append(usage)

        self.optimizer.usage_history[ResourceType.COMPUTE] = usage_list

        recommendations = self.optimizer.generate_optimization_recommendations()

        assert isinstance(recommendations, list)
        for rec in recommendations:
            assert isinstance(rec, CostOptimizationAction)
            assert rec.resource_type in ResourceType
            assert rec.priority in Priority
            assert 0 <= rec.estimated_savings_percent <= 100

    def test_simulate_optimization_impact(self):
        """Test simulation of optimization impact."""
        # Create sample actions
        actions = [
            CostOptimizationAction(
                action_id="test_action_1",
                action_type="rightsizing",
                description="Rightsize compute resources",
                resource_type=ResourceType.COMPUTE,
                estimated_savings_percent=20,
                estimated_savings_absolute=100.0,
                implementation_effort="medium",
                risk_level="low",
                priority=Priority.HIGH,
                detailed_steps=["Step 1", "Step 2"],
                impact_analysis={},
                timeline_days=14,
            )
        ]

        # Add some baseline data
        usage_list = [
            ResourceUsage(
                resource_type=ResourceType.COMPUTE,
                current_usage=100.0,
                historical_average=100.0,
                peak_usage=150.0,
                unit_cost=0.1,
                total_cost=50.0,
                utilization_percent=60,
                timestamp=datetime.utcnow(),
            )
        ]
        self.optimizer.usage_history[ResourceType.COMPUTE] = usage_list

        simulation = self.optimizer.simulate_optimization_impact(
            actions, simulation_days=90
        )

        assert "total_estimated_savings" in simulation
        assert "cost_reduction_percent" in simulation
        assert "implementation_timeline" in simulation
        assert "confidence_score" in simulation

    def test_predict_future_costs(self):
        """Test future cost prediction."""
        # Add usage history
        usage_list = []
        for i in range(30):
            usage = ResourceUsage(
                resource_type=ResourceType.COMPUTE,
                current_usage=100.0,
                historical_average=100.0,
                peak_usage=150.0,
                unit_cost=0.1,
                total_cost=10.0 + i * 0.1,  # Slight upward trend
                utilization_percent=70,
                timestamp=datetime.now(timezone.utc) - timedelta(days=30 - i),
            )
            usage_list.append(usage)

        self.optimizer.usage_history[ResourceType.COMPUTE] = usage_list

        predictions = self.optimizer.predict_future_costs(
            forecast_days=90,
            growth_scenarios={"conservative": 0.05, "aggressive": 0.20},
        )

        assert "scenarios" in predictions
        assert "conservative" in predictions["scenarios"]
        assert "aggressive" in predictions["scenarios"]
        assert "key_assumptions" in predictions

    def test_track_optimization_results(self):
        """Test tracking optimization results for learning."""
        # First generate a recommendation
        action = CostOptimizationAction(
            action_id="test_tracking_1",
            action_type="rightsizing",
            description="Test action",
            resource_type=ResourceType.COMPUTE,
            estimated_savings_percent=20,
            estimated_savings_absolute=100.0,
            implementation_effort="medium",
            risk_level="low",
            priority=Priority.HIGH,
            detailed_steps=[],
            impact_analysis={},
            timeline_days=14,
        )

        self.optimizer.optimization_actions = [action]

        # Track results
        self.optimizer.track_optimization_results(
            action_id="test_tracking_1",
            actual_savings=90.0,  # Slightly less than predicted
            implementation_cost=200.0,
            performance_impact={"latency_increase_ms": 5},
        )

        assert "optimization_results" in self.optimizer.learning_data
        assert len(self.optimizer.learning_data["optimization_results"]) == 1

        result = self.optimizer.learning_data["optimization_results"][0]
        assert result["actual_savings"] == 90.0
        assert result["predicted_savings"] == 100.0


class TestBudgetAllocator:
    """Test budget allocation functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.allocator = BudgetAllocator()

    def test_allocate_budget(self):
        """Test budget allocation across resource types."""
        total_budget = 1000.0

        # Note: allocate_budget expects priorities and historical_usage
        priorities = {
            ResourceType.COMPUTE: 0.5,
            ResourceType.STORAGE: 0.3,
            ResourceType.NETWORK: 0.2,
        }
        historical_usage = {
            ResourceType.COMPUTE: [400.0, 450.0, 420.0],
            ResourceType.STORAGE: [200.0, 180.0, 190.0],
            ResourceType.NETWORK: [100.0, 120.0, 110.0],
        }

        allocations = self.allocator.allocate_budget(
            total_budget=total_budget,
            priorities=priorities,
            historical_usage=historical_usage,
        )

        assert len(allocations) > 0

        # Check that allocations sum to approximately total budget
        total_allocated = sum(alloc.allocated_budget for alloc in allocations.values())
        assert (
            abs(total_allocated - total_budget) < 1.0
        )  # Allow small rounding differences

        # Check allocation structure
        for resource_type, allocation in allocations.items():
            assert isinstance(resource_type, ResourceType)
            assert allocation.allocated_budget > 0
            assert allocation.remaining_budget == allocation.allocated_budget
            assert allocation.spent_amount == 0

    def test_allocate_budget_with_constraints(self):
        """Test budget allocation with constraints."""
        total_budget = 1000.0

        priorities = {
            ResourceType.COMPUTE: 0.5,
            ResourceType.STORAGE: 0.3,
            ResourceType.NETWORK: 0.2,
        }
        historical_usage = {
            ResourceType.COMPUTE: [400.0, 450.0, 420.0],
            ResourceType.STORAGE: [200.0, 180.0, 190.0],
            ResourceType.NETWORK: [100.0, 120.0, 110.0],
        }
        constraints = {
            ResourceType.COMPUTE: (400.0, float("inf")),  # min 400
            ResourceType.STORAGE: (0.0, 100.0),  # max 100
        }

        allocations = self.allocator.allocate_budget(
            total_budget=total_budget,
            priorities=priorities,
            historical_usage=historical_usage,
            constraints=constraints,
        )

        # Check constraints are respected
        compute_allocation = allocations[ResourceType.COMPUTE].allocated_budget
        storage_allocation = allocations[ResourceType.STORAGE].allocated_budget

        assert compute_allocation >= 400.0
        assert storage_allocation <= 100.0

    def test_allocate_budget_handles_zero_priorities(self):
        """Allocator should handle zero or missing priority weights without crashing."""
        total_budget = 600.0
        priorities = {
            ResourceType.COMPUTE: 0.0,
            ResourceType.STORAGE: 0.0,
            ResourceType.NETWORK: 0.0,
        }

        allocations = self.allocator.allocate_budget(
            total_budget=total_budget,
            priorities=priorities,
            historical_usage={},
        )

        # With all zero priorities, allocator should distribute evenly
        assert len(allocations) == len(priorities)
        expected_allocation = total_budget / len(priorities)
        for allocation in allocations.values():
            assert (
                pytest.approx(allocation.allocated_budget, rel=1e-6)
                == expected_allocation
            )


class TestResourceOptimizer:
    """Test resource optimization functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.optimizer = ResourceOptimizer()

    def test_optimize_compute_configuration(self):
        """Test compute configuration optimization."""
        current_config = {"instance_type": "medium", "cpu_cores": 4, "memory_gb": 8}

        performance_requirements = {"target_cpu_utilization": 50}  # Low utilization

        cost_constraints = {"max_hourly_cost": 1.0}

        optimization = self.optimizer.optimize_resource_configuration(
            resource_type=ResourceType.COMPUTE,
            current_config=current_config,
            performance_requirements=performance_requirements,
            cost_constraints=cost_constraints,
        )

        assert "optimized_config" in optimization
        assert "expected_improvements" in optimization
        assert "implementation_steps" in optimization

    def test_optimize_storage_configuration(self):
        """Test storage configuration optimization."""
        current_config = {"storage_type": "standard", "size_gb": 100}

        performance_requirements = {
            "access_frequency_per_day": 0.5  # Low access frequency
        }

        cost_constraints = {"max_monthly_cost": 50.0}

        optimization = self.optimizer.optimize_resource_configuration(
            resource_type=ResourceType.STORAGE,
            current_config=current_config,
            performance_requirements=performance_requirements,
            cost_constraints=cost_constraints,
        )

        assert "optimized_config" in optimization
        assert (
            optimization["optimized_config"]["storage_type"] == "cold_storage"
        )  # Should recommend cold storage


class TestSmartScheduler:
    """Test intelligent scheduling functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.scheduler = SmartScheduler()

    def test_schedule_optimization(self):
        """Test optimization scheduling."""
        optimization_request = {
            "function_name": "test_function",
            "estimated_duration": 3600,
            "estimated_cost": 20.0,
            "priority": "medium",
        }

        cost_preferences = {"cost_priority": "high"}

        time_constraints = {
            "deadline": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
            "max_delay_hours": 12,
        }

        schedule = self.scheduler.schedule_optimization(
            optimization_request=optimization_request,
            cost_preferences=cost_preferences,
            time_constraints=time_constraints,
        )

        assert "scheduled_start_time" in schedule
        assert "estimated_duration" in schedule
        assert "estimated_cost" in schedule
        assert "priority_score" in schedule
        assert "resource_allocation" in schedule

        # Should have some cost optimization applied
        assert schedule["estimated_cost"] <= optimization_request["estimated_cost"]

    def test_balanced_schedule_preserves_critical_jobs(self):
        """Balanced scheduling should keep all jobs in the queue and schedule each one."""
        now = datetime.now(timezone.utc)
        self.scheduler.schedule_windows = [
            ScheduleWindow(
                start_time=now,
                end_time=now + timedelta(hours=4),
                cost_multiplier=1.0,
            )
        ]

        jobs = [
            ScheduledJob(
                job_id="job-critical",
                job_type="analysis",
                priority=JobPriority.CRITICAL,
                estimated_duration=timedelta(hours=1),
                estimated_cost=10.0,
                resource_requirements={},
            ),
            ScheduledJob(
                job_id="job-high",
                job_type="analysis",
                priority=JobPriority.HIGH,
                estimated_duration=timedelta(hours=1),
                estimated_cost=12.0,
                resource_requirements={},
            ),
            ScheduledJob(
                job_id="job-normal",
                job_type="analysis",
                priority=JobPriority.NORMAL,
                estimated_duration=timedelta(hours=1),
                estimated_cost=8.0,
                resource_requirements={},
            ),
        ]

        for job in jobs:
            self.scheduler.add_job(job)

        queue_reference = self.scheduler.job_queue
        pre_queue_ids = [job.job_id for job in queue_reference]

        scheduled = self.scheduler.schedule_jobs("balanced")

        # All jobs should be scheduled and remain in the queue
        expected_ids = {job.job_id for job in jobs}
        assert set(scheduled.keys()) == expected_ids
        assert {job.job_id for job in self.scheduler.job_queue} == expected_ids
        assert queue_reference is self.scheduler.job_queue
        assert [job.job_id for job in self.scheduler.job_queue] == pre_queue_ids

        # Critical and normal jobs both have scheduled times assigned
        assert scheduled["job-critical"].scheduled_time is not None
        assert scheduled["job-normal"].scheduled_time is not None


if __name__ == "__main__":
    pytest.main([__file__])
