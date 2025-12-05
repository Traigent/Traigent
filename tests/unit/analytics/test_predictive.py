"""Tests for predictive analytics module."""

from datetime import UTC, datetime, timedelta

import pytest

from traigent.analytics.predictive import (
    CostForecaster,
    ForecastPeriod,
    ForecastResult,
    PerformanceForecaster,
    PredictiveAnalytics,
    TrendAnalyzer,
    UsageMetric,
)


class TestPredictiveAnalyzer:
    """Test PredictiveAnalyzer functionality."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = PredictiveAnalytics()

        assert hasattr(analyzer, "analyze_patterns")
        assert hasattr(analyzer, "predict_future")
        assert hasattr(analyzer, "generate_insights")

    def test_analyze_optimization_data(self):
        """Test optimization data analysis."""
        analyzer = PredictiveAnalytics()

        # Mock historical data
        historical_data = [
            {
                "timestamp": datetime.now(UTC) - timedelta(days=i),
                "config": {"model": "gpt-3.5", "temperature": 0.7},
                "metrics": {
                    "accuracy": 0.85 + i * 0.01,
                    "cost": 0.05 - i * 0.001,
                    "latency": 100 - i * 5,
                },
            }
            for i in range(10)
        ]

        analysis = analyzer.analyze_patterns(historical_data)

        assert "summary" in analysis
        assert "trends" in analysis
        assert "predictions" in analysis
        assert "recommendations" in analysis
        assert "confidence" in analysis

    def test_predict_future_metrics(self):
        """Test future metrics prediction."""
        analyzer = PredictiveAnalytics()

        current_metrics = {"accuracy": 0.90, "cost": 0.04, "latency": 80}

        predictions = analyzer.predict_future(
            current_metrics, forecast_period=ForecastPeriod.WEEKLY
        )

        assert "metrics" in predictions
        assert "confidence_intervals" in predictions
        assert "risk_factors" in predictions

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        analyzer = PredictiveAnalytics()

        analysis = analyzer.analyze_patterns([])

        assert analysis["summary"]["status"] == "insufficient_data"
        assert len(analysis["trends"]) == 0
        assert analysis["confidence"] == 0.0


class TestTrendAnalyzer:
    """Test TrendAnalyzer functionality."""

    def test_analyze_trends(self):
        """Test trend analysis."""
        analyzer = TrendAnalyzer()

        # Create sample time series data with increasing trend
        data_points = [
            {
                "timestamp": datetime.now(UTC) - timedelta(hours=i),
                "value": 10 + (24 - i) * 0.5,
            }
            for i in range(24, 0, -1)
        ]

        trends = analyzer.analyze_trends(data_points, "value")

        assert "direction" in trends
        assert "strength" in trends
        assert "volatility" in trends
        assert "change_points" in trends
        assert trends["direction"] == "increasing"

    def test_seasonal_detection(self):
        """Test seasonal pattern detection."""
        analyzer = TrendAnalyzer()

        # Create data with daily seasonality
        data_points = []
        for day in range(7):
            for hour in range(24):
                timestamp = datetime.now(UTC) - timedelta(days=day, hours=hour)
                # Peak at noon, low at midnight
                value = 50 + 30 * abs(12 - hour) / 12
                data_points.append({"timestamp": timestamp, "value": value})

        seasonality = analyzer.detect_seasonality(data_points, "value")

        assert seasonality["has_seasonality"]
        assert seasonality["period"] == "daily"
        assert "pattern_strength" in seasonality

    def test_anomaly_detection(self):
        """Test anomaly detection in trends."""
        analyzer = TrendAnalyzer()

        # Normal data with one anomaly
        data_points = [
            {
                "timestamp": datetime.now(UTC) - timedelta(hours=i),
                "value": 10 + i * 0.1,
            }
            for i in range(20, 0, -1)
        ]
        # Insert anomaly
        data_points[10]["value"] = 100

        anomalies = analyzer.detect_anomalies(data_points, "value")

        assert len(anomalies) > 0
        assert anomalies[0]["severity"] == "high"


class TestCostForecaster:
    """Test CostForecaster functionality."""

    def test_forecast_costs(self):
        """Test cost forecasting."""
        forecaster = CostForecaster()

        historical_costs = [
            {
                "timestamp": datetime.now(UTC) - timedelta(days=i),
                "cost": 100 + i * 10,
                "usage": {"requests": 1000 + i * 100},
            }
            for i in range(30, 0, -1)
        ]

        forecast = forecaster.forecast_costs(
            historical_costs, forecast_period=ForecastPeriod.MONTHLY
        )

        assert isinstance(forecast, ForecastResult)
        assert forecast.predicted_values is not None
        assert forecast.confidence_intervals is not None
        assert forecast.trend_direction is not None
        assert forecast.forecast_horizon_days > 0

    def test_budget_alert(self):
        """Test budget alert generation."""
        forecaster = CostForecaster()

        current_spend = 800
        budget = 1000
        days_remaining = 10

        alert = forecaster.check_budget_alert(current_spend, budget, days_remaining)

        assert alert["alert_level"] in ["none", "warning", "critical"]
        assert "projected_overage" in alert
        assert "recommended_actions" in alert


class TestOptimizationOutcomePredictor:
    """Test OptimizationOutcomePredictor functionality."""

    def test_predict_optimization_outcome(self):
        """Test optimization outcome prediction."""
        predictor = PerformanceForecaster()

        current_state = {
            "config": {"model": "gpt-3.5", "temperature": 0.7},
            "metrics": {"accuracy": 0.85, "cost": 0.05},
        }

        prediction = predictor.forecast(current_state, ForecastPeriod.DAILY)

        assert "expected_improvement" in prediction
        assert "confidence" in prediction
        assert "risks" in prediction
        assert "recommendation" in prediction

    def test_compare_configurations(self):
        """Test configuration comparison."""
        predictor = PerformanceForecaster()

        configs = [
            {"model": "gpt-3.5", "temperature": 0.7},
            {"model": "gpt-4", "temperature": 0.3},
            {"model": "gpt-3.5", "temperature": 0.1},
        ]

        comparison = predictor.forecast(configs[0], ForecastPeriod.DAILY)

        assert comparison is not None
        assert isinstance(comparison, dict)
        assert "expected_improvement" in comparison
        assert "confidence" in comparison


class TestResourceUsagePredictor:
    """Test ResourceUsagePredictor functionality."""

    def test_predict_resource_usage(self):
        """Test resource usage prediction."""
        predictor = TrendAnalyzer()

        historical_usage = [
            {
                "timestamp": datetime.now(UTC) - timedelta(hours=i),
                "cpu": 50 + i * 2,
                "memory": 4000 + i * 100,
                "requests": 100 + i * 10,
            }
            for i in range(24, 0, -1)
        ]

        prediction = predictor.analyze_trends(historical_usage, "cpu")

        assert prediction is not None
        assert "direction" in prediction

    def test_capacity_planning(self):
        """Test capacity planning recommendations."""
        predictor = TrendAnalyzer()

        # Provide sufficient data for trend analysis
        data = [
            {
                "timestamp": datetime.now(UTC) - timedelta(hours=i),
                "value": 50 + i,
            }
            for i in range(5)
        ]
        plan = predictor.analyze_trends(data, "value")

        assert plan is not None
        assert "direction" in plan


class TestScalingRecommender:
    """Test ScalingRecommender functionality."""

    def test_recommend_scaling(self):
        """Test scaling recommendations."""
        recommender = CostForecaster()

        data = [{"timestamp": datetime.now(UTC), "cost": 100}]
        recommendations = recommender.forecast_costs(
            data, forecast_period=ForecastPeriod.DAILY
        )

        assert recommendations is not None
        assert isinstance(recommendations, ForecastResult)

    def test_auto_scaling_policy(self):
        """Test auto-scaling policy generation."""
        recommender = CostForecaster()

        data = [{"timestamp": datetime.now(UTC), "cost": 100}]
        policy = recommender.forecast_costs(data, forecast_period=ForecastPeriod.DAILY)

        assert policy is not None
        assert isinstance(policy, ForecastResult)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        analyzer = PredictiveAnalytics()

        # Only 2 data points
        sparse_data = [
            {"timestamp": datetime.now(UTC), "metrics": {"accuracy": 0.85}},
            {
                "timestamp": datetime.now(UTC) - timedelta(hours=1),
                "metrics": {"accuracy": 0.84},
            },
        ]

        analysis = analyzer.analyze_patterns(sparse_data)
        assert analysis["confidence"] < 0.5
        assert "insufficient_data" in analysis["warnings"]

    def test_outlier_handling(self):
        """Test handling of outliers in data."""
        analyzer = TrendAnalyzer()

        # Data with extreme outliers
        data_points = [
            {"timestamp": datetime.now(UTC) - timedelta(hours=i), "value": 10}
            for i in range(10)
        ]
        # Add outliers
        data_points[3]["value"] = 1000
        data_points[7]["value"] = -500

        trends = analyzer.analyze_trends(data_points, "value", remove_outliers=True)

        assert trends["outliers_removed"] == 2
        assert "cleaned_trend" in trends

    def test_missing_metrics(self):
        """Test handling of missing metrics."""
        predictor = PerformanceForecaster()

        incomplete_state = {
            "config": {"model": "gpt-3.5"},
            "metrics": {"accuracy": 0.85},  # Missing cost
        }

        prediction = predictor.forecast(incomplete_state, ForecastPeriod.DAILY)

        assert "warnings" in prediction
        assert "missing_metrics" in prediction["warnings"]

    @pytest.mark.parametrize(
        "period",
        [
            ForecastPeriod.DAILY,
            ForecastPeriod.WEEKLY,
            ForecastPeriod.MONTHLY,
            ForecastPeriod.QUARTERLY,
        ],
    )
    def test_all_forecast_periods(self, period):
        """Test all forecast period options."""
        forecaster = CostForecaster()

        historical_costs = [
            {
                "timestamp": datetime.now(UTC) - timedelta(days=i),
                "cost": 100 + i * 5,
            }
            for i in range(60, 0, -1)
        ]

        forecast = forecaster.forecast_costs(historical_costs, forecast_period=period)

        assert isinstance(forecast, ForecastResult)
        assert forecast.period == period
        assert len(forecast.predicted_values) > 0

    def test_concurrent_predictions(self):
        """Test thread safety of predictions."""
        import threading

        analyzer = PredictiveAnalytics()
        results = []
        errors = []

        def make_prediction():
            try:
                result = analyzer.predict_future(
                    {"accuracy": 0.85}, forecast_period=ForecastPeriod.DAILY
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_prediction) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    def test_generate_comprehensive_forecast(self):
        """Comprehensive forecast should aggregate component forecasts."""
        analyzer = PredictiveAnalytics()

        # Populate cost forecaster history
        now = datetime.now(UTC)
        for i in range(5):
            analyzer.cost_forecaster.add_usage_data(
                UsageMetric(
                    timestamp=now - timedelta(days=i),
                    optimizations_count=10,
                    total_trials=100,
                    total_duration_seconds=200.0,
                    compute_cost=50.0 + i,
                    storage_cost=10.0 + i * 0.5,
                    api_calls=1000,
                    active_users=25,
                )
            )

        forecast = analyzer.generate_comprehensive_forecast(forecast_days=30)

        assert "cost_forecasts" in forecast
        assert "performance_forecasts" in forecast
        assert forecast["forecast_period_days"] == 30
        assert isinstance(forecast["cost_forecasts"], ForecastResult)
        assert len(forecast["recommendations"]) >= 0
