"""Tests for cost optimization AI intelligence module."""

from datetime import datetime, timedelta, timezone

import pytest

from traigent.analytics.intelligence import CostOptimizationAI


class TestCostOptimizationAI:
    """Test CostOptimizationAI functionality."""

    def test_initialization(self):
        """Test AI initialization with different configurations."""
        # Default initialization
        ai = CostOptimizationAI()
        assert ai.min_confidence == 0.7
        assert ai.anomaly_threshold == 2.0

        # Custom initialization
        ai_custom = CostOptimizationAI(
            min_confidence=0.9, anomaly_threshold=3.0, enable_ml=True
        )
        assert ai_custom.min_confidence == 0.9
        assert ai_custom.anomaly_threshold == 3.0

    def test_analyze_cost_patterns(self):
        """Test cost pattern analysis."""
        ai = CostOptimizationAI()

        # Create usage data with patterns
        usage_data = []
        base_cost = 100
        for day in range(30):
            for hour in range(24):
                timestamp = datetime.now(timezone.utc) - timedelta(days=day, hours=hour)
                # Weekly pattern (weekends cheaper)
                weekly_factor = 0.7 if timestamp.weekday() >= 5 else 1.0
                # Daily pattern (peak hours more expensive)
                daily_factor = 1.5 if 9 <= hour <= 17 else 0.8

                cost = base_cost * weekly_factor * daily_factor
                usage_data.append(
                    {
                        "timestamp": timestamp,
                        "cost": cost,
                        "resource_type": "compute",
                        "usage": {
                            "cpu_hours": cost / 0.1,
                            "memory_gb_hours": cost / 0.05,
                        },
                    }
                )

        analysis = ai.analyze_cost_patterns(usage_data)

        assert "patterns" in analysis
        assert "anomalies" in analysis
        assert "trends" in analysis
        assert "summary" in analysis

        # Should detect weekly and daily patterns
        patterns = analysis["patterns"]
        assert any(p["type"] == "weekly" for p in patterns)
        assert any(p["type"] == "daily" for p in patterns)

    def test_get_optimization_recommendations(self):
        """Test optimization recommendation generation."""
        ai = CostOptimizationAI()

        current_usage = {
            "compute": {"instances": 10, "utilization": 0.3, "cost_per_hour": 1.0},
            "storage": {
                "total_gb": 1000,
                "accessed_percentage": 0.2,
                "cost_per_gb": 0.1,
            },
            "network": {"egress_gb": 500, "cost_per_gb": 0.12},
        }

        objectives = ["reduce_cost", "maintain_performance"]
        constraints = {"min_availability": 0.99, "max_latency_ms": 100}

        recommendations = ai.get_optimization_recommendations(
            current_usage, objectives, constraints
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        for rec in recommendations:
            assert isinstance(rec, dict)
            assert "resource_type" in rec
            assert "priority" in rec
            assert "confidence" in rec
            assert "estimated_savings" in rec
            assert 0 <= rec["confidence"] <= 1
            assert rec["estimated_savings"] >= 0

    def test_simulate_optimization_impact(self):
        """Test optimization impact simulation."""
        ai = CostOptimizationAI()

        optimization_params = {
            "compute_reduction": 0.3,  # Reduce compute by 30%
            "storage_tiering": True,  # Enable storage tiering
            "spot_instances": 0.5,  # Use 50% spot instances
        }

        current_metrics = {
            "daily_cost": 1000,
            "performance_score": 0.95,
            "availability": 0.995,
        }

        simulation = ai.simulate_optimization_impact(
            optimization_params, duration_days=30, current_metrics=current_metrics
        )

        assert "cost_savings" in simulation
        assert "performance_impact" in simulation
        assert "risk_assessment" in simulation
        assert "timeline" in simulation

        # Cost savings should be positive
        assert simulation["cost_savings"]["total"] > 0
        assert simulation["cost_savings"]["percentage"] > 0

    def test_simulate_optimization_impact_zero_baseline(self):
        """Zero baseline cost should not cause division errors."""
        ai = CostOptimizationAI()

        optimization_params = {
            "compute_reduction": 0.5,
            "storage_tiering": True,
        }

        current_metrics = {
            "daily_cost": 0,
            "performance_score": 0.95,
        }

        simulation = ai.simulate_optimization_impact(
            optimization_params, duration_days=7, current_metrics=current_metrics
        )

        assert simulation["current_cost"] == 0
        assert simulation["projected_cost"] == 0
        assert simulation["cost_savings"]["total"] == 0
        assert simulation["cost_savings"]["percentage"] == 0
        assert simulation["savings_percentage"] == 0

    def test_identify_usage_anomalies(self):
        """Test usage anomaly detection."""
        ai = CostOptimizationAI()

        # Normal usage pattern with anomalies
        usage_data = []
        for i in range(100):
            timestamp = datetime.now(timezone.utc) - timedelta(hours=i)
            cost = 100 + (i % 24) * 2  # Normal daily pattern

            # Insert anomalies
            if i in [25, 50, 75]:
                cost = cost * 5  # 5x spike

            usage_data.append(
                {"timestamp": timestamp, "cost": cost, "resource_type": "compute"}
            )

        anomalies = ai.identify_usage_anomalies(usage_data)

        assert len(anomalies) >= 3
        for anomaly in anomalies:
            assert isinstance(anomaly, dict)
            assert "severity" in anomaly
            assert "deviation_score" in anomaly
            assert anomaly["severity"] in ["low", "medium", "high"]
            assert anomaly["deviation_score"] > ai.anomaly_threshold

    def test_predict_future_costs(self):
        """Test future cost prediction."""
        ai = CostOptimizationAI()

        # Historical data with growth trend
        historical_data = []
        base_cost = 1000
        growth_rate = 0.05  # 5% monthly growth

        for month in range(12, 0, -1):
            for day in range(30):
                timestamp = datetime.now(timezone.utc) - timedelta(
                    days=month * 30 + day
                )
                cost = base_cost * (1 + growth_rate) ** (12 - month)
                historical_data.append(
                    {
                        "timestamp": timestamp,
                        "cost": cost + (day * 10),  # Daily variation
                        "usage": {"requests": int(cost * 100)},
                    }
                )

        prediction = ai.predict_future_costs(
            historical_data, forecast_days=90, include_seasonality=True
        )

        assert "predictions" in prediction
        assert "confidence_intervals" in prediction
        assert "growth_rate" in prediction
        assert "seasonality_factors" in prediction

        # Should detect positive growth
        assert prediction["growth_rate"] > 0

    def test_optimize_resource_allocation(self):
        """Test resource allocation optimization."""
        ai = CostOptimizationAI()

        resources = {
            "compute": {
                "current_allocation": 100,
                "utilization": 0.4,
                "cost_per_unit": 0.1,
            },
            "storage": {
                "current_allocation": 1000,
                "utilization": 0.6,
                "cost_per_unit": 0.01,
            },
            "memory": {
                "current_allocation": 500,
                "utilization": 0.8,
                "cost_per_unit": 0.05,
            },
        }

        workload_requirements = {
            "min_compute": 30,
            "min_storage": 500,
            "min_memory": 350,
            "peak_multiplier": 2.0,
        }

        optimization = ai.optimize_resource_allocation(
            resources, workload_requirements, optimization_goal="cost"
        )

        assert "recommended_allocation" in optimization
        assert "estimated_savings" in optimization
        assert "utilization_improvement" in optimization

        # Should recommend reduction for underutilized resources
        assert optimization["recommended_allocation"]["compute"] < 100

    def test_generate_report_handles_missing_execution_time(self, monkeypatch):
        """Report generation should succeed when execution times are unavailable."""
        ai = CostOptimizationAI()

        # Ensure no execution data is present
        ai.optimization_results.clear()

        monkeypatch.setattr(
            CostOptimizationAI,
            "analyze_current_state",
            lambda self: {
                "total_resources": 0,
                "total_cost_last_period": 0.0,
                "resource_analysis": {},
                "optimization_opportunities": [],
            },
        )

        report = ai.generate_report()

        assert "Average Execution Time" in report
        assert "0.0s" in report

    def test_multi_cloud_optimization(self):
        """Test multi-cloud cost optimization."""
        ai = CostOptimizationAI()

        cloud_usage = {
            "aws": {
                "compute": {"cost": 5000, "instances": 50},
                "storage": {"cost": 1000, "gb": 10000},
            },
            "azure": {
                "compute": {"cost": 4500, "instances": 45},
                "storage": {"cost": 1200, "gb": 12000},
            },
            "gcp": {
                "compute": {"cost": 4000, "instances": 40},
                "storage": {"cost": 800, "gb": 8000},
            },
        }

        recommendations = ai.optimize_multi_cloud(
            cloud_usage,
            constraints={
                "data_sovereignty": {"eu": "azure", "us": "aws"},
                "min_providers": 2,
            },
        )

        assert "workload_distribution" in recommendations
        assert "estimated_savings" in recommendations
        assert "migration_plan" in recommendations

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        ai = CostOptimizationAI()

        # Empty data
        with pytest.raises(ValueError):
            ai.analyze_cost_patterns([])

        # Invalid data
        invalid_data = [{"timestamp": "invalid", "cost": "not_a_number"}]
        analysis = ai.analyze_cost_patterns(invalid_data, safe_mode=True)
        assert "error" in analysis

        # Extreme values
        extreme_data = [
            {
                "timestamp": datetime.now(timezone.utc),
                "cost": 1e10,  # Very high cost
                "resource_type": "compute",
            }
        ]
        analysis = ai.analyze_cost_patterns(extreme_data)
        assert len(analysis["anomalies"]) > 0

    def test_recommendation_prioritization(self):
        """Test recommendation prioritization logic."""
        ai = CostOptimizationAI()

        # Create multiple recommendations
        recommendations = [
            {
                "resource_type": "compute",
                "action": "Reduce instances",
                "description": "Scale down underutilized instances",
                "estimated_savings": 5000,
                "implementation_effort": "low",
                "risk_level": "low",
                "priority": "high",
                "confidence": 0.9,
            },
            {
                "resource_type": "storage",
                "action": "Enable tiering",
                "description": "Move cold data to cheaper storage",
                "estimated_savings": 2000,
                "implementation_effort": "medium",
                "risk_level": "low",
                "priority": "medium",
                "confidence": 0.8,
            },
            {
                "resource_type": "network",
                "action": "Optimize routing",
                "description": "Use regional endpoints",
                "estimated_savings": 500,
                "implementation_effort": "high",
                "risk_level": "medium",
                "priority": "low",
                "confidence": 0.7,
            },
        ]

        prioritized = ai.prioritize_recommendations(
            recommendations, criteria=["savings", "effort", "risk"]
        )

        # Should be sorted by priority and savings
        assert prioritized[0]["priority"] == "high"
        assert (
            prioritized[0]["estimated_savings"] >= prioritized[1]["estimated_savings"]
        )

    def test_seasonal_cost_analysis(self):
        """Test seasonal cost pattern analysis."""
        ai = CostOptimizationAI()

        # Create data with seasonal patterns - use specific dates to ensure correct months
        usage_data = []
        datetime(2023, 1, 1)  # Use fixed date to ensure predictable months

        for month in range(1, 13):  # 1-12 for Jan-Dec
            base_cost = 1000
            # Holiday season spike in November (11) and December (12)
            seasonal_factor = 2.0 if month in [11, 12] else 1.0

            for day in range(1, 31):  # 1-30 days
                try:
                    timestamp = datetime(2023, month, day)
                    cost = base_cost * seasonal_factor
                    usage_data.append(
                        {
                            "timestamp": timestamp,
                            "cost": cost,
                            "resource_type": "compute",
                        }
                    )
                except ValueError:
                    # Skip invalid dates like Feb 30
                    continue

        analysis = ai.analyze_cost_patterns(usage_data)

        # Should detect seasonal pattern
        seasonal_patterns = [p for p in analysis["patterns"] if p["type"] == "seasonal"]
        assert len(seasonal_patterns) > 0
        # Should detect peaks in November (11) and December (12)
        assert set(seasonal_patterns[0]["peak_months"]) == {11, 12}

    def test_cost_allocation_analysis(self):
        """Test cost allocation and attribution analysis."""
        ai = CostOptimizationAI()

        cost_data = {
            "departments": {
                "engineering": {"compute": 5000, "storage": 2000},
                "data_science": {"compute": 8000, "storage": 5000},
                "operations": {"compute": 2000, "storage": 1000},
            },
            "projects": {
                "ml_training": {"cost": 10000, "department": "data_science"},
                "web_app": {"cost": 3000, "department": "engineering"},
                "monitoring": {"cost": 2000, "department": "operations"},
            },
        }

        allocation = ai.analyze_cost_allocation(cost_data)

        assert "by_department" in allocation
        assert "by_resource_type" in allocation
        assert "inefficiencies" in allocation
        assert "reallocation_opportunities" in allocation

    def test_external_pricing_integration(self):
        """Test integration with pricing data (uses static fallback for aws/azure)."""
        ai = CostOptimizationAI()

        # fetch_current_pricing uses static data for aws/azure providers
        pricing = ai.fetch_current_pricing(["aws", "azure"])

        assert "aws" in pricing
        assert "compute" in pricing["aws"]
        assert (
            pricing["aws"]["compute"]["spot"] < pricing["aws"]["compute"]["on_demand"]
        )

    def test_concurrent_analysis(self):
        """Test thread safety of analysis operations."""
        import threading

        ai = CostOptimizationAI()
        results = []
        errors = []

        def analyze_data():
            try:
                data = [
                    {
                        "timestamp": datetime.now(timezone.utc) - timedelta(hours=i),
                        "cost": 100 + i * 10,
                        "resource_type": "compute",
                    }
                    for i in range(100)
                ]
                result = ai.analyze_cost_patterns(data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=analyze_data) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5
        # All results should be consistent
        assert all(
            r["summary"]["total_cost"] == results[0]["summary"]["total_cost"]
            for r in results
        )
