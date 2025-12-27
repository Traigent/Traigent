"""
Tests for enterprise deployment features
"""

import json
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

from traigent.security.enterprise import (
    DeploymentMode,
    EnterpriseDeploymentManager,
    HealthStatus,
    MetricsCollector,
    SLAConfiguration,
    SLAMonitor,
    SLATier,
    SystemMetrics,
)


class TestSLAConfiguration:
    """Test SLAConfiguration class"""

    def test_sla_config_from_tier(self):
        """Test SLA configuration creation from tier"""
        contact = "admin@test.com"

        # Basic tier
        basic_config = SLAConfiguration.from_tier(SLATier.BASIC, contact)
        assert basic_config.tier == SLATier.BASIC
        assert basic_config.uptime_target == 99.5
        assert basic_config.response_time_target_ms == 2000
        assert basic_config.primary_contact == contact

        # Enterprise tier
        enterprise_config = SLAConfiguration.from_tier(SLATier.ENTERPRISE, contact)
        assert enterprise_config.tier == SLATier.ENTERPRISE
        assert enterprise_config.uptime_target == 99.99
        assert enterprise_config.response_time_target_ms == 250
        assert enterprise_config.throughput_target_rps == 5000
        assert enterprise_config.error_rate_target == 0.1


class TestSystemMetrics:
    """Test SystemMetrics class"""

    def test_system_metrics_creation(self):
        """Test system metrics creation"""
        metrics = SystemMetrics(
            cpu_usage_percent=45.2,
            memory_usage_percent=60.1,
            requests_per_second=150.5,
            error_rate_percent=0.5,
        )

        assert metrics.cpu_usage_percent == 45.2
        assert metrics.memory_usage_percent == 60.1
        assert metrics.requests_per_second == 150.5
        assert metrics.error_rate_percent == 0.5
        assert metrics.health_status == HealthStatus.HEALTHY
        assert isinstance(metrics.timestamp, datetime)

    def test_system_metrics_serialization(self):
        """Test system metrics serialization"""
        metrics = SystemMetrics(
            cpu_usage_percent=25.0,
            active_optimizations=5,
            health_status=HealthStatus.DEGRADED,
            health_score=75.5,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["system"]["cpu_usage_percent"] == 25.0
        assert metrics_dict["traigent"]["active_optimizations"] == 5
        assert metrics_dict["health"]["status"] == HealthStatus.DEGRADED.value
        assert metrics_dict["health"]["score"] == 75.5
        assert "timestamp" in metrics_dict
        assert isinstance(metrics_dict["timestamp"], str)


class TestMetricsCollector:
    """Test MetricsCollector class"""

    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization"""
        collector = MetricsCollector()

        assert isinstance(collector.start_time, datetime)
        assert collector.request_counts == []
        assert collector.response_times == []
        assert collector.error_counts == []

    def test_record_request(self):
        """Test recording request metrics"""
        collector = MetricsCollector()

        # Record successful request
        collector.record_request(response_time_ms=150.0, is_error=False)

        assert len(collector.request_counts) == 1
        assert len(collector.response_times) == 1
        assert collector.response_times[0] == 150.0
        assert len(collector.error_counts) == 0

        # Record error request
        collector.record_request(response_time_ms=500.0, is_error=True)

        assert len(collector.request_counts) == 2
        assert len(collector.response_times) == 2
        assert len(collector.error_counts) == 1

    @patch("traigent.security.enterprise.psutil")
    def test_collect_system_metrics(self, mock_psutil):
        """Test collecting system metrics"""
        # Mock psutil calls
        mock_psutil.cpu_percent.return_value = 25.5
        mock_psutil.virtual_memory.return_value = Mock(percent=60.2)
        mock_psutil.disk_usage.return_value = Mock(percent=45.0)
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1000000, bytes_recv=2000000
        )

        collector = MetricsCollector()

        # Add some request data
        collector.record_request(100.0, False)
        collector.record_request(200.0, False)
        collector.record_request(300.0, True)  # Error

        metrics = collector.collect_system_metrics()

        assert metrics.cpu_usage_percent == 25.5
        assert metrics.memory_usage_percent == 60.2
        assert metrics.disk_usage_percent == 45.0
        assert metrics.network_bytes_sent == 1000000
        assert metrics.network_bytes_recv == 2000000
        assert metrics.requests_per_second > 0
        assert metrics.average_response_time_ms > 0
        assert metrics.error_rate_percent > 0
        assert isinstance(metrics.health_score, float)
        assert isinstance(metrics.health_status, HealthStatus)

    def test_record_optuna_event(self):
        """Ensure Optuna telemetry events are captured."""

        collector = MetricsCollector()
        collector.record_optuna_event(
            "trial_suggested",
            {"study_name": "test-study", "payload": {"score": 0.9}},
        )

        events = collector.get_optuna_events()
        assert len(events) == 1
        recorded = events[0]
        assert recorded["event"] == "trial_suggested"
        assert recorded["study_name"] == "test-study"

    def test_health_score_calculation(self):
        """Test health score calculation"""
        collector = MetricsCollector()

        # Test perfect health (low resource usage, fast response, no errors)
        health_score = collector._calculate_health_score(
            cpu=10.0,  # Low CPU
            memory=20.0,  # Low memory
            disk=15.0,  # Low disk
            rps=100.0,  # Good RPS
            response_time=50.0,  # Fast response
            error_rate=0.0,  # No errors
        )

        assert health_score >= 90.0  # Should be high

        # Test poor health (high resource usage, slow response, high errors)
        poor_health_score = collector._calculate_health_score(
            cpu=95.0,  # High CPU
            memory=90.0,  # High memory
            disk=85.0,  # High disk
            rps=1.0,  # Low RPS
            response_time=5000.0,  # Very slow response
            error_rate=10.0,  # High error rate
        )

        assert poor_health_score <= 20.0  # Should be low


class TestSLAMonitor:
    """Test SLAMonitor class"""

    def test_sla_monitor_initialization(self):
        """Test SLA monitor initialization"""
        sla_config = SLAConfiguration.from_tier(SLATier.STANDARD, "admin@test.com")
        metrics_collector = MetricsCollector()
        monitor = SLAMonitor(sla_config, metrics_collector)

        assert monitor.sla_config == sla_config
        assert monitor.metrics_collector == metrics_collector
        assert monitor.alert_handlers == []
        assert monitor.sla_history == []
        assert not monitor.running

    def test_add_alert_handler(self):
        """Test adding alert handlers"""
        sla_config = SLAConfiguration.from_tier(SLATier.STANDARD, "admin@test.com")
        metrics_collector = MetricsCollector()
        monitor = SLAMonitor(sla_config, metrics_collector)

        alert_handler = Mock()
        monitor.add_alert_handler(alert_handler)

        assert len(monitor.alert_handlers) == 1
        assert monitor.alert_handlers[0] == alert_handler

    def test_sla_compliance_checking(self):
        """Test SLA compliance checking"""
        sla_config = SLAConfiguration.from_tier(SLATier.STANDARD, "admin@test.com")
        metrics_collector = MetricsCollector()
        monitor = SLAMonitor(sla_config, metrics_collector)

        # Create metrics that meet SLA
        good_metrics = SystemMetrics(
            requests_per_second=600,  # Above target (500)
            average_response_time_ms=800,  # Below target (1000)
            error_rate_percent=1.0,  # Below target (2.0)
            health_status=HealthStatus.HEALTHY,
        )

        sla_status = monitor._check_sla_compliance(good_metrics)

        assert sla_status["uptime"]["compliant"]
        assert sla_status["response_time"]["compliant"]
        assert sla_status["throughput"]["compliant"]
        assert sla_status["error_rate"]["compliant"]

        # Create metrics that violate SLA
        bad_metrics = SystemMetrics(
            requests_per_second=100,  # Below target (500)
            average_response_time_ms=1500,  # Above target (1000)
            error_rate_percent=5.0,  # Above target (2.0)
            health_status=HealthStatus.CRITICAL,
        )

        sla_status = monitor._check_sla_compliance(bad_metrics)

        assert not sla_status["uptime"]["compliant"]
        assert not sla_status["response_time"]["compliant"]
        assert not sla_status["throughput"]["compliant"]
        assert not sla_status["error_rate"]["compliant"]

    def test_sla_report_generation(self):
        """Test SLA report generation"""
        sla_config = SLAConfiguration.from_tier(SLATier.ENTERPRISE, "admin@test.com")
        metrics_collector = MetricsCollector()
        monitor = SLAMonitor(sla_config, metrics_collector)

        # Add some history data
        now = datetime.now(UTC)
        for i in range(5):
            metrics = SystemMetrics(
                requests_per_second=5000 + i * 100,
                average_response_time_ms=200 + i * 10,
                error_rate_percent=0.05 + i * 0.01,
            )
            sla_status = monitor._check_sla_compliance(metrics)

            monitor.sla_history.append(
                {
                    "timestamp": (now - timedelta(hours=i)).isoformat(),
                    "metrics": metrics.to_dict(),
                    "sla_status": sla_status,
                }
            )

        # Generate report
        report = monitor.get_sla_report(hours=24)

        assert report["period_hours"] == 24
        assert report["data_points"] == 5
        assert "sla_targets" in report
        assert "compliance" in report
        assert "overall_compliance" in report

        targets = report["sla_targets"]
        assert targets["uptime"] == 99.99
        assert targets["response_time_ms"] == 250
        assert targets["error_rate"] == 0.1

    def test_sla_report_insufficient_data(self):
        """Test SLA report with insufficient data"""
        sla_config = SLAConfiguration.from_tier(SLATier.BASIC, "admin@test.com")
        metrics_collector = MetricsCollector()
        monitor = SLAMonitor(sla_config, metrics_collector)

        # No history data
        report = monitor.get_sla_report(hours=24)

        assert report["period_hours"] == 24
        assert report["data_points"] == 0
        assert report["sla_compliance"] == "insufficient_data"


class TestEnterpriseDeploymentManager:
    """Test EnterpriseDeploymentManager class"""

    def test_deployment_manager_initialization(self):
        """Test deployment manager initialization"""
        manager = EnterpriseDeploymentManager(DeploymentMode.VPC_DEDICATED)

        assert manager.deployment_mode == DeploymentMode.VPC_DEDICATED
        assert isinstance(manager.metrics_collector, MetricsCollector)
        assert manager.sla_monitor is None
        assert isinstance(manager.config, dict)

    def test_deployment_config_loading(self):
        """Test deployment configuration loading"""
        # Test cloud public config
        cloud_manager = EnterpriseDeploymentManager(DeploymentMode.CLOUD_PUBLIC)
        assert cloud_manager.config["load_balancer_enabled"]
        assert cloud_manager.config["auto_scaling_enabled"]
        assert cloud_manager.config["backup_retention_days"] == 30

        # Test VPC dedicated config
        vpc_manager = EnterpriseDeploymentManager(DeploymentMode.VPC_DEDICATED)
        assert vpc_manager.config["backup_retention_days"] == 90
        assert "vpc_config" in vpc_manager.config

        # Test on-premise config
        onprem_manager = EnterpriseDeploymentManager(DeploymentMode.ON_PREMISE)
        assert not onprem_manager.config["load_balancer_enabled"]
        assert not onprem_manager.config["auto_scaling_enabled"]
        assert onprem_manager.config["backup_retention_days"] == 365
        assert "on_premise_config" in onprem_manager.config

    def test_sla_monitoring_setup(self):
        """Test SLA monitoring setup"""
        manager = EnterpriseDeploymentManager(DeploymentMode.CLOUD_PUBLIC)

        # Setup SLA monitoring
        manager.setup_sla_monitoring(SLATier.ENTERPRISE, "admin@test.com")

        assert manager.sla_monitor is not None
        assert manager.sla_monitor.sla_config.tier == SLATier.ENTERPRISE
        assert manager.sla_monitor.sla_config.primary_contact == "admin@test.com"
        assert len(manager.sla_monitor.alert_handlers) >= 1  # Default handler

    @patch("traigent.security.enterprise.psutil")
    def test_deployment_status(self, mock_psutil):
        """Test getting deployment status"""

        # Mock psutil with proper side_effect to avoid blocking
        def mock_cpu_percent(interval=None):
            return 30.0

        mock_psutil.cpu_percent.side_effect = mock_cpu_percent
        mock_psutil.virtual_memory.return_value = Mock(percent=50.0)
        mock_psutil.disk_usage.return_value = Mock(percent=40.0)
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1000, bytes_recv=2000
        )

        manager = EnterpriseDeploymentManager(DeploymentMode.CLOUD_PRIVATE)

        status = manager.get_deployment_status()

        assert status["deployment_mode"] == DeploymentMode.CLOUD_PRIVATE.value
        assert "health_status" in status
        assert "health_score" in status
        assert "uptime_seconds" in status
        assert "metrics" in status
        assert "config" in status
        assert status["uptime_seconds"] >= 0

    @patch("traigent.security.enterprise.psutil")
    def test_health_check(self, mock_psutil):
        """Test comprehensive health check"""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value = Mock(percent=45.0)
        mock_psutil.disk_usage.return_value = Mock(percent=35.0)
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1000, bytes_recv=2000
        )

        manager = EnterpriseDeploymentManager(DeploymentMode.CLOUD_PUBLIC)

        health_check = manager.perform_health_check()

        assert health_check["overall_status"] in ["healthy", "unhealthy"]
        assert "timestamp" in health_check
        assert "checks" in health_check

        checks = health_check["checks"]
        assert "system" in checks
        assert "database" in checks
        assert "external_services" in checks

        # System check should have status and score
        assert "status" in checks["system"]
        if checks["system"]["status"] != "error":
            assert "score" in checks["system"]
            assert "details" in checks["system"]

    def test_create_backup(self):
        """Test backup creation"""
        manager = EnterpriseDeploymentManager(DeploymentMode.ON_PREMISE)

        backup_info = manager.create_backup()

        assert "backup_id" in backup_info
        assert backup_info["backup_id"].startswith("backup_")
        assert "timestamp" in backup_info
        assert backup_info["deployment_mode"] == DeploymentMode.ON_PREMISE.value
        assert backup_info["status"] == "completed"
        assert backup_info["size_bytes"] > 0
        assert "retention_until" in backup_info

        # Retention should be based on deployment config
        assert manager.config["backup_retention_days"] == 365  # On-premise default

    @patch("traigent.security.enterprise.psutil")
    def test_enterprise_dashboard(self, mock_psutil):
        """Test enterprise dashboard generation"""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 20.0
        mock_psutil.virtual_memory.return_value = Mock(percent=40.0)
        mock_psutil.disk_usage.return_value = Mock(percent=30.0)
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=5000, bytes_recv=10000
        )

        manager = EnterpriseDeploymentManager(DeploymentMode.VPC_DEDICATED)

        # Setup SLA monitoring for dashboard
        manager.setup_sla_monitoring(SLATier.PREMIUM, "admin@test.com")

        dashboard = manager.get_enterprise_dashboard()

        assert "timestamp" in dashboard
        assert "deployment" in dashboard
        assert "health" in dashboard
        assert "performance" in dashboard
        assert "capacity" in dashboard

        deployment = dashboard["deployment"]
        assert deployment["mode"] == DeploymentMode.VPC_DEDICATED.value
        assert "uptime_hours" in deployment
        assert "config" in deployment

        performance = dashboard["performance"]
        assert "cpu_usage" in performance
        assert "memory_usage" in performance
        assert "requests_per_second" in performance
        assert "response_time_ms" in performance

        # Should include SLA info
        assert "sla" in dashboard

    def test_shutdown(self):
        """Test graceful shutdown"""
        manager = EnterpriseDeploymentManager(DeploymentMode.CLOUD_PUBLIC)

        # Setup SLA monitoring
        manager.setup_sla_monitoring(SLATier.STANDARD, "admin@test.com")

        # Shutdown should complete without errors
        manager.shutdown()

        # SLA monitor should be stopped
        assert not manager.sla_monitor.running

    def test_config_file_loading(self):
        """Test loading configuration from file"""
        EnterpriseDeploymentManager(DeploymentMode.CLOUD_PUBLIC)

        # Create temporary config file
        config_data = {
            "custom_setting": "test_value",
            "monitoring_interval_seconds": 30,
            "backup_retention_days": 60,
        }

        config_file = f"enterprise_config_{DeploymentMode.CLOUD_PUBLIC.value}.json"

        try:
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            # Create new manager that should load the config file
            new_manager = EnterpriseDeploymentManager(DeploymentMode.CLOUD_PUBLIC)

            # Should have custom settings
            assert new_manager.config["custom_setting"] == "test_value"
            assert new_manager.config["monitoring_interval_seconds"] == 30
            assert new_manager.config["backup_retention_days"] == 60

            # Should still have default settings not overridden
            assert "load_balancer_enabled" in new_manager.config

        finally:
            # Cleanup
            if os.path.exists(config_file):
                os.unlink(config_file)

    def test_default_alert_handler(self):
        """Test default alert handler"""
        manager = EnterpriseDeploymentManager(DeploymentMode.CLOUD_PUBLIC)

        # Test alert handler doesn't crash
        alert_data = {
            "message": "Test alert",
            "details": {"metric": "response_time", "value": 1500},
        }

        # Should not raise exception
        manager._default_alert_handler("response_time_alert", alert_data)

    @patch("traigent.security.enterprise.psutil")
    def test_monitoring_with_different_intervals(self, mock_psutil):
        """Test monitoring with different intervals"""
        # Mock psutil
        mock_psutil.cpu_percent.return_value = 15.0
        mock_psutil.virtual_memory.return_value = Mock(percent=35.0)
        mock_psutil.disk_usage.return_value = Mock(percent=25.0)
        mock_psutil.net_io_counters.return_value = Mock(
            bytes_sent=1000, bytes_recv=2000
        )

        # Set custom monitoring interval
        manager = EnterpriseDeploymentManager(DeploymentMode.ON_PREMISE)
        manager.config["monitoring_interval_seconds"] = 10

        # Setup SLA monitoring
        manager.setup_sla_monitoring(SLATier.BASIC, "admin@test.com")

        # SLA monitor should be running
        assert manager.sla_monitor.running

        # Cleanup
        manager.shutdown()
        assert not manager.sla_monitor.running
