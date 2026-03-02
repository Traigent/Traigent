"""Unit tests for deployment.

Tests for enterprise deployment and monitoring functionality,
including deployment modes, health checks, SLA monitoring, and backup management.
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Reliability
# Traceability: FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from itertools import count
from unittest.mock import MagicMock, patch

import pytest

from traigent.security.deployment import (
    BackupManager,
    DeploymentManager,
    DeploymentMode,
    HealthCheck,
    HealthChecker,
    HealthStatus,
    SLAMetrics,
    SLAMonitor,
)


class TestDeploymentMode:
    """Tests for DeploymentMode enum."""

    def test_deployment_mode_values(self) -> None:
        """Test all deployment mode values are accessible."""
        assert DeploymentMode.CLOUD.value == "cloud"
        assert DeploymentMode.VPC.value == "vpc"
        assert DeploymentMode.ON_PREMISE.value == "on_premise"
        assert DeploymentMode.HYBRID.value == "hybrid"

    def test_deployment_mode_equality(self) -> None:
        """Test deployment mode equality comparisons."""
        assert DeploymentMode.CLOUD == DeploymentMode.CLOUD
        assert DeploymentMode.VPC != DeploymentMode.CLOUD

    def test_deployment_mode_from_string(self) -> None:
        """Test creating deployment mode from string value."""
        mode = DeploymentMode("cloud")
        assert mode == DeploymentMode.CLOUD


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self) -> None:
        """Test all health status values are accessible."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_health_status_equality(self) -> None:
        """Test health status equality comparisons."""
        assert HealthStatus.HEALTHY == HealthStatus.HEALTHY
        assert HealthStatus.DEGRADED != HealthStatus.HEALTHY

    def test_health_status_from_string(self) -> None:
        """Test creating health status from string value."""
        status = HealthStatus("healthy")
        assert status == HealthStatus.HEALTHY


class TestHealthCheck:
    """Tests for HealthCheck dataclass."""

    def test_health_check_creation_with_defaults(self) -> None:
        """Test creating health check with default timestamp."""
        check = HealthCheck(
            service="test-service",
            status=HealthStatus.HEALTHY,
            response_time_ms=50.0,
        )
        assert check.service == "test-service"
        assert check.status == HealthStatus.HEALTHY
        assert check.response_time_ms == 50.0
        assert isinstance(check.timestamp, datetime)
        assert check.timestamp.tzinfo == UTC
        assert check.details == {}

    def test_health_check_creation_with_custom_timestamp(self) -> None:
        """Test creating health check with custom timestamp."""
        custom_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        check = HealthCheck(
            service="test-service",
            status=HealthStatus.HEALTHY,
            response_time_ms=50.0,
            timestamp=custom_time,
        )
        assert check.timestamp == custom_time

    def test_health_check_with_details(self) -> None:
        """Test creating health check with additional details."""
        details = {"connections": 45, "query_time_avg_ms": 12}
        check = HealthCheck(
            service="database",
            status=HealthStatus.HEALTHY,
            response_time_ms=30.0,
            details=details,
        )
        assert check.details == details
        assert check.details["connections"] == 45

    def test_health_check_all_statuses(self) -> None:
        """Test health check with all possible status values."""
        for status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.UNKNOWN,
        ]:
            check = HealthCheck(service="test", status=status, response_time_ms=100.0)
            assert check.status == status


class TestSLAMetrics:
    """Tests for SLAMetrics dataclass."""

    def test_sla_metrics_with_defaults(self) -> None:
        """Test creating SLA metrics with default values."""
        metrics = SLAMetrics()
        assert metrics.uptime_percentage == 99.99
        assert metrics.response_time_p95_ms == 500.0
        assert metrics.error_rate_percentage == 0.01
        assert isinstance(metrics.measured_at, datetime)
        assert metrics.measured_at.tzinfo == UTC

    def test_sla_metrics_with_custom_values(self) -> None:
        """Test creating SLA metrics with custom values."""
        custom_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC)
        metrics = SLAMetrics(
            uptime_percentage=99.95,
            response_time_p95_ms=750.0,
            error_rate_percentage=0.05,
            measured_at=custom_time,
        )
        assert metrics.uptime_percentage == 99.95
        assert metrics.response_time_p95_ms == 750.0
        assert metrics.error_rate_percentage == 0.05
        assert metrics.measured_at == custom_time

    def test_sla_metrics_edge_values(self) -> None:
        """Test SLA metrics with edge case values."""
        metrics = SLAMetrics(
            uptime_percentage=100.0,
            response_time_p95_ms=0.0,
            error_rate_percentage=0.0,
        )
        assert metrics.uptime_percentage == 100.0
        assert metrics.response_time_p95_ms == 0.0
        assert metrics.error_rate_percentage == 0.0


class TestDeploymentManagerInit:
    """Tests for DeploymentManager initialization."""

    def test_init_with_default_mode(self) -> None:
        """Test initialization with default CLOUD mode."""
        manager = DeploymentManager()
        assert manager.mode == DeploymentMode.CLOUD
        assert manager.services == {}
        assert manager.configuration is not None

    def test_init_with_cloud_mode(self) -> None:
        """Test initialization with CLOUD mode."""
        manager = DeploymentManager(mode=DeploymentMode.CLOUD)
        assert manager.mode == DeploymentMode.CLOUD
        assert manager.configuration["auto_scaling"] is True
        assert manager.configuration["load_balancing"] is True
        assert manager.configuration["multi_az"] is True
        assert manager.configuration["backup_retention_days"] == 30

    def test_init_with_vpc_mode(self) -> None:
        """Test initialization with VPC mode."""
        manager = DeploymentManager(mode=DeploymentMode.VPC)
        assert manager.mode == DeploymentMode.VPC
        assert manager.configuration["vpc_id"] == "vpc-12345"
        assert manager.configuration["private_subnets"] is True
        assert manager.configuration["nat_gateway"] is True
        assert manager.configuration["security_groups"] == ["sg-security"]

    def test_init_with_on_premise_mode(self) -> None:
        """Test initialization with ON_PREMISE mode."""
        manager = DeploymentManager(mode=DeploymentMode.ON_PREMISE)
        assert manager.mode == DeploymentMode.ON_PREMISE
        assert manager.configuration["data_residency"] == "local"
        assert manager.configuration["air_gapped"] is True
        assert manager.configuration["custom_ca"] is True

    def test_init_with_hybrid_mode(self) -> None:
        """Test initialization with HYBRID mode."""
        manager = DeploymentManager(mode=DeploymentMode.HYBRID)
        assert manager.mode == DeploymentMode.HYBRID
        assert manager.configuration["cloud_backup"] is True
        assert manager.configuration["local_processing"] is True
        assert manager.configuration["sync_interval"] == 3600


class TestDeploymentManagerDeployService:
    """Tests for DeploymentManager.deploy_service method."""

    @pytest.fixture
    def manager(self) -> DeploymentManager:
        """Create test deployment manager instance."""
        return DeploymentManager(mode=DeploymentMode.CLOUD)

    def test_deploy_service_basic(self, manager: DeploymentManager) -> None:
        """Test basic service deployment."""
        config = {"replicas": 3, "port": 8080}
        result = manager.deploy_service("api-service", config)

        assert result["service_name"] == "api-service"
        assert result["mode"] == "cloud"
        assert result["config"] == config
        assert result["status"] == "deployed"
        assert "deployed_at" in result
        assert isinstance(result["deployed_at"], str)

    def test_deploy_service_stores_in_services(
        self, manager: DeploymentManager
    ) -> None:
        """Test deployed service is stored in services dict."""
        config = {"replicas": 3}
        manager.deploy_service("api-service", config)

        assert "api-service" in manager.services
        assert manager.services["api-service"]["service_name"] == "api-service"
        assert manager.services["api-service"]["config"] == config

    def test_deploy_multiple_services(self, manager: DeploymentManager) -> None:
        """Test deploying multiple services."""
        config1 = {"replicas": 3}
        config2 = {"replicas": 5}

        manager.deploy_service("api-service", config1)
        manager.deploy_service("worker-service", config2)

        assert len(manager.services) == 2
        assert "api-service" in manager.services
        assert "worker-service" in manager.services

    def test_deploy_service_empty_config(self, manager: DeploymentManager) -> None:
        """Test deploying service with empty config."""
        result = manager.deploy_service("minimal-service", {})

        assert result["service_name"] == "minimal-service"
        assert result["config"] == {}
        assert result["status"] == "deployed"

    def test_deploy_service_complex_config(self, manager: DeploymentManager) -> None:
        """Test deploying service with complex configuration."""
        config = {
            "replicas": 3,
            "resources": {"cpu": "2", "memory": "4Gi"},
            "env": {"DB_HOST": "localhost", "PORT": "5432"},
            "volumes": ["/data", "/logs"],
        }
        result = manager.deploy_service("complex-service", config)

        assert result["config"] == config
        assert result["config"]["resources"]["cpu"] == "2"

    @patch("traigent.security.deployment.logger")
    def test_deploy_service_logs_info(
        self, mock_logger: MagicMock, manager: DeploymentManager
    ) -> None:
        """Test deployment logs informational message."""
        manager.deploy_service("test-service", {})
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Deployed service test-service" in log_message
        assert "cloud mode" in log_message

    def test_deploy_service_preserves_mode_in_result(self) -> None:
        """Test deployment result includes correct mode for different deployment modes."""
        for mode in [
            DeploymentMode.CLOUD,
            DeploymentMode.VPC,
            DeploymentMode.ON_PREMISE,
            DeploymentMode.HYBRID,
        ]:
            manager = DeploymentManager(mode=mode)
            result = manager.deploy_service("test-service", {})
            assert result["mode"] == mode.value


class TestSLAMonitorInit:
    """Tests for SLAMonitor initialization."""

    def test_init_with_default_target(self) -> None:
        """Test initialization with default target uptime."""
        monitor = SLAMonitor()
        assert monitor.target_uptime == 99.99
        assert monitor.metrics_history == []
        assert monitor.downtime_events == []

    def test_init_with_custom_target(self) -> None:
        """Test initialization with custom target uptime."""
        monitor = SLAMonitor(target_uptime=99.95)
        assert monitor.target_uptime == 99.95


class TestSLAMonitorRecordMetrics:
    """Tests for SLAMonitor.record_metrics method."""

    @pytest.fixture
    def monitor(self) -> SLAMonitor:
        """Create test SLA monitor instance."""
        return SLAMonitor(target_uptime=99.9)

    def test_record_metrics_within_sla(self, monitor: SLAMonitor) -> None:
        """Test recording metrics that meet SLA targets."""
        metrics = monitor.record_metrics(
            uptime=99.95, response_time_p95=450.0, error_rate=0.01
        )

        assert metrics.uptime_percentage == 99.95
        assert metrics.response_time_p95_ms == 450.0
        assert metrics.error_rate_percentage == 0.01
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0] == metrics

    def test_record_metrics_below_sla(self, monitor: SLAMonitor) -> None:
        """Test recording metrics below SLA targets triggers violation."""
        metrics = monitor.record_metrics(
            uptime=99.5, response_time_p95=600.0, error_rate=0.05
        )

        assert metrics.uptime_percentage == 99.5
        assert len(monitor.downtime_events) == 1
        violation = monitor.downtime_events[0]
        assert violation["metric"] == "uptime"
        assert violation["actual_value"] == 99.5
        assert violation["target_value"] == 99.9

    def test_record_metrics_stores_in_history(self, monitor: SLAMonitor) -> None:
        """Test multiple metrics are stored in history."""
        monitor.record_metrics(uptime=99.95, response_time_p95=450.0, error_rate=0.01)
        monitor.record_metrics(uptime=99.92, response_time_p95=480.0, error_rate=0.02)

        assert len(monitor.metrics_history) == 2

    @patch("traigent.security.deployment.logger")
    def test_record_metrics_logs_info(
        self, mock_logger: MagicMock, monitor: SLAMonitor
    ) -> None:
        """Test metrics recording logs informational message."""
        monitor.record_metrics(uptime=99.95, response_time_p95=450.0, error_rate=0.01)

        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Recorded SLA metrics" in log_message
        assert "99.95%" in log_message
        assert "450.0ms" in log_message


class TestSLAMonitorRecordSLAViolation:
    """Tests for SLAMonitor._record_sla_violation method."""

    @pytest.fixture
    def monitor(self) -> SLAMonitor:
        """Create test SLA monitor instance."""
        return SLAMonitor(target_uptime=99.9)

    def test_record_sla_violation_medium_severity(self, monitor: SLAMonitor) -> None:
        """Test violation with medium severity (actual >= 95% of target)."""
        monitor._record_sla_violation("uptime", 99.85, 99.9)

        assert len(monitor.downtime_events) == 1
        violation = monitor.downtime_events[0]
        assert violation["metric"] == "uptime"
        assert violation["actual_value"] == 99.85
        assert violation["target_value"] == 99.9
        assert violation["severity"] == "medium"
        assert "violation_time" in violation

    def test_record_sla_violation_high_severity(self, monitor: SLAMonitor) -> None:
        """Test violation with high severity (actual < 95% of target)."""
        monitor._record_sla_violation("uptime", 94.0, 99.9)

        assert len(monitor.downtime_events) == 1
        violation = monitor.downtime_events[0]
        assert violation["severity"] == "high"

    def test_record_sla_violation_multiple_violations(
        self, monitor: SLAMonitor
    ) -> None:
        """Test recording multiple violations."""
        monitor._record_sla_violation("uptime", 99.85, 99.9)
        monitor._record_sla_violation("response_time", 600.0, 500.0)

        assert len(monitor.downtime_events) == 2
        assert monitor.downtime_events[0]["metric"] == "uptime"
        assert monitor.downtime_events[1]["metric"] == "response_time"

    @patch("traigent.security.deployment.logger")
    def test_record_sla_violation_logs_warning(
        self, mock_logger: MagicMock, monitor: SLAMonitor
    ) -> None:
        """Test violation logging."""
        monitor._record_sla_violation("uptime", 99.5, 99.9)

        mock_logger.warning.assert_called_once()
        log_message = mock_logger.warning.call_args[0][0]
        assert "SLA violation" in log_message
        assert "uptime" in log_message


class TestSLAMonitorGetSLAReport:
    """Tests for SLAMonitor.get_sla_report method."""

    @pytest.fixture
    def monitor(self) -> SLAMonitor:
        """Create test SLA monitor instance."""
        return SLAMonitor(target_uptime=99.9)

    def test_get_sla_report_no_metrics(self, monitor: SLAMonitor) -> None:
        """Test report generation with no metrics available."""
        report = monitor.get_sla_report(period_days=30)
        assert "error" in report
        assert report["error"] == "No metrics available for period"

    def test_get_sla_report_with_recent_metrics(self, monitor: SLAMonitor) -> None:
        """Test report generation with recent metrics."""
        monitor.record_metrics(uptime=99.95, response_time_p95=450.0, error_rate=0.01)
        monitor.record_metrics(uptime=99.92, response_time_p95=480.0, error_rate=0.02)

        report = monitor.get_sla_report(period_days=30)

        assert report["period_days"] == 30
        assert report["target_uptime"] == 99.9
        assert report["actual_uptime"] == pytest.approx((99.95 + 99.92) / 2)
        assert report["sla_compliance"] is True
        assert report["average_response_time_ms"] == pytest.approx((450.0 + 480.0) / 2)
        assert report["average_error_rate"] == pytest.approx((0.01 + 0.02) / 2)
        assert report["violations_count"] == 0
        assert report["violations"] == []

    def test_get_sla_report_with_violations(self, monitor: SLAMonitor) -> None:
        """Test report includes violations."""
        monitor.record_metrics(uptime=99.5, response_time_p95=450.0, error_rate=0.01)

        report = monitor.get_sla_report(period_days=30)

        assert report["violations_count"] == 1
        assert len(report["violations"]) == 1
        assert report["sla_compliance"] is False

    def test_get_sla_report_filters_old_metrics(self, monitor: SLAMonitor) -> None:
        """Test report filters out metrics outside the period."""
        # Create old metric
        old_metric = SLAMetrics(
            uptime_percentage=99.0,
            response_time_p95_ms=1000.0,
            error_rate_percentage=1.0,
            measured_at=datetime.now(UTC) - timedelta(days=60),
        )
        monitor.metrics_history.append(old_metric)

        # Add recent metric
        monitor.record_metrics(uptime=99.95, response_time_p95=450.0, error_rate=0.01)

        report = monitor.get_sla_report(period_days=30)

        # Should only include recent metric
        assert report["actual_uptime"] == 99.95
        assert report["average_response_time_ms"] == 450.0

    def test_get_sla_report_filters_old_violations(self, monitor: SLAMonitor) -> None:
        """Test report filters out violations outside the period."""
        # Create old violation
        old_violation = {
            "metric": "uptime",
            "actual_value": 90.0,
            "target_value": 99.9,
            "violation_time": (datetime.now(UTC) - timedelta(days=60)).isoformat(),
            "severity": "high",
        }
        monitor.downtime_events.append(old_violation)

        # Add recent violation
        monitor.record_metrics(uptime=99.5, response_time_p95=450.0, error_rate=0.01)

        report = monitor.get_sla_report(period_days=30)

        # Should only include recent violation
        assert report["violations_count"] == 1

    def test_get_sla_report_custom_period(self, monitor: SLAMonitor) -> None:
        """Test report generation with custom period."""
        monitor.record_metrics(uptime=99.95, response_time_p95=450.0, error_rate=0.01)

        report = monitor.get_sla_report(period_days=7)

        assert report["period_days"] == 7


class TestHealthCheckerInit:
    """Tests for HealthChecker initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        checker = HealthChecker()
        assert checker.health_history == []
        assert checker.check_interval == 60


class TestHealthCheckerCheckServiceHealth:
    """Tests for HealthChecker.check_service_health method."""

    @pytest.fixture
    def checker(self) -> HealthChecker:
        """Create test health checker instance."""
        return HealthChecker()

    def test_check_service_health_database(self, checker: HealthChecker) -> None:
        """Test health check for database service."""
        health = checker.check_service_health("database")

        assert health.service == "database"
        assert health.status == HealthStatus.HEALTHY
        assert health.response_time_ms > 0
        assert health.details["connections"] == 45
        assert health.details["query_time_avg_ms"] == 12
        assert isinstance(health.timestamp, datetime)

    def test_check_service_health_api(self, checker: HealthChecker) -> None:
        """Test health check for API service."""
        health = checker.check_service_health("api")

        assert health.service == "api"
        assert health.status == HealthStatus.HEALTHY
        assert health.details["active_requests"] == 23
        assert health.details["queue_length"] == 2

    def test_check_service_health_optimizer(self, checker: HealthChecker) -> None:
        """Test health check for optimizer service."""
        health = checker.check_service_health("optimizer")

        assert health.service == "optimizer"
        assert health.status == HealthStatus.HEALTHY
        assert health.details["active_optimizations"] == 5
        assert health.details["cpu_usage"] == 67

    def test_check_service_health_unknown_service(self, checker: HealthChecker) -> None:
        """Test health check for unknown service returns UNKNOWN status."""
        health = checker.check_service_health("unknown-service")

        assert health.service == "unknown-service"
        assert health.status == HealthStatus.UNKNOWN
        assert health.details == {}

    def test_check_service_health_stores_in_history(
        self, checker: HealthChecker
    ) -> None:
        """Test health check results are stored in history."""
        checker.check_service_health("database")
        checker.check_service_health("api")

        assert len(checker.health_history) == 2

    @patch("traigent.security.deployment.logger")
    def test_check_service_health_logs_debug(
        self, mock_logger: MagicMock, checker: HealthChecker
    ) -> None:
        """Test health check logs debug message."""
        checker.check_service_health("database")

        mock_logger.debug.assert_called_once()
        log_message = mock_logger.debug.call_args[0][0]
        assert "Health check for database" in log_message
        assert "healthy" in log_message

    @patch("traigent.security.deployment.time.time")
    def test_check_service_health_measures_response_time(
        self, mock_time: MagicMock, checker: HealthChecker
    ) -> None:
        """Test health check measures response time correctly."""
        # Mock time to return consistent values
        mock_time.side_effect = [1000.0, 1000.5]  # 500ms difference

        health = checker.check_service_health("database")

        # Response time should be approximately 500ms
        assert health.response_time_ms == pytest.approx(500.0, abs=1.0)

    def test_check_service_health_handles_exceptions(
        self, checker: HealthChecker
    ) -> None:
        """Test health check handles exceptions gracefully."""
        # Force an exception by mocking time.time to fail
        with patch("traigent.security.deployment.time.time") as mock_time:
            # First call starts timing, exception during check,
            # third call in exception handler
            mock_time.side_effect = [
                1000.0,
                Exception("Time error"),
                1000.5,
            ]

            health = checker.check_service_health("database")

            # Should handle the exception and return UNHEALTHY
            assert health.status == HealthStatus.UNHEALTHY
            assert "error" in health.details
            assert "Time error" in str(health.details["error"])


class TestHealthCheckerGetSystemHealth:
    """Tests for HealthChecker.get_system_health method."""

    @pytest.fixture
    def checker(self) -> HealthChecker:
        """Create test health checker instance."""
        return HealthChecker()

    def test_get_system_health_all_healthy(self, checker: HealthChecker) -> None:
        """Test system health when all services are healthy or unknown."""
        result = checker.get_system_health()

        # Note: storage service returns UNKNOWN, so overall should be degraded
        assert result["overall_status"] == "degraded"
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)
        assert "services" in result
        assert len(result["services"]) == 4  # database, api, optimizer, storage

    def test_get_system_health_includes_all_services(
        self, checker: HealthChecker
    ) -> None:
        """Test system health checks all expected services."""
        result = checker.get_system_health()

        services = result["services"]
        assert "database" in services
        assert "api" in services
        assert "optimizer" in services
        assert "storage" in services

    def test_get_system_health_service_details(self, checker: HealthChecker) -> None:
        """Test system health includes service details."""
        result = checker.get_system_health()

        db_health = result["services"]["database"]
        assert db_health["status"] == "healthy"
        assert "response_time_ms" in db_health
        assert "details" in db_health
        assert db_health["details"]["connections"] == 45

    def test_get_system_health_creates_health_history(
        self, checker: HealthChecker
    ) -> None:
        """Test system health check creates entries in history."""
        checker.get_system_health()

        # Should have 4 health checks (one per service)
        assert len(checker.health_history) == 4

    @patch.object(HealthChecker, "check_service_health")
    def test_get_system_health_degraded_status(
        self, mock_check: MagicMock, checker: HealthChecker
    ) -> None:
        """Test system health shows degraded when some services degraded."""
        # Mock health checks to return mixed statuses
        mock_check.side_effect = [
            HealthCheck(
                service="database",
                status=HealthStatus.HEALTHY,
                response_time_ms=50.0,
            ),
            HealthCheck(
                service="api", status=HealthStatus.DEGRADED, response_time_ms=100.0
            ),
            HealthCheck(
                service="optimizer",
                status=HealthStatus.HEALTHY,
                response_time_ms=75.0,
            ),
            HealthCheck(
                service="storage", status=HealthStatus.UNKNOWN, response_time_ms=200.0
            ),
        ]

        result = checker.get_system_health()
        assert result["overall_status"] == "degraded"

    @patch.object(HealthChecker, "check_service_health")
    def test_get_system_health_unhealthy_status(
        self, mock_check: MagicMock, checker: HealthChecker
    ) -> None:
        """Test system health shows unhealthy when any service unhealthy."""
        # Mock health checks to return at least one unhealthy
        mock_check.side_effect = [
            HealthCheck(
                service="database",
                status=HealthStatus.HEALTHY,
                response_time_ms=50.0,
            ),
            HealthCheck(
                service="api", status=HealthStatus.UNHEALTHY, response_time_ms=1000.0
            ),
            HealthCheck(
                service="optimizer",
                status=HealthStatus.HEALTHY,
                response_time_ms=75.0,
            ),
            HealthCheck(
                service="storage", status=HealthStatus.HEALTHY, response_time_ms=60.0
            ),
        ]

        result = checker.get_system_health()
        assert result["overall_status"] == "unhealthy"


class TestBackupManagerInit:
    """Tests for BackupManager initialization."""

    def test_init_with_default_interval(self) -> None:
        """Test initialization with default backup interval."""
        manager = BackupManager()
        assert manager.backup_interval == 24
        assert manager.backups == []
        assert manager.retention_days == 30

    def test_init_with_custom_interval(self) -> None:
        """Test initialization with custom backup interval."""
        manager = BackupManager(backup_interval_hours=12)
        assert manager.backup_interval == 12


class TestBackupManagerCreateBackup:
    """Tests for BackupManager.create_backup method."""

    @pytest.fixture
    def manager(self) -> BackupManager:
        """Create test backup manager instance."""
        return BackupManager()

    def test_create_backup_with_default_type(self, manager: BackupManager) -> None:
        """Test creating backup with default full type."""
        backup = manager.create_backup()

        assert backup["type"] == "full"
        assert backup["status"] == "completed"
        assert backup["size_gb"] == 2.5
        assert "backup_id" in backup
        assert backup["backup_id"].startswith("backup_")
        assert "created_at" in backup
        assert "retention_until" in backup

    def test_create_backup_with_custom_type(self, manager: BackupManager) -> None:
        """Test creating backup with custom type."""
        backup = manager.create_backup(backup_type="incremental")

        assert backup["type"] == "incremental"

    def test_create_backup_stores_in_backups_list(self, manager: BackupManager) -> None:
        """Test created backup is stored in backups list."""
        manager.create_backup()

        assert len(manager.backups) == 1
        assert manager.backups[0]["type"] == "full"

    def test_create_backup_multiple_backups(self, manager: BackupManager) -> None:
        """Test creating multiple backups."""
        manager.create_backup(backup_type="full")
        manager.create_backup(backup_type="incremental")

        assert len(manager.backups) == 2

    @patch("traigent.security.deployment.time.time")
    def test_create_backup_unique_ids(
        self, mock_time: MagicMock, manager: BackupManager
    ) -> None:
        """Test each backup gets unique ID."""
        # Use an unbounded counter so unrelated background calls to time.time
        # in this module do not exhaust side effects and cause StopIteration.
        mock_time.side_effect = (float(ts) for ts in count(start=1000, step=1))

        backup1 = manager.create_backup()
        backup2 = manager.create_backup()

        assert backup1["backup_id"] != backup2["backup_id"]
        assert backup1["backup_id"].startswith("backup_")
        assert backup2["backup_id"].startswith("backup_")

    def test_create_backup_retention_period(self, manager: BackupManager) -> None:
        """Test backup retention period is set correctly."""
        backup = manager.create_backup()

        created_at = datetime.fromisoformat(backup["created_at"])
        retention_until = datetime.fromisoformat(backup["retention_until"])
        expected_retention = created_at + timedelta(days=30)

        # Allow small time difference due to processing
        assert abs((retention_until - expected_retention).total_seconds()) < 1

    @patch("traigent.security.deployment.logger")
    def test_create_backup_logs_info(
        self, mock_logger: MagicMock, manager: BackupManager
    ) -> None:
        """Test backup creation logs informational message."""
        manager.create_backup(backup_type="full")

        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Created full backup" in log_message


class TestBackupManagerListBackups:
    """Tests for BackupManager.list_backups method."""

    @pytest.fixture
    def manager(self) -> BackupManager:
        """Create test backup manager instance."""
        return BackupManager()

    def test_list_backups_empty(self, manager: BackupManager) -> None:
        """Test listing backups when none exist."""
        backups = manager.list_backups()
        assert backups == []

    def test_list_backups_with_valid_backups(self, manager: BackupManager) -> None:
        """Test listing valid backups."""
        manager.create_backup()
        manager.create_backup()

        backups = manager.list_backups()
        assert len(backups) == 2

    def test_list_backups_filters_expired(self, manager: BackupManager) -> None:
        """Test list filters out expired backups."""
        # Create expired backup
        expired_backup = {
            "backup_id": "backup_expired",
            "type": "full",
            "created_at": (datetime.now(UTC) - timedelta(days=60)).isoformat(),
            "size_gb": 2.5,
            "status": "completed",
            "retention_until": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        }
        manager.backups.append(expired_backup)

        # Create valid backup
        manager.create_backup()

        backups = manager.list_backups()
        assert len(backups) == 1
        assert backups[0]["backup_id"] != "backup_expired"


class TestBackupManagerRestoreBackup:
    """Tests for BackupManager.restore_backup method."""

    @pytest.fixture
    def manager(self) -> BackupManager:
        """Create test backup manager instance."""
        return BackupManager()

    def test_restore_backup_existing(self, manager: BackupManager) -> None:
        """Test restoring from existing backup."""
        backup = manager.create_backup()
        backup_id = backup["backup_id"]

        restore = manager.restore_backup(backup_id)

        assert restore["backup_id"] == backup_id
        assert restore["status"] == "in_progress"
        assert restore["estimated_duration_minutes"] == 30
        assert "restore_started_at" in restore

    def test_restore_backup_not_found(self, manager: BackupManager) -> None:
        """Test restoring from non-existent backup returns error."""
        restore = manager.restore_backup("nonexistent_backup")

        assert "error" in restore
        assert "not found" in restore["error"]

    @patch("traigent.security.deployment.logger")
    def test_restore_backup_logs_info(
        self, mock_logger: MagicMock, manager: BackupManager
    ) -> None:
        """Test restore operation logs informational message."""
        backup = manager.create_backup()
        manager.restore_backup(backup["backup_id"])

        mock_logger.info.assert_called()
        # Find the restore log message (not the create backup message)
        restore_log_found = False
        for call in mock_logger.info.call_args_list:
            log_message = call[0][0]
            if "Started restore from backup" in log_message:
                restore_log_found = True
                break
        assert restore_log_found


class TestBackupManagerCleanupExpiredBackups:
    """Tests for BackupManager.cleanup_expired_backups method."""

    @pytest.fixture
    def manager(self) -> BackupManager:
        """Create test backup manager instance."""
        return BackupManager()

    def test_cleanup_expired_backups_none_expired(self, manager: BackupManager) -> None:
        """Test cleanup when no backups are expired."""
        manager.create_backup()
        manager.create_backup()

        cleaned = manager.cleanup_expired_backups()

        assert cleaned == 0
        assert len(manager.backups) == 2

    def test_cleanup_expired_backups_some_expired(self, manager: BackupManager) -> None:
        """Test cleanup removes only expired backups."""
        # Create expired backup
        expired_backup = {
            "backup_id": "backup_expired",
            "type": "full",
            "created_at": (datetime.now(UTC) - timedelta(days=60)).isoformat(),
            "size_gb": 2.5,
            "status": "completed",
            "retention_until": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        }
        manager.backups.append(expired_backup)

        # Create valid backup
        manager.create_backup()

        cleaned = manager.cleanup_expired_backups()

        assert cleaned == 1
        assert len(manager.backups) == 1
        assert manager.backups[0]["backup_id"] != "backup_expired"

    def test_cleanup_expired_backups_all_expired(self, manager: BackupManager) -> None:
        """Test cleanup when all backups are expired."""
        # Create multiple expired backups
        for i in range(3):
            expired_backup = {
                "backup_id": f"backup_expired_{i}",
                "type": "full",
                "created_at": (datetime.now(UTC) - timedelta(days=60 + i)).isoformat(),
                "size_gb": 2.5,
                "status": "completed",
                "retention_until": (
                    datetime.now(UTC) - timedelta(days=1 + i)
                ).isoformat(),
            }
            manager.backups.append(expired_backup)

        cleaned = manager.cleanup_expired_backups()

        assert cleaned == 3
        assert len(manager.backups) == 0

    @patch("traigent.security.deployment.logger")
    def test_cleanup_expired_backups_logs_info(
        self, mock_logger: MagicMock, manager: BackupManager
    ) -> None:
        """Test cleanup logs info when backups are cleaned."""
        # Create expired backup
        expired_backup = {
            "backup_id": "backup_expired",
            "type": "full",
            "created_at": (datetime.now(UTC) - timedelta(days=60)).isoformat(),
            "size_gb": 2.5,
            "status": "completed",
            "retention_until": (datetime.now(UTC) - timedelta(days=1)).isoformat(),
        }
        manager.backups.append(expired_backup)

        manager.cleanup_expired_backups()

        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "Cleaned up 1 expired backups" in log_message

    @patch("traigent.security.deployment.logger")
    def test_cleanup_expired_backups_no_log_when_none_cleaned(
        self, mock_logger: MagicMock, manager: BackupManager
    ) -> None:
        """Test cleanup doesn't log when no backups are cleaned."""
        manager.create_backup()

        manager.cleanup_expired_backups()

        # Should not call logger.info for cleanup (only for create_backup)
        assert mock_logger.info.call_count == 1  # Only from create_backup
