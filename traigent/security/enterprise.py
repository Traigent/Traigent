"""
Enterprise Deployment Features

Implements enterprise-grade deployment and operational features including:
- VPC integration and private cloud deployments
- Enterprise SLAs and monitoring
- High availability and disaster recovery
- Performance monitoring and optimization
- Security hardening and compliance
"""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Reliability FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import psutil

logger = logging.getLogger(__name__)


class DeploymentMode(Enum):
    """Enterprise deployment modes"""

    CLOUD_PUBLIC = "cloud_public"
    CLOUD_PRIVATE = "cloud_private"
    VPC_DEDICATED = "vpc_dedicated"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"


class SLATier(Enum):
    """Service Level Agreement tiers"""

    BASIC = "basic"  # 99.5% uptime
    STANDARD = "standard"  # 99.9% uptime
    PREMIUM = "premium"  # 99.95% uptime
    ENTERPRISE = "enterprise"  # 99.99% uptime


class HealthStatus(Enum):
    """System health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class SLAConfiguration:
    """SLA configuration and metrics"""

    tier: SLATier
    uptime_target: float  # Percentage
    response_time_target_ms: int
    throughput_target_rps: int  # Requests per second
    error_rate_target: float  # Percentage

    # Contact and escalation
    primary_contact: str
    escalation_contacts: list[str] = field(default_factory=list)

    # Notification settings
    alert_thresholds: dict[str, float] = field(default_factory=dict)
    notification_channels: list[str] = field(default_factory=list)

    @classmethod
    def from_tier(cls, tier: SLATier, primary_contact: str) -> SLAConfiguration:
        """Create SLA configuration from tier"""
        configs = {
            SLATier.BASIC: cls(
                tier=tier,
                uptime_target=99.5,
                response_time_target_ms=2000,
                throughput_target_rps=100,
                error_rate_target=5.0,
                primary_contact=primary_contact,
                alert_thresholds={
                    "uptime": 99.0,
                    "response_time": 3000,
                    "error_rate": 7.0,
                },
            ),
            SLATier.STANDARD: cls(
                tier=tier,
                uptime_target=99.9,
                response_time_target_ms=1000,
                throughput_target_rps=500,
                error_rate_target=2.0,
                primary_contact=primary_contact,
                alert_thresholds={
                    "uptime": 99.5,
                    "response_time": 1500,
                    "error_rate": 3.0,
                },
            ),
            SLATier.PREMIUM: cls(
                tier=tier,
                uptime_target=99.95,
                response_time_target_ms=500,
                throughput_target_rps=1000,
                error_rate_target=1.0,
                primary_contact=primary_contact,
                alert_thresholds={
                    "uptime": 99.8,
                    "response_time": 750,
                    "error_rate": 1.5,
                },
            ),
            SLATier.ENTERPRISE: cls(
                tier=tier,
                uptime_target=99.99,
                response_time_target_ms=250,
                throughput_target_rps=5000,
                error_rate_target=0.1,
                primary_contact=primary_contact,
                alert_thresholds={
                    "uptime": 99.95,
                    "response_time": 500,
                    "error_rate": 0.5,
                },
            ),
        }
        return configs.get(tier, configs[SLATier.BASIC])


@dataclass
class SystemMetrics:
    """System performance and health metrics"""

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # System resources
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0

    # Application metrics
    active_connections: int = 0
    requests_per_second: float = 0.0
    average_response_time_ms: float = 0.0
    error_rate_percent: float = 0.0

    # TraiGent specific metrics
    active_optimizations: int = 0
    completed_optimizations_today: int = 0
    queue_depth: int = 0

    # Health indicators
    health_status: HealthStatus = HealthStatus.HEALTHY
    health_score: float = 100.0  # 0-100 scale

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "system": {
                "cpu_usage_percent": self.cpu_usage_percent,
                "memory_usage_percent": self.memory_usage_percent,
                "disk_usage_percent": self.disk_usage_percent,
                "network_bytes_sent": self.network_bytes_sent,
                "network_bytes_recv": self.network_bytes_recv,
            },
            "application": {
                "active_connections": self.active_connections,
                "requests_per_second": self.requests_per_second,
                "average_response_time_ms": self.average_response_time_ms,
                "error_rate_percent": self.error_rate_percent,
            },
            "traigent": {
                "active_optimizations": self.active_optimizations,
                "completed_optimizations_today": self.completed_optimizations_today,
                "queue_depth": self.queue_depth,
            },
            "health": {"status": self.health_status.value, "score": self.health_score},
        }


class MetricsCollector:
    """Collects system and application metrics"""

    def __init__(self) -> None:
        self.start_time = datetime.utcnow()
        self.request_counts: list[Any] = []
        self.response_times: list[Any] = []
        self.error_counts: list[Any] = []
        self.optuna_events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net_io = psutil.net_io_counters()

            # Calculate application metrics
            now = datetime.utcnow()

            with self._lock:
                # RPS calculation (last minute)
                recent_requests = [
                    t for t in self.request_counts if now - t < timedelta(minutes=1)
                ]
                rps = len(recent_requests) / 60.0

                # Average response time (last 100 requests)
                avg_response_time = sum(self.response_times[-100:]) / max(
                    len(self.response_times[-100:]), 1
                )

                # Error rate (last hour)
                recent_errors = [
                    t for t in self.error_counts if now - t < timedelta(hours=1)
                ]
                total_requests = len(
                    [t for t in self.request_counts if now - t < timedelta(hours=1)]
                )
                error_rate = (len(recent_errors) / max(total_requests, 1)) * 100

            # Calculate health score
            health_score = self._calculate_health_score(
                cpu_percent,
                memory.percent,
                disk.percent,
                rps,
                avg_response_time,
                error_rate,
            )

            # Determine health status
            if health_score >= 90:
                health_status = HealthStatus.HEALTHY
            elif health_score >= 70:
                health_status = HealthStatus.DEGRADED
            elif health_score >= 50:
                health_status = HealthStatus.UNHEALTHY
            else:
                health_status = HealthStatus.CRITICAL

            return SystemMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                requests_per_second=rps,
                average_response_time_ms=avg_response_time,
                error_rate_percent=error_rate,
                health_status=health_status,
                health_score=health_score,
            )

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return SystemMetrics(health_status=HealthStatus.CRITICAL, health_score=0.0)

    def _calculate_health_score(
        self,
        cpu: float,
        memory: float,
        disk: float,
        rps: float,
        response_time: float,
        error_rate: float,
    ) -> float:
        """Calculate overall health score"""
        # Weight factors
        weights = {
            "cpu": 0.2,
            "memory": 0.2,
            "disk": 0.1,
            "response_time": 0.3,
            "error_rate": 0.2,
        }

        # Component scores (0-100)
        cpu_score = max(0, 100 - cpu)
        memory_score = max(0, 100 - memory)
        disk_score = max(0, 100 - disk)
        response_time_score = max(0, 100 - (response_time / 10))  # 1000ms = 0 score
        error_rate_score = max(0, 100 - (error_rate * 10))  # 10% error = 0 score

        # Weighted average
        health_score = (
            cpu_score * weights["cpu"]
            + memory_score * weights["memory"]
            + disk_score * weights["disk"]
            + response_time_score * weights["response_time"]
            + error_rate_score * weights["error_rate"]
        )

        return round(health_score, 1)

    def record_request(self, response_time_ms: float, is_error: bool = False) -> None:
        """Record request metrics"""
        now = datetime.utcnow()

        with self._lock:
            self.request_counts.append(now)
            self.response_times.append(response_time_ms)

            if is_error:
                self.error_counts.append(now)

            # Clean old data (keep last 24 hours)
            cutoff = now - timedelta(hours=24)
            self.request_counts = [t for t in self.request_counts if t > cutoff]
            self.response_times = self.response_times[-10000:]  # Keep last 10k
            self.error_counts = [t for t in self.error_counts if t > cutoff]

    def record_optuna_event(self, event: str, payload: dict[str, Any]) -> None:
        """Record an Optuna optimizer telemetry event."""

        enriched = {"event": event, **payload}
        with self._lock:
            self.optuna_events.append(enriched)
            self.optuna_events = self.optuna_events[-1000:]

    def get_optuna_events(self) -> list[dict[str, Any]]:
        """Return a snapshot of recorded Optuna events."""

        with self._lock:
            return list(self.optuna_events)


class SLAMonitor:
    """Monitors SLA compliance and alerts.

    Thread Safety:
        This class is thread-safe. Uses an RLock (_lock) to protect:
        - alert_handlers list (add/iteration)
        - sla_history list (append/read)
        The monitoring loop runs in a separate daemon thread.
    """

    def __init__(
        self, sla_config: SLAConfiguration, metrics_collector: MetricsCollector
    ) -> None:
        self.sla_config = sla_config
        self.metrics_collector = metrics_collector
        self.alert_handlers: list[Callable[[str, dict[str, Any]], None]] = []
        self.sla_history: list[dict[str, Any]] = []
        self.running = False
        self.monitor_thread: threading.Thread | None = None
        self._lock = threading.RLock()

    def add_alert_handler(self, handler: Callable[[str, dict[str, Any]], None]) -> None:
        """Add alert handler (thread-safe)."""
        with self._lock:
            self.alert_handlers.append(handler)

    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """Start SLA monitoring"""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval_seconds,), daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started SLA monitoring")

    def stop_monitoring(self) -> None:
        """Stop SLA monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Stopped SLA monitoring")

    def _monitor_loop(self, interval_seconds: int) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self.metrics_collector.collect_system_metrics()
                sla_status = self._check_sla_compliance(metrics)

                # Store SLA history
                with self._lock:
                    self.sla_history.append(
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "metrics": metrics.to_dict(),
                            "sla_status": sla_status,
                        }
                    )

                    # Keep last 24 hours of history
                    cutoff = datetime.utcnow() - timedelta(hours=24)
                    self.sla_history = [
                        h
                        for h in self.sla_history
                        if datetime.fromisoformat(h["timestamp"]) > cutoff
                    ]

                # Check for alerts
                self._check_alerts(metrics, sla_status)

                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in SLA monitoring loop: {e}")
                time.sleep(interval_seconds)

    def _check_sla_compliance(self, metrics: SystemMetrics) -> dict[str, Any]:
        """Check SLA compliance for current metrics"""

        # Calculate uptime percentage (simplified)
        uptime_percentage = (
            100.0 if metrics.health_status != HealthStatus.CRITICAL else 0.0
        )

        return {
            "uptime": {
                "current": uptime_percentage,
                "target": self.sla_config.uptime_target,
                "compliant": uptime_percentage >= self.sla_config.uptime_target,
            },
            "response_time": {
                "current": metrics.average_response_time_ms,
                "target": self.sla_config.response_time_target_ms,
                "compliant": metrics.average_response_time_ms
                <= self.sla_config.response_time_target_ms,
            },
            "throughput": {
                "current": metrics.requests_per_second,
                "target": self.sla_config.throughput_target_rps,
                "compliant": metrics.requests_per_second
                >= self.sla_config.throughput_target_rps,
            },
            "error_rate": {
                "current": metrics.error_rate_percent,
                "target": self.sla_config.error_rate_target,
                "compliant": metrics.error_rate_percent
                <= self.sla_config.error_rate_target,
            },
        }

    def _check_alerts(self, metrics: SystemMetrics, sla_status: dict[str, Any]) -> None:
        """Check for alert conditions"""
        alerts = []

        # Check health status
        if metrics.health_status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
            alerts.append(
                {
                    "type": "health_critical",
                    "message": f"System health is {metrics.health_status.value}",
                    "details": {
                        "health_score": metrics.health_score,
                        "status": metrics.health_status.value,
                    },
                }
            )

        # Check SLA violations
        for metric_name, status in sla_status.items():
            if not status["compliant"]:
                alerts.append(
                    {
                        "type": "sla_violation",
                        "message": f"SLA violation: {metric_name}",
                        "details": status,
                    }
                )

        # Check against alert thresholds
        thresholds = self.sla_config.alert_thresholds

        if metrics.average_response_time_ms > thresholds.get(
            "response_time", float("inf")
        ):
            alerts.append(
                {
                    "type": "response_time_alert",
                    "message": f"Response time {metrics.average_response_time_ms}ms exceeds threshold",
                    "details": {
                        "current": metrics.average_response_time_ms,
                        "threshold": thresholds.get("response_time"),
                    },
                }
            )

        if metrics.error_rate_percent > thresholds.get("error_rate", float("inf")):
            alerts.append(
                {
                    "type": "error_rate_alert",
                    "message": f"Error rate {metrics.error_rate_percent}% exceeds threshold",
                    "details": {
                        "current": metrics.error_rate_percent,
                        "threshold": thresholds.get("error_rate"),
                    },
                }
            )

        # Send alerts
        for alert in alerts:
            alert_type = str(alert["type"])  # Ensure string type
            self._send_alert(alert_type, alert)

    def _send_alert(self, alert_type: str, alert_data: dict[str, Any]) -> None:
        """Send alert to handlers (thread-safe)."""
        # Copy handlers under lock to avoid race with add_alert_handler
        with self._lock:
            handlers = list(self.alert_handlers)
        for handler in handlers:
            try:
                handler(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Error sending alert: {e}")

    def get_sla_report(self, hours: int = 24) -> dict[str, Any]:
        """Get SLA compliance report"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        with self._lock:
            recent_history = [
                h
                for h in self.sla_history
                if datetime.fromisoformat(h["timestamp"]) > cutoff
            ]

        if not recent_history:
            return {
                "period_hours": hours,
                "data_points": 0,
                "sla_compliance": "insufficient_data",
            }

        # Calculate averages
        total_points = len(recent_history)

        uptime_violations = sum(
            1 for h in recent_history if not h["sla_status"]["uptime"]["compliant"]
        )
        response_time_violations = sum(
            1
            for h in recent_history
            if not h["sla_status"]["response_time"]["compliant"]
        )
        error_rate_violations = sum(
            1 for h in recent_history if not h["sla_status"]["error_rate"]["compliant"]
        )

        avg_response_time = (
            sum(
                h["metrics"]["application"]["average_response_time_ms"]
                for h in recent_history
            )
            / total_points
        )
        avg_error_rate = (
            sum(
                h["metrics"]["application"]["error_rate_percent"]
                for h in recent_history
            )
            / total_points
        )

        return {
            "period_hours": hours,
            "data_points": total_points,
            "sla_targets": {
                "uptime": self.sla_config.uptime_target,
                "response_time_ms": self.sla_config.response_time_target_ms,
                "error_rate": self.sla_config.error_rate_target,
            },
            "compliance": {
                "uptime": {
                    "violations": uptime_violations,
                    "percentage": ((total_points - uptime_violations) / total_points)
                    * 100,
                },
                "response_time": {
                    "violations": response_time_violations,
                    "average_ms": avg_response_time,
                    "percentage": (
                        (total_points - response_time_violations) / total_points
                    )
                    * 100,
                },
                "error_rate": {
                    "violations": error_rate_violations,
                    "average_percent": avg_error_rate,
                    "percentage": (
                        (total_points - error_rate_violations) / total_points
                    )
                    * 100,
                },
            },
            "overall_compliance": (
                "compliant"
                if (
                    uptime_violations + response_time_violations + error_rate_violations
                )
                == 0
                else "violations_detected"
            ),
        }


class EnterpriseDeploymentManager:
    """Manages enterprise deployment features"""

    def __init__(
        self, deployment_mode: DeploymentMode = DeploymentMode.CLOUD_PUBLIC
    ) -> None:
        self.deployment_mode = deployment_mode
        self.metrics_collector = MetricsCollector()
        self.sla_monitor: SLAMonitor | None = None
        self.config: dict[str, Any] = {}

        # Load deployment configuration
        self._load_deployment_config()

    def _load_deployment_config(self) -> None:
        """Load deployment-specific configuration"""
        config_file = f"enterprise_config_{self.deployment_mode.value}.json"

        # Default configurations for different deployment modes
        default_configs = {
            DeploymentMode.CLOUD_PUBLIC: {
                "load_balancer_enabled": True,
                "auto_scaling_enabled": True,
                "backup_retention_days": 30,
                "monitoring_interval_seconds": 60,
                "log_level": "INFO",
            },
            DeploymentMode.VPC_DEDICATED: {
                "load_balancer_enabled": True,
                "auto_scaling_enabled": True,
                "backup_retention_days": 90,
                "monitoring_interval_seconds": 30,
                "log_level": "INFO",
                "vpc_config": {
                    "subnet_id": "subnet-12345",
                    "security_group_id": "sg-12345",
                },
            },
            DeploymentMode.ON_PREMISE: {
                "load_balancer_enabled": False,
                "auto_scaling_enabled": False,
                "backup_retention_days": 365,
                "monitoring_interval_seconds": 30,
                "log_level": "DEBUG",
                "on_premise_config": {
                    "data_center": "primary",
                    "cluster_nodes": ["node1", "node2", "node3"],
                },
            },
        }

        self.config = default_configs.get(
            self.deployment_mode, default_configs[DeploymentMode.CLOUD_PUBLIC]
        )

        # Try to load from file if exists
        try:
            if os.path.exists(config_file):
                with open(config_file) as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
                logger.info(f"Loaded deployment config from {config_file}")
        except Exception as e:
            logger.warning(f"Could not load config file {config_file}: {e}")

    def setup_sla_monitoring(self, sla_tier: SLATier, primary_contact: str) -> None:
        """Setup SLA monitoring"""
        sla_config = SLAConfiguration.from_tier(sla_tier, primary_contact)
        self.sla_monitor = SLAMonitor(sla_config, self.metrics_collector)

        # Add default alert handler
        self.sla_monitor.add_alert_handler(self._default_alert_handler)

        # Start monitoring
        interval = self.config.get("monitoring_interval_seconds", 60)
        self.sla_monitor.start_monitoring(interval)

        logger.info(f"Setup SLA monitoring for {sla_tier.value} tier")

    def _default_alert_handler(
        self, alert_type: str, alert_data: dict[str, Any]
    ) -> None:
        """Default alert handler (logs alerts)"""
        logger.warning(f"SLA Alert [{alert_type}]: {alert_data['message']}")

        # In production, this would integrate with:
        # - Email/SMS notifications
        # - Slack/Teams webhooks
        # - PagerDuty/Opsgenie
        # - Custom alerting systems

    def get_deployment_status(self) -> dict[str, Any]:
        """Get current deployment status"""
        metrics = self.metrics_collector.collect_system_metrics()

        status = {
            "deployment_mode": self.deployment_mode.value,
            "health_status": metrics.health_status.value,
            "health_score": metrics.health_score,
            "uptime_seconds": (
                datetime.utcnow() - self.metrics_collector.start_time
            ).total_seconds(),
            "metrics": metrics.to_dict(),
            "config": self.config,
        }

        # Add SLA status if monitoring is enabled
        if self.sla_monitor:
            status["sla_report"] = self.sla_monitor.get_sla_report(hours=1)

        return status

    def perform_health_check(self) -> dict[str, Any]:
        """Perform comprehensive health check"""
        health_checks = {}

        # System health
        try:
            metrics = self.metrics_collector.collect_system_metrics()
            health_checks["system"] = {
                "status": "healthy" if metrics.health_score > 70 else "unhealthy",
                "score": metrics.health_score,
                "details": metrics.to_dict(),
            }
        except Exception as e:
            health_checks["system"] = {"status": "error", "error": str(e)}

        # Database connectivity (mock)
        health_checks["database"] = {"status": "healthy", "response_time_ms": 5}

        # External services (mock)
        health_checks["external_services"] = {
            "status": "healthy",
            "services_checked": ",".join(
                [
                    "auth_service",
                    "billing_service",
                    "notification_service",
                ]
            ),
        }

        # Overall status
        all_healthy = all(
            check.get("status") == "healthy" for check in health_checks.values()
        )

        return {
            "overall_status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": health_checks,
        }

    def create_backup(self) -> dict[str, Any]:
        """Create system backup"""
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        # In production, this would:
        # - Backup database
        # - Backup configuration files
        # - Backup user data
        # - Store in secure location

        backup_info = {
            "backup_id": backup_id,
            "timestamp": datetime.utcnow().isoformat(),
            "deployment_mode": self.deployment_mode.value,
            "size_bytes": 1024 * 1024 * 100,  # Mock 100MB
            "retention_until": (
                datetime.utcnow()
                + timedelta(days=self.config.get("backup_retention_days", 30))
            ).isoformat(),
            "status": "completed",
        }

        logger.info(f"Created backup {backup_id}")
        return backup_info

    def get_enterprise_dashboard(self) -> dict[str, Any]:
        """Get enterprise dashboard data"""
        metrics = self.metrics_collector.collect_system_metrics()

        dashboard = {
            "timestamp": datetime.utcnow().isoformat(),
            "deployment": {
                "mode": self.deployment_mode.value,
                "uptime_hours": (
                    datetime.utcnow() - self.metrics_collector.start_time
                ).total_seconds()
                / 3600,
                "config": self.config,
            },
            "health": {
                "status": metrics.health_status.value,
                "score": metrics.health_score,
            },
            "performance": {
                "cpu_usage": metrics.cpu_usage_percent,
                "memory_usage": metrics.memory_usage_percent,
                "disk_usage": metrics.disk_usage_percent,
                "requests_per_second": metrics.requests_per_second,
                "response_time_ms": metrics.average_response_time_ms,
                "error_rate": metrics.error_rate_percent,
            },
            "capacity": {
                "active_optimizations": metrics.active_optimizations,
                "queue_depth": metrics.queue_depth,
                "completed_today": metrics.completed_optimizations_today,
            },
        }

        # Add SLA information if available
        if self.sla_monitor:
            dashboard["sla"] = self.sla_monitor.get_sla_report(hours=24)

        return dashboard

    def shutdown(self) -> None:
        """Graceful shutdown"""
        if self.sla_monitor:
            self.sla_monitor.stop_monitoring()

        logger.info("Enterprise deployment manager shutdown completed")
