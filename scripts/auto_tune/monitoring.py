#!/usr/bin/env python3
"""
Monitoring and alerting system for auto-tuning pipeline.
Provides health checks, metrics collection, and alerting.
"""

import ipaddress
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import psutil
import requests

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from security_utils import AuditLogger, safe_file_read, safe_file_write, setup_logging

# Initialize logging
logger = setup_logging(__name__, "monitoring.log")
audit_logger = AuditLogger("monitoring_audit.jsonl")


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check result."""

    name: str
    status: HealthStatus
    message: str
    timestamp: str
    details: Dict[str, Any] = None


@dataclass
class Metric:
    """Performance metric."""

    name: str
    value: float
    unit: str
    timestamp: str
    tags: Dict[str, str] = None


@dataclass
class Alert:
    """Alert notification."""

    severity: AlertSeverity
    title: str
    message: str
    timestamp: str
    details: Dict[str, Any] = None


class SystemMonitor:
    """System resource monitor."""

    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__)

    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        return psutil.cpu_percent(interval=1)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "used_gb": mem.used / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent": mem.percent,
        }

    def get_disk_usage(self, path: str = "/") -> Dict[str, float]:
        """Get disk usage statistics."""
        disk = psutil.disk_usage(path)
        return {
            "total_gb": disk.total / (1024**3),
            "used_gb": disk.used / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent": disk.percent,
        }

    def check_system_health(self) -> HealthCheck:
        """Check overall system health."""
        cpu = self.get_cpu_usage()
        memory = self.get_memory_usage()
        disk = self.get_disk_usage()

        # Determine health status
        if cpu > 90 or memory["percent"] > 90 or disk["percent"] > 90:
            status = HealthStatus.CRITICAL
            message = "System resources critically high"
        elif cpu > 80 or memory["percent"] > 80 or disk["percent"] > 80:
            status = HealthStatus.UNHEALTHY
            message = "System resources high"
        elif cpu > 70 or memory["percent"] > 70 or disk["percent"] > 70:
            status = HealthStatus.DEGRADED
            message = "System resources elevated"
        else:
            status = HealthStatus.HEALTHY
            message = "System resources normal"

        return HealthCheck(
            name="system_resources",
            status=status,
            message=message,
            timestamp=datetime.now().isoformat(),
            details={"cpu_percent": cpu, "memory": memory, "disk": disk},
        )


class PipelineMonitor:
    """Pipeline execution monitor."""

    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__)
        self.metrics_file = Path("metrics.json")

    def check_dvc_status(self) -> HealthCheck:
        """Check DVC pipeline status."""
        try:
            # Check if DVC is initialized
            if not Path(".dvc").exists():
                return HealthCheck(
                    name="dvc_pipeline",
                    status=HealthStatus.UNHEALTHY,
                    message="DVC not initialized",
                    timestamp=datetime.now().isoformat(),
                )

            # Check for pipeline file
            if not Path("dvc.yaml").exists():
                return HealthCheck(
                    name="dvc_pipeline",
                    status=HealthStatus.DEGRADED,
                    message="No pipeline defined",
                    timestamp=datetime.now().isoformat(),
                )

            return HealthCheck(
                name="dvc_pipeline",
                status=HealthStatus.HEALTHY,
                message="DVC pipeline configured",
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            logger.error(f"Error checking DVC status: {e}")
            return HealthCheck(
                name="dvc_pipeline",
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}",
                timestamp=datetime.now().isoformat(),
            )

    def check_optimization_status(self) -> HealthCheck:
        """Check optimization results status."""
        results_file = Path("optimization_results/latest.json")

        if not results_file.exists():
            return HealthCheck(
                name="optimization",
                status=HealthStatus.DEGRADED,
                message="No optimization results found",
                timestamp=datetime.now().isoformat(),
            )

        try:
            # Check file age
            file_age = time.time() - results_file.stat().st_mtime

            if file_age > 86400:  # Older than 24 hours
                status = HealthStatus.DEGRADED
                message = "Optimization results are stale"
            else:
                status = HealthStatus.HEALTHY
                message = "Optimization results are current"

            # Load and check results
            content = safe_file_read(results_file)
            if content:
                data = json.loads(content)
                trials = data.get("trials", [])

                return HealthCheck(
                    name="optimization",
                    status=status,
                    message=message,
                    timestamp=datetime.now().isoformat(),
                    details={
                        "trials_count": len(trials),
                        "file_age_hours": file_age / 3600,
                    },
                )

        except Exception as e:
            logger.error(f"Error checking optimization status: {e}")

        return HealthCheck(
            name="optimization",
            status=HealthStatus.UNHEALTHY,
            message="Failed to check optimization status",
            timestamp=datetime.now().isoformat(),
        )

    def collect_pipeline_metrics(self) -> List[Metric]:
        """Collect pipeline execution metrics."""
        metrics = []

        # Check performance report
        perf_file = Path("performance_report.json")
        if perf_file.exists():
            try:
                content = safe_file_read(perf_file)
                if content:
                    data = json.loads(content)
                    current = data.get("current_metrics", {})

                    metrics.extend(
                        [
                            Metric(
                                name="accuracy",
                                value=current.get("accuracy", 0),
                                unit="ratio",
                                timestamp=datetime.now().isoformat(),
                            ),
                            Metric(
                                name="latency",
                                value=current.get("avg_latency", 0),
                                unit="seconds",
                                timestamp=datetime.now().isoformat(),
                            ),
                            Metric(
                                name="cost",
                                value=current.get("total_cost", 0),
                                unit="usd",
                                timestamp=datetime.now().isoformat(),
                            ),
                            Metric(
                                name="trials",
                                value=current.get("trials_completed", 0),
                                unit="count",
                                timestamp=datetime.now().isoformat(),
                            ),
                        ]
                    )
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

        return metrics


class AlertManager:
    """Alert management and notification system."""

    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__)
        self.alerts: List[Alert] = []
        self.alert_file = Path("alerts.jsonl")

    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now().isoformat(),
            details=details or {},
        )

        self.alerts.append(alert)
        self._save_alert(alert)
        self._send_notifications(alert)

        return alert

    def _save_alert(self, alert: Alert):
        """Save alert to file."""
        try:
            existing = safe_file_read(self.alert_file) or ""
            new_line = json.dumps(asdict(alert)) + "\n"
            safe_file_write(self.alert_file, existing + new_line, backup=False)
        except Exception as e:
            logger.error(f"Failed to save alert: {e}")

    def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        # Log alert
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(f"{alert.title}: {alert.message}")
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(f"{alert.title}: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(f"{alert.title}: {alert.message}")
        else:
            logger.info(f"{alert.title}: {alert.message}")

        # Send to webhook if configured
        webhook_url = os.environ.get("ALERT_WEBHOOK")
        if webhook_url:
            self._send_webhook(webhook_url, alert)

    def _send_webhook(self, url: str, alert: Alert):
        """Send alert to webhook."""
        try:
            if not self._is_safe_webhook_url(url):
                logger.error("Refusing to send webhook to unsafe URL: %s", url)
                return
            payload = {
                "text": f"{alert.severity.value.upper()}: {alert.title}",
                "attachments": [
                    {
                        "color": self._get_alert_color(alert.severity),
                        "text": alert.message,
                        "fields": [
                            {"title": k, "value": str(v), "short": True}
                            for k, v in (alert.details or {}).items()
                        ],
                        "timestamp": alert.timestamp,
                    }
                ],
            }

            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()

        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

    @staticmethod
    def _is_safe_webhook_url(url: str) -> bool:
        """Validate webhook URL to reduce SSRF risk."""
        try:
            parsed = urlparse(url)
        except Exception:
            return False
        if parsed.scheme != "https":
            return False
        if not parsed.hostname:
            return False

        hostname = parsed.hostname.lower()
        if hostname in {"localhost", "127.0.0.1"} or hostname.endswith(".local"):
            return False

        try:
            ip_addr = ipaddress.ip_address(hostname)
        except ValueError:
            ip_addr = None

        if ip_addr is not None:
            if ip_addr.is_private or ip_addr.is_loopback or ip_addr.is_link_local:
                return False
            if ip_addr.is_multicast or ip_addr.is_reserved:
                return False
            return True

        try:
            addr_infos = socket.getaddrinfo(hostname, None)
        except (socket.gaierror, OSError):
            return False

        for _family, _socktype, _proto, _canon, sockaddr in addr_infos:
            ip_str = sockaddr[0]
            try:
                resolved_ip = ipaddress.ip_address(ip_str)
            except ValueError:
                return False
            if (
                resolved_ip.is_private
                or resolved_ip.is_loopback
                or resolved_ip.is_link_local
            ):
                return False
            if resolved_ip.is_multicast or resolved_ip.is_reserved:
                return False

        return True

    def _get_alert_color(self, severity: AlertSeverity) -> str:
        """Get color for alert severity."""
        colors = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#990000",
        }
        return colors.get(severity, "#808080")


class MonitoringService:
    """Main monitoring service orchestrator."""

    def __init__(self):
        self.logger = setup_logging(self.__class__.__name__)
        self.system_monitor = SystemMonitor()
        self.pipeline_monitor = PipelineMonitor()
        self.alert_manager = AlertManager()
        self.status_file = Path("monitoring_status.json")

    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        checks = []

        # System health
        system_health = self.system_monitor.check_system_health()
        checks.append(asdict(system_health))

        # Pipeline health
        dvc_health = self.pipeline_monitor.check_dvc_status()
        checks.append(asdict(dvc_health))

        opt_health = self.pipeline_monitor.check_optimization_status()
        checks.append(asdict(opt_health))

        # Overall status
        statuses = [check["status"] for check in checks]
        if HealthStatus.CRITICAL.value in statuses:
            overall = HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY.value in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED.value in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        result = {
            "overall_status": overall.value,
            "timestamp": datetime.now().isoformat(),
            "checks": checks,
        }

        # Save status
        safe_file_write(self.status_file, json.dumps(result, indent=2), backup=False)

        # Create alerts for critical issues
        if overall == HealthStatus.CRITICAL:
            self.alert_manager.create_alert(
                AlertSeverity.CRITICAL,
                "Critical Health Issues",
                "One or more critical health issues detected",
                {"checks": checks},
            )
        elif overall == HealthStatus.UNHEALTHY:
            self.alert_manager.create_alert(
                AlertSeverity.ERROR,
                "System Unhealthy",
                "System health issues detected",
                {"checks": checks},
            )

        return result

    def collect_metrics(self) -> List[Metric]:
        """Collect all metrics."""
        metrics = []

        # System metrics
        cpu = self.system_monitor.get_cpu_usage()
        metrics.append(
            Metric(
                name="system.cpu",
                value=cpu,
                unit="percent",
                timestamp=datetime.now().isoformat(),
            )
        )

        memory = self.system_monitor.get_memory_usage()
        metrics.append(
            Metric(
                name="system.memory",
                value=memory["percent"],
                unit="percent",
                timestamp=datetime.now().isoformat(),
            )
        )

        # Pipeline metrics
        pipeline_metrics = self.pipeline_monitor.collect_pipeline_metrics()
        metrics.extend(pipeline_metrics)

        # Save metrics
        metrics_data = [asdict(m) for m in metrics]
        safe_file_write(
            Path("metrics.json"), json.dumps(metrics_data, indent=2), backup=False
        )

        return metrics

    def check_thresholds(self, metrics: List[Metric]):
        """Check metrics against thresholds."""
        thresholds = {
            "system.cpu": 80,
            "system.memory": 80,
            "cost": 10.0,
            "latency": 5.0,
        }

        for metric in metrics:
            if metric.name in thresholds:
                threshold = thresholds[metric.name]
                if metric.value > threshold:
                    self.alert_manager.create_alert(
                        AlertSeverity.WARNING,
                        f"{metric.name} threshold exceeded",
                        f"{metric.name} is {metric.value:.2f} {metric.unit}, threshold is {threshold}",
                        {"metric": asdict(metric)},
                    )


def main():
    """Main monitoring entry point."""
    print("🔍 Running monitoring checks...")

    service = MonitoringService()

    # Run health checks
    health = service.run_health_checks()
    print(f"Overall health: {health['overall_status']}")

    # Collect metrics
    metrics = service.collect_metrics()
    print(f"Collected {len(metrics)} metrics")

    # Check thresholds
    service.check_thresholds(metrics)

    # Audit log
    audit_logger.log_event(
        "monitoring_complete",
        {"health": health["overall_status"], "metrics_count": len(metrics)},
    )

    print("✅ Monitoring complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
