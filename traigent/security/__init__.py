"""Enterprise security and compliance module for Traigent SDK."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

from importlib import import_module
from typing import Any

from .audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditSeverity,
    ComplianceFramework,
    EnrichmentProviderUnavailableError,
    EventProcessor,
    SecurityMonitor,
)

_LAZY_EXPORTS = {
    "MultiFactorAuth": "traigent.security.auth",
    "DeploymentManager": "traigent.security.deployment",
    "SLAMonitor": "traigent.security.deployment",
    "HealthChecker": "traigent.security.deployment",
    "BackupManager": "traigent.security.deployment",
    "EncryptionManager": "traigent.security.encryption",
    "PIIDetector": "traigent.security.encryption",
    "DataClassifier": "traigent.security.encryption",
    "SecureStorage": "traigent.security.encryption",
    "TenantManager": "traigent.security.tenant",
    "TenantContext": "traigent.security.tenant",
    "TenantIsolation": "traigent.security.tenant",
    "BillingIntegration": "traigent.security.tenant",
    "Tenant": "traigent.security.tenant",
    "TenantQuotas": "traigent.security.tenant",
    "TenantUsage": "traigent.security.tenant",
    "TenantTier": "traigent.security.tenant",
    "TenantStatus": "traigent.security.tenant",
}

__all__ = [
    # Authentication & Authorization
    "MultiFactorAuth",
    # Data Protection & Encryption
    "EncryptionManager",
    "PIIDetector",
    "DataClassifier",
    "SecureStorage",
    # Audit & Compliance
    "AuditSeverity",
    "AuditEvent",
    "AuditEventType",
    "ComplianceFramework",
    "AuditLogger",
    "SecurityMonitor",
    "EventProcessor",
    "EnrichmentProviderUnavailableError",
    # Multi-Tenancy
    "TenantManager",
    "TenantContext",
    "TenantIsolation",
    "BillingIntegration",
    "Tenant",
    "TenantQuotas",
    "TenantUsage",
    "TenantTier",
    "TenantStatus",
    # Enterprise Deployment
    "DeploymentManager",
    "SLAMonitor",
    "HealthChecker",
    "BackupManager",
]


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
