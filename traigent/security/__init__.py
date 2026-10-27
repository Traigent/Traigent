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

_TRAIGENT_SECURITY_DEPLOYMENT = "traigent.security.deployment"
_TRAIGENT_SECURITY_ENCRYPTION = "traigent.security.encryption"
_TRAIGENT_SECURITY_TENANT = "traigent.security.tenant"

_LAZY_EXPORTS = {
    "MultiFactorAuth": "traigent.security.auth",
    "DeploymentManager": _TRAIGENT_SECURITY_DEPLOYMENT,
    "SLAMonitor": _TRAIGENT_SECURITY_DEPLOYMENT,
    "HealthChecker": _TRAIGENT_SECURITY_DEPLOYMENT,
    "BackupManager": _TRAIGENT_SECURITY_DEPLOYMENT,
    "EncryptionManager": _TRAIGENT_SECURITY_ENCRYPTION,
    "PIIDetector": _TRAIGENT_SECURITY_ENCRYPTION,
    "DataClassifier": _TRAIGENT_SECURITY_ENCRYPTION,
    "SecureStorage": _TRAIGENT_SECURITY_ENCRYPTION,
    "TenantManager": _TRAIGENT_SECURITY_TENANT,
    "TenantContext": _TRAIGENT_SECURITY_TENANT,
    "TenantIsolation": _TRAIGENT_SECURITY_TENANT,
    "BillingIntegration": _TRAIGENT_SECURITY_TENANT,
    "Tenant": _TRAIGENT_SECURITY_TENANT,
    "TenantQuotas": _TRAIGENT_SECURITY_TENANT,
    "TenantUsage": _TRAIGENT_SECURITY_TENANT,
    "TenantTier": _TRAIGENT_SECURITY_TENANT,
    "TenantStatus": _TRAIGENT_SECURITY_TENANT,
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
