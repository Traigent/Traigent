"""Multi-tenant support and resource management."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security CONC-Quality-Reliability FUNC-SECURITY REQ-SEC-010 SYNC-CloudHybrid

from __future__ import annotations

import secrets
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, cast

from ..utils.logging import get_logger

logger = get_logger(__name__)


class TenantStatus(Enum):
    """Enumeration of possible tenant account statuses.

    Defines the lifecycle states a tenant can be in, from active usage
    through suspension, cancellation, and deletion.
    """

    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    CANCELLED = "cancelled"
    DELETED = "deleted"


class TenantTier(Enum):
    """Service tier levels for multi-tenant pricing and feature access.

    Each tier provides different resource quotas, feature access, and support levels.
    Higher tiers unlock advanced features and increased resource limits.
    """

    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TenantQuotas:
    """Resource quotas and feature flags for tenant accounts.

    Defines limits on API usage, concurrent operations, storage, and users,
    as well as feature flags for advanced functionality. These quotas are
    typically determined by the tenant's service tier.

    Attributes:
        max_optimizations_per_month: Monthly limit on optimization runs
        max_concurrent_optimizations: Simultaneous optimization limit
        max_users: Maximum number of users per tenant
        max_storage_gb: Storage quota in gigabytes
        max_api_calls_per_hour: Hourly API rate limit
        max_api_calls_per_minute: Minute API rate limit
        advanced_algorithms_enabled: Access to advanced optimization algorithms
        custom_metrics_enabled: Ability to define custom evaluation metrics
        priority_support_enabled: Access to priority support channels
        sla_guarantee_enabled: Service level agreement guarantees
        white_label_enabled: White-label branding options
    """

    max_optimizations_per_month: int = 1000
    max_concurrent_optimizations: int = 10
    max_users: int = 5
    max_storage_gb: int = 100
    max_api_calls_per_hour: int = 1000
    max_api_calls_per_minute: int = 100

    # Feature flags
    advanced_algorithms_enabled: bool = False
    custom_metrics_enabled: bool = False
    priority_support_enabled: bool = False
    sla_guarantee_enabled: bool = False
    white_label_enabled: bool = False

    @classmethod
    def from_tier(cls, tier: TenantTier) -> TenantQuotas:
        """Create quota configuration based on service tier.

        Returns pre-configured quota limits and feature flags appropriate
        for the specified service tier. Higher tiers provide increased
        limits and additional features.

        Args:
            tier: The service tier to generate quotas for

        Returns:
            TenantQuotas configured for the specified tier
        """
        if tier == TenantTier.FREE:
            return cls(
                max_optimizations_per_month=100,
                max_concurrent_optimizations=1,
                max_users=1,
                max_storage_gb=1,
                max_api_calls_per_hour=100,
                max_api_calls_per_minute=10,
                advanced_algorithms_enabled=False,
                custom_metrics_enabled=False,
                priority_support_enabled=False,
                sla_guarantee_enabled=False,
                white_label_enabled=False,
            )
        elif tier == TenantTier.BASIC:
            return cls(
                max_optimizations_per_month=1000,
                max_concurrent_optimizations=5,
                max_users=5,
                max_storage_gb=10,
                max_api_calls_per_hour=500,
                max_api_calls_per_minute=50,
                advanced_algorithms_enabled=False,
                custom_metrics_enabled=True,
                priority_support_enabled=False,
                sla_guarantee_enabled=False,
                white_label_enabled=False,
            )
        elif tier == TenantTier.PROFESSIONAL:
            return cls(
                max_optimizations_per_month=5000,
                max_concurrent_optimizations=25,
                max_users=25,
                max_storage_gb=100,
                max_api_calls_per_hour=2000,
                max_api_calls_per_minute=200,
                advanced_algorithms_enabled=True,
                custom_metrics_enabled=True,
                priority_support_enabled=True,
                sla_guarantee_enabled=False,
                white_label_enabled=False,
            )
        else:  # ENTERPRISE
            return cls(
                max_optimizations_per_month=50000,
                max_concurrent_optimizations=100,
                max_users=100,
                max_storage_gb=1000,
                max_api_calls_per_hour=10000,
                max_api_calls_per_minute=1000,
                advanced_algorithms_enabled=True,
                custom_metrics_enabled=True,
                priority_support_enabled=True,
                sla_guarantee_enabled=True,
                white_label_enabled=True,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert quotas to dictionary."""
        return {
            "max_optimizations_per_month": self.max_optimizations_per_month,
            "max_concurrent_optimizations": self.max_concurrent_optimizations,
            "max_users": self.max_users,
            "max_storage_gb": self.max_storage_gb,
            "max_api_calls_per_hour": self.max_api_calls_per_hour,
            "max_api_calls_per_minute": self.max_api_calls_per_minute,
            "advanced_algorithms_enabled": self.advanced_algorithms_enabled,
            "custom_metrics_enabled": self.custom_metrics_enabled,
            "priority_support_enabled": self.priority_support_enabled,
            "sla_guarantee_enabled": self.sla_guarantee_enabled,
            "white_label_enabled": self.white_label_enabled,
        }


@dataclass
class TenantUsage:
    """Real-time usage tracking for tenant resource consumption.

    Maintains counters for various resources and API usage to enforce
    quota limits. Provides methods to reset counters at different time
    intervals (monthly, daily, hourly, minute).

    Attributes:
        optimizations_this_month: Count of optimizations in current month
        concurrent_optimizations: Currently running optimizations
        active_users: Number of active users
        storage_used_gb: Storage consumption in gigabytes
        api_calls_today: API calls made today
        api_calls_this_hour: API calls in current hour
        api_calls_this_minute: API calls in current minute
        last_updated: Timestamp of last usage update
    """

    optimizations_this_month: int = 0
    concurrent_optimizations: int = 0
    active_users: int = 0
    storage_used_gb: float = 0.0
    api_calls_today: int = 0
    api_calls_this_hour: int = 0
    api_calls_this_minute: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def reset_monthly_counters(self) -> None:
        """Reset monthly usage counters at the start of a new billing cycle.

        Resets optimization count for the month while preserving other metrics.
        Updates the last_updated timestamp.
        """
        self.optimizations_this_month = 0
        self.last_updated = datetime.utcnow()

    def reset_daily_counters(self) -> None:
        """Reset daily API call counter at midnight UTC.

        Clears the daily API call count while preserving hourly and minute counts.
        Updates the last_updated timestamp.
        """
        self.api_calls_today = 0
        self.last_updated = datetime.utcnow()

    def reset_hourly_counters(self) -> None:
        """Reset hourly API call counter at the start of each hour.

        Clears the hourly API call count while preserving minute count.
        Updates the last_updated timestamp.
        """
        self.api_calls_this_hour = 0
        self.last_updated = datetime.utcnow()

    def reset_minute_counters(self) -> None:
        """Reset per-minute API call counter for rate limiting.

        Clears the minute API call count to enforce per-minute rate limits.
        Updates the last_updated timestamp.
        """
        self.api_calls_this_minute = 0
        self.last_updated = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert usage statistics to dictionary format.

        Returns:
            Dictionary containing all usage metrics and last_updated timestamp
        """
        return {
            "optimizations_this_month": self.optimizations_this_month,
            "concurrent_optimizations": self.concurrent_optimizations,
            "active_users": self.active_users,
            "storage_used_gb": self.storage_used_gb,
            "api_calls_today": self.api_calls_today,
            "api_calls_this_hour": self.api_calls_this_hour,
            "api_calls_this_minute": self.api_calls_this_minute,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class Tenant:
    """Tenant entity."""

    tenant_id: str
    name: str
    contact_email: str
    tier: TenantTier = TenantTier.BASIC
    status: TenantStatus = TenantStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    trial_ends_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    quotas: TenantQuotas | None = None
    usage: TenantUsage = field(default_factory=TenantUsage)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False, compare=False
    )

    _RESOURCE_MAP = {
        "optimizations_per_month": (
            "optimizations_this_month",
            "max_optimizations_per_month",
        ),
        "concurrent_optimizations": (
            "concurrent_optimizations",
            "max_concurrent_optimizations",
        ),
        "users": ("active_users", "max_users"),
        "storage": ("storage_used_gb", "max_storage_gb"),
        "api_calls_per_hour": ("api_calls_this_hour", "max_api_calls_per_hour"),
        "api_calls_per_minute": ("api_calls_this_minute", "max_api_calls_per_minute"),
        "api_calls_today": ("api_calls_today", None),
    }

    def __post_init__(self) -> None:
        """Initialize tenant after creation."""
        # Set quotas based on tier if not explicitly provided
        if self.quotas is None:
            # If no tier is explicitly set and we're defaulting, use default quotas
            # rather than tier-based quotas (for backward compatibility)
            if self.tier == TenantTier.FREE:
                # But for FREE tier, always use the tier-specific (limited) quotas
                self.quotas = TenantQuotas.from_tier(self.tier)
            else:
                # For other tiers, use tier-specific quotas
                self.quotas = TenantQuotas.from_tier(self.tier)

    def _ensure_quotas(self) -> TenantQuotas:
        """Return tenant quotas, lazily initialising if missing."""
        if self.quotas is None:
            self.quotas = TenantQuotas.from_tier(self.tier)
        return self.quotas

    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE

    def check_quota(self, resource_type: str, amount: int = 1) -> bool:
        """Check if tenant has quota for resource."""
        self._validate_inputs(resource_type, amount)
        with self._lock:
            if not self.is_active():
                return False
            return self._has_quota_unlocked(resource_type, amount)

    def consume_quota(self, resource_type: str, amount: int = 1) -> bool:
        """Consume quota for resource."""
        self._validate_inputs(resource_type, amount)
        with self._lock:
            if not self.is_active():
                return False
            if not self._has_quota_unlocked(resource_type, amount):
                return False
            self._apply_consumption_unlocked(resource_type, amount)
            self.usage.last_updated = datetime.utcnow()
            return True

    def release_quota(self, resource_type: str, amount: int = 1) -> None:
        """Release quota for resource."""
        self._validate_inputs(resource_type, amount)
        with self._lock:
            usage_attr, _ = self._RESOURCE_MAP[resource_type]
            current = getattr(self.usage, usage_attr)
            if resource_type == "storage":
                new_value = max(0.0, current - amount)
            else:
                new_value = max(0, current - amount)
            setattr(self.usage, usage_attr, new_value)
            self.usage.last_updated = datetime.utcnow()

    def _validate_inputs(self, resource_type: str, amount: int | float) -> None:
        if resource_type not in self._RESOURCE_MAP:
            raise ValueError(f"Unknown resource type: {resource_type}")
        if amount <= 0:
            raise ValueError("amount must be positive")

    def _has_quota_unlocked(self, resource_type: str, amount: int | float) -> bool:
        quotas = self._ensure_quotas()
        usage_attr, quota_attr = self._RESOURCE_MAP[resource_type]
        current_usage = cast(float, getattr(self.usage, usage_attr))

        if resource_type == "api_calls_today":
            daily_limit = quotas.max_api_calls_per_hour * 24
            return current_usage + amount <= daily_limit

        if quota_attr is None:
            return False

        quota_limit = cast(float, getattr(quotas, quota_attr))
        return current_usage + amount <= quota_limit

    def _apply_consumption_unlocked(
        self, resource_type: str, amount: int | float
    ) -> None:
        usage_attr, _ = self._RESOURCE_MAP[resource_type]
        current = getattr(self.usage, usage_attr)
        setattr(self.usage, usage_attr, current + amount)

    def upgrade_tier(self, new_tier: TenantTier) -> bool:
        """Upgrade tenant tier."""
        tier_order = [
            TenantTier.FREE,
            TenantTier.BASIC,
            TenantTier.PROFESSIONAL,
            TenantTier.ENTERPRISE,
        ]

        if tier_order.index(new_tier) <= tier_order.index(self.tier):
            return False

        self.tier = new_tier
        self.quotas = TenantQuotas.from_tier(new_tier)
        logger.info(f"Upgraded tenant {self.tenant_id} to {new_tier.value}")
        return True

    def get_quota_utilization(self) -> dict[str, float]:
        """Get quota utilization percentages."""
        if not self.quotas:
            return {}

        return {
            "optimizations_per_month": (
                self.usage.optimizations_this_month
                / self.quotas.max_optimizations_per_month
            )
            * 100,
            "concurrent_optimizations": (
                self.usage.concurrent_optimizations
                / self.quotas.max_concurrent_optimizations
            )
            * 100,
            "users": (self.usage.active_users / self.quotas.max_users) * 100,
            "storage": (self.usage.storage_used_gb / self.quotas.max_storage_gb) * 100,
            "api_calls_per_hour": (
                self.usage.api_calls_this_hour / self.quotas.max_api_calls_per_hour
            )
            * 100,
            "api_calls_per_minute": (
                self.usage.api_calls_this_minute / self.quotas.max_api_calls_per_minute
            )
            * 100,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert tenant to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "contact_email": self.contact_email,
            "tier": self.tier.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "trial_ends_at": (
                self.trial_ends_at.isoformat() if self.trial_ends_at else None
            ),
            "metadata": self.metadata,
            "quotas": self.quotas.to_dict() if self.quotas else {},
            "usage": self.usage.to_dict(),
        }


class TenantContext:
    """Thread-local context for tenant isolation."""

    def __init__(self) -> None:
        """Initialize tenant context."""
        self._local = threading.local()

    def set_tenant(self, tenant_id: str) -> None:
        """Set current tenant ID."""
        self._local.tenant_id = tenant_id

    def get_tenant(self) -> str | None:
        """Get current tenant ID."""
        return getattr(self._local, "tenant_id", None)

    def clear_tenant(self) -> None:
        """Clear current tenant."""
        if hasattr(self._local, "tenant_id"):
            delattr(self._local, "tenant_id")

    @contextmanager
    def tenant_scope(self, tenant_id: str) -> Generator[None, None, None]:
        """Context manager for tenant scope."""
        old_tenant = self.get_tenant()
        self.set_tenant(tenant_id)
        try:
            yield
        finally:
            if old_tenant:
                self.set_tenant(old_tenant)
            else:
                self.clear_tenant()


class TenantManager:
    """Manages multi-tenant operations."""

    def __init__(self) -> None:
        """Initialize tenant manager."""
        self.tenants: dict[str, Tenant] = {}
        self.context = TenantContext()
        self._quota_violations: list[dict[str, Any]] = []
        self.usage_history: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()

    @property
    def quota_violations(self) -> dict[str, list[dict[str, Any]]]:
        """Get quota violations grouped by tenant."""
        with self._lock:
            violations_by_tenant: dict[str, list[dict[str, Any]]] = {}
            for violation in self._quota_violations:
                tenant_id = violation["tenant_id"]
                if tenant_id not in violations_by_tenant:
                    violations_by_tenant[tenant_id] = []
                violations_by_tenant[tenant_id].append(violation)
            return violations_by_tenant

    def create_tenant(
        self,
        name: str,
        contact_email: str,
        tier: TenantTier = TenantTier.FREE,
        status: TenantStatus = TenantStatus.ACTIVE,
    ) -> Tenant:
        """Create new tenant."""
        tenant_id = secrets.token_urlsafe(16)

        # Set trial end date if status is TRIAL
        trial_ends_at = None
        if status == TenantStatus.TRIAL:
            trial_ends_at = datetime.utcnow() + timedelta(days=30)

        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            contact_email=contact_email,
            tier=tier,
            status=status,
            trial_ends_at=trial_ends_at,
            quotas=TenantQuotas.from_tier(tier),
        )

        with self._lock:
            self.tenants[tenant_id] = tenant

        logger.info(f"Created tenant {name} ({tenant_id}) with {tier.value} tier")
        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        """Get tenant by ID."""
        with self._lock:
            return self.tenants.get(tenant_id)

    def list_tenants(
        self, status: TenantStatus | None = None, tier: TenantTier | None = None
    ) -> list[Tenant]:
        """List tenants with optional filtering."""
        with self._lock:
            tenants = list(self.tenants.values())

        if status is not None:
            tenants = [t for t in tenants if t.status == status]

        if tier is not None:
            tenants = [t for t in tenants if t.tier == tier]

        return tenants

    def update_tenant(self, tenant_id: str, **updates) -> bool:
        """Update tenant properties."""
        with self._lock:
            tenant = self.tenants.get(tenant_id)
            if not tenant:
                return False

            for key, value in updates.items():
                if hasattr(tenant, key):
                    setattr(tenant, key, value)

            # Update the timestamp
            tenant.updated_at = datetime.utcnow()

        logger.info(f"Updated tenant {tenant_id}: {updates}")
        return True

    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant (soft delete by marking as deleted)."""
        with self._lock:
            tenant = self.tenants.get(tenant_id)
            if tenant:
                tenant.status = TenantStatus.DELETED
                tenant.updated_at = datetime.utcnow()
                logger.info(f"Soft deleted tenant {tenant_id}")
                return True
        return False

    def check_quota(
        self, resource_type: str, amount: int = 1, tenant_id: str | None = None
    ) -> bool:
        """Check quota for current or specified tenant."""
        if tenant_id is None:
            tenant_id = self.context.get_tenant()

        if not tenant_id:
            return False

        with self._lock:
            tenant = self.tenants.get(tenant_id)

        if not tenant:
            return False

        return tenant.check_quota(resource_type, amount)

    def consume_quota(
        self, resource_type: str, amount: int = 1, tenant_id: str | None = None
    ) -> bool:
        """Consume quota for current or specified tenant."""
        if tenant_id is None:
            tenant_id = self.context.get_tenant()

        if not tenant_id:
            return False

        with self._lock:
            tenant = self.tenants.get(tenant_id)

        if not tenant:
            return False

        success = tenant.consume_quota(resource_type, amount)
        if not success:
            # Record quota violation
            with self._lock:
                self._quota_violations.append(
                    {
                        "tenant_id": tenant_id,
                        "resource": resource_type,  # As expected by tests
                        "requested_amount": amount,  # As expected by tests
                        "timestamp": datetime.utcnow(),
                        "quota_limit": getattr(
                            tenant.quotas, f"max_{resource_type}", "unknown"
                        ),
                        "current_usage": getattr(
                            tenant.usage,
                            resource_type.replace("_per_month", "_this_month"),
                            "unknown",
                        ),
                    }
                )

        return success

    def release_quota(
        self, resource_type: str, amount: int = 1, tenant_id: str | None = None
    ) -> None:
        """Release quota for current or specified tenant."""
        if tenant_id is None:
            tenant_id = self.context.get_tenant()

        if not tenant_id:
            return

        with self._lock:
            tenant = self.tenants.get(tenant_id)

        if tenant:
            tenant.release_quota(resource_type, amount)

    def reset_usage_counters(
        self, tenant_id: str, counter_type: str = "monthly"
    ) -> bool:
        """Reset usage counters for tenant."""
        with self._lock:
            tenant = self.tenants.get(tenant_id)
            if not tenant:
                return False

            # Record current usage before reset
            if tenant_id not in self.usage_history:
                self.usage_history[tenant_id] = []

            self.usage_history[tenant_id].append(
                {
                    "reset_type": counter_type,
                    "timestamp": datetime.utcnow(),
                    "previous_usage": {
                        "optimizations_this_month": tenant.usage.optimizations_this_month,
                        "api_calls_today": tenant.usage.api_calls_today,
                        "api_calls_this_hour": tenant.usage.api_calls_this_hour,
                        "api_calls_this_minute": tenant.usage.api_calls_this_minute,
                    },
                }
            )

            if counter_type == "monthly":
                tenant.usage.reset_monthly_counters()
            elif counter_type == "daily":
                tenant.usage.reset_daily_counters()
            elif counter_type == "hourly":
                tenant.usage.reset_hourly_counters()
            elif counter_type == "minute":
                tenant.usage.reset_minute_counters()

        logger.info(f"Reset {counter_type} counters for tenant {tenant_id}")
        return True

    def get_tenant_analytics(self, tenant_id: str) -> dict[str, Any]:
        """Get analytics for specific tenant."""
        with self._lock:
            tenant = self.tenants.get(tenant_id)
            if not tenant:
                return {}

            # Get tenant-specific data
            tenant_violations = [
                v for v in self._quota_violations if v["tenant_id"] == tenant_id
            ]
            tenant_usage_history = self.usage_history.get(tenant_id, [])

            analytics = {
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "tenant_name": tenant.name,  # Add both for compatibility
                "tier": tenant.tier.value,
                "status": tenant.status.value,
                "created_at": tenant.created_at.isoformat(),
                "quota_utilization": tenant.get_quota_utilization(),
                "current_usage": tenant.usage.to_dict(),
                "quotas": (
                    tenant.quotas.to_dict() if tenant.quotas else {}
                ),  # Use 'quotas' as expected by tests
                "quota_limits": (
                    tenant.quotas.to_dict() if tenant.quotas else {}
                ),  # Keep both for compatibility
                "usage_history": tenant_usage_history,
                "quota_violations": tenant_violations,
                "metrics": {
                    "total_violations": len(tenant_violations),
                    "usage_resets": len(tenant_usage_history),
                    "utilization_score": sum(tenant.get_quota_utilization().values())
                    / len(tenant.get_quota_utilization()),
                },
                "violations_count": len(tenant_violations),
            }

            # Add tenant info for backwards compatibility
            analytics["tenant_info"] = {
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "tier": tenant.tier.value,
                "status": tenant.status.value,
                "created_at": tenant.created_at.isoformat(),
            }

            return analytics

    def get_system_analytics(self) -> dict[str, Any]:
        """Get system-wide analytics."""
        with self._lock:
            total_tenants = len(self.tenants)
            active_tenants = len([t for t in self.tenants.values() if t.is_active()])

            tier_distribution: dict[str, int] = {}
            for tenant in self.tenants.values():
                tier = tenant.tier.value
                tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

            total_violations = len(self._quota_violations)

            # Calculate total usage across all tenants
            total_usage = {
                "optimizations_this_month": sum(
                    t.usage.optimizations_this_month for t in self.tenants.values()
                ),
                "api_calls_today": sum(
                    t.usage.api_calls_today for t in self.tenants.values()
                ),
                "api_calls_this_hour": sum(
                    t.usage.api_calls_this_hour for t in self.tenants.values()
                ),
                "active_users": sum(
                    t.usage.active_users for t in self.tenants.values()
                ),
                "storage_used_gb": sum(
                    t.usage.storage_used_gb for t in self.tenants.values()
                ),
            }

            return {
                "tenant_counts": {
                    "total": total_tenants,
                    "active": active_tenants,
                    "inactive": total_tenants - active_tenants,
                },
                "tier_distribution": tier_distribution,
                "total_usage": total_usage,
                "quota_violations": {
                    "total": total_violations,
                    "recent_24h": len(
                        [
                            v
                            for v in self._quota_violations
                            if v["timestamp"] > datetime.utcnow() - timedelta(hours=24)
                        ]
                    ),
                },
                "quota_violations_today": len(
                    [
                        v
                        for v in self._quota_violations
                        if v["timestamp"].date() == datetime.utcnow().date()
                    ]
                ),
            }

    def get_current_tenant(self) -> Tenant | None:
        """Get current tenant object from context."""
        tenant_id = self.context.get_tenant()
        if tenant_id:
            with self._lock:
                return self.tenants.get(tenant_id)
        return None

    def tenant_scope(self, tenant_id: str):
        """Context manager for tenant scope."""
        return self.context.tenant_scope(tenant_id)


# Legacy classes for backward compatibility
class TenantIsolation:
    """Ensures data isolation between tenants."""

    def __init__(self) -> None:
        """Initialize tenant isolation."""
        self.tenant_data: dict[str, dict[str, Any]] = {}

    def get_tenant_data(self, tenant_id: str, data_type: str) -> dict[str, Any]:
        """Get tenant-specific data."""
        if tenant_id not in self.tenant_data:
            self.tenant_data[tenant_id] = {}

        if data_type not in self.tenant_data[tenant_id]:
            self.tenant_data[tenant_id][data_type] = {}

        return cast(dict[str, Any], self.tenant_data[tenant_id][data_type])


class BillingIntegration:
    """Integrates with billing systems."""

    def __init__(self) -> None:
        """Initialize billing integration."""
        self.usage_records: list[dict[str, Any]] = []

    def record_usage(
        self, tenant_id: str, service: str, quantity: int, unit_cost: float
    ) -> None:
        """Record billable usage."""
        record = {
            "tenant_id": tenant_id,
            "service": service,
            "quantity": quantity,
            "unit_cost": unit_cost,
            "total_cost": quantity * unit_cost,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.usage_records.append(record)
        logger.debug(
            f"Recorded billing usage: {service} x{quantity} for tenant {tenant_id}"
        )

    def generate_invoice(
        self, tenant_id: str, period_start: datetime, period_end: datetime
    ) -> dict[str, Any]:
        """Generate invoice for tenant."""
        tenant_records = [
            r
            for r in self.usage_records
            if r["tenant_id"] == tenant_id
            and period_start <= datetime.fromisoformat(r["timestamp"]) <= period_end
        ]

        total_cost = sum(r["total_cost"] for r in tenant_records)

        return {
            "tenant_id": tenant_id,
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "line_items": tenant_records,
            "total_cost": total_cost,
            "generated_at": datetime.utcnow().isoformat(),
        }
