"""
Tests for multi-tenant support systems
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from traigent.security.tenant import (
    Tenant,
    TenantContext,
    TenantManager,
    TenantQuotas,
    TenantStatus,
    TenantTier,
    TenantUsage,
)


class TestTenantQuotas:
    """Test TenantQuotas class"""

    def test_default_quotas(self):
        """Test default quota creation"""
        quotas = TenantQuotas()

        assert quotas.max_optimizations_per_month == 1000
        assert quotas.max_concurrent_optimizations == 10
        assert quotas.max_users == 5
        assert not quotas.advanced_algorithms_enabled

    def test_quotas_from_tier(self):
        """Test quota creation from tier"""
        # Free tier
        free_quotas = TenantQuotas.from_tier(TenantTier.FREE)
        assert free_quotas.max_optimizations_per_month == 100
        assert free_quotas.max_users == 1
        assert not free_quotas.advanced_algorithms_enabled

        # Enterprise tier
        enterprise_quotas = TenantQuotas.from_tier(TenantTier.ENTERPRISE)
        assert enterprise_quotas.max_optimizations_per_month == 50000
        assert enterprise_quotas.max_users == 100
        assert enterprise_quotas.advanced_algorithms_enabled
        assert enterprise_quotas.priority_support_enabled
        assert enterprise_quotas.sla_guarantee_enabled

    def test_quotas_serialization(self):
        """Test quota serialization"""
        quotas = TenantQuotas(
            max_optimizations_per_month=5000, advanced_algorithms_enabled=True
        )

        quota_dict = quotas.to_dict()

        assert quota_dict["max_optimizations_per_month"] == 5000
        assert quota_dict["advanced_algorithms_enabled"] is True
        assert isinstance(quota_dict, dict)


class TestTenantUsage:
    """Test TenantUsage class"""

    def test_usage_initialization(self):
        """Test usage initialization"""
        usage = TenantUsage()

        assert usage.optimizations_this_month == 0
        assert usage.concurrent_optimizations == 0
        assert usage.api_calls_today == 0
        assert usage.active_users == 0
        assert isinstance(usage.last_updated, datetime)

    def test_usage_reset_counters(self):
        """Test resetting usage counters"""
        usage = TenantUsage()
        usage.optimizations_this_month = 100
        usage.api_calls_today = 1000
        usage.api_calls_this_minute = 50

        # Reset monthly
        usage.reset_monthly_counters()
        assert usage.optimizations_this_month == 0
        assert usage.api_calls_today == 1000  # Should not be reset

        # Reset daily
        usage.reset_daily_counters()
        assert usage.api_calls_today == 0

        # Reset minute
        usage.api_calls_this_minute = 10
        usage.reset_minute_counters()
        assert usage.api_calls_this_minute == 0

    def test_usage_serialization(self):
        """Test usage serialization"""
        usage = TenantUsage()
        usage.optimizations_this_month = 50
        usage.active_users = 3

        usage_dict = usage.to_dict()

        assert usage_dict["optimizations_this_month"] == 50
        assert usage_dict["active_users"] == 3
        assert "last_updated" in usage_dict
        assert isinstance(usage_dict["last_updated"], str)


class TestTenant:
    """Test Tenant class"""

    def test_tenant_creation(self):
        """Test tenant creation"""
        tenant = Tenant(
            tenant_id="tenant123",
            name="Test Tenant",
            contact_email="admin@test.com",
            tier=TenantTier.PROFESSIONAL,
        )

        assert tenant.tenant_id == "tenant123"
        assert tenant.name == "Test Tenant"
        assert tenant.contact_email == "admin@test.com"
        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.is_active()

        # Should have professional tier quotas
        assert tenant.quotas.advanced_algorithms_enabled
        assert tenant.quotas.custom_metrics_enabled

    def test_tenant_quota_checking(self):
        """Test tenant quota checking"""
        tenant = Tenant(
            tenant_id="tenant123",
            name="Test Tenant",
            contact_email="admin@test.com",
            tier=TenantTier.FREE,  # Limited quotas
        )

        # Should have quota for optimizations
        assert tenant.check_quota("optimizations_per_month", 1)
        assert tenant.check_quota("optimizations_per_month", 100)
        assert not tenant.check_quota(
            "optimizations_per_month", 200
        )  # Exceeds free tier limit

        # Should have quota for users
        assert tenant.check_quota("users", 1)
        assert not tenant.check_quota("users", 2)  # Free tier allows only 1 user

    def test_tenant_quota_consumption(self):
        """Test tenant quota consumption"""
        tenant = Tenant(
            tenant_id="tenant123",
            name="Test Tenant",
            contact_email="admin@test.com",
            tier=TenantTier.BASIC,
        )

        # Consume optimization quota
        assert tenant.consume_quota("optimizations_per_month", 50)
        assert tenant.usage.optimizations_this_month == 50

        # Consume more
        assert tenant.consume_quota("optimizations_per_month", 100)
        assert tenant.usage.optimizations_this_month == 150

        # Try to exceed quota (basic tier has 1000 limit)
        assert tenant.consume_quota("optimizations_per_month", 850)
        assert tenant.usage.optimizations_this_month == 1000

        # Should fail to consume more
        assert not tenant.consume_quota("optimizations_per_month", 1)
        assert tenant.usage.optimizations_this_month == 1000

    def test_tenant_quota_release(self):
        """Test tenant quota release"""
        tenant = Tenant(
            tenant_id="tenant123", name="Test Tenant", contact_email="admin@test.com"
        )

        # Consume some concurrent optimizations
        tenant.consume_quota("concurrent_optimizations", 5)
        assert tenant.usage.concurrent_optimizations == 5

        # Release some
        tenant.release_quota("concurrent_optimizations", 2)
        assert tenant.usage.concurrent_optimizations == 3

        # Release more than available (should not go negative)
        tenant.release_quota("concurrent_optimizations", 5)
        assert tenant.usage.concurrent_optimizations == 0

    def test_tenant_quota_rejects_non_positive_amount(self):
        """Ensure negative or zero consumption amounts are rejected."""
        tenant = Tenant(
            tenant_id="tenant123",
            name="Test Tenant",
            contact_email="admin@test.com",
            tier=TenantTier.BASIC,
        )

        with pytest.raises(ValueError):
            tenant.consume_quota("optimizations_per_month", 0)

        with pytest.raises(ValueError):
            tenant.consume_quota("optimizations_per_month", -1)

        with pytest.raises(ValueError):
            tenant.release_quota("optimizations_per_month", 0)

        with pytest.raises(ValueError):
            tenant.check_quota("optimizations_per_month", -10)

    def test_tenant_quota_rejects_unknown_resource(self):
        """Ensure invalid resource types raise errors."""
        tenant = Tenant(
            tenant_id="tenant123",
            name="Test Tenant",
            contact_email="admin@test.com",
            tier=TenantTier.BASIC,
        )

        with pytest.raises(ValueError):
            tenant.consume_quota("unknown_resource", 1)

    def test_concurrent_quota_consumption_is_atomic(self):
        """Concurrent quota consumption must not exceed configured limits."""
        tenant = Tenant(
            tenant_id="tenant123",
            name="Test Tenant",
            contact_email="admin@test.com",
            tier=TenantTier.BASIC,
        )

        limit = tenant.quotas.max_concurrent_optimizations

        def attempt_consume():
            return tenant.consume_quota("concurrent_optimizations", 1)

        with ThreadPoolExecutor(max_workers=limit * 2) as executor:
            futures = [executor.submit(attempt_consume) for _ in range(limit * 2)]
            results = [future.result() for future in futures]

        successful_consumes = sum(results)
        assert successful_consumes == limit
        assert tenant.usage.concurrent_optimizations == limit

    def test_tenant_tier_upgrade(self):
        """Test tenant tier upgrade"""
        tenant = Tenant(
            tenant_id="tenant123",
            name="Test Tenant",
            contact_email="admin@test.com",
            tier=TenantTier.FREE,
        )

        original_quota_limit = tenant.quotas.max_optimizations_per_month

        # Upgrade to professional
        assert tenant.upgrade_tier(TenantTier.PROFESSIONAL)
        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.quotas.max_optimizations_per_month > original_quota_limit
        assert tenant.quotas.advanced_algorithms_enabled

        # Cannot downgrade
        assert not tenant.upgrade_tier(TenantTier.BASIC)
        assert tenant.tier == TenantTier.PROFESSIONAL

    def test_tenant_quota_utilization(self):
        """Test tenant quota utilization calculation"""
        tenant = Tenant(
            tenant_id="tenant123",
            name="Test Tenant",
            contact_email="admin@test.com",
            tier=TenantTier.BASIC,  # 1000 optimizations/month limit
        )

        # Consume some quota
        tenant.consume_quota("optimizations_per_month", 500)
        tenant.consume_quota("users", 3)  # Basic tier allows 5 users

        utilization = tenant.get_quota_utilization()

        assert utilization["optimizations_per_month"] == 50.0  # 500/1000 * 100
        assert utilization["users"] == 60.0  # 3/5 * 100
        assert utilization["concurrent_optimizations"] == 0.0  # None consumed

    def test_inactive_tenant_quota_check(self):
        """Test quota checking for inactive tenant"""
        tenant = Tenant(
            tenant_id="tenant123",
            name="Test Tenant",
            contact_email="admin@test.com",
            status=TenantStatus.SUSPENDED,
        )

        # Inactive tenant should not have any quota
        assert not tenant.is_active()
        assert not tenant.check_quota("optimizations_per_month", 1)
        assert not tenant.consume_quota("optimizations_per_month", 1)

    def test_tenant_serialization(self):
        """Test tenant serialization"""
        tenant = Tenant(
            tenant_id="tenant123",
            name="Test Tenant",
            contact_email="admin@test.com",
            tier=TenantTier.PROFESSIONAL,
        )

        tenant_dict = tenant.to_dict()

        assert tenant_dict["tenant_id"] == "tenant123"
        assert tenant_dict["name"] == "Test Tenant"
        assert tenant_dict["tier"] == TenantTier.PROFESSIONAL.value
        assert tenant_dict["status"] == TenantStatus.ACTIVE.value
        assert "created_at" in tenant_dict
        assert isinstance(tenant_dict["created_at"], str)


class TestTenantContext:
    """Test TenantContext class"""

    def test_set_get_tenant(self):
        """Test setting and getting tenant context"""
        context = TenantContext()

        # Initially no tenant
        assert context.get_tenant() is None

        # Set tenant
        context.set_tenant("tenant123")
        assert context.get_tenant() == "tenant123"

        # Clear tenant
        context.clear_tenant()
        assert context.get_tenant() is None

    def test_tenant_scope_context_manager(self):
        """Test tenant scope context manager"""
        context = TenantContext()

        # Set initial tenant
        context.set_tenant("tenant_original")

        # Use scope for different tenant
        with context.tenant_scope("tenant_scoped"):
            assert context.get_tenant() == "tenant_scoped"

        # Should restore original tenant
        assert context.get_tenant() == "tenant_original"

    def test_nested_tenant_scopes(self):
        """Test nested tenant scopes"""
        context = TenantContext()

        context.set_tenant("tenant1")

        with context.tenant_scope("tenant2"):
            assert context.get_tenant() == "tenant2"

            with context.tenant_scope("tenant3"):
                assert context.get_tenant() == "tenant3"

            # Should restore tenant2
            assert context.get_tenant() == "tenant2"

        # Should restore tenant1
        assert context.get_tenant() == "tenant1"


class TestTenantManager:
    """Test TenantManager class"""

    def test_create_tenant(self):
        """Test tenant creation"""
        manager = TenantManager()

        tenant = manager.create_tenant(
            name="Test Company",
            contact_email="admin@testcompany.com",
            tier=TenantTier.PROFESSIONAL,
        )

        assert tenant.name == "Test Company"
        assert tenant.contact_email == "admin@testcompany.com"
        assert tenant.tier == TenantTier.PROFESSIONAL

        # Should be stored in manager
        retrieved_tenant = manager.get_tenant(tenant.tenant_id)
        assert retrieved_tenant == tenant

    def test_list_tenants(self):
        """Test listing tenants with filters"""
        manager = TenantManager()

        # Create tenants with different tiers and statuses
        tenant1 = manager.create_tenant("Company 1", "admin1@test.com", TenantTier.FREE)
        tenant2 = manager.create_tenant(
            "Company 2", "admin2@test.com", TenantTier.PROFESSIONAL
        )
        tenant3 = manager.create_tenant(
            "Company 3", "admin3@test.com", TenantTier.ENTERPRISE
        )

        # Suspend one tenant
        manager.update_tenant(tenant2.tenant_id, status=TenantStatus.SUSPENDED)

        # List all tenants
        all_tenants = manager.list_tenants()
        assert len(all_tenants) == 3

        # List by status
        active_tenants = manager.list_tenants(status=TenantStatus.ACTIVE)
        assert len(active_tenants) == 2
        assert tenant1 in active_tenants
        assert tenant3 in active_tenants

        suspended_tenants = manager.list_tenants(status=TenantStatus.SUSPENDED)
        assert len(suspended_tenants) == 1
        assert tenant2 in suspended_tenants

        # List by tier
        enterprise_tenants = manager.list_tenants(tier=TenantTier.ENTERPRISE)
        assert len(enterprise_tenants) == 1
        assert tenant3 in enterprise_tenants

    def test_update_tenant(self):
        """Test updating tenant properties"""
        manager = TenantManager()

        tenant = manager.create_tenant("Test Company", "admin@test.com")
        original_updated_at = tenant.updated_at

        # Update tenant
        assert manager.update_tenant(
            tenant.tenant_id, name="Updated Company Name", tier=TenantTier.PROFESSIONAL
        )

        # Check updates
        assert tenant.name == "Updated Company Name"
        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.updated_at > original_updated_at

        # Update non-existent tenant
        assert not manager.update_tenant("nonexistent", name="Test")

    def test_delete_tenant(self):
        """Test tenant deletion (soft delete)"""
        manager = TenantManager()

        tenant = manager.create_tenant("Test Company", "admin@test.com")
        assert tenant.status == TenantStatus.ACTIVE

        # Delete tenant (soft delete)
        assert manager.delete_tenant(tenant.tenant_id)
        assert tenant.status == TenantStatus.DELETED

        # Tenant should still exist but be marked as deleted
        retrieved_tenant = manager.get_tenant(tenant.tenant_id)
        assert retrieved_tenant is not None
        assert retrieved_tenant.status == TenantStatus.DELETED

    def test_quota_management_with_context(self):
        """Test quota management with tenant context"""
        manager = TenantManager()

        tenant = manager.create_tenant(
            "Test Company", "admin@test.com", TenantTier.BASIC
        )

        # Set tenant context
        manager.context.set_tenant(tenant.tenant_id)

        # Check quota
        assert manager.check_quota("optimizations_per_month", 100)
        assert not manager.check_quota(
            "optimizations_per_month", 2000
        )  # Exceeds basic limit

        # Consume quota
        assert manager.consume_quota("optimizations_per_month", 100)
        assert tenant.usage.optimizations_this_month == 100

        # Release quota
        manager.release_quota(
            "concurrent_optimizations", 1
        )  # Should handle gracefully even if none consumed

    def test_quota_violation_tracking(self):
        """Test quota violation tracking"""
        manager = TenantManager()

        tenant = manager.create_tenant(
            "Test Company", "admin@test.com", TenantTier.FREE  # Very limited quotas
        )

        manager.context.set_tenant(tenant.tenant_id)

        # Try to consume more than allowed
        assert not manager.consume_quota(
            "optimizations_per_month", 200
        )  # Free tier limit is 100

        # Should record violation
        violations = manager.quota_violations[tenant.tenant_id]
        assert len(violations) == 1

        violation = violations[0]
        assert violation["resource"] == "optimizations_per_month"
        assert violation["requested_amount"] == 200
        assert "timestamp" in violation

    def test_usage_counter_reset(self):
        """Test resetting usage counters"""
        manager = TenantManager()

        tenant = manager.create_tenant("Test Company", "admin@test.com")

        # Consume some quota
        tenant.consume_quota("optimizations_per_month", 50)
        tenant.consume_quota("api_calls_today", 1000)

        # Reset monthly counters
        assert manager.reset_usage_counters(tenant.tenant_id, "monthly")
        assert tenant.usage.optimizations_this_month == 0
        assert tenant.usage.api_calls_today == 1000  # Should not be reset

        # Should store usage history
        history = manager.usage_history[tenant.tenant_id]
        assert len(history) == 1
        assert history[0]["reset_type"] == "monthly"

    def test_tenant_analytics(self):
        """Test tenant analytics generation"""
        manager = TenantManager()

        tenant = manager.create_tenant(
            "Test Company", "admin@test.com", TenantTier.PROFESSIONAL
        )

        # Generate some usage
        tenant.consume_quota("optimizations_per_month", 100)
        tenant.consume_quota("users", 5)

        # Create usage history
        manager.reset_usage_counters(tenant.tenant_id, "monthly")

        # Create quota violation
        manager.context.set_tenant(tenant.tenant_id)
        manager.consume_quota(
            "optimizations_per_month", 10000
        )  # Should fail and create violation

        # Get analytics
        analytics = manager.get_tenant_analytics(tenant.tenant_id)

        assert analytics["tenant_id"] == tenant.tenant_id
        assert analytics["tenant_name"] == "Test Company"
        assert analytics["tier"] == TenantTier.PROFESSIONAL.value
        assert "current_usage" in analytics
        assert "quotas" in analytics
        assert "quota_utilization" in analytics
        assert "usage_history" in analytics
        assert "quota_violations" in analytics
        assert "metrics" in analytics

        # Should have violation
        assert len(analytics["quota_violations"]) > 0

    def test_system_analytics(self):
        """Test system-wide analytics"""
        manager = TenantManager()

        # Create tenants with different tiers
        manager.create_tenant("Company 1", "admin1@test.com", TenantTier.FREE)
        manager.create_tenant("Company 2", "admin2@test.com", TenantTier.PROFESSIONAL)
        manager.create_tenant("Company 3", "admin3@test.com", TenantTier.ENTERPRISE)

        analytics = manager.get_system_analytics()

        assert analytics["tenant_counts"]["total"] == 3
        assert analytics["tenant_counts"]["active"] == 3
        assert analytics["tier_distribution"][TenantTier.FREE.value] == 1
        assert analytics["tier_distribution"][TenantTier.PROFESSIONAL.value] == 1
        assert analytics["tier_distribution"][TenantTier.ENTERPRISE.value] == 1
        assert "total_usage" in analytics
        assert "quota_violations_today" in analytics

    def test_tenant_scope_context_manager(self):
        """Test tenant scope context manager"""
        manager = TenantManager()

        tenant1 = manager.create_tenant("Company 1", "admin1@test.com")
        tenant2 = manager.create_tenant("Company 2", "admin2@test.com")

        # Set initial context
        manager.context.set_tenant(tenant1.tenant_id)
        assert manager.context.get_tenant() == tenant1.tenant_id

        # Use scope for different tenant
        with manager.tenant_scope(tenant2.tenant_id):
            assert manager.context.get_tenant() == tenant2.tenant_id

            # Operations in this scope should affect tenant2
            current_tenant = manager.get_current_tenant()
            assert current_tenant.tenant_id == tenant2.tenant_id

        # Should restore original tenant
        assert manager.context.get_tenant() == tenant1.tenant_id

    def test_no_tenant_context_operations(self):
        """Test operations without tenant context"""
        manager = TenantManager()

        # Operations without tenant context should handle gracefully
        assert not manager.check_quota("optimizations_per_month", 1)
        assert not manager.consume_quota("optimizations_per_month", 1)

        # Should log warnings but not crash
        manager.release_quota("optimizations_per_month", 1)  # Should handle gracefully

        # Current tenant should be None
        assert manager.get_current_tenant() is None
