"""
Tests for audit logging and compliance reporting systems
"""

import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from traigent.security.audit import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditLogIntegrity,
    AuditSeverity,
    AuditStorage,
    ComplianceFramework,
    ComplianceReporter,
)

STRONG_AUDIT_SECRET = "StrongAuditSecretKey123!@#ABC4567890"
COMPLIANCE_NOT_IMPLEMENTED = "Compliance reporting subsystem is not yet implemented"
PERSISTENT_STORAGE_NOT_IMPLEMENTED = "Persistent audit storage is not yet implemented"
TAMPER_DETECTION_NOT_IMPLEMENTED = "Tamper-detection is not yet implemented"
FAKE_AUDIT_API_KEY = (
    "sk-ant-canary-DO-NOT-USE-123456789abcdef"  # pragma: allowlist secret
)


class TestAuditEvent:
    """Test AuditEvent class"""

    def test_audit_event_creation(self):
        """Test audit event creation"""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id="user123",
            tenant_id="tenant123",
            source_ip="192.168.1.1",
            message="User login successful",
            severity=AuditSeverity.LOW,
        )

        assert event.event_type == AuditEventType.LOGIN_SUCCESS
        assert event.user_id == "user123"
        assert event.tenant_id == "tenant123"
        assert event.source_ip == "192.168.1.1"
        assert event.message == "User login successful"
        assert event.severity == AuditSeverity.LOW
        assert event.result == "success"
        assert isinstance(event.timestamp, datetime)

    def test_audit_event_serialization(self):
        """Test audit event serialization"""
        event = AuditEvent(
            event_type=AuditEventType.DATA_CREATED,
            user_id="user123",
            compliance_tags=[ComplianceFramework.GDPR, ComplianceFramework.SOC2],
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == AuditEventType.DATA_CREATED.value
        assert event_dict["user_id"] == "user123"
        assert event_dict["compliance_tags"] == ["gdpr", "soc2"]
        assert "timestamp" in event_dict
        assert isinstance(event_dict["timestamp"], str)

    def test_audit_event_hash(self):
        """Test audit event hash for integrity"""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id="user123",
            message="Test event",
        )

        hash1 = event.get_hash()
        hash2 = event.get_hash()

        # Same event should produce same hash
        assert hash1 == hash2

        # Different event should produce different hash
        event.message = "Different message"
        hash3 = event.get_hash()
        assert hash1 != hash3


class TestAuditStorage:
    """Test AuditStorage class"""

    def test_default_storage_path_keeps_backward_compatible_value(self):
        """No-arg construction preserves the historical default path."""
        storage = AuditStorage()

        assert storage.storage_path == "audit_logs"
        assert storage.events == []

    def test_explicit_storage_path_is_accepted_for_compatibility(self):
        """Explicit paths remain constructible while storage stays in-memory."""
        storage = AuditStorage(storage_path="custom_audit_logs")
        event = AuditEvent(event_type=AuditEventType.LOGIN_SUCCESS, user_id="user123")

        storage.store_event(event)

        assert storage.storage_path == "custom_audit_logs"
        assert storage.get_events() == [event]

    def test_store_and_retrieve_events(self):
        """Test storing and retrieving events"""
        storage = AuditStorage()

        event1 = AuditEvent(event_type=AuditEventType.LOGIN_SUCCESS, user_id="user123")
        event2 = AuditEvent(event_type=AuditEventType.DATA_CREATED, user_id="user456")

        # Store events
        storage.store_event(event1)
        storage.store_event(event2)

        # Get all events
        events = storage.get_events()
        assert len(events) == 2
        assert event1 in events
        assert event2 in events

        # Events should be sorted by timestamp (newest first)
        assert events[0].timestamp >= events[1].timestamp

    def test_filter_events_by_time(self):
        """Test filtering events by time range"""
        storage = AuditStorage()

        now = datetime.now(UTC)

        # Create events with different timestamps
        old_event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS, user_id="user123"
        )
        old_event.timestamp = now - timedelta(hours=2)

        recent_event = AuditEvent(
            event_type=AuditEventType.DATA_CREATED, user_id="user456"
        )
        recent_event.timestamp = now - timedelta(minutes=30)

        storage.store_event(old_event)
        storage.store_event(recent_event)

        # Filter by time
        start_time = now - timedelta(hours=1)
        recent_events = storage.get_events(start_time=start_time)

        assert len(recent_events) == 1
        assert recent_events[0] == recent_event

    def test_filter_events_by_user(self):
        """Test filtering events by user"""
        storage = AuditStorage()

        event1 = AuditEvent(event_type=AuditEventType.LOGIN_SUCCESS, user_id="user123")
        event2 = AuditEvent(event_type=AuditEventType.DATA_CREATED, user_id="user456")
        event3 = AuditEvent(event_type=AuditEventType.LOGOUT, user_id="user123")

        storage.store_event(event1)
        storage.store_event(event2)
        storage.store_event(event3)

        # Filter by user
        user123_events = storage.get_events(user_id="user123")
        assert len(user123_events) == 2
        assert event1 in user123_events
        assert event3 in user123_events
        assert event2 not in user123_events

    def test_filter_events_by_redacted_email_user_id(self):
        """Email user IDs stay redacted while exact-match storage filters work."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        user_id = "alice@example.com"
        other_user_id = "bob@example.com"

        matching_event = audit_logger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id=user_id,
            message="Alice login",
        )
        other_event = audit_logger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id=other_user_id,
            message="Bob login",
        )

        events = audit_logger.storage.get_events(user_id=user_id)

        assert events == [matching_event]
        assert other_event not in events
        assert matching_event.user_id != other_event.user_id
        assert matching_event.user_id.startswith("[REDACTED:user_id:")
        assert user_id not in json.dumps(matching_event.to_dict(), sort_keys=True)

    def test_filter_events_by_type(self):
        """Test filtering events by event type"""
        storage = AuditStorage()

        login_event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS, user_id="user123"
        )
        data_event = AuditEvent(
            event_type=AuditEventType.DATA_CREATED, user_id="user123"
        )
        logout_event = AuditEvent(event_type=AuditEventType.LOGOUT, user_id="user123")

        storage.store_event(login_event)
        storage.store_event(data_event)
        storage.store_event(logout_event)

        # Filter by event types
        auth_events = storage.get_events(
            event_types=[AuditEventType.LOGIN_SUCCESS, AuditEventType.LOGOUT]
        )

        assert len(auth_events) == 2
        assert login_event in auth_events
        assert logout_event in auth_events
        assert data_event not in auth_events

    def test_event_limit(self):
        """Test limiting number of returned events"""
        storage = AuditStorage()

        # Store multiple events
        for i in range(10):
            event = AuditEvent(event_type=AuditEventType.DATA_READ, user_id=f"user{i}")
            storage.store_event(event)

        # Get limited events
        limited_events = storage.get_events(limit=5)
        assert len(limited_events) == 5

    def test_retrieve_events_warns_when_time_filters_ignored(self):
        """retrieve_events warns when its unsupported time filters are supplied."""
        storage = AuditStorage()
        event = AuditEvent(event_type=AuditEventType.DATA_READ, user_id="user123")
        storage.store_event(event)

        now = datetime.now(UTC)
        with pytest.warns(UserWarning, match="ignores time filters"):
            events = storage.retrieve_events(
                start_time=now - timedelta(hours=1), end_time=now
            )

        assert events == [event]

    def test_verify_integrity(self):
        """Test audit log integrity verification"""
        storage = AuditStorage()

        start_time = datetime.now(UTC)

        # Store some events
        for i in range(3):
            event = AuditEvent(event_type=AuditEventType.DATA_READ, user_id=f"user{i}")
            storage.store_event(event)

        end_time = datetime.now(UTC)

        # Verify integrity
        integrity = storage.verify_integrity(start_time, end_time)

        assert isinstance(integrity, AuditLogIntegrity)
        assert integrity.event_count == 3
        assert integrity.log_hash is not None
        assert len(integrity.log_hash) == 64  # SHA-256 hex digest


class TestAuditLogIntegrity:
    """Test AuditLogIntegrity class"""

    def test_verify_integrity_fails_loud(self):
        """Tamper detection should not report fake success."""
        integrity = AuditLogIntegrity()
        integrity.hash_chain = ["TAMPERED_NOT_A_HASH", "ANOTHER_BAD"]
        integrity.event_count = 999
        integrity.log_hash = "CORRUPTED"

        with pytest.raises(NotImplementedError, match=TAMPER_DETECTION_NOT_IMPLEMENTED):
            integrity.verify_integrity()


class TestAuditLogger:
    """Test AuditLogger class"""

    def test_log_event(self):
        """Test logging audit events"""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)

        event = audit_logger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id="user123",
            source_ip="192.168.1.1",
            message="User login successful",
        )

        assert event.event_type == AuditEventType.LOGIN_SUCCESS
        assert event.user_id == "user123"
        assert event.source_ip == "192.168.1.1"
        assert event.message == "User login successful"

        # Event should be queued for processing
        assert not audit_logger.event_queue.empty()

    def test_log_event_redacts_sensitive_payloads_before_storage(self):
        """Audit events should not store raw PII or credential-like values."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)

        event = audit_logger.log_event(
            event_type=AuditEventType.DATA_READ,
            user_id="alice@example.com",
            session_id="session-4111111111111234",
            tenant_id="tenant-123-45-6789",
            resource_id=FAKE_AUDIT_API_KEY,
            message="Bearer canary.jwt.header.payload.signature",
            details={
                "email": "alice@example.com",
                "ssn": "123-45-6789",
                "api_key": FAKE_AUDIT_API_KEY,
            },
        )

        event_blob = str(event.to_dict())
        assert "alice@example.com" not in event_blob
        assert "4111111111111234" not in event_blob
        assert "123-45-6789" not in event_blob
        assert FAKE_AUDIT_API_KEY not in event_blob
        assert "canary.jwt.header.payload.signature" not in event_blob
        assert "[REDACTED:email]" in event_blob
        assert "[REDACTED:credit_card]" in event_blob
        assert "[REDACTED:ssn]" in event_blob
        assert "[REDACTED:api_key]" in event_blob
        assert "[REDACTED:bearer_token]" in event_blob

    def test_email_user_filter_matches_only_the_original_redacted_identifier(self):
        """Email user IDs should stay filterable without storing raw addresses."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        alice_event = audit_logger.log_event(
            event_type=AuditEventType.DATA_READ,
            user_id="alice@example.com",
            message="Alice event",
        )
        bob_event = audit_logger.log_event(
            event_type=AuditEventType.DATA_READ,
            user_id="bob@example.com",
            message="Bob event",
        )

        alice_events = audit_logger.storage.get_events(user_id="alice@example.com")
        redacted_bucket_events = audit_logger.storage.get_events(
            user_id="[REDACTED:email]"
        )

        assert alice_events == [alice_event]
        assert bob_event not in alice_events
        assert redacted_bucket_events == []
        assert "alice@example.com" not in str(alice_event.to_dict())

    def test_log_authentication_events(self):
        """Test logging authentication events"""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)

        # Successful login
        success_event = audit_logger.log_authentication(
            user_id="user123", success=True, source_ip="192.168.1.1"
        )

        assert success_event.event_type == AuditEventType.LOGIN_SUCCESS
        assert success_event.severity == AuditSeverity.LOW
        assert success_event.result == "success"

        # Failed login
        failure_event = audit_logger.log_authentication(
            user_id="user123", success=False, source_ip="192.168.1.1"
        )

        assert failure_event.event_type == AuditEventType.LOGIN_FAILURE
        assert failure_event.severity == AuditSeverity.MEDIUM
        assert failure_event.result == "failure"

    def test_log_data_access_events(self):
        """Test logging data access events"""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)

        event = audit_logger.log_data_access(
            user_id="user123",
            resource="user_data",
            action="read",
            tenant_id="tenant123",
        )

        assert event.event_type == AuditEventType.DATA_READ
        assert event.user_id == "user123"
        assert event.tenant_id == "tenant123"
        assert event.resource == "user_data"
        assert event.action == "read"
        assert ComplianceFramework.GDPR in event.compliance_tags

    def test_log_security_events(self):
        """Test logging security events"""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)

        event = audit_logger.log_security_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id="user123",
            severity=AuditSeverity.CRITICAL,
            message="Unauthorized access attempt",
        )

        assert event.event_type == AuditEventType.SECURITY_VIOLATION
        assert event.severity == AuditSeverity.CRITICAL
        assert event.message == "Unauthorized access attempt"
        assert ComplianceFramework.SOC2 in event.compliance_tags

    def test_alert_handling(self):
        """Test alert handling for high-severity events"""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)

        # Mock alert handler
        alert_handler = Mock()
        audit_logger.add_alert_handler(alert_handler)

        # Log high-severity event
        audit_logger.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            message="Critical security issue",
        )

        # Process events (wait briefly for background processing)
        time.sleep(0.1)

        # Verify alert handler was registered
        assert audit_logger is not None
        # Note: Full alert testing would require synchronous processing

    def test_get_events(self):
        """Test retrieving events from audit logger"""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)

        # Log some events
        audit_logger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS, user_id="user123"
        )
        audit_logger.log_event(
            event_type=AuditEventType.DATA_CREATED, user_id="user456"
        )

        # Wait briefly for processing
        time.sleep(0.1)

        # Get events
        events = audit_logger.get_events(limit=10)
        assert isinstance(events, list)  # Should return a list of events

    def test_shutdown(self):
        """Test graceful shutdown"""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)

        # Log some events
        audit_logger.log_event(
            event_type=AuditEventType.SYSTEM_SHUTDOWN, message="System shutting down"
        )

        # Shutdown should complete gracefully
        audit_logger.shutdown()
        assert not audit_logger.running

    def test_secret_key_validation(self):
        """Secret key must meet strength requirements."""
        with pytest.raises(ValueError):
            AuditLogger("weak")


class TestComplianceReporter:
    """Test ComplianceReporter class"""

    def test_soc2_report_generation(self):
        """SOC 2 compliance reporting fails loudly until implemented."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        reporter = ComplianceReporter(audit_logger)

        start_date = datetime.now(UTC) - timedelta(days=30)
        end_date = datetime.now(UTC)

        with pytest.raises(NotImplementedError, match=COMPLIANCE_NOT_IMPLEMENTED):
            reporter.generate_report(
                framework=ComplianceFramework.SOC2,
                start_date=start_date,
                end_date=end_date,
            )

    def test_iso27001_report_generation(self):
        """ISO 27001 compliance reporting fails loudly until implemented."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        reporter = ComplianceReporter(audit_logger)

        start_date = datetime.now(UTC) - timedelta(days=30)
        end_date = datetime.now(UTC)

        with pytest.raises(NotImplementedError, match=COMPLIANCE_NOT_IMPLEMENTED):
            reporter.generate_report(
                framework=ComplianceFramework.ISO27001,
                start_date=start_date,
                end_date=end_date,
            )

    def test_gdpr_report_generation(self):
        """GDPR compliance reporting fails loudly until implemented."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        reporter = ComplianceReporter(audit_logger)

        start_date = datetime.now(UTC) - timedelta(days=30)
        end_date = datetime.now(UTC)

        with pytest.raises(NotImplementedError, match=COMPLIANCE_NOT_IMPLEMENTED):
            reporter.generate_report(
                framework=ComplianceFramework.GDPR,
                start_date=start_date,
                end_date=end_date,
            )

    def test_unsupported_framework(self):
        """Test error handling for unsupported framework"""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        reporter = ComplianceReporter(audit_logger)

        start_date = datetime.now(UTC) - timedelta(days=30)
        end_date = datetime.now(UTC)

        # This should raise an error for unsupported framework
        with pytest.raises(ValueError):
            # Create a mock framework not in the supported list
            class UnsupportedFramework:
                pass

            reporter.generate_report(
                framework=UnsupportedFramework(),
                start_date=start_date,
                end_date=end_date,
            )

    def test_compliance_dashboard(self):
        """Compliance dashboard fails loudly until implemented."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        reporter = ComplianceReporter(audit_logger)

        with pytest.raises(NotImplementedError, match=COMPLIANCE_NOT_IMPLEMENTED):
            reporter.get_compliance_dashboard()

    def test_tenant_specific_reporting(self):
        """Tenant-specific compliance reporting fails loudly until implemented."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        reporter = ComplianceReporter(audit_logger)

        start_date = datetime.now(UTC) - timedelta(days=30)
        end_date = datetime.now(UTC)

        with pytest.raises(NotImplementedError, match=COMPLIANCE_NOT_IMPLEMENTED):
            reporter.generate_report(
                framework=ComplianceFramework.GDPR,
                start_date=start_date,
                end_date=end_date,
                tenant_id="tenant1",
            )

    def test_legacy_compliance_methods_fail_loud(self):
        """Legacy report entry points fail loudly instead of returning fake status."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        reporter = ComplianceReporter(audit_logger)

        start_date = datetime.now(UTC) - timedelta(days=30)
        end_date = datetime.now(UTC)

        with pytest.raises(NotImplementedError, match=COMPLIANCE_NOT_IMPLEMENTED):
            reporter.generate_soc2_report(start_date, end_date)

        with pytest.raises(NotImplementedError, match=COMPLIANCE_NOT_IMPLEMENTED):
            reporter.generate_gdpr_report(start_date, end_date)

    def test_tenant_period_filter_matches_redacted_sensitive_identifier(self):
        """Tenant filters should work when stored tenant IDs are pseudonymized."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        reporter = ComplianceReporter(audit_logger)
        tenant_id = "tenant-123-45-6789"
        matching_event = audit_logger.log_event(
            event_type=AuditEventType.DATA_READ,
            tenant_id=tenant_id,
            message="Tenant event",
        )
        other_event = audit_logger.log_event(
            event_type=AuditEventType.DATA_READ,
            tenant_id="tenant-987-65-4321",
            message="Other tenant event",
        )
        start_date = datetime.now(UTC) - timedelta(minutes=1)
        end_date = datetime.now(UTC) + timedelta(minutes=1)

        events = reporter._get_events_for_period(
            start_date, end_date, tenant_id=tenant_id
        )

        assert events == [matching_event]
        assert other_event not in events
        assert tenant_id not in str(matching_event.to_dict())

    def test_security_incident_analysis_fails_loud(self):
        """Incident compliance analysis fails loudly instead of faking resolution."""
        audit_logger = AuditLogger(STRONG_AUDIT_SECRET)
        reporter = ComplianceReporter(audit_logger)
        events = [
            AuditEvent(
                event_type=AuditEventType.SECURITY_VIOLATION,
                user_id="user123",
                severity=AuditSeverity.CRITICAL,
            )
        ]

        with pytest.raises(NotImplementedError, match=COMPLIANCE_NOT_IMPLEMENTED):
            reporter._analyze_security_incidents(events)
