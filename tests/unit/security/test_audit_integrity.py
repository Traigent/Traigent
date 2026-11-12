"""Tamper-evidence coverage for every persisted audit event field."""

from __future__ import annotations

import copy
from dataclasses import fields
from datetime import timedelta

import pytest

from traigent.security.audit import AuditEventType, AuditLogger, AuditSeverity

SECRET = "Str0ng!AuditSecret-ForTests-0123456789"  # pragma: allowlist secret

MUTATIONS = {
    "event_id": lambda e: "other-id",
    "event_type": lambda e: AuditEventType.LOGIN_SUCCESS,
    "user_id": lambda e: "mallory",
    "tenant_id": lambda e: "other-tenant",
    "message": lambda e: "edited",
    "severity": lambda e: AuditSeverity.CRITICAL,
    "timestamp": lambda e: e.timestamp + timedelta(seconds=1),
    "session_id": lambda e: "sess-x",
    "resource_id": lambda e: "res-x",
    "resource_type": lambda e: "type-x",
    "resource": lambda e: "resource-x",
    "action": lambda e: "delete",
    "result": lambda e: "success",
    "ip_address": lambda e: "9.9.9.9",
    "source_ip": lambda e: "8.8.8.8",
    "user_agent": lambda e: "evil/1.0",
    "details": lambda e: {"injected": True},
    "compliance_tags": lambda e: ["forged"],
}


def _logger_with_event() -> tuple[AuditLogger, object]:
    logger = AuditLogger(SECRET)
    logger.log_event(
        AuditEventType.LOGIN_FAILURE,
        user_id="alice",
        session_id="s1",
        tenant_id="t1",
        resource_id="r1",
        resource_type="rt",
        resource="res",
        action="login",
        result="failure",
        source_ip="1.2.3.4",
        user_agent="ua",
        message="failed",
        details={"k": "v"},
        severity=AuditSeverity.MEDIUM,
        compliance_tags=["gdpr"],
    )
    return logger, logger.events[0]


def test_mutation_table_covers_every_persisted_field() -> None:
    names = {f.name for f in fields(_logger_with_event()[1])} - {"checksum"}
    assert names == set(MUTATIONS)


def test_untouched_log_verifies_clean() -> None:
    logger, _ = _logger_with_event()
    result = logger.verify_integrity()
    assert result["tampered_events"] == []
    assert result["legacy_events"] == []
    assert result["chain_integrity"] is True


@pytest.mark.parametrize("field_name", sorted(MUTATIONS))
def test_editing_any_field_fails_verification(field_name: str) -> None:
    logger, event = _logger_with_event()
    original = copy.deepcopy(getattr(event, field_name))
    new_value = MUTATIONS[field_name](event)
    assert new_value != original
    setattr(event, field_name, new_value)
    result = logger.verify_integrity()
    assert [t["event_id"] for t in result["tampered_events"]] == [event.event_id]


def test_legacy_checksum_is_flagged_legacy_not_valid() -> None:
    logger, event = _logger_with_event()
    event.checksum = logger._calculate_legacy_checksum(event, None)
    result = logger.verify_integrity()
    assert result["tampered_events"] == []
    assert result["legacy_events"] == [event.event_id]


def test_legacy_entry_with_edit_is_tampered() -> None:
    logger, event = _logger_with_event()
    event.checksum = logger._calculate_legacy_checksum(event, None)
    event.user_id = "mallory"
    result = logger.verify_integrity()
    assert len(result["tampered_events"]) == 1
    assert result["legacy_events"] == []


def test_missing_checksum_is_tampered() -> None:
    logger, event = _logger_with_event()
    event.checksum = None
    assert len(logger.verify_integrity()["tampered_events"]) == 1


def test_integrity_status_values() -> None:
    logger, event = _logger_with_event()
    assert logger.verify_integrity()["integrity_status"] == "verified"
    event.user_id = "mallory"
    assert logger.verify_integrity()["integrity_status"] == "tampered"


def test_legacy_entries_are_never_a_plain_pass() -> None:
    logger, event = _logger_with_event()
    event.checksum = logger._calculate_legacy_checksum(event, None)
    logger.event_chain_hash = None
    logger.event_chain_hash = logger._update_chain_hash(event)  # as an old log
    # Edit a field the legacy checksum never covered.
    event.tenant_id = "other-tenant"
    result = logger.verify_integrity()
    assert result["tampered_events"] == []
    assert result["integrity_status"] == "indeterminate_legacy"
    assert result["chain_integrity"] is False


COLLISION_PAIRS = [
    ({1}, ["1"]),
    (1, "1"),
    ((1, 2), [1, 2]),
    (True, 1),
    (1, 1.0),
    (None, "None"),
    ({1, 2}, frozenset({1, 2})),
    ({"a": 1}, {"a": "1"}),
]


@pytest.mark.parametrize(("left", "right"), COLLISION_PAIRS)
def test_canonicalisation_distinguishes_types(left: object, right: object) -> None:
    logger, event = _logger_with_event()
    event.details = {"v": left}
    a = logger._calculate_checksum_for_verification(event, None)
    event.details = {"v": right}
    b = logger._calculate_checksum_for_verification(event, None)
    assert a != b


def test_every_value_type_has_its_own_tag() -> None:
    from datetime import UTC, datetime

    samples = {
        "none": None,
        "bool": True,
        "int": 1,
        "float": 1.5,
        "str": "x",
        "bytes": b"x",
        "datetime": datetime(2026, 1, 1, tzinfo=UTC),
        "enum": AuditSeverity.LOW,
        "dict": {"a": 1},
        "list": [1],
        "tuple": (1,),
        "set": {1},
        "frozenset": frozenset({1}),
        "other": object,
    }
    tags = {name: AuditLogger._canon(v)[0] for name, v in samples.items()}
    assert len(set(tags.values())) == len(tags), tags
