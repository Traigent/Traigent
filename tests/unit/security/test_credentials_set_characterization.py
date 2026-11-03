"""Characterization of EnhancedCredentialStore.set (validation, expiry, storage)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from traigent.security import credentials as credentials_module
from traigent.security.config import SecurityProfile
from traigent.security.credentials import (
    CredentialType,
    EnhancedCredentialStore,
    SecurityLevel,
)
from traigent.utils.exceptions import AuthenticationError

NOW = datetime(2030, 1, 1, 12, 0, 0, tzinfo=UTC)
WEAK_FIELD_VALUE = "demo-" + "secret-1"
GOOD_FIELD_VALUE = "Zq7kP2x" + "LmN"
GOOD = "Zq7#kP2x"  # exactly the minimum length, no weak pattern


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


class _Logger:
    def __init__(self, trace):
        self._trace = trace

    def _rec(self, level, msg, args):
        self._trace.append(("log", level, msg, args))

    def info(self, msg, *args, **kwargs):
        self._rec("info", msg, args)

    def warning(self, msg, *args, **kwargs):
        self._rec("warning", msg, args)

    def debug(self, msg, *args, **kwargs):
        self._rec("debug", msg, args)

    def error(self, msg, *args, **kwargs):
        self._rec("error", msg, args)


@pytest.fixture
def env(tmp_path, monkeypatch):
    store = EnhancedCredentialStore(
        master_password="test-passphrase-12345",  # noqa: S106  # pragma: allowlist secret
        storage_path=tmp_path / "creds",
        security_level=SecurityLevel.HIGH,
    )
    trace: list = []
    state = SimpleNamespace(
        flags=SimpleNamespace(
            profile=SecurityProfile.PRODUCTION, allow_weak_credentials=False
        )
    )

    def flags():
        trace.append(("flags",))
        return state.flags

    monkeypatch.setattr(credentials_module, "get_security_flags", flags)
    monkeypatch.setattr(credentials_module, "logger", _Logger(trace))
    monkeypatch.setattr(credentials_module, "datetime", _FrozenDatetime)

    real_event = store._security_event
    real_encrypt = store._encrypt_value
    real_save = store._save_secure_credentials
    real_audit = store._audit_operation

    def event(kind, details):
        trace.append(
            (
                "event",
                kind,
                dict(details),
                store.get_security_metrics()["short_value_blocked"],
            )
        )
        real_event(kind, details)

    def encrypt(value):
        trace.append(("encrypt", value))
        return real_encrypt(value)

    def save():
        trace.append(("save", store._lock._is_owned()))
        real_save()

    def audit(op, details):
        trace.append(("audit", op, dict(details), store._lock._is_owned()))
        real_audit(op, details)

    monkeypatch.setattr(store, "_security_event", event)
    monkeypatch.setattr(store, "_encrypt_value", encrypt)
    monkeypatch.setattr(store, "_save_secure_credentials", save)
    monkeypatch.setattr(store, "_audit_operation", audit)
    return SimpleNamespace(store=store, trace=trace, state=state)


def _kinds(trace):
    return [entry[0] for entry in trace]


def _nothing_stored(env):
    assert env.store._credentials == {}
    assert "encrypt" not in _kinds(env.trace)
    assert "save" not in _kinds(env.trace)
    assert "audit" not in _kinds(env.trace)


class TestKeywordHandling:
    def test_alias_overrides_credential_type(self, env):
        env.store.set("n", GOOD, CredentialType.SECRET, cred_type=CredentialType.TOKEN)
        assert env.store._credentials["n"].type is CredentialType.TOKEN

    def test_alias_with_unknown_keyword_lists_only_the_unknown(self, env):
        with pytest.raises(TypeError) as err:
            env.store.set("n", GOOD, cred_type=CredentialType.TOKEN, zeta=1, alpha=2)
        assert str(err.value) == "Unexpected keyword arguments: zeta, alpha"
        assert env.trace == []
        assert env.store._credentials == {}

    def test_unknown_keyword_is_rejected_before_flags(self, env):
        with pytest.raises(TypeError) as err:
            env.store.set("n", GOOD, other=1)
        assert str(err.value) == "Unexpected keyword arguments: other"
        assert env.trace == []

    def test_default_type_is_secret(self, env):
        env.store.set("n", GOOD)
        assert env.store._credentials["n"].type is CredentialType.SECRET


class TestValueValidation:
    def test_none_is_denied_after_flags_are_read(self, env):
        with pytest.raises(ValueError) as err:
            env.store.set("n", None)
        assert str(err.value) == "Credential value cannot be None"
        assert env.trace == [("flags",)]

    def test_empty_is_denied(self, env):
        with pytest.raises(ValueError) as err:
            env.store.set("n", "")
        assert str(err.value) == "Credential value cannot be empty"
        assert env.trace == [("flags",)]

    def test_non_string_without_length_propagates(self, env):
        with pytest.raises(TypeError):
            env.store.set("n", 12345678)
        assert env.trace == [("flags",)]
        assert env.store._credentials == {}

    @pytest.mark.parametrize("length", range(1, 8))
    def test_short_value_is_denied_with_full_evidence(self, env, length):
        value = GOOD[:length]
        with pytest.raises(ValueError) as err:
            env.store.set("short-name", value)
        assert str(err.value) == (
            "Credential value too short; minimum length is 8 characters"
        )
        details = {
            "name": "short-name",
            "length": length,
            "profile": "production",
            "minimum_length": 8,
        }
        assert env.trace == [
            ("flags",),
            # counter incremented before the event is raised
            ("event", "rejected_short_credential_value", details, 1),
            (
                "log",
                "warning",
                "Security event: type=%s security_level=%s",
                ("rejected_short_credential_value", "high"),
            ),
            (
                "log",
                "warning",
                "Rejecting short credential value for %s (length=%d) under profile %s",
                ("short-name", length, "production"),
            ),
        ]
        assert env.store.get_security_metrics()["short_value_blocked"] == 1
        _nothing_stored(env)

    def test_minimum_length_is_accepted(self, env):
        env.store.set("n", GOOD)
        assert len(GOOD) == 8
        assert "n" in env.store._credentials
        assert env.store.get_security_metrics()["short_value_blocked"] == 0

    def test_short_check_precedes_weak_check(self, env):
        with pytest.raises(ValueError):
            env.store.set("n", "admin")
        assert env.trace[1][1] == "rejected_short_credential_value"


class TestWeakValues:
    @pytest.mark.parametrize(
        "value",
        [
            "password123!",
            "abc12345xyz",
            "my-admin-key",
            "demo-secret1",
            "your-key-here",
            "xxxxxxxxxx",
            "placeholder1",
            "an-example-k",
            "UPPER-ADMIN-1",
        ],
    )
    def test_weak_value_is_denied(self, env, value):
        with pytest.raises(AuthenticationError) as err:
            env.store.set("n", value)
        assert str(err.value) == "Credential value appears to be weak or a placeholder"
        assert env.trace[0] == ("flags",)
        assert env.trace[1] == (
            "event",
            "weak_credential_detected",
            {"name": "n", "field": "value"},
            0,
        )
        assert _kinds(env.trace) == ["flags", "event", "log"]
        _nothing_stored(env)

    def test_weak_field_in_structured_value_is_named(self, env):
        value = json.dumps({"user": "admin", "client_secret": WEAK_FIELD_VALUE})
        with pytest.raises(AuthenticationError) as err:
            env.store.set("n", value)
        assert str(err.value) == (
            "Credential field 'client_secret' appears to be weak or a placeholder"
        )
        assert env.trace[1] == (
            "event",
            "weak_credential_detected",
            {"name": "n", "field": "client_secret"},
            0,
        )
        _nothing_stored(env)

    def test_non_secret_fields_of_structured_value_are_ignored(self, env):
        env.store.set(
            "n", json.dumps({"user": "admin@dev.local", "token": GOOD_FIELD_VALUE})
        )
        assert "n" in env.store._credentials

    def test_weak_test_value_is_allowed_only_when_flag_set(self, env):
        env.state.flags = SimpleNamespace(
            profile=SecurityProfile.DEVELOPMENT, allow_weak_credentials=True
        )
        env.store.set("n", "test_password123")
        assert "n" in env.store._credentials
        assert "event" not in _kinds(env.trace)

    @pytest.mark.parametrize(
        "value",
        ["test_password123", "dummy-admin-key", "sample-demo-1", "x_admin_test"],
    )
    def test_test_shaped_prefixes_and_suffixes_pass_when_allowed(self, env, value):
        env.state.flags = SimpleNamespace(
            profile=SecurityProfile.DEVELOPMENT, allow_weak_credentials=True
        )
        env.store.set("n", value)
        assert "n" in env.store._credentials

    def test_weak_test_value_is_denied_when_flag_unset(self, env):
        with pytest.raises(AuthenticationError):
            env.store.set("n", "test_password123")
        _nothing_stored(env)

    def test_weak_non_test_value_is_denied_even_when_flag_set(self, env):
        env.state.flags = SimpleNamespace(
            profile=SecurityProfile.DEVELOPMENT, allow_weak_credentials=True
        )
        with pytest.raises(AuthenticationError) as err:
            env.store.set("n", "my-admin-key")
        assert str(err.value) == "Credential value appears to be weak or a placeholder"
        _nothing_stored(env)

    def test_event_callback_error_replaces_the_denial(self, env):
        def boom(event):
            raise RuntimeError("callback")

        env.store.audit_callback = boom
        with pytest.raises(RuntimeError, match="callback"):
            env.store.set("n", "my-admin-key")
        _nothing_stored(env)


class TestExpiry:
    def test_no_expiry_by_default(self, env):
        env.store.set("n", GOOD)
        cred = env.store._credentials["n"]
        assert cred.expires_at is None
        assert cred.created_at == NOW

    def test_days_are_added_to_now(self, env):
        env.store.set("n", GOOD, expires_in_days=3)
        assert env.store._credentials["n"].expires_at == NOW + timedelta(days=3)

    def test_zero_days_means_no_expiry_at_standard_level(self, env):
        env.store.set("n", GOOD, expires_in_days=0)
        assert env.store._credentials["n"].expires_at is None

    def test_maximum_level_forces_ninety_days(self, env):
        env.store.security_level = SecurityLevel.MAXIMUM
        env.store.set("n", GOOD)
        assert env.store._credentials["n"].expires_at == NOW + timedelta(days=90)
        assert audit_expiry(env) == (NOW + timedelta(days=90)).isoformat()

    def test_maximum_level_treats_zero_days_as_unset(self, env):
        env.store.security_level = SecurityLevel.MAXIMUM
        env.store.set("n", GOOD, expires_in_days=0)
        assert env.store._credentials["n"].expires_at == NOW + timedelta(days=90)

    def test_maximum_level_honours_explicit_days(self, env):
        env.store.security_level = SecurityLevel.MAXIMUM
        env.store.set("n", GOOD, expires_in_days=5)
        assert env.store._credentials["n"].expires_at == NOW + timedelta(days=5)

    def test_explicit_time_beats_days(self, env):
        when = NOW + timedelta(days=40)
        env.store.set("n", GOOD, expires_in_days=3, expires_at=when)
        assert env.store._credentials["n"].expires_at == when

    def test_explicit_time_beats_maximum_default(self, env):
        env.store.security_level = SecurityLevel.MAXIMUM
        when = NOW + timedelta(days=400)
        env.store.set("n", GOOD, expires_at=when)
        assert env.store._credentials["n"].expires_at == when

    def test_naive_explicit_time_is_taken_as_utc(self, env):
        naive = datetime(2031, 5, 6, 7, 8, 9)
        env.store.set("n", GOOD, expires_at=naive)
        stored = env.store._credentials["n"].expires_at
        assert stored == naive.replace(tzinfo=UTC)
        assert stored.tzinfo is UTC

    def test_aware_explicit_time_keeps_its_zone(self, env):
        zone = timezone(timedelta(hours=5))
        aware = datetime(2031, 5, 6, 7, 8, 9, tzinfo=zone)
        env.store.set("n", GOOD, expires_at=aware)
        assert env.store._credentials["n"].expires_at.tzinfo is zone

    def test_past_explicit_time_is_recorded_not_denied(self, env):
        past = NOW - timedelta(microseconds=1)
        env.store.set("n", GOOD, expires_at=past)
        assert env.store._credentials["n"].expires_at == past
        decrypted = []
        env.store._decrypt_value = lambda *args: decrypted.append(args)
        assert env.store.get("n", check_env=False) is None
        assert decrypted == []
        event = next(e for e in env.trace if e[0] == "event")
        assert event == (
            "event",
            "credential_expiration_in_past",
            {"name": "n", "expires_at": past.isoformat()},
            0,
        )

    def test_present_instant_is_not_in_the_past(self, env):
        env.store.set("n", GOOD, expires_at=NOW)
        assert "event" not in _kinds(env.trace)
        assert env.store._credentials["n"].expires_at == NOW

    def test_past_event_follows_encryption(self, env):
        env.store.set("n", GOOD, expires_at=NOW - timedelta(days=1))
        assert _kinds(env.trace)[:3] == ["flags", "encrypt", "event"]

    def test_naive_past_time_is_normalized_before_comparison(self, env):
        env.store.set("n", GOOD, expires_at=datetime(2000, 1, 1))
        assert "event" in _kinds(env.trace)


def audit_expiry(env):
    return next(e for e in env.trace if e[0] == "audit")[2]["expires_at"]


class TestStorage:
    def test_success_sequence_and_lock_scope(self, env):
        env.store.set("n", GOOD, CredentialType.API_KEY, expires_in_days=1)
        assert env.trace == [
            ("flags",),
            ("encrypt", GOOD),
            ("save", True),
            ("log", "info", "Saved 1 secure credentials", ()),
            (
                "audit",
                "credential_created",
                {
                    "name": "n",
                    "type": "api_key",
                    "expires_at": (NOW + timedelta(days=1)).isoformat(),
                },
                False,
            ),
            ("log", "info", "Securely stored credential 'n'", ()),
        ]

    def test_audit_expiry_is_none_without_expiry(self, env):
        env.store.set("n", GOOD)
        assert audit_expiry(env) is None

    def test_record_fields(self, env):
        env.store.set("n", GOOD, metadata={"a": 1})
        cred = env.store._credentials["n"]
        assert cred.name == "n"
        assert cred.security_level is SecurityLevel.HIGH
        assert cred.metadata == {"a": 1}
        assert cred.created_at == NOW
        assert cred.last_accessed is None
        assert cred.access_count == 0
        assert len(cred.nonce) == env.store.NONCE_LENGTH
        assert cred.associated_data
        assert GOOD.encode() not in cred._encrypted_value

    def test_encrypted_parts_are_stored_together(self, env):
        parts = (b"cipher", b"nonce-bytes", b"aad")
        env.store._encrypt_value = lambda value: parts
        env.store.set("n", GOOD)
        cred = env.store._credentials["n"]
        assert (cred._encrypted_value, cred.nonce, cred.associated_data) == parts

    def test_value_round_trips(self, env):
        env.store.set("n", GOOD)
        assert env.store.get("n") == GOOD

    @pytest.mark.parametrize("given", [None, {}])
    def test_empty_metadata_becomes_new_dict(self, env, given):
        env.store.set("n", GOOD, metadata=given)
        assert env.store._credentials["n"].metadata == {}
        if given is not None:
            assert env.store._credentials["n"].metadata is not given

    def test_metadata_is_copied(self, env):
        meta = {"a": 1}
        env.store.set("n", GOOD, metadata=meta)
        meta["b"] = 2
        assert env.store._credentials["n"].metadata == {"a": 1}

    def test_existing_name_is_replaced(self, env):
        env.store.set("n", GOOD, metadata={"v": 1})
        env.store.set("n", "Lm9#wR4t", metadata={"v": 2})
        assert env.store._credentials["n"].metadata == {"v": 2}
        assert env.store.get("n") == "Lm9#wR4t"

    def test_encryption_error_propagates_without_side_effects(self, env):
        def boom(value):
            raise RuntimeError("encrypt")

        env.store._encrypt_value = boom
        with pytest.raises(RuntimeError, match="encrypt"):
            env.store.set("n", GOOD)
        assert env.store._credentials == {}
        assert "audit" not in _kinds(env.trace)

    def test_save_error_propagates_after_registration_and_skips_audit(self, env):
        def boom():
            raise RuntimeError("save")

        env.store._save_secure_credentials = boom
        with pytest.raises(RuntimeError, match="save"):
            env.store.set("n", GOOD)
        assert "n" in env.store._credentials
        assert "audit" not in _kinds(env.trace)
        assert ("log", "info", "Securely stored credential 'n'", ()) not in env.trace

    def test_audit_error_propagates_after_save(self, env):
        def boom(op, details):
            raise RuntimeError("audit")

        env.store._audit_operation = boom
        with pytest.raises(RuntimeError, match="audit"):
            env.store.set("n", GOOD)
        assert "n" in env.store._credentials
        assert ("log", "info", "Securely stored credential 'n'", ()) not in env.trace
