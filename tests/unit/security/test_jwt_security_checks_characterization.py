"""Characterization of SecureJWTValidator._perform_security_checks."""

from __future__ import annotations

import logging

import pytest

from traigent.security.jwt_validator import (
    JWTSecurityError,
    SecureJWTValidator,
    ValidationMode,
)

LOGGER_NAME = "traigent.security.jwt_validator"


@pytest.fixture
def validator(monkeypatch: pytest.MonkeyPatch) -> SecureJWTValidator:
    monkeypatch.delenv("TRAIGENT_JWT_BYPASS", raising=False)
    monkeypatch.delenv("TRAIGENT_ALLOWED_SCOPES", raising=False)
    return SecureJWTValidator(
        validation_mode=ValidationMode.DEVELOPMENT,
        issuer="iss",
        audience="aud",
        development_public_key="unused",
    )


def _warnings(caplog: pytest.LogCaptureFixture) -> list[tuple[str, int, str]]:
    return [
        (r.name, r.levelno, r.getMessage())
        for r in caplog.records
        if r.name == LOGGER_NAME and r.getMessage().startswith("Suspicious")
    ]


class TestSuspiciousClaims:
    def test_each_true_claim_is_warned_in_fixed_order(self, validator, caplog):
        payload = {"bypass": True, "admin": True, "superuser": True, "root": True}
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            assert validator._perform_security_checks(payload) is None
        assert _warnings(caplog) == [
            (LOGGER_NAME, logging.WARNING, "Suspicious claim detected: admin"),
            (LOGGER_NAME, logging.WARNING, "Suspicious claim detected: root"),
            (LOGGER_NAME, logging.WARNING, "Suspicious claim detected: superuser"),
            (LOGGER_NAME, logging.WARNING, "Suspicious claim detected: bypass"),
        ]

    @pytest.mark.parametrize("value", [False, "true", 1, None, "yes"])
    def test_only_boolean_true_triggers_warning(self, validator, caplog, value):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            validator._perform_security_checks({"admin": value})
        assert _warnings(caplog) == []

    def test_unrelated_claims_do_not_warn(self, validator, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            validator._perform_security_checks({"sub": "u", "Admin": True})
        assert _warnings(caplog) == []

    def test_warning_does_not_deny(self, validator):
        validator._perform_security_checks({"admin": True, "scope": "anything"})


class TestScopeEnforcement:
    def test_no_allowlist_accepts_any_scope(self, validator):
        validator._perform_security_checks({"scope": "whatever other"})

    @pytest.mark.parametrize("env", ["", "   ", " , ,, "])
    def test_blank_allowlist_means_unrestricted(self, validator, monkeypatch, env):
        monkeypatch.setenv("TRAIGENT_ALLOWED_SCOPES", env)
        validator._perform_security_checks({"scope": "anything"})

    def test_string_scope_inside_allowlist_passes(self, validator, monkeypatch):
        monkeypatch.setenv("TRAIGENT_ALLOWED_SCOPES", " read , write ,")
        validator._perform_security_checks({"scope": "read write"})

    def test_string_scope_outside_allowlist_denied_first_offender(
        self, validator, monkeypatch
    ):
        monkeypatch.setenv("TRAIGENT_ALLOWED_SCOPES", "read")
        with pytest.raises(JWTSecurityError) as exc:
            validator._perform_security_checks({"scope": "read delete admin"})
        assert str(exc.value) == "Invalid scope: delete"
        assert exc.value.__cause__ is None

    def test_list_scope_outside_allowlist_denied(self, validator, monkeypatch):
        monkeypatch.setenv("TRAIGENT_ALLOWED_SCOPES", "read,write")
        with pytest.raises(JWTSecurityError) as exc:
            validator._perform_security_checks({"scope": ["read", "nope", "x"]})
        assert str(exc.value) == "Invalid scope: nope"

    def test_list_scope_inside_allowlist_passes(self, validator, monkeypatch):
        monkeypatch.setenv("TRAIGENT_ALLOWED_SCOPES", "read,write")
        validator._perform_security_checks({"scope": ["write", "read"]})

    def test_empty_scope_string_passes_when_allowlist_set(self, validator, monkeypatch):
        monkeypatch.setenv("TRAIGENT_ALLOWED_SCOPES", "read")
        validator._perform_security_checks({"scope": ""})

    def test_absent_scope_skips_check(self, validator, monkeypatch):
        monkeypatch.setenv("TRAIGENT_ALLOWED_SCOPES", "read")
        validator._perform_security_checks({"sub": "u"})

    def test_non_string_list_member_denied(self, validator, monkeypatch):
        monkeypatch.setenv("TRAIGENT_ALLOWED_SCOPES", "read")
        with pytest.raises(JWTSecurityError) as exc:
            validator._perform_security_checks({"scope": [1]})
        assert str(exc.value) == "Invalid scope: 1"

    @pytest.mark.parametrize("scope", [None, 5])
    def test_malformed_non_iterable_scope_raises_type_error(
        self, validator, monkeypatch, scope
    ):
        monkeypatch.setenv("TRAIGENT_ALLOWED_SCOPES", "read")
        with pytest.raises(TypeError):
            validator._perform_security_checks({"scope": scope})

    @pytest.mark.parametrize("scope", [None, 5])
    def test_malformed_scope_ignored_without_allowlist(self, validator, scope):
        validator._perform_security_checks({"scope": scope})

    def test_warning_precedes_scope_denial(self, validator, monkeypatch, caplog):
        monkeypatch.setenv("TRAIGENT_ALLOWED_SCOPES", "read")
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            with pytest.raises(JWTSecurityError):
                validator._perform_security_checks({"admin": True, "scope": "x"})
        assert _warnings(caplog) == [
            (LOGGER_NAME, logging.WARNING, "Suspicious claim detected: admin")
        ]

    def test_allowlist_read_only_when_scope_present(self, validator, monkeypatch):
        calls: list[str] = []
        import traigent.security.jwt_validator as mod

        real = mod.os.getenv

        def spy(key, *a, **k):
            if key == "TRAIGENT_ALLOWED_SCOPES":
                calls.append(key)
            return real(key, *a, **k)

        monkeypatch.setattr(mod.os, "getenv", spy)
        validator._perform_security_checks({"sub": "u"})
        assert calls == []
        validator._perform_security_checks({"scope": "x"})
        assert calls == ["TRAIGENT_ALLOWED_SCOPES"]
