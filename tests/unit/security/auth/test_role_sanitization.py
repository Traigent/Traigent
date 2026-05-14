"""Role-claim sanitization tests for IdP-authenticated users."""

from __future__ import annotations

import pytest

from traigent.security.auth.helpers import sanitize_roles


def test_sanitize_roles_defaults_only_when_claim_is_absent() -> None:
    assert sanitize_roles(None, strict=True) == ["user"]
    assert sanitize_roles([], strict=True) == ["user"]


def test_sanitize_roles_strict_rejects_malformed_claims() -> None:
    with pytest.raises(ValueError, match="Invalid role claim"):
        sanitize_roles(["admin", "bad role with spaces"], strict=True)

    with pytest.raises(ValueError, match="Invalid role claim"):
        sanitize_roles({"role": "admin"}, strict=True)


def test_sanitize_roles_strict_accepts_common_idp_separators() -> None:
    assert sanitize_roles(
        ["admin", "api:read", "team.member", "org/admin"], strict=True
    ) == ["admin", "api:read", "team.member", "org/admin"]


def test_sanitize_roles_legacy_non_strict_falls_back() -> None:
    assert sanitize_roles(["bad role with spaces"]) == ["user"]
