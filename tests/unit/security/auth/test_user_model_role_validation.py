"""SDK#939 regression tests for User model role validation.

These tests pin the fail-closed behavior of `User.__post_init__`:
no roles → safe default; non-empty invalid roles → raise.

Issue: https://github.com/Traigent/Traigent/issues/939
"""

from __future__ import annotations

import pytest

from traigent.security.auth.models import User

_VALID_USER_KWARGS = {
    "user_id": "u_123",
    "username": "alice",
    "email": "alice@example.com",
    "metadata": {},
}


class TestUserRoleValidation_SDK939:
    """Pin: User.__post_init__ does not silently demote invalid IdP
    role claims to ["user"]."""

    def test_user_with_no_roles_defaults_to_user(self):
        """`roles=None`/`[]` → safe default. This is the LEGITIMATE
        path: caller has no IdP claim and explicitly defaults."""
        # roles=None
        u = User(**_VALID_USER_KWARGS, roles=None)
        assert u.roles == ["user"]
        # roles=[]
        u = User(**_VALID_USER_KWARGS, roles=[])
        assert u.roles == ["user"]

    def test_user_with_valid_roles_preserves_them(self):
        """Valid string roles pass through (lowercased + trimmed)."""
        u = User(**_VALID_USER_KWARGS, roles=["Admin", " operator "])
        assert u.roles == ["admin", "operator"]

    def test_user_with_all_invalid_roles_raises(self):
        """SDK#939 fix: a non-empty roles list whose every item is
        invalid (non-strings, empty strings) must RAISE — not
        silently demote to ['user']. Pre-fix this returned ['user']
        and quietly hid the upstream misconfiguration."""
        invalid_inputs = [
            [123, 456],          # all non-strings
            [None, None],        # all None
            ["", "  "],          # all empty/whitespace
            [{"role": "admin"}], # dict, not string
        ]
        for invalid in invalid_inputs:
            with pytest.raises(ValueError, match="SDK#939"):
                User(**_VALID_USER_KWARGS, roles=invalid)

    def test_user_with_mixed_valid_and_invalid_roles_keeps_valid(self):
        """A mix of valid + invalid: keep the valid ones (caller
        provided some real roles, we preserve them; we only fail
        when EVERYTHING is invalid)."""
        u = User(**_VALID_USER_KWARGS, roles=["admin", 123, None, "user"])
        assert u.roles == ["admin", "user"]

    def test_user_with_string_role_not_list_coerced(self):
        """Backward-compat: a single string `roles="admin"` is coerced
        to `["admin"]` (preserves pre-fix behavior for this happy case)."""
        u = User(**_VALID_USER_KWARGS, roles="admin")
        assert u.roles == ["admin"]
