"""User model for authentication."""

# Traceability: CONC-Layer-Infra CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .helpers import EMAIL_PATTERN


@dataclass
class User:
    """User model for authentication."""

    user_id: str
    username: str
    email: str
    roles: list[str]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate user data after initialization."""
        if not self.user_id or not isinstance(self.user_id, str):
            raise ValueError("Invalid user_id") from None

        if not self.username or not re.match(r"^[a-zA-Z0-9_-]{3,32}$", self.username):
            raise ValueError("Invalid username format")

        if not self.email or not EMAIL_PATTERN.match(self.email):
            raise ValueError("Invalid email format")

        # Role normalization (SDK#939):
        # The auth flow callers (OIDC at oidc.py:138, SAML at saml.py:114)
        # already pass `roles` through `sanitize_roles(strict=True)`,
        # which raises on invalid IdP claims. This block is defense-in-
        # depth for direct `User(...)` construction outside the
        # OIDC/SAML happy path.
        #
        # Distinguish two cases:
        #   1. Caller passes `roles=None`/`[]`/`""`/falsy → no roles
        #      provided. Default to `["user"]` (legitimate default for
        #      programmatic User construction without an IdP).
        #   2. Caller passes a non-empty `roles` value but every item
        #      filters out as invalid (non-strings, empty strings) →
        #      raise. This is the silent-fallback bug class #939
        #      flagged: previously the User would quietly become a
        #      `["user"]`-roled account, hiding the upstream
        #      misconfiguration.
        roles_provided = bool(self.roles)  # truthy = caller asserted SOMETHING
        if not isinstance(self.roles, list):
            self.roles = [str(self.roles)] if self.roles else []
        else:
            self.roles = list(self.roles)

        normalized = [
            r.strip().lower() for r in self.roles if isinstance(r, str) and r.strip()
        ]
        if not normalized:
            if roles_provided:
                # Case 2: caller provided roles but all were invalid.
                raise ValueError(
                    "Invalid roles: all provided role claims filtered out as "
                    "invalid (SDK#939 fail-closed). Pass `roles=None` or "
                    "`roles=[]` for the default ['user'] role; do not pass a "
                    "non-empty roles list with no string items."
                )
            # Case 1: nothing provided → safe default.
            self.roles: list[Any] = ["user"]
        else:
            self.roles = normalized

        if not isinstance(self.metadata, dict):
            self.metadata: dict[str, Any] = {}
