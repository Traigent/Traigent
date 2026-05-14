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
        """Validate user data after initialization.

        Role-validation contract (SDK#939):
          * `roles=None`/`[]`/falsy → safe default `["user"]`. Use this
            for programmatic User construction without an IdP claim.
          * `roles=["admin", "operator"]` → preserved (lowercased+trimmed).
          * `roles=[123, 456]` (non-empty AND every item invalid) → raises
            ValueError. Pre-fix this silently demoted to `["user"]`,
            hiding upstream IdP misconfigurations.
          * `roles=["admin", 123]` (mixed valid + invalid) → raises
            ValueError. Mirrors `sanitize_roles(strict=True)` semantics:
            ANY malformed item in a caller-provided list signals
            confusion or attack and must surface as a hard error.

        OIDC/SAML callers already pre-sanitize via
        `helpers.sanitize_roles(strict=True)`; this defense-in-depth
        validation catches direct `User(...)` construction outside the
        OIDC/SAML happy path.
        """
        if not self.user_id or not isinstance(self.user_id, str):
            raise ValueError("Invalid user_id") from None

        if not self.username or not re.match(r"^[a-zA-Z0-9_-]{3,32}$", self.username):
            raise ValueError("Invalid username format")

        if not self.email or not EMAIL_PATTERN.match(self.email):
            raise ValueError("Invalid email format")

        # Role normalization (SDK#939 fail-closed):
        # Three cases (see method docstring above):
        #   1. Empty/falsy → safe default ["user"].
        #   2. Non-empty AND every item invalid → raise.
        #   3. Non-empty AND any item invalid → raise (mirrors
        #      sanitize_roles(strict=True) — defense-in-depth, per
        #      Codex Q3 of PR #969).
        roles_provided = bool(self.roles)  # truthy = caller asserted SOMETHING
        if not isinstance(self.roles, list):
            self.roles = [str(self.roles)] if self.roles else []
        else:
            self.roles = list(self.roles)

        if roles_provided:
            normalized: list[str] = []
            for r in self.roles:
                if isinstance(r, str) and r.strip():
                    normalized.append(r.strip().lower())
                else:
                    raise ValueError(
                        f"Invalid roles: at least one item is not a non-empty "
                        f"string (SDK#939 fail-closed; mirrors "
                        f"sanitize_roles(strict=True) semantics). Got "
                        f"item of type {type(r).__name__}={r!r}. Pass "
                        f"roles=None or roles=[] for the default ['user'] "
                        f"role; do not mix valid and invalid items."
                    )
            self.roles = normalized
        else:
            # Case 1: nothing provided → safe default.
            self.roles: list[Any] = ["user"]

        if not isinstance(self.metadata, dict):
            self.metadata: dict[str, Any] = {}
