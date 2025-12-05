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

        if not isinstance(self.roles, list):
            self.roles = [str(self.roles)] if self.roles else ["user"]

        self.roles = [r.strip().lower() for r in self.roles if isinstance(r, str)]
        if not self.roles:
            self.roles: list[Any] = ["user"]

        if not isinstance(self.metadata, dict):
            self.metadata: dict[str, Any] = {}
