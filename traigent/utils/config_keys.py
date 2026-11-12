"""Shared rule for tuned-variable (configuration_space) key names."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

# Dot-separated identifier segments; same grammar as the JavaScript SDK and the
# TVL loader.
_CONFIG_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


def validate_configuration_space_keys(keys: Iterable[Any]) -> None:
    """Raise ``ValueError`` if any key is not a dot-separated identifier.

    The offending key is not echoed; the message gives its position instead.
    """
    for index, key in enumerate(keys):
        if not isinstance(key, str) or not _CONFIG_KEY_RE.match(key):
            raise ValueError(
                f"configuration_space key #{index + 1} is not a valid identifier: "
                "keys must be dot-separated segments of letters, digits and "
                "underscores, each starting with a letter or underscore"
            )
