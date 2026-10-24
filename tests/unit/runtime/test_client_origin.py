"""RuntimeClient validates its backend origin structurally at construction."""

from __future__ import annotations

import pytest

from traigent.runtime.client import RuntimeClient
from traigent.runtime.model import RuntimeRequirementError


@pytest.mark.parametrize(
    "bad",
    [
        "ftp://api.example.com",
        "https://",
        "api.example.com",
        "https://user:pass@api.example.com",  # pragma: allowlist secret
        "https://user@api.example.com",
        "https://api.example.com?q=1",
        "https://api.example.com/;p=1",
        "https://api.example.com#frag",
        "https://api.example.com /x",
        "https://api.example.com\n",
        "https://api.example.com\x00",
        "https://api.example.com\t",
    ],
)
def test_invalid_origin_rejected_without_echo(bad: str) -> None:
    with pytest.raises(RuntimeRequirementError) as exc:
        RuntimeClient("p1", base_url=bad, api_key="k")
    assert exc.value.code == "invalid_requirement"
    assert "example" not in str(exc.value)


@pytest.mark.parametrize(
    ("good", "origin"),
    [
        ("https://api.example.com", "https://api.example.com"),
        ("http://localhost:5000", "http://localhost:5000"),
        ("https://api.example.com/", "https://api.example.com"),
        ("https://api.example.com/base//", "https://api.example.com/base"),
    ],
)
def test_valid_origin_accepted(good: str, origin: str) -> None:
    assert RuntimeClient("p1", base_url=good, api_key="k")._origin == origin
