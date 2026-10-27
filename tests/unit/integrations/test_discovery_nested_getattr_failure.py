"""A failing attribute access must not abort discovery of later attributes."""

from __future__ import annotations

from traigent.integrations.utils.discovery import ParameterDiscovery


class _ExplodingFirstAttribute:
    @property
    def aaa_broken(self):  # sorts first in dir()
        raise RuntimeError("attribute access failed")

    def create_completion(self, prompt: str = "") -> str:
        return prompt


def test_getattr_failure_on_first_attribute_does_not_abort_search() -> None:
    found: list[dict] = []

    ParameterDiscovery._search_nested_methods(_ExplodingFirstAttribute(), found)

    assert [entry["path"] for entry in found] == ["create_completion"]
