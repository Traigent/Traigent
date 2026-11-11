"""register_discovery() evicts the cached instance of the replaced class (Traigent#3158)."""

from __future__ import annotations

import pytest

from traigent.integrations.model_discovery import registry


class _A:
    pass


class _B:
    pass


@pytest.fixture(autouse=True)
def _isolate():
    saved = (dict(registry._discovery_registry), dict(registry._discovery_instances))
    yield
    registry._discovery_registry.clear()
    registry._discovery_registry.update(saved[0])
    registry._discovery_instances.clear()
    registry._discovery_instances.update(saved[1])


def test_reregistering_replaces_cached_instance() -> None:
    registry.register_discovery("demo-3158", _A)
    assert isinstance(registry.get_model_discovery("demo-3158"), _A)

    registry.register_discovery("demo-3158", _B)

    assert isinstance(registry.get_model_discovery("demo-3158"), _B)
