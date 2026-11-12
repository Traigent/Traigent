"""Registries must validate input and warn on override instead of silently replacing (#3159)."""

from __future__ import annotations

import logging

import pytest

from traigent.agents.executor import AgentExecutor
from traigent.agents.platforms import PlatformRegistry
from traigent.effectuation import contracts
from traigent.integrations import mappings, providers
from traigent.integrations.model_discovery import registry as disc
from traigent.integrations.model_discovery.base import ModelDiscovery


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    monkeypatch.setattr(
        mappings, "PARAMETER_MAPPINGS", dict(mappings.PARAMETER_MAPPINGS)
    )
    monkeypatch.setattr(mappings, "METHOD_MAPPINGS", dict(mappings.METHOD_MAPPINGS))
    monkeypatch.setattr(
        contracts, "STRATEGY_REGISTRY", dict(contracts.STRATEGY_REGISTRY)
    )
    monkeypatch.setattr(providers, "_MODEL_TIERS", dict(providers._MODEL_TIERS))
    monkeypatch.setattr(providers, "_FALLBACK_MODELS", dict(providers._FALLBACK_MODELS))
    monkeypatch.setattr(
        PlatformRegistry, "_executors", dict(PlatformRegistry._executors)
    )
    monkeypatch.setattr(disc, "_discovery_registry", dict(disc._discovery_registry))


def test_parameter_mapping_warns_copies_and_validates(caplog):
    src = {"x": "y"}
    with caplog.at_level(logging.WARNING):
        mappings.register_parameter_mapping("openai.OpenAI", src)
    assert "Overriding" in caplog.text
    src["z"] = "q"
    assert mappings.PARAMETER_MAPPINGS["openai.OpenAI"] == {"x": "y"}
    with pytest.raises(ValueError):
        mappings.register_parameter_mapping("", {"a": "b"})
    with pytest.raises(ValueError):
        mappings.register_parameter_mapping("a.B", {"a": 1})  # type: ignore[dict-item]


def test_method_mapping_warns_copies_and_validates(caplog):
    mappings.register_method_mapping("a.B", {"m": ["p"]})
    src = {"m": ["q"]}
    with caplog.at_level(logging.WARNING):
        mappings.register_method_mapping("a.B", src)
    assert "Overriding" in caplog.text
    src["m"].append("z")
    assert mappings.METHOD_MAPPINGS["a.B"] == {"m": ["q"]}
    with pytest.raises(ValueError):
        mappings.register_method_mapping("a.B", {"m": "notalist"})  # type: ignore[dict-item]


def test_strategy_duplicate_id_warns(caplog):
    class S:
        strategy_id = "dup-s"
        supported_kinds: set[str] = set()

    contracts.register_strategy(S())
    with caplog.at_level(logging.WARNING):
        contracts.register_strategy(S())
    assert "Overriding" in caplog.text


def test_provider_tiers_warns_validates_and_copies(caplog):
    providers.register_provider_tiers("dup_prov", {"fast": ["a"]})
    src = {"fast": ["b"]}
    with caplog.at_level(logging.WARNING):
        providers.register_provider_tiers("dup_prov", src)
    assert "Overriding" in caplog.text
    src["fast"].append("c")
    assert providers._MODEL_TIERS["dup_prov"] == {"fast": ["b"]}
    with pytest.raises(ValueError):
        providers.register_provider_tiers("", {"fast": ["a"]})
    with pytest.raises(ValueError):
        providers.register_provider_tiers("p", {"fast": "a"})  # type: ignore[dict-item]


def test_executor_override_warns_and_validates(caplog):
    class E(AgentExecutor):  # type: ignore[misc]
        pass

    with caplog.at_level(logging.WARNING):
        PlatformRegistry.register_executor("openai", E)
    assert "Overriding" in caplog.text
    with pytest.raises(ValueError):
        PlatformRegistry.register_executor("x", object)  # type: ignore[arg-type]


def test_discovery_override_warns_and_validates(caplog):
    class D(ModelDiscovery):  # type: ignore[misc]
        pass

    disc.register_discovery("dup-d", D)

    class D2(ModelDiscovery):  # type: ignore[misc]
        pass

    with caplog.at_level(logging.WARNING):
        disc.register_discovery("dup-d", D2)
    assert "Overriding" in caplog.text
    with pytest.raises(ValueError):
        disc.register_discovery("x", object)  # type: ignore[arg-type]
