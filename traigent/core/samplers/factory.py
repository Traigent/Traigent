"""Sampler registry and construction helpers."""

# Traceability: CONC-Layer-Core CONC-Quality-Usability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import BaseSampler
from .random_sampler import RandomSampler


class SamplerFactory:
    """Central registry for sampler implementations."""

    _registry: dict[str, type[BaseSampler]] = {
        "random": RandomSampler,
    }

    @classmethod
    def register(cls, name: str, sampler_cls: type[BaseSampler]) -> None:
        if not name:
            raise ValueError("Sampler name must be non-empty.")
        if not issubclass(sampler_cls, BaseSampler):
            raise TypeError("sampler_cls must inherit from BaseSampler.")
        cls._registry[name.lower()] = sampler_cls

    @classmethod
    def create(cls, config: Mapping[str, Any] | None = None) -> BaseSampler:
        config = config or {}
        sampler_type = str(config.get("type", "random")).lower()
        params = dict(config.get("params", {}))

        sampler_cls = cls._registry.get(sampler_type)
        if sampler_cls is None:
            raise ValueError(
                f"Unknown sampler '{sampler_type}'. "
                f"Registered: {sorted(cls._registry.keys())}"
            )

        return sampler_cls(**params)
