"""Sampler interfaces and default implementations.
# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

Example
-------

::

    from traigent.core.samplers import create_sampler

    sampler = create_sampler(
        {
            "params": {
                "population": list(range(100)),
                "sample_limit": 10,
                "replace": False,
                "seed": 42,
            },
        }
    )

    while True:
        sample = sampler.sample()
        if sample is None:
            break
        process(sample)

"""

# Traceability: CONC-Layer-Core CONC-Quality-Usability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from collections.abc import Mapping
from typing import Any

from .random_sampler import RandomSampler, RandomSamplerPlan


def create_sampler(config: Mapping[str, Any] | None = None) -> RandomSampler[Any]:
    """Create a random sampler from a compact config payload.

    The sampler module currently supports a single implementation (`RandomSampler`).
    """
    config = config or {}
    sampler_type = str(config.get("type", "random")).lower()
    if sampler_type != "random":
        raise ValueError(
            f"Unsupported sampler type {sampler_type!r}; only 'random' is available."
        )

    params = dict(config.get("params", {}))
    return RandomSampler(**params)


__all__ = ["RandomSampler", "RandomSamplerPlan", "create_sampler"]
