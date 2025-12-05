"""Sampler interfaces and default implementations.
# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

Example
-------

::

    from traigent.core.samplers import SamplerFactory

    sampler = SamplerFactory.create(
        {
            "type": "random",
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

from .base import BaseSampler
from .factory import SamplerFactory
from .random_sampler import RandomSampler, RandomSamplerPlan

__all__ = ["BaseSampler", "RandomSampler", "RandomSamplerPlan", "SamplerFactory"]
