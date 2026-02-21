"""Unit tests for simplified sampler construction helpers."""

# Traceability: CONC-Layer-Core CONC-Quality-Usability
# FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import pytest

from traigent.core.samplers import RandomSampler, create_sampler


class TestCreateSampler:
    """Tests for create_sampler helper."""

    def test_create_sampler_defaults_to_random(self) -> None:
        sampler = create_sampler({"params": {"population": [1, 2, 3]}})

        assert isinstance(sampler, RandomSampler)

    def test_create_sampler_explicit_random_type(self) -> None:
        sampler = create_sampler(
            {"type": "random", "params": {"population": [10, 20], "seed": 7}}
        )

        assert isinstance(sampler, RandomSampler)
        assert sampler.sample() in {10, 20}

    def test_create_sampler_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported sampler type"):
            create_sampler({"type": "custom", "params": {"population": [1, 2, 3]}})

    def test_create_sampler_requires_population(self) -> None:
        with pytest.raises(TypeError, match="population"):
            create_sampler({})

    def test_create_sampler_propagates_random_sampler_validation(self) -> None:
        with pytest.raises(ValueError, match="sample_limit must be a positive integer"):
            create_sampler(
                {
                    "params": {
                        "population": [1, 2, 3],
                        "sample_limit": -1,
                    }
                }
            )

    def test_random_sampler_direct_usage_remains_supported(self) -> None:
        sampler = RandomSampler(population=[1, 2, 3], sample_limit=2, seed=0)

        assert sampler.sample() in {1, 2, 3}
        assert sampler.sample() in {1, 2, 3}
        assert sampler.sample() is None
        assert sampler.exhausted is True
