"""Unit tests for sampler utilities."""

from __future__ import annotations

import asyncio
import threading

import pytest

from traigent.core.samplers import RandomSampler
from traigent.core.samplers.random_sampler import RandomSamplerPlan


def test_random_sampler_without_replacement_exhausts_population():
    population = [1, 2, 3]
    sampler = RandomSampler(population, replace=False, sample_limit=None, seed=42)

    seen: set[int] = set()
    for _ in range(len(population)):
        sample = sampler.sample()
        assert sample is not None
        assert sample not in seen  # no duplicates when replace=False
        seen.add(sample)

    # Further sampling should return None once the population is exhausted.
    assert sampler.sample() is None
    assert sampler.exhausted


def test_random_sampler_reset_restores_state():
    sampler = RandomSampler([7], sample_limit=1, replace=True, seed=123)

    assert sampler.sample() == 7
    assert sampler.sample() is None  # limit reached

    sampler.reset()
    assert not sampler.exhausted
    assert sampler.sample() == 7  # reset restores ability to draw again


def test_random_sampler_clone_produces_independent_instances():
    population = ["a", "b", "c"]
    sampler = RandomSampler(population, replace=False, seed=99)

    clone = sampler.clone()

    # Draw from the original sampler; it should transition toward exhaustion.
    first_original = sampler.sample()
    assert first_original in population

    # Clone should still have access to the full population.
    clone_samples = {clone.sample(), clone.sample(), clone.sample()}
    assert clone_samples == set(population)
    assert clone.exhausted

    # Original sampler should still be able to draw the remaining items.
    remaining = {sampler.sample(), sampler.sample()}
    remaining.discard(None)
    assert remaining == set(population) - {first_original}
    assert sampler.exhausted


def test_random_sampler_plan_roundtrip():
    population = list(range(10))
    sampler = RandomSampler(population, replace=False, sample_limit=5, seed=5)

    plan = sampler.create_plan()
    sampler.apply_plan(plan)

    draws = [sampler.sample() for _ in range(len(plan))]
    assert sampler.sample() is None
    assert sampler.exhausted

    expected = [population[idx] for idx in plan.indices]
    assert draws == expected

    sampler.reset()
    sampler.apply_plan(plan)
    assert [sampler.sample() for _ in range(len(plan))] == expected


def test_random_sampler_plan_resume_random():
    population = list(range(6))
    sampler = RandomSampler(population, replace=False, sample_limit=None, seed=11)

    plan = sampler.create_plan(draws=2)
    sampler.apply_plan(plan, resume_random_after_plan=True)

    first_two = [sampler.sample(), sampler.sample()]
    assert first_two == [population[idx] for idx in plan.indices]

    remaining = set()
    while True:
        value = sampler.sample()
        if value is None:
            break
        remaining.add(value)

    assert remaining == set(population) - set(first_two)
    assert sampler.exhausted


def test_random_sampler_plan_serialization_and_validation():
    population = list(range(4))
    sampler = RandomSampler(population, replace=False, seed=7)

    plan = sampler.create_plan(draws=3)
    plan_dict = plan.to_dict()
    restored = RandomSamplerPlan.from_dict(plan_dict)

    clone = RandomSampler(population, replace=False, seed=19)
    clone.apply_plan(restored)
    results = [clone.sample() for _ in range(len(restored))]
    assert results == [population[idx] for idx in restored.indices]

    tampered = restored.to_dict()
    tampered["fingerprint"] = "deadbeef"
    tampered_plan = RandomSamplerPlan.from_dict(tampered)

    with pytest.raises(ValueError):
        clone.apply_plan(tampered_plan, strict=True)

    clone.apply_plan(tampered_plan, strict=False)


def test_random_sampler_plan_thread_safety():
    population = list(range(20))
    sampler = RandomSampler(population, replace=False, sample_limit=10, seed=3)
    plan = sampler.create_plan()
    sampler.apply_plan(plan)

    collected: list[int] = []
    lock = threading.Lock()

    def worker() -> None:
        while True:
            value = sampler.sample()
            if value is None:
                return
            with lock:
                collected.append(value)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    expected = [population[idx] for idx in plan.indices]
    assert len(collected) == len(expected)
    assert sorted(collected) == sorted(expected)
    assert sampler.exhausted


@pytest.mark.asyncio
async def test_random_sampler_plan_async_to_thread():
    population = list(range(5))
    sampler = RandomSampler(population, replace=False, sample_limit=5, seed=0)
    plan = sampler.create_plan()
    sampler.apply_plan(plan)

    results: list[int] = []
    collect_lock = threading.Lock()

    async def runner() -> None:
        while True:
            # Yield between draws so two async consumers interleave predictably
            # without depending on interpreter-specific threadpool teardown.
            await asyncio.sleep(0)
            value = sampler.sample()
            if value is None:
                return
            with collect_lock:
                results.append(value)

    await asyncio.gather(runner(), runner())

    expected = [population[idx] for idx in plan.indices]
    assert sorted(results) == sorted(expected)


def test_random_sampler_plan_shared_across_instances():
    population = list(range(12))
    seed_sampler = RandomSampler(population, replace=False, seed=21)

    plan = seed_sampler.create_plan(draws=6)

    sampler_a = RandomSampler(population, replace=False, plan=plan)
    sampler_b = RandomSampler(population, replace=False, plan=plan)

    draws_a = [sampler_a.sample() for _ in range(6)]
    draws_b = [sampler_b.sample() for _ in range(6)]

    assert draws_a == draws_b == [population[idx] for idx in plan.indices]

    # Both samplers should be exhausted after consuming the plan (when no resume_random_after_plan)
    assert sampler_a.sample() is None
    assert sampler_a.exhausted
    assert sampler_b.sample() is None
    assert sampler_b.exhausted
