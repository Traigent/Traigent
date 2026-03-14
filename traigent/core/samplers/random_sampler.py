"""Random sampling utility built on :mod:`random`."""

# Traceability: CONC-Layer-Core CONC-Quality-Reliability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def _normalise_for_hash(item: Any) -> bytes:
    if isinstance(item, bytes):
        return item
    if isinstance(item, str):
        return item.encode("utf-8")
    try:
        return json.dumps(item, sort_keys=True, default=str).encode("utf-8")
    except TypeError:
        return repr(item).encode("utf-8")


def _compute_population_fingerprint(population: Sequence[Any]) -> str:
    hasher = hashlib.sha256()
    for element in population:
        hasher.update(_normalise_for_hash(element))
        hasher.update(b"\x00")
    return hasher.hexdigest()


@dataclass(frozen=True)
class RandomSamplerPlan:
    """Serializable plan describing an exact sampling order."""

    indices: tuple[int, ...]
    replace: bool
    population_size: int
    sample_limit: int | None
    fingerprint: str | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "indices": list(self.indices),
            "replace": self.replace,
            "population_size": self.population_size,
            "sample_limit": self.sample_limit,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RandomSamplerPlan:
        raw_indices = data.get("indices", [])
        if not isinstance(raw_indices, (list, tuple)):
            raise TypeError("plan 'indices' must be a list or tuple")
        indices = tuple(int(idx) for idx in raw_indices)
        sample_limit_raw = data.get("sample_limit")
        sample_limit = None if sample_limit_raw is None else int(sample_limit_raw)
        population_size_raw = data.get("population_size", len(indices))
        population_size = int(population_size_raw)
        fingerprint = data.get("fingerprint")
        if fingerprint is not None and not isinstance(fingerprint, str):
            raise TypeError("plan 'fingerprint' must be a string when provided")
        return cls(
            indices=indices,
            replace=bool(data.get("replace", False)),
            population_size=population_size,
            sample_limit=sample_limit,
            fingerprint=fingerprint,
        )


class RandomSampler(Generic[T]):
    """Sample elements uniformly at random until a limit is reached.

    Notes
    -----
    * Thread-safe: internal state mutations are guarded by the sampler's re-entrant lock.
      For parallel execution, prefer :meth:`clone` to obtain independent instances per worker.
    * With-replacement sampling runs in O(1). Without replacement maintains a shrinking pool.
    * When exhausted, :meth:`sample` returns ``None`` and :pyattr:`exhausted` becomes ``True``.
    * Deterministic plans can be exported/imported via :meth:`create_plan` / :meth:`apply_plan`.
    """

    def __init__(
        self,
        population: Sequence[T],
        *,
        sample_limit: int | None = None,
        replace: bool = False,
        seed: int | None = None,
        plan: RandomSamplerPlan | Mapping[str, Any] | None = None,
        resume_random_after_plan: bool = False,
    ) -> None:
        self._lock = RLock()
        self._exhausted: bool = False
        if sample_limit is not None and sample_limit <= 0:
            raise ValueError("sample_limit must be a positive integer or None")
        if not population:
            raise ValueError("population must contain at least one element")

        self._population: Sequence[T] = population
        self._population_size = len(population)
        self._replace = bool(replace)
        self._sample_limit = sample_limit
        self._rng = random.Random(seed)
        self._seed = seed
        self._samples_drawn = 0

        self._index_pool: list[int] | None = None
        if not self._replace:
            self._index_pool = list(range(self._population_size))
        self._initial_index_pool: tuple[int, ...] | None = (
            tuple(self._index_pool) if self._index_pool is not None else None
        )
        self._initial_rng_state = self._rng.getstate()

        self._fixed_plan: tuple[int, ...] | None = None
        self._fixed_cursor = 0
        self._resume_random_after_plan = bool(resume_random_after_plan)
        self._fingerprint_cache: str | None = None

        if plan is not None:
            self.apply_plan(plan, strict=False, reset_state=False)

    @property
    def supports_plans(self) -> bool:
        return True

    @property
    def exhausted(self) -> bool:
        """Return ``True`` when the sampler cannot yield more samples."""
        with self._lock:
            return self._exhausted

    def reset(self) -> None:
        """Reset internal state so sampling can start over."""
        with self._lock:
            self._exhausted = False
            self._samples_drawn = 0
            if not self._replace:
                if self._initial_index_pool is None:
                    self._initial_index_pool = tuple(range(self._population_size))
                self._index_pool = list(self._initial_index_pool)
            else:
                self._index_pool = None
            self._fixed_cursor = 0

    def _mark_exhausted(self) -> None:
        """Mark the sampler as exhausted."""
        with self._lock:
            self._exhausted = True

    def _ensure_fingerprint(self) -> str:
        if self._fingerprint_cache is None:
            self._fingerprint_cache = _compute_population_fingerprint(self._population)
        return self._fingerprint_cache

    def _remaining_capacity(
        self,
        pool: list[int] | None,
        samples_drawn: int,
    ) -> int | None:
        remaining_limit: int | None = None
        if self._sample_limit is not None:
            remaining_limit = max(self._sample_limit - samples_drawn, 0)

        if self._replace:
            return remaining_limit

        pool_remaining = len(pool) if pool is not None else 0
        if remaining_limit is None:
            return pool_remaining
        return min(remaining_limit, pool_remaining)

    def _resolve_plan_size(self, requested: int | None, remaining: int | None) -> int:
        if requested is not None and requested < 0:
            raise ValueError("draws must be a non-negative integer or None")

        if remaining is None:
            if requested is None:
                raise ValueError(
                    "draws must be provided when sampler has no natural limit"
                )
            return requested

        if requested is None:
            return remaining
        return min(requested, remaining)

    def _simulate_draw_sequence(
        self,
        pool: list[int] | None,
        rng_state: Any,
        plan_size: int,
    ) -> list[int]:
        if plan_size <= 0:
            return []

        rng_copy = random.Random()
        rng_copy.setstate(rng_state)

        sequence: list[int] = []
        if self._replace:
            for _ in range(plan_size):
                sequence.append(rng_copy.randrange(self._population_size))
            return sequence

        working_pool = (
            list(range(self._population_size)) if pool is None else list(pool)
        )
        for _ in range(plan_size):
            if not working_pool:
                break
            pos = rng_copy.randrange(len(working_pool))
            idx = working_pool.pop(pos)
            sequence.append(idx)
        return sequence

    def create_plan(
        self,
        *,
        draws: int | None = None,
        from_start: bool = True,
        include_fingerprint: bool = True,
        **kwargs: Any,
    ) -> RandomSamplerPlan:
        """Materialise a deterministic sampling plan.

        Args:
            draws: Optional cap on the number of draws in the plan.
            from_start: Generate the plan from the initial state rather than the current state.
            include_fingerprint: Attach a population fingerprint for validation.

        Returns:
            RandomSamplerPlan describing the sampling order.
        """

        with self._lock:
            if self._fixed_plan is not None:
                if from_start:
                    indices_source = self._fixed_plan
                else:
                    indices_source = self._fixed_plan[self._fixed_cursor :]

                planned = tuple(
                    indices_source[:draws] if draws is not None else indices_source
                )
                fingerprint = (
                    self._ensure_fingerprint() if include_fingerprint else None
                )
                return RandomSamplerPlan(
                    indices=planned,
                    replace=self._replace,
                    population_size=self._population_size,
                    sample_limit=self._sample_limit,
                    fingerprint=fingerprint,
                )

            pool = (
                list(self._initial_index_pool)
                if from_start and self._initial_index_pool is not None
                else (list(self._index_pool) if self._index_pool is not None else None)
            )
            samples_drawn = 0 if from_start else self._samples_drawn
            rng_state = self._initial_rng_state if from_start else self._rng.getstate()
            remaining = self._remaining_capacity(pool, samples_drawn)
            plan_size = self._resolve_plan_size(draws, remaining)
            sequence = self._simulate_draw_sequence(pool, rng_state, plan_size)
            fingerprint = self._ensure_fingerprint() if include_fingerprint else None

            return RandomSamplerPlan(
                indices=tuple(sequence),
                replace=self._replace,
                population_size=self._population_size,
                sample_limit=self._sample_limit,
                fingerprint=fingerprint,
            )

    def apply_plan(
        self,
        plan: Any,
        *,
        strict: bool = True,
        resume_random_after_plan: bool | None = None,
        reset_state: bool = True,
        **kwargs: Any,
    ) -> None:
        """Configure the sampler to follow a pre-computed plan.

        Args:
            plan: Plan object or its mapping representation.
            strict: Validate compatibility between the plan and sampler configuration.
            resume_random_after_plan: Continue random sampling once the plan is exhausted.
            reset_state: Reset sampler counters before applying the plan.
            **kwargs: Additional arguments for compatibility.
        """

        resolved_plan = (
            plan
            if isinstance(plan, RandomSamplerPlan)
            else RandomSamplerPlan.from_dict(plan)
        )

        with self._lock:
            self._validate_plan_compatibility(resolved_plan, strict)

            for idx in resolved_plan.indices:
                if not 0 <= idx < self._population_size:
                    raise ValueError(
                        f"Plan index {idx} out of range for population size {self._population_size}"
                    )

            if reset_state:
                self.reset()
            else:
                self._exhausted = False

            self._fixed_plan = tuple(resolved_plan.indices)
            self._fixed_cursor = 0
            if resume_random_after_plan is not None:
                self._resume_random_after_plan = bool(resume_random_after_plan)

    def _post_sample_bookkeeping(self) -> None:
        if self._sample_limit is not None and self._samples_drawn >= self._sample_limit:
            self._mark_exhausted()
            return
        if not self._replace and self._index_pool is not None and not self._index_pool:
            self._mark_exhausted()

    def _consume_plan_entry(self) -> T | None:
        if self._fixed_plan is None:
            return None

        if self._fixed_cursor >= len(self._fixed_plan):
            if self._resume_random_after_plan:
                self._fixed_plan = None
                self._fixed_cursor = 0
            else:
                self._mark_exhausted()
            return None

        index = self._fixed_plan[self._fixed_cursor]
        self._fixed_cursor += 1
        if not self._replace and self._index_pool is not None:
            try:
                self._index_pool.remove(index)
            except ValueError:
                # Already removed via previous plan application; ignore.
                pass

        choice = self._population[index]
        self._samples_drawn += 1
        self._post_sample_bookkeeping()
        return choice

    def sample(self, **kwargs: Any) -> T | None:
        with self._lock:
            if self._exhausted:
                return None

            if (
                self._sample_limit is not None
                and self._samples_drawn >= self._sample_limit
            ):
                self._mark_exhausted()
                return None

            planned_choice = self._consume_plan_entry()
            if planned_choice is not None:
                return planned_choice
            if self._exhausted:
                return None

            if self._replace:
                index = self._rng.randrange(self._population_size)
                choice = self._population[index]
            else:
                if self._index_pool is None:
                    raise RuntimeError("Index pool not initialized for sampling")
                if not self._index_pool:
                    self._mark_exhausted()
                    return None
                pos = self._rng.randrange(len(self._index_pool))
                index = self._index_pool.pop(pos)
                choice = self._population[index]

            self._samples_drawn += 1
            self._post_sample_bookkeeping()
            return choice

    def clone(self) -> RandomSampler[T]:
        """Create an independent sampler with identical configuration.

        Returns:
            New RandomSampler with same population/limits but fresh RNG state.
            Note: Plans are NOT copied - the clone starts without any fixed plan.
        """
        return RandomSampler(
            population=self._population,
            sample_limit=self._sample_limit,
            replace=self._replace,
            seed=None,
        )

    def _validate_plan_compatibility(
        self, plan: RandomSamplerPlan, strict: bool
    ) -> None:
        if plan.replace != self._replace:
            raise ValueError("Plan replace flag does not match sampler configuration.")

        if self._sample_limit is not None and len(plan.indices) > self._sample_limit:
            raise ValueError("Plan length exceeds sampler sample_limit configuration.")

        if not self._replace:
            indices_set = set(plan.indices)
            if len(indices_set) != len(plan.indices):
                raise ValueError("Plan contains duplicate indices while replace=False.")

        if strict:
            if plan.population_size != self._population_size:
                raise ValueError(
                    "Plan population size does not match sampler population."
                )
            if (
                plan.sample_limit is not None
                and self._sample_limit is not None
                and plan.sample_limit > self._sample_limit
            ):
                raise ValueError(
                    "Plan sample_limit exceeds sampler sample_limit configuration."
                )
            if plan.fingerprint:
                fingerprint = self._ensure_fingerprint()
                if fingerprint != plan.fingerprint:
                    raise ValueError(
                        "Plan fingerprint does not match sampler population."
                    )

        for idx in plan.indices:
            if not 0 <= idx < self._population_size:
                raise ValueError(
                    f"Plan index {idx} out of range for population size {self._population_size}"
                )
