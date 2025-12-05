"""Sampling strategy interfaces used by Traigent orchestration."""

# Traceability: CONC-Layer-Core CONC-Quality-Maintainability FUNC-OPT-ALGORITHMS REQ-OPT-ALG-004 SYNC-OptimizationFlow

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock
from typing import Any


class BaseSampler(ABC):
    """Abstract base class for sampling strategies.

    Implementations should keep track of their own exhaustion state and return
    ``None`` when no additional samples are available. Consumers can also query
    :pyattr:`exhausted` to short-circuit polling when desired.
    """

    def __init__(self) -> None:
        self._exhausted: bool = False
        self._lock = RLock()

    @property
    def supports_plans(self) -> bool:
        """Return ``True`` if the sampler can export/apply deterministic plans."""

        return False

    @property
    def exhausted(self) -> bool:
        """Return ``True`` when the sampler cannot yield more samples."""

        with self._lock:
            return self._exhausted

    def reset(self) -> None:
        """Reset internal state so sampling can start over."""

        with self._lock:
            self._exhausted = False
            self._reset_impl()

    @abstractmethod
    def sample(self, **kwargs: Any) -> Any | None:
        """Return the next sample or ``None`` if the sampler is exhausted."""

    def create_plan(self, **kwargs: Any) -> Any:
        """Return a reusable sampling plan if supported."""

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support plan creation."
        )

    def apply_plan(self, plan: Any, **kwargs: Any) -> None:
        """Configure the sampler to follow a pre-computed plan if supported."""

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support applying plans."
        )

    def _mark_exhausted(self) -> None:
        """Mark the sampler as exhausted."""

        with self._lock:
            self._exhausted = True

    @abstractmethod
    def clone(self) -> BaseSampler:
        """Create an independent sampler with identical configuration."""

    def _reset_impl(self) -> None:
        """Hook for subclasses to reset internal state safely."""
        return
