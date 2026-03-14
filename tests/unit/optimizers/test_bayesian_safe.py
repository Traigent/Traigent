"""Dependency-light coverage tests for bayesian optimizer fixes."""

from __future__ import annotations

import numpy as np

import traigent.optimizers.bayesian as bayesian_module


class _FakeGP:
    def predict(self, _X, return_std=True):  # type: ignore[no-untyped-def]
        assert return_std is True
        return np.array([[1.0]]), np.array([1e-16])


class _FakeNorm:
    @staticmethod
    def cdf(values: np.ndarray) -> np.ndarray:
        return np.full_like(values, 0.5, dtype=float)

    @staticmethod
    def pdf(values: np.ndarray) -> np.ndarray:
        return np.zeros_like(values, dtype=float)


def test_expected_improvement_treats_near_zero_sigma_as_zero(monkeypatch) -> None:
    monkeypatch.setattr(bayesian_module, "norm", _FakeNorm(), raising=False)

    optimizer = object.__new__(bayesian_module.BayesianOptimizer)
    optimizer.gp = _FakeGP()
    optimizer.xi = 0.01

    ei = bayesian_module.BayesianOptimizer._expected_improvement(
        optimizer,
        np.array([[0.5]]),
        y_best=0.5,
    )

    assert ei.shape == (1,)
    assert ei[0] == 0.0
