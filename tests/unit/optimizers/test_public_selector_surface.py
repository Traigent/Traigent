"""Public smoke test for the optimizer selector surface.

Guards against "complete but undiscoverable" drift between the feature
matrix at ``docs/feature_matrices/optimizers.yml`` and the public API
surface exposed by ``traigent.optimizers``:

- Every implementation marked ``status: complete`` must be importable
  from ``traigent.optimizers`` (export check).
- Every implementation that declares a ``selectable_by`` algorithm name
  must be registered in ``list_optimizers()`` and constructible through
  ``get_optimizer()`` (selector check).

Tracks issue #919.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

import traigent.optimizers as opt_pkg
from traigent.optimizers import get_optimizer, list_optimizers
from traigent.optimizers.base import BaseOptimizer

REPO_ROOT = Path(__file__).resolve().parents[3]
OPTIMIZER_MATRIX_PATH = REPO_ROOT / "docs/feature_matrices/optimizers.yml"


def _load_matrix() -> dict[str, Any]:
    with OPTIMIZER_MATRIX_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _iter_implementations() -> list[dict[str, Any]]:
    matrix = _load_matrix()
    implementations: list[dict[str, Any]] = []
    for capability in matrix.get("capabilities", []):
        cap_id = capability.get("id", "<unknown>")
        for impl in capability.get("implementations", []):
            entry = dict(impl)
            entry["_capability_id"] = cap_id
            implementations.append(entry)
    return implementations


_BATCH_OPTIMIZER_CLASS_NAMES = {
    "ParallelBatchOptimizer",
    "MultiObjectiveBatchOptimizer",
    "AdaptiveBatchOptimizer",
}


def _complete_batch_implementations() -> list[dict[str, Any]]:
    return [
        impl
        for impl in _iter_implementations()
        if impl.get("class_name") in _BATCH_OPTIMIZER_CLASS_NAMES
        and impl.get("status") == "complete"
    ]


def _selectable_implementations() -> list[dict[str, Any]]:
    return [impl for impl in _iter_implementations() if impl.get("selectable_by")]


class TestPublicOptimizerSelectorSurface:
    """Selector-surface contract tests for documented optimizers."""

    def test_batch_optimizers_are_exported_from_package(self) -> None:
        """Each documented complete batch optimizer is importable.

        Mirrors the user-facing pattern ``from traigent.optimizers import X``.
        """
        complete = _complete_batch_implementations()
        assert (
            complete
        ), "Expected at least one complete batch optimizer in feature matrix"
        missing = [
            impl["class_name"]
            for impl in complete
            if not hasattr(opt_pkg, impl["class_name"])
        ]
        assert not missing, (
            "Documented-complete batch optimizers missing from "
            f"traigent.optimizers public surface: {missing}"
        )

    def test_batch_optimizers_are_registry_selectable(self) -> None:
        """Each batch optimizer with ``selectable_by`` is in the registry.

        Future drift: if a new batch optimizer is marked complete but
        not registered, this assertion fails and prompts an update.
        """
        registered = set(list_optimizers())
        for impl in _complete_batch_implementations():
            name = impl.get("selectable_by")
            assert name, (
                f"Batch optimizer {impl['class_name']} is marked complete but "
                "has no `selectable_by` field. Either register it under a public "
                "algorithm name or document non-selectability in the feature matrix."
            )
            assert name in registered, (
                f"selectable_by='{name}' for {impl['class_name']} is not in the "
                f"public registry. Registered: {sorted(registered)}"
            )

    @pytest.mark.parametrize(
        "name,expected_class",
        [
            ("parallel_batch", "ParallelBatchOptimizer"),
            ("multi_objective_batch", "MultiObjectiveBatchOptimizer"),
            ("adaptive_batch", "AdaptiveBatchOptimizer"),
        ],
    )
    def test_batch_optimizers_instantiable_via_get_optimizer(
        self, name: str, expected_class: str
    ) -> None:
        """``get_optimizer`` returns the documented class for each batch name."""
        config_space = {"x": [1, 2, 3], "y": [0.1, 0.2]}
        objectives = ["accuracy"]

        instance = get_optimizer(name, config_space, objectives)

        assert isinstance(instance, BaseOptimizer)
        assert type(instance).__name__ == expected_class
        assert instance.config_space == config_space
        assert instance.objectives == objectives

    def test_selectable_by_entries_resolve_to_declared_class(self) -> None:
        """``selectable_by`` entries must resolve to their declared class.

        Prevents silent renames or accidental adapter substitution.
        """
        config_space = {"x": [1, 2]}
        objectives = ["accuracy"]

        for impl in _selectable_implementations():
            name = impl["selectable_by"]
            declared_class = impl["class_name"]
            instance = get_optimizer(name, config_space, objectives)
            assert type(instance).__name__ == declared_class, (
                f"Feature matrix declares selectable_by='{name}' resolves to "
                f"{declared_class}, but registry returned {type(instance).__name__}"
            )
