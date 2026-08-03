"""Regression coverage for the removed named strategy-preset selection API.

The epsilon / quality-floor / Pareto-frontier advisory *selection* rules
(``max_accuracy_then_cheapest_within_epsilon``, ``quality_floor_min_cost``,
``pareto_frontier``) were IP pulled back out of the SDK: that selection logic
is the technique and belongs in the backend, not in a package users can read.
``traigent.api.strategy_presets`` is deleted; nothing in the public surface may
still construct or expose a named preset.

Note: ``traigent.api.types.PresetSelection`` (the generic advisory-selection
*data container*, and ``OptimizationResult.preset_selection``) is deliberately
NOT covered here — it is retained so that results persisted by an older SDK
build can still be read back and displayed. Nothing in this build can ever
populate it with a live named-preset selection any more (see
``tests/unit/cli/test_results_commands.py::test_results_rerank_preset_option_is_gone``
for the CLI-side counterpart).
"""

from __future__ import annotations

import importlib.util

import pytest

import traigent
import traigent.api

REMOVED_MODULES = ("traigent.api.strategy_presets",)

REMOVED_ROOT_AND_API_EXPORTS = (
    "ADVISORY_SELECTION_NOTICE",
    "NormalizedStrategyPreset",
    "StrategyPresetError",
    "StrategyPresetValidationError",
    "UnknownStrategyPresetError",
    "VALID_PRESET_NAMES",
    "normalize_strategy_preset",
    "select_strategy_preset",
)


def test_strategy_presets_module_is_not_importable() -> None:
    for module_name in REMOVED_MODULES:
        assert importlib.util.find_spec(module_name) is None, (
            f"{module_name} is still importable"
        )

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("traigent.api.strategy_presets")


def test_no_preset_symbol_is_exported_from_traigent_or_traigent_api() -> None:
    for export_name in REMOVED_ROOT_AND_API_EXPORTS:
        assert export_name not in traigent.__all__, (
            f"{export_name} is still in traigent.__all__"
        )
        assert export_name not in traigent.api.__all__, (
            f"{export_name} is still in traigent.api.__all__"
        )
        assert not hasattr(traigent, export_name), (
            f"traigent still has attribute {export_name}"
        )
        assert not hasattr(traigent.api, export_name), (
            f"traigent.api still has attribute {export_name}"
        )
