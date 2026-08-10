"""``traigent validate-config`` must not pass a config that cannot run.

The command reads ``"algorithm"`` out of the JSON config and hands it to
``OptimizationValidator.validate_optimization_config`` as its ``strategy``
argument, where it was accepted and discarded. A config naming one of the three
retired strategy presets therefore printed "Configuration validation passed" —
the one route whose entire purpose is to say in advance whether a config is
runnable, answering yes for a config that raises the moment it is used.

Scope of the fix these pin: the three retired names only. ``validate-config``
still does not validate algorithm names in general, so an ordinary typo
(``"algorithm": "gird"``) still passes — asserted below so the gap is recorded
rather than mistaken for coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from traigent.cli.main import validate_config
from traigent.utils.validation import OptimizationValidator

REMOVED_PRESET_NAMES = (
    "max_accuracy_then_cheapest_within_epsilon",
    "quality_floor_min_cost",
    "pareto_frontier",
)


def _config_file(tmp_path: Path, algorithm: str) -> Path:
    """A minimal config whose only questionable field is ``algorithm``."""
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "configuration_space": {"model": ["gpt-4o", "gpt-4o-mini"]},
                "objectives": ["accuracy"],
                "algorithm": algorithm,
            }
        )
    )
    return path


@pytest.mark.parametrize("preset_name", REMOVED_PRESET_NAMES)
def test_validate_config_fails_a_retired_preset_name(
    tmp_path: Path, preset_name: str
) -> None:
    """The command reports failure and names the removal, not "passed"."""
    result = CliRunner().invoke(
        validate_config, [str(_config_file(tmp_path, preset_name))]
    )

    output = result.output
    assert "Configuration validation passed" not in output
    assert "Configuration validation failed" in output
    assert preset_name in output
    lowered = output.lower()
    assert "named strategy preset" in lowered
    assert "removed" in lowered
    assert "algorithm=" in lowered
    assert "objectives=" in lowered


def test_validate_config_still_passes_a_legal_algorithm(tmp_path: Path) -> None:
    """A config the refusal must not touch."""
    result = CliRunner().invoke(validate_config, [str(_config_file(tmp_path, "grid"))])

    assert "Configuration validation passed" in result.output


def test_validate_config_does_not_check_algorithm_names_in_general(
    tmp_path: Path,
) -> None:
    """Recorded gap, not coverage: only the retired names are refused.

    No general algorithm-name validation exists on this route and none was
    built here, so a plain typo still validates clean. This test exists so that
    the limit is written down where the next reader looks.
    """
    result = CliRunner().invoke(validate_config, [str(_config_file(tmp_path, "gird"))])

    assert "Configuration validation passed" in result.output


@pytest.mark.parametrize("preset_name", REMOVED_PRESET_NAMES)
def test_validator_reports_the_retired_name_as_an_error(preset_name: str) -> None:
    """The validator itself, below the CLI: an error, so ``is_valid`` is False."""
    result = OptimizationValidator.validate_optimization_config(
        {"model": ["gpt-4o"]}, ["accuracy"], None, preset_name
    )

    assert not result.is_valid
    assert any(preset_name in error.message for error in result.errors)


def test_validator_leaves_a_legal_algorithm_valid() -> None:
    result = OptimizationValidator.validate_optimization_config(
        {"model": ["gpt-4o"]}, ["accuracy"], None, "grid"
    )

    assert result.is_valid
