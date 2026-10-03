"""Rejection tests for the removed inert ``mock`` / ``mock_mode_config`` params.

Traigent#1370 Item 6: both parameters were inert and are removed. Supplying
either must fail loudly through every public entry point instead of being
silently ignored.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from traigent.api.decorators import LegacyOptimizeArgs, optimize
from traigent.api.parameter_validator import validate_optimize_parameters
from traigent.cli.main import cli
from traigent.core.optimized_function import OptimizedFunction
from traigent.utils.exceptions import ValidationError

REMOVED = ["mock_mode_config", "mock"]
SPACE = {"x": [1, 2]}


def _fn(x: int = 1) -> int:
    return x


@pytest.mark.parametrize("name", REMOVED)
@pytest.mark.parametrize("value", [{"enabled": True}, None])
def test_decorator_rejects_removed_parameter(name: str, value: Any) -> None:
    """Rejected even when the value is None (recorder shortcut)."""
    with pytest.raises(TypeError, match=name):
        optimize(configuration_space=SPACE, **{name: value})(_fn)


@pytest.mark.parametrize("name", REMOVED)
def test_decorator_rejects_removed_parameter_when_traigent_disabled(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TRAIGENT_DISABLED", "1")
    with pytest.raises(TypeError, match=name):
        optimize(configuration_space=SPACE, **{name: {"enabled": True}})(_fn)


@pytest.mark.parametrize("name", REMOVED)
def test_decorator_rejects_removed_parameter_via_legacy(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        optimize(configuration_space=SPACE, legacy={name: {"enabled": True}})(_fn)
    with pytest.raises(TypeError, match=name):
        optimize(
            configuration_space=SPACE,
            legacy=LegacyOptimizeArgs(extra={name: {"enabled": True}}),
        )(_fn)


def test_mock_mode_options_is_gone() -> None:
    import traigent.api.decorators as decorators

    assert not hasattr(decorators, "MockModeOptions")
    assert "MockModeOptions" not in decorators.__all__


@pytest.mark.asyncio
@pytest.mark.parametrize("name", REMOVED)
async def test_runtime_optimize_rejects_removed_parameter(name: str) -> None:
    decorated = optimize(configuration_space=SPACE, objectives=["accuracy"])(_fn)
    with pytest.raises(TypeError, match=name):
        await decorated.optimize(max_trials=1, **{name: {"enabled": True}})


@pytest.mark.parametrize("name", REMOVED)
def test_direct_optimized_function_rejects_removed_parameter(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        OptimizedFunction(
            func=_fn, configuration_space=SPACE, **{name: {"enabled": True}}
        )


@pytest.mark.parametrize("name", REMOVED)
def test_validate_optimize_parameters_rejects_removed_parameter(name: str) -> None:
    with pytest.raises(ValidationError, match=name):
        validate_optimize_parameters(configuration_space=SPACE, **{name: {}})


@pytest.mark.parametrize("name", REMOVED)
def test_validate_config_cli_reports_removed_parameter(
    name: str, tmp_path: Any
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "configuration_space": {"x": [1, 2]},
                "objectives": ["accuracy"],
                name: {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(cli, ["validate-config", str(config)])
    assert "validation failed" in result.output
    assert name in result.output
    assert "passed" not in result.output
