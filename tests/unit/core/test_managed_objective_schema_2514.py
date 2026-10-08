"""#2514: the managed (cloud) path must forward the declared ObjectiveSchema.

``InteractiveOptimizer`` was built from objective names only, so a custom
objective whose orientation was declared on the decorator failed with "has no
declared orientation" on the connected run, while the offline run (which keeps
the schema) passed.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

import traigent
from traigent.core.objectives import create_default_objectives
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.optimizers.interactive_optimizer import InteractiveOptimizer


class _AuthenticatedTestManager:
    """Synthetic authenticated session; no credential is stored or transmitted."""

    def has_api_key(self) -> bool:
        return True


class _StopAfterConstruction(Exception):
    """Ends the run once the managed optimizer has been built."""


def _fake_backend_client() -> Any:
    from traigent.core.session_types import SessionCreationResult

    client = SimpleNamespace(
        create_session=Mock(return_value=SessionCreationResult.connected("s-2514")),
        get_session_mapping=Mock(
            return_value=SimpleNamespace(
                experiment_id="exp-2514", experiment_run_id="run-2514"
            )
        ),
        upload_example_features=Mock(return_value=True),
        submit_result=Mock(),
        request_trial_slot=AsyncMock(return_value="slot-unused"),
        _submit_trial_result_via_session=AsyncMock(return_value=True),
        update_trial_weighted_scores=AsyncMock(return_value=True),
        finalize_session_sync=Mock(return_value={"status": "completed"}),
        close=AsyncMock(),
    )
    client.auth_manager = _AuthenticatedTestManager()
    client.auth = client.auth_manager
    return client


@pytest.mark.asyncio
async def test_managed_path_builds_optimizer_with_declared_custom_orientation(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
    monkeypatch.setenv("TRAIGENT_API_KEY", "tg_test_key")

    schema = create_default_objectives(
        ["graded_accuracy", "cost"],
        orientations={"graded_accuracy": "maximize", "cost": "minimize"},
    )

    @traigent.optimize(
        algorithm="bayesian",
        offline=False,
        eval_dataset=Dataset(
            [EvaluationExample({"text": "case-0"}, "ok")], name="managed_2514"
        ),
        objectives=schema,
        configuration_space={"temperature": [0.0, 1.0]},
        injection_mode="parameter",
    )
    def answer(text: str, config) -> str:
        return "ok"

    built: list[Any] = []

    def _build_real(config_space, objectives, **kwargs):
        # Build the REAL optimizer with exactly what the managed path passes;
        # pre-fix this raised "Objective 'graded_accuracy' has no declared
        # orientation".
        try:
            built.append(InteractiveOptimizer(config_space, objectives, **kwargs))
        except Exception as exc:  # recorded, asserted below
            built.append(exc)
        raise _StopAfterConstruction

    with (
        patch(
            "traigent.optimizers.interactive_optimizer.InteractiveOptimizer",
            side_effect=_build_real,
        ) as managed,
        patch("traigent.cloud.client.TraigentCloudClient"),
        patch(
            "traigent.core.backend_session_manager.BackendSessionManager"
            ".create_backend_client",
            return_value=_fake_backend_client(),
        ),
        contextlib.suppress(Exception),
    ):
        await answer.optimize()

    managed.assert_called_once()
    assert len(built) == 1
    optimizer = built[0]
    assert isinstance(optimizer, InteractiveOptimizer), optimizer
    assert optimizer.objective_orientations == {
        "graded_accuracy": "maximize",
        "cost": "minimize",
    }
