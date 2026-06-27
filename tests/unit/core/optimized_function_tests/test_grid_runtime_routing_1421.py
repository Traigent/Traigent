"""Regression tests for #1421: runtime ``optimize(algorithm="grid")`` must stay
local + exhaustive even when a portal API key is present.

Root cause (pre-fix): ``OptimizedFunction`` resolved and stored its execution
policy at construction WITHOUT the runtime algorithm. A decorator with the
default ``algorithm="auto"`` therefore stored a cloud-capable ``CLOUD_BRAIN``
policy. A runtime override such as ``optimize(algorithm="grid")`` reused that
stale policy, so with a portal key the SDK opened the backend-guided/typed cloud
session (Optuna/TPE sampling -> duplicate configs + skipped grid cells) instead
of running the exhaustive local ``GridSearchOptimizer``.

The fix recomputes the policy from the resolved runtime algorithm in
``_execute_optimization`` (see ``_policy_for_runtime_algorithm``), so grid/random
resolve to ``LOCAL_ONLY`` and ``_try_cloud_execution`` is skipped.
"""

from __future__ import annotations

import itertools
from unittest.mock import patch

import pytest

import traigent
from traigent.api.types import (
    OptimizationResult,
    OptimizationStatus,
)
from traigent.config.types import (
    ExecutionIntent,
    ExecutionMode,
    ResolvedExecutionPolicy,
    resolve_execution_policy,
)
from traigent.core.execution_policy_runtime import SOURCE_CLOUD_BRAIN
from traigent.core.optimization_pipeline import create_traigent_config
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample

# A small fully-enumerable categorical space: 3 * 2 = 6 unique configs.
_GRID_SPACE = {
    "temperature": [0.0, 0.5, 1.0],
    "model": ["a", "b"],
}
_EXPECTED_CONFIGS = {
    (("model", m), ("temperature", t))
    for t, m in itertools.product(_GRID_SPACE["temperature"], _GRID_SPACE["model"])
}


def _dataset() -> Dataset:
    return Dataset(
        [
            EvaluationExample({"text": "case-0"}, "ok"),
            EvaluationExample({"text": "case-1"}, "ok"),
        ],
        name="grid_runtime_routing_1421",
    )


def _config_key(config: dict) -> tuple:
    # Only the grid dimensions matter for exhaustiveness comparison.
    return tuple(sorted((k, config[k]) for k in _GRID_SPACE if k in config))


def _auto_optimized_function() -> OptimizedFunction:
    """A wrapper whose STORED policy is the cloud-capable ``auto`` default
    (NOT a decorator-level ``grid``) — the construction state that triggered the
    bug on the runtime override path.

    Built the way ``@traigent.optimize()`` builds it for the default
    ``algorithm="auto"``: an explicit cloud-brain ``execution_policy`` plus the
    matching legacy ``HYBRID`` execution mode (see decorators._runtime_execution
    _mode_for_policy). The direct ``OptimizedFunction`` constructor otherwise
    defaults to local ``edge_analytics``, which would not reproduce the bug.
    """

    auto_policy = resolve_execution_policy(algorithm="auto", source_hint="optimize")
    opt_func = OptimizedFunction(
        func=lambda text, **cfg: "ok",
        configuration_space=_GRID_SPACE,
        objectives=["accuracy"],
        eval_dataset=_dataset(),
        algorithm="auto",
        execution_mode=ExecutionMode.HYBRID,
        execution_policy=auto_policy,
    )
    # Sanity: the stored policy is the cloud-brain auto policy, exactly the
    # stale-policy precondition the bug depends on.
    assert isinstance(opt_func.execution_policy, ResolvedExecutionPolicy)
    assert opt_func.execution_policy.algorithm == "auto"
    assert opt_func.execution_policy.intent is ExecutionIntent.CLOUD_BRAIN
    return opt_func


class TestRuntimeGridPolicyRecompute:
    """Unit-level coverage of the policy recompute helper (the primary fix)."""

    def test_runtime_grid_recomputes_to_local_only(self) -> None:
        opt_func = _auto_optimized_function()
        stored = opt_func.execution_policy

        recomputed = opt_func._policy_for_runtime_algorithm(stored, "grid")

        assert recomputed is not None
        assert recomputed.algorithm == "grid"
        assert recomputed.intent is ExecutionIntent.LOCAL_ONLY

    def test_runtime_random_recomputes_to_local_only(self) -> None:
        opt_func = _auto_optimized_function()
        recomputed = opt_func._policy_for_runtime_algorithm(
            opt_func.execution_policy, "random"
        )
        assert recomputed is not None
        assert recomputed.intent is ExecutionIntent.LOCAL_ONLY

    def test_runtime_auto_keeps_cloud_brain(self) -> None:
        """No-op when the runtime algorithm matches the stored auto policy:
        genuinely cloud algorithms are NOT regressed to local."""
        opt_func = _auto_optimized_function()
        stored = opt_func.execution_policy
        recomputed = opt_func._policy_for_runtime_algorithm(stored, "auto")
        assert recomputed is stored
        assert recomputed.intent is ExecutionIntent.CLOUD_BRAIN

    def test_runtime_smart_override_keeps_cloud(self) -> None:
        """A smart runtime override on an auto wrapper is NOT regressed to local.

        The recompute is deliberately narrow (only ``grid``/``random`` flip to
        local); a smart override keeps the stored cloud-capable policy and is
        routed/validated downstream as before — never local.
        """
        opt_func = _auto_optimized_function()
        stored = opt_func.execution_policy
        recomputed = opt_func._policy_for_runtime_algorithm(stored, "optuna")
        assert recomputed is stored
        assert recomputed.intent is not ExecutionIntent.LOCAL_ONLY

    def test_runtime_unknown_algorithm_keeps_stored_policy(self) -> None:
        """An unknown runtime algorithm is NOT validated/rejected by the
        recompute; it keeps the stored policy so the existing optimizer-lookup
        error path (get_optimizer -> OptimizationError) is preserved."""
        opt_func = _auto_optimized_function()
        stored = opt_func.execution_policy
        recomputed = opt_func._policy_for_runtime_algorithm(stored, "unknown_algorithm")
        assert recomputed is stored

    def test_offline_preserved_on_recompute(self) -> None:
        """An offline (LOCAL_ONLY) stored policy stays offline after recompute."""
        opt_func = _auto_optimized_function()
        offline_policy = resolve_execution_policy(algorithm="auto", offline=True)
        assert offline_policy.offline is True
        recomputed = opt_func._policy_for_runtime_algorithm(offline_policy, "grid")
        assert recomputed is not None
        assert recomputed.offline is True
        assert recomputed.intent is ExecutionIntent.LOCAL_ONLY


class TestRuntimeGridSkipsCloud:
    """``_try_cloud_execution`` must NOT open a cloud session for runtime grid."""

    @pytest.mark.asyncio
    async def test_try_cloud_returns_none_for_recomputed_grid_policy(
        self, monkeypatch
    ) -> None:
        # Portal key present AND online (the autouse conftest fixture forces
        # TRAIGENT_OFFLINE_MODE=true for non-cloud tests, which would short
        # circuit cloud routing regardless of policy; clear it so the cloud-vs-
        # local decision is driven purely by the resolved policy intent — the
        # exact production condition #1421 reported).
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        monkeypatch.setenv("TRAIGENT_API_KEY", "tg-fake-portal-key-1421")
        opt_func = _auto_optimized_function()

        # The recomputed (correct) policy for runtime grid.
        grid_policy = opt_func._policy_for_runtime_algorithm(
            opt_func.execution_policy, "grid"
        )
        config = create_traigent_config(
            execution_mode=opt_func.execution_mode,
            local_storage_path=None,
            minimal_logging=False,
            privacy_enabled=False,
            execution_policy=grid_policy,
            result_source="explicit_local",
        )

        with (
            patch(
                "traigent.optimizers.interactive_optimizer.InteractiveOptimizer"
            ) as MockInteractive,
            patch("traigent.cloud.client.TraigentCloudClient") as MockClient,
        ):
            result = await opt_func._try_cloud_execution(
                _dataset(),
                None,
                None,
                _GRID_SPACE,
                {},
                config,
                {},  # artifact_fingerprint_payload — routing test, fingerprints not exercised
                False,
                None,
                True,
                None,
                None,
            )

        assert result is None
        MockInteractive.assert_not_called()
        MockClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_negative_control_stale_auto_policy_routes_to_cloud(
        self, monkeypatch
    ) -> None:
        """NEGATIVE CONTROL: with the STALE auto policy (the pre-fix condition),
        the very same runtime-grid call DOES open the cloud session — proving
        the 'no cloud call' assertion fails against the bug.
        """
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        monkeypatch.setenv("TRAIGENT_API_KEY", "tg-fake-portal-key-1421")
        opt_func = _auto_optimized_function()

        # The STALE policy: the construction-time auto/cloud-brain policy that the
        # buggy code reused for a runtime grid override.
        stale_policy = opt_func.execution_policy
        assert stale_policy.intent is ExecutionIntent.CLOUD_BRAIN
        config = create_traigent_config(
            execution_mode=opt_func.execution_mode,
            local_storage_path=None,
            minimal_logging=False,
            privacy_enabled=False,
            execution_policy=stale_policy,
            result_source=SOURCE_CLOUD_BRAIN,
        )

        # Make InteractiveOptimizer construction raise a sentinel so we can prove
        # the cloud path was entered (it would otherwise need a live backend).
        class _CloudEntered(RuntimeError):
            pass

        def _boom(*_a, **_k):
            raise _CloudEntered("cloud session-create entered")

        with (
            patch(
                "traigent.optimizers.interactive_optimizer.InteractiveOptimizer",
                side_effect=_boom,
            ) as MockInteractive,
            patch("traigent.cloud.client.TraigentCloudClient"),
        ):
            with pytest.raises(_CloudEntered):
                await opt_func._try_cloud_execution(
                    _dataset(),
                    None,
                    None,
                    _GRID_SPACE,
                    {},
                    config,
                    {},  # artifact_fingerprint_payload — routing test, fingerprints not exercised
                    False,
                    None,
                    True,
                    None,
                    None,
                )

        # The cloud optimizer WAS constructed under the stale policy: this is the
        # exact duplicate-sampling code path #1421 reported. The recompute fix
        # prevents reaching it (see the test above).
        MockInteractive.assert_called_once()


class TestRuntimeGridExhaustiveEndToEnd:
    """Full local run: runtime grid evaluates the entire cartesian product once,
    with a portal key present and zero cloud session/next-trial calls."""

    @pytest.mark.asyncio
    async def test_runtime_grid_is_exhaustive_with_api_key(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.chdir(tmp_path)
        # Online + portal key: the production condition for #1421.
        monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "false")
        monkeypatch.setenv("TRAIGENT_API_KEY", "tg-fake-portal-key-1421")
        monkeypatch.setenv("TRAIGENT_COST_APPROVED", "true")
        # The fake key cannot mint a backend *tracking* session, so allow the
        # local-only grid to proceed (this gate is about run-tracking, not about
        # the optimization routing the fix changes). The optimization cloud
        # session path is asserted-never-called via the mocks below.
        monkeypatch.setenv("TRAIGENT_ALLOW_UNTRACKED", "1")

        seen: list[dict] = []

        @traigent.optimize(
            eval_dataset=_dataset(),
            objectives=["accuracy"],
            configuration_space=_GRID_SPACE,
            injection_mode="parameter",
        )
        def answer(text: str, config) -> str:
            seen.append({k: config.get(k) for k in _GRID_SPACE})
            return "ok"

        # Stored policy is the cloud-capable auto default (the bug precondition).
        assert answer.execution_policy.intent is ExecutionIntent.CLOUD_BRAIN

        with (
            patch(
                "traigent.optimizers.interactive_optimizer.InteractiveOptimizer"
            ) as MockInteractive,
            patch("traigent.cloud.client.TraigentCloudClient") as MockClient,
        ):
            result = await answer.optimize(algorithm="grid")

        # No cloud session-create / next-trial: the runtime grid stayed local.
        MockInteractive.assert_not_called()
        MockClient.assert_not_called()

        assert isinstance(result, OptimizationResult)
        assert result.status is OptimizationStatus.COMPLETED

        # Exhaustive: every cartesian-product config evaluated exactly once.
        recorded = [_config_key(t.config) for t in result.trials]
        assert len(recorded) == len(_EXPECTED_CONFIGS), (
            f"expected {len(_EXPECTED_CONFIGS)} grid trials, got {len(recorded)}: "
            f"{recorded}"
        )
        assert len(set(recorded)) == len(recorded), f"duplicate configs: {recorded}"
        assert set(recorded) == _EXPECTED_CONFIGS, (
            f"missing/extra grid cells. got={set(recorded)} "
            f"expected={_EXPECTED_CONFIGS}"
        )
