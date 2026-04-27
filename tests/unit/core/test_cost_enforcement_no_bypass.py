"""S2-B Round 3 — pin that TRAIGENT_MOCK_LLM never bypasses cost enforcement.

Codex's re-review flagged five sites in ``traigent/core/cost_enforcement.py``
that read ``TRAIGENT_MOCK_LLM`` and behaviorally short-circuited cost
enforcement (approval, permit acquisition, accounting, async tracking,
sync tracking). If the env var ever leaked into production, customers'
cost limits and accounting would be silently bypassed — a security/billing
concern.

These tests pin the post-fix behavior:

* ``check_and_approve`` must NOT auto-return True when mock-mode is set.
* ``acquire_permit`` / ``acquire_permit_async`` must NOT return synthetic
  ``id=0`` permits and must respect real limits.
* ``track_cost`` / ``track_cost_async`` must update ``accumulated_cost``.
* ``release_permit`` / ``release_permit_async`` must go through the real
  locked path.
* The bypass machinery (``_check_mock_mode`` / ``is_mock_mode`` /
  ``_mock_mode_cached``) is removed from the public+private API.
* ``CostEstimator.check_cost_approval`` no longer skips when mock-mode is set.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

from traigent.core.cost_enforcement import CostEnforcer, CostEnforcerConfig


@pytest.fixture
def mock_env() -> dict[str, str]:
    """Common env dict with TRAIGENT_MOCK_LLM=true plus auxiliary flags off."""
    return {
        "TRAIGENT_MOCK_LLM": "true",
        "TRAIGENT_COST_APPROVED": "false",
        "TRAIGENT_REQUIRE_COST_TRACKING": "false",
        "TRAIGENT_STRICT_COST_ACCOUNTING": "false",
    }


class TestCostEnforcerNoMockBypass:
    """CostEnforcer must not honor TRAIGENT_MOCK_LLM at any of the 5 flagged sites."""

    def test_bypass_attributes_are_removed(self, mock_env: dict[str, str]) -> None:
        """The mock-mode caching/property/method must be gone."""
        with patch.dict(os.environ, mock_env, clear=False):
            enforcer = CostEnforcer()
            assert not hasattr(enforcer, "_check_mock_mode")
            assert not hasattr(enforcer, "is_mock_mode")
            assert not hasattr(enforcer, "_mock_mode_cached")

    def test_check_and_approve_does_not_short_circuit_on_mock_env(
        self, mock_env: dict[str, str]
    ) -> None:
        """check_and_approve must NOT auto-return True on TRAIGENT_MOCK_LLM=true.

        Pre-fix: the mock_env branch returned True without consulting
        approval rules. Post-fix: approval must be requested (we patch
        ``_request_user_approval`` to deny so the test is deterministic).
        """
        with patch.dict(os.environ, mock_env, clear=False):
            enforcer = CostEnforcer(CostEnforcerConfig(limit=1.0, approved=False))
            with patch(
                "traigent.core.cost_enforcement.CostEnforcer._request_user_approval",
                return_value=False,
            ) as request_approval:
                # Estimated cost above limit, no auto-approve, no token.
                assert enforcer.check_and_approve(1000.0) is False
                request_approval.assert_called_once_with(1000.0)

    def test_acquire_permit_does_not_return_bypass_id_zero(
        self, mock_env: dict[str, str]
    ) -> None:
        """acquire_permit must allocate a real permit id (>=1) under mock env."""
        with patch.dict(os.environ, mock_env, clear=False):
            enforcer = CostEnforcer(
                CostEnforcerConfig(limit=10.0, estimated_cost_per_trial=0.01)
            )
            permit = enforcer.acquire_permit()
            assert permit.is_granted
            # Pre-fix mock-mode short-circuit returned id=0; real permits start at 1.
            assert permit.id != 0
            assert enforcer._in_flight_count == 1

    def test_acquire_permit_async_does_not_return_bypass_id_zero(
        self, mock_env: dict[str, str]
    ) -> None:
        """acquire_permit_async must allocate a real permit id (>=1) under mock env."""

        async def run() -> None:
            with patch.dict(os.environ, mock_env, clear=False):
                enforcer = CostEnforcer(
                    CostEnforcerConfig(limit=10.0, estimated_cost_per_trial=0.01)
                )
                permit = await enforcer.acquire_permit_async()
                assert permit.is_granted
                assert permit.id != 0
                assert enforcer._in_flight_count == 1

        asyncio.run(run())

    def test_track_cost_updates_accumulated_cost_under_mock_env(
        self, mock_env: dict[str, str]
    ) -> None:
        """track_cost must update accumulated_cost when TRAIGENT_MOCK_LLM=true."""
        with patch.dict(os.environ, mock_env, clear=False):
            enforcer = CostEnforcer(CostEnforcerConfig(limit=10.0))
            permit = enforcer.acquire_permit()
            enforcer.track_cost(0.42, permit=permit)
            assert enforcer.accumulated_cost == pytest.approx(0.42)
            # Real bookkeeping increments the trial count.
            assert enforcer._trial_count == 1

    def test_track_cost_async_updates_accumulated_cost_under_mock_env(
        self, mock_env: dict[str, str]
    ) -> None:
        """track_cost_async must update accumulated_cost under mock env."""

        async def run() -> None:
            with patch.dict(os.environ, mock_env, clear=False):
                enforcer = CostEnforcer(CostEnforcerConfig(limit=10.0))
                permit = await enforcer.acquire_permit_async()
                await enforcer.track_cost_async(0.18, permit=permit)
                assert enforcer.accumulated_cost == pytest.approx(0.18)
                assert enforcer._trial_count == 1

        asyncio.run(run())

    def test_acquire_permit_respects_real_limits_under_mock_env(
        self, mock_env: dict[str, str]
    ) -> None:
        """Tight limits must still deny permits under mock env (no auto-grant)."""
        with patch.dict(os.environ, mock_env, clear=False):
            # Limit equals one trial. After tracking one trial's actual cost,
            # the next permit reservation must be denied.
            enforcer = CostEnforcer(
                CostEnforcerConfig(limit=0.10, estimated_cost_per_trial=0.05)
            )
            first = enforcer.acquire_permit()
            assert first.is_granted, "First permit within budget should be granted"
            enforcer.track_cost(0.10, permit=first)  # Consume entire budget.

            # Now budget is exhausted; the bypass would have granted id=0,
            # but the real path must deny.
            second = enforcer.acquire_permit()
            assert not second.is_granted, (
                "Mock-mode must not override real cost limits after exhaustion"
            )
            assert second.id == -1, "Denied permit must use sentinel id=-1, not bypass id=0"

    def test_release_permit_uses_real_locked_path_under_mock_env(
        self, mock_env: dict[str, str]
    ) -> None:
        """release_permit must traverse the real locked release path under mock env."""
        with patch.dict(os.environ, mock_env, clear=False):
            enforcer = CostEnforcer(CostEnforcerConfig(limit=10.0))
            permit = enforcer.acquire_permit()
            assert permit.id in enforcer._active_permits
            assert enforcer._in_flight_count == 1

            released = enforcer.release_permit(permit)
            assert released is True
            # Real release-path bookkeeping clears the permit.
            assert permit.id not in enforcer._active_permits
            assert enforcer._in_flight_count == 0


class TestCostEstimatorNoMockBypass:
    """CostEstimator.check_cost_approval must not skip on TRAIGENT_MOCK_LLM=true."""

    def test_check_cost_approval_does_not_short_circuit(
        self, mock_env: dict[str, str]
    ) -> None:
        from traigent.core.cost_enforcement import OptimizationAborted
        from traigent.core.cost_estimator import CostEstimator

        with patch.dict(os.environ, mock_env, clear=False):
            enforcer = CostEnforcer(CostEnforcerConfig(limit=0.01, approved=False))

            estimator = CostEstimator.__new__(CostEstimator)
            estimator._cost_enforcer = enforcer
            # Force a high estimate via patching the estimation method
            with (
                patch.object(
                    CostEstimator,
                    "estimate_optimization_cost",
                    return_value=999.0,
                ),
                patch(
                    "traigent.core.cost_enforcement.CostEnforcer._request_user_approval",
                    return_value=False,
                ),
            ):
                with pytest.raises(OptimizationAborted):
                    estimator.check_cost_approval(dataset=None)
