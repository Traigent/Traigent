"""Regression tests for the #2352 review: the managed-run heartbeat gate must
read the per-call RESOLVED execution mode, not the construction-time
``self.execution_mode`` string.

Two concrete bugs, both from the same stale signal:

* FALSE POSITIVE -- a per-call ``algorithm`` override (e.g.
  ``optimize(algorithm="grid")`` on a decorator built with ``algorithm="auto"``)
  flips the *actual* routing to ``LOCAL_ONLY`` / no-egress (#1421), but
  ``self.execution_mode`` still reads ``"hybrid"`` from construction, so the
  heartbeat fires for a run that never leaves the machine.
* FALSE NEGATIVE -- an external-evaluator run resolves to
  ``ExecutionMode.HYBRID_API``, which the old gate (``== HYBRID.value``)
  excluded outright, so a managed run gets zero heartbeat -- the literal
  #1601 symptom the feature exists to fix.

``OptimizedFunction._resolve_runtime_execution_mode`` (added by this fix)
re-derives the effective mode for THIS call, mirroring
``decorators._runtime_execution_mode_for_policy`` and reusing the same
``_policy_for_runtime_algorithm`` recompute ``_execute_optimization`` already
performs (issue #1421).
"""

from __future__ import annotations

from traigent.config.types import (
    ExecutionIntent,
    ExecutionMode,
    ResolvedExecutionPolicy,
    resolve_execution_policy,
)
from traigent.core.optimized_function import OptimizedFunction, _resolve_callbacks
from traigent.evaluators.base import Dataset, EvaluationExample

_SPACE = {"temperature": [0.0, 0.5, 1.0], "model": ["a", "b"]}


def _dataset() -> Dataset:
    return Dataset(
        [EvaluationExample({"text": "case-0"}, "ok")],
        name="runtime_execution_mode_2352",
    )


def _auto_hybrid_optimized_function() -> OptimizedFunction:
    """Constructed the way ``@traigent.optimize()`` builds the ``algorithm="auto"``
    default: a stored cloud-capable ``auto``/``CLOUD_BRAIN`` policy and the
    matching legacy ``HYBRID`` execution mode -- the exact construction-time
    state that goes stale on a runtime override.
    """
    auto_policy = resolve_execution_policy(algorithm="auto", source_hint="optimize")
    opt_func = OptimizedFunction(
        func=lambda text, **cfg: "ok",
        configuration_space=_SPACE,
        objectives=["accuracy"],
        eval_dataset=_dataset(),
        algorithm="auto",
        execution_mode=ExecutionMode.HYBRID,
        execution_policy=auto_policy,
    )
    assert isinstance(opt_func.execution_policy, ResolvedExecutionPolicy)
    assert opt_func.execution_policy.intent is ExecutionIntent.CLOUD_BRAIN
    assert opt_func.execution_mode == ExecutionMode.HYBRID.value
    return opt_func


def _hybrid_api_optimized_function() -> OptimizedFunction:
    """An external-evaluator wrapper: construction-time mode is HYBRID_API."""
    policy = resolve_execution_policy(algorithm="auto", source_hint="optimize")
    opt_func = OptimizedFunction(
        func=lambda text, **cfg: "ok",
        configuration_space=_SPACE,
        objectives=["accuracy"],
        eval_dataset=_dataset(),
        algorithm="auto",
        execution_mode=ExecutionMode.HYBRID_API,
        execution_policy=policy,
    )
    assert opt_func.execution_mode == ExecutionMode.HYBRID_API.value
    return opt_func


class TestResolveRuntimeExecutionMode:
    """Direct coverage of the new per-call resolver."""

    def test_local_algorithm_override_resolves_to_local(self) -> None:
        """The false-positive case: a local runtime override on a HYBRID/auto
        wrapper must resolve to LOCAL, even though ``self.execution_mode`` is
        still the stale ``"hybrid"`` from construction."""
        opt_func = _auto_hybrid_optimized_function()

        resolved = opt_func._resolve_runtime_execution_mode("grid")

        assert resolved is ExecutionMode.LOCAL
        # The whole point: the per-call answer diverges from the stale attribute.
        assert resolved.value != opt_func.execution_mode

    def test_smart_override_stays_managed(self) -> None:
        """A smart override (#1681, CLOUD_REQUIRED) is still a managed run."""
        opt_func = _auto_hybrid_optimized_function()
        resolved = opt_func._resolve_runtime_execution_mode("bayesian")
        assert resolved is ExecutionMode.HYBRID

    def test_unchanged_auto_stays_managed(self) -> None:
        opt_func = _auto_hybrid_optimized_function()
        resolved = opt_func._resolve_runtime_execution_mode(None)
        assert resolved is ExecutionMode.HYBRID

    def test_external_evaluator_resolves_to_hybrid_api(self) -> None:
        """The false-negative case: an external-evaluator run must resolve to
        HYBRID_API regardless of the runtime algorithm."""
        opt_func = _hybrid_api_optimized_function()

        resolved = opt_func._resolve_runtime_execution_mode(None)

        assert resolved is ExecutionMode.HYBRID_API


class TestManagedHeartbeatGateCoversHybridApi:
    """``_resolve_callbacks`` itself: the widened gate (``!= LOCAL``, matching
    ``_log_execution_mode_warnings``' own definition of "managed") must inject
    the heartbeat for HYBRID_API, not only HYBRID -- the literal #1601 symptom
    for the external-evaluator sibling path."""

    def test_hybrid_api_gets_a_heartbeat(self, monkeypatch) -> None:
        import sys as _sys

        from traigent.utils.callbacks import ManagedProgressCallback

        monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)
        resolved = _resolve_callbacks(
            [], None, None, execution_mode=ExecutionMode.HYBRID_API.value
        )
        assert any(isinstance(c, ManagedProgressCallback) for c in resolved), (
            "HYBRID_API (external-evaluator) managed runs must get a heartbeat"
        )

    def test_local_gets_no_heartbeat(self, monkeypatch) -> None:
        import sys as _sys

        from traigent.utils.callbacks import ManagedProgressCallback

        monkeypatch.setattr(_sys.stdin, "isatty", lambda: False)
        resolved = _resolve_callbacks(
            [], None, None, execution_mode=ExecutionMode.LOCAL.value
        )
        assert not any(isinstance(c, ManagedProgressCallback) for c in resolved)


class _Aborted(RuntimeError):
    """Sentinel: raised from the spy to stop optimize() right after the
    reviewed call site, without running the rest of the optimization."""


class TestOptimizeCallSitePassesResolvedMode:
    """Wiring, not just the helper: ``optimize()``'s own call to
    ``_resolve_callbacks`` (the reviewed call site, optimized_function.py:~2050)
    must pass the per-call resolved mode, not ``self.execution_mode``. Drives
    the REAL ``optimize()`` coroutine and intercepts at the call site itself,
    aborting immediately after so the rest of the (unrelated) optimization
    pipeline never has to run.
    """

    async def _capture_execution_mode(self, opt_func, algorithm) -> list[str]:
        import traigent.core.optimized_function as of_mod

        captured: list[str] = []

        def spy(*args, **kwargs):
            captured.append(kwargs.get("execution_mode"))
            raise _Aborted

        original = of_mod._resolve_callbacks
        of_mod._resolve_callbacks = spy
        try:
            try:
                await opt_func.optimize(algorithm=algorithm)
            except _Aborted:
                pass
        finally:
            of_mod._resolve_callbacks = original
        return captured

    async def test_local_override_call_site_passes_local(self) -> None:
        opt_func = _auto_hybrid_optimized_function()

        captured = await self._capture_execution_mode(opt_func, "grid")

        assert captured == [ExecutionMode.LOCAL.value], (
            "optimize(algorithm='grid') must pass the per-call resolved LOCAL "
            f"mode to _resolve_callbacks, not the stale self.execution_mode; got {captured}"
        )

    async def test_unchanged_auto_call_site_passes_hybrid(self) -> None:
        opt_func = _auto_hybrid_optimized_function()

        captured = await self._capture_execution_mode(opt_func, None)

        assert captured == [ExecutionMode.HYBRID.value]
