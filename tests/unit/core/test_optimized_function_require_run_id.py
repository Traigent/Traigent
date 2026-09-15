"""ExecutionOptions.require_run_id contract tests (G1 v1.0.1 (g)).

The option is a plain ``bool`` (default ``False``, no tri-state at the public
surface). Left at the default it changes nothing: the SDK still defers to
``TRAIGENT_REQUIRE_RUN_ID`` exactly as attempt 1 wired it
(``backend_session_manager.py::_effective_require_run_id``). Set to ``True``
it forces the same early ``RunIdMissingError`` regardless of the env var.
"""

from __future__ import annotations

from typing import Any

from traigent.api.decorators import ExecutionOptions, optimize
from traigent.api.types import TrialResult
from traigent.config.types import TraigentConfig
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import BaseEvaluator, Dataset, EvaluationResult
from traigent.optimizers.base import BaseOptimizer


class _StubOptimizer(BaseOptimizer):
    """Minimal concrete optimizer: never called, only satisfies the ABC."""

    def suggest_next_trial(self, history: list[TrialResult]) -> dict[str, Any]:
        return {}

    def should_stop(self, history: list[TrialResult]) -> bool:
        return True


class _StubEvaluator(BaseEvaluator):
    """Minimal concrete evaluator: never called, only satisfies the ABC."""

    async def evaluate(
        self,
        func: Any,
        config: dict[str, Any],
        dataset: Dataset,
        *,
        sample_lease: Any = None,
        progress_callback: Any = None,
        budget: Any = None,
    ) -> EvaluationResult:
        raise NotImplementedError


def _make_optimized_function(**kwargs: Any) -> OptimizedFunction:
    def func(text: str) -> str:
        return text

    kwargs.setdefault("configuration_space", {"p": [0]})
    kwargs.setdefault("objectives", ["accuracy"])
    return OptimizedFunction(func=func, **kwargs)


def _build_orchestrator(optimized_func: OptimizedFunction) -> Any:
    """Build a real orchestrator the same way OptimizedFunction does, offline."""
    optimizer = _StubOptimizer({"p": [0]}, ["accuracy"])
    evaluator = _StubEvaluator()
    return optimized_func._build_optimization_orchestrator(
        optimizer=optimizer,
        evaluator=evaluator,
        max_trials=1,
        max_total_examples_value=None,
        timeout=None,
        callbacks=None,
        traigent_config=TraigentConfig(no_egress=True, enable_usage_analytics=False),
        effective_parallel_trials=None,
        samples_include_pruned_value=True,
        algorithm_kwargs={},
        artifact_fingerprint_payload={},
    )


class TestRequireRunIdExecutionOption:
    def test_default_is_false(self) -> None:
        assert ExecutionOptions().require_run_id is False

    def test_true_accepted(self) -> None:
        assert ExecutionOptions(require_run_id=True).require_run_id is True

    def test_execution_bundle_plumbs_onto_optimized_function(self) -> None:
        @optimize(
            configuration_space={"model": ["cheap", "strong"]},
            objectives=["accuracy"],
            execution=ExecutionOptions(require_run_id=True),
        )
        def test_func(text: str, model: str = "cheap") -> str:
            return f"Response: {text} ({model})"

        assert isinstance(test_func, OptimizedFunction)
        assert test_func.require_run_id is True

    def test_default_false_without_execution_options(self) -> None:
        @optimize(
            configuration_space={"model": ["cheap", "strong"]},
            objectives=["accuracy"],
        )
        def test_func(text: str, model: str = "cheap") -> str:
            return f"Response: {text} ({model})"

        assert test_func.require_run_id is False


class TestRequireRunIdOrchestratorPlumbing:
    """The field must reach orchestrator.config the same way sibling options do."""

    def test_true_forces_orchestrator_config_true(self) -> None:
        optimized_func = _make_optimized_function(require_run_id=True)
        orchestrator = _build_orchestrator(optimized_func)
        assert orchestrator.config.get("require_run_id") is True

    def test_default_leaves_orchestrator_config_unset(self) -> None:
        optimized_func = _make_optimized_function()
        orchestrator = _build_orchestrator(optimized_func)
        assert orchestrator.config.get("require_run_id") is None


class TestRequireRunIdEnvPrecedence:
    """Precedence: explicit True overrides the env var; default defers to it."""

    def test_explicit_true_overrides_absent_env(self, monkeypatch) -> None:
        monkeypatch.delenv("TRAIGENT_REQUIRE_RUN_ID", raising=False)
        optimized_func = _make_optimized_function(require_run_id=True)
        orchestrator = _build_orchestrator(optimized_func)
        assert (
            orchestrator.backend_session_manager._effective_require_run_id() is True
        )

    def test_default_defers_to_env_true(self, monkeypatch) -> None:
        monkeypatch.setenv("TRAIGENT_REQUIRE_RUN_ID", "true")
        optimized_func = _make_optimized_function()
        orchestrator = _build_orchestrator(optimized_func)
        assert (
            orchestrator.backend_session_manager._effective_require_run_id() is True
        )

    def test_default_defers_to_env_absent(self, monkeypatch) -> None:
        monkeypatch.delenv("TRAIGENT_REQUIRE_RUN_ID", raising=False)
        optimized_func = _make_optimized_function()
        orchestrator = _build_orchestrator(optimized_func)
        assert (
            orchestrator.backend_session_manager._effective_require_run_id() is False
        )
