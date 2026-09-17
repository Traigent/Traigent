"""ExecutionOptions.require_run_id contract tests (G1 v1.0.1 (g)).

The option is tri-state (``bool | None``, default ``None`` = unspecified).
Unspecified changes nothing: the SDK still defers to
``TRAIGENT_REQUIRE_RUN_ID`` exactly as attempt 1 wired it
(``backend_session_manager.py::_effective_require_run_id``). An explicit
``True`` OR ``False`` always overrides the environment variable -- attempt 2
lost this for explicit ``False`` (G1 v1.0.1 (g), F2): omitting the forwarded
key for a falsy value made explicit ``False`` indistinguishable from
"unspecified", so it lost to ``TRAIGENT_REQUIRE_RUN_ID=true`` instead of
overriding it.
"""

from __future__ import annotations

from typing import Any

import pytest

from traigent.api.decorators import ExecutionOptions, optimize
from traigent.api.types import TrialResult
from traigent.cloud.client import RunIdMissingError, SessionContractError
from traigent.config.types import TraigentConfig
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import (
    BaseEvaluator,
    Dataset,
    EvaluationExample,
    EvaluationResult,
)
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
    def test_default_is_unspecified(self) -> None:
        assert ExecutionOptions().require_run_id is None

    def test_true_accepted(self) -> None:
        assert ExecutionOptions(require_run_id=True).require_run_id is True

    def test_explicit_false_accepted_and_distinct_from_unspecified(self) -> None:
        assert ExecutionOptions(require_run_id=False).require_run_id is False

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

    def test_unspecified_without_execution_options(self) -> None:
        @optimize(
            configuration_space={"model": ["cheap", "strong"]},
            objectives=["accuracy"],
        )
        def test_func(text: str, model: str = "cheap") -> str:
            return f"Response: {text} ({model})"

        assert test_func.require_run_id is None

    def test_explicit_false_execution_bundle_plumbs_onto_optimized_function(
        self,
    ) -> None:
        @optimize(
            configuration_space={"model": ["cheap", "strong"]},
            objectives=["accuracy"],
            execution=ExecutionOptions(require_run_id=False),
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

    def test_explicit_false_forwards_key_as_false(self) -> None:
        """F2 regression: explicit False must reach orchestrator.config as
        False, not be omitted like "unspecified" is (G1 v1.0.1 (g), F2)."""
        optimized_func = _make_optimized_function(require_run_id=False)
        orchestrator = _build_orchestrator(optimized_func)
        assert orchestrator.config.get("require_run_id") is False


class TestRequireRunIdEnvPrecedence:
    """Precedence: explicit True overrides the env var; default defers to it."""

    def test_explicit_true_overrides_absent_env(self, monkeypatch) -> None:
        monkeypatch.delenv("TRAIGENT_REQUIRE_RUN_ID", raising=False)
        optimized_func = _make_optimized_function(require_run_id=True)
        orchestrator = _build_orchestrator(optimized_func)
        assert orchestrator.backend_session_manager._effective_require_run_id() is True

    def test_default_defers_to_env_true(self, monkeypatch) -> None:
        monkeypatch.setenv("TRAIGENT_REQUIRE_RUN_ID", "true")
        optimized_func = _make_optimized_function()
        orchestrator = _build_orchestrator(optimized_func)
        assert orchestrator.backend_session_manager._effective_require_run_id() is True

    def test_default_defers_to_env_absent(self, monkeypatch) -> None:
        monkeypatch.delenv("TRAIGENT_REQUIRE_RUN_ID", raising=False)
        optimized_func = _make_optimized_function()
        orchestrator = _build_orchestrator(optimized_func)
        assert orchestrator.backend_session_manager._effective_require_run_id() is False


class TestRequireRunIdTriStateMatrix:
    """F2 (G1 v1.0.1 (g)): explicit True/False × env true/false/unset, full
    matrix. An explicit option value always wins; unspecified defers to
    TRAIGENT_REQUIRE_RUN_ID (unset resolves to False)."""

    @pytest.mark.parametrize(
        "option_value,env_value,expected",
        [
            # explicit True always wins
            (True, "true", True),
            (True, "false", True),
            (True, None, True),
            # explicit False always wins -- this is the F2 regression case
            (False, "true", False),
            (False, "false", False),
            (False, None, False),
            # unspecified (None) defers to the environment
            (None, "true", True),
            (None, "false", False),
            (None, None, False),
        ],
    )
    def test_effective_flag(
        self, monkeypatch, option_value, env_value, expected
    ) -> None:
        if env_value is None:
            monkeypatch.delenv("TRAIGENT_REQUIRE_RUN_ID", raising=False)
        else:
            monkeypatch.setenv("TRAIGENT_REQUIRE_RUN_ID", env_value)

        kwargs = {} if option_value is None else {"require_run_id": option_value}
        optimized_func = _make_optimized_function(**kwargs)
        orchestrator = _build_orchestrator(optimized_func)

        assert (
            orchestrator.backend_session_manager._effective_require_run_id() is expected
        )


class TestRequireRunIdPublicEntryPoint:
    """F1 (G1 v1.0.1 (g)): the typed exception must reach the caller of the
    PUBLIC ``.optimize()`` entry point unchanged -- not wrapped into a generic
    OptimizationError with the typed error demoted to ``__cause__``."""

    @staticmethod
    def _dataset() -> Dataset:
        return Dataset(
            [EvaluationExample({"text": "case-0"}, "ok")],
            name="require_run_id_public_entry_point",
        )

    @staticmethod
    def _make_evaluator(counter: list[int]):
        def _evaluator(func, config, example):
            counter.append(1)
            raise AssertionError(
                "evaluator must never be called: require_run_id must fail "
                "closed at session-create time, before any trial"
            )

        return _evaluator

    def _decorated(self, counter: list[int]):
        @optimize(
            eval_dataset=self._dataset(),
            objectives=["accuracy"],
            configuration_space={"p": [0]},
            execution=ExecutionOptions(require_run_id=True, offline=True),
            custom_evaluator=self._make_evaluator(counter),
        )
        def answer(text: str, config) -> str:
            return "ok"

        return answer

    @pytest.mark.asyncio
    async def test_raises_run_id_missing_error_not_optimization_error(self) -> None:
        counter: list[int] = []
        answer = self._decorated(counter)

        with pytest.raises(RunIdMissingError):
            await answer.optimize()

        assert counter == []

    @pytest.mark.asyncio
    async def test_raises_session_contract_error(self) -> None:
        """RunIdMissingError IS a SessionContractError -- callers catching the
        broader typed base must see it too, unwrapped."""
        counter: list[int] = []
        answer = self._decorated(counter)

        with pytest.raises(SessionContractError):
            await answer.optimize()

        assert counter == []
