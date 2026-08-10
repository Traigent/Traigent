"""Call-time retired-preset rejection for a decorator that declares objectives (#2100).

The shape reported in #2100 is specifically this one:

    @traigent.optimize(..., objectives=["quality"])   # objectives on the DECORATOR
    fn.optimize_sync(strategy="quality_floor_min_cost")   # preset name at CALL time,
                                                          # objectives NOT repeated

Before the named-strategy-preset surface was removed, that call silently
replaced the decorator-declared objectives with the preset's own
``("accuracy", "cost")`` for the duration of the run, optimized for the wrong
thing, and returned the wrong winner. Nothing raised and nothing warned, and
because ``optimize()`` restored ``self.objective_schema`` in a ``finally``,
``fn.objectives`` still read ``["quality"]`` afterwards — so the substitution
was invisible even post-hoc.

What these tests pin is the *entry point*: the retired preset name is refused
before any trial is evaluated. They deliberately do not claim that "objectives
cannot be substituted" (no code substitutes them any more) or that "the quality
floor binds" (there is no floor any more).

Sibling coverage, not duplicated here:

* ``tests/unit/api/test_strategy_preset_absence.py`` — module/symbol absence,
  and the rejection message for ``traigent.optimize(strategy=...)`` and for
  ``OptimizedFunction._resolve_runtime_strategy_argument`` called directly.
* ``tests/unit/api/test_decorators.py::TestRemovedDecoratorCompatibilityOptions``
  — decorator-level rejection, and ``optimize_sync(strategy_params=...)`` on a
  function with no decorator objectives.

Not asserted here, on purpose: the residual #2101 mis-scoring, where a graded
``metric_functions`` scorer registered under a name other than ``accuracy``
still leaves ``metrics["accuracy"] == 0.0``. That reproduces identically on
``develop``, where no preset code has ever existed, so it is not caused by
anything this change touches; its root cause is being fixed separately. Pinning
it here would cement the buggy behaviour as expected.
"""

from __future__ import annotations

from typing import Any

import pytest

import traigent
from traigent.core.optimized_function import OptimizedFunction
from traigent.evaluators.base import Dataset, EvaluationExample

# Declared on the decorator, and never repeated at call time. This is the value
# the retired preset used to overwrite.
DECORATOR_OBJECTIVES = ["quality"]

RETIRED_PRESET_CALLS = [
    # The exact shape reported in #2100.
    ("quality_floor_min_cost", {"floor": 0.8}),
    # The params-free preset: `strategy=` is then the only kwarg that differs
    # from a legal call, so the preset *name* is provably what is refused —
    # not the separately-rejected `strategy_params=`.
    ("pareto_frontier", None),
]


def assert_names_the_removal(message: str, preset_name: str) -> None:
    """Assert the message tells the user the *preset* is gone and what replaces it.

    Keyed to the substantive wording, not to the version number: "0.27.0" is
    owner-configurable, so pinning it would add brittleness without adding
    proof. What is pinned is the part a user acts on — that this name was a
    named strategy preset, that such presets were removed, and where to go
    instead (``algorithm=`` / ``objectives=``).

    The failure this exists to catch: a message like "strategy=... is not a
    valid optimizer; presets were removed in 0.27.0 ..." still contains
    "removed", so a bare ``"removed" in message`` check passes while the user
    is told their name is a bad optimizer rather than a retired feature.
    """
    lowered = message.lower()
    assert preset_name in message, message
    assert "named strategy preset" in lowered, message
    assert "removed" in lowered, message
    assert "algorithm=" in lowered, message
    assert "objectives=" in lowered, message


class _Recorder:
    """Records anything that would only happen if a trial actually ran."""

    def __init__(self) -> None:
        self.scored: list[tuple[Any, Any]] = []
        self.evaluated: list[Any] = []
        self.executed: list[Any] = []

    def score(
        self, actual_output: Any = None, expected_output: Any = None, **_: Any
    ) -> float:
        self.scored.append((actual_output, expected_output))
        return 1.0

    def evaluate(self, func: Any, config: Any, example: Any) -> dict[str, float]:
        self.evaluated.append(config)
        return {"quality": 1.0}


@pytest.fixture
def recorder() -> _Recorder:
    return _Recorder()


@pytest.fixture(autouse=True)
def _no_trial_may_run(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    """Make reaching execution a loud failure rather than a real (paid) run."""

    async def _fail(self: Any, *args: Any, **kwargs: Any) -> Any:
        recorder.executed.append(kwargs)
        raise AssertionError(
            "_execute_optimization ran even though the call was supposed to be "
            "rejected before any trial"
        )

    monkeypatch.setattr(OptimizedFunction, "_execute_optimization", _fail)


@pytest.fixture
def decorated_function(recorder: _Recorder) -> OptimizedFunction:
    """A function whose objectives are declared on the decorator only."""
    dataset = Dataset(
        [
            EvaluationExample(input_data={"text": "hello"}, expected_output="HELLO"),
            EvaluationExample(input_data={"text": "world"}, expected_output="WORLD"),
        ]
    )

    @traigent.optimize(
        configuration_space={"model": ["fast", "smart"]},
        objectives=DECORATOR_OBJECTIVES,
        eval_dataset=dataset,
        metric_functions={"quality": recorder.score},
        max_trials=2,
    )
    def classify(text: str) -> str:
        config = traigent.get_config()
        return f"{config.get('model', 'fast')}:{text.upper()}"

    return classify


@pytest.mark.parametrize("strategy,strategy_params", RETIRED_PRESET_CALLS)
def test_calltime_retired_preset_name_raises_typeerror_naming_the_removal_2100(
    decorated_function: OptimizedFunction,
    strategy: str,
    strategy_params: dict[str, Any] | None,
) -> None:
    """The #2100 entry point raises, with a message that names the removal.

    Proves the call is refused and that the message identifies the name as a
    *named strategy preset* that was removed, and points at ``algorithm=`` /
    ``objectives=`` — not merely that the word "removed" appears somewhere in
    it. Nothing here claims anything about what the run would have optimized.
    """
    with pytest.raises(TypeError) as excinfo:
        decorated_function.optimize_sync(
            strategy=strategy,
            strategy_params=strategy_params,
            # objectives= is deliberately NOT passed: the decorator's are the
            # only ones in play, which is what made #2100 invisible.
        )

    assert_names_the_removal(str(excinfo.value), strategy)


def test_calltime_retired_preset_rejection_leaves_decorator_objectives_unchanged_2100(
    decorated_function: OptimizedFunction,
) -> None:
    """The rejected call does not touch the decorator-declared objectives.

    Proves the objective schema is the same object before and after — the
    rejection happens before the schema is swapped, so there is no swap to
    restore.
    """
    schema_before = decorated_function.objective_schema

    with pytest.raises(TypeError):
        decorated_function.optimize_sync(
            strategy="quality_floor_min_cost", strategy_params={"floor": 0.8}
        )

    assert decorated_function.objectives == DECORATOR_OBJECTIVES
    assert decorated_function.objective_schema is schema_before


class TestConstructorDoor:
    """``OptimizedFunction(...)`` is the third door onto the removed feature.

    Its ``**kwargs`` accepted ``strategy=``, ``strategy_params=`` and the former
    internal ``strategy_preset=`` as unknown keywords: they were stored in
    ``_decorator_runtime_overrides``, merged into the optimizer's
    ``algorithm_config``, and never read. Construction succeeded and the run
    returned an ordinary winner for a call that asked for a preset — #2100's
    silent-substitution shape, reached through the public constructor rather
    than through ``optimize()``.
    """

    @staticmethod
    def _construct(**kwargs: Any) -> OptimizedFunction:
        def plain(text: str) -> str:
            return text

        return OptimizedFunction(
            func=plain,
            configuration_space={"model": ["fast", "smart"]},
            objectives=DECORATOR_OBJECTIVES,
            **kwargs,
        )

    @pytest.mark.parametrize(
        "preset_name", ["quality_floor_min_cost", "pareto_frontier"]
    )
    @pytest.mark.parametrize("parameter", ["strategy", "strategy_preset"])
    def test_constructor_refuses_a_retired_preset_name(
        self, parameter: str, preset_name: str
    ) -> None:
        """By name, with the same message the other two doors give."""
        with pytest.raises(TypeError) as excinfo:
            self._construct(**{parameter: preset_name})

        assert_names_the_removal(str(excinfo.value), preset_name)

    def test_constructor_refuses_strategy_params(self) -> None:
        """Any non-``None`` ``strategy_params`` is refused, naming the removal."""
        with pytest.raises(TypeError) as excinfo:
            self._construct(strategy_params={"floor": 0.8})

        message = str(excinfo.value)
        assert "strategy_params is no longer supported" in message
        assert "named strategy preset" in message.lower()

    def test_constructor_still_accepts_unrelated_kwargs(self) -> None:
        """The guard must not turn the open ``**kwargs`` door into a wall.

        Two things that legitimately flow into ``_decorator_runtime_overrides``
        keep working, and a non-preset ``strategy=`` value keeps whatever
        behaviour it already had rather than being swept into the refusal.
        """
        function = self._construct(
            strategy="grid", experiment_name="unrelated", max_trials=3
        )

        assert function.max_trials == 3
        assert function._decorator_runtime_overrides["strategy"] == "grid"

    def test_constructor_refusal_is_not_a_generic_unknown_kwarg_error(self) -> None:
        """``strategy_params=None`` is indistinguishable from not passing it."""
        function = self._construct(strategy=None, strategy_params=None)

        assert function.objectives == DECORATOR_OBJECTIVES


def test_calltime_retired_preset_is_rejected_before_any_evaluation_runs_2100(
    decorated_function: OptimizedFunction, recorder: _Recorder
) -> None:
    """Nothing is evaluated: no scorer call, no evaluator call, no execution.

    Proves the rejection is upstream of evaluation by checking recorders that
    a single trial would have populated, rather than inferring it from the
    exception alone.
    """
    with pytest.raises(TypeError):
        decorated_function.optimize_sync(
            strategy="quality_floor_min_cost",
            strategy_params={"floor": 0.8},
            custom_evaluator=recorder.evaluate,
        )

    assert recorder.scored == []
    assert recorder.evaluated == []
    assert recorder.executed == []
