"""traigent check scores the baseline with the function's own scorers (#2654)."""

from __future__ import annotations

from types import SimpleNamespace

from traigent.cli.optimization_validator import OptimizationValidator
from traigent.cli.validation_types import OptimizedFunction


def _info(**func_attrs):
    func = SimpleNamespace(execution_policy=None, algorithm=None, offline=True)
    for key, value in func_attrs.items():
        setattr(func, key, value)
    return OptimizedFunction(
        name="f",
        func=func,
        decorator_config={},
        default_params={},
        eval_dataset=None,
        objectives=["quality"],
    )


def quality(output, expected):
    return 1.0 if output == expected else 0.0


def test_baseline_evaluator_receives_metric_functions():
    info = _info(metric_functions={"quality": quality}, scoring_function=None)
    evaluator = OptimizationValidator()._create_evaluator(info, is_mock_mode=False)
    assert evaluator.metric_functions.get("quality") is quality


def test_baseline_evaluator_receives_scoring_function_as_accuracy():
    def scorer(output, expected=None, llm_metrics=None):
        return 1.0

    info = _info(metric_functions=None, scoring_function=scorer)
    info.objectives = ["accuracy"]
    evaluator = OptimizationValidator()._create_evaluator(info, is_mock_mode=False)
    assert evaluator.metric_functions.get("accuracy") is scorer
