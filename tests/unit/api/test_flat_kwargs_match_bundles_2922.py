"""Flat ``@optimize`` kwargs run the same checks as their grouped bundles (#2922)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

import traigent

SPACE = {"temperature": [0.1, 0.5]}


def _decorate(**kwargs):
    @traigent.optimize(configuration_space=SPACE, **kwargs)
    def fn(text: str) -> str:
        return text

    return fn


@pytest.mark.parametrize(
    ("flat", "bundle_key", "bundle", "field"),
    [
        (
            {"scoring_function": "x"},
            "evaluation",
            {"scoring_function": "x"},
            "scoring_function",
        ),
        (
            {"metric_functions": "x"},
            "evaluation",
            {"metric_functions": "x"},
            "metric_functions",
        ),
        ({"config_param": 5}, "injection", {"config_param": 5}, "config_param"),
        (
            {"custom_evaluator": 3},
            "evaluation",
            {"custom_evaluator": 3},
            "custom_evaluator",
        ),
        ({"eval_dataset": 42}, "evaluation", {"eval_dataset": 42}, "eval_dataset"),
        (
            {"framework_targets": "openai"},
            "injection",
            {"framework_targets": "openai"},
            "framework_targets",
        ),
    ],
)
def test_flat_matches_bundle_rejection(flat, bundle_key, bundle, field):
    with pytest.raises(PydanticValidationError) as bundled:
        _decorate(**{bundle_key: bundle})
    with pytest.raises(PydanticValidationError) as flat_exc:
        _decorate(**flat)
    assert field in str(flat_exc.value)
    assert str(flat_exc.value).splitlines()[1:] == str(bundled.value).splitlines()[1:]


@pytest.mark.parametrize(
    "bad", ["x", [1], float("nan")], ids=["str", "list", "nan-value"]
)
def test_default_config_rejected_at_decoration(bad):
    value = {"temperature": bad} if isinstance(bad, float) else bad
    with pytest.raises((TypeError, ValueError), match="default_config"):
        _decorate(default_config=value)


def test_call_time_callbacks_not_iterable_names_argument():
    with pytest.raises(TypeError, match="callbacks"):
        from traigent.core.optimized_function import _resolve_callbacks

        _resolve_callbacks(5, None, None, execution_mode="local")


def test_stored_callbacks_not_a_callback_rejected():
    from traigent.core.optimized_function import _coerce_callbacks

    with pytest.raises(TypeError, match="callbacks"):
        _coerce_callbacks(5)


@pytest.mark.parametrize("field", ["effectuation", "auto_override_frameworks"])
def test_flat_bool_flags_coerce_like_bundle(field):
    from traigent.api.decorators import _validate_flat_bundle_options

    none = dict.fromkeys(
        (
            "eval_dataset",
            "custom_evaluator",
            "scoring_function",
            "metric_functions",
            "config_param",
            "auto_override_frameworks",
            "framework_targets",
            "effectuation",
        )
    )
    none[field] = "no"
    assert _validate_flat_bundle_options(**none)[field] is False


@pytest.mark.parametrize("field", ["effectuation", "auto_override_frameworks"])
def test_flat_declined_flag_conflicts_with_bundle_true(field):
    with pytest.raises(TypeError, match=f"Conflicting values for '{field}'"):
        _decorate(**{field: "no"}, injection={field: True})


def test_flat_framework_targets_iterator_not_consumed():
    from traigent.api.decorators import _validate_flat_bundle_options

    none = dict.fromkeys(
        (
            "eval_dataset",
            "custom_evaluator",
            "scoring_function",
            "metric_functions",
            "config_param",
            "auto_override_frameworks",
            "framework_targets",
            "effectuation",
        )
    )
    none["framework_targets"] = iter(["openai.OpenAI.chat.completions.create"])
    assert _validate_flat_bundle_options(**none)["framework_targets"] == [
        "openai.OpenAI.chat.completions.create"
    ]


def test_flat_tuple_eval_dataset_matches_bundle_list():
    rows = ({"input": {"text": "a"}, "output": "a"},)
    fn_flat = _decorate(eval_dataset=rows)
    fn_bundle = _decorate(evaluation={"eval_dataset": rows})
    assert fn_flat.eval_dataset == fn_bundle.eval_dataset
    assert isinstance(fn_flat.eval_dataset, list)


def test_callback_missing_hooks_rejected():
    from traigent.core.optimized_function import _coerce_callbacks

    class Partial:
        def on_trial_complete(self, *a):
            return None

    with pytest.raises(TypeError, match="on_optimization_start"):
        _coerce_callbacks([Partial()])
