"""Pre-trial objective-to-metric binding check (#2668)."""

from __future__ import annotations

import logging

from traigent.core.objective_preflight import warn_unproducible_objectives


def _warn(caplog, objectives, metric_functions=None, scoring_function=None):
    with caplog.at_level(logging.WARNING, logger="traigent.core.objective_preflight"):
        return warn_unproducible_objectives(
            metric_functions or {}, objectives, scoring_function
        )


def test_typo_objective_warns_with_did_you_mean(caplog):
    unmatched = _warn(caplog, ["acuracy"])
    assert unmatched == ["acuracy"]
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "Did you mean 'accuracy'" in text
    assert "acuracy" not in text  # customer-chosen names are never echoed


def test_builtin_objectives_do_not_warn(caplog):
    assert _warn(caplog, ["accuracy", "cost", "latency", "score"]) == []
    assert not caplog.records


def test_metric_function_covers_objective(caplog):
    assert _warn(caplog, ["relevance"], {"relevance": lambda **_: 1.0}) == []


def test_any_user_scorer_makes_binding_undecidable(caplog):
    # A mapping-returning scorer may emit any key, so stay silent.
    assert _warn(caplog, ["relevance"], {"scores": lambda **_: {}}) == []
    assert _warn(caplog, ["relevance"], None, lambda **_: {}) == []
    assert not caplog.records


def test_hostile_objective_name_is_not_echoed(caplog):
    evil = "x\nINFO forged /home/alice/secret.txt Alice_has_diabetes"
    unmatched = _warn(caplog, [evil])
    assert unmatched == [evil]
    text = " ".join(r.getMessage() for r in caplog.records)
    assert caplog.records
    assert "forged" not in text and "alice" not in text.lower()
    assert "\n" not in text


def test_misnamed_metric_key_warns_without_echoing_names(caplog):
    # objectives=["accuracy"], metric_functions={"acc": f}: the user's scorer is
    # silently ignored and the built-in accuracy is used instead.
    _warn(caplog, ["accuracy"], {"acc": lambda output, expected: 1.0})
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "built-in objective 'accuracy'" in text
    assert "may not be scored" in text
    assert "'acc'" not in text  # customer-chosen keys are never echoed


def test_informational_metric_next_to_objective_does_not_warn(caplog):
    _warn(
        caplog,
        ["accuracy"],
        {"accuracy": lambda **_: 1.0, "brevity": lambda **_: 1.0},
    )
    assert not caplog.records


def test_unrelated_extra_metric_key_does_not_warn(caplog):
    _warn(caplog, ["accuracy"], {"brevity": lambda **_: 1.0})
    assert not caplog.records


def test_mapping_returning_scorer_does_not_warn(caplog):
    _warn(caplog, ["accuracy"], {"acc": lambda **_: {"accuracy": 1.0}})

    def typed(output, expected) -> dict:
        return {"accuracy": 1.0}

    _warn(caplog, ["accuracy"], {"acc": typed})
    assert not caplog.records


def test_scoring_function_already_supplying_objective_does_not_warn(caplog):
    _warn(
        caplog,
        ["accuracy"],
        {"acc": lambda output, expected: 1.0},
        lambda output, expected: 1.0,
    )
    assert not caplog.records
