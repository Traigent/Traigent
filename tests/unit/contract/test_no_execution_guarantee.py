"""The headline guarantee: the contract executes nothing.

Agent bodies, metric bodies and custom evaluators are never called; no
``ConfigurationContext`` is entered; the network is never touched; and a secret
sitting in ``config`` never leaks into the serialized report (only its KEY does).
"""

from __future__ import annotations

import json
import socket
import warnings

import pytest

import traigent.config.context as context_module
from traigent.contract import validate_evaluation_contract

DATASET = [{"input": {"question": "q1"}, "output": "gold"}]


def _bomb_agent(question):
    raise AssertionError("the target function must never be executed")


def _bomb_agent_with_config(question, config):
    raise AssertionError("the target function must never be executed")


def _bomb_metric(output, expected):
    raise AssertionError("the metric function must never be executed")


@pytest.mark.parametrize("mode", ["context", "parameter", "seamless"])
def test_bodies_never_execute_across_all_modes(mode):
    agent = _bomb_agent_with_config if mode == "parameter" else _bomb_agent
    config = {"model": "gpt-4"} if mode != "context" else None

    report = validate_evaluation_contract(
        func=agent,
        dataset=DATASET,
        scoring_function=_bomb_metric,
        injection_mode=mode,
        config=config,
    )
    # A full report was produced although every callable body would raise.
    assert report.injection.effective_mode == mode
    assert len(report.call_shapes) == 1
    assert len(report.evaluator_bindings) == 1


def test_custom_evaluator_never_executes():
    import traigent

    def bomb_custom_eval(func, config, example):
        raise AssertionError("the custom evaluator must never be executed")

    def raw_agent(question):
        raise AssertionError("the target function must never be executed")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt_agent = traigent.optimize(
            configuration_space={"model": ["a", "b"]},
            custom_evaluator=bomb_custom_eval,
        )(raw_agent)

    report = validate_evaluation_contract(func=opt_agent, dataset=DATASET)
    assert "custom_evaluator" in report.unsupported


def test_configuration_context_never_entered(monkeypatch):
    def boom(self, *args, **kwargs):
        raise AssertionError("ConfigurationContext must never be entered")

    monkeypatch.setattr(context_module.ConfigurationContext, "__enter__", boom)

    for mode in ("context", "parameter", "seamless"):
        agent = _bomb_agent_with_config if mode == "parameter" else _bomb_agent
        config = {"model": "gpt-4"} if mode != "context" else None
        report = validate_evaluation_contract(
            func=agent,
            dataset=DATASET,
            scoring_function=_bomb_metric,
            injection_mode=mode,
            config=config,
        )
        assert report.injection.effective_mode == mode


def test_network_never_touched(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("no socket may be opened during contract validation")

    monkeypatch.setattr(socket, "socket", blocked)

    report = validate_evaluation_contract(
        func=_bomb_agent, dataset=DATASET, scoring_function=_bomb_metric
    )
    assert len(report.call_shapes) == 1


def test_secret_value_never_serialized():
    sentinel_secret = "sk-live-DO-NOT-LEAK-abcdef123456"
    report = validate_evaluation_contract(
        func=lambda question: "x",
        dataset=[{"input": {"question": "q1"}}],
        config={"api_key": sentinel_secret, "model": "gpt-4"},
    )
    serialized = json.dumps(report.to_dict())

    # The KEY is reported...
    assert "api_key" in serialized
    assert report.injection.config_keys == ("api_key", "model")
    # ...but the secret VALUE never appears anywhere in the report.
    assert sentinel_secret not in serialized
