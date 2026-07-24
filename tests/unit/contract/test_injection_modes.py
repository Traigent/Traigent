"""Injection-resolution stage of the evaluation contract.

Exercises context / parameter / seamless modes, grouped vs flat option
equivalence and conflict, the removed ``attribute``/``decorator`` modes, the
``OptimizedFunction`` decorator path, and the invariant that the dead
``_traigent_injection_mode`` attribute is ignored.
"""

from __future__ import annotations

import pytest

import traigent
from traigent.api.decorators import InjectionOptions
from traigent.contract import ContractCode, validate_evaluation_contract

from ._support import (
    error_codes,
    find_finding,
    finding_codes,
    source_unavailable_function,
)

DATASET = [{"input": {"question": "q1"}}]


def _agent(question):
    return "x"


def _agent_with_config(question, config):
    return "x"


# --------------------------------------------------------------------------- #
# Context mode
# --------------------------------------------------------------------------- #
def test_context_mode_default():
    report = validate_evaluation_contract(func=_agent, dataset=DATASET)
    assert report.injection.effective_mode == "context"
    assert report.injection.source == "default"
    assert report.injection.provider_class == "ContextBasedProvider"
    assert report.injection.setup_ok is True


def test_context_mode_does_not_inject_config_into_call():
    report = validate_evaluation_contract(
        func=_agent, dataset=DATASET, config={"model": "m", "temperature": "0.2"}
    )
    shape = report.call_shapes[0]
    # Config keys are REPORTED, but never appear in the (context-mode) call.
    assert report.injection.config_keys == ("model", "temperature")
    assert shape.runtime_keyword_names == ("question",)
    assert shape.effective_keyword_names == ("question",)


# --------------------------------------------------------------------------- #
# Parameter mode
# --------------------------------------------------------------------------- #
def test_parameter_mode_present():
    report = validate_evaluation_contract(
        func=_agent_with_config,
        dataset=DATASET,
        injection_mode="parameter",
        config={"model": "m"},
    )
    assert report.ok
    assert report.injection.effective_mode == "parameter"
    assert report.injection.provider_class == "ParameterBasedProvider"
    assert report.injection.setup_ok is True
    # The injected config parameter shows up in the EFFECTIVE call shape only.
    shape = report.call_shapes[0]
    assert "config" in shape.effective_keyword_names
    assert "config" not in shape.runtime_keyword_names
    assert shape.bind_ok is True


def test_parameter_mode_missing_param():
    report = validate_evaluation_contract(
        func=_agent, dataset=DATASET, injection_mode="parameter"
    )
    assert report.ok is False
    assert report.injection.setup_ok is False
    finding = find_finding(report, ContractCode.INJECTION_CONFIG_PARAM_MISSING)
    assert finding is not None
    assert finding.severity == "error"


def test_parameter_mode_custom_config_param_name():
    def agent(question, cfg):
        return "x"

    report = validate_evaluation_contract(
        func=agent,
        dataset=DATASET,
        injection_mode="parameter",
        config_param="cfg",
        config={"model": "m"},
    )
    assert report.ok
    assert report.injection.config_param == "cfg"
    assert "cfg" in report.call_shapes[0].effective_keyword_names


def test_parameter_mode_custom_param_missing_reports_name():
    def agent(question):
        return "x"

    report = validate_evaluation_contract(
        func=agent, dataset=DATASET, injection_mode="parameter", config_param="settings"
    )
    finding = find_finding(report, ContractCode.INJECTION_CONFIG_PARAM_MISSING)
    assert finding is not None
    assert "settings" in finding.message


# --------------------------------------------------------------------------- #
# Grouped vs flat options
# --------------------------------------------------------------------------- #
def test_grouped_options_equivalent_to_flat():
    flat = validate_evaluation_contract(
        func=_agent_with_config,
        dataset=DATASET,
        injection_mode="parameter",
        config_param="config",
    )
    grouped = validate_evaluation_contract(
        func=_agent_with_config,
        dataset=DATASET,
        injection_options=InjectionOptions(
            injection_mode="parameter", config_param="config"
        ),
    )
    assert flat.injection.effective_mode == grouped.injection.effective_mode
    assert flat.injection.config_param == grouped.injection.config_param
    assert flat.injection.source == "flat_kwargs"
    assert grouped.injection.source == "injection_options"


def test_grouped_flat_conflict():
    report = validate_evaluation_contract(
        func=_agent_with_config,
        dataset=DATASET,
        injection_mode="parameter",
        injection_options=InjectionOptions(injection_mode="seamless"),
    )
    assert report.ok is False
    assert error_codes(report) == [ContractCode.INJECTION_OPTIONS_CONFLICT]


# --------------------------------------------------------------------------- #
# Removed modes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("removed_mode", ["attribute", "decorator"])
def test_removed_modes_unsupported(removed_mode):
    report = validate_evaluation_contract(
        func=_agent, dataset=DATASET, injection_mode=removed_mode
    )
    assert report.ok is False
    finding = find_finding(report, ContractCode.INJECTION_MODE_UNSUPPORTED)
    assert finding is not None
    assert finding.severity == "error"


# --------------------------------------------------------------------------- #
# Seamless mode
# --------------------------------------------------------------------------- #
def test_seamless_source_available_reports_injected_names():
    def agent(question):
        model = "default"
        return model

    report = validate_evaluation_contract(
        func=agent,
        dataset=DATASET,
        injection_mode="seamless",
        config={"model": "gpt-4"},
    )
    assert report.injection.effective_mode == "seamless"
    assert report.injection.provider_class == "SeamlessParameterProvider"
    assert report.injection.setup_ok is True
    assert report.injection.seamless_injected_names == ("model",)
    assert finding_codes(report) == []


def test_seamless_source_unavailable_warns():
    # A function whose source cannot be introspected -> AST injection fails.
    report = validate_evaluation_contract(
        func=source_unavailable_function(),
        dataset=DATASET,
        injection_mode="seamless",
        config={"model": "gpt-4"},
    )
    finding = find_finding(report, ContractCode.SEAMLESS_INJECTION_UNAVAILABLE)
    assert finding is not None
    assert finding.severity == "warning"
    # Best-effort: seamless failure is never fatal.
    assert report.ok is True


def test_seamless_on_plain_lambda_succeeds():
    # NOTE (reality vs matrix): the matrix expected a lambda to raise
    # SEAMLESS_INJECTION_UNAVAILABLE, but a plain lambda has retrievable source
    # and no reassignable variables, so it transforms cleanly with an empty
    # injected-names set. The genuine failure path is source_unavailable above.
    report = validate_evaluation_contract(
        func=lambda question: "x",
        dataset=DATASET,
        injection_mode="seamless",
        config={"model": "gpt-4"},
    )
    assert report.injection.setup_ok is True
    assert report.injection.seamless_injected_names == ()
    assert find_finding(report, ContractCode.SEAMLESS_INJECTION_UNAVAILABLE) is None


# --------------------------------------------------------------------------- #
# OptimizedFunction path
# --------------------------------------------------------------------------- #
def test_optimized_function_source():
    @traigent.optimize(configuration_space={"model": ["a", "b"]})
    def opt_agent(question):
        return traigent.get_config().model

    report = validate_evaluation_contract(func=opt_agent, dataset=DATASET)
    assert report.injection.source == "optimized_function"
    assert report.injection.effective_mode == "context"


def test_optimized_function_parameter_mode_from_decorator():
    @traigent.optimize(
        configuration_space={"model": ["a", "b"]},
        injection_mode="parameter",
        config_param="config",
    )
    def opt_agent(question, config):
        return config.model

    report = validate_evaluation_contract(
        func=opt_agent, dataset=DATASET, config={"model": "m"}
    )
    assert report.injection.effective_mode == "parameter"
    assert report.injection.config_param == "config"
    assert "config" in report.call_shapes[0].effective_keyword_names


# --------------------------------------------------------------------------- #
# The dead attribute must be ignored
# --------------------------------------------------------------------------- #
def test_traigent_injection_mode_attribute_is_ignored():
    """The unused ``_traigent_injection_mode`` attribute must not affect shaping.

    Production never sets it and always resolves "context"; the contract must do
    the same regardless of any stray attribute a caller planted on the function.
    """

    def agent(question):
        return "x"

    # Plant a bogus attribute that, if honoured, would switch to parameter mode.
    agent._traigent_injection_mode = "parameter"  # type: ignore[attr-defined]

    report = validate_evaluation_contract(
        func=agent, dataset=DATASET, config={"model": "m"}
    )
    shape = report.call_shapes[0]
    # Still context mode: no config merged into the runtime call shape.
    assert report.injection.effective_mode == "context"
    assert shape.runtime_keyword_names == ("question",)
    assert shape.effective_keyword_names == ("question",)
