"""Regression coverage for boolean configuration_space cloud diagnostics."""

import logging
from unittest.mock import Mock

from traigent.cloud.api_operations import ApiOperations, _typed_configuration_space
from traigent.cloud.models import SessionCreationRequest

LOGGER_NAME = "traigent.cloud.api_operations"


def _request(configuration_space):
    return SessionCreationRequest(
        function_name="answer_question",
        configuration_space=configuration_space,
        objectives=["accuracy"],
        dataset_metadata={"size": 3},
        max_trials=5,
        metadata={"evaluation_set": "dev"},
    )


def _bool_config_warnings(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
        and "configuration_space parameter(s)" in record.getMessage()
        and "boolean values" in record.getMessage()
    ]


def test_boolean_choice_list_warns_once_with_field_and_workaround(caplog):
    config_space = {
        "include_schema": [True, False],
        "model": ["cheap", "strong"],
        "top_k": [1, 3],
        "temperature": {"low": 0.0, "high": 1.0},
    }

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        normalized = _typed_configuration_space(config_space)

    warnings = _bool_config_warnings(caplog)
    assert len(warnings) == 1
    warning = warnings[0]
    assert "include_schema" in warning
    assert "#1488" in warning
    assert "strings" in warning
    assert "0/1" in warning
    assert normalized["include_schema"] == {
        "type": "categorical",
        "choices": [True, False],
    }
    assert normalized["model"] == {
        "type": "categorical",
        "choices": ["cheap", "strong"],
    }


def test_non_boolean_configuration_space_does_not_warn(caplog):
    config_space = {
        "model": ["cheap", "strong"],
        "top_k": [1, 3],
        "temperature": {"low": 0.0, "high": 1.0},
        "mode": {"type": "categorical", "choices": ["with", "without"]},
        "ratio": (0.1, 0.5),
    }

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        _typed_configuration_space(config_space)

    assert _bool_config_warnings(caplog) == []


def test_scalar_bool_and_typed_categorical_bool_are_detected(caplog):
    config_space = {
        "enabled": True,
        "include_schema": {"type": "categorical", "choices": [True, False]},
        "format": "json",
    }

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        normalized = _typed_configuration_space(config_space)

    warnings = _bool_config_warnings(caplog)
    assert len(warnings) == 1
    warning = warnings[0]
    assert "enabled" in warning
    assert "include_schema" in warning
    assert normalized["enabled"] == {"type": "categorical", "choices": [True]}
    assert normalized["include_schema"] is config_space["include_schema"]


def test_values_key_in_typed_categorical_bool_is_detected(caplog):
    config_space = {
        "use_context": {"type": "categorical", "values": [True, False]},
    }

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        normalized = _typed_configuration_space(config_space)

    warnings = _bool_config_warnings(caplog)
    assert len(warnings) == 1
    assert "use_context" in warnings[0]
    assert normalized["use_context"] is config_space["use_context"]


def test_explicit_legacy_payload_warns_once_without_changing_search_space(
    monkeypatch, caplog
):
    monkeypatch.setenv("TRAIGENT_SESSION_CONTRACT", "legacy")
    config_space = {
        "include_schema": [True, False],
        "model": ["cheap", "strong"],
    }

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        payload = ApiOperations(Mock())._build_session_payload(
            _request(config_space), max_trials=5
        )

    warnings = _bool_config_warnings(caplog)
    assert len(warnings) == 1
    assert "include_schema" in warnings[0]
    assert payload["search_space"] is config_space
