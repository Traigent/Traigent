"""configure_integrations warns about options that nothing reads (#3125)."""

from __future__ import annotations

import warnings
from dataclasses import fields

import pytest

import traigent.integrations.config as config_module
from traigent.integrations.config import IntegrationConfig, configure_integrations


@pytest.fixture(autouse=True)
def _restore_config():
    saved = config_module.integration_config
    config_module._WARNED_UNUSED_OPTIONS.clear()
    config_module.integration_config = IntegrationConfig()
    yield
    config_module.integration_config = saved


def test_non_default_value_warns_that_option_has_no_effect():
    with pytest.warns(UserWarning, match="strict_mode.*no effect"):
        configure_integrations(strict_mode=True)


@pytest.mark.parametrize("name", [f.name for f in fields(IntegrationConfig)])
def test_every_unread_option_warns_when_changed(name):
    default = getattr(IntegrationConfig(), name)
    if isinstance(default, bool):
        changed = not default
    elif isinstance(default, float):
        changed = 0.5
    else:
        changed = default + 1
    with pytest.warns(UserWarning, match=name):
        configure_integrations(**{name: changed})


def test_explicit_default_value_still_warns():
    with pytest.warns(UserWarning, match="sanitize_parameters"):
        configure_integrations(sanitize_parameters=True)


def test_warns_once_per_option_per_process():
    with pytest.warns(UserWarning, match="strict_mode"):
        configure_integrations(strict_mode=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        configure_integrations(strict_mode=False)


def test_warning_turned_into_error_is_not_recorded_as_warned():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning):
            configure_integrations(strict_mode=True)
    with pytest.warns(UserWarning, match="strict_mode"):
        configure_integrations(strict_mode=True)
