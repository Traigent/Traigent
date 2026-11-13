"""One boolean grammar for TRAIGENT_* flags (#2989).

Spellings are stripped and case-folded; unrecognised values resolve to the
protective value (approval/relaxing flags off, privacy flags on only for
recognised spellings).
"""

from __future__ import annotations

import pytest

TRUE_SPELLINGS = ["true", " true", "True", "1", "yes", "on", "yes\n", "ON "]
FALSE_SPELLINGS = ["", "0", "False", "off", "no", "maybe", "treu"]


def test_run_approved_exactly_one(monkeypatch):
    from traigent.core.ci_approval import _check_env_var_approval

    monkeypatch.setenv("TRAIGENT_RUN_APPROVED", "1")
    assert _check_env_var_approval() is True


@pytest.mark.parametrize(
    "value", ["true", " true", "True", "yes", "on", "1 ", " 1", "1\n", "", "0"]
)
def test_run_approved_extra_spellings_do_not_approve(monkeypatch, value):
    from traigent.core.ci_approval import _check_env_var_approval

    monkeypatch.setenv("TRAIGENT_RUN_APPROVED", value)
    assert _check_env_var_approval() is False


@pytest.mark.parametrize("value", ["true", "True", "TRUE"])
def test_cost_approved_true_spellings(monkeypatch, tmp_path, value):
    from traigent.core.cost_enforcement import is_cost_preapproved

    monkeypatch.setenv("TRAIGENT_COST_APPROVED", value)
    assert is_cost_preapproved(token_path=tmp_path / "none") is True


@pytest.mark.parametrize(
    "value",
    ["1", "yes", "on", " true", "true\n", "", "0", "off", "maybe", "treu"],
)
def test_cost_approved_only_true_approves(monkeypatch, tmp_path, value):
    from traigent.core.cost_enforcement import is_cost_preapproved

    monkeypatch.setenv("TRAIGENT_COST_APPROVED", value)
    assert is_cost_preapproved(token_path=tmp_path / "none") is False


@pytest.mark.parametrize("name", ["TRAIGENT_DEV_MODE", "TRAIGENT_GENERATE_MOCKS"])
@pytest.mark.parametrize("value", ["true", "True", "1", "yes", "YES"])
def test_dev_toggles_prior_spellings_still_work(monkeypatch, name, value):
    from traigent.cloud.credential_manager import CredentialManager

    monkeypatch.delenv("TRAIGENT_DEV_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_GENERATE_MOCKS", raising=False)
    monkeypatch.setenv(name, value)
    assert CredentialManager._is_development_environment() is True


@pytest.mark.parametrize("name", ["TRAIGENT_DEV_MODE", "TRAIGENT_GENERATE_MOCKS"])
@pytest.mark.parametrize("value", ["on", " true", "true\n", "", "0", "off", "maybe"])
def test_dev_toggles_extra_spellings_stay_off(monkeypatch, name, value):
    from traigent.cloud.credential_manager import CredentialManager

    monkeypatch.delenv("TRAIGENT_DEV_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_GENERATE_MOCKS", raising=False)
    monkeypatch.setenv(name, value)
    assert CredentialManager._is_development_environment() is False


@pytest.mark.parametrize("value", ["1", "yes", "on", " true", "TRUE", "yes\n"])
def test_metrics_generate_mocks_only_true(monkeypatch, value):
    from traigent.utils.env_config import is_exact_flag

    monkeypatch.setenv("TRAIGENT_GENERATE_MOCKS", value)
    expected = value.lower() == "true"
    assert (
        is_exact_flag("TRAIGENT_GENERATE_MOCKS", frozenset({"true"}), lower=True)
        is expected
    )


def test_exact_flag_unset_is_false(monkeypatch):
    from traigent.utils.env_config import is_exact_flag

    monkeypatch.delenv("TRAIGENT_X_FLAG", raising=False)
    assert is_exact_flag("TRAIGENT_X_FLAG", frozenset({"1"})) is False


@pytest.mark.parametrize(
    "name", ["TRAIGENT_EDGE_ANALYTICS_MODE", "TRAIGENT_EXECUTION_MODE"]
)
@pytest.mark.parametrize("value", TRUE_SPELLINGS + ["local", " edge_analytics\n"])
def test_edge_analytics_privacy_flag_protective(monkeypatch, name, value):
    from traigent.utils.env_config import is_edge_analytics_env_enabled

    monkeypatch.delenv("TRAIGENT_EDGE_ANALYTICS_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_EXECUTION_MODE", raising=False)
    monkeypatch.setenv(name, value)
    assert is_edge_analytics_env_enabled() is True


@pytest.mark.parametrize("value", FALSE_SPELLINGS)
def test_edge_analytics_off_spellings(monkeypatch, value):
    from traigent.utils.env_config import is_edge_analytics_env_enabled

    monkeypatch.delenv("TRAIGENT_EDGE_ANALYTICS_MODE", raising=False)
    monkeypatch.setenv("TRAIGENT_EXECUTION_MODE", value)
    assert is_edge_analytics_env_enabled() is False


@pytest.mark.parametrize("value", ["false", "0", "off", ""])
def test_debug_flag_false_spellings_keep_clean_output(monkeypatch, value):
    from traigent.utils.exceptions import ConfigurationError, _traigent_excepthook

    monkeypatch.setenv("TRAIGENT_DEBUG", value)
    with pytest.raises(SystemExit):
        _traigent_excepthook(ConfigurationError, ConfigurationError("x"), None)
