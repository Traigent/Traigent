"""One boolean grammar for TRAIGENT_* flags (#2989).

Spellings are stripped and case-folded; unrecognised values resolve to the
protective value (approval/relaxing flags off, privacy flags on only for
recognised spellings).
"""

from __future__ import annotations

import pytest

TRUE_SPELLINGS = ["true", " true", "True", "1", "yes", "on", "yes\n", "ON "]
FALSE_SPELLINGS = ["", "0", "False", "off", "no", "maybe", "treu"]


@pytest.mark.parametrize("value", TRUE_SPELLINGS)
def test_run_approved_accepts_canonical_spellings(monkeypatch, value):
    from traigent.core.ci_approval import _check_env_var_approval

    monkeypatch.setenv("TRAIGENT_RUN_APPROVED", value)
    assert _check_env_var_approval() is True


@pytest.mark.parametrize("value", FALSE_SPELLINGS)
def test_run_approved_unrecognised_is_not_approved(monkeypatch, value):
    from traigent.core.ci_approval import _check_env_var_approval

    monkeypatch.setenv("TRAIGENT_RUN_APPROVED", value)
    assert _check_env_var_approval() is False


@pytest.mark.parametrize("value", ["true", " true", "True", "TRUE\n"])
def test_cost_approved_true_spellings(monkeypatch, tmp_path, value):
    from traigent.core.cost_enforcement import is_cost_preapproved

    monkeypatch.setenv("TRAIGENT_COST_APPROVED", value)
    assert is_cost_preapproved(token_path=tmp_path / "none") is True


@pytest.mark.parametrize("value", ["1", "yes", "on", "", "0", "off", "maybe", "treu"])
def test_cost_approved_only_true_approves(monkeypatch, tmp_path, value):
    from traigent.core.cost_enforcement import is_cost_preapproved

    monkeypatch.setenv("TRAIGENT_COST_APPROVED", value)
    assert is_cost_preapproved(token_path=tmp_path / "none") is False


@pytest.mark.parametrize("name", ["TRAIGENT_DEV_MODE", "TRAIGENT_GENERATE_MOCKS"])
@pytest.mark.parametrize("value", TRUE_SPELLINGS)
def test_dev_toggles_share_grammar(monkeypatch, name, value):
    from traigent.cloud.credential_manager import CredentialManager

    monkeypatch.delenv("TRAIGENT_DEV_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_GENERATE_MOCKS", raising=False)
    monkeypatch.setenv(name, value)
    assert CredentialManager._is_development_environment() is True


@pytest.mark.parametrize("name", ["TRAIGENT_DEV_MODE", "TRAIGENT_GENERATE_MOCKS"])
@pytest.mark.parametrize("value", FALSE_SPELLINGS)
def test_dev_toggles_unrecognised_stay_off(monkeypatch, name, value):
    from traigent.cloud.credential_manager import CredentialManager

    monkeypatch.delenv("TRAIGENT_DEV_MODE", raising=False)
    monkeypatch.delenv("TRAIGENT_GENERATE_MOCKS", raising=False)
    monkeypatch.setenv(name, value)
    assert CredentialManager._is_development_environment() is False


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
