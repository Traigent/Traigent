"""#3162: traigent.api.config_builder shares the one global config store."""

from __future__ import annotations

import pytest

from traigent.api import config_builder as cb
from traigent.api import functions as fn


@pytest.fixture(autouse=True)
def _restore_global_config():
    saved = dict(fn._GLOBAL_CONFIG)
    yield
    fn._GLOBAL_CONFIG.clear()
    fn._GLOBAL_CONFIG.update(saved)


def test_single_store_object():
    assert cb._GLOBAL_CONFIG is fn._GLOBAL_CONFIG


def test_configure_is_visible_through_config_builder():
    fn.configure(parallel_workers=7)
    assert cb.get_global_config()["parallel_workers"] == 7


def test_config_builder_update_is_visible_through_functions():
    cb.update_global_config(execution_mode="cloud")
    assert fn.get_global_config()["execution_mode"] == "cloud"
    assert fn._GLOBAL_CONFIG["execution_mode"] == "cloud"


def test_configuration_builder_default_reads_shared_store():
    fn._GLOBAL_CONFIG["execution_mode"] = "edge_analytics"
    assert cb.ConfigurationBuilder().get_global_config()["execution_mode"] == (
        "edge_analytics"
    )


def test_clear_global_config_keeps_functions_usable():
    cb.update_global_config(execution_mode="cloud")
    cb.clear_global_config()
    assert "execution_mode" not in fn.get_global_config()
    # configure() still works: required default keys were restored
    fn.configure(api_keys={"openai": "k"})
    assert fn.get_global_config()["parallel_workers"] == 1


def test_clear_global_config_does_not_drop_configured_api_keys():
    fn._GLOBAL_CONFIG["api_keys"]["openai"] = "k"
    cb.clear_global_config()
    assert fn.get_global_config()["api_keys"] == {"openai": "k"}


def test_private_builder_config_stays_private():
    private: dict = {}
    builder = cb.ConfigurationBuilder(global_config=private)
    builder.update_global_config(execution_mode="cloud")
    assert "execution_mode" not in fn.get_global_config()
    builder.clear_global_config()
    assert private == {}
