import pytest

from traigent.config.types import TraigentConfig
from traigent.integrations.llms.cohere_plugin import CoherePlugin


@pytest.mark.unit
def test_cohere_plugin_mappings():
    plugin = CoherePlugin()
    mappings = plugin.get_parameter_mappings()
    assert mappings["model"] == "model"
    assert mappings["top_p"] == "p"
    assert mappings["system"] == "preamble"


@pytest.mark.unit
def test_cohere_plugin_overrides():
    plugin = CoherePlugin()
    config = TraigentConfig(
        model="command-r",
        temperature=0.7,
        top_p=0.9,
        custom_params={"system": "You are a helpful assistant"},
    )

    kwargs = {"message": "Hello"}
    overridden = plugin.apply_overrides(kwargs, config)

    assert overridden["model"] == "command-r"
    assert overridden["temperature"] == 0.7
    assert overridden["p"] == 0.9
    assert overridden["preamble"] == "You are a helpful assistant"
