"""Tests for the CONTEXT-mode parameter-shadowing warning (issue #1372).

When a function already declares the tuned knobs as parameters and is wrapped
with the default ``injection_mode=InjectionMode.CONTEXT`` (which does NOT
override function parameters), every trial silently runs with the signature
defaults. The decorator must emit a loud warning instead of failing silently.
"""

import warnings

import pytest

from traigent.api.decorators import optimize


def test_context_mode_param_shadowing_warns():
    """A knob declared as a function param under CONTEXT mode warns loudly."""
    with pytest.warns(UserWarning, match="injection_mode is CONTEXT"):

        @optimize(
            configuration_space={"model": ["a", "b"], "temperature": [0.1, 0.9]},
            algorithm="grid",
        )
        def generate(question, *, model="default", temperature=0.0):
            return f"{model}:{temperature}"

    # The warning names the actually-shadowed knobs.
    with pytest.warns(UserWarning) as record:

        @optimize(
            configuration_space={"model": ["a", "b"], "temperature": [0.1, 0.9]},
            algorithm="grid",
        )
        def generate2(question, *, model="default", temperature=0.0):
            return f"{model}:{temperature}"

    msg = str(record[0].message)
    assert "model" in msg and "temperature" in msg


def test_no_warn_when_body_reads_get_config():
    """A body that reads traigent.get_config() honors the knobs — no warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any matching warning would raise

        @optimize(
            configuration_space={"model": ["a", "b"]},
            algorithm="grid",
        )
        def generate(question, *, model="default"):
            import traigent

            cfg = traigent.get_config()
            return cfg.get("model", model)


def test_no_warn_when_no_param_overlap():
    """Knobs that are NOT function params (pure CONTEXT pattern) do not warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        @optimize(
            configuration_space={"model": ["a", "b"]},
            algorithm="grid",
        )
        def generate(question):
            return question


def test_no_warn_in_parameter_mode():
    """PARAMETER mode injects explicitly, so an overlap is expected and silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        @optimize(
            configuration_space={"model": ["a", "b"]},
            injection_mode="parameter",
            config_param="config",
            algorithm="grid",
        )
        def generate(question, model="default", config=None):
            return model
