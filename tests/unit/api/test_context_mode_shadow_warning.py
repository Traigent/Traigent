"""Tests for the CONTEXT-mode parameter-shadowing warning (issue #1372).

When a function already declares the tuned knobs as parameters and is wrapped
with the default ``injection_mode=InjectionMode.CONTEXT`` (which does NOT
override function parameters), every trial silently runs with the signature
defaults. The decorator must emit a loud warning instead of failing silently.
"""

import warnings

import pytest

from traigent.api.decorators import _warn_context_mode_param_shadowing, optimize


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


def test_no_warn_when_no_param_overlap_and_reads_get_config():
    """No param overlap but the body reads get_config() — correct CONTEXT pattern."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        @optimize(
            configuration_space={"model": ["a", "b"]},
            algorithm="grid",
        )
        def generate(question):
            import traigent

            cfg = traigent.get_config()
            return cfg.get("model", "default")


def test_phantom_best_config_warns_when_no_overlap_and_no_get_config():
    """Phantom best_config regression (T6): a knob that is neither a parameter nor
    read via get_config() under CONTEXT mode must warn — previously silent.

    The knob ``temperature`` is hardcoded in the body, never a parameter, and the
    function never reads get_config(). Every trial runs identically, yet the
    optimizer would report a confident best_config. This is the naive default case
    that used to slip through the param-overlap-only guard.
    """
    with pytest.warns(UserWarning, match="phantom best_config"):

        @optimize(
            configuration_space={"temperature": [0.1, 0.9]},
            algorithm="grid",
        )
        def generate(question):
            temperature = 0.0  # hardcoded; the sweep can never touch it
            return f"{question}:{temperature}"


def test_phantom_warning_names_the_knobs():
    """The phantom warning lists the config-space knob(s) that can never vary."""
    with pytest.warns(UserWarning) as record:

        @optimize(
            configuration_space={"temperature": [0.1, 0.9], "top_p": [0.5, 1.0]},
            algorithm="grid",
        )
        def generate(question):
            return question

    msg = str(record[0].message)
    assert "temperature" in msg and "top_p" in msg


def test_graceful_degradation_when_source_unavailable():
    """An unintrospectable callable (no retrievable source, e.g. exec-compiled /
    C-extension / notebook) must not crash and must not emit the phantom warning —
    the phantom check degrades to a silent skip when it cannot read the body."""
    ns: dict = {}
    exec("def gen(question):\n    return question", ns)
    fn = ns["gen"]  # inspect.getsource -> OSError (no source file)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise
        _warn_context_mode_param_shadowing(
            fn, {"temperature": [0.1, 0.9]}, "context", None
        )


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
