"""Tests that public ``traigent.api.*`` submodules declare ``__all__`` (issue #1446).

Without ``__all__`` every non-underscore top-level name — including internal
helpers and resolution-only classes — leaks via ``import *``, IDE autocomplete,
and autodoc. These tests pin the curated export surface.
"""

import importlib

import pytest

_SUBMODULES = [
    "agent_inference",
    "config_builder",
    "constraint_builders",
    "decorators",
    "functions",
    "parameter_validator",
    "strategy_presets",
    "types",
]


@pytest.mark.parametrize("name", _SUBMODULES)
def test_submodule_declares_all(name):
    mod = importlib.import_module(f"traigent.api.{name}")
    assert hasattr(mod, "__all__"), f"{name} is missing __all__"
    # Every declared name must actually exist on the module.
    missing = [n for n in mod.__all__ if not hasattr(mod, n)]
    assert not missing, f"{name}.__all__ references missing names: {missing}"
    # No private names and no module-level logger noise leak into the surface.
    assert "logger" not in mod.__all__
    assert all(not n.startswith("_") for n in mod.__all__)


def test_decorators_does_not_export_internals():
    """The internals named in #1446 must no longer leak via import *."""
    ns: dict = {}
    exec("from traigent.api.decorators import *", ns)  # noqa: S102
    for internal in (
        "get_optimize_default",
        "LegacyOptimizeArgs",
        "ResolvedExecutionOptions",
    ):
        assert internal not in ns, f"{internal} still leaks via import *"
    # Public API is still exported.
    assert "optimize" in ns
    assert "EvaluationOptions" in ns
