"""Tests for the traigent.traigent_integration compatibility shim."""

from __future__ import annotations

import importlib
import warnings


def test_compat_shim_exports_old_and_new_client_names():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        module = importlib.import_module("traigent.traigent_integration")

    assert module.OptiGenClient is module.TraigentClient
    assert module.__all__ == ["OptiGenClient", "TraigentClient"]


def test_compat_shim_warning_points_to_traigent_client():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.reload(importlib.import_module("traigent.traigent_integration"))

    assert caught
    assert "compatibility shim" in str(caught[-1].message)
    assert "traigent.traigent_client" in str(caught[-1].message)
