"""#1894: traigent-tracing add-ons older than 0.1.1 are ignored."""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
import types

import pytest

from traigent.core import tracing


def _fake_plugin(version):
    fake = types.ModuleType("traigent_tracing")
    if version is not None:
        fake.__version__ = version
    for name, val in {
        "OTEL_AVAILABLE": True,
        "TRACING_ENABLED": True,
        "SecureIdGenerator": object,
        "clear_test_context": lambda: None,
        "get_test_context": dict,
        "set_test_context": lambda *a, **k: None,
        "example_evaluation_span": lambda *a, **k: None,
        "optimization_session_span": lambda *a, **k: None,
        "trial_span": lambda *a, **k: None,
        "record_example_result": lambda *a, **k: None,
        "record_optimization_complete": lambda *a, **k: None,
        "record_trial_result": lambda *a, **k: None,
        "get_tracer": lambda: "plugin-tracer",
    }.items():
        setattr(fake, name, val)
    return fake


@pytest.fixture
def reload_with(monkeypatch):
    def _do(version, *, metadata=None):
        def _md(name):
            if metadata is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return metadata

        monkeypatch.setattr(importlib.metadata, "version", _md)
        monkeypatch.setitem(sys.modules, "traigent_tracing", _fake_plugin(version))
        return importlib.reload(tracing)

    yield _do
    monkeypatch.delitem(sys.modules, "traigent_tracing", raising=False)
    importlib.reload(tracing)


@pytest.mark.parametrize("version", ["0.1.0", "0.0.9", "garbage", "0.1.1rc1", None])
def test_old_or_unparseable_plugin_uses_embedded(reload_with, caplog, version):
    with caplog.at_level(logging.WARNING, logger="traigent.core.tracing"):
        mod = reload_with(version)
    assert mod._PLUGIN_AVAILABLE is False
    assert mod.get_tracer is not sys.modules["traigent_tracing"].get_tracer
    msgs = [
        r.getMessage() for r in caplog.records if "traigent-tracing" in r.getMessage()
    ]
    assert len(msgs) == 1
    assert "0.1.0" not in msgs[0].replace("0.1.1", "")


def test_metadata_version_preferred_over_module_attr(reload_with):
    assert reload_with("9.9.9", metadata="0.1.0")._PLUGIN_AVAILABLE is False
    assert reload_with("0.0.1", metadata="0.1.1")._PLUGIN_AVAILABLE is True


@pytest.mark.parametrize("version", ["0.1.1", "0.2.0", "1.0"])
def test_accepted_plugin_is_used_and_gated(reload_with, caplog, monkeypatch, version):
    with caplog.at_level(logging.WARNING, logger="traigent.core.tracing"):
        mod = reload_with(version)
    assert mod._PLUGIN_AVAILABLE is True
    monkeypatch.setattr(mod, "_is_offline", lambda: False)
    assert mod.get_tracer() == "plugin-tracer"
    assert hasattr(mod.get_tracer, "__wrapped__")  # offline gate kept (#2240)
    assert not [r for r in caplog.records if "traigent-tracing" in r.getMessage()]
