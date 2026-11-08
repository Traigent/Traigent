"""#2940: unmapped or non-class framework targets must not count as applied."""

import logging
import sys
import types

import pytest

from traigent.integrations.framework_override import FrameworkOverrideManager


@pytest.fixture
def fake_module(monkeypatch):
    mod = types.ModuleType("fake_fw_2940")

    class Unmapped:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def a_function(**kwargs):
        return kwargs

    mod.Unmapped = Unmapped
    mod.a_function = a_function
    monkeypatch.setitem(sys.modules, "fake_fw_2940", mod)
    return mod


def test_unmapped_class_not_applied(fake_module, caplog):
    mgr = FrameworkOverrideManager()
    original = fake_module.Unmapped.__init__
    with caplog.at_level(logging.WARNING):
        mgr.activate_overrides(["fake_fw_2940.Unmapped"])
    assert "fake_fw_2940.Unmapped" not in mgr.applied_targets
    assert fake_module.Unmapped.__init__ is original
    assert "no parameter or method mapping" in caplog.text
    assert "Supported targets" in caplog.text


def test_function_target_not_applied(fake_module, caplog):
    mgr = FrameworkOverrideManager()
    mgr.register_framework_target("fake_fw_2940.a_function", {"model": "model"})
    with caplog.at_level(logging.WARNING):
        mgr.activate_overrides(["fake_fw_2940.a_function"])
    assert "fake_fw_2940.a_function" not in mgr.applied_targets
    assert "_traigent_original_init" not in vars(fake_module.a_function)
    assert "not a class" in caplog.text


def test_mapped_class_still_applied(fake_module):
    mgr = FrameworkOverrideManager()
    mgr.register_framework_target("fake_fw_2940.Unmapped", {"model": "model"})
    mgr.activate_overrides(["fake_fw_2940.Unmapped"])
    assert "fake_fw_2940.Unmapped" in mgr.applied_targets
