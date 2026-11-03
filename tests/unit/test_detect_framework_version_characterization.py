"""Golden characterization for FrameworkDiscovery.detect_framework_version."""

from __future__ import annotations

import sys
import types

from traigent.integrations.utils import discovery as disc


def _fn():
    for name in dir(disc):
        obj = getattr(disc, name)
        if isinstance(obj, type) and hasattr(obj, "detect_framework_version"):
            return obj.detect_framework_version
    raise AssertionError("detect_framework_version not found")


detect = _fn()


def test_class_attrs_in_priority_order():
    class A:
        __version__ = "1"
        _version = "2"

    class B:
        _version = "2"
        VERSION = "3"

    class C:
        VERSION = 3
        version = "4"

    assert detect(A) == "1"
    assert detect(B) == "2"
    assert detect(C) == "4"


def test_non_string_class_attrs_fall_through_to_module():
    mod = types.ModuleType("fake_fw_mod_char")
    mod.VERSION = "9.9"
    mod._version = 5
    sys.modules[mod.__name__] = mod
    try:

        class K:
            __version__ = 1
            __module__ = mod.__name__

        assert detect(K) == "9.9"
    finally:
        del sys.modules[mod.__name__]


def test_module_attr_priority():
    mod = types.ModuleType("fake_fw_mod_char2")
    mod.__version__ = 7
    mod._version = "2.0"
    mod.version = "3.0"
    sys.modules[mod.__name__] = mod
    try:

        class K:
            __module__ = mod.__name__

        assert detect(K) == "2.0"
    finally:
        del sys.modules[mod.__name__]


def test_none_when_nothing_found():
    class K:
        __module__ = "no_such_module_char_xyz"

    assert detect(K) is None
    assert detect(int) is None


def test_exception_returns_none():
    class Boom:
        def __getattribute__(self, name):
            raise RuntimeError("x")

    class Meta(type):
        def __getattribute__(cls, name):
            if name in ("__version__", "_version", "VERSION", "version"):
                raise RuntimeError("x")
            return super().__getattribute__(name)

    class K(metaclass=Meta):
        pass

    assert detect(K) is None
