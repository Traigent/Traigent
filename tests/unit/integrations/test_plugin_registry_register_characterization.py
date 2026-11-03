"""Characterization tests for PluginRegistry.register.

They pin the duplicate/priority decisions, the class and package mapping
updates, the order and count of reads on the plugin and on the registry maps,
and the state left behind when the plugin misbehaves part-way through.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import patch

import pytest

import traigent.integrations.plugin_registry as registry_module
from traigent.integrations.base_plugin import (
    IntegrationPlugin,
    IntegrationPriority,
    PluginMetadata,
)
from traigent.integrations.plugin_registry import PluginRegistry
from traigent.utils.exceptions import TraigentError


def _register(target: PluginRegistry, plugin: object) -> Any:
    """Call ``register`` with a duck-typed plugin and return its result."""
    return target.register(cast(IntegrationPlugin, plugin))


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[PluginRegistry]:
    """Fresh registry without built-in discovery; singleton restored after."""
    monkeypatch.setattr(PluginRegistry, "_discover_builtin_plugins", lambda self: None)
    monkeypatch.setattr(PluginRegistry, "_instance", None)
    instance = PluginRegistry()
    yield instance
    monkeypatch.setattr(PluginRegistry, "_instance", None)


class _Priority:
    """Priority stand-in that records every read of ``value``."""

    def __init__(self, value: int, events: list[Any], label: str) -> None:
        self._value = value
        self._events = events
        self._label = label

    @property
    def value(self) -> int:
        self._events.append(("priority", self._label))
        return self._value


class _Plugin:
    """Plugin double exposing metadata and target classes with read tracing."""

    def __init__(
        self,
        name: str,
        classes: Any,
        *,
        priority: int = 1,
        packages: list[str] | None = None,
        events: list[Any] | None = None,
    ) -> None:
        self.events: list[Any] = events if events is not None else []
        self._classes = classes
        self._metadata = PluginMetadata(
            name=name,
            version="1.0",
            supported_packages=packages if packages is not None else ["pkg"],
            priority=cast(Any, _Priority(priority, self.events, name)),
        )

    @property
    def metadata(self) -> PluginMetadata:
        self.events.append(("metadata", self._metadata.name))
        return self._metadata

    def get_target_classes(self) -> Any:
        self.events.append(("classes", self._metadata.name))
        if isinstance(self._classes, BaseException):
            raise self._classes
        return self._classes


def test_first_registration_stores_plugin_and_both_mappings(
    registry: PluginRegistry,
) -> None:
    plugin = _Plugin("alpha", ["mod.A", "mod.B"], packages=["p1", "p2"])
    assert _register(registry, plugin) is None
    assert registry._plugins == {"alpha": plugin}
    assert registry._class_to_plugin == {"mod.A": "alpha", "mod.B": "alpha"}
    assert registry._package_to_plugin == {"p1": ["alpha"], "p2": ["alpha"]}


def test_duplicate_with_equal_priority_is_refused_and_state_untouched(
    registry: PluginRegistry,
) -> None:
    first = _Plugin("alpha", ["mod.A"], priority=1)
    _register(registry, first)
    with pytest.raises(TraigentError) as excinfo:
        _register(registry, _Plugin("alpha", ["mod.Z"], priority=1))
    assert str(excinfo.value) == "Plugin 'alpha' is already registered"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True
    assert registry._plugins == {"alpha": first}
    assert registry._class_to_plugin == {"mod.A": "alpha"}


def test_duplicate_with_lower_priority_is_refused(registry: PluginRegistry) -> None:
    _register(registry, _Plugin("alpha", ["mod.A"], priority=2))
    with pytest.raises(TraigentError):
        _register(registry, _Plugin("alpha", ["mod.A"], priority=1))


def test_duplicate_with_higher_priority_replaces_and_logs(
    registry: PluginRegistry,
) -> None:
    _register(registry, _Plugin("alpha", ["mod.A"], priority=1))
    newer = _Plugin("alpha", ["mod.A", "mod.N"], priority=2, packages=["q"])
    with patch.object(registry_module, "logger") as log:
        _register(registry, newer)
    assert log.method_calls[0][0] == "info"
    assert [c.args for c in log.info.call_args_list] == [
        ("Replacing plugin alpha with higher priority version",)
    ]
    assert registry._plugins == {"alpha": newer}
    assert registry._class_to_plugin == {"mod.A": "alpha", "mod.N": "alpha"}
    assert registry._package_to_plugin == {"pkg": ["alpha"], "q": ["alpha"]}


def test_class_claimed_by_other_plugin_moves_only_on_strictly_higher_priority(
    registry: PluginRegistry,
) -> None:
    _register(registry, _Plugin("low", ["c.lower", "c.equal", "c.higher"], priority=1))
    _register(registry, _Plugin("mid", ["c.lower"], priority=2))
    _register(registry, _Plugin("mid2", ["c.higher"], priority=2))
    assert registry._class_to_plugin == {
        "c.lower": "mid",
        "c.equal": "low",
        "c.higher": "mid2",
    }
    _register(registry, _Plugin("same", ["c.lower", "c.equal"], priority=2))
    assert registry._class_to_plugin == {
        "c.lower": "mid",
        "c.equal": "same",
        "c.higher": "mid2",
    }
    _register(registry, _Plugin("weak", ["c.lower"], priority=0))
    assert registry._class_to_plugin["c.lower"] == "mid"


def test_repeated_class_in_one_plugin_keeps_first_mapping_name(
    registry: PluginRegistry,
) -> None:
    plugin = _Plugin("alpha", ["mod.A", "mod.A"])
    _register(registry, plugin)
    assert registry._class_to_plugin == {"mod.A": "alpha"}


def test_dangling_class_mapping_raises_key_error_after_partial_update(
    registry: PluginRegistry,
) -> None:
    registry._class_to_plugin["mod.B"] = "ghost"
    plugin = _Plugin("alpha", ["mod.A", "mod.B", "mod.C"])
    with pytest.raises(KeyError) as excinfo:
        _register(registry, plugin)
    assert excinfo.value.args == ("ghost",)
    assert registry._plugins == {"alpha": plugin}
    assert registry._class_to_plugin == {"mod.B": "ghost", "mod.A": "alpha"}
    assert registry._package_to_plugin == {}


def test_class_map_and_plugin_map_operations_are_exact(
    registry: PluginRegistry,
) -> None:
    ops: list[Any] = []

    class _TracedDict(dict[str, Any]):
        def __init__(self, label: str) -> None:
            super().__init__()
            self._label = label

        def __contains__(self, key: object) -> bool:
            ops.append((self._label, "contains", key))
            return super().__contains__(key)

        def __getitem__(self, key: str) -> Any:
            ops.append((self._label, "getitem", key))
            return super().__getitem__(key)

        def __setitem__(self, key: str, value: Any) -> None:
            ops.append((self._label, "setitem", key))
            super().__setitem__(key, value)

    registry._plugins = cast(Any, _TracedDict("plugins"))
    registry._class_to_plugin = _TracedDict("classes")
    _register(registry, _Plugin("base", ["x.one"], priority=1))
    del ops[:]
    _register(registry, _Plugin("top", ["x.one", "x.two"], priority=3))
    assert ops == [
        ("plugins", "contains", "top"),
        ("plugins", "setitem", "top"),
        ("classes", "contains", "x.one"),
        ("classes", "getitem", "x.one"),
        ("plugins", "getitem", "base"),
        ("classes", "setitem", "x.one"),
        ("classes", "contains", "x.two"),
        ("classes", "setitem", "x.two"),
    ]


def test_plugin_read_trace_without_conflicts(registry: PluginRegistry) -> None:
    plugin = _Plugin("alpha", ["mod.A", "mod.B"], packages=["p1"])
    _register(registry, plugin)
    assert plugin.events == [
        ("metadata", "alpha"),
        ("classes", "alpha"),
        ("metadata", "alpha"),
    ]


def test_plugin_read_trace_with_conflicts_reads_priority_pairs(
    registry: PluginRegistry,
) -> None:
    events: list[Any] = []
    first = _Plugin("one", ["c.A", "c.B"], priority=1, events=events)
    _register(registry, first)
    del events[:]
    second = _Plugin("two", ["c.A", "c.B", "c.C"], priority=5, events=events)
    _register(registry, second)
    assert events == [
        ("metadata", "two"),
        ("classes", "two"),
        ("metadata", "two"),
        ("priority", "two"),
        ("metadata", "one"),
        ("priority", "one"),
        ("metadata", "two"),
        ("priority", "two"),
        ("metadata", "one"),
        ("priority", "one"),
        ("metadata", "two"),
    ]


def test_duplicate_check_reads_existing_then_new_priority(
    registry: PluginRegistry,
) -> None:
    events: list[Any] = []
    _register(registry, _Plugin("dup", [], priority=1, events=events))
    del events[:]
    _register(registry, _Plugin("dup", [], priority=2, events=events))
    assert events[:5] == [
        ("metadata", "dup"),
        ("metadata", "dup"),
        ("priority", "dup"),
        ("metadata", "dup"),
        ("priority", "dup"),
    ]


def test_get_target_classes_raising_leaves_plugin_stored_without_mappings(
    registry: PluginRegistry,
) -> None:
    plugin = _Plugin("alpha", RuntimeError("no classes"))
    with pytest.raises(RuntimeError, match="no classes"):
        _register(registry, plugin)
    assert registry._plugins == {"alpha": plugin}
    assert registry._class_to_plugin == {}
    assert registry._package_to_plugin == {}


def test_non_iterable_classes_leave_plugin_stored(registry: PluginRegistry) -> None:
    plugin = _Plugin("alpha", 5)
    with pytest.raises(TypeError):
        _register(registry, plugin)
    assert registry._plugins == {"alpha": plugin}
    assert registry._class_to_plugin == {}


def test_class_iterator_failing_midway_keeps_earlier_mappings(
    registry: PluginRegistry,
) -> None:
    def classes() -> Iterator[str]:
        yield "mod.A"
        raise RuntimeError("stopped")

    plugin = _Plugin("alpha", classes())
    with pytest.raises(RuntimeError, match="stopped"):
        _register(registry, plugin)
    assert registry._class_to_plugin == {"mod.A": "alpha"}
    assert registry._package_to_plugin == {}


def test_unhashable_class_name_raises_type_error(registry: PluginRegistry) -> None:
    plugin = _Plugin("alpha", [["not", "hashable"]])
    with pytest.raises(TypeError):
        _register(registry, plugin)
    assert registry._plugins == {"alpha": plugin}
    assert registry._class_to_plugin == {}


def test_class_iterable_is_consumed_once(registry: PluginRegistry) -> None:
    consumed = iter(["mod.A", "mod.B"])
    plugin = _Plugin("alpha", consumed)
    _register(registry, plugin)
    assert registry._class_to_plugin == {"mod.A": "alpha", "mod.B": "alpha"}
    assert list(consumed) == []


def test_package_mapping_shares_names_across_plugins_without_duplicates(
    registry: PluginRegistry,
) -> None:
    _register(registry, _Plugin("alpha", [], packages=["shared", "only-a"]))
    _register(registry, _Plugin("beta", [], packages=["shared", "shared"]))
    assert registry._package_to_plugin == {
        "shared": ["alpha", "beta"],
        "only-a": ["alpha"],
    }


def test_invalid_metadata_is_rejected_before_any_state_change(
    registry: PluginRegistry,
) -> None:
    class _NoMetadata:
        metadata = None

        def get_target_classes(self) -> list[str]:
            raise AssertionError("must not be reached")

    with pytest.raises(TraigentError, match="must define metadata"):
        _register(registry, _NoMetadata())
    assert registry._plugins == {}
    assert registry._class_to_plugin == {}


def test_metadata_name_is_normalized_before_use(registry: PluginRegistry) -> None:
    plugin = _Plugin("  padded  ", ["mod.A"])
    _register(registry, plugin)
    assert registry._plugins == {"padded": plugin}
    assert registry._class_to_plugin == {"mod.A": "padded"}


def test_priority_enum_values_compare_through_value(registry: PluginRegistry) -> None:
    class _Enumerated(_Plugin):
        def __init__(self, name: str, priority: IntegrationPriority) -> None:
            super().__init__(name, ["mod.A"])
            self._metadata.priority = priority

    _register(registry, _Enumerated("normal", IntegrationPriority.NORMAL))
    _register(registry, _Enumerated("high", IntegrationPriority.HIGH))
    assert registry._class_to_plugin == {"mod.A": "high"}
