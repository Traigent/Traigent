"""Characterization tests for ``ParameterDiscovery.find_completion_methods``.

These pin the observable behaviour (result ordering, path naming, depth limit,
instantiation gating, dummy-argument selection and error swallowing) so the
function can be restructured without changing what it returns or when it
constructs the inspected class.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from traigent.integrations.utils import discovery as discovery_module
from traigent.integrations.utils.discovery import ParameterDiscovery

MODULE_LOGGER = "traigent.integrations.utils.discovery"


def _flags(
    auto_discovery: bool = False,
    emit_security_telemetry: bool = False,
    profile: str = "standard",
) -> SimpleNamespace:
    return SimpleNamespace(
        auto_discovery=auto_discovery,
        emit_security_telemetry=emit_security_telemetry,
        profile=SimpleNamespace(value=profile),
    )


@pytest.fixture
def real_run(monkeypatch):
    """Pretend we are not running under pytest so only flags/opt-ins decide.

    pytest re-sets ``PYTEST_CURRENT_TEST`` at the start of the call phase, so
    hide it from the module under test instead of deleting it in setup.
    """
    real_getenv = discovery_module.os.getenv

    def fake_getenv(key, default=None):
        if key == "PYTEST_CURRENT_TEST":
            return None
        return real_getenv(key, default)

    monkeypatch.setattr(discovery_module.os, "getenv", fake_getenv)


def _names(result: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(m["name"], m["path"]) for m in result]


class Leaf:
    def create_item(self):
        return 1

    def helper(self):
        return 2

    def Chat_Stream(self):  # keyword match is case-insensitive
        return 3


class Mid:
    leaf = Leaf()
    value = 5  # non-callable, not a function: recursed into (no matches)

    def generate(self):
        return 1


class Holder:
    mid = Mid()
    _private = Leaf()

    def completion(self):
        return 1

    def other(self):
        return 2


def test_class_level_search_collects_in_dir_order_with_dotted_paths():
    result = ParameterDiscovery.find_completion_methods(Holder)

    assert _names(result) == [
        ("completion", "completion"),
        ("generate", "mid.generate"),
        ("Chat_Stream", "mid.leaf.Chat_Stream"),
        ("create_item", "mid.leaf.create_item"),
    ]
    assert all(m["parameters"] == {} for m in result)


def test_private_attributes_and_non_keyword_methods_are_ignored():
    result = ParameterDiscovery.find_completion_methods(Holder)

    paths = [m["path"] for m in result]
    assert not any("_private" in p for p in paths)
    assert "other" not in [m["name"] for m in result]
    assert "helper" not in [m["name"] for m in result]


class Deep:
    class _Hidden:
        pass

    def __init__(self, depth: int) -> None:
        pass


class _L3:
    def create_here(self):
        return 1


class _L2:
    l3 = _L3()


class _L1:
    l2 = _L2()


class _L0:
    l1 = _L1()


def test_search_depth_is_limited_to_three_levels():
    # _L0 (depth 3) -> l1 (2) -> l2 (1) -> l3 reached with depth 0: not searched.
    assert ParameterDiscovery.find_completion_methods(_L0) == []

    class Shallow:
        l2 = _L2()

    result = ParameterDiscovery.find_completion_methods(Shallow)
    assert _names(result) == [("create_here", "l2.l3.create_here")]


def test_classes_and_builtins_are_not_recursed_into():
    class WithNested:
        Inner = _L3  # class attribute: skipped for recursion
        builtin = len

    assert ParameterDiscovery.find_completion_methods(WithNested) == []


def test_attribute_access_error_skips_only_that_attribute(caplog):
    """A failing ``getattr`` is logged with its path; siblings are still searched."""

    class Exploding:
        @property
        def boom(self):
            raise RuntimeError("no access")

        def create_ok(self):
            return 1

    class Holder2:
        exploding = Exploding()

        def generate(self):
            return 1

    with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER):
        result = ParameterDiscovery.find_completion_methods(Holder2)

    assert _names(result) == [
        ("create_ok", "exploding.create_ok"),
        ("generate", "generate"),
    ]
    assert any(
        "Could not inspect attribute at exploding.boom: no access" in r.getMessage()
        for r in caplog.records
    )


def test_dir_failure_is_logged_and_returns_partial_results(caplog):
    class BadDir:
        def __dir__(self):
            raise RuntimeError("dir broke")

    with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER):
        result = ParameterDiscovery.find_completion_methods(BadDir())

    assert result == []
    assert any(
        "Error during nested method search: dir broke" in r.getMessage()
        for r in caplog.records
    )


class Built:
    """Instance attribute exposes a completion method only after __init__."""

    instances = 0
    kwargs: dict[str, Any] = {}

    def __init__(self) -> None:
        type(self).instances += 1
        self.client = Leaf()


def _reset(cls) -> None:
    cls.instances = 0
    cls.kwargs = {}


def test_instance_discovery_off_under_pytest_without_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(
        discovery_module, "get_security_flags", lambda: _flags(auto_discovery=True)
    )
    _reset(Built)

    result = ParameterDiscovery.find_completion_methods(Built)

    assert Built.instances == 0
    assert result == []


def test_explicit_opt_in_instantiates_even_under_pytest(monkeypatch):
    monkeypatch.setattr(
        discovery_module, "get_security_flags", lambda: _flags(auto_discovery=False)
    )

    class OptIn(Built):
        __traigent_allow_instance_discovery__ = True

    _reset(OptIn)
    result = ParameterDiscovery.find_completion_methods(OptIn)

    assert OptIn.instances == 1
    assert _names(result) == [
        ("Chat_Stream", "client.Chat_Stream"),
        ("create_item", "client.create_item"),
    ]


def test_instance_discovery_attribute_is_read_exactly_once(monkeypatch):
    """A flapping attribute (False, then True) must not flip the decision."""
    monkeypatch.setattr(
        discovery_module, "get_security_flags", lambda: _flags(auto_discovery=True)
    )
    reads: list[int] = []

    class FlappingMeta(type):
        @property
        def __traigent_allow_instance_discovery__(cls) -> bool:
            reads.append(1)
            return len(reads) > 1

    class Flapping(Built, metaclass=FlappingMeta):
        pass

    _reset(Flapping)
    result = ParameterDiscovery.find_completion_methods(Flapping)

    assert len(reads) == 1
    assert Flapping.instances == 0
    assert result == []


def test_explicit_opt_out_skips_instantiation_and_logs(monkeypatch, real_run, caplog):
    monkeypatch.setattr(
        discovery_module, "get_security_flags", lambda: _flags(auto_discovery=True)
    )

    class OptOut(Built):
        __traigent_allow_instance_discovery__ = False

    _reset(OptOut)
    with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER):
        result = ParameterDiscovery.find_completion_methods(OptOut)

    assert OptOut.instances == 0
    assert result == []
    assert any("explicit opt-out" in r.getMessage() for r in caplog.records)


def test_auto_discovery_flag_instantiates_outside_pytest(monkeypatch, real_run):
    monkeypatch.setattr(
        discovery_module, "get_security_flags", lambda: _flags(auto_discovery=True)
    )

    class Auto(Built):
        pass

    _reset(Auto)
    result = ParameterDiscovery.find_completion_methods(Auto)

    assert Auto.instances == 1
    assert [m["path"] for m in result] == ["client.Chat_Stream", "client.create_item"]


def test_no_flag_and_no_opt_in_means_no_instantiation(monkeypatch, real_run):
    monkeypatch.setattr(
        discovery_module, "get_security_flags", lambda: _flags(auto_discovery=False)
    )

    class Never(Built):
        pass

    _reset(Never)
    assert ParameterDiscovery.find_completion_methods(Never) == []
    assert Never.instances == 0


def test_telemetry_info_logged_only_for_flag_driven_instantiation(
    monkeypatch, real_run, caplog
):
    monkeypatch.setattr(
        discovery_module,
        "get_security_flags",
        lambda: _flags(auto_discovery=False, emit_security_telemetry=True),
    )

    class Opted(Built):
        __traigent_allow_instance_discovery__ = True

    _reset(Opted)
    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        ParameterDiscovery.find_completion_methods(Opted)
    # explicit setting present -> no telemetry message even though the flag is on
    assert not [r for r in caplog.records if "Auto discovery" in r.getMessage()]

    monkeypatch.setattr(
        discovery_module,
        "get_security_flags",
        lambda: _flags(auto_discovery=True, emit_security_telemetry=True, profile="p"),
    )

    class Flagged(Built):
        pass

    _reset(Flagged)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        ParameterDiscovery.find_completion_methods(Flagged)
    # auto_discovery true -> condition "not flags.auto_discovery" fails -> no log
    assert not [r for r in caplog.records if "Auto discovery" in r.getMessage()]


def test_telemetry_not_emitted_when_flag_off(monkeypatch, real_run, caplog):
    monkeypatch.setattr(
        discovery_module,
        "get_security_flags",
        lambda: _flags(auto_discovery=False, emit_security_telemetry=False),
    )

    class Opted(Built):
        __traigent_allow_instance_discovery__ = True

    _reset(Opted)
    with caplog.at_level(logging.INFO, logger=MODULE_LOGGER):
        ParameterDiscovery.find_completion_methods(Opted)
    assert not [r for r in caplog.records if "Auto discovery" in r.getMessage()]


class Required:
    seen: dict[str, Any] = {}

    def __init__(
        self,
        name: str,
        count: int,
        ratio: float,
        enabled: bool,
        blob,
        typed_other: list,
        opt_str: str = "x",
        opt_int=3,
    ) -> None:
        type(self).seen = {
            "name": name,
            "count": count,
            "ratio": ratio,
            "enabled": enabled,
            "blob": blob,
            "typed_other": typed_other,
            "opt_str": opt_str,
            "opt_int": opt_int,
        }
        self.sub = Leaf()


def test_string_annotations_do_not_infer_a_dummy_type(monkeypatch):
    """With postponed annotations the inferred "type" is a string -> None."""
    monkeypatch.setattr(discovery_module, "get_security_flags", lambda: _flags())

    class Opted(Required):
        __traigent_allow_instance_discovery__ = True

    Opted.seen = {}
    result = ParameterDiscovery.find_completion_methods(Opted)

    assert Opted.seen == {
        "name": None,
        "count": None,
        "ratio": None,
        "enabled": None,
        "blob": None,
        "typed_other": None,
        "opt_str": "x",
        "opt_int": 3,
    }
    assert _names(result) == [
        ("Chat_Stream", "sub.Chat_Stream"),
        ("create_item", "sub.create_item"),
    ]


def test_dummy_arguments_follow_real_type_annotations(monkeypatch):
    import inspect

    monkeypatch.setattr(discovery_module, "get_security_flags", lambda: _flags())
    params = [
        inspect.Parameter("name", inspect.Parameter.KEYWORD_ONLY, annotation=str),
        inspect.Parameter("count", inspect.Parameter.KEYWORD_ONLY, annotation=int),
        inspect.Parameter("ratio", inspect.Parameter.KEYWORD_ONLY, annotation=float),
        inspect.Parameter("flag", inspect.Parameter.KEYWORD_ONLY, annotation=bool),
        inspect.Parameter("other", inspect.Parameter.KEYWORD_ONLY, annotation=list),
        inspect.Parameter("bare", inspect.Parameter.KEYWORD_ONLY),
        inspect.Parameter(
            "opt", inspect.Parameter.KEYWORD_ONLY, annotation=int, default=9
        ),
    ]

    class Typed:
        __traigent_allow_instance_discovery__ = True
        seen: dict[str, Any] = {}

        def __init__(self, **kwargs: Any) -> None:
            type(self).seen = kwargs
            self.sub = Leaf()

    Typed.__init__.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD), *params]
    )

    result = ParameterDiscovery.find_completion_methods(Typed)

    assert Typed.seen == {
        "name": "dummy",
        "count": 0,
        "ratio": 0.0,
        "flag": False,
        "other": None,
        "bare": None,
    }
    assert [m["path"] for m in result] == ["sub.Chat_Stream", "sub.create_item"]


def test_all_optional_init_params_still_instantiates_with_no_arguments(
    monkeypatch,
):
    monkeypatch.setattr(discovery_module, "get_security_flags", lambda: _flags())

    class AllOptional:
        __traigent_allow_instance_discovery__ = True
        calls: list[dict[str, Any]] = []

        def __init__(self, a: int = 1, b: str = "z") -> None:
            type(self).calls.append({"a": a, "b": b})
            self.sub = Leaf()

    result = ParameterDiscovery.find_completion_methods(AllOptional)
    assert AllOptional.calls == [{"a": 1, "b": "z"}]
    assert [m["path"] for m in result] == ["sub.Chat_Stream", "sub.create_item"]


def test_instantiation_failure_is_swallowed_and_class_results_kept(monkeypatch, caplog):
    monkeypatch.setattr(discovery_module, "get_security_flags", lambda: _flags())

    class Failing:
        __traigent_allow_instance_discovery__ = True

        def create_cls(self):
            return 1

        def __init__(self) -> None:
            raise RuntimeError("cannot build")

    with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER):
        result = ParameterDiscovery.find_completion_methods(Failing)

    assert _names(result) == [("create_cls", "create_cls")]
    assert any(
        "Could not create instance for discovery: cannot build" in r.getMessage()
        for r in caplog.records
    )


def test_class_methods_found_before_instance_methods(monkeypatch):
    monkeypatch.setattr(discovery_module, "get_security_flags", lambda: _flags())

    class Both:
        __traigent_allow_instance_discovery__ = True

        def create_cls(self):
            return 1

        def __init__(self) -> None:
            self.sub = Leaf()

    result = ParameterDiscovery.find_completion_methods(Both)
    assert [m["path"] for m in result] == [
        "create_cls",
        "create_cls",
        "sub.Chat_Stream",
        "sub.create_item",
    ]


def test_instance_input_is_searched_without_instantiation(monkeypatch):
    monkeypatch.setattr(
        discovery_module, "get_security_flags", lambda: _flags(auto_discovery=True)
    )
    instance = Holder()

    result = ParameterDiscovery.find_completion_methods(instance)  # type: ignore[arg-type]

    assert [m["path"] for m in result] == [
        "completion",
        "mid.generate",
        "mid.leaf.Chat_Stream",
        "mid.leaf.create_item",
    ]


def test_flag_lookup_failure_is_swallowed(monkeypatch, caplog):
    def boom():
        raise RuntimeError("flags unavailable")

    monkeypatch.setattr(discovery_module, "get_security_flags", boom)
    with caplog.at_level(logging.DEBUG, logger=MODULE_LOGGER):
        result = ParameterDiscovery.find_completion_methods(Holder)

    # Class-level results gathered before the flag lookup survive.
    assert _names(result)[0] == ("completion", "completion")
    assert any(
        "Could not find completion methods" in r.getMessage() for r in caplog.records
    )
