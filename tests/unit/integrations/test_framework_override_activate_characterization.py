"""Characterization tests for FrameworkOverrideManager.activate_overrides."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from traigent.integrations import framework_override as fo
from traigent.integrations.framework_override import FrameworkOverrideManager
from traigent.utils.exceptions import ConfigurationError

MOD = "fake_activate_fw_mod"


class _Client:
    def __init__(self, *args, **kwargs):
        self.args = args


@pytest.fixture
def manager():
    mgr = FrameworkOverrideManager()
    yield mgr
    mgr._override_active.enabled = False


@pytest.fixture
def fake_module():
    class Client(_Client):
        pass

    module = types.ModuleType(MOD)
    module.Client = Client
    sys.modules[MOD] = module
    yield module
    sys.modules.pop(MOD, None)


@pytest.fixture
def log():
    with patch.object(fo, "logger") as lg:
        yield lg


def _warnings(lg):
    return [c.args[0] for c in lg.warning.call_args_list]


def _infos(lg):
    return [c.args[0] for c in lg.info.call_args_list]


def test_success_patches_and_registers(manager, fake_module, log):
    target = f"{MOD}.Client"
    manager._method_mappings[target] = {"gen": ["temperature"], "other": ["x"]}
    original_init = fake_module.Client.__init__
    with patch.object(manager, "_apply_method_override") as apply:
        manager.activate_overrides([target])
    assert manager._override_active.enabled is True
    assert fake_module.Client._traigent_original_init is original_init
    assert fake_module.Client.__init__ is not original_init
    assert [c.args[1:] for c in apply.call_args_list] == [
        (target, "gen"),
        (target, "other"),
    ]
    assert manager.applied_targets == [target]
    assert _warnings(log) == []


def test_existing_original_init_is_kept(manager, fake_module, log):
    sentinel = object()
    fake_module.Client._traigent_original_init = sentinel
    with patch.object(
        manager, "_create_override_constructor", return_value=lambda *a, **k: None
    ) as create:
        manager.activate_overrides([f"{MOD}.Client"])
    assert create.call_args.args[0] is sentinel
    assert fake_module.Client._traigent_original_init is sentinel


def test_already_registered_target_is_skipped(manager, fake_module, log):
    target = f"{MOD}.Client"
    manager.register_active_override(target, object)
    original_init = fake_module.Client.__init__
    manager.activate_overrides([target])
    assert fake_module.Client.__init__ is original_init
    assert manager._override_active.enabled is True
    assert _warnings(log) == []


def test_builtin_mock_target_is_noop(manager, log):
    manager.activate_overrides(["MockOpenAI", "MockLangChainOpenAI"])
    assert manager._override_active.enabled is True
    assert manager.applied_targets == []
    assert _warnings(log) == []


def test_invalid_target_raises_before_enabling(manager, fake_module, log):
    original_init = fake_module.Client.__init__
    with pytest.raises(ConfigurationError):
        manager.activate_overrides([f"{MOD}.Client", "langchain"])
    assert manager._override_active.enabled is False
    assert fake_module.Client.__init__ is original_init


def test_registered_bare_target_is_not_revalidated(manager, log):
    manager.register_active_override("bare", object)
    manager.activate_overrides(["bare"])
    assert manager._override_active.enabled is True


def test_import_error_reports_detail(manager, log):
    manager.activate_overrides(["no_such_activate_mod_xyz.Client"])
    warns = _warnings(log)
    assert len(warns) == 1
    assert warns[0].startswith(
        "Could not override framework target: no_such_activate_mod_xyz.Client: "
    )
    assert "no_such_activate_mod_xyz" in warns[0]
    assert manager.applied_targets == []


def test_attribute_error_missing_class_reports_detail(manager, fake_module, log):
    manager.activate_overrides([f"{MOD}.Missing"])
    warns = _warnings(log)
    assert len(warns) == 1
    assert warns[0].startswith(f"Could not override framework target: {MOD}.Missing: ")
    assert manager.applied_targets == []


@pytest.mark.parametrize("marker", ["PydanticUserError", "__modify_schema__"])
def test_import_error_pydantic_marker(manager, log, marker):
    with patch.object(
        manager, "_create_override_constructor", side_effect=ImportError(f"x {marker}")
    ):
        sys.modules[MOD] = types.ModuleType(MOD)
        sys.modules[MOD].Client = type("Client", (), {})
        try:
            manager.activate_overrides([f"{MOD}.Client"])
        finally:
            sys.modules.pop(MOD, None)
    warns = _warnings(log)
    assert warns[0] == f"Pydantic compatibility issue with {MOD}.Client: x {marker}"
    assert _infos(log) == [
        f"Consider upgrading {MOD} package for Pydantic v2 compatibility"
    ]
    # override_applied stays False, no failure detail is recorded
    assert warns[1] == f"Could not override framework target: {MOD}.Client"
    assert len(warns) == 2


class PydanticUserError(Exception):
    pass


def _boom_with(exc):
    sys.modules[MOD] = types.ModuleType(MOD)
    sys.modules[MOD].Client = type("Client", (), {})
    return patch.object(
        FrameworkOverrideManager, "_create_override_constructor", side_effect=exc
    )


def test_generic_exception_pydantic_by_type(manager, log):
    try:
        with _boom_with(PydanticUserError("bad")):
            manager.activate_overrides([f"{MOD}.Client"])
    finally:
        sys.modules.pop(MOD, None)
    warns = _warnings(log)
    assert warns[0] == (
        f"Pydantic compatibility issue with {MOD}.Client: PydanticUserError: bad"
    )
    assert _infos(log) == [
        f"Skipping {MOD}.Client due to Pydantic v2 compatibility issues. "
        "Consider upgrading the package."
    ]
    assert warns[1] == f"Could not override framework target: {MOD}.Client"


def test_generic_exception_pydantic_by_message(manager, log):
    try:
        with _boom_with(RuntimeError("Some PYDANTIC problem")):
            manager.activate_overrides([f"{MOD}.Client"])
    finally:
        sys.modules.pop(MOD, None)
    assert _warnings(log)[0].startswith(f"Pydantic compatibility issue with {MOD}.Client")


def test_generic_exception_unexpected(manager, log):
    try:
        with _boom_with(RuntimeError("kaput")):
            manager.activate_overrides([f"{MOD}.Client"])
    finally:
        sys.modules.pop(MOD, None)
    warns = _warnings(log)
    assert warns[0] == f"Unexpected error loading {MOD}.Client: RuntimeError: kaput"
    assert warns[1] == f"Could not override framework target: {MOD}.Client"
    assert _infos(log) == []


def test_failure_does_not_stop_later_targets(manager, fake_module, log):
    manager.activate_overrides(["no_such_activate_mod_xyz.Client", f"{MOD}.Client"])
    assert manager.applied_targets == [f"{MOD}.Client"]
