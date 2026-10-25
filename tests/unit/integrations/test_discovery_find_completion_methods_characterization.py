"""Characterization tests for ParameterDiscovery.find_completion_methods.

These pin the exact current behaviour (results, ordering, duplicates, log
messages and instantiation side effects) so the function can be refactored
safely.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from traigent.integrations.utils import discovery as discovery_module
from traigent.integrations.utils.discovery import ParameterDiscovery

find = ParameterDiscovery.find_completion_methods


def _flags(auto_discovery=False, telemetry=False):
    return SimpleNamespace(
        auto_discovery=auto_discovery,
        emit_security_telemetry=telemetry,
        profile=SimpleNamespace(value="test-profile"),
    )


def _entry(name, path):
    return {"name": name, "path": path, "parameters": {}}


@pytest.fixture
def mock_logger():
    with patch.object(discovery_module, "logger") as m:
        yield m


@pytest.fixture
def no_pytest_env(monkeypatch):
    # pytest re-sets PYTEST_CURRENT_TEST at the start of the call phase, so
    # the variable is hidden through os.getenv instead of the environment.
    real_getenv = discovery_module.os.getenv

    def fake_getenv(key, default=None):
        if key == "PYTEST_CURRENT_TEST":
            return default
        return real_getenv(key, default)

    monkeypatch.setattr(discovery_module.os, "getenv", fake_getenv)


# ---------------------------------------------------------------------------
# Class-level search
# ---------------------------------------------------------------------------


def test_class_phase_finds_functions_in_dir_order_and_keywords_case_insensitive():
    class Api:
        __traigent_allow_instance_discovery__ = False

        def create(self):
            pass

        def Generate_text(self):
            pass

        def chat(self):
            pass

        def completion(self):
            pass

        def complete(self):
            pass

        def unrelated(self):
            pass

        def _create_private(self):
            pass

    assert find(Api) == [
        _entry("Generate_text", "Generate_text"),
        _entry("chat", "chat"),
        _entry("complete", "complete"),
        _entry("completion", "completion"),
        _entry("create", "create"),
    ]


def test_bound_classmethod_is_found_but_classes_and_builtins_are_not():
    class Inner:
        def create(self):
            pass

    class Api:
        __traigent_allow_instance_discovery__ = False
        create_class = Inner
        create_builtin = len

        @classmethod
        def create_cm(cls):
            pass

    assert find(Api) == [_entry("create_cm", "create_cm")]


def test_keyword_attr_that_is_not_callable_is_not_reported_but_is_recursed():
    class Leaf:
        def create(self):
            pass

    class Api:
        __traigent_allow_instance_discovery__ = False
        create_client = Leaf()

    assert find(Api) == [_entry("create", "create_client.create")]


def test_nested_paths_and_depth_limit():
    class L3:
        def create(self):
            pass

    class L2:
        deep = L3()

        def chat(self):
            pass

    class L1:
        level2 = L2()

    class Api:
        __traigent_allow_instance_discovery__ = False
        level1 = L1()

        def create(self):
            pass

    # level1.level2.chat is at depth 3 (found); level1.level2.deep.create is
    # at depth 4 (not searched).
    assert find(Api) == [
        _entry("create", "create"),
        _entry("chat", "level1.level2.chat"),
    ]


def test_non_class_object_is_searched_without_instantiation():
    class Obj:
        def generate(self):
            pass

    with patch.object(discovery_module, "get_security_flags") as flags:
        result = find(Obj())

    assert result == [_entry("generate", "generate")]
    flags.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling inside the recursive search
# ---------------------------------------------------------------------------


def test_dir_failure_is_logged_and_returns_empty(mock_logger):
    class BadDir:
        def __dir__(self):
            raise RuntimeError("no dir")

    assert find(BadDir()) == []
    mock_logger.debug.assert_called_once_with(
        "Error during nested method search: no dir"
    )


def test_attribute_error_after_first_attr_logs_previous_path_and_continues(
    mock_logger,
):
    class Flaky:
        def alpha_create(self):
            pass

        @property
        def bad(self):
            raise RuntimeError("boom")

        def zzz_create(self):
            pass

    assert find(Flaky()) == [
        _entry("alpha_create", "alpha_create"),
        _entry("zzz_create", "zzz_create"),
    ]
    mock_logger.debug.assert_called_once_with(
        "Could not inspect attribute at alpha_create: boom"
    )


def test_attribute_error_on_first_attr_aborts_the_scan(mock_logger):
    class Flaky:
        @property
        def bad(self):
            raise RuntimeError("boom")

        def create_after(self):
            pass

    assert find(Flaky()) == []
    assert mock_logger.debug.call_count == 1
    message = mock_logger.debug.call_args.args[0]
    assert message.startswith("Error during nested method search: ")
    assert "current_path" in message


def test_recursion_failure_in_nested_object_is_isolated(mock_logger):
    class BadNested:
        def __dir__(self):
            raise ValueError("nested dir")

    class Holder:
        a_bad = BadNested()

        def create(self):
            pass

    assert find(Holder()) == [_entry("create", "create")]
    mock_logger.debug.assert_called_once_with(
        "Error during nested method search: nested dir"
    )


# ---------------------------------------------------------------------------
# Instance-level discovery gate
# ---------------------------------------------------------------------------


def _counting_class(**attrs):
    class Api:
        init_calls = []

        def __init__(self):
            type(self).init_calls.append(1)
            self.client = SimpleNamespace(chat_fn=self._chat_fn)

        def _chat_fn(self):
            pass

        def create(self):
            pass

    for key, value in attrs.items():
        setattr(Api, key, value)
    return Api


def test_pytest_env_blocks_instantiation_without_explicit_setting(mock_logger):
    cls = _counting_class()
    with patch.object(
        discovery_module, "get_security_flags", return_value=_flags(True)
    ):
        result = find(cls)

    assert cls.init_calls == []
    assert result == [_entry("create", "create")]
    assert mock_logger.debug.call_args.args[0] == (
        "Skipping instantiation of %s during discovery (explicit opt-out)"
    )


def test_explicit_false_logs_skip_with_cls(mock_logger):
    cls = _counting_class(__traigent_allow_instance_discovery__=False)
    with patch.object(
        discovery_module, "get_security_flags", return_value=_flags(True)
    ):
        find(cls)

    assert cls.init_calls == []
    mock_logger.debug.assert_called_once_with(
        "Skipping instantiation of %s during discovery (explicit opt-out)", cls
    )


def test_explicit_true_instantiates_and_appends_instance_results(mock_logger):
    cls = _counting_class(__traigent_allow_instance_discovery__=True)
    with patch.object(
        discovery_module, "get_security_flags", return_value=_flags(False)
    ):
        result = find(cls)

    assert cls.init_calls == [1]
    # class phase first, then instance phase (duplicates preserved)
    assert result == [
        _entry("create", "create"),
        _entry("chat_fn", "client.chat_fn"),
        _entry("create", "create"),
    ]
    mock_logger.info.assert_not_called()


def test_auto_discovery_flag_allows_instantiation_when_not_under_pytest(
    no_pytest_env,
):
    cls = _counting_class()
    with patch.object(
        discovery_module, "get_security_flags", return_value=_flags(True)
    ):
        result = find(cls)

    assert cls.init_calls == [1]
    assert len(result) == 3


def test_auto_discovery_off_skips_when_not_under_pytest(no_pytest_env, mock_logger):
    cls = _counting_class()
    with patch.object(
        discovery_module, "get_security_flags", return_value=_flags(False)
    ):
        result = find(cls)

    assert cls.init_calls == []
    assert result == [_entry("create", "create")]
    mock_logger.debug.assert_called_once_with(
        "Skipping instantiation of %s during discovery (explicit opt-out)", cls
    )


def test_falsy_non_false_setting_neither_skips_nor_instantiates(mock_logger):
    cls = _counting_class(__traigent_allow_instance_discovery__=0)
    with patch.object(
        discovery_module, "get_security_flags", return_value=_flags(True)
    ):
        result = find(cls)

    assert cls.init_calls == []
    assert result == [_entry("create", "create")]
    mock_logger.debug.assert_not_called()


# ---------------------------------------------------------------------------
# Dummy argument construction
# ---------------------------------------------------------------------------


def test_dummy_args_by_annotation_and_defaults_skipped():
    captured = {}

    class Api:
        __traigent_allow_instance_discovery__ = True

        def __init__(
            self,
            name: str,
            count: int,
            ratio: float,
            flag: bool,
            other: list,
            unannotated,
            optional: str = "keep",
        ):
            captured.update(
                name=name,
                count=count,
                ratio=ratio,
                flag=flag,
                other=other,
                unannotated=unannotated,
                optional=optional,
            )

        def generate(self):
            pass

    with patch.object(
        discovery_module, "get_security_flags", return_value=_flags(False)
    ):
        result = find(Api)

    assert captured == {
        "name": "dummy",
        "count": 0,
        "ratio": 0.0,
        "flag": False,
        "other": None,
        "unannotated": None,
        "optional": "keep",
    }
    assert result == [_entry("generate", "generate"), _entry("generate", "generate")]
    assert type(captured["count"]) is int
    assert captured["flag"] is False


def test_default_typed_param_is_not_given_a_dummy_value():
    captured = {}

    class Api:
        __traigent_allow_instance_discovery__ = True

        def __init__(self, temperature=0.5):
            captured["temperature"] = temperature

    with patch.object(
        discovery_module, "get_security_flags", return_value=_flags(False)
    ):
        find(Api)

    assert captured == {"temperature": 0.5}


def test_instantiation_failure_is_logged_and_class_results_kept(mock_logger):
    class Api:
        __traigent_allow_instance_discovery__ = True

        def __init__(self, required: str):
            raise ValueError("cannot build")

        def create(self):
            pass

    with patch.object(
        discovery_module, "get_security_flags", return_value=_flags(False)
    ):
        result = find(Api)

    assert result == [_entry("create", "create")]
    mock_logger.debug.assert_called_once_with(
        "Could not create instance for discovery: cannot build"
    )


def test_outer_failure_returns_results_collected_so_far(mock_logger):
    class Api:
        def create(self):
            pass

    with patch.object(
        discovery_module, "get_security_flags", side_effect=RuntimeError("flags down")
    ):
        result = find(Api)

    assert result == [_entry("create", "create")]
    mock_logger.debug.assert_called_once_with(
        f"Could not find completion methods for {Api}: flags down"
    )
