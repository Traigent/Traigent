"""Optimization-only mode refuses free text in the search space and default config.

Owner ruling 2026-10-28 (A): the optimizer's search space and default config may carry
numbers, booleans, ``None`` and identifier strings only; any other string is customer
content and the whole request is refused with the closed contract error.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as oo

PROMPT = "You are a careful medical assistant who never reveals patient notes"
BAD_KEY = "patient name key with spaces"

_TYPED_KEY = "configuration_space"
_LEGACY_KEY = "search_space"
_CASES = [
    pytest.param(oo.project_session_create, _TYPED_KEY, id="typed"),
    pytest.param(oo.project_legacy_session_create, _LEGACY_KEY, id="legacy"),
]
_MODELS = [
    "gpt-4o-mini",
    "gpt-4.1",
    "claude-3-5-sonnet",
    "claude-3-5-sonnet-20241022",
    "anthropic/claude-3-opus",
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "models/gemini-1.5-pro",
]


@pytest.fixture(autouse=True)
def _clean_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    yield
    oo.reset_latch()


def _body(space_key: str, space: Any, **extra: Any) -> dict[str, Any]:
    return {space_key: space, **extra}


GOOD_SPACE = {
    "model": {"type": "categorical", "choices": ["gpt-4o-mini", "claude-3-5-sonnet"]},
    "temperature": {"type": "float", "low": 0.1, "high": 0.9, "log": False},
    "top_k": {"type": "int", "low": 1, "high": 8, "step": 1},
    "use_cache": {"type": "categorical", "choices": [True, False, None]},
    "mixed": [1, 2.5, "a", "b"],
}


@pytest.mark.parametrize(("project", "key"), _CASES)
def test_numeric_bool_and_identifier_space_passes_unchanged(project: Any, key: str):
    body = _body(
        key, copy.deepcopy(GOOD_SPACE), default_config={"model": "gpt-4o-mini"}
    )
    out = project(copy.deepcopy(body))
    assert out[key] == GOOD_SPACE
    assert out["default_config"] == {"model": "gpt-4o-mini"}


@pytest.mark.parametrize("model", _MODELS)
def test_common_model_names_are_identifiers(model: str) -> None:
    assert oo.ID_RE.fullmatch(model) is not None
    out = oo.project_session_create({_TYPED_KEY: {"model": [model]}})
    assert out[_TYPED_KEY] == {"model": [model]}


@pytest.mark.parametrize(("project", "key"), _CASES)
def test_prompt_in_list_choice_is_refused(project: Any, key: str) -> None:
    space = {"system_prompt": {"type": "categorical", "choices": ["ok_id", PROMPT]}}
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, space))


@pytest.mark.parametrize(("project", "key"), _CASES)
def test_prompt_nested_in_dict_is_refused(project: Any, key: str) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"a": {"b": {"c": [{"d": PROMPT}]}}}))


@pytest.mark.parametrize(("project", "key"), _CASES)
def test_non_identifier_dict_key_is_refused(project: Any, key: str) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {BAD_KEY: [1, 2]}))
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"ok": {BAD_KEY: 1}}))


@pytest.mark.parametrize(("project", "key"), _CASES)
def test_default_config_with_prompt_is_refused(project: Any, key: str) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"model": ["gpt-4o-mini"]}, default_config={"p": PROMPT}))


@pytest.mark.parametrize(("project", "key"), _CASES)
def test_error_echoes_neither_key_nor_value(project: Any, key: str) -> None:
    bodies = [
        _body(key, {"p": [PROMPT]}),
        _body(key, {BAD_KEY: [1]}),
        _body(key, {"p": [1]}, default_config={BAD_KEY: PROMPT}),
        _body(key, {"p": [1]}, warm_start_from=PROMPT),
    ]
    for body in bodies:
        with pytest.raises(oo.MetadataContractViolation) as caught:
            project(body)
        text = f"{caught.value!s} {caught.value.args!r} {caught.value!r}"
        assert PROMPT not in text
        assert BAD_KEY not in text
        assert "medical" not in text
        assert caught.value.args == (oo.OPTIMIZATION_ONLY_METADATA_INVALID_MESSAGE,)


@pytest.mark.parametrize(("project", "key"), _CASES)
def test_warm_start_from_identifier_passes_and_free_text_refused(
    project: Any, key: str
) -> None:
    out = project(_body(key, {"a": [1]}, warm_start_from="opt_123-abc"))
    assert out["warm_start_from"] == "opt_123-abc"
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"a": [1]}, warm_start_from=PROMPT))
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"a": [1]}, warm_start_from=5))


@pytest.mark.parametrize(("project", "key"), _CASES)
def test_boundaries(project: Any, key: str) -> None:
    assert project(_body(key, []))[key] == []
    assert project(_body(key, {}))[key] == {}
    assert project(_body(key, {"a": []}))[key] == {"a": []}
    for bad in (float("nan"), float("inf"), float("-inf"), 1e16):
        with pytest.raises(oo.MetadataContractViolation):
            project(_body(key, {"a": [bad]}))
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"a": [object()]}))
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"a": [b"bytes"]}))
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"a": [""]}))
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"a": ["x" * 129]}))
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"a": list(range(oo.MAX_LIST + 1))}))
    deep: Any = 1
    for _ in range(oo.MAX_DEPTH + 2):
        deep = [deep]
    with pytest.raises(oo.MetadataContractViolation):
        project(_body(key, {"a": deep}))


@pytest.mark.parametrize(("project", "key"), _CASES)
def test_absent_or_none_fields_are_untouched(project: Any, key: str) -> None:
    assert key not in project({})
    out = project(_body(key, None, default_config=None, warm_start_from=None))
    assert out == {key: None, "default_config": None, "warm_start_from": None}


@pytest.mark.parametrize(
    "phase",
    [oo.PHASE_SESSION_CREATE, oo.PHASE_SESSION_CREATE_LEGACY],
)
def test_mode_off_leaves_prompt_search_space_untouched(phase: str) -> None:
    key = _TYPED_KEY if phase == oo.PHASE_SESSION_CREATE else _LEGACY_KEY
    body = _body(key, {"system_prompt": [PROMPT]}, default_config={"p": PROMPT})
    assert oo.is_active() is False
    assert oo.project_for_send(phase, body) is body
    assert body[key] == {"system_prompt": [PROMPT]}


@pytest.mark.parametrize(
    ("phase", "key"),
    [
        (oo.PHASE_SESSION_CREATE, _TYPED_KEY),
        (oo.PHASE_SESSION_CREATE_LEGACY, _LEGACY_KEY),
    ],
)
def test_mode_on_refuses_prompt_search_space_through_send_boundary(
    phase: str, key: str
) -> None:
    body = _body(key, {"system_prompt": [PROMPT]})
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_send(phase, body, explicit=True)
    ok = _body(key, {"model": ["gpt-4o-mini"]})
    assert oo.project_for_send(phase, ok, explicit=True)[key] == ok[key]
