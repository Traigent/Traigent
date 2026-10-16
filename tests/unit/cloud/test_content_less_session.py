"""Content-less session create, phase 3 group 1 (identity), SDK side.

Opt-in off must leave the legacy request untouched; opt-in on declares the
``content_less_v1`` capability, a closed environment and opaque refs, and never sends
customer-chosen identity.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.cloud import content_less as cl
from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.client import SessionContractError
from traigent.cloud.models import SessionCreationRequest

CANARY_AGENT = "cnryAgent_Rn8F-3hSx1Qj5"
CANARY_EVAL_SET = "cnryEvalSet_Wt2G-6kYd9Vb3"
CANARY_DATASET_ID = "cnryDsId_Mu3Q-4cRy8Xn1"
CANARY_EVALUATOR = "cnryEvalId_Ag1Z-7oLj5Tf9"
CANARY_FILE = "cnryFile_Xq7K-2vMb9Rt4"
CANARY_DISPLAY = "cnryDisplay_Zz1Q-9aBc4"

AGENT_REF = "agentref_" + "ab" * 16
DATASET_REF = "datasetref_" + "cd" * 16
EVALUATOR_REF = "evaluatorref_" + "ef" * 16
IDENTITY_CANARIES = (
    CANARY_AGENT,
    CANARY_EVAL_SET,
    CANARY_DATASET_ID,
    CANARY_EVALUATOR,
    CANARY_FILE,
    CANARY_DISPLAY,
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "TRAIGENT_CONTENT_LESS",
        "TRAIGENT_AGENT_REF",
        "TRAIGENT_ENVIRONMENT",
        "OTEL_RESOURCE_ATTRIBUTES",
        "ENVIRONMENT",
        "TRAIGENT_ENV",
        "TRAIGENT_SESSION_CONTRACT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRAIGENT_RESULTS_FOLDER", str(tmp_path))


def _request(**overrides: Any) -> SessionCreationRequest:
    base: dict[str, Any] = {
        "function_name": "pkg.mod.fn (abc123)",
        "agent_key": CANARY_AGENT,
        "configuration_space": {"temperature": [0.1, 0.9]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 2, "name": CANARY_EVAL_SET},
        "metadata": {
            "function_name": "pkg/mod.py_fn",
            "function_display_name": CANARY_DISPLAY,
            "function_slug": "slug_abc123",
            "evaluation_set": CANARY_EVAL_SET,
            "function_module": "pkg.mod",
        },
        "dataset_id": CANARY_DATASET_ID,
        "evaluator_id": CANARY_EVALUATOR,
        "max_trials": 3,
    }
    base.update(overrides)
    return SessionCreationRequest(**base)


def _ops() -> ApiOperations:
    return ApiOperations(MagicMock())


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    async def json(self) -> dict[str, Any]:
        return self._body


def _created(**extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "session_id": "s1",
        "status": "active",
        "metadata": {"experiment_id": "e", "experiment_run_id": "r"},
        "tenant_id": "tenant_" + "1" * 32,
        "project_id": "project_" + "2" * 32,
        "capabilities": ["content_less_v1"],
        "agent_ref": AGENT_REF,
        "dataset_ref": DATASET_REF,
        "evaluator_ref": EVALUATOR_REF,
    }
    body.update(extra)
    return body


# --------------------------------------------------------------------- opt-in off


def test_opt_in_off_is_the_legacy_request() -> None:
    ops = _ops()
    request = _request()
    payload = ops._build_session_payload(request, 3)
    legacy = ops._build_typed_session_payload(request, 3)
    assert json.dumps(payload, sort_keys=True) == json.dumps(legacy, sort_keys=True)
    for key in ("capabilities", "environment", "agent_ref", "dataset_ref"):
        assert key not in payload
    assert payload["agent_key"] == CANARY_AGENT
    assert payload["dataset_id"] == CANARY_DATASET_ID
    assert payload["metadata"]["evaluation_set"] == CANARY_EVAL_SET


def test_explicit_false_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_CONTENT_LESS", "1")
    payload = _ops()._build_session_payload(
        _request(content_less=cl.ContentLessOptions(enabled=False)), 3
    )
    assert "capabilities" not in payload


# ---------------------------------------------------------------------- opt-in on


def test_env_opt_in_declares_capability_environment_and_no_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_CONTENT_LESS", "1")
    monkeypatch.setenv("TRAIGENT_ENVIRONMENT", "staging")
    payload = _ops()._build_session_payload(_request(), 3)
    assert payload["capabilities"] == ["content_less_v1"]
    assert payload["environment"] == "staging"
    text = json.dumps(payload)
    assert not [c for c in IDENTITY_CANARIES if c in text]
    for key in cl.LEGACY_TOP_LEVEL:
        assert key not in payload
    assert "name" not in payload["dataset_metadata"]
    for key in cl.LEGACY_METADATA:
        assert key not in payload["metadata"]
    # the function-derived code identifier stays
    assert payload["metadata"]["function_name"] == "pkg/mod.py_fn"
    assert payload["function_name"] == "pkg.mod.fn (abc123)"


def test_option_opt_in_sends_explicit_agent_ref() -> None:
    options = cl.ContentLessOptions(enabled=True, agent_ref=AGENT_REF)
    payload = _ops()._build_session_payload(_request(content_less=options), 3)
    assert payload["agent_ref"] == AGENT_REF
    assert "agent_key" not in payload


def test_agent_ref_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_AGENT_REF", AGENT_REF)
    options = cl.ContentLessOptions(enabled=True)
    assert (
        _ops()._build_session_payload(_request(content_less=options), 3)["agent_ref"]
        == AGENT_REF
    )


def test_explicit_agent_ref_beats_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_AGENT_REF", "agentref_" + "11" * 16)
    options = cl.ContentLessOptions(enabled=True, agent_ref=AGENT_REF)
    assert cl.resolve_agent_ref(options) == AGENT_REF


@pytest.mark.parametrize(
    "bad",
    [
        "Alice_has_diabetes",
        "agentref_XYZ",
        "datasetref_" + "ab" * 16,
        "agentref_" + "AB" * 16,
        "agentref_" + "ab" * 15,
    ],
)
def test_invalid_agent_ref_is_refused_locally_without_echo(bad: str) -> None:
    options = cl.ContentLessOptions(enabled=True, agent_ref=bad)
    with pytest.raises(cl.ContentLessRefusal) as info:
        _ops()._build_session_payload(_request(content_less=options), 3)
    assert info.value.code == cl.INVALID_AGENT_REF
    assert bad not in str(info.value)
    assert isinstance(info.value, SessionContractError)


def test_invalid_agent_ref_from_environment_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_AGENT_REF", "Alice_has_diabetes")
    with pytest.raises(cl.ContentLessRefusal):
        cl.resolve_agent_ref(cl.ContentLessOptions(enabled=True))


def test_legacy_contract_cannot_carry_content_less(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRAIGENT_SESSION_CONTRACT", "legacy")
    with pytest.raises(SessionContractError):
        _ops()._build_session_payload(
            _request(content_less=cl.ContentLessOptions(enabled=True)), 3
        )


# --------------------------------------------------------------------- environment


def test_environment_precedence_chain() -> None:
    env = {
        "TRAIGENT_ENVIRONMENT": "test",
        "OTEL_RESOURCE_ATTRIBUTES": "service.name=x,deployment.environment.name=staging",
        "ENVIRONMENT": "development",
        "TRAIGENT_ENV": "production",
    }
    assert cl.resolve_environment("staging", env) == "staging"  # explicit
    assert cl.resolve_environment(None, env) == "test"  # TRAIGENT_ENVIRONMENT
    del env["TRAIGENT_ENVIRONMENT"]
    assert cl.resolve_environment(None, env) == "staging"  # OTel
    del env["OTEL_RESOURCE_ATTRIBUTES"]
    assert cl.resolve_environment(None, env) == "development"  # ENVIRONMENT
    del env["ENVIRONMENT"]
    assert cl.resolve_environment(None, env) == "production"  # legacy TRAIGENT_ENV


def test_environment_is_default_and_never_production_when_unset() -> None:
    assert cl.resolve_environment(None, {}) == "default"
    assert cl.resolve_environment(None, {"ENVIRONMENT": "  ", "TRAIGENT_ENV": ""}) == (
        "default"
    )
    # nothing in the SDK's own process flags (CI, K8s, cloud) is read as a guess
    assert cl.resolve_environment(
        None, {"CI": "true", "KUBERNETES_SERVICE_HOST": "x"}
    ) == ("default")


@pytest.mark.parametrize("bad", ["Alice_has_diabetes", "live", "production2", "prd"])
def test_environment_outside_the_enum_is_refused_without_echo(bad: str) -> None:
    for env, explicit in (({}, bad), ({"ENVIRONMENT": bad}, None)):
        with pytest.raises(cl.ContentLessRefusal) as info:
            cl.resolve_environment(explicit, env)
        assert info.value.code == cl.INVALID_ENVIRONMENT
        assert bad not in str(info.value)


def test_environment_is_case_and_whitespace_normalised() -> None:
    assert cl.resolve_environment(" Production ", {}) == "production"


def test_environment_option_reaches_the_payload() -> None:
    options = cl.ContentLessOptions(enabled=True, environment="test")
    assert (
        _ops()._build_session_payload(_request(content_less=options), 3)["environment"]
        == "test"
    )


# --------------------------------------------------------------------- responses


def test_backend_without_the_capability_is_refused() -> None:
    ops = _ops()
    ctx = (cl.RefCache(), "p", {}, {})
    for body in (
        {"session_id": "s"},
        {"session_id": "s", "capabilities": []},
        {"session_id": "s", "capabilities": ["something_else"]},
        {"session_id": "s", "capabilities": "content_less_v1"},
    ):
        with pytest.raises(cl.ContentLessRefusal) as info:
            asyncio.run(
                ops._parse_session_response(_Response(body), content_less_ctx=ctx)
            )
        assert info.value.code == cl.BACKEND_UPGRADE_REQUIRED


def test_refs_are_cached_per_scope_and_reused() -> None:
    ops = _ops()
    options = cl.ContentLessOptions(enabled=True)
    request = _request(content_less=options)
    first = ops._build_session_payload(request, 3)
    assert "agent_ref" not in first and "dataset_ref" not in first
    ctx = ops._content_less_ctx
    parsed = asyncio.run(
        ops._parse_session_response(_Response(_created()), content_less_ctx=ctx)
    )
    assert parsed[0] == "s1"

    second = ops._build_session_payload(_request(content_less=options), 3)
    assert second["agent_ref"] == AGENT_REF
    assert second["dataset_ref"] == DATASET_REF
    assert second["evaluator_ref"] == EVALUATOR_REF
    assert not [c for c in IDENTITY_CANARIES if c in json.dumps(second)]

    # a different dataset label does not pick up the first dataset's ref
    other = ops._build_session_payload(
        _request(content_less=options, dataset_id="somethingElse"), 3
    )
    assert "dataset_ref" not in other and other["agent_ref"] == AGENT_REF


def test_ref_cache_holds_no_plaintext_names(tmp_path: Path) -> None:
    ops = _ops()
    options = cl.ContentLessOptions(enabled=True)
    ops._build_session_payload(_request(content_less=options), 3)
    asyncio.run(
        ops._parse_session_response(
            _Response(_created()), content_less_ctx=ops._content_less_ctx
        )
    )
    text = (tmp_path / "content_less_refs.json").read_text(encoding="utf-8")
    assert AGENT_REF in text
    assert not [c for c in IDENTITY_CANARIES if c in text]
    assert "pkg/mod.py_fn" not in text


def test_malformed_response_refs_are_not_cached(tmp_path: Path) -> None:
    ops = _ops()
    options = cl.ContentLessOptions(enabled=True)
    ops._build_session_payload(_request(content_less=options), 3)
    asyncio.run(
        ops._parse_session_response(
            _Response(_created(agent_ref="Alice_has_diabetes", dataset_ref=None)),
            content_less_ctx=ops._content_less_ctx,
        )
    )
    again = ops._build_session_payload(_request(content_less=options), 3)
    assert "agent_ref" not in again and "dataset_ref" not in again


def test_refs_are_scoped_to_the_credential() -> None:
    cache = cl.RefCache()
    keys = {"agent": "fn"}
    cache.store("principal-a", "t", "p", keys, {"agent": AGENT_REF})
    assert cache.lookup("principal-a", keys) == {"agent": AGENT_REF}
    assert cache.lookup("principal-b", keys) == {}
    cache.forget("principal-a")
    assert cache.lookup("principal-a", keys) == {}


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("prod", "production"),
        ("PROD", "production"),
        ("dev", "development"),
        ("stage", "staging"),
        ("STG", "staging"),
        ("testing", "test"),
        (" Testing ", "test"),
    ],
)
def test_environment_aliases(alias: str, expected: str) -> None:
    assert cl.resolve_environment(alias, {}) == expected
    assert cl.resolve_environment(None, {"ENVIRONMENT": alias}) == expected
    assert cl.resolve_environment(None, {"TRAIGENT_ENV": alias}) == expected
