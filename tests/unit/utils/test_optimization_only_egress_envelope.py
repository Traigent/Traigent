"""Optimization-only session-create projection is default-deny at the top level.

Sol review of #2777: the projectors copied the whole body and validated selected fields, so
``budget={"prompt": ...}``, a sentence in a constraint, a ``10**400`` integer, a smart-pruning
error echoing customer keys, and a ``content_identity``-named tunable escaped the closed
contract. Every top-level key now needs an entry in the envelope allowlist.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.models import SessionCreationRequest
from traigent.cloud.smart_pruning import normalize_smart_pruning_options
from traigent.utils import optimization_only_egress as oo

PROMPT = "You are a careful medical assistant who never reveals patient notes"
SYNTHETIC_KEY = "zq_customer_secret_key"
SYNTHETIC_VALUE = "zq_customer_secret_value"

_PROJECTORS = [
    pytest.param(oo.project_session_create, "configuration_space", id="typed"),
    pytest.param(oo.project_legacy_session_create, "search_space", id="legacy"),
]


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        oo.ENV_VAR,
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
    oo.reset_latch()
    yield
    oo.reset_latch()


# ----------------------------------------------------------- the real producers' bodies


def _full_request(**overrides: Any) -> SessionCreationRequest:
    fingerprints = {
        "dataset": "fp1:" + "a" * 64,
        "agent": "fp1:" + "b" * 64,
        "evaluator": "fp1:" + "c" * 64,
        "config_space": "fp1:" + "d" * 64,
    }
    base: dict[str, Any] = {
        "function_name": "pkg.mod.fn",
        "agent_key": "agent_one",
        "run_title": PROMPT,
        "run_description": PROMPT,
        "configuration_space": {"temperature": [0.1, 0.9], "model": ["gpt-4o-mini"]},
        "objectives": ["accuracy", {"name": "cost", "orientation": "minimize"}],
        "dataset_metadata": {"size": 4, "name": "eval_set", "privacy_mode": True},
        "max_trials": 5,
        "budget": {"max_cost_usd": 2.5, "max_wallclock_ms": 30000},
        "constraints": {
            "structural": [{"id": "c1", "require": "params.temperature <= 0.9"}],
            "derived": [
                {
                    "when": 'params.model == "gpt-4o-mini"',
                    "then": "metrics.accuracy >= 0.5",
                }
            ],
        },
        "default_config": {"temperature": 0.5},
        "warm_start_from": "prior_exp_1",
        "smart_pruning": {"label": "balanced", "epsilon": 0.1, "confidence": 0.9},
        "promotion_policy": {"min_improvement": 0.01},
        "tvl_governance": None,
        "optimization_strategy": {
            "algorithm": "optuna",
            "tracking_mode": "native_local",
        },
        "user_id": None,
        "billing_tier": "standard",
        "metadata": {"evaluation_set": "eval_set", "privacy_mode": True},
        "artifact_fingerprints": fingerprints,
        "fingerprint_meta": {
            "algorithm": "fp1",
            "dataset_example_count": 4,
            "source_available": True,
        },
        "evaluator_id": "evaluator_one",
        "task_type": "multiple_choice",
        "dataset_id": "dataset_one",
        "content_identity": {"scheme": "ci1", "provenance": "declared"},
    }
    base.update(overrides)
    return SessionCreationRequest(**base)


def _ops() -> ApiOperations:
    client = MagicMock()
    client.backend_config.api_base_url = "https://api.example.test"
    return ApiOperations(client)


def _cloud_client_body(request: SessionCreationRequest) -> dict[str, Any]:
    stub = SimpleNamespace(_ensure_owner_metadata=lambda metadata: metadata or {})
    return TraigentCloudClient._serialize_session_request(stub, request)


def _sync_manager_body() -> dict[str, Any]:
    from traigent.cloud.sync_manager import SyncManager
    from traigent.storage.local_storage import OptimizationSession

    session = OptimizationSession(
        session_id="s1",
        function_name="pkg.mod.fn",
        created_at="2026-10-28T00:00:00",
        updated_at="2026-10-28T00:00:00",
        status="completed",
        total_trials=0,
        completed_trials=0,
        best_score=None,
        best_config=None,
        optimization_config={"search_space": {"temperature": [0.1, 0.9]}},
        trials=[],
        metadata={"agent_key": "agent_one", "evaluation_set": "eval_set"},
    )
    manager = object.__new__(SyncManager)
    return SyncManager.convert_session_to_traigent_format(manager, session)[
        "session_create"
    ]


def _producer_bodies(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    request = _full_request()
    typed = _ops()._build_session_payload(request, max_trials=5)
    monkeypatch.setenv("TRAIGENT_SESSION_CONTRACT", "legacy")
    legacy = _ops()._build_session_payload(
        _full_request(optimization_strategy=None, promotion_policy=None),
        max_trials=5,
    )
    monkeypatch.delenv("TRAIGENT_SESSION_CONTRACT")
    monkeypatch.setenv("TRAIGENT_CONTENT_LESS", "1")
    monkeypatch.setenv("TRAIGENT_ENVIRONMENT", "test")
    content_less = _ops()._build_session_payload(request, max_trials=5)
    monkeypatch.delenv("TRAIGENT_CONTENT_LESS")
    return {
        "typed": typed,
        "legacy": legacy,
        "content_less": content_less,
        "cloud_client": _cloud_client_body(request),
        "sync_manager": _sync_manager_body(),
    }


@pytest.mark.parametrize(
    "producer", ["typed", "legacy", "content_less", "cloud_client", "sync_manager"]
)
def test_allowlist_covers_every_real_producer_body(
    producer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _producer_bodies(monkeypatch)[producer]
    # Producer bodies reach the projector through the AUTO phase; run it the same way.
    projected = oo.project_for_phase(oo.PHASE_SESSION_CREATE_AUTO, body)
    assert set(projected) <= set(body)
    text = json.dumps(projected, default=str)
    assert PROMPT not in text  # run_title / run_description are dropped


def test_producer_bodies_exercise_the_budget_and_smart_pruning_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    typed = _producer_bodies(monkeypatch)["typed"]
    assert typed["budget"] == {"max_cost_usd": 2.5, "max_wallclock_ms": 30000}
    assert typed["smart_pruning"]["label"] == "balanced"
    assert set(typed["artifact_versions"]) == {
        "agent",
        "dataset",
        "evaluator",
        "config_space",
    }
    projected = oo.project_session_create(typed)
    assert projected["budget"] == typed["budget"]
    assert projected["artifact_versions"] == typed["artifact_versions"]


# ------------------------------------------------------------------- default-deny


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
def test_unknown_top_level_key_is_refused(project: Any, space_key: str) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        project({space_key: {"a": [1, 2]}, "brand_new_field": "x"})


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
def test_unknown_key_with_content_value_is_refused_without_echo(
    project: Any, space_key: str
) -> None:
    with pytest.raises(oo.MetadataContractViolation) as caught:
        project({space_key: {"a": [1]}, SYNTHETIC_KEY: SYNTHETIC_VALUE})
    assert SYNTHETIC_KEY not in str(caught.value) + repr(caught.value)
    assert SYNTHETIC_VALUE not in str(caught.value) + repr(caught.value)


def test_typed_budget_with_prompt_is_refused() -> None:
    body = {"configuration_space": {"a": [1]}, "budget": {"prompt": PROMPT}}
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_session_create(body)


@pytest.mark.parametrize(
    "budget",
    [
        {"prompt": "private customer prompt"},
        {"max_cost_usd": "free text here"},
        {"max_cost_usd": -1},
        {"max_cost_usd": {"nested": 1}},
        {"bad key": 1},
        ["max_cost_usd"],
    ],
)
def test_budget_must_be_closed_numbers(budget: Any) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_session_create({"configuration_space": {"a": [1]}, "budget": budget})


def test_budget_of_numbers_passes() -> None:
    budget = {"max_cost_usd": 1.5, "max_wallclock_ms": 1000}
    out = oo.project_session_create(
        {"configuration_space": {"a": [1]}, "budget": budget}
    )
    assert out["budget"] == budget


def test_typed_alternate_search_space_key_is_checked() -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_session_create(
            {"configuration_space": {"a": [1]}, "search_space": {"p": PROMPT}}
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("smart_pruning", {"label": PROMPT}),
        ("smart_pruning", {"label": "balanced", "note": PROMPT}),
        ("artifact_fingerprints", {"dataset": PROMPT}),
        ("artifact_versions", {"agent": {"schema": PROMPT}}),
        ("user_id", PROMPT),
        ("billing_tier", PROMPT),
        ("evaluator_code_digest", PROMPT),
        ("environment", PROMPT),
        ("capabilities", [PROMPT]),
    ],
)
def test_envelope_specs_refuse_free_text(key: str, value: Any) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_session_create({"configuration_space": {"a": [1]}, key: value})


def test_legacy_smart_pruning_and_fingerprints_are_checked() -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_legacy_session_create(
            {"search_space": {"a": [1]}, "smart_pruning": {"label": PROMPT}}
        )
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_legacy_session_create(
            {"search_space": {"a": [1]}, "artifact_fingerprints": {"agent": PROMPT}}
        )


# ------------------------------------------------------------- constraint syntax


@pytest.mark.parametrize(
    "expr",
    [
        "You are a careful medical assistant",
        "never reveal patient notes please",
        "the quick brown fox",
        "hello world; import os",
    ],
)
def test_constraint_sentence_is_refused(expr: str) -> None:
    for key in ("require", "when", "then", "expr"):
        with pytest.raises(oo.MetadataContractViolation):
            oo.project_session_create(
                {
                    "configuration_space": {"a": [1]},
                    "constraints": {"structural": [{key: expr}]},
                }
            )


@pytest.mark.parametrize(
    "expr",
    [
        "params.temperature <= 0.9",
        "metrics.accuracy >= 0.8",
        'params.model == "accurate"',
        "params.max_tokens >= 64",
        "env.quota > 0",
        "True",
        "params.x > 1 and params.y < 2 or not params.z",
        "min(params.a, params.b) + abs(params.c) * 2 > 3",
        "math.sqrt(params.x) < 3",
        "params.x > 1 && params.y < 2",
        "params.x = 3",
    ],
)
def test_real_constraint_expressions_still_pass(expr: str) -> None:
    out = oo.project_session_create(
        {
            "configuration_space": {"a": [1]},
            "constraints": {"derived": [{"require": expr}]},
        }
    )
    assert out["constraints"]["derived"] == [{"require": expr}]


@pytest.mark.parametrize(
    "expr", ["__import__('os')", "open(params.x)", "params.__class__ > 1", "x ** 2"]
)
def test_constraint_outside_the_dsl_is_refused(expr: str) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_session_create(
            {
                "configuration_space": {"a": [1]},
                "constraints": {"derived": [{"require": expr}]},
            }
        )


# ---------------------------------------------------------------- numeric overflow


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
def test_huge_int_in_search_space_is_a_contract_violation(
    project: Any, space_key: str
) -> None:
    for value in (10**400, -(10**400), {"low": 0, "high": [10**400]}):
        with pytest.raises(oo.MetadataContractViolation):
            project({space_key: {"nested": {"deep": [value]}}})


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
def test_huge_int_in_default_config_is_a_contract_violation(
    project: Any, space_key: str
) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        project({space_key: {"a": [1]}, "default_config": {"a": 10**400}})


def test_huge_int_in_closed_tree_is_dropped_not_raised() -> None:
    assert oo.s_num()(5, oo._Ctx(), 0) == 5
    with pytest.raises(oo._Drop):
        oo.s_num()(10**400, oo._Ctx(), 0)
    with pytest.raises(oo._Drop):
        oo.s_closed_tree(10**400, oo._Ctx(), 0)


# --------------------------------------------------------------- smart-pruning errors


def test_smart_pruning_error_names_no_key_or_value() -> None:
    with pytest.raises(ValueError) as caught:
        normalize_smart_pruning_options(
            {"label": "balanced", SYNTHETIC_KEY: SYNTHETIC_VALUE}
        )
    exc = caught.value
    assert SYNTHETIC_KEY not in str(exc) + repr(exc)
    assert SYNTHETIC_VALUE not in str(exc) + repr(exc)
    assert exc.__cause__ is None
    assert "smart_pruning" in str(exc)


def test_smart_pruning_bad_value_is_not_echoed() -> None:
    with pytest.raises(ValueError) as caught:
        normalize_smart_pruning_options({"label": SYNTHETIC_VALUE})
    assert SYNTHETIC_VALUE not in str(caught.value) + repr(caught.value)
    assert caught.value.__cause__ is None


# ------------------------------------------------------------------- size accounting


@pytest.mark.parametrize(("project", "space_key"), _PROJECTORS)
def test_content_identity_named_tunable_gets_no_size_exemption(
    project: Any, space_key: str
) -> None:
    chunk = {f"k{i}": ["x" * 128] * 256 for i in range(24)}
    space = {"content_identity": chunk, "ordinary": chunk}
    # ``x`` * 128 is identifier-shaped, so only the size limit can refuse it.
    with pytest.raises(oo.MetadataContractViolation):
        project({space_key: space})
    with pytest.raises(oo.MetadataContractViolation):
        project({space_key: {"renamed": chunk, "ordinary": chunk}})


# ----------------------------------------- every dedicated-handler key, wrong-typed

SECRET = "private customer prompt"  # pragma: allowlist secret
_DEDICATED_TYPED = [
    "function_name", "agent_key", "agent_id", "dataset_id", "evaluator_id",
    "evaluator_definition_id", "run_title", "run_description", "objectives",
    "metadata", "dataset_metadata", "content_identity", "code_revision",
    "optimization_strategy", "promotion_policy", "tvl_governance",
    "fingerprint_meta", "task_type", "constraints", "configuration_space",
    "search_space", "default_config", "warm_start_from",
]  # fmt: skip
_DEDICATED_LEGACY = [
    "problem_statement", "optimization_config", "dataset", "metadata", "search_space",
    "default_config", "warm_start_from", "evaluator_id", "evaluator_definition_id",
    "fingerprint_meta",
]  # fmt: skip
_WRONG_VALUES = [
    pytest.param({"prompt": SECRET}, id="nested"),
    pytest.param(SECRET, id="bare"),
    pytest.param([SECRET], id="list"),
    pytest.param({"a": {"prompt": [SECRET]}}, id="deep"),
]


def _send_and_enforce(phase: str, body: dict[str, Any]) -> str:
    projected = oo.project_for_send(phase, body, explicit=True)
    return json.dumps(projected, default=str)


def _assert_refused_or_clean(phase: str, body: dict[str, Any]) -> None:
    try:
        projected = oo.project_for_send(phase, body, explicit=True)
        raw = json.dumps(projected).encode()
        oo.enforce_outgoing(
            "POST", "https://api.example.test/api/v1/sessions", {}, raw, explicit=True
        )
    except (oo.MetadataContractViolation, oo.OptimizationOnlyRefusal):
        return
    assert SECRET not in raw.decode()


@pytest.mark.parametrize("value", _WRONG_VALUES)
@pytest.mark.parametrize("key", _DEDICATED_TYPED)
def test_typed_dedicated_handlers_never_pass_nested_content(
    key: str, value: Any
) -> None:
    body: dict[str, Any] = {"configuration_space": {"a": [1]}}
    body[key] = value
    _assert_refused_or_clean(oo.PHASE_SESSION_CREATE_AUTO, body)


@pytest.mark.parametrize("value", _WRONG_VALUES)
@pytest.mark.parametrize("key", _DEDICATED_LEGACY)
def test_legacy_dedicated_handlers_never_pass_nested_content(
    key: str, value: Any
) -> None:
    body: dict[str, Any] = {"problem_statement": "p", "search_space": {"a": [1]}}
    body[key] = value
    _assert_refused_or_clean(oo.PHASE_SESSION_CREATE_AUTO, body)


@pytest.mark.parametrize(
    "value", ["private customer prompt", {"prompt": SECRET}, 7, [SECRET]]
)
def test_objectives_wrong_shapes_are_refused(value: Any) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_session_create(
            {"configuration_space": {"a": [1]}, "objectives": value}
        )


@pytest.mark.parametrize("value", [{"prompt": SECRET}, 7, [SECRET], True])
def test_task_type_container_or_number_is_refused(value: Any) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_session_create(
            {"configuration_space": {"a": [1]}, "task_type": value}
        )


def test_task_type_free_text_string_is_dropped() -> None:
    out = oo.project_session_create(
        {"configuration_space": {"a": [1]}, "task_type": SECRET}
    )
    assert "task_type" not in out


# ------------------------------------------- the wire form, not the Python object

import asyncio  # noqa: E402

from traigent.labels import transport_gate as tg  # noqa: E402

_SESSIONS_URL = "https://api.example.test/api/v1/sessions"


class _LyingKey(str):
    """Hashes and compares as ``billing_tier`` but serializes as its (secret) content."""

    def __hash__(self) -> int:
        return hash("billing_tier")

    def __eq__(self, other: object) -> bool:
        return other == "billing_tier"


class _LyingValue(str):
    def __hash__(self) -> int:
        return hash("test")

    def __eq__(self, other: object) -> bool:
        return True


class _OddDict(dict):  # type: ignore[type-arg]
    """A dict whose ``keys``/``__iter__``/``items`` lie about what ``json`` will write."""

    def keys(self) -> Any:  # type: ignore[override]
        return ["a"]

    def __iter__(self) -> Any:
        return iter(["a"])

    def items(self) -> Any:  # type: ignore[override]
        return [("a", [1])]


class _StrDiffers(str):
    def __str__(self) -> str:
        return "harmless_id"

    def __hash__(self) -> int:
        return hash("harmless_id")

    def __eq__(self, other: object) -> bool:
        return other == "harmless_id"


class _FakeTransport:
    _traigent_oo_explicit = True
    _traigent_gate_origins = frozenset({("https", "api.example.test", 443)})
    _json_serialize = staticmethod(json.dumps)
    headers: dict[str, str] = {}
    sent: list[bytes] = []

    async def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        if "data" in kwargs:
            data = kwargs["data"]
            body = data if isinstance(data, bytes) else data._value
        else:
            body = json.dumps(kwargs["json"]).encode()
        type(self).sent.append(bytes(body))
        return SimpleNamespace(status=200)


def _lying_bodies() -> list[dict[str, Any]]:
    base: dict[str, Any] = {"configuration_space": {"a": [1]}}
    return [
        {**base, _LyingKey(SECRET): "standard"},
        {**base, "environment": _LyingValue(SECRET)},
        {**base, "budget": {_LyingKey(SECRET): 1}},
        {**base, "metadata": _OddDict({"a": [1], "hidden": SECRET})},
        {"configuration_space": _OddDict({"a": [1], "hidden": SECRET})},
        {**base, "billing_tier": _StrDiffers(SECRET)},
    ]


@pytest.mark.parametrize("index", range(6))
def test_projection_judges_the_serialized_form(index: int) -> None:
    body = _lying_bodies()[index]
    try:
        projected = oo.project_for_send(
            oo.PHASE_SESSION_CREATE_AUTO, body, explicit=True
        )
    except (oo.MetadataContractViolation, oo.OptimizationOnlyRefusal):
        return
    assert type(projected) is dict
    assert SECRET not in json.dumps(projected)


@pytest.mark.parametrize("index", range(6))
def test_aiohttp_wrapper_never_sends_lying_bodies(index: int) -> None:
    body = _lying_bodies()[index]
    _FakeTransport.sent = []
    session = tg._build_aiohttp(_FakeTransport)()

    async def go() -> None:
        projected = oo.project_for_send(
            oo.PHASE_SESSION_CREATE_AUTO, body, explicit=True
        )
        await session._request("POST", _SESSIONS_URL, json=projected)

    try:
        asyncio.run(go())
    except (oo.MetadataContractViolation, oo.OptimizationOnlyRefusal):
        pass
    assert all(SECRET.encode() not in sent for sent in _FakeTransport.sent)


def test_aiohttp_wrapper_enforces_unprojected_lying_json_directly() -> None:
    """Even a body that skipped projection is judged on the bytes the transport writes."""
    _FakeTransport.sent = []
    session = tg._build_aiohttp(_FakeTransport)()
    body = {"configuration_space": {"a": [1]}, "environment": _LyingValue(SECRET)}
    with pytest.raises(oo.OptimizationOnlyRefusal):
        asyncio.run(session._request("POST", _SESSIONS_URL, json=body))
    assert _FakeTransport.sent == []


def test_aiohttp_wrapper_sends_the_checked_bytes() -> None:
    _FakeTransport.sent = []
    session = tg._build_aiohttp(_FakeTransport)()
    body = oo.project_for_send(
        oo.PHASE_SESSION_CREATE_AUTO, {"configuration_space": {"a": [1]}}, explicit=True
    )
    asyncio.run(session._request("POST", _SESSIONS_URL, json=body))
    assert json.loads(_FakeTransport.sent[0]) == body


def test_nan_in_search_space_still_refuses() -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_send(
            oo.PHASE_SESSION_CREATE_AUTO,
            {"configuration_space": {"a": [float("nan")]}},
            explicit=True,
        )


def test_mode_off_returns_the_original_object() -> None:
    body = {"configuration_space": {"a": [1]}, "environment": _LyingValue(SECRET)}
    assert (
        oo.project_for_send(oo.PHASE_SESSION_CREATE_AUTO, body, explicit=False) is body
    )


class _SubStr(str):
    """A plain str subclass: equal and hashed like str, written like str."""


@pytest.mark.parametrize("value", _WRONG_VALUES)
@pytest.mark.parametrize("key", _DEDICATED_TYPED)
def test_typed_handlers_with_str_subclass_values(key: str, value: Any) -> None:
    def wrap(v: Any) -> Any:
        if isinstance(v, str):
            return _SubStr(v)
        if isinstance(v, list):
            return [wrap(i) for i in v]
        if isinstance(v, dict):
            return {_SubStr(k): wrap(i) for k, i in v.items()}
        return v

    body: dict[str, Any] = {"configuration_space": {"a": [1]}}
    body[_SubStr(key)] = wrap(value)
    _assert_refused_or_clean(oo.PHASE_SESSION_CREATE_AUTO, body)


# ----------------------------------- sent bytes == checked bytes (requests, aiohttp form)

from unittest.mock import patch  # noqa: E402

import requests as _requests  # noqa: E402

_FINALIZE_URL = "https://example.test/api/v1/sessions/s/finalize"


class _ChangingDict(dict):  # type: ignore[type-arg]
    """Serializes harmlessly the first two times, then as a secret."""

    def __init__(self) -> None:
        super().__init__(session_id="s")
        self.n = 0

    def items(self) -> Any:  # type: ignore[override]
        self.n += 1
        return {"session_id": "s" if self.n <= 2 else SECRET}.items()


class _OnceDict(dict):  # type: ignore[type-arg]
    def __init__(self) -> None:
        super().__init__(session_id="s")
        self.n = 0

    def items(self) -> Any:  # type: ignore[override]
        self.n += 1
        if self.n > 1:
            raise ValueError("serializer consumed twice")
        return super().items()


@pytest.fixture
def _gate_off(monkeypatch: pytest.MonkeyPatch) -> Any:
    tg.set_mode("off")
    yield
    tg.set_mode("off")


def test_requests_path_sends_exactly_the_checked_bytes(
    monkeypatch: pytest.MonkeyPatch, _gate_off: Any
) -> None:
    from traigent.runtime.client import RuntimeClient

    monkeypatch.setenv(oo.ENV_VAR, "1")
    sent: list[bytes] = []

    def capture(self: Any, request: Any, **kw: Any) -> Any:
        sent.append(request.body)
        response = _requests.Response()
        response.status_code = 200
        response._content = b"{}"
        response.request = request
        return response

    client = RuntimeClient("p", base_url="https://example.test", api_key="dummy")
    with (
        patch.object(_requests.Session, "send", capture),
        patch(
            "traigent.cloud.client.cloud_backend_egress_disabled", lambda *a, **k: False
        ),
    ):
        client.send("POST", _FINALIZE_URL, _ChangingDict())
    assert len(sent) == 1
    assert SECRET.encode() not in sent[0]


def test_preflight_judges_data_when_it_takes_precedence_over_json(
    monkeypatch: pytest.MonkeyPatch, _gate_off: Any
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    # requests sends ``data`` when it is non-empty, never the (clean) json body.
    with pytest.raises(oo.OptimizationOnlyRefusal):
        tg.preflight_requests(
            "POST",
            _FINALIZE_URL,
            json={"session_id": "s"},
            data=json.dumps({"session_id": SECRET, "x": SECRET}),
        )


def test_mode_off_preflight_serializes_nothing(
    monkeypatch: pytest.MonkeyPatch, _gate_off: Any
) -> None:
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    once = _OnceDict()
    tg.preflight_requests("POST", _FINALIZE_URL, json=once)
    assert once.n == 0
    assert tg.freeze_json_body(once) is once
    prepared = _requests.Request("POST", _FINALIZE_URL, json=once).prepare()
    assert prepared.body == b'{"session_id": "s"}'


def test_mode_off_aiohttp_wrapper_serializes_once_and_untouched() -> None:
    calls: list[int] = []

    def serializer(value: Any) -> str:
        calls.append(1)
        return json.dumps(value)

    class _Plain(_FakeTransport):
        _traigent_oo_explicit = None
        _json_serialize = staticmethod(serializer)

    _Plain.sent = []
    session = tg._build_aiohttp(_Plain)()
    asyncio.run(session._request("POST", _SESSIONS_URL, json={"a": 1}))
    assert calls == []  # the gate added no serialization of its own while off


def test_aiohttp_form_is_encoded_once_and_lying_str_is_refused() -> None:
    class _LyingGrant(str):
        def __eq__(self, other: object) -> bool:
            return other == "refresh_token"

        def __hash__(self) -> int:
            return hash("refresh_token")

    _FakeTransport.sent = []
    session = tg._build_aiohttp(_FakeTransport)()
    url = "https://api.example.test/api/v1/oauth/token"
    with pytest.raises(oo.OptimizationOnlyRefusal):
        asyncio.run(
            session._request("POST", url, data={"grant_type": _LyingGrant(SECRET)})
        )
    assert all(SECRET.encode() not in b for b in _FakeTransport.sent)
    # a legitimate form still goes out as urlencoded bytes
    asyncio.run(session._request("POST", url, data={"grant_type": "refresh_token"}))
    assert _FakeTransport.sent[-1] == b"grant_type=refresh_token"


def test_results_projector_keeps_unknown_content_keys() -> None:
    """A key that hashes/compares as ``config`` but WRITES secret content is judged by what it
    writes: the result projectors are default-deny, so it is refused."""

    class _Key(str):
        def __hash__(self) -> int:
            return hash("config")

        def __eq__(self, other: object) -> bool:
            return other == "config"

    body = {_Key(SECRET): 1}
    with pytest.raises(oo.MetadataContractViolation) as caught:
        oo.project_for_send(oo.PHASE_RESULTS, body, explicit=True)
    assert SECRET not in str(caught.value)


# ------------------------------------------------ fail closed when serialization fails


class _Unserializable(dict):  # type: ignore[type-arg]
    def items(self) -> Any:  # type: ignore[override]
        raise ValueError("cannot serialize")


def test_freeze_refuses_instead_of_returning_the_original(
    monkeypatch: pytest.MonkeyPatch, _gate_off: Any
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    with pytest.raises(oo.OptimizationOnlyRefusal):
        tg.freeze_json_body(_Unserializable(a=1))
    with pytest.raises(oo.OptimizationOnlyRefusal):
        tg.freeze_json_body({"a": float("nan")})


def test_freeze_mode_off_returns_the_object_untouched(
    monkeypatch: pytest.MonkeyPatch, _gate_off: Any
) -> None:
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    value = _Unserializable(a=1)
    assert tg.freeze_json_body(value) is value


def test_runtime_client_refuses_an_unserializable_body(
    monkeypatch: pytest.MonkeyPatch, _gate_off: Any
) -> None:
    from traigent.runtime.client import RuntimeClient

    monkeypatch.setenv(oo.ENV_VAR, "1")
    sent: list[Any] = []
    client = RuntimeClient("p", base_url="https://example.test", api_key="dummy")
    with (
        patch.object(_requests.Session, "send", lambda *a, **k: sent.append(a)),
        patch(
            "traigent.cloud.client.cloud_backend_egress_disabled", lambda *a, **k: False
        ),
        pytest.raises(Exception),  # noqa: B017 - any closed refusal, nothing sent
    ):
        client.send("POST", _FINALIZE_URL, _Unserializable(session_id="s"))
    assert all(SECRET.encode() not in repr(a).encode() for a in sent)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"json": {"a": 1}},
        {"data": {"grant_type": "refresh_token"}},
    ],
)
def test_aiohttp_encoding_failure_refuses_and_sends_nothing(
    kwargs: dict[str, Any],
) -> None:
    def boom(_value: Any) -> str:
        raise ValueError("serializer failed")

    class _Broken(_FakeTransport):
        _json_serialize = staticmethod(boom)

    _Broken.sent = []
    session = tg._build_aiohttp(_Broken)()
    url = (
        "https://api.example.test/api/v1/oauth/token"
        if "data" in kwargs
        else _SESSIONS_URL
    )
    with (
        patch("aiohttp.FormData", side_effect=ValueError("encode failed")),
        pytest.raises(oo.OptimizationOnlyRefusal),
    ):
        asyncio.run(session._request("POST", url, **kwargs))
    assert _Broken.sent == []


def test_every_gated_json_call_site_passes_a_frozen_body() -> None:
    """Census of every ``preflight_requests(..., json=X)`` under traigent/: X must be assigned
    from ``freeze_json_body`` in the same function (or ``kwargs["json"]`` be). Sites known at
    writing: RuntimeClient.send, cloud/auth.py API-key validation, workflow_traces ingest, and
    BackendIntegratedClient publish_best_config / process-record request / example-feature
    upload. External-endpoint calls (nous_auth refresh) are exempt: only the origin is judged. A new json-sending gated call that skips freezing fails here."""
    import ast

    root = Path(oo.__file__).resolve().parents[1]
    problems: list[str] = []
    for path in root.rglob("*.py"):
        if path.name == "transport_gate.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            frozen: set[str] = set()
            frozen_kwargs = False
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                    f = node.value.func
                    name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
                    if name == "freeze_json_body":
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                frozen.add(target.id)
                            elif isinstance(target, ast.Subscript):
                                frozen_kwargs = True
            for node in ast.walk(fn):
                if not (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", getattr(node.func, "attr", ""))
                    == "preflight_requests"
                ):
                    continue
                if any(
                    kw.arg == "external"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                ):
                    continue  # external endpoints: only the origin is judged, never the body
                for kw in node.keywords:
                    if kw.arg == "json" and not (
                        isinstance(kw.value, ast.Name) and kw.value.id in frozen
                    ):
                        problems.append(f"{path.name}:{node.lineno} json not frozen")
                    if kw.arg is None and not frozen_kwargs:
                        problems.append(
                            f"{path.name}:{node.lineno} **kwargs not frozen"
                        )
    assert problems == []


def test_publish_best_config_sends_the_frozen_body(
    monkeypatch: pytest.MonkeyPatch, _gate_off: Any
) -> None:
    import inspect

    from traigent.cloud import backend_client

    source = inspect.getsource(backend_client)
    assert source.count("freeze_json_body(payload)") >= 2
