"""Parity, adversarial and mode-switch tests for ``optimization_only_egress``.

Parity: ``tests/fixtures/optimization_only_parity/expected.json`` holds what the BACKEND's
own projector (TraigentBackend ``2d5cceda5``) returned for ``cases.json``; the SDK port must
agree with it case by case. See that directory's README for how it was generated.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from traigent.utils import optimization_only_egress as oo

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "optimization_only_parity"
CASES: dict[str, dict[str, Any]] = json.loads((FIXTURES / "cases.json").read_text())
EXPECTED: dict[str, dict[str, Any]] = json.loads(
    (FIXTURES / "expected.json").read_text()
)

_RESULT_KEYS = ("metrics", "measures", "summary_stats", "metadata", "error_message")


def _canon(value: Any) -> str:
    """Order-insensitive, NaN-tolerant canonical form for equality."""
    return json.dumps(value, sort_keys=True, allow_nan=True, default=repr)


def _sdk(case: dict[str, Any]) -> Any:
    phase = case["phase"]
    return oo.project_for_phase(phase, case["input"], declared=case["declared"])


@pytest.fixture(autouse=True)
def _clean_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(oo.ENV_VAR, raising=False)
    oo.reset_latch()
    yield
    oo.reset_latch()


def test_pinned_contract_version() -> None:
    assert oo.METADATA_CONTRACT_VERSION == 1
    assert len(CASES) == len(EXPECTED) >= 40


@pytest.mark.parametrize("name", sorted(CASES))
def test_parity_with_backend_projection(name: str) -> None:
    case, expected = CASES[name], EXPECTED[name]
    if "violation" in expected:
        with pytest.raises(oo.MetadataContractViolation):
            _sdk(case)
        return
    got = _sdk(case)
    want = expected["ok"]
    if case["phase"] in ("results", "summary_submission"):
        # the backend projects the five free-form members; the SDK also keeps the rest of
        # the body, so compare exactly the members the backend returned and that were sent
        for key in _RESULT_KEYS:
            if key in case["input"]:
                assert _canon(got[key]) == _canon(want[key]), key
        return
    assert _canon(got) == _canon(want)


@pytest.mark.parametrize("name", sorted(CASES))
def test_input_is_never_mutated(name: str) -> None:
    case = CASES[name]
    before = _canon(case["input"])
    try:
        _sdk(case)
    except oo.MetadataContractViolation:
        pass
    assert _canon(case["input"]) == before


@pytest.mark.parametrize("name", sorted(n for n in CASES if "ok" in EXPECTED[n]))
def test_projection_is_idempotent(name: str) -> None:
    case = CASES[name]
    once = _sdk(case)
    twice = oo.project_for_phase(case["phase"], once, declared=case["declared"])
    # Backend property, ported faithfully: a ``code_revision`` label becomes ``cr_<hash>``,
    # which is not itself revision-shaped, so a second pass tokenizes the token again.
    # Nothing in the SDK projects twice (each attempt reuses one projected copy).
    for tree in (once, twice):
        if isinstance(tree, dict):
            tree.pop("code_revision", None)
    assert _canon(twice) == _canon(once)


def test_code_revision_token_is_not_idempotent_backend_property() -> None:
    once = oo.project_for_phase(
        oo.PHASE_SESSION_CREATE, {"function_name": "f", "code_revision": "my label"}
    )
    twice = oo.project_for_phase(oo.PHASE_SESSION_CREATE, once)
    assert once["code_revision"].startswith("cr_")
    assert twice["code_revision"] != once["code_revision"]
    # hex revisions (what the SDK normally sends) are fixed points
    rev = oo.project_for_phase(
        oo.PHASE_SESSION_CREATE,
        {"function_name": "f", "code_revision": "abc1234-dirty"},
    )
    assert rev["code_revision"] == "abc1234-dirty"


def test_returned_projection_shares_no_state_with_input() -> None:
    payload = copy.deepcopy(CASES["fixture_session_create_typed"]["input"])
    out = oo.project_for_phase(oo.PHASE_SESSION_CREATE, payload)
    out["metadata"]["agent_configuration"]["agents"].clear()
    assert payload["metadata"]["agent_configuration"]["agents"]


# ---------------------------------------------------------------- adversarial


def test_tokenization_prefixes_and_shape() -> None:
    import hashlib
    import re

    out = oo.project_for_phase(
        oo.PHASE_SESSION_CREATE,
        {
            "function_name": "has spaces (x)",
            "agent_key": "My Agent",
            "dataset_id": "Alice's data",
            "evaluator_id": "eval one",
            "evaluator_definition_id": "eval two",
            "code_revision": "free label",
            "metadata": {"evaluation_set": "my set"},
            "dataset_metadata": {"size": 1, "name": "Alice has diabetes"},
            "promotion_policy": {
                "chance_constraints": [
                    {"name": "text name", "threshold": 0.1, "confidence": 0.9}
                ]
            },
        },
    )
    digest = hashlib.sha256(b"has spaces (x)").hexdigest()[:16]
    assert out["function_name"] == f"fn_{digest}"
    assert re.fullmatch(r"ak_[0-9a-f]{16}", out["agent_key"])
    assert re.fullmatch(r"ds_[0-9a-f]{16}", out["dataset_id"])
    assert re.fullmatch(r"ev_[0-9a-f]{16}", out["evaluator_id"])
    assert re.fullmatch(r"ev_[0-9a-f]{16}", out["evaluator_definition_id"])
    assert re.fullmatch(r"cr_[0-9a-f]{16}", out["code_revision"])
    assert re.fullmatch(r"es_[0-9a-f]{16}", out["metadata"]["evaluation_set"])
    assert re.fullmatch(r"dn_[0-9a-f]{16}", out["dataset_metadata"]["name"])
    assert re.fullmatch(
        r"cc_[0-9a-f]{16}", out["promotion_policy"]["chance_constraints"][0]["name"]
    )
    assert oo.token("fn", "x") == oo.token("fn", "x") != oo.token("fn", "y")


def test_identifier_shaped_values_are_kept_verbatim() -> None:
    out = oo.project_for_phase(
        oo.PHASE_SESSION_CREATE,
        {"function_name": "pkg.mod:fn/v-1_a", "agent_key": "a1"},
    )
    assert out["function_name"] == "pkg.mod:fn/v-1_a"
    assert out["agent_key"] == "a1"


@pytest.mark.parametrize("bad", ["", "   ", 5, ["a"], {"a": 1}])
def test_invalid_identity_is_refused_not_rewritten(bad: Any) -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_phase(
            oo.PHASE_SESSION_CREATE, {"function_name": "ok", "agent_key": bad}
        )


def test_free_text_labels_and_nested_unknown_keys_are_dropped() -> None:
    out = oo.project_for_phase(
        oo.PHASE_RESULTS,
        {
            "trial_id": "t1",
            "metrics": {"accuracy": 0.5},
            "metadata": {
                "notes": "Alice has diabetes",
                "nested": {"a": {"b": ["secret"]}},
                "comparability": {"label": "text", "scope": "x y", "n": 3},
            },
            "free_text_top_level": "Alice has diabetes",
            "free_dict_top_level": {"a": "b"},
        },
    )
    blob = json.dumps(out)
    assert "Alice" not in blob and "secret" not in blob and "free_" not in blob
    assert out["metadata"]["comparability"] == {"n": 3}


def test_error_message_collapses_to_closed_vocabulary() -> None:
    def err(text: Any) -> Any:
        return oo.project_for_phase(
            oo.PHASE_RESULTS, {"trial_id": "t", "error_message": text}
        )["error_message"]

    assert err("prompt 'tell me about Alice' failed") == "trial_failed"
    assert err("vendor_rate_limit") == "vendor_rate_limit"
    assert err("") is None
    assert err(None) is None


def test_undeclared_metrics_dropped_declared_kept() -> None:
    body = {"trial_id": "t", "metrics": {"accuracy": 1, "mine": 2, "other": 3}}
    out = oo.project_for_phase(oo.PHASE_RESULTS, body, declared=["mine"])
    assert out["metrics"] == {"accuracy": 1, "mine": 2}
    out = oo.project_for_phase(oo.PHASE_RESULTS, body)
    assert out["metrics"] == {"accuracy": 1}
    # a declared name that is not a valid metric key never widens the set
    out = oo.project_for_phase(oo.PHASE_RESULTS, body, declared=["bad name"])
    assert out["metrics"] == {"accuracy": 1}


def test_registry_supplies_declared_metrics_for_session() -> None:
    oo.register_session_objectives("sess-1", [{"metric": "mine"}, "accuracy"])
    body = {"trial_id": "t", "metrics": {"mine": 2, "other": 3}}
    out = oo.project_for_phase(oo.PHASE_RESULTS, body, session_id="sess-1")
    assert out["metrics"] == {"mine": 2}
    out = oo.project_for_phase(oo.PHASE_RESULTS, body, session_id="unknown-session")
    assert out["metrics"] == {}


def test_registry_is_bounded() -> None:
    for i in range(oo._MAX_TRACKED_SESSIONS + 10):
        oo.register_session_objectives(f"s{i}", ["a"])
    assert oo.declared_for("s0") == ()
    assert oo.declared_for(f"s{oo._MAX_TRACKED_SESSIONS + 9}") == ("a",)


@pytest.mark.parametrize("value", [True, False, math.nan, math.inf, -math.inf, 1e16])
def test_bool_nonfinite_and_huge_numbers_are_not_metrics(value: Any) -> None:
    out = oo.project_for_phase(
        oo.PHASE_RESULTS, {"trial_id": "t", "metrics": {"accuracy": value, "cost": 1.0}}
    )
    assert out["metrics"] == {"cost": 1.0}


def test_bool_is_not_an_int_for_strict_fields() -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_phase(
            oo.PHASE_RESULTS, {"trial_id": "t", "metadata": {"trial_number": True}}
        )


def test_oversized_structures_are_refused() -> None:
    ids = [f"p{i}".ljust(128, "x") for i in range(256)]
    big = {
        "function_name": "f",
        "metadata": {"agent_configuration": {"agents": {"a": {"parameter_keys": ids}}}},
    }
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_phase(oo.PHASE_SESSION_CREATE, big)
    rows = [{"example_id": f"e{i}", "metrics": {"accuracy": 1.0}} for i in range(2000)]
    # lists are truncated, never an unbounded send
    out = oo.project_for_phase(oo.PHASE_RESULTS, {"trial_id": "t", "measures": rows})
    assert len(out["measures"]) <= 1000
    deep: Any = "leaf"
    for _ in range(40):
        deep = {"a": deep}
    out = oo.project_for_phase(oo.PHASE_RESULTS, {"trial_id": "t", "metadata": deep})
    assert out["metadata"] == {}


def test_contract_version_mismatch_is_refused() -> None:
    for version in (0, 2, True, "1"):
        with pytest.raises(oo.MetadataContractViolation):
            oo.project_for_phase(
                oo.PHASE_FINALIZE, {oo.CONTRACT_VERSION_KEY: version, "reason": "x"}
            )
    out = oo.project_for_phase(
        oo.PHASE_FINALIZE, {oo.CONTRACT_VERSION_KEY: 1, "reason": "x"}
    )
    assert oo.CONTRACT_VERSION_KEY not in out


def test_finalize_metadata_is_emptied_and_reason_closed() -> None:
    out = oo.project_for_phase(
        oo.PHASE_FINALIZE,
        {
            "reason": "free text here",
            "metadata": {"a": "b"},
            "include_full_history": True,
        },
    )
    assert out == {
        "reason": oo.DEFAULT_FINALIZE_REASON,
        "metadata": {},
        "include_full_history": True,
    }


def test_endpoint_is_never_kept_for_external_evaluator() -> None:
    out = oo.project_for_phase(
        oo.PHASE_SESSION_CREATE,
        {
            "function_name": "f",
            "optimization_strategy": {
                "execution_options": {
                    "external_evaluator": {"endpoint": "https://u:p@h.example/x"}
                }
            },
        },
    )
    blob = json.dumps(out)
    assert "h.example" not in blob and "u:p" not in blob
    assert "endpoint_redacted" in blob


def test_session_shape_is_classified_or_refused() -> None:
    legacy = oo.project_for_phase(
        oo.PHASE_SESSION_CREATE_AUTO, {"problem_statement": "free text", "metadata": {}}
    )
    assert legacy["problem_statement"] == oo.DEFAULT_PROBLEM_STATEMENT
    typed = oo.project_for_phase(
        oo.PHASE_SESSION_CREATE_AUTO, {"function_name": "f", "configuration_space": {}}
    )
    assert typed["function_name"] == "f"
    for ambiguous in ({}, {"function_name": "f", "problem_statement": "p"}, []):
        with pytest.raises(oo.MetadataContractViolation):
            oo.project_for_phase(oo.PHASE_SESSION_CREATE_AUTO, ambiguous)


def test_unknown_phase_fails_locally() -> None:
    with pytest.raises(ValueError, match="unknown"):
        oo.project_for_phase("not_a_phase", {})


def test_next_trial_projects_previous_results() -> None:
    out = oo.project_for_phase(
        oo.PHASE_NEXT_TRIAL,
        {
            "session_id": "s1",
            "previous_results": [
                {
                    "trial_id": "t1",
                    "metrics": {"accuracy": 1, "zz": 2},
                    "error_message": "x y",
                }
            ],
            "request_metadata": {"k": "free text value", "n": 1},
        },
    )
    assert out["previous_results"][0]["metrics"] == {"accuracy": 1}
    assert out["previous_results"][0]["error_message"] == "trial_failed"
    assert out["request_metadata"] == {"n": 1}


def test_intermediate_report_is_closed_scalars_only() -> None:
    ok = {
        "session_id": "s",
        "trial_id": "t",
        "running_score": 0.5,
        "objective_name": "accuracy",
    }
    assert oo.project_for_phase(oo.PHASE_INTERMEDIATE_REPORT, ok) == ok
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_phase(
            oo.PHASE_INTERMEDIATE_REPORT, {**ok, "objective_name": "free text"}
        )


def test_embedded_examples_are_refused_when_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"function_name": "f", "examples": [{"input": "text"}]}
    assert oo.project_for_send(oo.PHASE_SESSION_CREATE, body) is body  # mode off
    monkeypatch.setenv(oo.ENV_VAR, "1")
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.project_for_send(oo.PHASE_SESSION_CREATE, body)
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.project_for_send(
            oo.PHASE_SESSION_CREATE,
            {"function_name": "f", "metadata": {"dataset": {"examples": ["text"]}}},
        )
    # a knob that merely carries the word "examples" is configuration, not content
    ok = oo.project_for_send(
        oo.PHASE_SESSION_CREATE,
        {"function_name": "f", "configuration_space": {"examples": {"choices": ["a"]}}},
    )
    assert ok["configuration_space"] == {"examples": {"choices": ["a"]}}


# ---------------------------------------------------------------- mode switch


def test_unset_is_legacy_and_send_helpers_are_identity() -> None:
    assert oo.is_active() is False
    payload = {"problem_statement": "free text", "anything": object()}
    assert oo.project_for_send(oo.PHASE_SESSION_CREATE_AUTO, payload) is payload
    oo.refuse_if_active("dataset_upload")  # no raise
    oo.refuse_embedded_examples({"examples": ["x"]})  # no raise


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " On "])
def test_env_truthy(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(oo.ENV_VAR, raw)
    assert oo.is_active() is True


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", " Off "])
def test_env_falsy(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(oo.ENV_VAR, raw)
    assert oo.is_active() is False


@pytest.mark.parametrize("raw", ["maybe", "2", "enabled", "tru"])
def test_env_invalid_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, raw)
    with pytest.raises(oo.OptimizationOnlyConfigError, match=oo.ENV_VAR):
        oo.is_active()


def test_blank_env_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "  ")
    assert oo.is_active() is False


def test_explicit_argument_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    assert oo.is_active(False) is False
    monkeypatch.setenv(oo.ENV_VAR, "0")
    assert oo.is_active(True) is True
    monkeypatch.delenv(oo.ENV_VAR)
    assert oo.is_active(True) is True
    assert oo.is_active(None) is False


def test_explicit_argument_must_be_bool_or_none() -> None:
    with pytest.raises(oo.OptimizationOnlyConfigError):
        oo.is_active("sometimes")  # type: ignore[arg-type]


def test_explicit_of_ignores_non_bool_attributes() -> None:
    class Mockish:
        optimization_only = object()

    class Real:
        optimization_only = True

    assert oo.explicit_of(Mockish()) is None
    assert oo.explicit_of(Real()) is True
    assert oo.explicit_of(object()) is None


CLOSED_BODY = {
    "success": False,
    "error": "optimization_only",
    "error_code": "optimization_only",
    "message": "This workspace is limited to optimization; this operation is not available.",
}


SCOPE_A = "https://backend.example.test|" + "a" * 64
SCOPE_B = "https://backend.example.test|" + "b" * 64


def test_closed_403_latches_the_mode_for_that_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "0")
    assert oo.is_active(scope=SCOPE_A) is False
    assert oo.note_response(403, json.dumps(CLOSED_BODY), SCOPE_A) is True
    assert oo.is_latched(SCOPE_A) is True
    assert oo.is_active(scope=SCOPE_A) is True
    # even an explicit opt-out cannot undo a latch for the SAME credential
    assert oo.is_active(False, scope=SCOPE_A) is True
    assert oo.is_active(oo.Policy(False, SCOPE_A)) is True
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.refuse_if_active("dataset_upload", explicit=oo.Policy(False, SCOPE_A))


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (403, {"error_code": "forbidden"}),
        (403, "not json"),
        (403, None),
        (401, CLOSED_BODY),
        (500, CLOSED_BODY),
        (403, [CLOSED_BODY]),
        (403, b"\xff\xfe"),
    ],
)
def test_other_responses_do_not_latch(status: int, body: Any) -> None:
    assert oo.note_response(status, body, SCOPE_A) is False
    assert oo.is_latched(SCOPE_A) is False


def test_refusal_for_one_credential_does_not_restrict_another() -> None:
    """Review finding 6: the latch is bound to (origin, credential fingerprint)."""
    oo.note_response(403, json.dumps(CLOSED_BODY), SCOPE_A)
    assert oo.is_active(oo.Policy(False, SCOPE_B)) is False  # B's explicit False stands
    assert oo.is_active(oo.Policy(None, SCOPE_B)) is False
    assert oo.is_active(oo.Policy(False, SCOPE_A)) is True
    assert oo.is_active(False) is False  # unattributed calls are never latched
    # a response with no credential to attribute it to latches nothing
    assert oo.note_response(403, json.dumps(CLOSED_BODY), None) is True
    assert oo.is_active(oo.Policy(False, None)) is False


def test_credential_scope_binds_origin_and_credential_without_storing_the_key() -> None:
    key_a, key_b = "uk_" + "a" * 43, "uk_" + "b" * 43
    a = oo.credential_scope("https://H.example/api/v1/sessions", {"X-API-Key": key_a})
    assert a == oo.credential_scope("https://h.example", {"x-api-key": key_a})
    assert a != oo.credential_scope("https://h.example", {"X-API-Key": key_b})
    assert a != oo.credential_scope("https://other.example", {"X-API-Key": key_a})
    assert key_a not in (a or "") and key_a[3:] not in (a or "")
    assert oo.credential_scope("https://h.example", {"Content-Type": "x"}) is None
    assert oo.credential_scope("https://h.example", None) is None
    bearer = oo.credential_scope("https://h.example", {"Authorization": "Bearer tok1"})
    assert bearer == oo.credential_scope("https://h.example", {"Authorization": "tok1"})


def test_jwt_takes_precedence_like_the_backend_authenticator() -> None:
    """Review 2 / P2: the backend tries JWT before the API key, so the scope follows it."""
    url = "https://h.example"
    jwt_a, jwt_b, key = "jwt-a." + "a" * 30, "jwt-b." + "b" * 30, "uk_" + "k" * 43
    a_with_key = oo.credential_scope(
        url, {"Authorization": f"Bearer {jwt_a}", "X-API-Key": key}
    )
    b_with_key = oo.credential_scope(
        url, {"Authorization": f"Bearer {jwt_b}", "X-API-Key": key}
    )
    assert a_with_key != b_with_key  # same API key, different JWTs: no collision
    # an irrelevant API-key header neither changes nor loses a JWT's scope
    jwt_only = oo.credential_scope(url, {"Authorization": f"Bearer {jwt_a}"})
    assert a_with_key == jwt_only
    assert (
        oo.credential_scope(
            url, {"Authorization": f"Bearer {jwt_a}", "X-API-Key": "uk_" + "z" * 43}
        )
        == jwt_only
    )
    # API-key-only and JWT-only clients stay distinct, and the key alone still scopes
    assert oo.credential_scope(url, {"X-API-Key": key}) not in (jwt_only, None)
    oo.note_response(403, json.dumps(CLOSED_BODY), a_with_key)
    assert oo.is_active(oo.Policy(False, jwt_only)) is True
    assert oo.is_active(oo.Policy(False, b_with_key)) is False


def test_scope_resolution_is_free_when_nothing_is_latched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review 2 / P3: mode off, nothing latched: no header resolution, no hashing."""
    calls: list[str] = []

    class Auth:
        def configured_credential_headers(self) -> dict[str, str]:
            calls.append("configured")
            return {"X-API-Key": "uk_" + "a" * 43}

    class Client:
        optimization_only = None
        auth = Auth()
        api_base_url = "https://h.example/api/v1"

    monkeypatch.setattr(
        oo, "credential_scope", lambda *a, **k: calls.append("hash") or None
    )
    for policy in (
        oo.policy_of(Client(), {"X-API-Key": "uk_" + "b" * 43}),
        oo.policy_of(Client()),
        oo.policy_from_headers("https://h.example", {"X-API-Key": "k"}),
        oo.policy_from_environment("https://h.example"),
        oo.policy_of(type("C", (), {"optimization_only": True, "auth": Auth()})()),
    ):
        assert policy.scope is None
    assert calls == []
    assert oo.scope_required(None) is False and oo.scope_required(True) is False
    oo.note_response(403, json.dumps(CLOSED_BODY), SCOPE_A)
    assert oo.scope_required(None) is True and oo.scope_required(True) is False
    oo.policy_of(Client())
    assert calls == ["hash", "configured", "hash"]


def test_latch_is_thread_safe_across_credentials() -> None:
    import threading

    scopes = [f"https://h.example|{i:064d}" for i in range(64)]
    threads = [
        threading.Thread(
            target=oo.note_response, args=(403, json.dumps(CLOSED_BODY), s)
        )
        for s in scopes
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(oo.is_latched(s) for s in scopes)
    assert not oo.is_latched("https://h.example|" + "f" * 64)


def test_closed_403_accepts_bytes_and_error_field() -> None:
    assert oo.is_closed_refusal(403, json.dumps(CLOSED_BODY).encode())
    assert oo.is_closed_refusal(403, {"error": "optimization_only"})


def test_session_id_from_url() -> None:
    assert oo.session_id_from_url("https://h/api/v1/sessions/abc/results") == "abc"
    assert oo.session_id_from_url("https://h/api/v1/sessions/abc?x=1") == "abc"
    assert oo.session_id_from_url("https://h/api/v1/other") is None


def test_closed_refusal_code_is_registered_with_session_types() -> None:
    from traigent.core.session_types import (
        CLOSED_REFUSAL_CODES,
        SessionCreationFailureDetail,
    )

    assert "optimization_only" in CLOSED_REFUSAL_CODES
    detail = SessionCreationFailureDetail.from_http_response(
        403, json.dumps(CLOSED_BODY)
    )
    assert detail.is_closed_refusal
    assert detail.safe_error_code == "optimization_only"


def _closed_resp_kinds() -> tuple[bytes, dict[str, str], str]:
    return (
        json.dumps(CLOSED_BODY).encode(),
        {"X-API-Key": "uk_" + "c" * 43},
        "https://h.example/api/v1/x",
    )


def test_transport_gate_latches_the_requests_credential_on_closed_403() -> None:
    import asyncio

    from traigent.labels.transport_gate import _latch_aiohttp

    class Resp:
        def __init__(self, status: int, body: bytes) -> None:
            self.status = status
            self._body = body

        async def read(self) -> bytes:
            return self._body

    body, headers, url = _closed_resp_kinds()
    scope = oo.credential_scope(url, headers)
    asyncio.run(_latch_aiohttp(Resp(500, body), url, headers))
    asyncio.run(_latch_aiohttp(Resp(403, b'{"error_code": "forbidden"}'), url, headers))
    assert oo.is_latched(scope) is False
    asyncio.run(_latch_aiohttp(Resp(403, body), url, headers))
    assert oo.is_latched(scope) is True
    assert oo.is_active(oo.Policy(False, scope)) is True
    other = oo.credential_scope(url, {"X-API-Key": "uk_" + "d" * 43})
    assert oo.is_active(oo.Policy(False, other)) is False


class _FakeRequest:
    def __init__(self, url: str, headers: dict[str, str]) -> None:
        self.url = url
        self.headers = headers
        self.method = "POST"
        self.content = b""


class _FakeHttpxResponse:
    def __init__(self, request: _FakeRequest, status: int, body: bytes) -> None:
        self.request = request
        self.status_code = status
        self._body = body
        self._read = False

    @property
    def content(self) -> bytes:
        if not self._read:
            raise RuntimeError("ResponseNotRead")
        return self._body

    def read(self) -> bytes:
        self._read = True
        return self._body

    async def aread(self) -> bytes:
        return self.read()

    def close(self) -> None:
        return None


def test_httpx_sync_and_async_wrappers_latch_on_closed_403() -> None:
    """Review finding 5: HTTPX responses used to be returned without being observed."""
    import asyncio

    from traigent.labels import transport_gate as gate

    body, headers, url = _closed_resp_kinds()
    scope = oo.credential_scope(url, headers)

    class SyncBase:
        def send(self, request: Any, **kwargs: Any) -> Any:
            return _FakeHttpxResponse(request, 403, body)

    class AsyncBase:
        async def send(self, request: Any, **kwargs: Any) -> Any:
            return _FakeHttpxResponse(request, 403, body)

    sync_client = gate._build_httpx_sync(SyncBase)()
    sync_client.send(_FakeRequest(url, headers))
    assert oo.is_latched(scope) is True

    oo.reset_latch()
    async_client = gate._build_httpx_async(AsyncBase)()
    asyncio.run(async_client.send(_FakeRequest(url, headers)))
    assert oo.is_latched(scope) is True


def test_module_level_requests_responses_are_observed() -> None:
    """Review finding 5: ``requests.post(...)`` bypasses the gated session entirely."""
    from traigent.labels.transport_gate import observe_requests_response

    body, headers, url = _closed_resp_kinds()

    class Resp:
        status_code = 403
        content = body
        request = _FakeRequest(url, headers)

    observe_requests_response(Resp())
    assert oo.is_latched(oo.credential_scope(url, headers)) is True


def test_backend_client_transient_check_observes_the_response() -> None:
    from traigent.cloud.backend_client import _raise_for_transient_backend_response

    body, headers, url = _closed_resp_kinds()

    class Resp:
        status_code = 403
        content = body
        text = ""
        request = _FakeRequest(url, headers)

    _raise_for_transient_backend_response(Resp(), "Best-config fetch")
    assert oo.is_latched(oo.credential_scope(url, headers)) is True


# ---------------------------------------------------------------- Director (finding 1)

DIRECTOR_SESSIONS = "/api/v1/director/sessions"
DIRECTOR_TURN = "/api/v1/director/sessions/s1/turn"


def test_director_turn_with_free_text_notes_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    leaking = {
        "session_revision": 1,
        "report": {
            "instruction_id": "i1",
            "status": "completed",
            "notes": "Alice has diabetes",
        },
    }
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.assert_director_request(DIRECTOR_TURN, leaking)


def test_director_allowlist_is_enforced_at_every_depth_and_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "1")
    ok_turn = {
        "session_revision": 2,
        "intent": "continue",
        "report": {
            "instruction_id": "i1",
            "status": "completed",
            "run_id": "run_1",
            "result": {
                "protocol": "p1",
                "judge": "j1",
                "orders": "both",
                "verdicts": [{"example_id": "e1", "verdict": "a"}],
            },
        },
        "client_report": {
            "validity_checks": [
                {"check": "c1", "status": "ok", "confidence_label": "high"}
            ]
        },
    }
    oo.assert_director_request(DIRECTOR_TURN, ok_turn)
    oo.assert_director_request(
        DIRECTOR_SESSIONS,
        {
            "project_id": "p1",
            "workflow_kind": "w",
            "run_ids": ["r1"],
            "environment": "dev",
        },
    )
    bad: list[tuple[str, dict[str, Any]]] = [
        (DIRECTOR_TURN, {**ok_turn, "message": "x"}),
        (DIRECTOR_TURN, {"report": {**ok_turn["report"], "detail": "x"}}),
        (
            DIRECTOR_TURN,
            {"report": {"result": {**ok_turn["report"]["result"], "rationale": "x"}}},
        ),
        (
            DIRECTOR_TURN,
            {"report": {"result": {"verdicts": [{"example_id": "e1", "note": "x"}]}}},
        ),
        (
            DIRECTOR_TURN,
            {"client_report": {"validity_checks": [{"check": "free text"}]}},
        ),
        (DIRECTOR_TURN, {"report": {"status": "Alice has diabetes"}}),
        (DIRECTOR_SESSIONS, {"project_id": "p1", "goal": "x"}),
        (DIRECTOR_SESSIONS, {"project_id": {"nested": "x y"}}),
        ("/api/v1/director/other", {"a": 1}),
    ]
    for path, body in bad:
        with pytest.raises(oo.OptimizationOnlyRefusal):
            oo.assert_director_request(path, body)


def test_director_check_is_a_noop_when_mode_off() -> None:
    oo.assert_director_request(DIRECTOR_TURN, {"report": {"notes": "free text"}})


# ----------------------------------------------- legacy session: allowlisted metadata only

_LEGACY_MARKERS = (
    "ROW_MARKER_a1",
    "EXAMPLE_MARKER_b2",
    "PATH_MARKER_c3",
    "NOTE_MARKER_d4",
)


def _legacy_body(**dataset: Any) -> dict[str, Any]:
    return {
        "problem_statement": "p",
        "search_space": {"model": ["a", "b"]},
        "optimization_config": {"algorithm": "grid", "max_trials": 3},
        "dataset": dataset,
    }


def test_legacy_session_dataset_keeps_only_allowlisted_metadata() -> None:
    body = _legacy_body(
        examples=[{"input": "ROW_MARKER_a1"}],
        rows=["EXAMPLE_MARKER_b2"],
        path="/data/PATH_MARKER_c3.jsonl",
        description="NOTE_MARKER_d4",
        metadata={"size": 7, "privacy_mode": True, "notes": "NOTE_MARKER_d4"},
    )
    out = oo.project_legacy_session_create(body)
    blob = json.dumps(out)
    assert not [m for m in _LEGACY_MARKERS if m in blob], blob
    assert set(out["dataset"]) <= {"name", "size", "metadata", "examples"}
    assert out["dataset"]["metadata"] == {"size": 7, "privacy_mode": True}
    # the projection of a projection is itself: the transport's "unchanged" check holds
    assert oo.project_legacy_session_create(out) == out


def test_legacy_session_dataset_metadata_free_text_values_are_dropped() -> None:
    out = oo.project_legacy_session_create(
        _legacy_body(metadata={"size": 3, "type": "NOTE_MARKER_d4", "name": "my set"})
    )
    blob = json.dumps(out)
    assert "NOTE_MARKER_d4" not in blob
    assert out["dataset"]["metadata"]["size"] == 3
    assert "type" not in out["dataset"]["metadata"]


def test_legacy_session_dataset_nonempty_examples_never_pass() -> None:
    out = oo.project_legacy_session_create(_legacy_body(examples=["ROW_MARKER_a1"]))
    assert "ROW_MARKER_a1" not in json.dumps(out)
    assert out["dataset"].get("examples", []) == []


@pytest.mark.parametrize(
    "field, value",
    [
        ("dataset", "ROW_MARKER_a1"),
        ("dataset", ["ROW_MARKER_a1"]),
        ("dataset", None),
        ("optimization_config", "NOTE_MARKER_d4"),
        ("optimization_config", ["NOTE_MARKER_d4"]),
        ("optimization_config", None),
    ],
)
def test_legacy_session_non_dict_sections_are_refused_before_transport(
    field: str, value: Any
) -> None:
    body = _legacy_body()
    body[field] = value
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_legacy_session_create(body)
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_for_phase(oo.PHASE_SESSION_CREATE_AUTO, body)


def test_legacy_session_dataset_metadata_non_dict_is_refused() -> None:
    with pytest.raises(oo.MetadataContractViolation):
        oo.project_legacy_session_create(_legacy_body(metadata="NOTE_MARKER_d4"))


def test_legacy_session_create_with_allowed_fields_still_succeeds() -> None:
    out = oo.project_legacy_session_create(
        _legacy_body(examples=[], metadata={"size": 100}, name="my_set", size=100)
    )
    assert out["dataset"] == {
        "examples": [],
        "metadata": {"size": 100},
        "name": "my_set",
        "size": 100,
    }
    assert out["search_space"] == {"model": ["a", "b"]}
    assert out["optimization_config"]["max_trials"] == 3
