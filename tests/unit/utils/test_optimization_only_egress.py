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


def test_closed_403_latches_the_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(oo.ENV_VAR, "0")
    assert oo.is_active() is False
    assert oo.note_response(403, json.dumps(CLOSED_BODY)) is True
    assert oo.is_latched() is True
    assert oo.is_active() is True
    assert oo.is_active(False) is True  # even an explicit opt-out cannot undo a latch
    with pytest.raises(oo.OptimizationOnlyRefusal):
        oo.refuse_if_active("dataset_upload", explicit=False)


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
    assert oo.note_response(status, body) is False
    assert oo.is_latched() is False


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


def test_transport_gate_latches_on_closed_403_body() -> None:
    import asyncio

    from traigent.labels.transport_gate import _latch_aiohttp

    class Resp:
        def __init__(self, status: int, body: bytes) -> None:
            self.status = status
            self._body = body

        async def read(self) -> bytes:
            return self._body

    asyncio.run(_latch_aiohttp(Resp(500, json.dumps(CLOSED_BODY).encode())))
    asyncio.run(_latch_aiohttp(Resp(403, b'{"error_code": "forbidden"}')))
    assert oo.is_latched() is False
    asyncio.run(_latch_aiohttp(Resp(403, json.dumps(CLOSED_BODY).encode())))
    assert oo.is_latched() is True
