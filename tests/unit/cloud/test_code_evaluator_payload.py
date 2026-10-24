"""Whole-payload canary tests for ``evaluator_code_digest`` (TraigentSchema #610).

The new field is a digest of the scorer's code. Canary tokens planted in the
scorer's source (comment, string literal, docstring, identifier), in dataset
rows and in the evaluator-id aliases must never reach the outgoing create-session
body. Content-less mode never carries the field.

Pre-existing hits (objective labels, ``function_name``) are NOT silently
allow-listed: they are strict ``xfail`` tests that reference spec section 11 of
the code-evaluator-identity design, so the gap stays visible and the test fails
(XPASS-strict) the day it is closed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from traigent.cloud import content_less as cl
from traigent.cloud.api_operations import ApiOperations
from traigent.cloud.client import TraigentCloudClient
from traigent.cloud.models import SessionCreationRequest
from traigent.evaluators.base import Dataset, EvaluationExample
from traigent.utils.artifact_fingerprints import (
    build_artifact_fingerprints,
    effective_evaluator_code_identity,
)

CANARY_COMMENT = "cnryComment_Kq4V-8nTz2Lm"
CANARY_LITERAL = "cnryLiteral_Bw7R-1xYd5Hc"
CANARY_DOC = "cnryDocstring_Fp9S-3gJu6Ne"
CANARY_IDENT = "cnryIdent_Zt2M-6aVq8Rb"
CANARY_ALIAS = "cnryAlias_Hx5D-9wKc1Pf"
CANARY_ROW = "cnryRow_Qe3N-7yLs4Vj"
CANARY_OBJECTIVE = "cnryObjective_Ua6T-2dMo9Gz"  # pragma: allowlist secret
CANARIES = (
    CANARY_COMMENT,
    CANARY_LITERAL,
    CANARY_DOC,
    CANARY_IDENT,
    CANARY_ALIAS,
    CANARY_ROW,
)

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def scorer(output, expected):
    """cnryDocstring_Fp9S-3gJu6Ne"""
    # cnryComment_Kq4V-8nTz2Lm
    cnryIdent_Zt2M_6aVq8Rb = "cnryLiteral_Bw7R-1xYd5Hc"
    return 1.0 if output == expected else len(cnryIdent_Zt2M_6aVq8Rb) * 0.0


#: The create-session body's top-level keys when origin is ``code`` (typed/v2 shape).
ALLOWED_TOP_LEVEL = frozenset(
    {
        "function_name",
        "configuration_space",
        "objectives",
        "dataset_metadata",
        "max_trials",
        "metadata",
        "user_id",
        "billing_tier",
        "artifact_fingerprints",
        "fingerprint_meta",
        "identity_version",
        "agent_id",
        "agent_id_source",
        "dataset_id",
        "dataset_id_source",
        "evaluator_id_source",
        "artifact_versions",
        "evaluator_code_digest",
        "optimization_strategy",
        "task_type",
        "agent_key",
        "budget",
        "constraints",
        "default_config",
        "promotion_policy",
        "smart_pruning",
        "tvl_governance",
        "warm_start_from",
        "run_title",
        "run_description",
        "cost_limit",
        "content_identity",
    }
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


def _digest() -> str:
    ident = effective_evaluator_code_identity(
        scoring_function=scorer, objectives=["accuracy"]
    )
    assert ident["origin"] == "code"
    return str(ident["digest"])


def _request(**overrides: Any) -> SessionCreationRequest:
    dataset = Dataset(
        [EvaluationExample({"q": CANARY_ROW}, CANARY_ROW)], name="canary-dataset"
    )
    fingerprints = build_artifact_fingerprints(
        dataset=dataset, scoring_function=scorer, configuration_space={"t": [0.1]}
    )
    base: dict[str, Any] = {
        "function_name": "pkg.mod.fn",
        "configuration_space": {"temperature": [0.1, 0.9]},
        "objectives": ["accuracy"],
        "dataset_metadata": {"size": 1},
        "max_trials": 2,
        "artifact_fingerprints": fingerprints["artifact_fingerprints"],
        "fingerprint_meta": fingerprints["fingerprint_meta"],
        "evaluator_code_digest": _digest(),
    }
    base.update(overrides)
    return SessionCreationRequest(**base)


def _ops() -> ApiOperations:
    client = MagicMock()
    client.backend_config.api_base_url = "https://api.example.test"
    return ApiOperations(client)


def _cloud_client_payload(request: SessionCreationRequest) -> dict[str, Any]:
    client = TraigentCloudClient.__new__(TraigentCloudClient)
    client.auth = MagicMock(get_owner_fingerprint=MagicMock(return_value=None))
    return client._serialize_session_request(request)


def _bodies(request: SessionCreationRequest) -> dict[str, dict[str, Any]]:
    return {
        "api_operations": _ops()._build_typed_session_payload(request, 2),
        "cloud_client": _cloud_client_payload(request),
    }


@pytest.mark.parametrize("serializer", ["api_operations", "cloud_client"])
def test_code_origin_body_is_closed_and_canary_free(serializer: str) -> None:
    body = _bodies(_request())[serializer]
    text = json.dumps(body, default=str)
    assert [c for c in CANARIES if c in text] == []
    assert DIGEST_RE.match(body["evaluator_code_digest"])
    assert body["identity_version"] == 2
    for alias in ("evaluator_id", "evaluator_definition_id", "evaluator_ref"):
        assert alias not in body
    assert body["evaluator_id_source"] == "unknown"
    assert set(body) <= ALLOWED_TOP_LEVEL, sorted(set(body) - ALLOWED_TOP_LEVEL)
    # the digest differs from the legacy fp1 evaluator value it sits beside
    fp1 = body["artifact_fingerprints"]["evaluator"]
    assert body["evaluator_code_digest"] != "sha256:" + fp1.removeprefix("fp1:")


@pytest.mark.parametrize("alias", ["evaluator_id", "evaluator_definition_id"])
@pytest.mark.parametrize("serializer", ["api_operations", "cloud_client"])
def test_declared_alias_wins_and_digest_is_not_sent(
    serializer: str, alias: str
) -> None:
    body = _bodies(_request(**{alias: CANARY_ALIAS}))[serializer]
    assert "evaluator_code_digest" not in body
    assert body["evaluator_id_source"] != "unknown"


def test_no_digest_field_when_not_set() -> None:
    body = _bodies(_request(evaluator_code_digest=None))["api_operations"]
    assert "evaluator_code_digest" not in body


@pytest.mark.parametrize(
    "bad",
    [
        "sha256:" + "A" * 64,
        "sha256:" + "a" * 63,
        "fp1:" + "a" * 64,
        "a" * 64,
        "sha256:" + "a" * 64 + "x",
    ],
)
def test_malformed_digest_is_refused_at_construction(bad: str) -> None:
    with pytest.raises(ValueError, match="evaluator_code_digest"):
        _request(evaluator_code_digest=bad)


def test_legacy_shape_never_carries_the_field() -> None:
    legacy = _ops()._build_legacy_session_payload(_request(), 2)
    assert "evaluator_code_digest" not in legacy
    assert CANARY_ROW not in json.dumps(legacy, default=str)


def test_content_less_body_has_no_field(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRAIGENT_CONTENT_LESS", "1")
    monkeypatch.setenv("TRAIGENT_ENVIRONMENT", "staging")
    request = _request()
    body = _ops()._build_session_payload(request, 2)
    assert body["capabilities"] == ["content_less_v1"]
    assert "evaluator_code_digest" not in body
    assert "evaluator_code_digest" in cl.LEGACY_TOP_LEVEL
    # explicit options path, through the cloud-client serializer too
    client_body = _cloud_client_payload(request)
    cl.prepare_payload(client_body, request, None)
    assert "evaluator_code_digest" not in client_body


def test_content_less_option_drops_the_field_even_when_set() -> None:
    request = _request(content_less=cl.ContentLessOptions(enabled=True))
    body = _ops()._build_session_payload(request, 2)
    assert "evaluator_code_digest" not in body
    assert DIGEST_RE.match(request.evaluator_code_digest)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "pre-existing gap, code-evaluator-identity spec section 11 item 1: a custom "
        "objective label travels verbatim in content-less mode"
    ),
)
def test_content_less_objective_label_does_not_cross() -> None:
    request = _request(
        objectives=[CANARY_OBJECTIVE],
        content_less=cl.ContentLessOptions(enabled=True),
    )
    body = _ops()._build_session_payload(request, 2)
    assert CANARY_OBJECTIVE not in json.dumps(body, default=str)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "pre-existing, code-evaluator-identity spec section 11 / section 4: "
        "objective names are customer-chosen labels already sent in standard mode"
    ),
)
def test_standard_body_has_no_objective_label() -> None:
    body = _ops()._build_typed_session_payload(
        _request(objectives=[CANARY_OBJECTIVE]), 2
    )
    assert CANARY_OBJECTIVE not in json.dumps(body, default=str)
