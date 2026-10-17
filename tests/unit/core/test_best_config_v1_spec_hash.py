"""One spec_hash rule for ``traigent.best_config.v1``, shared with the Backend.

The expected values are the language-neutral vectors vendored from TraigentSchema
(``best_config_schema.json`` ``x-traigent-spec-hash``). The Backend has applied this rule
since #819; before this rule the SDK hashed provenance and unfiltered validation, so a real
publish raised ``CloudBestConfigIntegrityError`` after the Backend had committed it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from traigent.core.best_config_runtime import (
    compute_spec_hash,
    resolve_cloud_cache_best_config,
    load_best_config_spec,
    write_cloud_cache_best_config,
    write_repo_best_config,
)
from traigent.core.config_state_manager import (
    CloudBestConfigIntegrityError,
    ConfigStateManager,
)
from traigent.utils.exceptions import ConfigurationError

_VECTORS = (
    Path(__file__).resolve().parents[2]
    / "fixtures/contracts/schema_083eb11/best_config_v1_spec_hash_vectors.json"
)
_PINNED_SHA256 = "dd72aa11a751c5d6cdf084ab5ba03ca203df73d9ce6ebb59ad86b3bfa6e98e00"


def _cases() -> list[dict[str, Any]]:
    return json.loads(_VECTORS.read_text(encoding="utf-8"))["cases"]


def _refusal_cases() -> list[dict[str, Any]]:
    return json.loads(_VECTORS.read_text(encoding="utf-8"))["refusal_cases"]


def _case(name: str) -> dict[str, Any]:
    return next(case for case in _cases() if case["name"] == name)


def test_vendored_vectors_are_the_pinned_schema_bytes() -> None:
    assert hashlib.sha256(_VECTORS.read_bytes()).hexdigest() == _PINNED_SHA256


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_compute_spec_hash_reproduces_every_vector(case: dict[str, Any]) -> None:
    request = case["request"]
    assert (
        compute_spec_hash(request["spec"], environment=request.get("environment"))
        == case["spec_hash"]
    )


def test_compute_spec_hash_does_not_mutate_its_input() -> None:
    spec = copy.deepcopy(_case("validation_nested_exclusions")["request"]["spec"])
    before = copy.deepcopy(spec)
    compute_spec_hash(spec)
    assert spec == before


def test_a_spec_envelope_other_than_v1_has_no_rule_here() -> None:
    spec = copy.deepcopy(_case("provenance_present")["request"]["spec"])
    spec["schema_version"] = "traigent.best_config.v2"
    with pytest.raises(ConfigurationError, match="schema_version"):
        compute_spec_hash(spec)


def _backend_returned(case_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """What the Backend answers for a request: its normalized spec and the vector digest."""
    case = _case(case_name)
    request_spec = copy.deepcopy(case["request"]["spec"])
    returned = copy.deepcopy(case["hashed_payload"])
    if "provenance" in request_spec:
        returned["provenance"] = request_spec["provenance"]
    if "validation" in request_spec:
        returned["validation"] = request_spec["validation"]
    return request_spec, {
        "spec": returned,
        "spec_hash": case["spec_hash"],
        "config_hash": "sha256:"
        + hashlib.sha256(
            json.dumps(
                returned["config"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }


@pytest.mark.parametrize(
    "case_name",
    [
        "provenance_present",
        "validation_nested_exclusions",
        "identifier_and_environment_normalized",
        "non_nfc_strings",
    ],
)
def test_publish_verification_accepts_the_backend_digest_over_the_returned_spec(
    case_name: str,
) -> None:
    sent, response = _backend_returned(case_name)
    ConfigStateManager._verify_cloud_response_hashes(response, response["spec"])
    # The request itself also verifies under the one rule (provenance never counts).
    assert compute_spec_hash(sent) == response["spec_hash"]


def test_publish_verification_still_rejects_a_wrong_digest() -> None:
    _sent, response = _backend_returned("provenance_present")
    response["spec_hash"] = "sha256:" + "0" * 64
    with pytest.raises(CloudBestConfigIntegrityError, match="spec_hash mismatch"):
        ConfigStateManager._verify_cloud_response_hashes(response, response["spec"])


def test_publish_hashes_the_returned_spec_not_the_request(monkeypatch) -> None:
    """The Backend may rewrite the request (trim config_id); the SDK verifies what came back."""
    case = _case("identifier_and_environment_normalized")
    sent = copy.deepcopy(case["request"]["spec"])
    assert sent["config_id"] != sent["config_id"].strip()
    returned = copy.deepcopy(case["hashed_payload"])
    response = {
        "spec": returned,
        "spec_hash": case["spec_hash"],
        "config_hash": "sha256:"
        + hashlib.sha256(
            json.dumps(
                returned["config"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    seen: list[dict[str, Any]] = []
    real = ConfigStateManager._verify_cloud_response_hashes

    def spy(data: dict[str, Any], spec: dict[str, Any]) -> None:
        seen.append(spec)
        real(data, spec)

    monkeypatch.setattr(
        ConfigStateManager, "_verify_cloud_response_hashes", staticmethod(spy)
    )
    ConfigStateManager._verify_publish_response(response, sent)
    assert seen == [returned]


def test_publish_response_without_a_stored_spec_fails_closed() -> None:
    sent = copy.deepcopy(_case("provenance_present")["request"]["spec"])
    with pytest.raises(CloudBestConfigIntegrityError, match="stored spec"):
        ConfigStateManager._verify_publish_response(
            {"spec_hash": "sha256:" + "0" * 64, "config_hash": "sha256:" + "0" * 64},
            sent,
        )


# --- stored hashes written under the old SDK rule fail closed with a clear error ---


def _old_rule_hash(spec: dict[str, Any]) -> str:
    keys = (
        "schema_version",
        "config_id",
        "function_ref",
        "environment",
        "config",
        "provenance",
        "validation",
    )
    payload = {key: spec[key] for key in keys if key in spec}
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def test_a_repo_manifest_with_an_old_rule_spec_hash_is_refused_with_a_clear_error(
    tmp_path: Path,
) -> None:
    write_repo_best_config(
        tmp_path,
        config_id="agent",
        config={"temperature": 0.2},
        provenance={"source": "unit"},
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    spec = json.loads((tmp_path / "agent.json").read_text())
    assert manifest["configs"]["agent"]["spec_hash"] == compute_spec_hash(spec)
    manifest["configs"]["agent"]["spec_hash"] = _old_rule_hash(spec)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="older spec_hash rule"):
        load_best_config_spec(
            tmp_path / "agent.json",
            manifest_entry=manifest["configs"]["agent"],
            strict=True,
        )


def test_a_cloud_cache_with_an_old_rule_spec_hash_is_refused(tmp_path: Path) -> None:
    spec = {
        "schema_version": "traigent.best_config.v1",
        "config_id": "agent",
        "environment": "default",
        "config": {"temperature": 0.2},
        "provenance": {"source": "unit"},
    }
    write_cloud_cache_best_config(tmp_path, spec, version=1)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["configs"]["agent"]["spec_hash"] = _old_rule_hash(spec)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ConfigurationError, match="older spec_hash rule"):
        resolve_cloud_cache_best_config(
            config_id="agent", cache_dir=tmp_path, strict=True
        )


def test_repo_manifest_hash_is_independent_of_provenance(tmp_path: Path) -> None:
    write_repo_best_config(
        tmp_path, config_id="a", config={"t": 1}, provenance={"source": "x"}
    )
    write_repo_best_config(tmp_path / "b", config_id="a", config={"t": 1})
    first = json.loads((tmp_path / "manifest.json").read_text())["configs"]["a"][
        "spec_hash"
    ]
    second = json.loads((tmp_path / "b" / "manifest.json").read_text())["configs"]["a"][
        "spec_hash"
    ]
    assert first == second


# --- a publish answer is bound to the request: environment and function_ref (sol F1 P2) ---


def _self_consistent_response(stored: dict[str, Any]) -> dict[str, Any]:
    return {
        "spec": stored,
        "spec_hash": compute_spec_hash(stored),
        "config_hash": "sha256:"
        + hashlib.sha256(
            json.dumps(
                stored["config"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }


def _sent_prod() -> dict[str, Any]:
    return {
        "schema_version": "traigent.best_config.v1",
        "config_id": "agent",
        "function_ref": "pkg:agent",
        "environment": "prod",
        "config": {"temperature": 0.2},
    }


def test_publish_response_for_a_different_environment_is_rejected() -> None:
    sent = _sent_prod()
    response = _self_consistent_response({**sent, "environment": "staging"})
    with pytest.raises(CloudBestConfigIntegrityError, match="spec_hash"):
        ConfigStateManager._verify_publish_response(response, sent, environment="prod")


def test_publish_response_for_a_different_function_ref_is_rejected() -> None:
    sent = _sent_prod()
    response = _self_consistent_response({**sent, "function_ref": "other:agent"})
    with pytest.raises(CloudBestConfigIntegrityError, match="spec_hash"):
        ConfigStateManager._verify_publish_response(response, sent, environment="prod")


def test_publish_response_environment_comes_from_the_envelope_when_the_spec_has_none() -> None:
    sent = _sent_prod()
    del sent["environment"]
    stored = {**sent, "environment": "prod"}
    ConfigStateManager._verify_publish_response(
        _self_consistent_response(stored), sent, environment="prod"
    )
    wrong = {**sent, "environment": "staging"}
    with pytest.raises(CloudBestConfigIntegrityError, match="spec_hash"):
        ConfigStateManager._verify_publish_response(
            _self_consistent_response(wrong), sent, environment="prod"
        )


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_every_vector_normalized_response_is_still_accepted(case: dict[str, Any]) -> None:
    request = case["request"]
    sent = copy.deepcopy(request["spec"])
    stored = copy.deepcopy(case["hashed_payload"])
    response = _self_consistent_response(stored)
    assert response["spec_hash"] == case["spec_hash"]
    ConfigStateManager._verify_publish_response(
        response, sent, environment=request.get("environment")
    )


# --- refusal vectors: NaN / +-Infinity and over-deep specs are refused, never hashed ---


@pytest.mark.parametrize("case", _refusal_cases(), ids=lambda case: case["name"])
def test_compute_spec_hash_refuses_every_refusal_vector(case: dict[str, Any]) -> None:
    request = copy.deepcopy(case["request"])
    substitution = case.get("non_finite_substitution")
    if substitution:
        target = request
        for key in substitution["path"][:-1]:
            target = target[key]
        target[substitution["path"][-1]] = float(substitution["value"])
    with pytest.raises(ConfigurationError):
        compute_spec_hash(request["spec"], environment=request.get("environment"))
