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
    check_publishable_spec,
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
_PINNED_SHA256 = "b8afcc0d23bb7134f000344dd488e217fa633562396811f021e49e850893db90"


def _cases() -> list[dict[str, Any]]:
    return json.loads(_VECTORS.read_text(encoding="utf-8"))["cases"]


def _historical_cases() -> list[dict[str, Any]]:
    return json.loads(_VECTORS.read_text(encoding="utf-8"))["historical_cases"]


def _ingress_refusal_cases() -> list[dict[str, Any]]:
    return json.loads(_VECTORS.read_text(encoding="utf-8"))["ingress_refusal_cases"]


def _refusal_cases() -> list[dict[str, Any]]:
    return json.loads(_VECTORS.read_text(encoding="utf-8"))["refusal_cases"]


def _case(name: str) -> dict[str, Any]:
    return next(case for case in _cases() if case["name"] == name)


def test_vendored_vectors_are_the_pinned_schema_bytes() -> None:
    assert hashlib.sha256(_VECTORS.read_bytes()).hexdigest() == _PINNED_SHA256


@pytest.mark.parametrize(
    "case", _cases() + _historical_cases(), ids=lambda case: case["name"]
)
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
                stored["config"],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
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


def test_publish_response_environment_comes_from_the_envelope_when_the_spec_has_none() -> (
    None
):
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


@pytest.mark.parametrize(
    "case", _cases() + _historical_cases(), ids=lambda case: case["name"]
)
def test_every_vector_normalized_response_is_still_accepted(
    case: dict[str, Any],
) -> None:
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


# --- finiteness is checked over the ORIGINAL spec, before NFC and exclusions (sol F1b P2) ---


def test_the_new_refusal_vectors_are_present() -> None:
    names = {case["name"] for case in _refusal_cases()}
    assert {
        "nan_in_excluded_validation_member_refused",
        "non_finite_lost_to_nfc_collision_refused",
    } <= names


# --- NFC-colliding keys: the hash keeps last-wins, a NEW publish is refused before the network ---


@pytest.mark.parametrize(
    "case",
    _ingress_refusal_cases() + _historical_cases(),
    ids=lambda case: case["name"],
)
def test_a_spec_with_nfc_colliding_keys_is_not_publishable(
    case: dict[str, Any],
) -> None:
    with pytest.raises(ConfigurationError) as caught:
        check_publishable_spec(copy.deepcopy(case["request"]["spec"]))
    assert "\u00e9" not in str(caught.value) and "e\u0301" not in str(caught.value)


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_every_ordinary_vector_spec_is_publishable(case: dict[str, Any]) -> None:
    check_publishable_spec(copy.deepcopy(case["request"]["spec"]))


def test_a_nested_collision_inside_validation_is_not_publishable() -> None:
    spec = copy.deepcopy(_case("provenance_present")["request"]["spec"])
    spec["validation"] = {"detail": [{"\u00e9": 1, "e\u0301": 2}]}
    with pytest.raises(ConfigurationError):
        check_publishable_spec(spec)


def test_the_hash_function_itself_still_accepts_colliding_keys() -> None:
    for case in _historical_cases():
        request = case["request"]
        assert (
            compute_spec_hash(request["spec"], environment=request.get("environment"))
            == case["spec_hash"]
        )


def _publishing_manager(monkeypatch, sent: list[Any]):
    import traigent.cloud.backend_client as backend_client

    class Client:
        def fetch_best_config_sync(self, *args, **kwargs):  # noqa: ARG002
            return None

        def publish_best_config_sync(self, spec, **kwargs):  # noqa: ARG002
            sent.append(spec)
            raise AssertionError("must not reach the network")

    monkeypatch.setattr(backend_client, "get_backend_client", lambda **kw: Client())


def test_publish_best_config_refuses_colliding_keys_before_any_request(
    monkeypatch,
) -> None:
    from traigent.core.optimized_function import OptimizedFunction

    sent: list[Any] = []
    _publishing_manager(monkeypatch, sent)

    def agent(text: str) -> str:
        return text

    func = OptimizedFunction(
        func=agent, configuration_space={"t": [1, 2]}, config_id="agent"
    )
    func.set_config({"\u00e9": 1, "e\u0301": 2})
    with pytest.raises(ConfigurationError, match="Unicode"):
        func.publish_best_config()
    assert sent == []


def test_backend_client_publish_wrapper_refuses_colliding_keys_before_any_request(
    monkeypatch,
) -> None:
    from traigent.cloud.backend_client import BackendIntegratedClient

    client = BackendIntegratedClient.__new__(BackendIntegratedClient)
    spec = copy.deepcopy(_historical_cases()[0]["request"]["spec"])
    with pytest.raises(ConfigurationError):
        client.publish_best_config_sync(spec)
    with pytest.raises(ConfigurationError):
        client.publish_best_config_sync(
            spec, agent_ref="agent_" + "a" * 32, expected_head_generation=0
        )


# --- publish response: the top-level identity fields must agree with the verified spec (P3) ---


def _prod_response(**top_level: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    sent = _sent_prod()
    response = _self_consistent_response(copy.deepcopy(sent))
    response.update(top_level)
    return sent, response


def test_publish_response_with_matching_top_level_identity_is_accepted() -> None:
    sent, response = _prod_response(
        config_id="agent", environment="prod", function_ref="pkg:agent"
    )
    ConfigStateManager._verify_publish_response(response, sent, environment="prod")


def test_publish_response_normalization_of_top_level_fields_is_accepted() -> None:
    sent = _sent_prod()
    sent["config_id"] = "  agent "
    sent["environment"] = "PROD"
    stored = {**sent, "config_id": "agent", "environment": "prod"}
    response = _self_consistent_response(stored)
    response.update(config_id="agent", environment="prod", function_ref="pkg:agent")
    ConfigStateManager._verify_publish_response(response, sent, environment="PROD")


@pytest.mark.parametrize(
    "field,value",
    [
        ("config_id", "other"),
        ("environment", "staging"),
        ("function_ref", "other:agent"),
        ("config_id", 7),
        ("environment", None),
    ],
)
def test_publish_response_with_contradictory_top_level_identity_is_rejected(
    field: str, value: Any
) -> None:
    sent, response = _prod_response(**{field: value})
    with pytest.raises(CloudBestConfigIntegrityError, match="top-level"):
        ConfigStateManager._verify_publish_response(response, sent, environment="prod")


def test_publish_response_with_all_three_contradictory_fields_is_rejected() -> None:
    sent, response = _prod_response(
        config_id="other", environment="staging", function_ref="other:agent"
    )
    with pytest.raises(CloudBestConfigIntegrityError):
        ConfigStateManager._verify_publish_response(response, sent, environment="prod")


# --- fetch: the fetched spec is bound to the request (sol F1b P2) ---


def _agent(text: str) -> str:
    return text


def _fetched(**override: Any) -> dict[str, Any]:
    from traigent.core.best_config_runtime import function_ref_for

    spec = {
        "schema_version": "traigent.best_config.v1",
        "config_id": "agent",
        "function_ref": function_ref_for(_agent),
        "environment": "prod",
        "config": {"t": 2},
    }
    spec.update(override)
    data = _self_consistent_response(spec)
    data.update(
        config_id=spec["config_id"],
        environment=spec["environment"],
        function_ref=spec["function_ref"],
        etag="etag-1",
        version=1,
    )
    return data


def _fetching_manager(monkeypatch, tmp_path, returned, requested=None):
    """A ConfigStateManager asking for agent/<_agent>/prod whose backend answers ``returned``."""
    import traigent.cloud.backend_client as backend_client

    class Client:
        def fetch_best_config_sync(self, config_id, **kwargs):
            if requested is not None:
                requested.append({"config_id": config_id, **kwargs})
            return returned

    monkeypatch.setattr(backend_client, "get_backend_client", lambda **kw: Client())
    manager = ConfigStateManager.__new__(ConfigStateManager)
    manager.func = _agent
    manager.config_id = "agent"
    manager.best_config_environment = "prod"
    manager.best_config_source = "cloud"
    manager.best_config_strict = False
    manager.best_config_cache_dir = str(tmp_path / "cache")
    manager.configuration_space = {"t": [1, 2]}
    manager.default_config = {"t": 1}
    return manager


def test_fetch_of_the_requested_config_is_loaded(monkeypatch, tmp_path) -> None:
    manager = _fetching_manager(monkeypatch, tmp_path, _fetched())
    snapshot = manager._resolve_cloud_fetch_best_config()
    assert snapshot is not None and snapshot.config["t"] == 2


@pytest.mark.parametrize(
    "override",
    [
        {"environment": "staging"},
        {"function_ref": "other:agent"},
        {"config_id": "other"},
    ],
    ids=["environment", "function_ref", "config_id"],
)
def test_fetch_of_another_environment_function_or_config_is_rejected_and_never_cached(
    monkeypatch, tmp_path, override: dict[str, Any]
) -> None:
    manager = _fetching_manager(monkeypatch, tmp_path, _fetched(**override))
    with pytest.raises(CloudBestConfigIntegrityError, match="request"):
        manager._resolve_cloud_fetch_best_config()
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").iterdir())


def test_fetch_response_top_level_identity_must_agree_with_the_spec(
    monkeypatch, tmp_path
) -> None:
    data = _fetched()
    data["environment"] = "staging"
    manager = _fetching_manager(monkeypatch, tmp_path, data)
    with pytest.raises(CloudBestConfigIntegrityError, match="top-level"):
        manager._resolve_cloud_fetch_best_config()


def test_fetch_binding_fails_closed_in_cloud_mode_and_is_ignored_otherwise(
    monkeypatch, tmp_path
) -> None:
    from traigent.core.best_config_runtime import (
        CloudPublishUnavailable,
        CloudPublishUnavailableReason,
    )

    manager = _fetching_manager(monkeypatch, tmp_path, _fetched(environment="staging"))
    with pytest.raises(CloudPublishUnavailable) as caught:
        manager._resolve_mode_snapshot()
    assert caught.value.reason is CloudPublishUnavailableReason.INTEGRITY_FAILED
    # Outside strict/pure-cloud modes the source is ignored: nothing is loaded.
    monkeypatch.chdir(tmp_path)
    manager.best_config_source = "cloud_then_repo"
    manager.best_config_cache_ttl_seconds = 60
    manager.best_config_stale_ok_ttl_seconds = None
    assert manager._resolve_mode_snapshot() is None


def test_cloud_cache_of_another_environment_or_function_is_not_loaded(tmp_path) -> None:
    spec = {
        "schema_version": "traigent.best_config.v1",
        "config_id": "agent",
        "function_ref": "pkg:agent",
        "environment": "staging",
        "config": {"t": 2},
    }
    write_cloud_cache_best_config(tmp_path, spec, version=1)
    kwargs = {"config_id": "agent", "cache_dir": tmp_path, "strict": True}
    with pytest.raises(ConfigurationError, match="environment"):
        resolve_cloud_cache_best_config(
            expected_function_ref="pkg:agent", expected_environment="prod", **kwargs
        )
    with pytest.raises(ConfigurationError, match="function_ref"):
        resolve_cloud_cache_best_config(
            expected_function_ref="pkg:other", expected_environment="staging", **kwargs
        )
    snapshot = resolve_cloud_cache_best_config(
        expected_function_ref="pkg:agent", expected_environment="staging", **kwargs
    )
    assert snapshot is not None and snapshot.config["t"] == 2
