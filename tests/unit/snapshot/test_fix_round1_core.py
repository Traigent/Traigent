"""Fix round 1 (stage A review): manifest validation, immutability, S3, metadata."""

from __future__ import annotations

import copy
import json
import logging
import tomllib
import traceback
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.unit.snapshot.fakes import FakeS3Client, client_error
from traigent.snapshot import (
    SnapshotErrorCode,
    SnapshotStoreError,
    build_execution_snapshot,
    publish_snapshot,
    verify_snapshot,
)
from traigent.snapshot.backends import LocalDirStore, S3Store
from traigent.snapshot.capture import serialize_example
from traigent.snapshot.manifest import (
    ExecutionSnapshot,
    canonical_manifest_bytes,
    digest_of,
    parse_manifest,
)

CANARY = "CANARY-SECRET-r1-4c2e"
# botocore 1.35.2 is the first release whose S3 PutObject model has IfNoneMatch
# (found by reading botocore/data/s3/2006-03-01/service-2.json.gz in the wheels
# of 1.34.162, 1.35.0, 1.35.1, 1.35.2 and 1.35.3); boto3 1.35.2 needs botocore>=1.35.2.
S3_FLOOR = (1, 35, 2)


def _clock() -> datetime:
    return datetime(2026, 10, 27, 12, 30, 45, tzinfo=UTC)


def _good() -> dict[str, Any]:
    snapshot = build_execution_snapshot(
        dataset=[{"input": 1, "expected": 1}], config={"a": 1}, clock=_clock
    )
    return json.loads(snapshot.manifest_bytes)


def _corrupt(manifest: dict[str, Any]) -> None:
    with pytest.raises(SnapshotStoreError) as caught:
        parse_manifest(canonical_manifest_bytes(manifest))
    assert caught.value.code is SnapshotErrorCode.CORRUPT
    assert caught.value.part == "manifest"


# ------------------------------------------------- A1 finding 1: structure


def test_a1_f1_minimal_manifest_is_rejected() -> None:
    _corrupt({"binding": {}, "format": "tes1", "parts": {}, "required_parts": []})


def test_a1_f1_minimal_manifest_stored_under_its_digest_does_not_verify(
    tmp_path: Path,
) -> None:
    store = LocalDirStore(tmp_path / "store")
    data = canonical_manifest_bytes(
        {"binding": {}, "format": "tes1", "parts": {}, "required_parts": []}
    )
    digest = digest_of(data)
    store.create_if_absent(f"snapshots/{digest[len('sha256:') :]}/manifest.json", data)
    with pytest.raises(SnapshotStoreError) as caught:
        verify_snapshot(store, digest)
    assert caught.value.code is SnapshotErrorCode.CORRUPT


def test_a1_f1_good_manifest_still_parses() -> None:
    parse_manifest(canonical_manifest_bytes(_good()))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m["required_parts"].remove("dataset/examples.jsonl"),
        lambda m: m["required_parts"].remove("config/config.json"),
        lambda m: m["required_parts"].remove("agent/git.json"),
        lambda m: m["parts"].pop("agent/callable.py"),
        lambda m: m["required_parts"].clear(),
        lambda m: m.pop("sdk_version"),
        lambda m: m.pop("created_at"),
        lambda m: m.update(created_at="yesterday"),
        lambda m: m.update(sdk_version=""),
        lambda m: m.update(binding={}),
        lambda m: m.update(binding={"kind": "unbound", "extra": 1}),
        lambda m: m.pop("fingerprints"),
        lambda m: m["fingerprints"].pop("dataset_fp1"),
        lambda m: m["fingerprints"].update(evaluator_code_identity="x"),
        lambda m: m.pop("snapshot_coverage"),
        lambda m: m["snapshot_coverage"].update(agent="maybe"),
        lambda m: m["required_parts"].append("config/config.json"),  # duplicate
    ],
    ids=lambda f: "mutation",
)
def test_a1_f1_each_mandatory_slot_and_metadata_field_is_enforced(mutate: Any) -> None:
    manifest = _good()
    mutate(manifest)
    _corrupt(manifest)


def test_a1_f1_evaluator_group_is_mandatory() -> None:
    manifest = _good()
    for path in [p for p in manifest["parts"] if p.startswith("evaluator/")]:
        del manifest["parts"][path]
        manifest["required_parts"].remove(path)
    _corrupt(manifest)


def test_a1_f1_capture_always_emits_every_mandatory_slot() -> None:
    from traigent.snapshot.manifest import MANDATORY_PARTS

    for kwargs in ({}, {"config": {"a": 1}}, {"agent": lambda q: q}):
        manifest = json.loads(build_execution_snapshot(**kwargs).manifest_bytes)
        assert set(MANDATORY_PARTS) <= set(manifest["required_parts"])
        assert set(MANDATORY_PARTS) <= set(manifest["parts"])


# ------------------------------------------- A1 finding 2: immutability


def _snapshot() -> ExecutionSnapshot:
    return build_execution_snapshot(
        dataset=[{"input": 1, "expected": 1}], config={"a": 1}, clock=_clock
    )


def _present(snapshot: ExecutionSnapshot) -> list[tuple[str, dict[str, Any]]]:
    return list(snapshot.present_parts())


def test_a1_f2_mutating_manifest_view_changes_nothing() -> None:
    snapshot = _snapshot()
    before_repr = repr(snapshot)
    before_manifest = copy.deepcopy(dict(snapshot.manifest))
    for attempt in (
        lambda m: m["fingerprints"].__setitem__("dataset_fp1", "sha256:" + "0" * 64),
        lambda m: m["required_parts"].append("x"),
        lambda m: m["parts"].clear(),
        lambda m: m.__setitem__("format", "tes9"),
        lambda m: m["snapshot_coverage"]["agent_gaps"].append("x"),
    ):
        try:
            attempt(snapshot.manifest)
        except (TypeError, AttributeError):
            pass  # an immutable view refusing the write is also fine
    assert copy.deepcopy(dict(snapshot.manifest)) == before_manifest
    assert repr(snapshot) == before_repr
    assert parse_manifest(snapshot.manifest_bytes) == before_manifest


def test_a1_f2_mutating_present_parts_entries_changes_nothing() -> None:
    snapshot = _snapshot()
    expected = [(path, dict(entry)) for path, entry in _present(snapshot)]
    for _, entry in _present(snapshot):
        try:
            entry["sha256"] = "sha256:" + "f" * 64
            entry["size"] = 0
        except (TypeError, AttributeError):
            pass
    assert [(p, dict(e)) for p, e in _present(snapshot)] == expected
    assert "f" * 64 not in repr(snapshot)


def test_a1_f2_publish_is_grounded_in_the_manifest_bytes(tmp_path: Path) -> None:
    snapshot = _snapshot()
    for _, entry in _present(snapshot):
        try:
            entry["sha256"] = "sha256:" + "e" * 64
        except (TypeError, AttributeError):
            pass
    try:
        snapshot.manifest["parts"].clear()
    except (TypeError, AttributeError):
        pass
    store = LocalDirStore(tmp_path / "store")
    ref = publish_snapshot(store, snapshot)
    verification = verify_snapshot(store, ref.manifest_digest)
    assert verification.integrity == "ok"
    assert ref.manifest_digest == digest_of(snapshot.manifest_bytes)


def test_a1_f2_unavailable_accessor_is_a_tuple_of_strings() -> None:
    reasons = _snapshot().unavailable()
    assert isinstance(reasons, tuple)
    assert all(isinstance(path, str) and isinstance(why, str) for path, why in reasons)


# ----------------------------------------------- A1 finding 3: S3 floor


def _floor_of(requirement: str) -> tuple[int, ...]:
    version = requirement.split(">=")[1].split(",")[0].strip()
    return tuple(int(piece) for piece in version.split("."))


def test_a1_f3_snapshot_s3_extra_floor_supports_if_none_match() -> None:
    root = Path(__file__).resolve().parents[3]
    extras = tomllib.loads((root / "pyproject.toml").read_text("utf-8"))["project"][
        "optional-dependencies"
    ]
    boto = [r for r in extras["snapshot-s3"] if r.startswith("boto3")]
    assert len(boto) == 1
    assert _floor_of(boto[0]) >= S3_FLOOR


@pytest.fixture
def real_s3_client() -> Iterator[Any]:
    boto3 = pytest.importorskip("boto3")
    from botocore import UNSIGNED
    from botocore.config import Config

    yield boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(signature_version=UNSIGNED, retries={"max_attempts": 0}),
    )


def test_a1_f3_real_client_model_accepts_the_conditional_put(
    real_s3_client: Any,
) -> None:
    from botocore.stub import Stubber

    model = real_s3_client.meta.service_model.operation_model("PutObject")
    assert "IfNoneMatch" in model.input_shape.members
    store = S3Store("my-bucket", "pre", client=real_s3_client)
    with Stubber(real_s3_client) as stubber:
        stubber.add_response(
            "put_object",
            {},
            {
                "Bucket": "my-bucket",
                "Key": "pre/blobs/x",
                "Body": b"data",
                "IfNoneMatch": "*",
            },
        )
        assert store.create_if_absent("blobs/x", b"data").value == "created"
        stubber.assert_no_pending_responses()


def test_a1_f3_real_client_412_is_read_and_compared(real_s3_client: Any) -> None:
    import io

    from botocore.response import StreamingBody
    from botocore.stub import Stubber

    store = S3Store("my-bucket", client=real_s3_client)
    with Stubber(real_s3_client) as stubber:
        stubber.add_client_error(
            "put_object", "PreconditionFailed", http_status_code=412
        )
        stubber.add_response(
            "get_object",
            {"Body": StreamingBody(io.BytesIO(b"data"), 4)},
            {"Bucket": "my-bucket", "Key": "blobs/x"},
        )
        assert store.create_if_absent("blobs/x", b"data").value == "existed_identical"
        stubber.assert_no_pending_responses()


# --------------------------------------------- A1 finding 4: 409 in probe


class ScriptedS3(FakeS3Client):
    """Answers the Nth put_object with a scripted exception, else behaves."""

    def __init__(self, endpoint: str, script: dict[int, Exception]) -> None:
        super().__init__(endpoint)
        self.script = script

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        index = len(self.put_calls)
        if index in self.script:
            self.put_calls.append(kwargs)
            raise self.script[index]
        return super().put_object(**kwargs)


@pytest.mark.parametrize("code", ["ConditionalRequestConflict", "409"])
def test_a1_f4_conflict_409_is_not_capability_evidence(code: str) -> None:
    client = ScriptedS3(
        "https://minio.example.com", {1: client_error(code, 409, "retry")}
    )
    store = S3Store("my-bucket", "p", client=client)
    with pytest.raises(SnapshotStoreError) as caught:
        store.create_if_absent("blobs/x", b"1")
    assert caught.value.code is SnapshotErrorCode.BACKEND_UNAVAILABLE
    assert ("my-bucket", "p/blobs/x") not in client.objects  # nothing real written
    # not remembered as supported: the next call probes again from scratch
    before = len(client.put_calls)
    store.create_if_absent("blobs/x", b"1")
    probes = [c for c in client.put_calls[before:] if "/.probe/" in c["Key"]]
    assert len(probes) == 2


def test_a1_f4_412_still_proves_the_capability() -> None:
    client = FakeS3Client("https://minio.example.com")
    store = S3Store("my-bucket", "p", client=client)
    store.create_if_absent("blobs/x", b"1")
    store.create_if_absent("blobs/y", b"2")
    assert len([c for c in client.put_calls if "/.probe/" in c["Key"]]) == 2


# ----------------------------------- A1 finding 5: unknown destination


class NoMetaClient:
    def __init__(self, *meta: Any) -> None:
        self.calls = 0
        if meta:
            self.meta = meta[0]

    def put_object(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {}

    def get_object(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        raise AssertionError("must not be reached")

    def delete_object(self, **_kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        return {}


@pytest.mark.parametrize(
    "meta",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(endpoint_url=None),
        SimpleNamespace(endpoint_url=""),
        SimpleNamespace(endpoint_url=5),
    ],
    ids=["no-meta", "no-attr", "none", "empty", "non-str"],
)
def test_a1_f5_client_without_discoverable_endpoint_is_refused(meta: Any) -> None:
    client = NoMetaClient() if meta is None else NoMetaClient(meta)
    with pytest.raises(SnapshotStoreError) as caught:
        S3Store("my-bucket", client=client)
    assert caught.value.code is SnapshotErrorCode.UNSAFE_DESTINATION
    assert client.calls == 0


# --------------------------------- A1 finding 6: construction sanitising


def _assert_sanitized(error: SnapshotStoreError) -> None:
    assert error.__cause__ is None and error.__suppress_context__ is True
    text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    assert CANARY not in text


def test_a1_f6_session_client_construction_error_is_sanitized(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Session:
        def client(self, _name: str) -> Any:
            raise RuntimeError(f"proxy https://user:{CANARY}@proxy.example failed")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(SnapshotStoreError) as caught:
            S3Store("my-bucket", session=Session())
    assert caught.value.code is SnapshotErrorCode.BACKEND_UNAVAILABLE
    assert caught.value.backend == "s3"
    _assert_sanitized(caught.value)
    assert CANARY not in caplog.text


def test_a1_f6_default_client_construction_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto3 = pytest.importorskip("boto3")

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError(f"bad endpoint {CANARY}")

    monkeypatch.setattr(boto3, "client", boom)
    with pytest.raises(SnapshotStoreError) as caught:
        S3Store("my-bucket")
    _assert_sanitized(caught.value)


def test_a1_f6_missing_extra_hint_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "boto3":
            raise ImportError("no boto3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"traigent\[snapshot-s3\]"):
        S3Store("my-bucket")


# ------------------------------------- A1 finding 8: falsy metadata kept


@pytest.mark.parametrize("value", [{}, False, 0, "", []])
def test_a1_f8_present_falsy_metadata_is_preserved_mapping(value: Any) -> None:
    row = serialize_example({"input": 1, "expected": 1, "metadata": value})
    assert "metadata" in row and row["metadata"] == value
    assert type(row["metadata"]) is type(value)


@pytest.mark.parametrize("value", [{}, False, 0])
def test_a1_f8_present_falsy_metadata_is_preserved_object(value: Any) -> None:
    example = SimpleNamespace(input_data={"q": 1}, expected_output=1, metadata=value)
    row = serialize_example(example)
    assert "metadata" in row and row["metadata"] == value


def test_a1_f8_none_metadata_stays_absent() -> None:
    assert "metadata" not in serialize_example(
        {"input": 1, "expected": 1, "metadata": None}
    )
    assert "metadata" not in serialize_example({"input": 1, "expected": 1})


def test_a1_f8_empty_set_metadata_makes_the_dataset_not_serializable() -> None:
    snapshot = build_execution_snapshot(
        dataset=[{"input": 1, "expected": 1, "metadata": set()}], clock=_clock
    )
    entry = snapshot.manifest["parts"]["dataset/examples.jsonl"]
    assert entry == {"status": "unavailable", "reason": "NOT_SERIALIZABLE"}


def test_a1_f8_falsy_metadata_lands_in_the_dataset_bytes() -> None:
    snapshot = build_execution_snapshot(
        dataset=[{"input": 1, "expected": 1, "metadata": {}}], clock=_clock
    )
    line = snapshot.parts["dataset/examples.jsonl"].decode("utf-8").strip()
    assert json.loads(line)["metadata"] == {}
