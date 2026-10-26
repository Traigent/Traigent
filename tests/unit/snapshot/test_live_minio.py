"""Optional live S3-compatible test (MinIO). Skipped unless env is set.

TRAIGENT_SNAPSHOT_MINIO_URL=http://127.0.0.1:9000
TRAIGENT_SNAPSHOT_MINIO_ACCESS_KEY / _SECRET_KEY / _BUCKET
"""

from __future__ import annotations

import os

import pytest

from traigent.snapshot import (
    SnapshotStoreError,
    build_execution_snapshot,
    publish_snapshot,
    verify_snapshot,
)
from traigent.snapshot.backends import S3Store

pytestmark = pytest.mark.skipif(
    not os.environ.get("TRAIGENT_SNAPSHOT_MINIO_URL"),
    reason="TRAIGENT_SNAPSHOT_MINIO_URL not set",
)


def test_minio_roundtrip_and_probe() -> None:
    boto3 = pytest.importorskip("boto3")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["TRAIGENT_SNAPSHOT_MINIO_URL"],
        aws_access_key_id=os.environ["TRAIGENT_SNAPSHOT_MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["TRAIGENT_SNAPSHOT_MINIO_SECRET_KEY"],
        region_name="us-east-1",
    )
    bucket = os.environ["TRAIGENT_SNAPSHOT_MINIO_BUCKET"]
    store = S3Store(bucket, "live-test", client=client, allow_insecure_local=True)
    snapshot = build_execution_snapshot(
        dataset=[{"input": 1, "expected": 1}], config={"a": 1}
    )
    try:
        ref = publish_snapshot(store, snapshot)
    except SnapshotStoreError as error:
        pytest.fail(f"endpoint failed the capability probe: {error.code.value}")
    assert verify_snapshot(store, ref.manifest_digest).integrity == "ok"
    publish_snapshot(store, snapshot)  # republish: identical, no error
