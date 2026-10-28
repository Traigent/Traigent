"""Optional live S3-compatible test (MinIO). Skipped unless env is set.

TRAIGENT_SNAPSHOT_MINIO_URL=http://127.0.0.1:9000
TRAIGENT_SNAPSHOT_MINIO_ACCESS_KEY / _SECRET_KEY / _BUCKET
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from tests.unit.snapshot.live_scenario import run_scenario
from traigent.snapshot import SnapshotStoreError
from traigent.snapshot.backends import S3Store

pytestmark = pytest.mark.skipif(
    not os.environ.get("TRAIGENT_SNAPSHOT_MINIO_URL"),
    reason="TRAIGENT_SNAPSHOT_MINIO_URL not set",
)


def test_minio_scenario_and_conditional_write_probe(tmp_path: Path) -> None:
    boto3 = pytest.importorskip("boto3")
    client = boto3.client(
        "s3",
        endpoint_url=os.environ["TRAIGENT_SNAPSHOT_MINIO_URL"],
        aws_access_key_id=os.environ["TRAIGENT_SNAPSHOT_MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["TRAIGENT_SNAPSHOT_MINIO_SECRET_KEY"],
        region_name="us-east-1",
    )
    bucket = os.environ["TRAIGENT_SNAPSHOT_MINIO_BUCKET"]
    try:
        client.create_bucket(Bucket=bucket)
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass
    prefix = f"live-{uuid.uuid4().hex[:12]}"
    store = S3Store(bucket, prefix, client=client, allow_insecure_local=True)

    def tamper(key: str, data: bytes) -> None:
        client.put_object(Bucket=bucket, Key=f"{prefix}/{key}", Body=data)

    try:
        facts = run_scenario(store, tamper, tmp_path)
    except SnapshotStoreError as error:
        pytest.fail(f"endpoint failed the capability probe: {error.code.value}")
    assert facts["objects_first_publish"] > 0
    # the probe ran against MinIO and left nothing behind
    listing = client.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/.probe/")
    assert listing.get("KeyCount", 0) == 0
