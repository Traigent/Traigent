"""Optional live GCS test (fake-gcs-server). Skipped unless env is set.

TRAIGENT_SNAPSHOT_FAKEGCS_URL=http://127.0.0.1:4443
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from tests.unit.snapshot.live_scenario import run_scenario
from traigent.snapshot.backends import GCSStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("TRAIGENT_SNAPSHOT_FAKEGCS_URL"),
    reason="TRAIGENT_SNAPSHOT_FAKEGCS_URL not set",
)


def test_fake_gcs_scenario(tmp_path: Path) -> None:
    storage = pytest.importorskip("google.cloud.storage")
    credentials = pytest.importorskip("google.auth.credentials")
    client = storage.Client(
        project="snapshot-live",
        credentials=credentials.AnonymousCredentials(),
        client_options={"api_endpoint": os.environ["TRAIGENT_SNAPSHOT_FAKEGCS_URL"]},
    )
    bucket_name = f"snap-{uuid.uuid4().hex[:12]}"
    client.create_bucket(bucket_name)
    store = GCSStore(bucket_name, "live", client=client, allow_insecure_local=True)

    def tamper(key: str, data: bytes) -> None:
        client.bucket(bucket_name).blob(f"live/{key}").upload_from_string(data)

    facts = run_scenario(store, tamper, tmp_path)
    assert facts["objects_first_publish"] > 0
