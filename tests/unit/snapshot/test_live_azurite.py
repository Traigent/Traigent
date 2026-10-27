"""Optional live Azure Blob test (Azurite). Skipped unless env is set.

TRAIGENT_SNAPSHOT_AZURITE_URL=http://127.0.0.1:10000/<account>
TRAIGENT_SNAPSHOT_AZURITE_ACCOUNT / _KEY (the emulator's published dev account)
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from tests.unit.snapshot.live_scenario import run_scenario
from traigent.snapshot.backends import AzureBlobStore

pytestmark = pytest.mark.skipif(
    not os.environ.get("TRAIGENT_SNAPSHOT_AZURITE_URL"),
    reason="TRAIGENT_SNAPSHOT_AZURITE_URL not set",
)


def test_azurite_scenario(tmp_path: Path) -> None:
    blob = pytest.importorskip("azure.storage.blob")
    container = f"snap{uuid.uuid4().hex[:12]}"
    client = blob.ContainerClient(
        os.environ["TRAIGENT_SNAPSHOT_AZURITE_URL"],
        container,
        credential={
            "account_name": os.environ["TRAIGENT_SNAPSHOT_AZURITE_ACCOUNT"],
            "account_key": os.environ["TRAIGENT_SNAPSHOT_AZURITE_KEY"],
        },
    )
    client.create_container()
    try:
        store = AzureBlobStore(client, "live", allow_insecure_local=True)

        def tamper(key: str, data: bytes) -> None:
            client.upload_blob(f"live/{key}", data, overwrite=True)

        facts = run_scenario(store, tamper, tmp_path)
        assert facts["objects_first_publish"] > 0
    finally:
        client.delete_container()
