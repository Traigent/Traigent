"""An older traigent-schema (no commitment_linkage module) fails closed and cleanly.

Deliberately no ``importorskip``: this must run exactly when the module is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from traigent.cli import certify_commands as cc
from traigent.cli.certify_commands import certify
from traigent.cloud.process_record_linkage import (
    CommitmentLinkageClientError,
    finalize_process_record,
)

SESSION = "bsess_" + "0" * 32
MODULE = "traigent_schema.certification.commitment_linkage"


class _Transport:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def finalize_process_record_sync(self, sid: str, payload: dict[str, Any]) -> Any:
        self.calls.append((sid, payload))
        return {"success": True, "data": {}}

    def get_process_record_sync(self, sid: str) -> Any:
        self.calls.append(sid)
        return {}


@pytest.fixture
def old_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    # None in sys.modules makes `import <module>` raise ImportError.
    monkeypatch.setitem(sys.modules, MODULE, None)


def test_old_schema_yields_closed_code_and_sends_nothing(
    old_schema: None, tmp_path: Path
) -> None:
    transport = _Transport()
    bundle = tmp_path / "b.json"
    with pytest.raises(CommitmentLinkageClientError) as err:
        finalize_process_record(
            transport,
            SESSION,
            {"project_ref": "p"},
            link_commitments=True,
            bundle_path=bundle,
        )
    assert err.value.code == "SCHEMA_PACKAGE_TOO_OLD"
    assert "commitment_linkage" in str(err.value)
    assert err.value.__cause__ is None and err.value.__suppress_context__ is False
    assert err.value.__context__ is None
    assert transport.calls == []
    assert not bundle.exists()


def test_cli_prints_closed_code_and_exits_nonzero(
    old_schema: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = _Transport()
    monkeypatch.setattr(cc, "_make_backend_client", lambda: transport)
    request = tmp_path / "req.json"
    request.write_text(json.dumps({"project_ref": "p"}))
    result = CliRunner().invoke(
        certify,
        [
            "finalize",
            SESSION,
            "--request",
            str(request),
            "--bundle",
            str(tmp_path / "b.json"),
            "--link-commitments",
        ],
    )
    assert result.exit_code == 1
    assert "FINALIZE_REFUSED SCHEMA_PACKAGE_TOO_OLD" in result.output
    assert "install a traigent-schema version" in result.output
    assert "Traceback" not in result.output
    assert transport.calls == []
