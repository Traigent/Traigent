from unittest.mock import patch

from traigent.connectors.models import ConnectionRef
from traigent.connectors.privacy import CustomerSideMinter, serialize_summary, validate_summary


def test_connector_paths_cannot_reach_backend_exporters():
    calls = []
    def intercepted(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("backend exporter invoked")

    with patch("traigent.observability.otel.exporter.TraigentOTLPExporter.export", intercepted):
        minter = CustomerSideMinter(ConnectionRef("langfuse"), b"x" * 32)
        run = {
            "schema_version": "1", "run_token": minter.mint("run", "synthetic"),
            "connector_kind": "langfuse", "connection_token": minter.connection_token,
            "command": "check", "status": "queued",
            "counts": {"observations_read": 0, "scores_read": 0, "rows_dropped_invalid": 0, "pages": 0, "items_written": 0},
            "guarantees": [],
        }
        assert validate_summary({key: value.value if hasattr(value, "value") else value for key, value in run.items()})
        serialize_summary(run, minter=minter)
    assert calls == []
