from contextlib import ExitStack
from unittest.mock import patch
from urllib import request as urllib_request

import aiohttp
import httpx
import requests

from traigent.connectors.models import ConnectionRef
from traigent.connectors.privacy import (
    CustomerSideMinter,
    serialize_summary,
    validate_summary,
)


def test_connector_operations_do_not_use_backend_egress_paths():
    calls: list[str] = []

    def intercepted(path: str):
        def reject(*args, **kwargs):
            calls.append(path)
            raise AssertionError(f"backend egress path invoked: {path}")

        return reject

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(requests.Session, "request", intercepted("requests"))
        )
        stack.enter_context(patch.object(httpx.Client, "request", intercepted("httpx")))
        stack.enter_context(
            patch.object(httpx.AsyncClient, "request", intercepted("httpx-async"))
        )
        stack.enter_context(
            patch.object(aiohttp.ClientSession, "_request", intercepted("aiohttp"))
        )
        stack.enter_context(
            patch.object(
                urllib_request.OpenerDirector, "open", intercepted("urllib-opener")
            )
        )
        stack.enter_context(
            patch.object(urllib_request, "urlopen", intercepted("urllib"))
        )
        stack.enter_context(
            patch(
                "traigent.observability.otel.exporter.TraigentOTLPExporter.export",
                intercepted("otlp-exporter"),
            )
        )
        stack.enter_context(
            patch(
                "traigent.observability.client.ObservabilityClient._send_payload_batch",
                intercepted("observability-ingest"),
            )
        )
        minter = CustomerSideMinter(ConnectionRef("langfuse"), b"x" * 32)
        run = {
            "schema_version": "1",
            "run_token": minter.mint("run", "synthetic"),
            "connector_kind": "langfuse",
            "connection_token": minter.connection_token,
            "command": "check",
            "status": "queued",
            "counts": {
                "observations_read": 0,
                "scores_read": 0,
                "rows_dropped_invalid": 0,
                "pages": 0,
                "items_written": 0,
            },
            "guarantees": [],
        }
        assert validate_summary(
            {
                key: value.value if hasattr(value, "value") else value
                for key, value in run.items()
            }
        )
        serialize_summary(run, minter=minter)
    assert calls == []
