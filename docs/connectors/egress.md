# Connector egress

`tests/unit/connectors/test_egress_boundaries.py::test_connector_operations_do_not_use_backend_egress_paths` replaces requests, httpx, aiohttp, and urllib request entry points, the OTLP exporter, and the observability ingest batch sender with failures while it validates and serializes a connector run summary. The test asserts that none of those replacements is called.

Summaries that may later be sent are closed allowlists validated by `traigent.connectors.models.validate_summary`.

See `docs/security/network-and-behaviour-manifest.md` for the SDK's existing network behaviour.
