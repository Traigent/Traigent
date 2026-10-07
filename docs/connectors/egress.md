# Connector egress

In this release, connector code paths send nothing to Traigent's backend. This is enforced by `tests/unit/connectors/test_egress_boundaries.py::test_connector_paths_cannot_reach_backend_exporters`.

Summaries that may later be sent are closed allowlists validated by `traigent.connectors.models.validate_summary`.

See `docs/security/network-and-behaviour-manifest.md` for the SDK's existing network behaviour.
