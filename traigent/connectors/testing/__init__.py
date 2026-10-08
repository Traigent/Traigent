"""Reusable conformance suites for connector authors.

Usage::

    from traigent.connectors.testing import ConnectorConformanceSuite, ConnectorHarness

    class MyHarness(ConnectorHarness):
        kind = "my-connector"
        ...  # drive your connector against a fake backend; see ConnectorHarness

    class TestMyConnectorConformance(ConnectorConformanceSuite):
        @staticmethod
        def make_harness() -> ConnectorHarness:
            return MyHarness()

Run it with ``pytest -n 0``. Each ``test_*`` method of the suite is one stateful
case (lost response, crash/resume, crash on write, duplicate rows, overlapping pages, expired
cursor, partial batch, connection collision, content rejection). ``CASES`` maps case names to
plain functions taking a harness, for use outside pytest. All canary strings
are synthetic; failure messages never echo them.
"""

from .harness import (
    CANARY,
    ConformanceFailure,
    Connection,
    ConnectorHarness,
    Fault,
    FaultKind,
)
from .suites import CASES, ConnectorConformanceSuite

__all__ = [
    "CANARY",
    "CASES",
    "ConformanceFailure",
    "Connection",
    "ConnectorConformanceSuite",
    "ConnectorHarness",
    "Fault",
    "FaultKind",
]
