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
case (lost response, simulated restart then resume, simulated restart then write
retry, duplicate rows, overlapping pages, expired cursor, partial batch, connection
collision, content rejection). ``CASES`` maps case names to plain functions taking a
harness, for use outside pytest. All canary strings are synthetic; failure messages
never echo them.

Limits: durability is self-attested by the connector under test, not verified by
this kit. The restart cases drop the harness object in-process and rebuild a new one
from the state the harness itself reports through ``persisted_state()`` (the suite's
own ``Connection`` handles are reused across the restart); what ``committed()`` and
``persisted_state()`` return is whatever the harness says it holds. The cases check
resume-after-restart and duplicate or lost-response handling through that reported
state. They do not show that data reached durable storage and cannot detect a
connector that reports durability without persisting. A real durability check needs
the harness to crash the connector out of process and inspect the backing store
itself, which this kit does not do.
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
