"""The transport gate in "refuse" mode refuses the known legacy fields before send.

Phase 2 evidence: with the gate installed in refuse mode the canary scenario cannot
complete a session-create, because the legacy content fields (agent name, knob names,
prompt templates, ...) are still in the body. The refusal must happen BEFORE the
transport: the capture never sees the refused request.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from traigent.labels import transport_gate as gate

from .canaries import CANARIES
from .scenario import run_scenario

pytestmark = pytest.mark.backend_online


@pytest.fixture()
def refuse_gate():
    gate.set_mode("refuse")
    gate.clear_refusals()
    gate.install_transport_gate()
    try:
        yield
    finally:
        gate.uninstall_transport_gate()
        gate.set_mode(None)


def test_legacy_fields_are_refused_before_send(
    refuse_gate, tmp_path_factory: pytest.TempPathFactory
) -> None:
    result = run_scenario(
        Path(tmp_path_factory.mktemp("gate-refuse")), direct_paths=False
    )

    refusals = gate.refusal_log()
    assert refusals, (
        "gate in refuse mode refused nothing: legacy fields reached the wire"
    )
    templates = {t for t, _ in refusals}
    assert "/api/v1/sessions" in templates, templates
    assert all(c.get("legacy_content", 0) > 0 for _, c in refusals)

    # Refused before send: nothing refused was ever captured by the transports.
    for template in templates:
        regex = "^" + template.replace("{session_id}", "[^/]+") + "/?$"
        import re

        sent = [
            r
            for r in result.requests
            if r.method == "POST" and re.match(regex, r.path) and r.body
        ]
        assert sent == [], f"{template} reached the transport after a refusal"

    # No canary value appears in the gate's own record.
    assert not any(v in repr(refusals) for v in CANARIES.values() if isinstance(v, str))
