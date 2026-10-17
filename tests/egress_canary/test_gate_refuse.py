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
    try:
        yield
    finally:
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


def _sessions_refusal(log: list[tuple[str, dict[str, int]]]) -> dict[str, int]:
    counts = [c for t, c in log if t == "/api/v1/sessions"]
    assert counts, f"no session-create refusal in {[t for t, _ in log]}"
    return counts[0]


def test_content_less_create_drops_group1_fields_but_later_groups_stay_refused(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Phase 3 group 1: the content-less create no longer trips the gate on identity fields.

    The refuse-mode gate still refuses it, but only for the later-group fields (knob and
    choice names, metric names, prompt templates, run narrative, ...), which stay recorded
    as open in leak_manifest.json. The legacy create's gate counts are strictly higher:
    the difference is the group-1 identity fields (agent key/id, dataset id, dataset and
    evaluation-set names, evaluator id, display names).
    """

    gate.set_mode("refuse")
    try:
        gate.clear_refusals()
        run_scenario(
            Path(tmp_path_factory.mktemp("gate-legacy")),
            direct_paths=False,
        )
        legacy = _sessions_refusal(gate.refusal_log())

        gate.clear_refusals()
        result = run_scenario(
            Path(tmp_path_factory.mktemp("gate-content-less")),
            direct_paths=False,
            content_less=True,
        )
        content_less = _sessions_refusal(gate.refusal_log())
    finally:
        gate.set_mode(None)

    # still refused for the later groups, before send
    assert content_less.get("legacy_content", 0) > 0
    assert not [
        r
        for r in result.requests
        if r.method == "POST" and r.path.endswith("/sessions")
    ], "a refused content-less create reached the transport"
    # but strictly less legacy content than the legacy create: group-1 fields are gone
    assert content_less["legacy_content"] < legacy["legacy_content"]
    assert content_less.get("unclassified", 0) < legacy.get("unclassified", 0)
