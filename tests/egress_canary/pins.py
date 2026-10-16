"""Ratchet pins, loaded from the checked-in manifest.

leak_manifest.json pins leak *identities* (route, location, field_path,
canary kind), not counts: a leak that moves to another field or location on
the same route is a removal plus an addition and fails.  register only
shrinks; each removal needs an entry in leak_dispositions.json.
"""

from __future__ import annotations

import json
from pathlib import Path

Identity = tuple[str, str, str, str]

_HERE = Path(__file__).parent


def _ids(rows: list[dict[str, str]]) -> set[Identity]:
    return {(r["route"], r["location"], r["field_path"], r["kind"]) for r in rows}


def load_manifest() -> tuple[set[Identity], set[Identity]]:
    data = json.loads((_HERE / "leak_manifest.json").read_text(encoding="utf-8"))
    return _ids(data["baseline"]), _ids(data["register"])


def load_dispositions() -> list[dict[str, str]]:
    data = json.loads((_HERE / "leak_dispositions.json").read_text(encoding="utf-8"))
    return list(data["dispositions"])


BASELINE, REGISTER = load_manifest()

#: The 43 (route, kind) pairs measured before the coverage expansion (default
#: config, identity fields, offline replay, failure paths). Kept so the report
#: can state which pairs are new.
PRE_EXPANSION_PAIRS: tuple[tuple[str, str], ...] = (
    ("GET /api/v1/best-configs/{id}", "agent_name"),
    ("POST /api/v1/best-configs", "agent_name"),
    ("POST /api/v1/best-configs", "choice_value"),
    ("POST /api/v1/best-configs", "knob_name"),
    ("POST /api/v1/best-configs", "metric_name"),
    ("POST /api/v1/example-sets", "agent_name"),
    ("POST /api/v1/example-sets", "dataset_file_name"),
    ("POST /api/v1/example-sets/{id}/upload", "example_id"),
    ("POST /api/v1/example-sets/{id}/upload", "input_text"),
    ("POST /api/v1/example-sets/{id}/upload", "metadata_key"),
    ("POST /api/v1/example-sets/{id}/upload", "metadata_value"),
    ("POST /api/v1/example-sets/{id}/upload", "output_text"),
    ("POST /api/v1/hybrid/optimizer/{id}/batch", "metadata_key"),
    ("POST /api/v1/hybrid/optimizer/{id}/batch", "metadata_value"),
    ("POST /api/v1/hybrid/optimizer/{id}/batch", "metric_name"),
    ("POST /api/v1/hybrid/sessions", "agent_name"),
    ("POST /api/v1/hybrid/sessions", "choice_value"),
    ("POST /api/v1/hybrid/sessions", "knob_name"),
    ("POST /api/v1/hybrid/sessions", "metadata_key"),
    ("POST /api/v1/hybrid/sessions", "metadata_value"),
    ("POST /api/v1/hybrid/sessions", "metric_name"),
    ("POST /api/v1/hybrid/sessions", "prompt_template"),
    ("POST /api/v1/hybrid/sessions", "run_title"),
    ("POST /api/v1/sessions", "agent_name"),
    ("POST /api/v1/sessions", "choice_value"),
    ("POST /api/v1/sessions", "evaluation_set"),
    ("POST /api/v1/sessions", "knob_name"),
    ("POST /api/v1/sessions", "metric_name"),
    ("POST /api/v1/sessions", "prompt_template"),
    ("POST /api/v1/sessions", "run_description"),
    ("POST /api/v1/sessions", "run_title"),
    ("POST /api/v1/sessions/{id}/finalize", "metric_name"),
    ("POST /api/v1/sessions/{id}/intermediate-report", "metric_name"),
    ("POST /api/v1/sessions/{id}/results", "choice_value"),
    ("POST /api/v1/sessions/{id}/results", "exception_message"),
    ("POST /api/v1/sessions/{id}/results", "knob_name"),
    ("POST /api/v1/sessions/{id}/results", "metadata_key"),
    ("POST /api/v1/sessions/{id}/results", "metadata_value"),
    ("POST /api/v1/sessions/{id}/results", "metric_name"),
    ("POST /api/v1/sessions/{id}/results", "model_output"),
    ("POST /api/v1/sessions/{id}/results", "prompt_template"),
    ("PUT /api/v1/configuration-runs/{id}/measures", "metric_name"),
    ("PUT /api/v1/configuration-runs/{id}/status", "exception_message"),
)
