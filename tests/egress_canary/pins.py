"""Ratchet pins.  Lower these only by fixing leaks; never raise them.

``PINNED_LEAK_PAIRS`` is the number of distinct (route, canary kind) pairs the
canary run finds on the wire.  ``BASELINE_PAIRS`` lists them, so a fix that is
offset by a new leak (same count, different pair) is still caught.
"""

from __future__ import annotations

PINNED_LEAK_PAIRS = 43

BASELINE_PAIRS: tuple[tuple[str, str], ...] = (
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
