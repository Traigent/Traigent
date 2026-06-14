"""Scaffold initial agent build playbooks."""

from __future__ import annotations

import json

from traigent.playbook.model import STAGE_ORDER

_STAGE_HINTS = {
    "dataset": "Pin an evaluation dataset with dataset_ref, revision, and optional holdout_ref.",
    "metric": "Pin a metric with metric_name, measure_type, and metric_output_type.",
    "evaluator": "Pin evaluator wiring with evaluator_ref, evaluation_method, and audit_ref.",
    "optimize": "Pin objectives plus the configuration_space_ref or last_run_id used for tuning.",
    "gate": "Pin baseline_artifact plus budgets and policy thresholds for promotion decisions.",
}


def scaffold_playbook(
    name: str,
    agent_type: str | None = None,
    entrypoint: str | None = None,
) -> str:
    """Render an initial YAML agent build playbook."""
    lines = [
        "# Agent build playbook for durable lifecycle state.",
        'playbook_version: "1.0.0"',
        "agent:",
        f"  name: {_yaml_string(name)}",
    ]
    if entrypoint:
        lines.append(f"  entrypoint: {_yaml_string(entrypoint)}")
    if agent_type:
        lines.append(f"  agent_type: {_yaml_string(agent_type)}")

    lines.append("stages:")
    for stage_name in STAGE_ORDER:
        lines.extend(
            [
                f"  # {_STAGE_HINTS[stage_name]}",
                f"  {stage_name}:",
                "    status: pending",
            ]
        )

    lines.extend(
        [
            "provenance:",
            '  created_by: "traigent playbook init"',
            "",
        ]
    )
    return "\n".join(lines)


def _yaml_string(value: str) -> str:
    return json.dumps(str(value))
