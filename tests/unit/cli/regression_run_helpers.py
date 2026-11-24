"""Shared builders for the regression run-safety tests (no tests here)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from traigent.cli.main import cli
from traigent.utils.regression_set import content_digest

MODULE_SOURCE = """
import sys
import traigent

@traigent.optimize(
    eval_dataset=[{"input": {"q": "pass"}, "output": "ok"}],
    objectives=["accuracy"],
    configuration_space={"mode": ["m"]},
    execution_mode="local",
    max_trials=1,
)
def answer(q: str, example_id: str = "") -> str:
    return "ok" if q.startswith("pass") else "wrong"
"""


def make_row(
    row_id: str, input_data: dict[str, Any], output: str = "ok"
) -> dict[str, Any]:
    return {
        "input": input_data,
        "output": output,
        "metadata": {
            "example_id": row_id,
            "content_digest": content_digest(input_data, output),
            "regression": {
                "schema_version": 1,
                "label": "seen_once",
                "fails": None,
                "reps": None,
                "provenance": [],
            },
        },
    }


def write_rows(ws: Path, rows: list[dict[str, Any]]) -> Path:
    path = ws / "reg" / "regression.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def read_rows(ws: Path) -> dict[str, dict[str, Any]]:
    lines = (ws / "reg" / "regression.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {r["metadata"]["example_id"]: r for r in rows}


def statuses(ws: Path) -> dict[str, str]:
    return {
        k: v["metadata"]["regression"].get("status", "unconfirmed")
        for k, v in read_rows(ws).items()
    }


def invoke(ws: Path, verb: str, *extra: str, module: str = "agent.py:answer"):
    (ws / "config.json").write_text(json.dumps({"mode": "m"}), encoding="utf-8")
    args = [
        "regression",
        verb,
        "--module",
        module,
        "--config",
        str(ws / "config.json"),
        "--dir",
        str(ws / "reg"),
        *extra,
    ]
    return CliRunner().invoke(cli, args)
