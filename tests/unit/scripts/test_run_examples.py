"""Tests for the example runner utility."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.examples.run_examples import ExampleRunner, main


def test_run_examples_passes_timeout_to_each_example(
    tmp_path: Path, monkeypatch
) -> None:
    example = tmp_path / "sample" / "run.py"
    example.parent.mkdir(parents=True)
    example.write_text("#!/usr/bin/env python3\n")

    runner = ExampleRunner(base_dir=str(tmp_path))
    observed: dict[str, object] = {}

    monkeypatch.setattr(runner, "validate_example_structure", lambda _: [])

    def fake_run_example(example_path: Path, timeout: int = 60) -> dict[str, object]:
        observed["example"] = example_path
        observed["timeout"] = timeout
        return {
            "file": str(example_path),
            "success": True,
            "duration": 0.1,
            "output": "",
            "error": "",
        }

    monkeypatch.setattr(runner, "run_example", fake_run_example)

    assert runner.run_examples("run.py", validate_structure=False, timeout=123)
    assert observed["example"] == example
    assert observed["timeout"] == 123


def test_main_passes_cli_timeout_to_runner(tmp_path: Path, monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, *, base_dir: str, verbose: bool) -> None:
            observed["base_dir"] = base_dir
            observed["verbose"] = verbose

        def run_examples(
            self, pattern: str, validate_structure: bool = True, timeout: int = 60
        ) -> bool:
            observed["pattern"] = pattern
            observed["validate_structure"] = validate_structure
            observed["timeout"] = timeout
            return True

        def generate_report(self) -> str:
            return "ok"

    monkeypatch.setattr("scripts.examples.run_examples.ExampleRunner", FakeRunner)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_examples.py",
            "--base",
            str(tmp_path),
            "--pattern",
            "run.py",
            "--timeout",
            "120",
        ],
    )

    assert main() == 0
    assert observed["base_dir"] == str(tmp_path)
    assert observed["pattern"] == "run.py"
    assert observed["timeout"] == 120
    assert observed["validate_structure"] is True
