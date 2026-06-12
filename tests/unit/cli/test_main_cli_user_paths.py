"""CLI user-path and strict validation regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from traigent.cli.main import cli


def test_validate_config_accepts_user_project_file_outside_package_dir(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "optimization.json"
    config_path.write_text(
        json.dumps(
            {
                "configuration_space": {"temperature": [0.0, 0.5]},
                "objectives": ["accuracy"],
                "algorithm": "grid",
            }
        )
    )

    result = CliRunner().invoke(cli, ["validate-config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "Configuration validation passed" in result.output
    assert "workspace" not in result.output.lower()


def test_generate_accepts_user_output_path_outside_package_dir(tmp_path: Path) -> None:
    output_path = tmp_path / "generated" / "traigent_example.py"

    result = CliRunner().invoke(cli, ["generate", "--output", str(output_path)])

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    assert "@traigent.optimize" in output_path.read_text()
    assert "workspace" not in result.output.lower()


def test_validate_strict_exits_nonzero_for_invalid_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.jsonl"
    dataset_path.write_text(json.dumps({"output": "missing input"}) + "\n")

    result = CliRunner().invoke(
        cli,
        ["validate", str(dataset_path), "--strict"],
        env={"TRAIGENT_DATASET_ROOT": str(tmp_path)},
    )

    assert result.exit_code != 0
    assert "Dataset content validation failed" in result.output


def test_validate_strict_exits_zero_for_valid_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "valid.jsonl"
    dataset_path.write_text(json.dumps({"input": "hello", "output": "world"}) + "\n")

    result = CliRunner().invoke(
        cli,
        ["validate", str(dataset_path), "--strict"],
        env={"TRAIGENT_DATASET_ROOT": str(tmp_path)},
    )

    assert result.exit_code == 0, result.output
    assert "Dataset content validation passed" in result.output
