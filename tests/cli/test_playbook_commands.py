from pathlib import Path

from click.testing import CliRunner

from traigent.cli.main import cli


def test_playbook_init_creates_file_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "traigent.playbook.yaml"
    runner = CliRunner()

    first = runner.invoke(
        cli,
        [
            "playbook",
            "init",
            "--name",
            "support-agent",
            "--agent-type",
            "rag",
            "--entrypoint",
            "agent:run",
            "--path",
            str(path),
        ],
    )
    assert first.exit_code == 0
    assert path.exists()
    original = path.read_text(encoding="utf-8")
    assert "evaluation dataset" in original

    second = runner.invoke(
        cli,
        ["playbook", "init", "--name", "other-agent", "--path", str(path)],
    )

    assert second.exit_code != 0
    assert "already exists" in second.output
    assert path.read_text(encoding="utf-8") == original


def test_playbook_validate_green_on_scaffold_and_red_on_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "traigent.playbook.yaml"
    runner = CliRunner()

    init_result = runner.invoke(
        cli,
        ["playbook", "init", "--name", "support-agent", "--path", str(path)],
    )
    assert init_result.exit_code == 0

    valid_result = runner.invoke(cli, ["playbook", "validate", "--path", str(path)])
    assert valid_result.exit_code == 0
    assert "playbook valid" in valid_result.output

    contents = path.read_text(encoding="utf-8")
    path.write_text(
        contents.replace("status: pending", "status: done", 1), encoding="utf-8"
    )

    invalid_result = runner.invoke(cli, ["playbook", "validate", "--path", str(path)])

    assert invalid_result.exit_code == 1
    assert "$.stages.dataset.status" in invalid_result.output
    assert "'done' is not one of" in invalid_result.output


def test_playbook_status_renders_all_five_stages(tmp_path: Path) -> None:
    path = tmp_path / "traigent.playbook.yaml"
    runner = CliRunner()

    init_result = runner.invoke(
        cli,
        ["playbook", "init", "--name", "support-agent", "--path", str(path)],
    )
    assert init_result.exit_code == 0

    status_result = runner.invoke(cli, ["playbook", "status", "--path", str(path)])

    assert status_result.exit_code == 0
    for stage_name in ("dataset", "metric", "evaluator", "optimize", "gate"):
        assert stage_name in status_result.output


def test_playbook_validate_rejects_non_yaml_path(tmp_path: Path) -> None:
    path = tmp_path / "playbook.txt"
    path.write_text("agent: {}\nstages: {}\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(cli, ["playbook", "validate", "--path", str(path)])

    assert result.exit_code != 0
    assert "must end with .yaml or .yml" in result.output
