"""CLI tests for the ``traigent doctor`` command (issue #1778).

Covers: the command exists and is wired into the CLI, exits non-zero on
FAIL, `--strict` promotes WARN to a failing exit code, `--json` emits a
parseable report, and the model/dataset/scorer checks work when their
inputs are given and SKIP cleanly when they are not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from traigent.cli.doctor_command import DoctorReport, doctor
from traigent.cli.main import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _clean_env() -> dict[str, str]:
    """Environment with every check-relevant var absent, mock mode off."""
    return {"TRAIGENT_SKIP_DOTENV": "true"}


class TestDoctorReport:
    """Unit tests for the DoctorReport aggregation helper."""

    def test_has_fail_and_has_warn(self) -> None:
        report = DoctorReport()
        report.add("Test", "PASS", "ok")
        assert not report.has_fail
        assert not report.has_warn

        report.add("Test", "WARN", "careful")
        assert report.has_warn
        assert not report.has_fail

        report.add("Test", "FAIL", "broken")
        assert report.has_fail

    def test_to_dict_summary_counts(self) -> None:
        report = DoctorReport()
        report.add("A", "PASS", "1")
        report.add("A", "PASS", "2")
        report.add("A", "WARN", "3")
        report.add("A", "FAIL", "4")
        report.add("A", "SKIP", "5")

        summary = report.to_dict()["summary"]
        assert summary == {"total": 5, "pass": 2, "warn": 1, "fail": 1, "skip": 1}


class TestDoctorCommandWiring:
    """The command must be registered under both `doctor` and `diagnose`."""

    def test_doctor_is_registered_on_cli(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["doctor", "--help"])
        assert result.exit_code == 0
        assert (
            "doctor" in result.output.lower() or "PASS/WARN/FAIL/SKIP" in result.output
        )

    def test_diagnose_alias_is_registered_on_cli(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["diagnose", "--help"])
        assert result.exit_code == 0


class TestDoctorExitCodes:
    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_exits_nonzero_on_fail(self, runner: CliRunner) -> None:
        # No OPENAI_API_KEY -> diagnose() reports a FAIL for the required
        # OPENAI_API_KEY environment variable.
        result = runner.invoke(doctor, [])
        assert result.exit_code == 1

    @patch.dict(
        os.environ,
        {**_clean_env(), "OPENAI_API_KEY": "sk-test"},  # pragma: allowlist secret
        clear=True,
    )
    def test_exits_zero_when_only_warnings(self, runner: CliRunner) -> None:
        result = runner.invoke(doctor, [])
        assert result.exit_code == 0

    @patch.dict(
        os.environ,
        {**_clean_env(), "OPENAI_API_KEY": "sk-test"},  # pragma: allowlist secret
        clear=True,
    )
    def test_strict_promotes_warn_to_failing_exit(self, runner: CliRunner) -> None:
        result = runner.invoke(doctor, ["--strict"])
        assert result.exit_code == 1


class TestDoctorJsonOutput:
    @patch.dict(
        os.environ,
        {**_clean_env(), "OPENAI_API_KEY": "sk-test"},  # pragma: allowlist secret
        clear=True,
    )
    def test_json_output_is_parseable_and_has_expected_shape(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(doctor, ["--json"])
        payload = json.loads(result.stdout)

        assert "checks" in payload
        assert "summary" in payload
        assert all(
            c["status"] in ("PASS", "WARN", "FAIL", "SKIP") for c in payload["checks"]
        )


class TestDoctorModelChecks:
    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_model_check_skipped_when_no_model_given(self, runner: CliRunner) -> None:
        result = runner.invoke(doctor, ["--json"])
        payload = json.loads(result.stdout)
        model_rows = [c for c in payload["checks"] if c["category"] == "Model"]
        assert len(model_rows) == 1
        assert model_rows[0]["status"] == "SKIP"

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_model_check_recognizes_known_model_and_pricing(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(doctor, ["--json", "--model", "gpt-4o-mini"])
        payload = json.loads(result.stdout)
        model_rows = [c for c in payload["checks"] if c["category"] == "Model"]
        # provider-recognition row + pricing-coverage row
        assert len(model_rows) == 2
        assert all(row["status"] == "PASS" for row in model_rows)

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_model_check_warns_on_unknown_model(self, runner: CliRunner) -> None:
        result = runner.invoke(
            doctor, ["--json", "--model", "definitely-not-a-real-model-id"]
        )
        payload = json.loads(result.stdout)
        model_rows = [c for c in payload["checks"] if c["category"] == "Model"]
        assert any(row["status"] == "WARN" for row in model_rows)


class TestDoctorDatasetChecks:
    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_dataset_check_skipped_when_no_dataset_given(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(doctor, ["--json"])
        payload = json.loads(result.stdout)
        dataset_rows = [c for c in payload["checks"] if c["category"] == "Dataset"]
        assert len(dataset_rows) == 1
        assert dataset_rows[0]["status"] == "SKIP"

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_dataset_check_passes_for_valid_dataset(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        dataset_file = tmp_path / "ds.jsonl"
        dataset_file.write_text(
            json.dumps({"input": {"text": "hi"}, "expected_output": "hello"}) + "\n"
        )

        result = runner.invoke(doctor, ["--json", "--dataset", str(dataset_file)])
        payload = json.loads(result.stdout)
        dataset_rows = [c for c in payload["checks"] if c["category"] == "Dataset"]
        assert len(dataset_rows) == 1
        assert dataset_rows[0]["status"] == "PASS"

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_dataset_check_fails_for_invalid_dataset(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        dataset_file = tmp_path / "bad.jsonl"
        dataset_file.write_text("not valid json\n")

        result = runner.invoke(doctor, ["--json", "--dataset", str(dataset_file)])
        payload = json.loads(result.stdout)
        dataset_rows = [c for c in payload["checks"] if c["category"] == "Dataset"]
        assert len(dataset_rows) == 1
        assert dataset_rows[0]["status"] == "FAIL"


class TestDoctorScorerChecks:
    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_scorer_check_skipped_when_no_scorer_given(self, runner: CliRunner) -> None:
        result = runner.invoke(doctor, ["--json"])
        payload = json.loads(result.stdout)
        scorer_rows = [c for c in payload["checks"] if c["category"] == "Scorer"]
        assert len(scorer_rows) == 1
        assert scorer_rows[0]["status"] == "SKIP"

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_scorer_check_passes_for_importable_callable(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(doctor, ["--json", "--scorer", "os.path:isabs"])
        payload = json.loads(result.stdout)
        scorer_rows = [c for c in payload["checks"] if c["category"] == "Scorer"]
        assert len(scorer_rows) == 1
        assert scorer_rows[0]["status"] == "PASS"

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_scorer_check_fails_for_missing_colon(self, runner: CliRunner) -> None:
        result = runner.invoke(doctor, ["--json", "--scorer", "not_a_valid_spec"])
        payload = json.loads(result.stdout)
        scorer_rows = [c for c in payload["checks"] if c["category"] == "Scorer"]
        assert len(scorer_rows) == 1
        assert scorer_rows[0]["status"] == "FAIL"

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_scorer_check_fails_for_unimportable_module(
        self, runner: CliRunner
    ) -> None:
        result = runner.invoke(doctor, ["--json", "--scorer", "no_such_module_xyz:fn"])
        payload = json.loads(result.stdout)
        scorer_rows = [c for c in payload["checks"] if c["category"] == "Scorer"]
        assert len(scorer_rows) == 1
        assert scorer_rows[0]["status"] == "FAIL"


class TestDoctorKeyChecks:
    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_traigent_key_recognized_prefix(self, runner: CliRunner) -> None:
        env = {
            **_clean_env(),
            "TRAIGENT_API_KEY": "uk_" + "0" * 43,  # pragma: allowlist secret
            "OPENAI_API_KEY": "sk-test",  # pragma: allowlist secret
        }
        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(doctor, ["--json"])
        payload = json.loads(result.stdout)
        auth_rows = [c for c in payload["checks"] if c["category"] == "Auth"]
        traigent_row = next(r for r in auth_rows if "TRAIGENT_API_KEY" in r["message"])
        assert traigent_row["status"] == "PASS"

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_traigent_key_unrecognized_prefix_fails(self, runner: CliRunner) -> None:
        env = {
            **_clean_env(),
            "TRAIGENT_API_KEY": "not-a-known-prefix",  # pragma: allowlist secret
            "OPENAI_API_KEY": "sk-test",  # pragma: allowlist secret
        }
        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(doctor, ["--json"])
        payload = json.loads(result.stdout)
        auth_rows = [c for c in payload["checks"] if c["category"] == "Auth"]
        traigent_row = next(r for r in auth_rows if "TRAIGENT_API_KEY" in r["message"])
        assert traigent_row["status"] == "FAIL"
