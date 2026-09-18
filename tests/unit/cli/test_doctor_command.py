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
    """A runner whose ``stdout`` really is stdout.

    ``traigent doctor --json`` writes JSON to stdout while the SDK's logging
    handler (``traigent/utils/logging.py``) writes to stderr, so piping the
    command into ``jq`` works. Under click < 8.2, however, ``CliRunner``
    defaults to ``mix_stderr=True`` and folds stderr INTO ``result.stdout``, so
    an INFO line like "Traigent SDK initialized successfully" lands in front of
    the JSON and every ``json.loads(result.stdout)`` in this file raises
    ``JSONDecodeError: Extra data``. Measured: 14 of 20 tests here fail on
    click 8.1.8 and pass on 8.2, with no code change -- the suite silently
    depended on the CI image's click version.

    ``mix_stderr`` was removed in click 8.2 (where the streams are already
    separate), hence the try/except rather than a version comparison.
    """
    try:
        return CliRunner(mix_stderr=False)  # type: ignore[call-arg]
    except TypeError:
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


# ===========================================================================
# Secret safety, offline honesty, and checks that could PASS without checking.
#
# The #2366 review returned NO-GO on all three. `traigent doctor --json` is
# built to be pasted into a ticket, so a leak here travels.
# ===========================================================================

SENTINEL = "tg_CANARY_0123456789abcdefSECRET"


class TestReportIsSecretSafe:
    def test_a_scorer_whose_import_raises_the_api_key_does_not_print_it(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        """The canary. This is the exact attack the previous code lost to.

        Importing a scorer runs the user's module. A module that raises an
        exception carrying a credential -- deliberately, or because some
        library echoed its config into the message -- had that text copied
        verbatim into the report and into ``--json``.
        """
        module_dir = tmp_path / "pkg"
        module_dir.mkdir()
        (module_dir / "leaky_scorer.py").write_text(
            "import os\nraise ImportError(os.environ['TRAIGENT_API_KEY'])\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(module_dir))
        monkeypatch.setenv("TRAIGENT_API_KEY", SENTINEL)
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")

        result = runner.invoke(
            doctor, ["--json", "--offline", "--scorer", "leaky_scorer:score"]
        )

        assert SENTINEL not in result.output, (
            "the API key reached the doctor report; --json is exactly what a "
            "user pastes into an issue"
        )
        # The failure must still be REPORTED -- redaction that hides the
        # problem as well as the secret is not a fix.
        assert "leaky_scorer" in result.output

    def test_scrub_masks_a_live_environment_value(self, monkeypatch) -> None:
        from traigent.utils.diagnostics import scrub

        env = {"TRAIGENT_API_KEY": SENTINEL}
        assert SENTINEL not in scrub(f"boom: {SENTINEL}", environ=env)

    def test_scrub_masks_url_credentials(self) -> None:
        """Password gone, host intact -- asserted by PARSING, not by substring.

        `"backend.example.com" in out` would also be satisfied by
        `https://evil.test/?x=backend.example.com`, which is why CodeQL flags
        that shape (py/incomplete-url-substring-sanitization). Parsing asserts
        the property the redaction actually has to hold.
        """
        from urllib.parse import urlsplit

        from traigent.utils.diagnostics import scrub

        out = scrub(
            "cannot reach https://admin:hunter2@backend.example.com/v1", environ={}
        )
        assert "hunter2" not in out

        url = urlsplit(out.removeprefix("cannot reach "))
        assert url.hostname == "backend.example.com"
        assert url.username == "admin"
        assert url.password != "hunter2"

    def test_scrub_leaves_ordinary_text_alone(self) -> None:
        from traigent.utils.diagnostics import scrub

        message = "No module named 'my_scorer'"
        assert scrub(message, environ={}) == message

    def test_backend_url_credentials_are_not_reported(self, monkeypatch) -> None:
        """The leak the old `"KEY" in var_name` rule could not see."""
        from urllib.parse import urlsplit

        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        monkeypatch.setenv(
            "TRAIGENT_BACKEND_URL", "https://svc:s3cr3t-pass@backend.example.com"
        )
        report = DiagnosticReport()
        TraigentDiagnostics._check_environment(report)

        assert "s3cr3t-pass" not in json.dumps(report.to_dict())

        # And the host survives, so the check is still useful. Parsed rather
        # than substring-matched, for the reason given above.
        reported = next(
            entry["message"]
            for entry in report.successes
            if entry["message"].startswith("TRAIGENT_BACKEND_URL = ")
        )
        url = urlsplit(reported.removeprefix("TRAIGENT_BACKEND_URL = "))
        assert url.hostname == "backend.example.com"
        assert url.password != "s3cr3t-pass"

    def test_an_api_key_value_is_never_quoted_in_the_report(self, monkeypatch) -> None:
        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        monkeypatch.setenv("TRAIGENT_API_KEY", SENTINEL)
        report = DiagnosticReport()
        TraigentDiagnostics._check_environment(report)
        assert SENTINEL not in json.dumps(report.to_dict())


class TestOfflineIsReal:
    def test_offline_opens_no_sockets(self, monkeypatch) -> None:
        """``--offline`` used to carry `noqa: ARG001` and do nothing."""
        import socket

        from traigent.utils.diagnostics import diagnose

        opened: list[object] = []

        def _record(*args, **kwargs):
            opened.append(args)
            raise OSError("blocked by test")

        monkeypatch.setattr(socket, "create_connection", _record)
        diagnose(offline=True)
        assert opened == [], f"offline run opened sockets: {opened}"

    def test_the_cli_flag_reaches_diagnose(self, runner, monkeypatch) -> None:
        """Drive the actual command, not just ``diagnose(offline=True)``.

        Found by a red control: reverting the CLI to call ``diagnose()`` with
        no argument left the direct-call test above green, because it never
        exercised the wiring. The `noqa: ARG001` bug was precisely a wiring
        bug, so the test has to go through click.
        """
        import socket

        opened: list[object] = []

        def _record(*args, **kwargs):
            opened.append(args)
            raise OSError("blocked by test")

        monkeypatch.setattr(socket, "create_connection", _record)
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")
        runner.invoke(doctor, ["--json", "--offline"])
        assert opened == [], (
            f"`doctor --offline` opened sockets: {opened}; the flag is not "
            "reaching diagnose()"
        )

    def test_without_the_flag_the_cli_does_open_sockets(
        self, runner, monkeypatch
    ) -> None:
        """Control for the test above, so it cannot pass vacuously."""
        import socket

        opened: list[object] = []

        def _record(*args, **kwargs):
            opened.append(args)
            raise OSError("blocked by test")

        monkeypatch.setattr(socket, "create_connection", _record)
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")
        runner.invoke(doctor, ["--json"])
        assert opened, "the CLI no longer reaches the network even without --offline"

    def test_the_socket_probe_actually_detects_a_connection(self, monkeypatch) -> None:
        """Red control: without offline, the same probe must see sockets.

        Otherwise the test above passes for the wrong reason -- e.g. if the
        network check moved and the monkeypatch no longer intercepts it.
        """
        import socket

        from traigent.utils.diagnostics import diagnose

        opened: list[object] = []

        def _record(*args, **kwargs):
            opened.append(args)
            raise OSError("blocked by test")

        monkeypatch.setattr(socket, "create_connection", _record)
        diagnose(offline=False)
        assert opened, (
            "the network check no longer goes through socket.create_connection"
        )


class TestPermissionProbeIsNonDestructive:
    def test_an_existing_test_permission_file_survives(
        self, tmp_path, monkeypatch
    ) -> None:
        """The probe used to write and unlink a FIXED `.test_permission`."""
        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        victim = tmp_path / ".test_permission"
        victim.write_text("the user's own file", encoding="utf-8")

        monkeypatch.setattr(
            TraigentDiagnostics,
            "_check_permissions",
            TraigentDiagnostics._check_permissions,
        )
        with patch.object(Path, "home", return_value=tmp_path):
            TraigentDiagnostics._check_permissions(DiagnosticReport())

        assert victim.exists(), "the diagnostic deleted a pre-existing user file"
        assert victim.read_text(encoding="utf-8") == "the user's own file"


class TestChecksCannotPassWithoutChecking:
    def test_an_invented_model_containing_a_priced_name_is_not_priced(
        self, runner, monkeypatch
    ) -> None:
        """Bidirectional substring matching reported PASS for a made-up id."""
        pytest.importorskip("litellm")
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")

        result = runner.invoke(
            doctor, ["--json", "--offline", "--model", "gpt-4o-of-my-own"]
        )
        payload = json.loads(result.output)
        pricing = [
            c
            for c in payload["checks"]
            if c["category"] == "Model" and "pricing" in c["message"]
        ]
        assert pricing, "no pricing check was reported"
        assert all(c["status"] != "PASS" for c in pricing), (
            f"an invented model id was reported as priced: {pricing}"
        )

    def test_a_real_model_is_still_reported_as_priced(
        self, runner, monkeypatch
    ) -> None:
        """Control for the test above: the tightened match must not reject everything."""
        litellm = pytest.importorskip("litellm")
        known = next(
            (k for k in ("gpt-4o", "gpt-4o-mini") if k in litellm.model_cost), None
        )
        if known is None:
            pytest.skip("no expected model in this litellm's price table")
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")

        result = runner.invoke(doctor, ["--json", "--offline", "--model", known])
        payload = json.loads(result.output)
        assert any(
            c["category"] == "Model"
            and c["status"] == "PASS"
            and "pricing" in c["message"]
            for c in payload["checks"]
        ), f"a real priced model was not recognized: {known}"

    def test_a_scorer_with_a_required_keyword_only_param_fails(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        """It would raise TypeError on the first trial; the check said PASS."""
        module_dir = tmp_path / "pkg2"
        module_dir.mkdir()
        (module_dir / "kwonly_scorer.py").write_text(
            "def score(output, expected, *, threshold):\n    return 1.0\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(module_dir))
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")

        result = runner.invoke(
            doctor, ["--json", "--offline", "--scorer", "kwonly_scorer:score"]
        )
        payload = json.loads(result.output)
        scorer_checks = [c for c in payload["checks"] if c["category"] == "Scorer"]
        assert any(c["status"] == "FAIL" for c in scorer_checks), (
            f"an uncallable scorer passed the preflight: {scorer_checks}"
        )

    def test_a_bindable_scorer_still_passes(
        self, runner, tmp_path, monkeypatch
    ) -> None:
        """Control: the tightened check must not reject a valid scorer."""
        module_dir = tmp_path / "pkg3"
        module_dir.mkdir()
        (module_dir / "ok_scorer.py").write_text(
            "def score(output, expected, *, threshold=0.5):\n    return 1.0\n",
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(module_dir))
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")

        result = runner.invoke(
            doctor, ["--json", "--offline", "--scorer", "ok_scorer:score"]
        )
        payload = json.loads(result.output)
        scorer_checks = [c for c in payload["checks"] if c["category"] == "Scorer"]
        assert any(c["status"] == "PASS" for c in scorer_checks), scorer_checks


class TestStrictIsUsable:
    def test_the_standing_chroma_advisory_does_not_fail_strict(self) -> None:
        """A gate that always fails gates nothing.

        The Chroma notice has identical text on every machine and describes our
        packaging, not the user's environment, so folding it as WARN made every
        completed `--strict` run exit 1.
        """
        from traigent.cli.doctor_command import DoctorReport, _fold_diagnostic_report
        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        diag = DiagnosticReport()
        TraigentDiagnostics._check_chroma_integration_availability(diag)
        report = DoctorReport()
        _fold_diagnostic_report(report, diag)

        assert not report.has_warn, (
            "the standing Chroma advisory still makes --strict exit 1"
        )
        assert any(
            TraigentDiagnostics.CHROMA_INTEGRATION_UNAVAILABLE in c.message
            for c in report.checks
        ), "the advisory must still be reported, just not as a gate failure"


class TestProviderKeysAreNotVendorSpecific:
    def test_anthropic_only_is_a_valid_setup(self, monkeypatch) -> None:
        """OPENAI_API_KEY was `required=True`, so an Anthropic user got a FAIL."""
        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-" + "x" * 32)

        report = DiagnosticReport()
        TraigentDiagnostics._check_provider_keys(report)
        assert not report.issues, report.issues

    def test_no_provider_key_at_all_is_still_an_issue(self, monkeypatch) -> None:
        """Control: the requirement was loosened, not removed."""
        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        for name in TraigentDiagnostics.PROVIDER_KEY_VARIABLES:
            monkeypatch.delenv(name, raising=False)

        report = DiagnosticReport()
        TraigentDiagnostics._check_provider_keys(report)
        assert report.issues
