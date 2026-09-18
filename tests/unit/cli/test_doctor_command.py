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
from urllib.parse import urlsplit

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


class TestVendorKeyCheckCoversEverySupportedVendor:
    """#1778 "vendor completeness": the 6-vendor hardcoded list previously
    silently un-checked HuggingFace, plus Bedrock/Azure OpenAI/OpenRouter,
    which the issue names explicitly. The presence check now derives from the
    SDK's canonical provider-support table (`provider_support.PROVIDER_SPECS`)
    instead of a literal, and a handful of documented non-table credential
    markers cover the vendors that authenticate some other way.
    """

    def _auth_row(self, payload: dict) -> dict:
        auth_rows = [c for c in payload["checks"] if c["category"] == "Auth"]
        return next(r for r in auth_rows if "vendor api key" in r["message"].lower())

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_a_huggingface_only_env_passes_the_vendor_check(
        self, runner: CliRunner
    ) -> None:
        """HF_TOKEN was never in the old 6-vendor literal list at all."""
        env = {**_clean_env(), "HF_TOKEN": "hf_" + "x" * 32}
        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(doctor, ["--json"])
        payload = json.loads(result.stdout)
        assert self._auth_row(payload)["status"] == "PASS"

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_a_bedrock_only_env_passes_the_vendor_check(
        self, runner: CliRunner
    ) -> None:
        """Bedrock authenticates via the AWS credential chain, not a
        provider_support.py env key -- doctor still must not call this a
        missing vendor key."""
        env = {**_clean_env(), "AWS_ACCESS_KEY_ID": "not-a-real-credential-marker"}
        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(doctor, ["--json"])
        payload = json.loads(result.stdout)
        assert self._auth_row(payload)["status"] == "PASS"

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_an_openrouter_only_env_passes_the_vendor_check(
        self, runner: CliRunner
    ) -> None:
        """OpenRouter has no ProviderSpec at all (reached only as an
        OpenAI-compatible client) -- issue #1778 names it explicitly."""
        env = {
            **_clean_env(),
            "OPENROUTER_API_KEY": "sk-or-" + "x" * 32,  # pragma: allowlist secret
        }
        with patch.dict(os.environ, env, clear=True):
            result = runner.invoke(doctor, ["--json"])
        payload = json.loads(result.stdout)
        assert self._auth_row(payload)["status"] == "PASS"

    @patch.dict(os.environ, _clean_env(), clear=True)
    def test_still_warns_with_no_vendor_credential_at_all(
        self, runner: CliRunner
    ) -> None:
        """Control: broadening the list must not make the WARN unreachable."""
        with patch.dict(os.environ, _clean_env(), clear=True):
            result = runner.invoke(doctor, ["--json"])
        payload = json.loads(result.stdout)
        assert self._auth_row(payload)["status"] == "WARN"


# ===========================================================================
# Secret safety, offline honesty, and checks that could PASS without checking.
#
# The #2366 review returned NO-GO on all three. `traigent doctor --json` is
# built to be pasted into a ticket, so a leak here travels.
# ===========================================================================

SENTINEL = "tg_CANARY_0123456789abcdefSECRET"  # pragma: allowlist secret


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

        # `result.output` (combined stdout+stderr) is deliberate here and
        # stricter than `.stdout`: a leaked key is a leak on EITHER stream.
        # Tests that PARSE the report use `.stdout`, because on click >= 8.2
        # `.output` interleaves the SDK's stderr log line with the JSON.
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
            "cannot reach https://admin:hunter2@backend.example.com/v1",  # pragma: allowlist secret
            environ={},
        )
        assert "hunter2" not in out  # pragma: allowlist secret

        url = urlsplit(out.removeprefix("cannot reach "))
        assert url.hostname == "backend.example.com"
        assert url.username == "admin"
        assert url.password != "hunter2"  # pragma: allowlist secret

    def test_scrub_leaves_ordinary_text_alone(self) -> None:
        from traigent.utils.diagnostics import scrub

        message = "No module named 'my_scorer'"
        assert scrub(message, environ={}) == message

    def test_backend_url_credentials_are_not_reported(self, monkeypatch) -> None:
        """The leak the old `"KEY" in var_name` rule could not see."""
        from urllib.parse import urlsplit

        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        monkeypatch.setenv(
            "TRAIGENT_BACKEND_URL",
            "https://svc:s3cr3t-pass@backend.example.com",  # pragma: allowlist secret
        )
        report = DiagnosticReport()
        TraigentDiagnostics._check_environment(report)

        assert "s3cr3t-pass" not in json.dumps(  # pragma: allowlist secret
            report.to_dict()
        )

        # And the host survives, so the check is still useful. Parsed rather
        # than substring-matched, for the reason given above.
        reported = next(
            entry["message"]
            for entry in report.successes
            if entry["message"].startswith("TRAIGENT_BACKEND_URL = ")
        )
        url = urlsplit(reported.removeprefix("TRAIGENT_BACKEND_URL = "))
        assert url.hostname == "backend.example.com"
        assert url.password != "s3cr3t-pass"  # pragma: allowlist secret

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
    """The probe used to write and unlink a FIXED `.test_permission`.

    The canary has to sit where the probe actually writes. An earlier version
    of this test planted it in the fake home ROOT while the probe writes into
    `<home>/.traigent`, so restoring the destructive implementation would not
    have touched it -- a test that could not fail. `Path.cwd` is patched too,
    because the probe also visits `cwd/logs` and `cwd/data` and would otherwise
    create directories in the real working tree during a test run.
    """

    @staticmethod
    def _probe_dirs(tmp_path: Path) -> list[Path]:
        return [tmp_path / ".traigent", tmp_path / "logs", tmp_path / "data"]

    def test_an_existing_test_permission_file_survives(self, tmp_path) -> None:
        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        victims = []
        for directory in self._probe_dirs(tmp_path):
            directory.mkdir(parents=True, exist_ok=True)
            victim = directory / ".test_permission"
            victim.write_text("the user's own file", encoding="utf-8")
            victims.append(victim)

        report = DiagnosticReport()
        with (
            patch.object(Path, "home", return_value=tmp_path),
            patch.object(Path, "cwd", return_value=tmp_path),
        ):
            TraigentDiagnostics._check_permissions(report)

        # The probe must have actually run somewhere, or the assertions below
        # are about a code path that never executed.
        assert any("Can write to" in s["message"] for s in report.successes)

        for victim in victims:
            assert victim.exists(), f"the diagnostic deleted {victim}"
            assert victim.read_text(encoding="utf-8") == "the user's own file"

    def test_the_probe_leaves_no_files_behind(self, tmp_path) -> None:
        """Cleanup, asserted where the probe actually writes."""
        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        before = {}
        for directory in self._probe_dirs(tmp_path):
            directory.mkdir(parents=True, exist_ok=True)
            before[directory] = set(directory.iterdir())

        with (
            patch.object(Path, "home", return_value=tmp_path),
            patch.object(Path, "cwd", return_value=tmp_path),
        ):
            TraigentDiagnostics._check_permissions(DiagnosticReport())

        for directory, contents in before.items():
            assert set(directory.iterdir()) == contents, (
                f"the probe left a file behind in {directory}"
            )


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
        payload = json.loads(result.stdout)
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
        payload = json.loads(result.stdout)
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
        payload = json.loads(result.stdout)
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
        payload = json.loads(result.stdout)
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


class TestOfflinePinsTheLiteLLMPriceTable:
    def test_offline_sets_the_local_cost_map_before_importing_litellm(
        self, runner, monkeypatch
    ) -> None:
        """LiteLLM fetches its price map on a cold import unless this is set.

        Found by review: skipping the connectivity probe is not the same as
        being offline. `--offline --model X` still reached the network through
        LiteLLM's own import, so the flag promised more than it delivered.
        """
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")

        runner.invoke(doctor, ["--json", "--offline", "--model", "gpt-4o"])

        assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"

    def test_without_offline_the_cost_map_is_left_alone(
        self, runner, monkeypatch
    ) -> None:
        """Control: the pin is scoped to --offline, not applied unconditionally."""
        monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")

        runner.invoke(doctor, ["--json", "--model", "gpt-4o"])

        assert "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ

    def test_the_offline_help_does_not_claim_more_than_it_delivers(self) -> None:
        """The old help said every check "is already local-only" -- it was not.

        `--scorer` executes the user's module, which this flag cannot sandbox,
        so the help has to say so rather than imply a guarantee.
        """
        offline_opt = next(p for p in doctor.params if "--offline" in p.opts)
        help_text = offline_opt.help or ""

        assert "no effect" not in help_text
        assert "scorer" in help_text.lower(), (
            "the caveat about --scorer executing user code must stay in the help"
        )


class TestEverySinkIsGuarded:
    """`scrub()` guarded ONE sink; an executing review found four more.

    Each test below replays a probe that leaked a sentinel before the fix. The
    lesson worth keeping: adding a sanitizer proves nothing until you enumerate
    the places output leaves the module. The first version sanitized scorer
    imports and left `--model`, `--dataset`, the environment URL and the
    initialization failure untouched.
    """

    def test_a_bare_token_userinfo_in_a_url_is_redacted(self, monkeypatch) -> None:
        """`https://<token>@host` -- no colon, so the first pattern missed it.

        This is the MORE common way an API URL carries a credential, and
        TRAIGENT_BACKEND_URL is the variable this module's own docstring names
        as the motivating leak.
        """
        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        # Deliberately NOT the `tg_`-prefixed SENTINEL: that matches the vendor
        # key-shape pattern, so the shape rule would redact it even with the URL
        # rule broken, and this test would pass without exercising the path it
        # names. Verified by mutation -- with SENTINEL, reverting the userinfo
        # pattern to colon-only left this test GREEN. A shapeless token isolates
        # the URL rule, which is the thing under test.
        shapeless = "OPAQUEUSERINFO1234567890"
        monkeypatch.setenv(
            "TRAIGENT_BACKEND_URL", f"https://{shapeless}@backend.example.com"
        )
        report = DiagnosticReport()
        TraigentDiagnostics._check_environment(report)

        assert shapeless not in json.dumps(report.to_dict())

        # Parsed, not substring-matched: `"backend.example.com" in blob` is also
        # satisfied by `https://evil.test/?x=backend.example.com`, which is why
        # CodeQL flags that shape (py/incomplete-url-substring-sanitization).
        # This is the second time this pattern slipped into this file; asserting
        # on the parsed host is both stronger and the thing actually meant.
        reported = next(
            entry["message"]
            for entry in report.successes
            if entry["message"].startswith("TRAIGENT_BACKEND_URL = ")
        )
        url = urlsplit(reported.removeprefix("TRAIGENT_BACKEND_URL = "))
        assert url.hostname == "backend.example.com", (
            "the host is the diagnostically useful part and must survive"
        )

    def test_a_password_containing_an_at_sign_is_masked_whole(self) -> None:
        """The old pattern stopped at the first `@` and left the tail exposed.

        Shapeless token for the same reason as above: a `tg_`-prefixed one
        would be caught by the vendor key-shape rule regardless of how the
        userinfo pattern behaves, so it would not test this at all.
        """
        from traigent.utils.diagnostics import scrub

        shapeless = "TAILOFTHEPASSWORD9876"
        out = scrub(f"cannot reach https://user:p@ss{shapeless}@host/v1", environ={})
        assert shapeless not in out

    def test_a_secret_passed_as_the_model_id_is_not_echoed(
        self, runner, monkeypatch
    ) -> None:
        """`--model` round-trips into the report; a user can paste anything."""
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")
        result = runner.invoke(doctor, ["--json", "--offline", "--model", SENTINEL])
        assert SENTINEL not in result.output

    def test_a_secret_in_the_dataset_path_is_not_echoed(
        self, runner, monkeypatch, tmp_path
    ) -> None:
        bad = tmp_path / f"{SENTINEL}.jsonl"
        bad.write_text("not valid jsonl\n", encoding="utf-8")
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")

        result = runner.invoke(doctor, ["--json", "--offline", "--dataset", str(bad)])
        assert SENTINEL not in result.output

    def test_an_initialization_failure_does_not_leak_the_key(self, monkeypatch) -> None:
        """Initialization reads config and talks to the backend.

        Its failure message is one of the likeliest places for a credential to
        surface, and it used raw `str(e)`.
        """
        from traigent.utils.diagnostics import DiagnosticReport, TraigentDiagnostics

        monkeypatch.setenv("TRAIGENT_API_KEY", SENTINEL)
        report = DiagnosticReport()
        with patch(
            "traigent.initialize",
            side_effect=RuntimeError(f"backend handshake rejected key {SENTINEL}"),
        ):
            TraigentDiagnostics._check_traigent_config(report)

        blob = json.dumps(report.to_dict())
        assert SENTINEL not in blob
        # The failure must still be legible, not swallowed along with the secret.
        assert "Failed to initialize" in blob


class TestScrubDoesNotDestroyDiagnostics:
    """The opposite failure, which the first design had badly.

    Masking every environment value meant a message's most useful part
    disappeared whenever it happened to contain a common variable's value --
    so the same error was legible on one machine and useless on another.
    Masking is now scoped to secret-NAMED variables plus credential shapes.
    """

    def test_a_terminal_type_survives(self) -> None:
        from traigent.utils.diagnostics import scrub

        message = "terminal type xterm-256color not found in terminfo database"
        assert scrub(message, environ={"TERM": "xterm-256color"}) == message

    def test_an_import_error_keeps_its_module_path(self) -> None:
        from traigent.utils.diagnostics import scrub

        message = "cannot import name 'x' from 'pkg' (/home/u/proj/pkg/__init__.py)"
        assert scrub(message, environ={"PWD": "/home/u/proj"}) == message

    def test_but_a_secret_named_variable_is_still_masked(self) -> None:
        """Control: the narrowing must not turn the masking off."""
        from traigent.utils.diagnostics import scrub

        out = scrub("auth rejected: s3cr3tvalue", environ={"MY_API_KEY": "s3cr3tvalue"})
        assert "s3cr3tvalue" not in out

    @pytest.mark.parametrize(
        "name", ["SVC_TOKEN", "DB_PASSWORD", "X_SECRET", "AUTH_HEADER", "SIG_SIGNATURE"]
    )
    def test_the_secret_name_patterns_each_match(self, name) -> None:
        from traigent.utils.diagnostics import scrub

        assert "swordfish" not in scrub("value=swordfish", environ={name: "swordfish"})


class TestSinksFoundInTheFourthRound:
    """Three more, after three rounds of "every sink is guarded".

    Worth keeping the count visible: each round I believed the enumeration was
    complete. The lesson is not "try harder" -- it is that a sanitizer needs a
    test per OUTPUT PATH, not per sanitizer.
    """

    def test_a_secret_passed_as_the_scorer_spec_is_not_echoed(
        self, runner, monkeypatch
    ) -> None:
        """`--scorer` round-trips like `--model` and `--dataset` already did."""
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")
        result = runner.invoke(doctor, ["--json", "--offline", "--scorer", SENTINEL])
        assert SENTINEL not in result.output

    def test_a_credential_in_a_url_query_parameter_is_redacted(self) -> None:
        """Userinfo is not the only place a URL carries a secret.

        `https://host/v1?token=...` is at least as common, and the userinfo
        pattern cannot see it. Matched on the parameter NAME so an opaque
        value with no recognizable shape is still caught.
        """
        from traigent.utils.diagnostics import redact_url_credentials

        out = redact_url_credentials("https://host/v1?token=opaque-canary-1234")
        assert "opaque-canary-1234" not in out
        assert "host/v1" in out

    def test_query_redaction_keeps_the_non_secret_parameters(self) -> None:
        """Control: it must not blank the whole query string."""
        from traigent.utils.diagnostics import redact_url_credentials

        out = redact_url_credentials("https://host/?api_key=abc123secret&model=gpt-4o")
        assert "abc123secret" not in out
        assert "model=gpt-4o" in out

    def test_a_clean_url_is_untouched(self) -> None:
        from traigent.utils.diagnostics import redact_url_credentials

        url = "https://host/v1?model=gpt-4o&n=3"
        assert redact_url_credentials(url) == url


class TestSinksFoundInTheFifthRound:
    """Five rounds. The count stays in the class name on purpose.

    This round came from an independent reviewer running the real CLI with
    sentinel values, not from reading the sanitizer. All three findings are
    places the sanitizer looked complete from the inside.
    """

    def test_a_secret_in_a_url_PATH_is_not_printed_in_the_environment_section(
        self, runner, monkeypatch
    ) -> None:
        """Userinfo and query masking cannot see a path.

        Measured on the previous head:
          TRAIGENT_BACKEND_URL=https://api.example.com/hook/<sentinel>
          -> "TRAIGENT_BACKEND_URL = https://api.example.com/hook/<sentinel>"
        printed in full, through the `*_URL` branch this feature added
        specifically so a URL could be shown safely.
        """
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")
        # Two things this canary must avoid, both of which would make the test
        # pass without the fix. It must not be the module-level SENTINEL --
        # that starts with `tg_`, which the vendor key-shape rule masks on its
        # own -- and its path must not be webhook-shaped, or the free-text
        # webhook rule would mask it and this test would prove nothing about
        # the environment section. `/v1/tenants/<id>` is the ordinary case:
        # a perfectly normal-looking path that happens to carry a secret.
        monkeypatch.setenv(
            "TRAIGENT_BACKEND_URL", "https://api.example.com/v1/tenants/CANARYVALUE"
        )
        result = runner.invoke(doctor, ["--json", "--offline"])
        assert "CANARYVALUE" not in result.output

    def test_the_host_still_survives_because_that_is_the_whole_point(
        self, runner, monkeypatch
    ) -> None:
        """Control: dropping the path must not drop the diagnosis.

        "Am I pointed at production or at localhost" is the question this line
        exists to answer. A fix that masked the whole URL would pass the test
        above and destroy the feature.
        """
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")
        monkeypatch.setenv(
            "TRAIGENT_BACKEND_URL", "https://api.example.com/v1/tenants/CANARYVALUE"
        )
        result = runner.invoke(doctor, ["--json", "--offline"])
        assert "api.example.com" in result.output

    @pytest.mark.parametrize(
        "url",
        [
            "https://hooks.slack.com/services/T00000000/B00000000/CANARYVALUE",
            "https://discord.com/api/webhooks/123456/CANARYVALUE",
            "https://example.com/hook/CANARYVALUE",
        ],
    )
    def test_a_webhook_path_credential_is_masked_in_free_text(self, url) -> None:
        """A webhook URL *is* the credential -- there is nothing else to
        authenticate with -- and it hides in the path, where neither the
        userinfo nor the query pattern looks."""
        from traigent.utils.diagnostics import redact_url_credentials

        assert "CANARYVALUE" not in redact_url_credentials(url)

    @pytest.mark.parametrize(
        "text",
        [
            "https://api.example.com/api/v1/runs/123 returned 404",
            "https://example.com/services/status is up",
        ],
    )
    def test_an_ordinary_url_path_survives(self, text) -> None:
        """Control: the path is usually the diagnosis.

        "404 on /api/v1/runs/123" is the useful half of that message, so a rule
        that masked every path would trade one silent failure for another.
        """
        from traigent.utils.diagnostics import redact_url_credentials

        assert redact_url_credentials(text) == text

    def test_a_webhook_env_value_is_masked_wherever_it_appears(self) -> None:
        """SLACK_WEBHOOK was printed in full while DATABASE_URL, SENTRY_DSN and
        MONGO_URI beside it were masked -- those carry the secret in userinfo,
        a webhook carries it in the path, and the NAME rule did not list it."""
        from traigent.utils.diagnostics import scrub

        hook = "https://hooks.example.com/abc/CANARYVALUE"
        out = scrub(f"POST failed: {hook}", environ={"SLACK_WEBHOOK": hook})
        assert "CANARYVALUE" not in out

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--model", "hf_CANARYVALUEabcdefghij"),
            ("--model", "AIzaCANARYVALUEabcdefghijklmnop"),
        ],
    )
    def test_a_vendor_key_doctor_knows_about_is_masked(
        self, runner, monkeypatch, flag, value
    ) -> None:
        """doctor's own `_ALL_VENDOR_KEY_MARKERS` names HF_TOKEN and
        GOOGLE_API_KEY as vendors it looks for, while their key shapes were
        missing from the masker -- so it echoed them back verbatim."""
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")
        result = runner.invoke(doctor, ["--json", "--offline", flag, value])
        assert "CANARYVALUE" not in result.output


class TestTheNameRuleMatchesSegmentsNotSubstrings:
    """The over-match the fourth round's commit message said it had fixed.

    `_SECRET_NAME_RE` was an unanchored substring search, so AUTHOR, XAUTHORITY,
    KEYBOARD_LAYOUT and MONKEY_HOME all counted as secret-named. With the
    minimum verbatim length at 4, a four-character value under any of them
    masked every occurrence of those four characters in the report.
    """

    @pytest.mark.parametrize(
        "name",
        ["GIT_AUTHOR_NAME", "AUTHOR", "XAUTHORITY", "KEYBOARD_LAYOUT", "MONKEY_HOME"],
    )
    def test_an_ordinary_variable_does_not_mask_the_message(self, name) -> None:
        from traigent.utils.diagnostics import scrub

        message = "No module named 'tests.helpers'; check your test layout"
        assert scrub(message, environ={name: "test"}) == message

    @pytest.mark.parametrize(
        "name",
        [
            "TRAIGENT_API_KEY",
            "HF_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "GITHUB_PAT",
            "SLACK_WEBHOOK",
            "DB_PASSWORD",
            "TRAIGENT_SESSION_ID",
        ],
    )
    def test_a_genuinely_secret_name_still_masks(self, name) -> None:
        """Control: narrowing the rule must not switch the masking off."""
        from traigent.utils.diagnostics import scrub

        assert "swordfish" not in scrub("value=swordfish", environ={name: "swordfish"})


class TestPricingKeepsProviderIdentity:
    """The over-match I introduced while fixing the original over-match.

    The first bug was bidirectional substring matching: `gpt-4o-of-my-own`
    reported PASS. My fix stripped ANY provider prefix, so `invented/gpt-4o`
    resolved to the bare `gpt-4o` and reported PASS for a provider that does
    not exist. Provider identity is part of the model's identity for pricing.
    """

    @pytest.mark.parametrize(
        "model_id,should_be_priced",
        [
            ("gpt-4o", True),
            ("openai/gpt-4o", True),
            ("invented/gpt-4o", False),
            ("gpt-4o-of-my-own", False),
        ],
    )
    def test_pricing_lookup(self, runner, monkeypatch, model_id, should_be_priced):
        litellm = pytest.importorskip("litellm")
        if "gpt-4o" not in litellm.model_cost:
            pytest.skip("this litellm build has no gpt-4o price entry")
        monkeypatch.setenv("TRAIGENT_SKIP_DOTENV", "true")

        result = runner.invoke(doctor, ["--json", "--offline", "--model", model_id])
        payload = json.loads(result.stdout)
        priced = any(
            c["category"] == "Model"
            and c["status"] == "PASS"
            and "pricing coverage" in c["message"]
            for c in payload["checks"]
        )
        assert priced is should_be_priced, (
            f"{model_id!r}: expected priced={should_be_priced}, got {priced}"
        )
