"""The approver authority, the event verifier and the authority config file."""

from __future__ import annotations

import pytest

from traigent.tuning.authority import (
    LOCAL_TRUSTED,
    PLATFORM,
    ApproverAuthority,
    EventVerifier,
    authority_from_config,
    load_authority,
    local_trusted_refused,
    sign_event,
    signing_input,
)
from traigent.tuning.errors import E_AUTHORITY_CONFIG, TuningError

from .conftest import tvl_free


def _issuers(var: str) -> dict:
    spec = {"secret_env": var}  # pragma: allowlist secret
    return {"event_issuers": {"ci": spec}}


def _refused(config: dict) -> TuningError:
    with pytest.raises(TuningError) as err:
        authority_from_config(config)
    assert err.value.code == E_AUTHORITY_CONFIG
    return err.value


@tvl_free
class TestLocalTrustedRefused:
    def test_a_non_production_process_and_store_is_trusted(self) -> None:
        assert local_trusted_refused("development") is False

    def test_a_store_without_a_recorded_environment_is_refused(self) -> None:
        assert local_trusted_refused(None) is True

    def test_a_production_store_is_refused_even_in_a_development_process(self) -> None:
        assert local_trusted_refused("production") is True

    def test_an_unknown_store_environment_is_refused(self) -> None:
        assert local_trusted_refused("somewhere-else") is True

    def test_an_unset_process_environment_counts_as_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("ENVIRONMENT", "TRAIGENT_ENVIRONMENT", "TRAIGENT_ENV"):
            monkeypatch.delenv(name, raising=False)
        assert local_trusted_refused("development") is True


@tvl_free
class TestApproverAuthority:
    def test_unknown_mode_is_refused(self) -> None:
        with pytest.raises(TuningError) as err:
            ApproverAuthority(frozenset({"a"}), "magic")
        assert err.value.code == E_AUTHORITY_CONFIG

    def test_an_empty_approver_list_is_refused(self) -> None:
        with pytest.raises(TuningError) as err:
            ApproverAuthority(frozenset())
        assert err.value.code == E_AUTHORITY_CONFIG

    def test_platform_mode_needs_an_authenticator(self) -> None:
        with pytest.raises(TuningError) as err:
            ApproverAuthority(frozenset({"a"}), PLATFORM)
        assert err.value.code == E_AUTHORITY_CONFIG

    def test_authorized_is_membership_of_a_string_reference(self) -> None:
        authority = ApproverAuthority(frozenset({"alice"}))
        assert authority.authorized("alice") is True
        assert authority.authorized("bob") is False
        assert authority.authorized(["alice"]) is False

    def test_local_trusted_authenticates_a_listed_approver_in_development(
        self,
    ) -> None:
        authority = ApproverAuthority(frozenset({"alice"}))
        assert authority.authenticated("alice", {}, "development") is True
        assert authority.authenticated("bob", {}, "development") is False

    def test_local_trusted_authenticates_nobody_in_a_production_store(self) -> None:
        authority = ApproverAuthority(frozenset({"alice"}))
        assert authority.authenticated("alice", {}, "production") is False

    def test_a_non_string_approver_is_never_authenticated(self) -> None:
        authority = ApproverAuthority(frozenset({"alice"}))
        assert authority.authenticated(42, {}, "development") is False

    def test_platform_authenticator_must_answer_a_literal_true(self) -> None:
        seen: list[tuple[str, dict]] = []

        def yes(approver: str, context: dict) -> bool:
            seen.append((approver, dict(context)))
            return True

        authority = ApproverAuthority(frozenset({"alice"}), PLATFORM, yes)
        assert authority.authenticated("carol", {"run": "r1"}) is True
        # the platform decides even for a person who is not listed: authorization
        # is a separate check
        assert seen == [("carol", {"run": "r1"})]
        assert authority.authorized("carol") is False

    @pytest.mark.parametrize("answer", [1, "yes", None, [True]])
    def test_a_truthy_non_bool_authenticator_answer_is_refused(
        self, answer: object
    ) -> None:
        authority = ApproverAuthority(
            frozenset({"alice"}),
            PLATFORM,
            lambda approver, context: answer,  # type: ignore[arg-type,return-value]
        )
        assert authority.authenticated("alice", {}) is False

    def test_an_authenticator_error_never_authenticates(self) -> None:
        def boom(approver: str, context: dict) -> bool:
            raise RuntimeError("platform down")

        authority = ApproverAuthority(frozenset({"alice"}), PLATFORM, boom)
        assert authority.authenticated("alice", {}) is False

    def test_an_async_authenticator_is_not_a_yes_and_is_closed(self) -> None:
        async def hook(approver: str, context: dict) -> bool:
            return True

        authority = ApproverAuthority(
            frozenset({"alice"}),
            PLATFORM,
            hook,  # type: ignore[arg-type]
        )
        with _no_unawaited_warning():
            assert authority.authenticated("alice", {}) is False


class _no_unawaited_warning:
    """Fail the test if a coroutine was left un-awaited and un-closed."""

    def __enter__(self) -> None:
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error", RuntimeWarning)

    def __exit__(self, *exc: object) -> None:
        import gc

        gc.collect()
        self._ctx.__exit__(*exc)


@tvl_free
class TestEventSigning:
    ENVELOPE = {"issuer": "ci", "event": "model_released", "id": "e1"}

    def test_the_signing_input_leaves_out_the_proof_fields(self) -> None:
        with_proof = {**self.ENVELOPE, "signature": "abc", "authenticated": True}
        assert signing_input(with_proof) == signing_input(self.ENVELOPE)
        assert b"signature" not in signing_input(with_proof)

    def test_the_signing_input_is_independent_of_key_order(self) -> None:
        reordered = dict(reversed(list(self.ENVELOPE.items())))
        assert signing_input(reordered) == signing_input(self.ENVELOPE)

    def test_a_str_and_a_bytes_secret_sign_identically(self) -> None:
        assert sign_event(self.ENVELOPE, "k") == sign_event(self.ENVELOPE, b"k")

    def test_a_different_secret_signs_differently(self) -> None:
        assert sign_event(self.ENVELOPE, "k1") != sign_event(self.ENVELOPE, "k2")

    def test_verify_accepts_only_a_matching_signature_from_a_trusted_issuer(
        self,
    ) -> None:
        verifier = EventVerifier({"ci": "secret"})
        signed = {**self.ENVELOPE, "signature": sign_event(self.ENVELOPE, "secret")}
        assert verifier.verify(signed) is True
        assert verifier.issuers == frozenset({"ci"})

    def test_verify_refuses_a_tampered_envelope(self) -> None:
        verifier = EventVerifier({"ci": "secret"})
        signed = {**self.ENVELOPE, "signature": sign_event(self.ENVELOPE, "secret")}
        assert verifier.verify({**signed, "id": "e2"}) is False

    def test_verify_never_reads_the_authenticated_field(self) -> None:
        verifier = EventVerifier({"ci": "secret"})
        assert verifier.verify({**self.ENVELOPE, "authenticated": True}) is False
        assert (
            verifier.verify({**self.ENVELOPE, "authenticated": True, "signature": "00"})
            is False
        )

    def test_verify_refuses_an_unknown_issuer(self) -> None:
        verifier = EventVerifier({"ci": "secret"})
        other = {**self.ENVELOPE, "issuer": "other"}
        other["signature"] = sign_event(other, "secret")
        assert verifier.verify(other) is False

    def test_verify_refuses_non_string_issuer_or_signature(self) -> None:
        verifier = EventVerifier({"ci": "secret"})
        assert verifier.verify({"issuer": 1, "signature": "x"}) is False
        assert verifier.verify({"issuer": "ci", "signature": 5}) is False

    def test_an_issuer_with_an_empty_secret_is_not_trusted(self) -> None:
        assert EventVerifier({"ci": "", "ok": b"k"}).issuers == frozenset({"ok"})


@tvl_free
class TestAuthorityFromConfig:
    def test_an_empty_config_configures_nothing(self) -> None:
        assert authority_from_config({}) == (None, None)

    def test_approvers_build_a_local_trusted_authority(self) -> None:
        authority, verifier = authority_from_config({"approvers": ["a", "b"]})
        assert authority is not None and verifier is None
        assert authority.approvers == frozenset({"a", "b"})
        assert authority.mode == LOCAL_TRUSTED

    @pytest.mark.parametrize("approvers", ["alice", b"alice", 5])
    def test_approvers_must_be_a_list(self, approvers: object) -> None:
        assert "approvers must be a list" in str(_refused({"approvers": approvers}))

    def test_platform_mode_cannot_come_from_a_file(self) -> None:
        err = _refused({"approvers": ["a"], "mode": "platform"})
        assert "cannot come from a file" in str(err)

    def test_event_issuers_read_their_key_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TUNING_TEST_ISSUER_KEY", "s3")
        _, verifier = authority_from_config(_issuers("TUNING_TEST_ISSUER_KEY"))
        assert verifier is not None
        envelope = {"issuer": "ci", "id": "1"}
        assert verifier.verify({**envelope, "signature": sign_event(envelope, "s3")})

    def test_event_issuers_must_be_a_mapping(self) -> None:
        assert "event_issuers must be a mapping" in str(
            _refused({"event_issuers": ["ci"]})
        )

    @pytest.mark.parametrize("spec", [None, "TOKEN", {}, {"secret_env": ""}])
    def test_an_issuer_must_name_a_secret_env_variable(self, spec: object) -> None:
        assert "needs a secret_env variable name" in str(
            _refused({"event_issuers": {"ci": spec}})
        )

    def test_an_unset_secret_env_variable_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TUNING_TEST_UNSET_KEY", raising=False)
        err = _refused(_issuers("TUNING_TEST_UNSET_KEY"))
        assert "environment variable is not set" in str(err)
        assert "TUNING_TEST_UNSET_KEY" not in str(err)  # the error names no secret


@tvl_free
class TestLoadAuthority:
    def test_reads_a_yaml_file(self, tmp_path) -> None:
        path = tmp_path / "authority.yml"
        path.write_text("approvers:\n  - alice\n", encoding="utf-8")
        authority, verifier = load_authority(path)
        assert authority is not None and authority.authorized("alice")
        assert verifier is None

    def test_reads_a_json_file_given_as_a_string_path(self, tmp_path) -> None:
        path = tmp_path / "authority.json"
        path.write_text('{"approvers": ["bob"]}', encoding="utf-8")
        authority, _ = load_authority(str(path))
        assert authority is not None and authority.authorized("bob")

    def test_a_missing_file_is_a_closed_config_error(self, tmp_path) -> None:
        with pytest.raises(TuningError) as err:
            load_authority(tmp_path / "absent.yml")
        assert err.value.code == E_AUTHORITY_CONFIG
        assert "unreadable" in str(err.value)

    def test_invalid_yaml_is_a_closed_config_error(self, tmp_path) -> None:
        path = tmp_path / "bad.yml"
        path.write_text("approvers: [unclosed", encoding="utf-8")
        with pytest.raises(TuningError) as err:
            load_authority(path)
        assert err.value.code == E_AUTHORITY_CONFIG

    def test_a_non_mapping_document_is_refused(self, tmp_path) -> None:
        path = tmp_path / "list.yml"
        path.write_text("- alice\n", encoding="utf-8")
        with pytest.raises(TuningError) as err:
            load_authority(path)
        assert "must hold a mapping" in str(err.value)
