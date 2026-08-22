"""Tests for the per-example signals the SDK derives locally.

These signals exist so evaluator quality can be assessed WITHOUT shipping prompts,
completions or gold labels off the client. The first test is therefore the one that
matters most: content must not survive into the payload.

``build_example_signals`` resolves its HMAC key material from the project API key
(``BackendConfig.get_api_key``), so most tests here patch that classmethod to a fixed
test key via the module-scoped ``_default_api_key`` autouse fixture. Tests that
specifically probe keying (fail-closed with no key, different keys -> different
digests) override it explicitly.
"""

from __future__ import annotations

import json

import pytest

from traigent.api.types import ExampleResult
from traigent.config.backend_config import BackendConfig
from traigent.utils.outcome_signals import (
    build_example_signals,
    example_digest,
    output_digest,
    verified_match,
)

#: Fixed HMAC key material for tests that call ``example_digest``/``output_digest``
#: directly (bypassing ``build_example_signals``' key resolution). Arbitrary bytes --
#: these tests only care that a key was supplied, not what it is.
_TEST_KEY = b"unit-test-hmac-key-material"


def _set_api_key(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    monkeypatch.setattr(BackendConfig, "get_api_key", classmethod(lambda cls: value))


@pytest.fixture(autouse=True)
def _default_api_key(monkeypatch: pytest.MonkeyPatch):
    """Give ``build_example_signals`` a resolvable project key by default.

    Without this, every ``build_example_signals`` call in this module would
    fail closed (no API key configured in the test environment) and every
    test below would see ``{}``, defeating the point of the test.
    """
    _set_api_key(monkeypatch, "test-project-key-default")


def _result(**overrides):
    payload = {
        "example_id": "ex-1",
        "input_data": {"q": "2+2?"},
        "expected_output": "4",
        "actual_output": "4",
        "metrics": {"accuracy": 1.0},
        "execution_time": 0.1,
        "success": True,
    }
    payload.update(overrides)
    return ExampleResult(**payload)


# --- the property the whole design rests on -------------------------------


@pytest.mark.parametrize("field", ["input_data", "expected_output", "actual_output"])
def test_no_content_reaches_the_payload(field: str) -> None:
    secret = "CANARY-7f3a-CONFIDENTIAL-VALUE"
    value = {"q": secret} if field == "input_data" else secret
    payload = json.dumps(build_example_signals(_result(**{field: value})))
    assert secret not in payload
    for fragment in ("CANARY", "CONFIDENTIAL", "7f3a"):
        assert fragment not in payload


def test_digests_are_fixed_width_hex() -> None:
    signals = build_example_signals(_result())
    for key in ("example_digest", "output_digest"):
        assert len(signals[key]) == 64
        assert all(c in "0123456789abcdef" for c in signals[key])


def test_signal_key_id_present_alongside_the_digests() -> None:
    signals = build_example_signals(_result())
    assert "signal_key_id" in signals
    key_id = signals["signal_key_id"]
    assert len(key_id) == 12
    assert all(c in "0123456789abcdef" for c in key_id)


# --- identity behaviour ---------------------------------------------------


def test_same_example_same_digest_across_calls_and_objects() -> None:
    assert (
        build_example_signals(_result())["example_digest"]
        == (build_example_signals(_result())["example_digest"])
    )


def test_dict_key_order_does_not_change_the_digest() -> None:
    a = example_digest({"a": 1, "b": 2}, "x", _TEST_KEY)
    b = example_digest({"b": 2, "a": 1}, "x", _TEST_KEY)
    assert a == b


def test_a_different_output_changes_only_the_output_digest() -> None:
    same = build_example_signals(_result())
    other = build_example_signals(_result(actual_output="5"))
    assert same["example_digest"] == other["example_digest"]
    assert same["output_digest"] != other["output_digest"]


def test_a_different_expected_answer_changes_the_example_digest() -> None:
    assert example_digest({"q": "x"}, "A", _TEST_KEY) != example_digest(
        {"q": "x"}, "B", _TEST_KEY
    )


def test_digest_domains_are_separated() -> None:
    """Identical bytes under different purposes must not collide."""
    assert example_digest("v", None, _TEST_KEY) != output_digest("v", _TEST_KEY)


# --- the match bit --------------------------------------------------------


def test_match_and_mismatch() -> None:
    assert build_example_signals(_result())["verified_match"] == 1.0
    assert build_example_signals(_result(actual_output="5"))["verified_match"] == 0.0


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_no_usable_expected_answer_omits_the_key_entirely(empty) -> None:
    """Absent is NOT zero.

    Coercing 'cannot be checked' to 'checked and failed' would understate quality on
    exactly the datasets that lack gold labels.
    """
    signals = build_example_signals(_result(expected_output=empty))
    assert "verified_match" not in signals
    assert verified_match("anything", empty) is None


def test_an_errored_example_counts_as_a_non_match_not_as_uncheckable() -> None:
    signals = build_example_signals(_result(error_message="provider timeout"))
    assert signals["verified_match"] == 0.0


def test_match_uses_the_same_predicate_as_the_builtin_scorer() -> None:
    """Not a second implementation -- a second one would drift from the first."""
    from traigent.evaluators.base import _accuracy_values_match

    for actual, expected in (("4", "4"), (" 4 ", "4"), ("4", "5"), (4, "4")):
        expected_bit = 1.0 if _accuracy_values_match(actual, expected) else 0.0
        assert verified_match(actual, expected) == expected_bit


# --- robustness -----------------------------------------------------------


def test_unserialisable_content_still_yields_signals() -> None:
    """A weird output is still a real output; it must not break the run."""

    class Odd:
        def __repr__(self) -> str:
            return "<odd>"

    signals = build_example_signals(_result(actual_output=Odd()))
    assert len(signals["output_digest"]) == 64


def test_a_broken_example_result_yields_no_signals_rather_than_raising() -> None:
    class Exploding:
        @property
        def input_data(self):
            raise RuntimeError("boom")

    assert build_example_signals(Exploding()) == {}


# --- canonicalisation must be stable across processes ---------------------


def test_default_object_repr_does_not_produce_an_unstable_digest() -> None:
    """An object with NO custom ``__repr__`` renders as
    ``<ClassName object at 0x...>`` -- the address differs across processes (and
    across runs within one process, under ASLR), so the SAME logical example
    would digest differently every time. That must never happen: the signal is
    omitted, not emitted unstably.
    """

    class NoCustomRepr:
        pass

    address_bearing_repr = repr(NoCustomRepr())
    assert "0x" in address_bearing_repr  # sanity: this really is address-bearing

    signals = build_example_signals(_result(actual_output=NoCustomRepr()))
    assert "output_digest" not in signals


def test_a_deterministic_custom_repr_still_yields_a_stable_digest() -> None:
    """Not every non-JSON object is unstable -- one with a content-based
    ``__repr__`` (no address) must still get a real, reproducible digest."""

    class Deterministic:
        def __repr__(self) -> str:
            return "<deterministic-marker>"

    first = output_digest(Deterministic(), _TEST_KEY)
    second = output_digest(Deterministic(), _TEST_KEY)
    assert first is not None
    assert first == second


def test_set_member_order_does_not_change_the_digest() -> None:
    """Set iteration order depends on hash randomisation and is not guaranteed
    stable across processes; the digest must not depend on it."""
    a = output_digest({"x", "y", "z"}, _TEST_KEY)
    b = output_digest({"z", "y", "x"}, _TEST_KEY)
    assert a is not None
    assert a == b


# --- signals read plain dicts, the actual wire form -----------------------


def test_build_example_signals_reads_a_plain_dict_not_just_an_object() -> None:
    """Trial metadata stores example results as redacted ``to_dict()`` payloads
    (plain dicts), not ``ExampleResult`` objects. ``getattr``-based reads return
    ``None`` for every field on a dict, collapsing every example to the same
    digest -- this is the production bug the signals exist to prevent.
    """
    from traigent.utils.outcome_signals import _derive_signal_key

    payload = {
        "input_data": {"q": "2+2?"},
        "expected_output": "4",
        "actual_output": "4",
        "error_message": None,
    }
    signals = build_example_signals(payload)
    # The dict path resolves its key via BackendConfig, patched by the autouse
    # fixture to "test-project-key-default" -- derive the same key here so the
    # comparison isn't a hardcoded, unrelated key.
    reference_key = _derive_signal_key("test-project-key-default")
    assert signals["example_digest"] == example_digest(
        {"q": "2+2?"}, "4", reference_key
    )
    assert signals["verified_match"] == 1.0


# --- a systemic failure must be observable, without leaking content -------


def test_a_signal_build_failure_logs_no_content(caplog) -> None:
    import logging

    secret = "CANARY-OBSERVABILITY-CONTENT-DO-NOT-LOG"

    class Exploding:
        @property
        def input_data(self):
            raise RuntimeError(secret)

    with caplog.at_level(logging.WARNING, logger="traigent.utils.outcome_signals"):
        result = build_example_signals(Exploding())

    assert result == {}
    log_text = caplog.text
    assert secret not in log_text
    # The failure must be observable (not merely silent), but content-free.
    assert "RuntimeError" in log_text


# --- keying: the confirmation-oracle fix (FIX 5) ---------------------------


def test_no_api_key_means_no_signals_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-closed: with no project API key, NONE of the four signal keys are
    emitted -- never a silent downgrade to an unkeyed (oracle-able) digest."""
    _set_api_key(monkeypatch, None)

    signals = build_example_signals(_result())

    assert signals == {}
    for key in ("example_digest", "output_digest", "verified_match", "signal_key_id"):
        assert key not in signals


def test_different_api_keys_give_different_digests_for_identical_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole point of keying: two projects (or one project before/after a
    key rotation) must not be able to compare digests over the same content and
    learn anything -- their digests must differ."""
    _set_api_key(monkeypatch, "project-key-alpha")
    signals_alpha = build_example_signals(_result())

    _set_api_key(monkeypatch, "project-key-beta")
    signals_beta = build_example_signals(_result())

    assert signals_alpha["example_digest"] != signals_beta["example_digest"]
    assert signals_alpha["output_digest"] != signals_beta["output_digest"]
    assert signals_alpha["signal_key_id"] != signals_beta["signal_key_id"]


def test_key_derivation_is_stable_across_calls_and_a_fresh_cache_for_the_same_key() -> (
    None
):
    """The KDF-derived key/id for a given API key must be reproducible --
    within a process (repeated calls, the lru_cache hit path) and starting from
    a fresh, empty cache (the lru_cache miss path, e.g. process restart) --
    while a different key still produces different material. Exercises the
    real derivation directly (not mocked), since this is exactly the KDF-cost
    contract ``_cached_key_pair`` exists to pay only once per key.
    """
    from traigent.utils.outcome_signals import _cached_key_pair, _derive_signal_key

    # Within-process stability, including repeated hits on the same cache entry.
    first_call = _cached_key_pair("stability-test-key")
    second_call = _cached_key_pair("stability-test-key")
    assert first_call == second_call

    # Stability is a property of the DERIVATION, not of the cache holding onto
    # an object: clearing the cache and recomputing from scratch (the
    # fresh-cache / cold-process case) must reproduce the identical key/id.
    _cached_key_pair.cache_clear()
    after_clear = _cached_key_pair("stability-test-key")
    assert after_clear == first_call

    # A different key must diverge -- the cache must not be leaking material
    # across distinct API key values.
    different_key_pair = _cached_key_pair("a-completely-different-key")
    assert different_key_pair != first_call
    assert different_key_pair[0] != first_call[0]
    assert different_key_pair[1] != first_call[1]

    # And the uncached derivation function agrees with the cached wrapper --
    # the cache is a pure memoization, not an alternate code path.
    assert _derive_signal_key("stability-test-key") == first_call[0]


def test_signal_key_id_is_not_the_api_key_or_derived_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The id must be a one-way tag: it cannot embed the key material itself."""
    api_key = "super-secret-project-api-key-do-not-leak"
    _set_api_key(monkeypatch, api_key)

    signals = build_example_signals(_result())

    assert api_key not in signals["signal_key_id"]
    assert len(signals["signal_key_id"]) < len(api_key)
