"""Committed hash-Bernoulli sampling executed locally."""

from __future__ import annotations

import stat
from decimal import Decimal

import pytest

from traigent.tuning import (
    CommittedSampler,
    LocalAuditQueue,
    TuningError,
)
from traigent.tuning import audit_sampling as mod
from traigent.tuning import sampler_from_monitor, verify_sampling
from traigent.tuning.errors import E_FLOAT_RATE, E_INVALID_TUNING

from .conftest import tvl_free

KEY = b"k" * 32


@tvl_free
def test_deterministic():
    a = CommittedSampler("0.25", KEY, "w1")
    b = CommittedSampler(Decimal("0.25"), KEY, "w1")
    ids = [f"u{i}" for i in range(200)]
    assert [a.keep(i) for i in ids] == [b.keep(i) for i in ids]
    assert [a.keep(i) for i in ids] == [a.keep(i) for i in ids]


@tvl_free
def test_empirical_rate():
    s = CommittedSampler("0.25", KEY, "w1")
    kept = sum(s.keep(f"id-{i}") for i in range(20000))
    assert abs(kept / 20000 - 0.25) < 0.02


@pytest.mark.parametrize(
    ("rate", "h", "expected"),
    [
        ("0", 0, False),
        ("1", 2**64 - 1, True),
        ("0.5", 2**63 - 1, True),
        ("0.5", 2**63, False),
        ("0.25", 2**62 - 1, True),
        ("0.25", 2**62, False),
    ],
)
@tvl_free
def test_rate_exact_at_boundaries(monkeypatch, rate, h, expected):
    monkeypatch.setattr(mod, "_hash_int", lambda key, window_id, unit_id: h)
    assert CommittedSampler(rate, KEY, "w").keep("x") is expected


@tvl_free
def test_commitment_before_and_after():
    s = CommittedSampler("0.1", KEY, "w9")
    before = s.commitment()
    assert before.mode == "hash_bernoulli_committed_v1"
    assert before.window_id == "w9"
    assert before.rate == "0.1"
    assert len(before.key_digest) == 64
    for i in range(50):
        s.keep(str(i))
    assert s.commitment() == before
    assert s.reveal() == KEY


@tvl_free
def test_verify_honest_flipped_and_wrong_key():
    s = CommittedSampler("0.5", KEY, "w")
    decisions = {f"u{i}": s.keep(f"u{i}") for i in range(100)}
    c = s.commitment()
    assert verify_sampling(c, s.reveal(), decisions)
    flipped = dict(decisions)
    flipped["u3"] = not flipped["u3"]
    assert not verify_sampling(c, s.reveal(), flipped)
    assert not verify_sampling(c, b"x" * 32, decisions)


@tvl_free
def test_default_key_is_random():
    assert (
        CommittedSampler("0.5", window_id="w").reveal()
        != CommittedSampler("0.5", window_id="w").reveal()
    )


@tvl_free
def test_float_rate_refused():
    with pytest.raises(TuningError) as exc:
        CommittedSampler(0.25, KEY, "w")
    assert exc.value.code == E_FLOAT_RATE


@tvl_free
def test_missing_sampling_is_none():
    assert sampler_from_monitor({"evidence": {}}, window_id="w") is None
    assert sampler_from_monitor({}, window_id="w") is None


@tvl_free
def test_wrong_mode_and_bad_rate_refused():
    with pytest.raises(TuningError) as exc:
        sampler_from_monitor(
            {"evidence": {"sampling": {"mode": "other", "rate": 0.1}}}, window_id="w"
        )
    assert exc.value.code == E_INVALID_TUNING
    with pytest.raises(TuningError):
        sampler_from_monitor(
            {
                "evidence": {
                    "sampling": {"mode": "hash_bernoulli_committed_v1", "rate": "abc"}
                }
            },
            window_id="w",
        )


@tvl_free
def test_sampler_from_monitor_reads_rate():
    m = {
        "evidence": {"sampling": {"mode": "hash_bernoulli_committed_v1", "rate": 0.25}}
    }
    s = sampler_from_monitor(m, window_id="w", key=KEY)
    assert s is not None and s.commitment().rate == "0.25"


@tvl_free
def test_queue_permissions_and_roundtrip(tmp_path):
    q = LocalAuditQueue(tmp_path / "audit")
    q.append("u1", {"a": 1}, window_id="w1")
    q.append("u2", {"b": "x"}, window_id="w1")
    assert stat.S_IMODE((tmp_path / "audit").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "audit" / "w1.jsonl").stat().st_mode) == 0o600
    assert q.list("w1") == [
        {"unit_id": "u1", "record": {"a": 1}},
        {"unit_id": "u2", "record": {"b": "x"}},
    ]
    assert q.list("other") == []
