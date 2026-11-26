"""Committed hash-Bernoulli sampling executed locally."""

from __future__ import annotations

import contextlib
import decimal
import hashlib
import os
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
    a = CommittedSampler("0.25", KEY, window_id="w1")
    b = CommittedSampler(Decimal("0.25"), KEY, window_id="w1")
    ids = [f"u{i}" for i in range(200)]
    assert [a.keep(i) for i in ids] == [b.keep(i) for i in ids]
    assert [a.keep(i) for i in ids] == [a.keep(i) for i in ids]


@tvl_free
def test_empirical_rate():
    s = CommittedSampler("0.25", KEY, window_id="w1")
    kept = sum(s.keep(f"id-{i}") for i in range(20000))
    assert abs(kept / 20000 - 0.25) < 0.02


@pytest.mark.parametrize(
    ("rate", "h", "expected"),
    [
        ("0.1", 2**64 // 10, True),
        ("0.1", 2**64 // 10 + 1, False),
        ("0.999", 2**64 - 1, False),
        ("0.5", 2**63 - 1, True),
        ("0.5", 2**63, False),
        ("0.25", 2**62 - 1, True),
        ("0.25", 2**62, False),
    ],
)
@tvl_free
def test_rate_exact_at_boundaries(monkeypatch, rate, h, expected):
    monkeypatch.setattr(mod, "_hash_int", lambda key, window_id, unit_id: h)
    assert CommittedSampler(rate, KEY, window_id="w").keep("x") is expected


@tvl_free
def test_commitment_before_and_after():
    s = CommittedSampler("0.1", KEY, window_id="w9")
    before = s.commitment()
    assert before.mode == "hash_bernoulli_committed_v1"
    assert before.window_id == "w9"
    assert before.rate == "0.1"
    assert len(before.key_digest) == 64
    for i in range(50):
        s.keep(str(i))
    assert s.commitment() == before
    assert s.reveal() == KEY


def _window(rate="0.5", n=100, window="w"):
    s = CommittedSampler(rate, KEY, window_id=window)
    eligible = [f"u{i}" for i in range(n)]
    kept = [u for u in eligible if s.keep(u)]
    return s, eligible, kept


@tvl_free
def test_verify_honest_and_wrong_key():
    s, eligible, kept = _window()
    c = s.commitment()
    assert verify_sampling(c, s.reveal(), eligible, kept)
    assert not verify_sampling(c, b"x" * 32, eligible, kept)


@tvl_free
def test_verify_rejects_incomplete_audit():
    s, eligible, kept = _window()
    c = s.commitment()
    # dropping a kept unit from the report
    assert not verify_sampling(c, KEY, eligible, kept[1:])
    # adding a unit the sampler did not keep
    dropped = next(u for u in eligible if u not in kept)
    assert not verify_sampling(c, KEY, eligible, [*kept, dropped])
    # shrinking the eligible list (hide a dropped-by-report unit)
    hidden = kept[0]
    assert not verify_sampling(c, KEY, [u for u in eligible if u != hidden], kept)
    # kept not a subset of eligible
    assert not verify_sampling(c, KEY, [], kept)
    assert not verify_sampling(c, KEY, eligible[:5], kept)


@tvl_free
def test_verify_empty_eligible():
    s = CommittedSampler("0.5", KEY, window_id="w")
    assert verify_sampling(s.commitment(), KEY, [], [])
    assert not verify_sampling(s.commitment(), KEY, [], ["u1"])


@tvl_free
def test_verify_rejects_duplicate_ids():
    s, eligible, kept = _window()
    c = s.commitment()
    assert not verify_sampling(c, KEY, [*eligible, eligible[0]], kept)
    assert not verify_sampling(c, KEY, eligible, [*kept, kept[0]])


@tvl_free
def test_verify_requires_mode_and_real_str_ids():
    s, eligible, kept = _window()
    c = s.commitment()
    bad = mod.SamplingCommitment("other", c.window_id, c.rate, c.key_digest)
    assert not verify_sampling(bad, KEY, eligible, kept)
    assert not verify_sampling(c, KEY, [1, 2], [])  # type: ignore[list-item]
    assert not verify_sampling(c, KEY, eligible, [b"u1"])  # type: ignore[list-item]


@tvl_free
def test_verify_rejects_out_of_range_rate_commitment():
    s, eligible, kept = _window()
    c = s.commitment()
    for rate in ("0", "1", "abc", "-0.1"):
        bad = mod.SamplingCommitment(c.mode, c.window_id, rate, c.key_digest)
        assert not verify_sampling(bad, KEY, eligible, kept)


@tvl_free
def test_rate_serialization_independent_of_decimal_context():
    with decimal.localcontext() as ctx:
        ctx.prec = 1
        s = CommittedSampler("0.49", KEY, window_id="w")
        c = s.commitment()
        eligible = [f"u{i}" for i in range(100)]
        kept = [u for u in eligible if s.keep(u)]
        assert c.rate == "0.49"
        assert verify_sampling(c, KEY, eligible, kept)


@tvl_free
@pytest.mark.parametrize("rate", ["0", "1", "-0.1", "1.5", 0, 1])
def test_sampler_rate_must_be_strictly_inside(rate):
    with pytest.raises(TuningError) as exc:
        CommittedSampler(rate, KEY, window_id="w")
    assert exc.value.code == E_INVALID_TUNING


@tvl_free
def test_window_id_required_and_valid():
    with pytest.raises(TypeError):
        CommittedSampler("0.5", KEY)  # type: ignore[call-arg]
    for bad in ("", "a\x00b"):
        with pytest.raises(TuningError) as exc:
            CommittedSampler("0.5", KEY, window_id=bad)
        assert exc.value.code == E_INVALID_TUNING
    s = CommittedSampler("0.5", KEY, window_id="w")
    with pytest.raises(TuningError):
        s.keep("a\x00b")
    with pytest.raises(TuningError):
        s.keep(1)  # type: ignore[arg-type]


@tvl_free
def test_key_must_be_32_plus_bytes():
    for bad in (b"short", "k" * 32, bytearray(b"k" * 32), None):
        if bad is None:
            continue
        with pytest.raises(TuningError):
            CommittedSampler("0.5", bad, window_id="w")  # type: ignore[arg-type]


@tvl_free
def test_message_encoding_is_unambiguous():
    # NUL-join would collide ("a\0b", "c") vs ("a", "b\0c"); length-prefix cannot.
    assert mod._hash_int(KEY, "ab", "c") != mod._hash_int(KEY, "a", "bc")


@tvl_free
def test_fixed_hmac_vector():
    key = bytes(range(32))
    assert mod._hash_int(key, "win-1", "unit-1") == 2324007256515656419
    assert CommittedSampler("0.5", key, window_id="win-1").keep("unit-1") is True
    assert CommittedSampler("0.1", key, window_id="win-1").keep("unit-1") is False


@tvl_free
def test_default_key_is_random():
    assert (
        CommittedSampler("0.5", window_id="w").reveal()
        != CommittedSampler("0.5", window_id="w").reveal()
    )


@tvl_free
def test_float_rate_refused():
    with pytest.raises(TuningError) as exc:
        CommittedSampler(0.25, KEY, window_id="w")
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


@tvl_free
def test_queue_refuses_unsafe_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(TuningError):
        LocalAuditQueue(link).append("u", {}, window_id="w")
    open_dir = tmp_path / "open"
    open_dir.mkdir()
    open_dir.chmod(0o755)
    with pytest.raises(TuningError):
        LocalAuditQueue(open_dir).append("u", {}, window_id="w")
    afile = tmp_path / "afile"
    afile.write_text("x")
    with pytest.raises(TuningError):
        LocalAuditQueue(afile).append("u", {}, window_id="w")


@tvl_free
def test_queue_refuses_symlink_and_open_files(tmp_path):
    d = tmp_path / "q"
    q = LocalAuditQueue(d)
    q.append("u", {}, window_id="w1")
    target = tmp_path / "target"
    target.write_text("")
    (d / "w2.jsonl").symlink_to(target)
    with pytest.raises(TuningError):
        q.append("u", {}, window_id="w2")
    (d / "w1.jsonl").chmod(0o644)
    with pytest.raises(TuningError):
        q.append("u", {}, window_id="w1")
    with pytest.raises(TuningError):
        q.list("w1")


@tvl_free
def test_queue_torn_tail_truncated_by_next_append(tmp_path):
    d = tmp_path / "q"
    q = LocalAuditQueue(d)
    q.append("u1", {"a": 1}, window_id="w")
    path = d / "w.jsonl"
    with open(path, "ab") as fh:
        fh.write(b'{"unit_id": "u2", "rec')
    assert q.list("w") == [{"unit_id": "u1", "record": {"a": 1}}]
    q.append("u3", {"c": 3}, window_id="w")
    assert path.read_bytes().count(b"\n") == 2
    assert q.list("w") == [
        {"unit_id": "u1", "record": {"a": 1}},
        {"unit_id": "u3", "record": {"c": 3}},
    ]


@tvl_free
def test_queue_torn_only_line_truncated_to_empty(tmp_path):
    d = tmp_path / "q"
    q = LocalAuditQueue(d)
    q.append("u1", {}, window_id="w")
    path = d / "w.jsonl"
    path.write_bytes(b'{"unit_id": "u')
    q.append("u2", {}, window_id="w")
    assert q.list("w") == [{"unit_id": "u2", "record": {}}]


@tvl_free
def test_queue_malformed_complete_line_raises(tmp_path):
    d = tmp_path / "q"
    q = LocalAuditQueue(d)
    q.append("u1", {}, window_id="w")
    with open(d / "w.jsonl", "ab") as fh:
        fh.write(b"SECRET-NOT-JSON\n")
    q.append("u3", {}, window_id="w")
    with pytest.raises(TuningError) as info:
        q.list("w")
    assert info.value.code == E_INVALID_TUNING
    assert "SECRET" not in str(info.value)


@tvl_free
def test_queue_append_holds_lock_around_write(tmp_path, monkeypatch):
    events = []
    real_write = os.write

    @contextlib.contextmanager
    def fake_lock(fd):
        events.append("enter")
        yield
        events.append("exit")

    def spy_write(fd, data):
        events.append("write")
        return real_write(fd, data)

    monkeypatch.setattr(mod, "_locked", fake_lock)
    monkeypatch.setattr(mod.os, "write", spy_write)
    LocalAuditQueue(tmp_path / "q").append("u", {}, window_id="w")
    assert events[0] == "enter" and events[-1] == "exit" and "write" in events[1:-1]


@tvl_free
def test_queue_list_refuses_symlinked_directory(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    LocalAuditQueue(real).append("u", {}, window_id="w")
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(TuningError):
        LocalAuditQueue(link).list("w")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX only")
@tvl_free
def test_queue_fifo_refused_without_hanging(tmp_path):
    d = tmp_path / "q"
    d.mkdir(mode=0o700)
    os.mkfifo(d / "w.jsonl", 0o600)
    q = LocalAuditQueue(d)
    with pytest.raises(TuningError):
        q.list("w")
    with pytest.raises(TuningError):
        q.append("u", {}, window_id="w")


@tvl_free
def test_verify_short_or_non_bytes_key_false():
    s = CommittedSampler("0.5", KEY, window_id="w")
    c = s.commitment()
    short = b"k" * 31
    c_short = mod.SamplingCommitment(
        c.mode, c.window_id, c.rate, hashlib.sha256(short).hexdigest()
    )
    assert verify_sampling(c_short, short, [], []) is False
    assert verify_sampling(c, bytearray(KEY), [], []) is False


@tvl_free
def test_queue_writes_whole_line_with_os_write_loop(tmp_path, monkeypatch):
    calls = []
    real_write = os.write

    def short_write(fd, data):
        calls.append(len(data))
        return real_write(fd, bytes(data)[:5])

    monkeypatch.setattr(mod.os, "write", short_write)
    q = LocalAuditQueue(tmp_path / "q")
    q.append("u1", {"a": 1}, window_id="w")
    monkeypatch.undo()
    assert len(calls) > 1
    assert q.list("w") == [{"unit_id": "u1", "record": {"a": 1}}]
