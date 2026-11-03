"""Characterization tests for ``verify_integrity_manifest``.

Pin every accept/deny outcome (legacy artifact, unreadable manifest, disallowed
names, uncovered files, missing or unreadable files, digest mismatch), the exact
messages, the order of the checks and collaborator calls, and error chaining.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from traigent.utils import file_versioning as fv

MANIFEST = "integrity.sha256.json"
META = "metadata.json"
TRIALS = "trials.json.gz"
UNREADABLE = "Integrity manifest is unreadable"
NOT_ALLOWED = "Integrity manifest names a file that is not allowed"
NOT_COVERED = "Integrity manifest does not cover all result files"
MISSING = "A result file listed in the integrity manifest is missing"
UNVERIFIABLE = "A result file could not be read for verification"
FAILED = "Result file failed integrity verification"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_manifest(directory: Path, payload: Any) -> None:
    (directory / MANIFEST).write_text(json.dumps(payload), encoding="utf-8")


def _setup(tmp_path: Path) -> Path:
    (tmp_path / META).write_bytes(b"meta-bytes")
    (tmp_path / TRIALS).write_bytes(b"trial-bytes")
    _write_manifest(
        tmp_path,
        {
            "version": 1,
            "sha256": {META: _digest(b"meta-bytes"), TRIALS: _digest(b"trial-bytes")},
        },
    )
    return tmp_path


def _expect(directory: Path, required: Any, message: str) -> ValueError:
    with pytest.raises(ValueError) as info:
        fv.verify_integrity_manifest(directory, required)
    assert type(info.value) is ValueError
    assert str(info.value) == message
    return info.value


def test_no_manifest_is_a_legacy_artifact(tmp_path: Path) -> None:
    (tmp_path / META).write_bytes(b"x")
    assert fv.verify_integrity_manifest(tmp_path, [META]) is False
    assert fv.verify_integrity_manifest(tmp_path, []) is False


def test_valid_manifest_verifies_all_required(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    assert fv.verify_integrity_manifest(directory, [META, TRIALS]) is True
    assert fv.verify_integrity_manifest(directory, (TRIALS, META)) is True


def test_only_required_files_are_hashed(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    (directory / TRIALS).write_bytes(b"tampered")
    assert fv.verify_integrity_manifest(directory, [META]) is True
    _expect(directory, [META, TRIALS], FAILED)


def test_unrequired_listed_file_may_be_missing(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    (directory / TRIALS).unlink()
    assert fv.verify_integrity_manifest(directory, [META]) is True


def test_empty_required_list_checks_names_only(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    (directory / META).unlink()
    assert fv.verify_integrity_manifest(directory, []) is True


def test_empty_listing_with_empty_required_is_valid(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"sha256": {}})
    assert fv.verify_integrity_manifest(tmp_path, []) is True
    _expect(tmp_path, [META], NOT_COVERED)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"{",
        b"[]",
        b"null",
        b"5",
        b'"text"',
        b"{}",
        b'{"sha256": null}',
        b'{"sha256": []}',
        b'{"sha256": "x"}',
        b'{"sha256": 5}',
        b'{"sha256": {"metadata.json": 5}}',
        b'{"sha256": {"metadata.json": null}}',
        b'{"sha256": {"metadata.json": ["a"]}}',
        b'{"sha256": {"metadata.json": "ok", "trials.json.gz": 1}}',
        b"\xff\xfe\x00",
    ],
)
def test_unreadable_manifest_shapes(tmp_path: Path, raw: bytes) -> None:
    (tmp_path / MANIFEST).write_bytes(raw)
    error = _expect(tmp_path, [META], UNREADABLE)
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


def test_manifest_that_is_a_directory_is_unreadable(tmp_path: Path) -> None:
    (tmp_path / MANIFEST).mkdir()
    error = _expect(tmp_path, [], UNREADABLE)
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


def test_unreadable_manifest_is_reported_before_anything_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / MANIFEST).write_bytes(b"{")
    calls: list[str] = []

    def child(d: Path, name: object) -> Path:
        calls.append(str(name))
        return Path(str(name))

    monkeypatch.setattr(fv, "_integrity_child", child)
    _expect(tmp_path, [META], UNREADABLE)
    assert calls == []


@pytest.mark.parametrize(
    "name",
    [
        "other.json",
        "../metadata.json",
        "sub/metadata.json",
        "sub\\metadata.json",
        "..",
        "/metadata.json",
        "metadata.json ",
        "",
        "METADATA.JSON",
    ],
)
def test_disallowed_names_are_rejected_even_when_not_required(
    tmp_path: Path, name: str
) -> None:
    directory = _setup(tmp_path)
    _write_manifest(
        directory, {"sha256": {META: _digest(b"meta-bytes"), name: "0" * 64}}
    )
    error = _expect(directory, [META], NOT_ALLOWED)
    assert name == "" or name not in str(error)


def test_names_are_validated_before_coverage_is_checked(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"sha256": {"bad.json": "0" * 64}})
    _expect(tmp_path, [META], NOT_ALLOWED)


def test_required_file_missing_from_listing(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    _write_manifest(directory, {"sha256": {META: _digest(b"meta-bytes")}})
    _expect(directory, [META, TRIALS], NOT_COVERED)


def test_required_unknown_name_is_not_covered(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    _expect(directory, ["other.json"], NOT_COVERED)


def test_unhashable_required_entry_raises_type_error(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    required: Any = [["metadata.json"]]
    with pytest.raises(TypeError):
        fv.verify_integrity_manifest(directory, required)


def test_missing_required_file(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    (directory / META).unlink()
    _expect(directory, [META], MISSING)


def test_dangling_symlink_is_reported_missing(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    (directory / META).unlink()
    (directory / META).symlink_to(directory / "nowhere")
    _expect(directory, [META], MISSING)


def test_symlinked_file_is_refused(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    real = directory / "real-copy"
    real.write_bytes(b"meta-bytes")
    (directory / META).unlink()
    (directory / META).symlink_to(real)
    _expect(directory, [META], NOT_ALLOWED)


def test_directory_in_place_of_file_is_refused(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    (directory / META).unlink()
    (directory / META).mkdir()
    _expect(directory, [META], NOT_ALLOWED)


def test_digest_mismatch(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    (directory / META).write_bytes(b"meta-bytez")
    _expect(directory, [META], FAILED)


def test_digest_comparison_is_exact_not_case_folded(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    _write_manifest(directory, {"sha256": {META: _digest(b"meta-bytes").upper()}})
    _expect(directory, [META], FAILED)


def test_digest_comparison_accepts_the_exact_value(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    _write_manifest(directory, {"sha256": {META: _digest(b"meta-bytes")}})
    assert fv.verify_integrity_manifest(directory, [META]) is True


def _install_spies(monkeypatch: pytest.MonkeyPatch, trace: list[str]) -> None:
    real_child = fv._integrity_child
    real_sha = fv.sha256_file
    real_lexists = os.path.lexists

    def child(d: Path, name: object) -> Path:
        trace.append(f"child:{name}")
        result: Path = real_child(d, name)
        return result

    def sha(d: Path, name: str) -> str:
        trace.append(f"sha:{name}")
        digest: str = real_sha(d, name)
        return digest

    def lexists(path: Any) -> bool:
        trace.append(f"lexists:{os.path.basename(str(path))}")
        found: bool = real_lexists(path)
        return found

    monkeypatch.setattr(fv, "_integrity_child", child)
    monkeypatch.setattr(fv, "sha256_file", sha)
    monkeypatch.setattr(fv.os.path, "lexists", lexists)


def test_check_order_and_collaborator_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _setup(tmp_path)
    trace: list[str] = []
    _install_spies(monkeypatch, trace)
    assert fv.verify_integrity_manifest(directory, [TRIALS, META]) is True
    # sha256_file resolves the child again internally, hence the repeated child call.
    assert trace == [
        f"child:{META}",
        f"child:{TRIALS}",
        f"child:{META}",
        f"lexists:{META}",
        f"sha:{META}",
        f"child:{META}",
        f"child:{TRIALS}",
        f"lexists:{TRIALS}",
        f"sha:{TRIALS}",
        f"child:{TRIALS}",
    ]


def test_unrequired_names_are_validated_but_not_probed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _setup(tmp_path)
    trace: list[str] = []
    _install_spies(monkeypatch, trace)
    assert fv.verify_integrity_manifest(directory, [TRIALS]) is True
    assert trace == [
        f"child:{META}",
        f"child:{TRIALS}",
        f"child:{TRIALS}",
        f"lexists:{TRIALS}",
        f"sha:{TRIALS}",
        f"child:{TRIALS}",
    ]


def test_first_failure_stops_later_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _setup(tmp_path)
    (directory / META).write_bytes(b"changed")
    hashed: list[str] = []
    real_sha = fv.sha256_file

    def sha(d: Path, name: str) -> str:
        hashed.append(name)
        digest: str = real_sha(d, name)
        return digest

    monkeypatch.setattr(fv, "sha256_file", sha)
    _expect(directory, [META, TRIALS], FAILED)
    assert hashed == [META]


def test_os_error_while_hashing_becomes_value_free_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _setup(tmp_path)

    def sha(d: Path, name: str) -> str:
        raise PermissionError(f"secret path {d}/{name}")

    monkeypatch.setattr(fv, "sha256_file", sha)
    error = _expect(directory, [META], UNVERIFIABLE)
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


def test_value_error_while_hashing_propagates_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _setup(tmp_path)

    def sha(d: Path, name: str) -> str:
        raise ValueError("custom hashing failure")

    monkeypatch.setattr(fv, "sha256_file", sha)
    _expect(directory, [META], "custom hashing failure")


@pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyboardInterrupt()])
def test_other_errors_while_hashing_propagate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    directory = _setup(tmp_path)

    def sha(d: Path, name: str) -> str:
        raise exc

    monkeypatch.setattr(fv, "sha256_file", sha)
    with pytest.raises(type(exc)):
        fv.verify_integrity_manifest(directory, [META])


def test_raise_once_hashing_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _setup(tmp_path)
    state = {"calls": 0}
    real_sha = fv.sha256_file

    def sha(d: Path, name: str) -> str:
        state["calls"] += 1
        if state["calls"] == 1:
            raise OSError("first")
        digest: str = real_sha(d, name)
        return digest

    monkeypatch.setattr(fv, "sha256_file", sha)
    _expect(directory, [META], UNVERIFIABLE)
    assert state["calls"] == 1


def test_child_validation_failure_on_verify_pass_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _setup(tmp_path)
    real_child = fv._integrity_child
    seen: list[str] = []

    def child(d: Path, name: object) -> Path:
        seen.append(str(name))
        if len(seen) == 3:
            raise RuntimeError("late")
        result: Path = real_child(d, name)
        return result

    monkeypatch.setattr(fv, "_integrity_child", child)
    with pytest.raises(RuntimeError, match="late"):
        fv.verify_integrity_manifest(directory, [META])
    assert seen == [META, TRIALS, META]


def test_manifest_with_extra_top_level_keys_is_accepted(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    _write_manifest(
        directory,
        {"version": 99, "extra": [1, 2], "sha256": {META: _digest(b"meta-bytes")}},
    )
    assert fv.verify_integrity_manifest(directory, [META]) is True


def test_error_messages_never_echo_paths_or_names(tmp_path: Path) -> None:
    directory = _setup(tmp_path)
    (directory / META).write_bytes(b"changed")
    error = _expect(directory, [META], FAILED)
    assert str(directory) not in str(error)
    assert META not in str(error)
