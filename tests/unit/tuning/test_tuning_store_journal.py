"""The tuning store's pure helpers: references, the checksummed journal, the head file.

None of these reach the TVL toolchain, so they run in every lane.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from traigent.tuning import controller as c
from traigent.tuning.errors import TuningError

from .conftest import tvl_free


def _code(err: pytest.ExceptionInfo[TuningError]) -> str:
    return err.value.code


@tvl_free
class TestDefaultStoreDir:
    def test_the_environment_is_part_of_the_path(self) -> None:
        assert c.default_store_dir("router", "staging") == Path(
            ".traigent/tuning/staging/router"
        )

    def test_the_environment_defaults_to_production(self) -> None:
        assert c.default_store_dir("router") == Path(
            ".traigent/tuning/production/router"
        )

    @pytest.mark.parametrize("agent", ["", "..", ".", "a/b", "a b", "x" * 129])
    def test_an_agent_that_is_not_an_opaque_reference_is_refused(
        self, agent: str
    ) -> None:
        with pytest.raises(TuningError) as err:
            c.default_store_dir(agent, "staging")
        assert _code(err) == c.E_INVALID_RECORD

    @pytest.mark.parametrize("env", ["..", "a/b", "../x"])
    def test_an_environment_that_could_climb_out_is_refused(self, env: str) -> None:
        with pytest.raises(TuningError) as err:
            c.default_store_dir("router", env)
        assert _code(err) == c.E_INVALID_RECORD


@tvl_free
class TestClocks:
    def test_utc_drops_microseconds_and_normalizes_the_zone(self) -> None:
        local = datetime(
            2026, 1, 1, 12, 0, 0, 999999, tzinfo=timezone(timedelta(hours=2))
        )
        assert c._utc(local) == datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)

    def test_a_naive_instant_is_taken_as_utc(self) -> None:
        assert c._utc(datetime(2026, 1, 1, 12, 0, 0)) == datetime(
            2026, 1, 1, 12, 0, 0, tzinfo=UTC
        )

    def test_instant_text_round_trips(self) -> None:
        moment = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
        assert c._instant_text(moment) == "2026-03-04T05:06:07Z"
        assert c._parse_instant("2026-03-04T05:06:07Z") == moment

    def test_no_fixed_instant_reads_the_live_clock(self) -> None:
        reading = c._effect_clock(None)()
        assert abs((datetime.now(UTC) - reading).total_seconds()) < 5

    def test_a_fixed_instant_still_advances_with_real_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ticks = iter([100.0, 107.0])
        monkeypatch.setattr(c.time, "monotonic", lambda: next(ticks))
        start = datetime(2026, 1, 1, tzinfo=UTC)
        assert c._effect_clock(start)() == start + timedelta(seconds=7)

    def test_a_fixed_instant_is_refused_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        with pytest.raises(TuningError) as err:
            c._effect_clock(datetime(2026, 1, 1, tzinfo=UTC))
        assert _code(err) == c.E_FIXED_CLOCK_IN_PRODUCTION

    def test_the_live_clock_is_allowed_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        assert isinstance(c._effect_clock(None)(), datetime)


@tvl_free
class TestOccurrenceRefs:
    def test_a_ref_splits_into_its_five_parts(self) -> None:
        assert c.parse_occurrence_ref("tuning/weekly/schedule/s1/3") == {
            "response": "tuning",
            "rule": "weekly",
            "scope_kind": "schedule",
            "scope": "s1",
            "j": 3,
        }

    @pytest.mark.parametrize(
        "ref",
        ["", "a/b/c/d", "a/b/c/d/e/f", "a/b/c/d/x", "a/b/c/d/-1", "a/b/c/d/"],
    )
    def test_a_malformed_ref_is_refused(self, ref: str) -> None:
        with pytest.raises(TuningError) as err:
            c.parse_occurrence_ref(ref)
        assert _code(err) == c.E_NOT_AN_OCCURRENCE

    def test_the_approve_context_names_the_decision_and_the_occurrence(self) -> None:
        assert c._approve_context("a/b/c/d/1") == {
            "decision": "approve_run",
            "occurrence": "a/b/c/d/1",
        }


@tvl_free
class TestFraming:
    def test_a_framed_record_round_trips(self) -> None:
        assert c._unframe(c._frame({"b": 1, "a": [2]})) == {"b": 1, "a": [2]}

    def test_the_frame_is_independent_of_key_order(self) -> None:
        assert c._frame({"a": 1, "b": 2}) == c._frame({"b": 2, "a": 1})

    def test_a_changed_body_fails_its_checksum(self) -> None:
        line = c._frame({"a": 1})
        with pytest.raises(ValueError, match="checksum"):
            c._unframe(line.replace('"a": 1', '"a": 2'))

    def test_a_line_without_a_checksum_is_refused(self) -> None:
        with pytest.raises(ValueError, match="checksum"):
            c._unframe('{"a": 1}')

    def test_legacy_lines_are_recognized_by_their_lack_of_a_frame(self) -> None:
        assert c._is_legacy_line(b'{"a": 1}') is True
        assert c._is_legacy_line(c._frame({"a": 1}).encode()) is False
        assert c._is_legacy_line(b"not json") is False
        assert c._is_legacy_line(b"[1, 2]") is False
        assert c._is_legacy_line(b"\xff\xfe") is False


def _journal(path: Path, *values: object) -> None:
    path.write_text("".join(c._frame(v) + "\n" for v in values), encoding="utf-8")


@tvl_free
class TestJournalScan:
    def test_a_missing_file_is_empty(self, tmp_path: Path) -> None:
        assert c._scan_lines(tmp_path, tmp_path / "none.jsonl") == ([], 0, False)
        assert c._read_lines(tmp_path, tmp_path / "none.jsonl") == []

    def test_every_record_is_read_back_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "h.jsonl"
        _journal(path, {"n": 1}, {"n": 2})
        values, valid, torn = c._scan_lines(tmp_path, path)
        assert values == [{"n": 1}, {"n": 2}]
        assert valid == path.stat().st_size
        assert torn is False

    def test_a_blank_line_is_skipped_and_counted_as_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "h.jsonl"
        path.write_text(c._frame({"n": 1}) + "\n\n" + c._frame({"n": 2}) + "\n")
        values, valid, torn = c._scan_lines(tmp_path, path)
        assert values == [{"n": 1}, {"n": 2}]
        assert (valid, torn) == (path.stat().st_size, False)

    def test_a_torn_final_line_is_reported_not_returned(self, tmp_path: Path) -> None:
        path = tmp_path / "h.jsonl"
        good = c._frame({"n": 1}) + "\n"
        path.write_text(good + c._frame({"n": 2})[:-5], encoding="utf-8")
        values, valid, torn = c._scan_lines(tmp_path, path)
        assert values == [{"n": 1}]
        assert valid == len(good)
        assert torn is True

    def test_a_damaged_committed_record_is_corruption(self, tmp_path: Path) -> None:
        path = tmp_path / "h.jsonl"
        bad = c._frame({"n": 2}).replace('"n": 2', '"n": 9')
        path.write_text(c._frame({"n": 1}) + "\n" + bad + "\n", encoding="utf-8")
        with pytest.raises(TuningError) as err:
            c._scan_lines(tmp_path, path)
        assert _code(err) == c.E_STORE_CORRUPT

    def test_a_plain_json_store_is_refused_in_the_old_format(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "h.jsonl"
        path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
        before = path.read_bytes()
        with pytest.raises(TuningError) as err:
            c._scan_lines(tmp_path, path)
        assert _code(err) == c.E_STORE_FORMAT
        assert path.read_bytes() == before  # refused untouched

    def test_reading_ignores_a_torn_tail_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "h.jsonl"
        path.write_text(c._frame({"n": 1}) + "\n" + "deadbeef", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=c.logger.name):
            assert c._read_lines(tmp_path, path) == [{"n": 1}]
        assert "torn final record" in caplog.text


@tvl_free
class TestJournalAppend:
    def test_append_creates_the_file_and_its_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "dir" / "h.jsonl"
        c._append_line(tmp_path, path, {"n": 1})
        assert c._read_lines(tmp_path, path) == [{"n": 1}]

    def test_appends_accumulate_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "h.jsonl"
        for n in range(3):
            c._append_line(tmp_path, path, {"n": n})
        assert [r["n"] for r in c._read_lines(tmp_path, path)] == [0, 1, 2]

    def test_a_prepared_append_repairs_first_and_then_only_writes(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "h.jsonl"
        path.write_text(c._frame({"n": 1}) + "\n" + "torn-tail", encoding="utf-8")
        append = c._prepare_append(tmp_path, path)
        assert path.read_text(encoding="utf-8") == c._frame({"n": 1}) + "\n"
        append({"n": 2})
        assert [r["n"] for r in c._read_lines(tmp_path, path)] == [1, 2]

    def test_repair_truncates_a_torn_tail_and_logs_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "h.jsonl"
        good = c._frame({"n": 1}) + "\n"
        path.write_text(good + "abc", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=c.logger.name):
            c._repair_tail(tmp_path, path)
        assert path.read_text(encoding="utf-8") == good
        assert "truncating a torn final record" in caplog.text

    def test_repair_terminates_a_valid_final_record_missing_its_newline(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "h.jsonl"
        path.write_text(c._frame({"n": 1}), encoding="utf-8")  # complete, no newline
        c._repair_tail(tmp_path, path)
        assert path.read_text(encoding="utf-8") == c._frame({"n": 1}) + "\n"
        assert c._read_lines(tmp_path, path) == [{"n": 1}]

    def test_repair_leaves_a_clean_journal_and_a_missing_file_alone(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "h.jsonl"
        c._repair_tail(tmp_path, path)  # nothing to repair, nothing created
        assert not path.exists()
        _journal(path, {"n": 1})
        before = path.read_bytes()
        c._repair_tail(tmp_path, path)
        assert path.read_bytes() == before


@tvl_free
class TestJsonFiles:
    def test_write_then_read_round_trips_and_leaves_no_temp_file(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "sub" / "meta.json"
        c._write_json(tmp_path, path, {"b": 2, "a": 1})
        assert c._read_json(tmp_path, path) == {"a": 1, "b": 2}
        assert [p.name for p in path.parent.iterdir()] == ["meta.json"]

    def test_a_failed_write_leaves_the_old_file_and_no_temp_file(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "meta.json"
        c._write_json(tmp_path, path, {"v": 1})
        with pytest.raises(TypeError):
            c._write_json(tmp_path, path, {"v": object()})
        assert c._read_json(tmp_path, path) == {"v": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["meta.json"]

    def test_a_missing_file_reads_as_the_default(self, tmp_path: Path) -> None:
        assert c._read_json(tmp_path, tmp_path / "absent.json") is None
        assert c._read_json(tmp_path, tmp_path / "absent.json", {"d": 1}) == {"d": 1}

    def test_read_head_returns_only_a_mapping(self, tmp_path: Path) -> None:
        assert c.read_head(tmp_path) is None
        (tmp_path / "head.json").write_text("[1]", encoding="utf-8")
        assert c.read_head(tmp_path) is None
        (tmp_path / "head.json").write_text('{"generation": 2}', encoding="utf-8")
        assert c.read_head(str(tmp_path)) == {"generation": 2}


def _write_head(store: Path, generation: int, config: dict, **meta: object) -> None:
    (store / "head.json").write_text(
        json.dumps({"generation": generation, "config": config}), encoding="utf-8"
    )
    if meta:
        (store / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


@tvl_free
class TestHeadLoader:
    def test_no_head_file_means_no_head(self, tmp_path: Path) -> None:
        loader = c.HeadLoader(tmp_path)
        assert loader.head() is None
        assert loader.config() is None
        assert loader.generation() is None

    def test_a_head_is_read_and_its_config_is_copied(self, tmp_path: Path) -> None:
        _write_head(tmp_path, 4, {"model": "m"})
        loader = c.HeadLoader(tmp_path)
        assert loader.generation() == 4
        config = loader.config()
        assert config == {"model": "m"}
        config["model"] = "changed"  # the caller's copy never edits the cache
        assert loader.config() == {"model": "m"}

    def test_a_rewritten_head_is_picked_up(self, tmp_path: Path) -> None:
        _write_head(tmp_path, 1, {"model": "a"})
        loader = c.HeadLoader(tmp_path)
        assert loader.generation() == 1
        _write_head(tmp_path, 2, {"model": "bb"})  # a different size: a new stamp
        assert loader.generation() == 2

    def test_a_store_of_another_environment_is_refused(self, tmp_path: Path) -> None:
        _write_head(tmp_path, 1, {"m": 1}, environment="staging")
        with pytest.raises(TuningError) as err:
            c.HeadLoader(tmp_path, "production").head()
        assert _code(err) == c.E_STORE_ENVIRONMENT_MISMATCH

    def test_a_store_of_the_same_environment_is_read(self, tmp_path: Path) -> None:
        _write_head(tmp_path, 1, {"m": 1}, environment="staging")
        assert c.HeadLoader(str(tmp_path), "staging").generation() == 1

    def test_without_an_expected_environment_meta_is_not_consulted(
        self, tmp_path: Path
    ) -> None:
        _write_head(tmp_path, 1, {"m": 1}, environment="anything")
        assert c.HeadLoader(tmp_path).generation() == 1


@tvl_free
class TestDomains:
    def test_a_set_domain_becomes_a_list_copy(self) -> None:
        spec = {"set": ["a", "b"]}
        out = c._normalized_domain(spec)
        assert out == ["a", "b"]
        out.append("c")
        assert spec == {"set": ["a", "b"]}

    def test_any_other_domain_is_passed_through_as_a_copy(self) -> None:
        spec = {"range": [0, 1]}
        out = c._normalized_domain(spec)
        assert out == spec and out is not spec

    def test_membership_in_a_list_or_set_domain(self) -> None:
        assert c._declared_contains(["a", "b"], "a") is True
        assert c._declared_contains(("a",), "z") is False
        assert c._declared_contains({"set": ["a"]}, "a") is True
        assert c._declared_contains({"set": ["a"]}, "b") is False

    def test_membership_in_a_range_domain_is_inclusive(self) -> None:
        spec = {"range": [0, 1]}
        assert c._declared_contains(spec, 0) is True
        assert c._declared_contains(spec, 1) is True
        assert c._declared_contains(spec, 0.5) is True
        assert c._declared_contains(spec, 1.5) is False

    def test_a_non_numeric_value_is_outside_a_range(self) -> None:
        assert c._declared_contains({"range": [0, 1]}, "x") is False
        assert c._declared_contains({"range": [0, 1]}, None) is False

    @pytest.mark.parametrize("spec", [{"range": [0]}, {"other": 1}, "text", 5, None])
    def test_an_unreadable_domain_fails_closed_as_containing_the_value(
        self, spec: object
    ) -> None:
        assert c._declared_contains(spec, "anything") is True

    def test_digests_must_be_a_collection_of_sha256_strings(self) -> None:
        digest = "sha256:" + "a" * 64
        assert c._well_formed_digests([digest]) is True
        assert c._well_formed_digests(frozenset({digest})) is True
        assert c._well_formed_digests([]) is True
        assert c._well_formed_digests(digest) is False  # a bare string
        assert c._well_formed_digests(None) is False
        assert c._well_formed_digests(["sha256:short"]) is False
        assert c._well_formed_digests([1]) is False


@tvl_free
class TestSmallRecords:
    def test_a_step_report_changed_only_when_something_ran(self) -> None:
        assert c.StepReport().changed is False
        assert (
            c.StepReport(awaiting_approval=["a/b/c/d/1"], refused=["x:y"]).changed
            is False
        )
        assert c.StepReport(started=["a1"]).changed is True
        assert c.StepReport(approved=["o"]).changed is True
        assert c.StepReport(finished=["a1"]).changed is True

    def test_an_acceptance_outcome_is_ok_only_when_written_or_reject_recorded(
        self,
    ) -> None:
        assert c.AcceptanceOutcome("a", "written", generation=2).ok is True
        assert c.AcceptanceOutcome("a", "reject_recorded").ok is True
        for result in ("refused", "invalid", "consumed"):
            assert c.AcceptanceOutcome("a", result, "why").ok is False

    def test_an_optimizer_run_exposes_the_wrapped_result(self) -> None:
        class Result:
            best_config = {"m": 1}

        run = c.OptimizerRun(Result(), frozenset({"sha256:" + "0" * 64}))
        assert run.best_config == {"m": 1}
        assert run.loaded_dataset_digests == frozenset({"sha256:" + "0" * 64})
        for name in ("__missing_dunder__", "no_such_field"):
            with pytest.raises(AttributeError):
                getattr(run, name)


@tvl_free
class TestStorePathContainment:
    def test_a_valid_store_file_round_trips(self, tmp_path: Path) -> None:
        c._write_json(tmp_path, tmp_path / "head.json", {"a": 1})
        assert c._read_json(tmp_path, tmp_path / "head.json") == {"a": 1}
        c._append_line(tmp_path, tmp_path / "history.jsonl", {"n": 1})
        assert c._read_lines(tmp_path, tmp_path / "history.jsonl") == [{"n": 1}]

    def test_a_dotdot_escaping_the_store_base_is_refused(self, tmp_path: Path) -> None:
        store = tmp_path / "store"
        store.mkdir()
        (tmp_path / "outside").mkdir()
        with pytest.raises(TuningError) as err:
            c._store_file(store / ".." / "outside" / "head.json", store)
        assert _code(err) == c.E_INVALID_RECORD
        with pytest.raises(TuningError):
            c._write_json(store, store / ".." / "outside" / "head.json", {})
        assert not (tmp_path / "outside" / "head.json").exists()

    def test_a_dotdot_that_stays_inside_the_base_is_normalised(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "sub").mkdir()
        assert (
            c._store_file(tmp_path / "sub" / ".." / "head.json", tmp_path)
            == (tmp_path / "head.json").resolve()
        )

    def test_a_link_planted_after_the_check_is_not_followed(
        self, tmp_path: Path
    ) -> None:
        """The open itself refuses a symlink (O_NOFOLLOW), not only the earlier check."""
        outside = tmp_path / "outside.jsonl"
        outside.write_text("untouched\n")
        store = tmp_path / "store"
        store.mkdir()
        path = store / "history.jsonl"
        append = c._prepare_append(store, path)  # checked, tail repaired, file absent
        path.symlink_to(outside)  # replaced between the check and the open
        with pytest.raises(TuningError) as err:
            append({"n": 1})
        assert _code(err) == c.E_INVALID_RECORD
        assert outside.read_text() == "untouched\n"
        with pytest.raises(TuningError):
            c._open_regular(path, os.O_RDONLY)

    def test_a_non_regular_file_is_refused_on_the_descriptor(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "head.json").mkdir()
        with pytest.raises(TuningError):
            c._open_regular(tmp_path / "head.json", os.O_RDONLY)

    def test_an_unreadable_file_error_does_not_echo_the_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_open = os.open

        def deny(path: Any, *args: Any, **kwargs: Any) -> int:
            if str(path).endswith("head.json"):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, *args, **kwargs)

        (tmp_path / "head.json").write_text("{}")
        monkeypatch.setattr(os, "open", deny)
        with pytest.raises(TuningError) as err:
            c._read_json(tmp_path, tmp_path / "head.json")
        assert _code(err) == c.E_INVALID_RECORD
        assert "head.json" not in str(err.value)
        assert str(tmp_path) not in str(err.value)

    def test_the_legacy_format_error_does_not_echo_the_store_path(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "history.jsonl").write_text('{"a": 1}\n')
        with pytest.raises(TuningError) as err:
            c._scan_lines(tmp_path, tmp_path / "history.jsonl")
        assert str(tmp_path) not in str(err.value)

    def test_a_symlinked_store_file_is_refused(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.json").write_text("{}")
        store = tmp_path / "store"
        store.mkdir()
        (store / "head.json").symlink_to(outside / "secret.json")
        for op in (
            lambda: c._read_json(store, store / "head.json"),
            lambda: c._write_json(store, store / "head.json", {}),
            lambda: c._scan_lines(store, store / "head.json"),
            lambda: c._append_line(store, store / "head.json", {}),
        ):
            with pytest.raises(TuningError) as err:
                op()
            assert _code(err) == c.E_INVALID_RECORD
            assert "secret" not in str(err.value)

    def test_a_nul_byte_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TuningError) as err:
            c._store_file(str(tmp_path / "head\0.json"), tmp_path)
        assert _code(err) == c.E_INVALID_RECORD
        with pytest.raises(TuningError):
            c.read_head(str(tmp_path) + "\0")


def test_a_failing_mkdir_is_a_closed_error_without_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "canary_store_dir"
    base.mkdir()

    def boom(self, *a, **k):
        raise PermissionError(13, "denied", str(self))

    monkeypatch.setattr(Path, "mkdir", boom)
    with pytest.raises(TuningError) as err:
        c._write_json(base, base / "sub" / "head.json", {"a": 1})
    assert _code(err) == c.E_INVALID_RECORD
    assert "canary_store_dir" not in str(err.value)
    assert err.value.__cause__ is None


@tvl_free
def test_a_verification_failure_after_the_temp_create_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The temp file is created, then the path check fails: the temp must be removed."""
    real_lstat = os.lstat

    def lstat(p, *args, **kwargs):  # the verification's lstat of the temp path fails
        if str(p).endswith(".tmp"):
            raise OSError("simulated")
        return real_lstat(p, *args, **kwargs)

    monkeypatch.setattr(c.os, "lstat", lstat)
    target = tmp_path / "head.json"
    with pytest.raises(TuningError) as err:
        c._write_json(tmp_path, target, {"a": 1})
    assert err.value.code == c.E_INVALID_RECORD
    assert str(tmp_path) not in str(err.value)
    assert not target.exists()
    assert [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []
