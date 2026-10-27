"""Characterization of ContentPolicy._attrs (span attribute egress policy)."""

from __future__ import annotations

import pytest

from traigent.observability.otel import contract as C
from traigent.observability.otel import policy as policy_module
from traigent.observability.otel.policy import ContentPolicy

SECRET = "sk-abcdefghijklmnopqrstuvwxyz0123456789"  # pragma: allowlist secret
MODES = ("metadata", "redacted", "record")
POLICY = ContentPolicy("metadata")


def _run(attrs, mode):
    return POLICY._attrs(attrs, mode)


class TestEmptyInputs:
    @pytest.mark.parametrize("mode", MODES)
    def test_none_and_empty_mapping(self, mode):
        assert _run(None, mode) == ({}, 0)
        assert _run({}, mode) == ({}, 0)


class TestNonStringKeys:
    @pytest.mark.parametrize("mode", MODES)
    def test_each_non_string_key_is_dropped_and_counted(self, mode):
        out, dropped = _run({1: "a", None: "b", b"k": "c", "ok.key": 5}, mode)
        assert dropped == (3 if mode == "record" else 4)
        assert 1 not in out and None not in out and b"k" not in out

    def test_record_mode_full_result_with_mixed_keys(self):
        assert _run({1: "a", "ok.key": 5, None: "b"}, "record") == (
            {"ok.key": 5},
            2,
        )

    def test_metadata_mode_full_result_with_mixed_keys(self):
        assert _run({1: "a", "ok.key": 5, None: "b"}, "metadata") == ({}, 3)


class TestAllowlistedKeys:
    @pytest.mark.parametrize("mode", MODES)
    def test_valid_value_is_kept(self, mode):
        assert _run({"gen_ai.request.model": "m-1"}, mode) == (
            {"gen_ai.request.model": "m-1"},
            0,
        )

    @pytest.mark.parametrize("mode", MODES)
    def test_invalid_value_is_dropped_and_never_recorded(self, mode):
        # A rejected allowlisted value must not fall through to the
        # record-mode pass-through (5 would be recorded as-is there).
        assert _run({"gen_ai.request.model": 5}, mode) == ({}, 1)

    @pytest.mark.parametrize("mode", MODES)
    def test_credential_shaped_value_is_dropped(self, mode):
        assert _run({"gen_ai.request.model": SECRET}, mode) == ({}, 1)

    @pytest.mark.parametrize("mode", MODES)
    def test_string_length_boundary(self, mode):
        ok = _run({"gen_ai.request.model": "a" * 255}, mode)
        bad = _run({"gen_ai.request.model": "a" * 256}, mode)
        assert ok == ({"gen_ai.request.model": "a" * 255}, 0)
        assert bad == ({}, 1)


class TestContentKeys:
    def test_metadata_mode_drops_content_keys(self):
        assert _run({"input.value": "hello"}, "metadata") == ({}, 1)

    def test_redacted_mode_replaces_content_keys(self):
        assert _run({"input.value": "hello", "gen_ai.prompt": 7}, "redacted") == (
            {
                "input.value": C.REDACTED_PLACEHOLDER,
                "gen_ai.prompt": C.REDACTED_PLACEHOLDER,
            },
            0,
        )

    def test_record_mode_records_content_keys(self):
        assert _run({"input.value": "hello"}, "record") == ({"input.value": "hello"}, 0)

    def test_redacted_mode_prefixed_safe_key_is_placeholder(self):
        key = C.CONTENT_METADATA_PREFIX + "tag_1"
        assert _run({key: "v"}, "redacted") == ({key: C.REDACTED_PLACEHOLDER}, 0)

    def test_redacted_mode_prefixed_unsafe_key_is_dropped(self):
        key = C.CONTENT_METADATA_PREFIX + "has space"
        assert _run({key: "v"}, "redacted") == ({}, 1)

    def test_redacted_mode_prefixed_key_length_boundary(self):
        ok = C.CONTENT_METADATA_PREFIX + "a" * (128 - len(C.CONTENT_METADATA_PREFIX))
        bad = ok + "a"
        assert _run({ok: "v"}, "redacted") == ({ok: C.REDACTED_PLACEHOLDER}, 0)
        assert _run({bad: "v"}, "redacted") == ({}, 1)

    def test_redacted_mode_unknown_key_is_dropped_not_placeholder(self):
        assert _run({"custom.key": "v"}, "redacted") == ({}, 1)

    def test_unknown_mode_drops_everything_not_allowlisted(self):
        assert _run({"input.value": "v", "custom.key": "v"}, "other") == ({}, 2)


class TestRecordMode:
    def test_unknown_safe_key_is_recorded(self):
        assert _run({"custom.key": "v"}, "record") == ({"custom.key": "v"}, 0)

    def test_scalar_and_sequence_values(self):
        out, dropped = _run(
            {"a": 1, "b": 2.5, "c": True, "d": ["x", "y"]},
            "record",
        )
        assert (out, dropped) == ({"a": 1, "b": 2.5, "c": True, "d": ("x", "y")}, 0)

    def test_secret_value_is_scrubbed(self):
        out, dropped = _run({"custom.key": SECRET}, "record")
        assert dropped == 0
        assert SECRET not in out["custom.key"]

    @pytest.mark.parametrize("key", ["api_key", "password", "authorization"])
    def test_credential_named_key_is_dropped(self, key):
        assert _run({key: "plain"}, "record") == ({}, 1)

    @pytest.mark.parametrize("key", ["has space", "a/b", "", "a" * 129])
    def test_unsafe_key_is_dropped(self, key):
        assert _run({key: "plain"}, "record") == ({}, 1)

    def test_safe_key_length_boundary(self):
        assert _run({"a" * 128: 1}, "record") == ({"a" * 128: 1}, 0)
        assert _run({"a" * 129: 1}, "record") == ({}, 1)

    def test_unsupported_value_is_dropped(self):
        assert _run({"k": {"nested": 1}, "j": None, "i": b"raw"}, "record") == ({}, 3)

    def test_oversize_string_is_truncated_not_dropped(self):
        limit = C.MAX_RECORD_ATTR_BYTES
        out, dropped = _run({"k": "a" * (limit + 10)}, "record")
        assert dropped == 0
        assert out == {"k": "a" * limit}

    def test_oversize_sequence_is_dropped(self):
        big = ["a" * 1000] * 100
        assert _run({"k": big}, "record") == ({}, 1)

    def test_insertion_order_is_preserved(self):
        out, _ = _run({"z": 1, "gen_ai.request.model": "m", "a": 2}, "record")
        assert list(out) == ["z", "gen_ai.request.model", "a"]


class _RaisingStr(str):
    def startswith(self, *args, **kwargs):
        raise RuntimeError("startswith read")


class TestExceptionPaths:
    @pytest.mark.parametrize("mode", MODES)
    def test_non_allowlisted_key_prefix_read_error_propagates(self, mode):
        # The content classification is evaluated for every non-allowlisted
        # string key, whatever the mode.
        with pytest.raises(RuntimeError, match="startswith read"):
            _run({_RaisingStr("custom.key"): "v"}, mode)

    @pytest.mark.parametrize("mode", MODES)
    def test_allowlisted_key_never_reaches_content_classification(self, mode):
        key = _RaisingStr("gen_ai.request.model")
        out, dropped = _run({key: "m"}, mode)
        assert (dict(out), dropped) == ({"gen_ai.request.model": "m"}, 0)

    def test_non_string_key_never_reaches_content_classification(self):
        assert _run({1: "v"}, "record") == ({}, 1)

    def test_items_error_propagates(self):
        class Broken(dict):
            def items(self):
                raise ValueError("items read")

        with pytest.raises(ValueError, match="items read"):
            _run(Broken(a=1), "record")

    def test_coerce_error_propagates(self, monkeypatch):
        def boom(value, spec):
            raise RuntimeError("coerce")

        monkeypatch.setattr(policy_module, "coerce_clean", boom)
        with pytest.raises(RuntimeError, match="coerce"):
            _run({"gen_ai.request.model": "m"}, "record")


class TestCollaboratorOrder:
    def _trace(self, monkeypatch):
        trace: list[tuple[str, str]] = []
        real_safe = C.is_safe_key
        real_cred = policy_module.is_credential_key_name

        def safe(key):
            trace.append(("safe", key))
            return real_safe(key)

        def cred(key):
            trace.append(("cred", key))
            return real_cred(key)

        monkeypatch.setattr(C, "is_safe_key", safe)
        monkeypatch.setattr(policy_module, "is_credential_key_name", cred)
        return trace

    def test_plain_key_in_record_mode(self, monkeypatch):
        trace = self._trace(monkeypatch)
        _run({"custom.key": "v"}, "record")
        assert trace == [("safe", "custom.key"), ("cred", "custom.key")]

    def test_prefixed_key_in_record_mode(self, monkeypatch):
        trace = self._trace(monkeypatch)
        key = C.CONTENT_METADATA_PREFIX + "x"
        _run({key: "v"}, "record")
        assert trace == [("safe", key), ("safe", key), ("cred", key)]

    def test_prefixed_unsafe_key_skips_credential_check(self, monkeypatch):
        trace = self._trace(monkeypatch)
        key = C.CONTENT_METADATA_PREFIX + "has space"
        _run({key: "v"}, "record")
        assert trace == [("safe", key), ("safe", key)]

    def test_prefixed_key_in_redacted_mode(self, monkeypatch):
        trace = self._trace(monkeypatch)
        key = C.CONTENT_METADATA_PREFIX + "x"
        _run({key: "v"}, "redacted")
        assert trace == [("safe", key)]

    def test_content_key_in_record_mode_still_checks_credential_name(self, monkeypatch):
        trace = self._trace(monkeypatch)
        _run({"input.value": "v"}, "record")
        assert trace == [("safe", "input.value"), ("cred", "input.value")]

    def test_metadata_mode_checks_prefix_only(self, monkeypatch):
        trace = self._trace(monkeypatch)
        key = C.CONTENT_METADATA_PREFIX + "x"
        _run({key: "v", "plain": "v"}, "metadata")
        assert trace == [("safe", key)]

    def test_allowlisted_key_touches_no_classifier(self, monkeypatch):
        trace = self._trace(monkeypatch)
        _run({"gen_ai.request.model": "m"}, "record")
        assert trace == []
