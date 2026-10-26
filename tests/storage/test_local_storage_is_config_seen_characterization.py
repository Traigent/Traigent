"""Characterization of ``LocalStorageManager.is_config_seen``.

``oracle_is_config_seen`` is a frozen copy of the pre-refactor method. The
live method and the oracle are driven against identical fake sessions and
compared on the result, on every call made on the storage object and the
sessions/trials/configs it reads (in order), and on the logged warning.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from traigent.storage import local_storage as ls
from traigent.storage.local_storage import LocalStorageManager


# --- frozen pre-refactor implementation ---
def oracle_is_config_seen(
    self,
    function_name: str,
    dataset_name: str,
    config: dict[str, Any],
    keys: list[str] | None = None,
) -> bool:
    """Check if configuration has been evaluated in any previous session.

    Args:
        function_name: Name of the function being optimized
        dataset_name: Name of the dataset being used
        config: Configuration to check
        keys: Optional list of config keys to use for comparison

    Returns:
        True if configuration has been seen and successfully evaluated before
    """
    user_config_keys = {
        key for key in config if isinstance(key, str) and not key.startswith("_")
    }
    base_keys = (
        {key for key in keys if isinstance(key, str) and not key.startswith("_")}
        | user_config_keys
        if keys
        else None
    )

    try:
        # List all sessions for this function
        sessions = self.list_sessions()

        for session_info in sessions:
            if session_info.function_name != function_name:
                continue

            # Load the actual session data
            session = self.load_session(session_info.session_id)
            if not session:
                continue

            # Check dataset match (prefer evaluation_set metadata)
            session_dataset = (
                session.metadata.get("evaluation_set") if session.metadata else None
            )
            if not session_dataset:
                session_dataset = (
                    session.metadata.get("dataset_name") if session.metadata else None
                )

            if session_dataset != dataset_name:
                continue

            # Check trials for matching config
            if session.trials:
                for trial in session.trials:
                    # Check if trial was successful (no error)
                    if trial.error is None:
                        # Compare the complete user-facing identity on both
                        # sides. Unioning realized user keys preserves spec
                        # awareness when a config space later expands or
                        # narrows, while ignoring runtime-only metadata.
                        if base_keys is None:
                            compare_keys = None
                        else:
                            trial_user_keys = {
                                key
                                for key in trial.config
                                if isinstance(key, str) and not key.startswith("_")
                            }
                            compare_keys = sorted(base_keys | trial_user_keys)
                        target_hash = self.compute_config_hash(config, compare_keys)
                        trial_hash = self.compute_config_hash(
                            trial.config, compare_keys
                        )
                        if trial_hash == target_hash:
                            return True

    except Exception as e:
        ls.logger.warning(
            "Error during config deduplication check (%s)", type(e).__name__
        )
        # Best effort - don't fail optimization due to dedup errors

    return False


class _Dict(dict):
    def __init__(self, data, log, tag):
        super().__init__(data)
        self._log = log
        self._tag = tag

    def __iter__(self):
        self._log.append((self._tag, "iter"))
        return super().__iter__()

    def get(self, key, *default):
        self._log.append((self._tag, "get", key))
        return super().get(key, *default)

    def __bool__(self):
        self._log.append((self._tag, "bool"))
        return super().__len__() > 0

    def items(self):
        self._log.append((self._tag, "items"))
        return super().items()


class _Obj:
    """Plain object whose attribute reads are logged."""

    def __init__(self, tag, log, **attrs):
        object.__setattr__(self, "_tag", tag)
        object.__setattr__(self, "_log", log)
        object.__setattr__(self, "_attrs", attrs)

    def __getattr__(self, name):
        attrs = object.__getattribute__(self, "_attrs")
        if name not in attrs:
            raise AttributeError(name)
        object.__getattribute__(self, "_log").append(
            (object.__getattribute__(self, "_tag"), name)
        )
        return attrs[name]


class _Boom(RuntimeError):
    pass


class _RecordingLogger:
    def __init__(self, log):
        self._log = log

    def warning(self, *args, **kwargs):
        self._log.append(("logger.warning", args, tuple(sorted(kwargs))))


def _build(scenario, log, monkeypatch):
    infos, trials_spec, fault = (
        scenario["sessions"],
        scenario["trials"],
        scenario["fault"],
    )
    manager = object.__new__(LocalStorageManager)
    sessions_by_id = {}
    listing = []
    for idx, info in enumerate(infos):
        sid = f"s{idx}"
        listing.append(
            _Obj(f"info{idx}", log, function_name=info["fn"], session_id=sid)
        )
        meta = info["meta"]
        trials = None
        if info["trials"] is not None:
            trials = [
                _Obj(
                    f"trial{idx}.{t_idx}",
                    log,
                    error=t["error"],
                    config=_Dict(t["config"], log, f"tcfg{idx}.{t_idx}"),
                )
                for t_idx, t in enumerate(trials_spec[info["trials"]])
            ]
        sessions_by_id[sid] = (
            None
            if info["loaded"] is False
            else _Obj(
                f"session{idx}",
                log,
                metadata=None if meta is None else _Dict(meta, log, f"meta{idx}"),
                trials=trials,
            )
        )
    calls = {"n": 0}

    def tick(label):
        calls["n"] += 1
        if fault == calls["n"]:
            raise _Boom(label)

    def list_sessions():
        log.append(("list_sessions",))
        tick("list")
        return listing

    def load_session(session_id):
        log.append(("load_session", session_id))
        tick("load")
        return sessions_by_id[session_id]

    real_hash = LocalStorageManager.compute_config_hash

    def compute_config_hash(config, keys=None):
        log.append(
            (
                "hash",
                tuple(sorted(map(repr, config))),
                None if keys is None else tuple(keys),
            )
        )
        tick("hash")
        return real_hash(manager, config, keys)

    manager.list_sessions = list_sessions
    manager.load_session = load_session
    manager.compute_config_hash = compute_config_hash
    monkeypatch.setattr(ls, "logger", _RecordingLogger(log))
    return manager


def _run(method, scenario, config, keys, monkeypatch):
    log: list = []
    manager = _build(scenario, log, monkeypatch)
    cfg = _Dict(config, log, "config")
    key_arg = None if keys is None else list(keys)
    try:
        result = ("ok", method(manager, "fn", "ds", cfg, key_arg))
    except Exception as exc:  # noqa: BLE001 - characterizing any exception
        result = ("err", type(exc).__name__)
    return result, log


_SAME = {"a": 1, "b": 2}
_TRIAL_SETS = {
    "match": [{"error": None, "config": {"a": 1, "b": 2}}],
    "miss": [{"error": None, "config": {"a": 9, "b": 2}}],
    "errored": [{"error": "x", "config": {"a": 1, "b": 2}}],
    "runtime_only": [{"error": None, "config": {"a": 1, "b": 2, "_opt": 5}}],
    "extra_key": [{"error": None, "config": {"a": 1, "b": 2, "c": 3}}],
    "mixed": [
        {"error": "x", "config": {"a": 1, "b": 2}},
        {"error": None, "config": {"a": 0, "b": 0}},
        {"error": None, "config": {"a": 1, "b": 2}},
    ],
    "empty": [],
    "falsy_errors": [
        {"error": "", "config": {"a": 1, "b": 2}},
        {"error": 0, "config": {"a": 1, "b": 2}},
        {"error": False, "config": {"a": 1, "b": 2}},
        {"error": [], "config": {"a": 1, "b": 2}},
    ],
}
_INFOS = [
    {
        "fn": "other",
        "meta": {"evaluation_set": "ds"},
        "trials": "match",
        "loaded": True,
    },
    {"fn": "fn", "meta": {"evaluation_set": "ds"}, "trials": "match", "loaded": False},
    {"fn": "fn", "meta": None, "trials": "match", "loaded": True},
    {"fn": "fn", "meta": {}, "trials": "match", "loaded": True},
    {"fn": "fn", "meta": {"evaluation_set": "ds"}, "trials": None, "loaded": True},
    {"fn": "fn", "meta": {"evaluation_set": "ds"}, "trials": "empty", "loaded": True},
    {"fn": "fn", "meta": {"evaluation_set": "ds"}, "trials": "match", "loaded": True},
    {"fn": "fn", "meta": {"evaluation_set": "ds"}, "trials": "miss", "loaded": True},
    {"fn": "fn", "meta": {"dataset_name": "ds"}, "trials": "match", "loaded": True},
    {
        "fn": "fn",
        "meta": {"evaluation_set": "zzz", "dataset_name": "ds"},
        "trials": "match",
        "loaded": True,
    },
    {
        "fn": "fn",
        "meta": {"evaluation_set": "", "dataset_name": "ds"},
        "trials": "match",
        "loaded": True,
    },
    {"fn": "fn", "meta": {"evaluation_set": "ds"}, "trials": "errored", "loaded": True},
    {
        "fn": "fn",
        "meta": {"evaluation_set": "ds"},
        "trials": "runtime_only",
        "loaded": True,
    },
    {
        "fn": "fn",
        "meta": {"evaluation_set": "ds"},
        "trials": "extra_key",
        "loaded": True,
    },
    {"fn": "fn", "meta": {"evaluation_set": "ds"}, "trials": "mixed", "loaded": True},
    {
        "fn": "fn",
        "meta": {"evaluation_set": "ds"},
        "trials": "falsy_errors",
        "loaded": True,
    },
    {
        "fn": "fn",
        "meta": {"evaluation_set": 0, "dataset_name": "ds"},
        "trials": "match",
        "loaded": True,
    },
]
_CONFIGS = [_SAME, {}, {"a": 1, 5: 2, "_x": 1}, {"a": 1, "b": 2, "_x": 9}]
_KEYS = [None, [], ["a"], ["a", "_x"], ["b", 3], ["a", "b", "c"]]


def _scenarios():
    yield {"sessions": [], "trials": _TRIAL_SETS, "fault": None}
    for info in _INFOS:
        for fault in (None, 1, 2, 3):
            yield {"sessions": [info], "trials": _TRIAL_SETS, "fault": fault}
    for first, second in itertools.product(range(len(_INFOS)), repeat=2):
        yield {
            "sessions": [_INFOS[first], _INFOS[second]],
            "trials": _TRIAL_SETS,
            "fault": None,
        }
    yield {"sessions": list(_INFOS), "trials": _TRIAL_SETS, "fault": None}
    for fault in range(1, 12):
        yield {"sessions": list(_INFOS), "trials": _TRIAL_SETS, "fault": fault}


def test_matches_oracle_across_scenarios(monkeypatch):
    checked = 0
    mismatches = []
    for scenario in _scenarios():
        for config, keys in itertools.product(_CONFIGS, _KEYS):
            got = _run(
                LocalStorageManager.is_config_seen, scenario, config, keys, monkeypatch
            )
            want = _run(oracle_is_config_seen, scenario, config, keys, monkeypatch)
            checked += 1
            if got != want:
                mismatches.append((scenario["fault"], config, keys, got, want))
                if len(mismatches) >= 3:
                    break
    assert not mismatches
    assert checked > 1000


@pytest.mark.parametrize(
    ("info_index", "config", "keys", "expected"),
    [
        (6, _SAME, None, True),
        (7, _SAME, None, False),
        (8, _SAME, None, True),
        (9, _SAME, None, False),
        (10, _SAME, None, True),
        (11, _SAME, None, False),
        (12, _SAME, ["a", "b"], True),
        (13, _SAME, None, False),
        (13, _SAME, ["a"], False),
        (0, _SAME, None, False),
        (1, _SAME, None, False),
        (2, _SAME, None, False),
    ],
)
def test_documented_outcomes(info_index, config, keys, expected, monkeypatch):
    scenario = {"sessions": [_INFOS[info_index]], "trials": _TRIAL_SETS, "fault": None}
    result, _log = _run(
        LocalStorageManager.is_config_seen, scenario, config, keys, monkeypatch
    )
    assert result == ("ok", expected)


def test_storage_errors_are_swallowed_and_warned(monkeypatch):
    scenario = {"sessions": [_INFOS[6]], "trials": _TRIAL_SETS, "fault": 1}
    result, log = _run(
        LocalStorageManager.is_config_seen, scenario, _SAME, None, monkeypatch
    )
    assert result == ("ok", False)
    assert (
        "logger.warning",
        ("Error during config deduplication check (%s)", "_Boom"),
        (),
    ) in log
