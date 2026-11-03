"""Characterization tests pinning the ``list`` CLI command (list_sessions)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from traigent.cli.local_commands import list_sessions

MOD = "traigent.cli.local_commands"


def _sessions(n):
    return [SimpleNamespace(session_id=f"s{i}") for i in range(n)]


def _summary(i, best=0.5, created="2025-01-02T03:04:05"):
    return {
        "function_name": f"fn{i}",
        "status": "completed",
        "completed_trials": i,
        "best_score": best,
        "created_at": created,
        "session_id": f"s{i}",
    }


# name -> (args, sessions, summaries by index, analytics, analytics_error, storage_error)
CASES = {
    "table_empty": ([], 0, {}, True, False, False),
    "table_rows": (
        [],
        3,
        {0: _summary(0, 0.123456), 1: _summary(1, None), 2: None},
        False,
        False,
        False,
    ),
    "table_limit": (
        ["--limit", "1"],
        3,
        {0: _summary(0), 1: _summary(1)},
        False,
        False,
        False,
    ),
    "table_status": (
        ["--status", "failed"],
        2,
        {0: _summary(0), 1: _summary(1)},
        False,
        False,
        False,
    ),
    "table_missing_created": (
        [],
        1,
        {0: {**_summary(0), "created_at": None}},
        False,
        False,
        False,
    ),
    "table_no_created_key": (
        [],
        1,
        {0: {k: v for k, v in _summary(0).items() if k != "created_at"}},
        False,
        False,
        False,
    ),
    "json_rows": (
        ["--format", "json"],
        3,
        {0: _summary(0), 1: None, 2: _summary(2)},
        False,
        False,
        False,
    ),
    "json_empty": (["--format", "json"], 0, {}, True, False, False),
    "analytics_on": ([], 2, {0: _summary(0), 1: _summary(1)}, True, False, False),
    "analytics_on_json": (
        ["--format", "json"],
        1,
        {0: _summary(0)},
        True,
        False,
        False,
    ),
    "analytics_on_no_sessions": ([], 0, {}, True, False, False),
    "analytics_fails": ([], 1, {0: _summary(0)}, True, True, False),
    "storage_error": ([], 1, {}, False, False, True),
    "missing_best_score_key": (
        [],
        1,
        {0: {k: v for k, v in _summary(0).items() if k != "best_score"}},
        False,
        False,
        False,
    ),
}


def _snapshot(name):
    args, n, summaries, analytics, analytics_err, storage_err = CASES[name]
    storage = MagicMock()
    sessions = _sessions(n)
    storage.list_sessions.return_value = sessions
    storage.get_session_summary.side_effect = lambda sid: summaries.get(int(sid[1:]))
    if storage_err:
        storage.list_sessions.side_effect = RuntimeError("disk")
    config = MagicMock()
    config.enable_usage_analytics = analytics
    config.get_local_storage_path.return_value = "/store"
    collect = MagicMock(side_effect=RuntimeError("net") if analytics_err else None)
    with (
        patch(f"{MOD}.TraigentConfig") as cfg_cls,
        patch(f"{MOD}.LocalStorageManager", return_value=storage) as storage_cls,
        patch(f"{MOD}.collect_and_submit_analytics", collect),
    ):
        cfg_cls.from_environment.return_value = config
        result = CliRunner().invoke(list_sessions, args)
    return {
        "exit": result.exit_code,
        "output": result.output,
        "calls": repr(
            (
                storage_cls.call_args,
                storage.list_sessions.call_args,
                storage.get_session_summary.call_args_list,
                [
                    c.args == (config,) and c.kwargs == {}
                    for c in collect.call_args_list
                ],
            )
        ),
    }


GOLDEN: dict = {
    "table_empty": {
        "exit": 0,
        "output": "No sessions found.\n",
        "calls": "(call('/store'), call(status=None), [], [])",
    },
    "table_rows": {
        "exit": 0,
        "output": "Session ID                     Function             Status       "
        "Trials   Best Score   Created\n"
        "----------------------------------------------------------------------------------------------------\n"
        "s0                             fn0                  completed    "
        "0        0.123        2025-01-02\n"
        "s1                             fn1                  completed    "
        "1        N/A          2025-01-02\n",
        "calls": "(call('/store'), call(status=None), [call('s0'), call('s1'), "
        "call('s2')], [])",
    },
    "table_limit": {
        "exit": 0,
        "output": "Session ID                     Function             Status       "
        "Trials   Best Score   Created\n"
        "----------------------------------------------------------------------------------------------------\n"
        "s0                             fn0                  completed    "
        "0        0.500        2025-01-02\n",
        "calls": "(call('/store'), call(status=None), [call('s0')], [])",
    },
    "table_status": {
        "exit": 0,
        "output": "Session ID                     Function             Status       "
        "Trials   Best Score   Created\n"
        "----------------------------------------------------------------------------------------------------\n"
        "s0                             fn0                  completed    "
        "0        0.500        2025-01-02\n"
        "s1                             fn1                  completed    "
        "1        0.500        2025-01-02\n",
        "calls": "(call('/store'), call(status='failed'), [call('s0'), call('s1')], [])",
    },
    "table_missing_created": {
        "exit": 0,
        "output": "Session ID                     Function             "
        "Status       Trials   Best Score   Created\n"
        "----------------------------------------------------------------------------------------------------\n"
        "s0                             fn0                  "
        "completed    0        0.500        None\n",
        "calls": "(call('/store'), call(status=None), [call('s0')], [])",
    },
    "table_no_created_key": {
        "exit": 0,
        "output": "Session ID                     Function             "
        "Status       Trials   Best Score   Created\n"
        "----------------------------------------------------------------------------------------------------\n"
        "s0                             fn0                  "
        "completed    0        0.500        \n",
        "calls": "(call('/store'), call(status=None), [call('s0')], [])",
    },
    "json_rows": {
        "exit": 0,
        "output": "[\n"
        "  {\n"
        '    "function_name": "fn0",\n'
        '    "status": "completed",\n'
        '    "completed_trials": 0,\n'
        '    "best_score": 0.5,\n'
        '    "created_at": "2025-01-02T03:04:05",\n'
        '    "session_id": "s0"\n'
        "  },\n"
        "  {\n"
        '    "function_name": "fn2",\n'
        '    "status": "completed",\n'
        '    "completed_trials": 2,\n'
        '    "best_score": 0.5,\n'
        '    "created_at": "2025-01-02T03:04:05",\n'
        '    "session_id": "s2"\n'
        "  }\n"
        "]\n",
        "calls": "(call('/store'), call(status=None), [call('s0'), call('s1'), call('s2')], "
        "[])",
    },
    "json_empty": {
        "exit": 0,
        "output": "[]\n",
        "calls": "(call('/store'), call(status=None), [], [])",
    },
    "analytics_on": {
        "exit": 0,
        "output": "Session ID                     Function             Status       "
        "Trials   Best Score   Created\n"
        "----------------------------------------------------------------------------------------------------\n"
        "s0                             fn0                  completed    "
        "0        0.500        2025-01-02\n"
        "s1                             fn1                  completed    "
        "1        0.500        2025-01-02\n",
        "calls": "(call('/store'), call(status=None), [call('s0'), call('s1')], [True])",
    },
    "analytics_on_json": {
        "exit": 0,
        "output": "[\n"
        "  {\n"
        '    "function_name": "fn0",\n'
        '    "status": "completed",\n'
        '    "completed_trials": 0,\n'
        '    "best_score": 0.5,\n'
        '    "created_at": "2025-01-02T03:04:05",\n'
        '    "session_id": "s0"\n'
        "  }\n"
        "]\n",
        "calls": "(call('/store'), call(status=None), [call('s0')], [True])",
    },
    "analytics_on_no_sessions": {
        "exit": 0,
        "output": "No sessions found.\n",
        "calls": "(call('/store'), call(status=None), [], [])",
    },
    "analytics_fails": {
        "exit": 0,
        "output": "Session ID                     Function             Status       "
        "Trials   Best Score   Created\n"
        "----------------------------------------------------------------------------------------------------\n"
        "s0                             fn0                  completed    "
        "0        0.500        2025-01-02\n",
        "calls": "(call('/store'), call(status=None), [call('s0')], [True])",
    },
    "storage_error": {
        "exit": 1,
        "output": "Error listing sessions: disk\n",
        "calls": "(call('/store'), call(status=None), [], [])",
    },
    "missing_best_score_key": {
        "exit": 1,
        "output": "Session ID                     Function             "
        "Status       Trials   Best Score   Created\n"
        "----------------------------------------------------------------------------------------------------\n"
        "Error listing sessions: 'best_score'\n",
        "calls": "(call('/store'), call(status=None), [call('s0')], [])",
    },
}


@pytest.mark.parametrize("name", list(CASES))
def test_list_sessions(name):
    assert _snapshot(name) == GOLDEN[name]
