"""Characterization tests pinning the ``sync`` CLI command (sync_to_cloud)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from traigent.cli.local_commands import sync_to_cloud

MOD = "traigent.cli.local_commands"

STATUS = {
    "total_sessions": 3,
    "completed_sessions": 2,
    "sync_eligible": 2,
    "total_trials": 5,
}
STATUS_NONE = {**STATUS, "sync_eligible": 0}
ALL_OK = {
    "total_sessions": 2,
    "synced_successfully": 2,
    "sync_errors": 0,
    "session_results": [
        {"session_id": "a", "status": "success"},
        {"session_id": "b", "status": "success"},
    ],
}
ALL_MIXED = {
    "total_sessions": 3,
    "synced_successfully": 1,
    "sync_errors": 2,
    "session_results": [
        {"session_id": "a", "status": "success"},
        {"session_id": "b", "status": "failed", "errors": ["bad"]},
        {"session_id": "c", "status": "failed"},
    ],
}
ALL_NONE_OK = {
    "total_sessions": 1,
    "synced_successfully": 0,
    "sync_errors": 1,
    "session_results": [{"session_id": "b", "status": "failed", "errors": ["x"]}],
}
ONE_OK = {"status": "success", "cloud_url": "https://portal/x"}
ONE_OK_NO_URL = {"status": "success"}
ONE_FAIL = {"status": "failed", "errors": ["e1", "e2"]}

# name -> (cli args, status, one-result, all-result, offline, api_key_from_cfg,
#          confirms, expect_sync_manager)
CASES = {
    "offline": ([], STATUS, None, None, True, "k", [], False),
    "no_key": ([], STATUS, None, None, False, None, [], False),
    "dry_no_key_all": (["--dry-run"], STATUS, None, ALL_OK, False, None, [], True),
    "no_eligible": (["--api-key", "k"], STATUS_NONE, None, None, False, None, [], True),
    "one_ok": (
        ["--api-key", "k", "--session-id", "s"],
        STATUS,
        ONE_OK,
        None,
        False,
        None,
        [],
        True,
    ),
    "one_ok_no_url": (
        ["--api-key", "k", "--session-id", "s"],
        STATUS,
        ONE_OK_NO_URL,
        None,
        False,
        None,
        [],
        True,
    ),
    "one_ok_dry": (
        ["--dry-run", "--session-id", "s"],
        STATUS,
        ONE_OK,
        None,
        False,
        None,
        [],
        True,
    ),
    "one_fail": (
        ["--api-key", "k", "--session-id", "s"],
        STATUS,
        ONE_FAIL,
        None,
        False,
        "cfgkey",
        [],
        True,
    ),
    "all_cancel": (
        ["--api-key", "k"],
        STATUS,
        None,
        ALL_OK,
        False,
        None,
        [False],
        True,
    ),
    "all_ok": (["--api-key", "k"], STATUS, None, ALL_OK, False, None, [True], True),
    "all_mixed": (
        ["--api-key", "k"],
        STATUS,
        None,
        ALL_MIXED,
        False,
        None,
        [True],
        True,
    ),
    "all_cleanup_yes": (
        ["--api-key", "k", "--cleanup"],
        STATUS,
        None,
        ALL_OK,
        False,
        None,
        [True, True],
        True,
    ),
    "all_cleanup_no": (
        ["--api-key", "k", "--cleanup"],
        STATUS,
        None,
        ALL_OK,
        False,
        None,
        [True, False],
        True,
    ),
    "all_cleanup_none_ok": (
        ["--api-key", "k", "--cleanup"],
        STATUS,
        None,
        ALL_NONE_OK,
        False,
        None,
        [True],
        True,
    ),
    "all_cleanup_dry": (
        ["--dry-run", "--cleanup"],
        STATUS,
        None,
        ALL_OK,
        False,
        None,
        [],
        True,
    ),
}


def _run(name):
    args, status, one, all_, offline, cfg_key, confirms, _ = CASES[name]
    manager = MagicMock()
    manager.get_sync_status.return_value = status
    manager.sync_session_to_cloud.return_value = one
    manager.sync_all_sessions.return_value = all_
    manager.cleanup_after_sync.return_value = {"sessions_deleted": 1}
    confirm_mock = MagicMock(side_effect=list(confirms))
    with (
        patch(f"{MOD}.TraigentConfig") as cfg_cls,
        patch(f"{MOD}._offline_requested", return_value=offline),
        patch(f"{MOD}.SyncManager", return_value=manager) as sm_cls,
        patch(
            "traigent.config.backend_config.BackendConfig.get_api_key",
            return_value=cfg_key,
        ),
        patch(f"{MOD}.click.confirm", confirm_mock),
    ):
        cfg_cls.from_environment.return_value = "CFG"
        result = CliRunner().invoke(sync_to_cloud, args)
    calls = {
        "sm_args": sm_cls.call_args,
        "status": manager.get_sync_status.call_count,
        "one": manager.sync_session_to_cloud.call_args_list,
        "all": manager.sync_all_sessions.call_args_list,
        "cleanup": manager.cleanup_after_sync.call_args_list,
        "preview": manager.get_cloud_analytics_preview.call_count,
        "confirm": confirm_mock.call_count,
    }
    return result, calls


def _snapshot(name):
    result, calls = _run(name)
    return {
        "exit": result.exit_code,
        "output": result.output,
        "calls": repr(calls),
    }


GOLDEN: dict = {
    "offline": {
        "exit": 1,
        "output": "Offline is enabled; portal sync is disabled.\n",
        "calls": "{'sm_args': None, 'status': 0, 'one': [], 'all': [], 'cleanup': [], "
        "'preview': 0, 'confirm': 0}",
    },
    "no_key": {
        "exit": 1,
        "output": "❌ API key required for portal sync. Use --api-key or set TRAIGENT_API_KEY "
        "environment variable.\n"
        "💡 Configure a key with `traigent auth login` or export TRAIGENT_API_KEY.\n",
        "calls": "{'sm_args': None, 'status': 0, 'one': [], 'all': [], 'cleanup': [], "
        "'preview': 0, 'confirm': 0}",
    },
    "dry_no_key_all": {
        "exit": 0,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔍 Dry Run Mode - Preview Only\n"
        "\n"
        "🔄 Syncing all 2 eligible sessions...\n"
        "\n"
        "📈 Sync Results:\n"
        "   Total sessions: 2\n"
        "   Synced successfully: 2\n"
        "   Errors: 0\n",
        "calls": "{'sm_args': call('CFG', None), 'status': 1, 'one': [], 'all': "
        "[call(True)], 'cleanup': [], 'preview': 0, 'confirm': 0}",
    },
    "no_eligible": {
        "exit": 0,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 0\n"
        "Total trials: 5\n"
        "\n"
        "💡 No completed sessions to sync. Complete some optimizations first!\n",
        "calls": "{'sm_args': call('CFG', 'k'), 'status': 1, 'one': [], 'all': [], "
        "'cleanup': [], 'preview': 0, 'confirm': 0}",
    },
    "one_ok": {
        "exit": 0,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔄 Syncing session: s\n"
        "✅ Session synced successfully!\n"
        "   Portal URL: https://portal/x\n"
        "\n"
        "🎯 Available in the Traigent portal:\n"
        "   • Cross-function optimization insights\n"
        "   • Performance trend analysis\n"
        "   • Team collaboration features\n"
        "   • Advanced visualization dashboard\n",
        "calls": "{'sm_args': call('CFG', 'k'), 'status': 1, 'one': [call('s', False)], 'all': "
        "[], 'cleanup': [], 'preview': 1, 'confirm': 0}",
    },
    "one_ok_no_url": {
        "exit": 0,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔄 Syncing session: s\n"
        "✅ Session synced successfully!\n"
        "   Portal URL: N/A\n"
        "\n"
        "🎯 Available in the Traigent portal:\n"
        "   • Cross-function optimization insights\n"
        "   • Performance trend analysis\n"
        "   • Team collaboration features\n"
        "   • Advanced visualization dashboard\n",
        "calls": "{'sm_args': call('CFG', 'k'), 'status': 1, 'one': [call('s', False)], "
        "'all': [], 'cleanup': [], 'preview': 1, 'confirm': 0}",
    },
    "one_ok_dry": {
        "exit": 0,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔍 Dry Run Mode - Preview Only\n"
        "\n"
        "🔄 Syncing session: s\n"
        "✅ Session synced successfully!\n",
        "calls": "{'sm_args': call('CFG', None), 'status': 1, 'one': [call('s', True)], "
        "'all': [], 'cleanup': [], 'preview': 0, 'confirm': 0}",
    },
    "one_fail": {
        "exit": 1,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔄 Syncing session: s\n"
        "❌ Sync failed:\n"
        "   • e1\n"
        "   • e2\n",
        "calls": "{'sm_args': call('CFG', 'k'), 'status': 1, 'one': [call('s', False)], "
        "'all': [], 'cleanup': [], 'preview': 0, 'confirm': 0}",
    },
    "all_cancel": {
        "exit": 0,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔄 Syncing all 2 eligible sessions...\n"
        "Sync cancelled.\n",
        "calls": "{'sm_args': call('CFG', 'k'), 'status': 1, 'one': [], 'all': [], "
        "'cleanup': [], 'preview': 0, 'confirm': 1}",
    },
    "all_ok": {
        "exit": 0,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔄 Syncing all 2 eligible sessions...\n"
        "\n"
        "📈 Sync Results:\n"
        "   Total sessions: 2\n"
        "   Synced successfully: 2\n"
        "   Errors: 0\n"
        "\n"
        "🎯 Available in the Traigent portal:\n"
        "   • Cross-function optimization insights\n"
        "   • Performance trend analysis\n"
        "   • Team collaboration features\n"
        "   • Advanced visualization dashboard\n",
        "calls": "{'sm_args': call('CFG', 'k'), 'status': 1, 'one': [], 'all': [call(False)], "
        "'cleanup': [], 'preview': 1, 'confirm': 1}",
    },
    "all_mixed": {
        "exit": 1,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔄 Syncing all 2 eligible sessions...\n"
        "\n"
        "📈 Sync Results:\n"
        "   Total sessions: 3\n"
        "   Synced successfully: 1\n"
        "   Errors: 2\n"
        "\n"
        "❌ Sessions with errors:\n"
        "   • b: ['bad']\n"
        "   • c: ['Unknown error']\n",
        "calls": "{'sm_args': call('CFG', 'k'), 'status': 1, 'one': [], 'all': "
        "[call(False)], 'cleanup': [], 'preview': 0, 'confirm': 1}",
    },
    "all_cleanup_yes": {
        "exit": 0,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔄 Syncing all 2 eligible sessions...\n"
        "\n"
        "📈 Sync Results:\n"
        "   Total sessions: 2\n"
        "   Synced successfully: 2\n"
        "   Errors: 0\n"
        "✅ Cleaned up 1 sessions (backed up first)\n"
        "\n"
        "🎯 Available in the Traigent portal:\n"
        "   • Cross-function optimization insights\n"
        "   • Performance trend analysis\n"
        "   • Team collaboration features\n"
        "   • Advanced visualization dashboard\n",
        "calls": "{'sm_args': call('CFG', 'k'), 'status': 1, 'one': [], 'all': "
        "[call(False)], 'cleanup': [call(['a', 'b'], keep_backup=True)], "
        "'preview': 1, 'confirm': 2}",
    },
    "all_cleanup_no": {
        "exit": 0,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔄 Syncing all 2 eligible sessions...\n"
        "\n"
        "📈 Sync Results:\n"
        "   Total sessions: 2\n"
        "   Synced successfully: 2\n"
        "   Errors: 0\n"
        "\n"
        "🎯 Available in the Traigent portal:\n"
        "   • Cross-function optimization insights\n"
        "   • Performance trend analysis\n"
        "   • Team collaboration features\n"
        "   • Advanced visualization dashboard\n",
        "calls": "{'sm_args': call('CFG', 'k'), 'status': 1, 'one': [], 'all': "
        "[call(False)], 'cleanup': [], 'preview': 1, 'confirm': 2}",
    },
    "all_cleanup_none_ok": {
        "exit": 1,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔄 Syncing all 2 eligible sessions...\n"
        "\n"
        "📈 Sync Results:\n"
        "   Total sessions: 1\n"
        "   Synced successfully: 0\n"
        "   Errors: 1\n"
        "\n"
        "❌ Sessions with errors:\n"
        "   • b: ['x']\n",
        "calls": "{'sm_args': call('CFG', 'k'), 'status': 1, 'one': [], 'all': "
        "[call(False)], 'cleanup': [], 'preview': 0, 'confirm': 1}",
    },
    "all_cleanup_dry": {
        "exit": 0,
        "output": "\n"
        "📊 Sync Status\n"
        "==============================\n"
        "Total sessions: 3\n"
        "Completed sessions: 2\n"
        "Sync eligible: 2\n"
        "Total trials: 5\n"
        "\n"
        "🔍 Dry Run Mode - Preview Only\n"
        "\n"
        "🔄 Syncing all 2 eligible sessions...\n"
        "\n"
        "📈 Sync Results:\n"
        "   Total sessions: 2\n"
        "   Synced successfully: 2\n"
        "   Errors: 0\n",
        "calls": "{'sm_args': call('CFG', None), 'status': 1, 'one': [], 'all': "
        "[call(True)], 'cleanup': [], 'preview': 0, 'confirm': 0}",
    },
}


@pytest.mark.parametrize("name", list(CASES))
def test_sync_to_cloud_output_and_calls(name):
    assert _snapshot(name) == GOLDEN[name]


def test_exception_is_reported_and_exits_one():
    manager = MagicMock()
    manager.get_sync_status.side_effect = RuntimeError("boom")
    with (
        patch(f"{MOD}.TraigentConfig"),
        patch(f"{MOD}._offline_requested", return_value=False),
        patch(f"{MOD}.SyncManager", return_value=manager),
    ):
        result = CliRunner().invoke(sync_to_cloud, ["--api-key", "k"])
    assert result.exit_code == 1
    assert "Error syncing to portal: boom" in result.output
