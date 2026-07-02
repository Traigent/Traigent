from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from traigent.examples._portal import PORTAL_API_KEYS_URL, PORTAL_URL
from traigent.examples.quickstart._env import configure_quickstart_env
from traigent.examples.tutorial_bootstrap import configure_tutorial_mock_mode
from traigent.testing import _reset_for_tests


@pytest.fixture(autouse=True)
def reset_mock_mode() -> None:
    _reset_for_tests()
    yield
    _reset_for_tests()


def test_packaged_examples_share_portal_urls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    quickstart_env: dict[str, str] = {}
    configure_quickstart_env(quickstart_env)
    assert PORTAL_API_KEYS_URL in capsys.readouterr().err

    tutorial_env: dict[str, str] = {}
    configure_tutorial_mock_mode(
        provider_env_keys=("ANTHROPIC_API_KEY",),
        tutorial_name="Simple Prompt Optimization",
        results_base=tmp_path,
        env=tutorial_env,
    )
    assert PORTAL_API_KEYS_URL in capsys.readouterr().err

    module = importlib.import_module("traigent.examples.quickstart.publish_and_verify")
    assert PORTAL_URL in (module.__doc__ or "")
