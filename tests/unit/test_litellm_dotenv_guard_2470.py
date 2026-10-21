"""#2470: TRAIGENT_SKIP_DOTENV must also stop litellm's own import-time .env load."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_PROBE = "import traigent, os; print(os.environ.get('LITELLM_MODE', '<unset>'))"


def _run(skip: str | None, litellm_mode: str | None = None) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("LITELLM_MODE",)}
    env.pop("TRAIGENT_SKIP_DOTENV", None)
    if skip is not None:
        env["TRAIGENT_SKIP_DOTENV"] = skip
    if litellm_mode is not None:
        env["LITELLM_MODE"] = litellm_mode
    env["PYTHONPATH"] = str(_ROOT)
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_skip_dotenv_pins_litellm_to_production_mode():
    assert _run("1") == "PRODUCTION"


def test_explicit_litellm_mode_is_respected():
    assert _run("1", litellm_mode="DEV") == "DEV"


def test_without_skip_dotenv_litellm_mode_is_untouched():
    assert _run(None) == "<unset>"
