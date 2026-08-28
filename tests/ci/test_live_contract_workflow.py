"""Static and shell-level checks for the live-contract secret handoff."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "live-contract.yml"
)
COMPOSE_SECRET_NAMES = (
    "JWT_SECRET_KEY",
    "SECRET_KEY",
    "TRAIGENT_MASTER_KEY",
    "TRAIGENT_ID_SECRET",
)


def _step_script(step_name: str) -> str:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    steps = workflow["jobs"]["sdk-backend-live-contract"]["steps"]
    return next(step["run"] for step in steps if step.get("name") == step_name)


def test_generated_compose_secrets_are_masked_before_github_env_write(
    tmp_path: Path,
) -> None:
    """Exercise the exact shell block with a harmless deterministic generator."""
    script = _step_script("Prepare compose environment")
    mask = "printf '::add-mask::%s\\n' \"$value\""
    persist = 'printf \'%s=%s\\n\' "$name" "$value" >> "$GITHUB_ENV"'
    assert script.index(mask) < script.index(persist)
    for name in COMPOSE_SECRET_NAMES:
        assert f"write_masked_env {name}" in script

    env_file = tmp_path / "github-env"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "openssl").write_text("#!/usr/bin/env bash\nprintf '%s\\n' test-value\n")
    (bin_dir / "openssl").chmod(0o755)

    result = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "GITHUB_ENV": str(env_file),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["::add-mask::test-value"] * 4
    assert env_file.read_text().splitlines() == [
        f"{name}=test-value" for name in COMPOSE_SECRET_NAMES
    ]


def test_live_contract_api_key_is_redirected_before_masking() -> None:
    """The key read cannot write its stdout to the Actions log before masking."""
    script = _step_script("Mint live-contract API key")

    assert "set -euo pipefail" in script
    assert 'key="$(compose exec' not in script
    assert 'compose exec -T backend cat /tmp/ci_api_key >"$key_file"' in script
    assert 'key="$(<"$key_file")"' in script
    assert script.index('>"$key_file"') < script.index(
        "printf '::add-mask::%s\\n' \"$key\""
    )
