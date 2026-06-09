"""Logging-hygiene guards for bundled mock walkthroughs."""

from __future__ import annotations

import ast
from pathlib import Path


def _mock_mode_config_keys(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "MOCK_MODE_CONFIG"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            return set()
        keys: set[str] = set()
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
        return keys
    return set()


def test_mock_walkthrough_configs_do_not_use_inert_mock_keys() -> None:
    """Bundled examples must not trip the SDK's own inert-key warning."""
    repo_root = Path(__file__).resolve().parents[3]
    walkthroughs = sorted((repo_root / "walkthrough" / "mock").glob("[0-9][0-9]_*.py"))
    assert walkthroughs

    inert_keys = {"optimizer", "sampler", "random_seed"}
    offenders = {
        path.name: sorted(_mock_mode_config_keys(path) & inert_keys)
        for path in walkthroughs
        if _mock_mode_config_keys(path) & inert_keys
    }

    assert offenders == {}
