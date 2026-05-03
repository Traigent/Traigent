from __future__ import annotations

import json
import os
from pathlib import Path

import traigent

IGNORED_FROM_CLIENT_PARITY: set[str] = set()

TRACKED_CONFIG_SYMBOLS = {
    "CoreMetricsConfig",
    "EnterpriseAdminConfig",
    "EvaluationConfig",
    "ObservabilityConfig",
    "ProjectManagementConfig",
    "PromptManagementConfig",
}


def _find_js_api_surface_snapshot() -> Path:
    env_path = os.environ.get("TRAIGENT_JS_SDK_PATH")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path).expanduser())

    current = Path(__file__).resolve()
    candidates.extend(parent / "traigent-js" for parent in current.parents)

    for js_repo in candidates:
        snapshot_path = (
            js_repo / "tests" / "integration" / "fixtures" / "api-surface.snapshot.json"
        )
        if snapshot_path.exists():
            return snapshot_path

    formatted = ", ".join(str(candidate) for candidate in candidates)
    raise AssertionError(
        "Missing JS API surface snapshot. Set TRAIGENT_JS_SDK_PATH or place "
        f"traigent-js beside a parent of this checkout. Checked: {formatted}"
    )


def test_python_public_clients_are_represented_in_js_api_surface() -> None:
    snapshot_path = _find_js_api_surface_snapshot()
    js_surface = json.loads(snapshot_path.read_text(encoding="utf-8"))
    js_root = set(js_surface["root"])
    # This guard tracks the public client/config migration surface. DTO/function
    # parity is covered separately through TraigentSchema generation and JS API
    # snapshot tests because JS intentionally exposes camelCase generated types.
    python_client_symbols = {
        name
        for name in traigent.__all__
        if name.endswith("Client") or name in TRACKED_CONFIG_SYMBOLS
    }

    missing = sorted(python_client_symbols - js_root - IGNORED_FROM_CLIENT_PARITY)
    assert missing == [], (
        "Python exported client/config symbols without JS parity coverage. "
        "Add the JS export/stub or document the omission in IGNORED_FROM_CLIENT_PARITY: "
        f"{missing}"
    )
