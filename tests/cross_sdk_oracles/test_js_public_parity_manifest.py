from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import traigent

JsonObject = dict[str, Any]


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    enterprise_root = os.environ.get("TRAIGENT_ENTERPRISE_ROOT")
    if enterprise_root:
        roots.append(Path(enterprise_root).expanduser())

    roots.extend(Path(__file__).resolve().parents)

    seen: set[Path] = set()
    unique_roots: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_roots.append(resolved)

    return unique_roots


def _resolve_existing_path(
    *,
    env_var: str,
    relative_path: Path,
    description: str,
) -> Path:
    override = os.environ.get(env_var)
    candidates = (
        [Path(override).expanduser()]
        if override
        else [root / relative_path for root in _candidate_roots()]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(candidate) for candidate in candidates)
    message = (
        f"Missing {description}. Set {env_var} or TRAIGENT_ENTERPRISE_ROOT. "
        f"Checked: {checked}"
    )
    if not override and os.environ.get("CI") != "true":
        pytest.skip(message)

    raise AssertionError(message)


def _load_json_object(path: Path, description: str) -> JsonObject:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{description} at {path} is not valid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise AssertionError(f"{description} at {path} must be a JSON object.")

    return payload


def _object_at(payload: JsonObject, path: tuple[str, ...], description: str) -> JsonObject:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            dotted = ".".join(path)
            raise AssertionError(f"{description} must define object field `{dotted}`.")
        value = value[key]

    if not isinstance(value, dict):
        dotted = ".".join(path)
        raise AssertionError(f"{description} field `{dotted}` must be an object.")

    return value


def _string_at(payload: JsonObject, path: tuple[str, ...], description: str) -> str:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            dotted = ".".join(path)
            raise AssertionError(f"{description} must define string field `{dotted}`.")
        value = value[key]

    if not isinstance(value, str) or not value:
        dotted = ".".join(path)
        raise AssertionError(f"{description} field `{dotted}` must be a non-empty string.")

    return value


def _string_list_at(payload: JsonObject, path: tuple[str, ...], description: str) -> list[str]:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            dotted = ".".join(path)
            raise AssertionError(f"{description} must define string-list field `{dotted}`.")
        value = value[key]

    dotted = ".".join(path)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AssertionError(f"{description} field `{dotted}` must be a list of strings.")

    return value


def _non_empty_string_list_at(
    payload: JsonObject,
    path: tuple[str, ...],
    description: str,
) -> list[str]:
    value = _string_list_at(payload, path, description)
    if not value:
        dotted = ".".join(path)
        raise AssertionError(f"{description} field `{dotted}` must not be empty.")

    return value


def _load_parity_manifest() -> JsonObject:
    manifest_path = _resolve_existing_path(
        env_var="TRAIGENT_PARITY_MANIFEST",
        relative_path=Path("TraigentSchema") / "parity" / "python-js-sdk.json",
        description="Python/JS parity manifest",
    )
    return _load_json_object(manifest_path, "Python/JS parity manifest")


def _load_js_surface() -> JsonObject:
    snapshot_path = _resolve_existing_path(
        env_var="TRAIGENT_JS_API_SURFACE_SNAPSHOT",
        relative_path=Path("traigent-js")
        / "tests"
        / "integration"
        / "fixtures"
        / "api-surface.snapshot.json",
        description="JS API surface snapshot",
    )
    return _load_json_object(snapshot_path, "JS API surface snapshot")


def _classified_symbols(manifest: JsonObject) -> set[str]:
    classifications = _object_at(manifest, ("classifications",), "Python/JS parity manifest")
    classified: set[str] = set()
    for classification, symbols in classifications.items():
        if not isinstance(symbols, list) or not all(isinstance(symbol, str) for symbol in symbols):
            raise AssertionError(
                "Python/JS parity manifest classification "
                f"`classifications.{classification}` must be a list of strings."
            )
        classified.update(symbols)

    return classified


def test_parity_manifest_declares_python_target() -> None:
    manifest = _load_parity_manifest()
    target_sha = _string_at(manifest, ("python", "targetSha"), "Python/JS parity manifest")
    expected_sha = os.environ.get("TRAIGENT_PARITY_EXPECTED_PYTHON_SHA")

    assert len(target_sha) == 40 and set(target_sha) <= set("0123456789abcdef"), (
        "Python/JS parity manifest `python.targetSha` must be a 40-character "
        f"lowercase Git SHA, got {target_sha!r}."
    )
    if expected_sha:
        assert target_sha == expected_sha


def test_python_public_clients_are_represented_in_js_api_surface() -> None:
    manifest = _load_parity_manifest()
    js_surface = _load_js_surface()
    js_root = set(_non_empty_string_list_at(js_surface, ("root",), "JS API surface snapshot"))

    required = set(
        _non_empty_string_list_at(
            manifest,
            ("javascript", "requiredRootExports"),
            "Python/JS parity manifest",
        )
    )
    missing = sorted(required - js_root)

    assert missing == [], (
        "JS root exports do not satisfy the schema-owned Python/JS parity manifest: "
        f"{missing}"
    )


def test_python_public_root_symbols_are_classified_in_parity_manifest() -> None:
    manifest = _load_parity_manifest()
    classified = _classified_symbols(manifest)
    missing = sorted(set(traigent.__all__) - classified)
    if missing and os.environ.get("CI") != "true":
        pytest.skip(
            "Local Python checkout exports symbols outside the pinned parity "
            f"manifest target: {missing}"
        )

    assert missing == [], (
        "Python exported root symbols are missing from the schema-owned "
        f"Python/JS parity manifest: {missing}"
    )
