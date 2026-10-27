"""Regenerate ``expected.json`` by running the BACKEND projector on ``cases.json``.

Usage (stdlib only, no Flask/DB; the backend's tenant guard is stubbed)::

    git -C <TraigentBackend> show 2d5cceda5:src/services/optimization_only_metadata_contract.py \\
        > /tmp/be/optimization_only_metadata_contract.py
    python -I generate_expected.py /tmp/be/optimization_only_metadata_contract.py

The backend module does ``from src.services import tenant_optimization_only``, which pulls in
Flask and SQLAlchemy models. None of that is used by the projectors, so a stub package is
injected into ``sys.modules`` before the module is loaded.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_backend(path: str):
    services = types.ModuleType("src.services")
    guard = types.ModuleType("src.services.tenant_optimization_only")
    services.tenant_optimization_only = guard  # type: ignore[attr-defined]
    src = types.ModuleType("src")
    src.services = services  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "src": src,
            "src.services": services,
            "src.services.tenant_optimization_only": guard,
        }
    )
    spec = importlib.util.spec_from_file_location("backend_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(c, case):
    phase, data, declared = case["phase"], case["input"], case["declared"]
    if phase in ("session_create",):
        return c.project_session_create(data)
    if phase == "session_create_legacy":
        return c.project_legacy_session_create(data)
    if phase in ("results", "summary_submission"):
        return c.project_trial_result(
            metrics=data.get("metrics"),
            measures=data.get("measures"),
            summary_stats=data.get("summary_stats"),
            metadata=data.get("metadata"),
            error_message=data.get("error_message"),
            declared=declared,
        )
    if phase == "finalize":
        return c.project_session_finalize(data)
    if phase == "configuration_run":
        return c.project_configuration_run(data, declared)
    if phase == "parent_run":
        return c.project_parent_run_create(data)
    if phase == "constraints":
        return c._project_constraints(data, c._Ctx(declared))
    if phase == "tvar_observation":
        return c._run(c._TVAR_OBSERVATION, data, c._Ctx(declared))
    raise SystemExit(f"unknown phase {phase}")


def main() -> None:
    c = load_backend(sys.argv[1])
    cases = json.loads((HERE / "cases.json").read_text())
    expected = {}
    for name, case in sorted(cases.items()):
        try:
            expected[name] = {"ok": run_case(c, case)}
        except c.MetadataContractViolation:
            expected[name] = {"violation": True}
    (HERE / "expected.json").write_text(
        json.dumps(expected, indent=1, sort_keys=True, allow_nan=True) + "\n"
    )
    print(f"{len(expected)} cases; contract v{c.METADATA_CONTRACT_VERSION}")


if __name__ == "__main__":
    main()
