"""The declared evaluator version (``EvaluatorVersionManifestV1``) of the evaluator in use.

Spec: TraigentSchema ``docs/identity/content-identity-v1.md`` section 9 and
``schemas/evaluation/evaluator_version_manifest_v1_schema.json``. The SDK sends
it as an ``EvaluatorVersionBindingV1`` with ``resolution:
"declared_at_session_start"``: the version it is about to score with. Only the
Backend can witness what actually scored (``witnessed_at_scoring``, M3), so a
certificate may never claim an evaluator version from this alone (rule C3).

How each field is built for the evaluator the run actually uses:

``evaluator_id``
    The declared evaluator id (``evaluator_definition_id``), else
    ``"sdk_local_evaluator"`` with ``evaluator_id_source: "fallback"`` on the
    wrapper (same fallback as the JS SDK).
``code_digest``
    fp2 ``efp2``. When the evaluator scores with user code (a
    ``scoring_function``, ``metric_functions`` or a user-defined evaluator
    class), ``source`` is each piece's decorator-free, dedented source under a
    ``# <slot>`` header, in slot-name order. When it scores only with the
    SDK's built-in scorers, ``external`` names them with the SDK version as the
    immutable revision. Unreadable source -> no manifest
    (``evaluator_manifest_unavailable``).
``config_digest``
    fp2 digest of ``{"metrics": [...], "bound": {slot: bound state}}`` -- the
    metric list and every closure / partial / instance value the scoring code
    carries. Not canonically serializable -> no manifest.
``helper_digests``
    Contents of the project-local source files defining that user code
    (never third-party or SDK files). ``{}`` for built-in scoring.
``judge``
    Explicit ``null``: the SDK has no generic way to see an LLM judge inside
    user scoring code. Judge calls it does intercept appear in the trial's
    ``observed_provider_versions``.
``objectives``
    The run's objectives, sorted by name. A ``band`` objective has no
    maximize/minimize orientation in this schema -> no manifest.
``dependency_versions``
    ``{}`` (JS parity). For built-in scoring the SDK version is already the
    ``efp2`` external revision.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from traigent.identity import agent_build as _ab
from traigent.identity.content_identity import (
    ContentIdentityError,
    compute_evaluator_version_digest,
)
from traigent.utils import fp2

__all__ = ["FALLBACK_EVALUATOR_ID", "build_evaluator_binding"]

#: Evaluator id used when none is declared (same as the JS SDK).
FALLBACK_EVALUATOR_ID = "sdk_local_evaluator"

_FOREIGN_KEY_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _scoring_code(evaluator: Any) -> dict[str, Any]:
    """``slot -> callable or class`` for the user code the evaluator scores with."""
    code: dict[str, Any] = {}
    scoring = getattr(evaluator, "scoring_function", None)
    if callable(scoring):
        code["scoring_function"] = scoring
    metric_functions = getattr(evaluator, "metric_functions", None)
    if isinstance(metric_functions, Mapping):
        for name, fn in metric_functions.items():
            if callable(fn):
                code[f"metric:{name}"] = fn
    cls = type(evaluator)
    module_file = getattr(inspect.getmodule(cls), "__file__", None)
    if isinstance(module_file, str) and not _ab._is_excluded(
        Path(module_file).resolve()
    ):
        code["evaluator_class"] = cls
    return code


def _source(obj: Any) -> str | None:
    if inspect.isclass(obj):
        try:
            text = inspect.getsource(obj)
        except (OSError, TypeError):
            return None
        return _ab.normalize_source(text, first_line=("class ",))
    target = _ab.innermost_callable(getattr(obj, "func", obj))
    return _ab._source_without_decorators(getattr(target, "__func__", target))


def _helper_digests(code: Mapping[str, Any]) -> dict[str, str] | None:
    digests: dict[str, str] = {}
    for obj in code.values():
        target = obj if inspect.isclass(obj) else _ab.innermost_callable(obj)
        try:
            filename = inspect.getsourcefile(getattr(target, "__func__", target))
        except TypeError:
            filename = None
        if not filename:
            return None
        path = Path(filename).resolve()
        if _ab._is_excluded(path):
            continue
        name = _ab.project_relative_name(path)
        if name is None:
            return None
        try:
            digests[name] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None
    return digests


def _objectives(objectives: Iterable[Any] | None) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    for objective in objectives or []:
        name = getattr(objective, "name", None)
        orientation = getattr(objective, "orientation", None)
        weight = getattr(objective, "weight", None)
        if isinstance(objective, Mapping):
            name = objective.get("name")
            orientation = objective.get("orientation")
            weight = objective.get("weight", 1.0)
        if orientation not in ("maximize", "minimize"):
            return None
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            return None
        rows.append({"name": name, "orientation": orientation, "weight": weight})
    if not rows:
        return None
    return sorted(rows, key=lambda row: str(row["name"]))


def build_evaluator_binding(
    evaluator: Any,
    *,
    objectives: Iterable[Any] | None,
    evaluator_id: str | None,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """``(EvaluatorVersionBindingV1 | None, evaluator_id_source, reason)``."""
    if evaluator is None:
        return None, None, "evaluator_manifest_unavailable"
    declared = evaluator_id.strip() if isinstance(evaluator_id, str) else ""
    if declared:
        if not _FOREIGN_KEY_ID.fullmatch(declared):
            return None, None, "evaluator_id_unavailable"
        resolved_id, source = declared, "declared"
    else:
        resolved_id, source = FALLBACK_EVALUATOR_ID, "fallback"

    code = _scoring_code(evaluator)
    sdk_version = _ab.sdk_version()
    try:
        if code:
            sources: list[str] = []
            for slot in sorted(code):
                text = _source(code[slot])
                if text is None:
                    return None, source, "evaluator_manifest_unavailable"
                sources.append(f"# {slot}\n{text}")
            efp2: dict[str, Any] = {
                "kind": "efp2",
                "runtime": "python",
                "source": "\n\n".join(sources),
            }
        else:
            efp2 = {
                "kind": "efp2",
                "runtime": "python",
                "external": {
                    "kind": "traigent_builtin",
                    "revision": f"{type(evaluator).__name__}@{sdk_version}",
                },
            }
        bound: dict[str, Any] = {}
        for slot in sorted(code):
            obj = code[slot]
            if inspect.isclass(obj):
                continue
            partial = obj if isinstance(obj, functools.partial) else None
            target = _ab.innermost_callable(partial.func if partial else obj)
            state = _ab._bound_state(partial, target)
            if state is None:
                return None, source, "evaluator_manifest_unavailable"
            if state:
                bound[slot] = state
        metrics = getattr(evaluator, "metrics", None)
        config = {
            "metrics": sorted(str(m) for m in metrics)
            if isinstance(metrics, (list, tuple))
            else [],
            "bound": bound,
        }
        helper_digests = _helper_digests(code)
        objective_rows = _objectives(objectives)
        if helper_digests is None or objective_rows is None:
            return None, source, "evaluator_manifest_unavailable"
        manifest = {
            "manifest_version": 1,
            "evaluator_id": resolved_id,
            "code_digest": str(fp2.digest(efp2)),
            "config_digest": str(fp2.digest(config)),
            "helper_digests": helper_digests,
            "judge": None,
            "objectives": objective_rows,
            "dependency_versions": {},
        }
        version_digest = compute_evaluator_version_digest(manifest)
    except (fp2.Fp2UnsupportedValue, ContentIdentityError, TypeError, ValueError):
        return None, source, "evaluator_manifest_unavailable"
    return (
        {
            "evaluator_id": resolved_id,
            "version_digest": version_digest,
            "resolution": "declared_at_session_start",
            "manifest": manifest,
        },
        source,
        None,
    )
