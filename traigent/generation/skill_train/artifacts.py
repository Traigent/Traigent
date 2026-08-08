"""Local artifact writing for skill training.

Artifacts in this module are local files only. ``best_skill.md`` and
``edit_apply_report.json`` may contain accepted edit content or rationales that
quote training data, so they are derived-from-training-data artifacts.
``training_log.jsonl`` is intentionally restricted to document hashes, split
names, scores, and decisions.

Artifact writes are controlled by ``SkillTrainOptions.write_artifacts``. The
trainer writes by default because ``train_skill`` is an explicit local training
action; callers can set the option to ``False`` to suppress all filesystem
writes while still receiving the full result object in memory. When the caller
does not provide ``artifacts_dir``, the trainer roots generated artifact
directories under the optimized function's local storage path when available,
falling back to ``.traigent``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from traigent.utils.secure_path import PathTraversalError, validate_path

from .trainer_result import SkillTrainResult


def _artifact_target(directory: Path, filename: str) -> Path:
    """Return the artifact path, refusing symlinked targets.

    Artifact filenames are fixed constants, so the only redirection primitive
    is a pre-existing symlink planted at one of those names; writing through it
    would clobber an arbitrary file outside the artifacts directory.
    """
    target = directory / filename
    if target.is_symlink():
        raise ValueError(
            f"refusing to write skill-train artifact through a symlink: {target}"
        )
    return target


def write_artifacts(
    directory: str | Path,
    result: SkillTrainResult,
    history: Sequence[dict[str, Any]],
    *,
    containment_root: str | Path | None = None,
) -> str:
    path = _resolve_artifact_directory(directory, containment_root=containment_root)
    path.mkdir(parents=True, exist_ok=True)

    _artifact_target(path, "best_skill.md").write_text(
        result.best_document, encoding="utf-8"
    )
    report = [record.to_dict() for record in result.all_edit_records]
    _artifact_target(path, "edit_apply_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    meta_skill = result.summary.get("meta_skill", "")
    if isinstance(meta_skill, str):
        _artifact_target(path, "meta_skill.md").write_text(meta_skill, encoding="utf-8")

    with _artifact_target(path, "training_log.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for entry in history:
            safe = {
                "doc_hash": entry.get("doc_hash"),
                "split": entry.get("split"),
                "score": entry.get("score"),
                "decision": entry.get("decision"),
            }
            handle.write(json.dumps(safe, sort_keys=True) + "\n")
    return str(path)


def _resolve_artifact_directory(
    directory: str | Path,
    *,
    containment_root: str | Path | None,
) -> Path:
    path = Path(directory).expanduser().resolve()
    if containment_root is None:
        return path

    root = Path(containment_root).expanduser().resolve()
    try:
        return validate_path(directory, allowed_base=root)
    except PathTraversalError as exc:
        raise ValueError(
            "skill-train artifacts directory escapes its storage root: "
            f"{path} is not under {root}"
        ) from exc


__all__ = ["write_artifacts"]
