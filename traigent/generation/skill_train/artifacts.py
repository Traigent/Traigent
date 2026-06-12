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

from .trainer_result import SkillTrainResult


def write_artifacts(
    directory: str | Path,
    result: SkillTrainResult,
    history: Sequence[dict[str, Any]],
) -> str:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    (path / "best_skill.md").write_text(result.best_document, encoding="utf-8")
    report = [record.to_dict() for record in result.all_edit_records]
    (path / "edit_apply_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    meta_skill = result.summary.get("meta_skill", "")
    if isinstance(meta_skill, str):
        (path / "meta_skill.md").write_text(meta_skill, encoding="utf-8")

    with (path / "training_log.jsonl").open("w", encoding="utf-8") as handle:
        for entry in history:
            safe = {
                "doc_hash": entry.get("doc_hash"),
                "split": entry.get("split"),
                "score": entry.get("score"),
                "decision": entry.get("decision"),
            }
            handle.write(json.dumps(safe, sort_keys=True) + "\n")
    return str(path)


__all__ = ["write_artifacts"]
