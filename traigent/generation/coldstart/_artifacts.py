"""Local-only artifact writer for a built cold-start eval set.

Mirrors ``traigent/generation/skill_train/artifacts.py``'s symlink-safe,
containment-rooted write pattern. This module is reached ONLY on the
EVAL_SET_BUILT path with at least one verified row already in hand -- a
fail-closed DISCOVERY_ONLY gap never calls into this module, so a gap can
never leave a partial file behind. Everything is built in memory first, then
the JSONL and the manifest are written and put into place together: both
land, or neither does. See ``_write_pair_atomically`` -- a caller must never
observe a JSONL without its manifest, even if the manifest write fails
partway through (e.g. its path is already a directory).
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from traigent.utils.secure_path import (
    PathTraversalError,
    sanitize_filename,
    validate_path,
)

from ._generation import VerifiedRow


def _artifact_target(directory: Path, filename: str) -> Path:
    """Return the artifact path, refusing symlinked targets.

    Artifact filenames are derived from ``dataset_name``, so the only
    redirection primitive is a pre-existing symlink planted at that name;
    writing through it would clobber an arbitrary file outside the output
    directory.
    """
    target = directory / filename
    if target.is_symlink():
        raise ValueError(
            f"refusing to write cold-start artifact through a symlink: {target}"
        )
    return target


def write_eval_set(
    output_dir: str | Path,
    dataset_name: str,
    rows: Sequence[VerifiedRow],
    *,
    plan_id: str,
    descriptor: Mapping[str, Any],
    containment_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write the tuning JSONL + manifest for one built eval set.

    Every row is required to already carry a passing ``ScoreReceipt`` (see
    ``_generation.generate_and_score``) -- this function does not verify
    anything itself, it only serializes what was already accepted.
    """
    safe_name = sanitize_filename(dataset_name)
    directory = _resolve_directory(output_dir, containment_root=containment_root)
    directory.mkdir(parents=True, exist_ok=True)

    eval_set_path = _artifact_target(directory, f"{safe_name}.jsonl")
    manifest_path = _artifact_target(directory, f"{safe_name}.manifest.json")

    jsonl_lines: list[str] = []
    receipt_summaries: list[dict[str, Any]] = []
    for inputs, output, receipt in rows:
        # Cold-start rows are synthetic candidates a LOCAL verifier accepted
        # -- never holdout examples. This field is set here, by executor
        # code, not by anything a caller (generator/verifier) can override.
        jsonl_lines.append(
            json.dumps(
                {
                    "input": dict(inputs),
                    "output": output,
                    "holdout": False,
                    "synthetic": True,
                },
                sort_keys=True,
            )
        )
        receipt_summaries.append(
            {
                "verifier_id": receipt.verifier_id,
                "verifier_kind": receipt.verifier_kind,
                "passed": receipt.passed,
                "provenance": receipt.provenance,
            }
        )

    manifest = {
        "plan_id": plan_id,
        "descriptor": dict(descriptor),
        "row_count": len(rows),
        "synthetic": True,
        "holdout": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "receipts": receipt_summaries,
    }

    # Build full file contents up front so a mid-write failure can never
    # leave a half-written JSONL/manifest pair on disk.
    eval_set_text = "\n".join(jsonl_lines) + ("\n" if jsonl_lines else "")
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)

    _write_pair_atomically(eval_set_path, eval_set_text, manifest_path, manifest_text)
    return eval_set_path, manifest_path


def _write_pair_atomically(
    eval_set_path: Path,
    eval_set_text: str,
    manifest_path: Path,
    manifest_text: str,
) -> None:
    """Put the JSONL and its manifest into place together, or neither at all.

    Each file is first written to a temporary sibling (same directory, so
    the follow-up rename is a same-filesystem, effectively-atomic
    ``os.replace``), then the two temporaries are moved into their final
    paths. If anything raises -- including the second rename, e.g. because
    ``manifest_path`` already exists as a directory (``IsADirectoryError``)
    -- any temporary already written and any final path already put in
    place by this call are removed before the exception propagates. A
    caller must never observe the JSONL at its final path without a
    manifest alongside it.
    """
    temp_eval_set = _sibling_temp_path(eval_set_path)
    temp_manifest = _sibling_temp_path(manifest_path)
    placed: list[Path] = []
    try:
        temp_eval_set.write_text(eval_set_text, encoding="utf-8")
        temp_manifest.write_text(manifest_text, encoding="utf-8")

        os.replace(temp_eval_set, eval_set_path)
        placed.append(eval_set_path)
        os.replace(temp_manifest, manifest_path)
        placed.append(manifest_path)
    except BaseException:
        for path in placed:
            path.unlink(missing_ok=True)
        raise
    finally:
        temp_eval_set.unlink(missing_ok=True)
        temp_manifest.unlink(missing_ok=True)


def _sibling_temp_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")


def _resolve_directory(
    directory: str | Path, *, containment_root: str | Path | None
) -> Path:
    path = Path(directory).expanduser().resolve()
    if containment_root is None:
        return path

    root = Path(containment_root).expanduser().resolve()
    try:
        return validate_path(directory, allowed_base=root)
    except PathTraversalError as exc:
        raise ValueError(
            "cold-start output directory escapes its containment root: "
            f"{path} is not under {root}"
        ) from exc
