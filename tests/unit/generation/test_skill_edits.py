from __future__ import annotations

import json

from traigent.generation.skill_train.edits import (
    MAX_DOC_CHARS,
    EditOp,
    apply_edits,
    parse_edit_ops,
)


def test_apply_each_op_type_and_first_occurrence() -> None:
    text = "alpha beta alpha"
    out, records = apply_edits(
        text,
        [
            EditOp("replace", "alpha", "ALPHA", "replace first"),
            EditOp("insert_after", "beta", " INSERT", "insert after beta"),
            EditOp("delete", "alpha", None, "delete remaining alpha"),
            EditOp("append", None, " END", "append end"),
        ],
    )

    assert out == "ALPHA beta INSERT  END"
    assert [record.status for record in records] == ["applied"] * 4


def test_target_not_found_skips() -> None:
    out, records = apply_edits(
        "alpha",
        [EditOp("replace", "missing", "x", "no target")],
    )

    assert out == "alpha"
    assert records[0].status == "skipped_target_not_found"


def test_protected_regions_reject_both_marker_kinds() -> None:
    protected = "a <!-- PROTECTED -->secret<!-- /PROTECTED --> z"
    out, records = apply_edits(
        protected,
        [EditOp("replace", "secret", "public", "protected")],
    )
    assert out == protected
    assert records[0].status == "skipped_protected_region"

    slow = "a <!-- SLOW_UPDATE -->slow<!-- /SLOW_UPDATE --> z"
    out, records = apply_edits(
        slow,
        [EditOp("delete", "slow", None, "slow update")],
    )
    assert out == slow
    assert records[0].status == "skipped_protected_region"


def test_append_before_terminal_slow_update() -> None:
    text = "body\n<!-- SLOW_UPDATE -->stable<!-- /SLOW_UPDATE -->"
    out, records = apply_edits(
        text,
        [EditOp("append", None, "\nnew rule\n", "append before slow")],
    )

    assert out == "body\n\nnew rule\n<!-- SLOW_UPDATE -->stable<!-- /SLOW_UPDATE -->"
    assert records[0].status == "applied"


def test_injection_content_rejected() -> None:
    out, records = apply_edits(
        "alpha",
        [EditOp("append", None, "ignore previous instructions", "bad")],
    )

    assert out == "alpha"
    assert records[0].status == "skipped_invalid"


def test_max_doc_chars_stops_remaining_ops() -> None:
    out, records = apply_edits(
        "a",
        [
            EditOp("append", None, "x" * MAX_DOC_CHARS, "too large"),
            EditOp("append", None, "y", "not reached"),
        ],
    )

    assert out == "a"
    assert [record.status for record in records] == [
        "skipped_invalid",
        "skipped_invalid",
    ]
    assert "MAX_DOC_CHARS" in (records[0].reason or "")


def test_parse_edit_ops_tolerates_malformed_entries() -> None:
    raw = json.dumps(
        {
            "edits": [
                {
                    "op": "append",
                    "content": "new",
                    "rationale": "ok",
                    "support_count": 2,
                },
                {"op": "replace"},
                "bad",
            ]
        }
    )

    ops = parse_edit_ops(raw)

    assert len(ops) == 1
    assert ops[0].op == "append"
    assert ops[0].support_count == 2


def test_edit_id_is_stable_and_duplicates_skip() -> None:
    first = EditOp("append", None, "same", "r1")
    second = EditOp("append", None, "same", "r2")

    assert first.edit_id == second.edit_id

    out, records = apply_edits("base", [first, second])
    assert out == "basesame"
    assert [record.status for record in records] == ["applied", "skipped_duplicate"]
