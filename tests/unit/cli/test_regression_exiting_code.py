"""User code that ends the process cannot make an unevaluated check pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.cli.regression_run_helpers import (
    MODULE_SOURCE,
    invoke,
    make_row,
    read_rows,
    write_rows,
)

EXIT_AT_IMPORT = MODULE_SOURCE.replace("import sys", "import sys\nsys.exit(0)")


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("TRAIGENT_BACKEND_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TRAIGENT_API_URL", "http://127.0.0.1:9")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "agent.py").write_text(MODULE_SOURCE, encoding="utf-8")
    write_rows(tmp_path, [make_row("P", {"q": "pass"})])
    assert invoke(tmp_path, "confirm", "--repeats", "2").exit_code == 0
    return tmp_path


def _swap_module(ws: Path, source: str) -> None:
    (ws / "agent.py").write_text(source, encoding="utf-8")


@pytest.mark.parametrize("code", [0, 1, None, "text"])
def test_a_module_that_exits_at_import_makes_check_inconclusive(
    ws: Path, code: object
) -> None:
    _swap_module(
        ws, MODULE_SOURCE.replace("import sys", f"import sys\nsys.exit({code!r})")
    )
    result = invoke(ws, "check", "--json")
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)  # exactly one object, nothing else
    assert payload["verdict"] == "error" and payload["error"] == "module_load_failed"
    assert payload["exit_code"] == 2


def test_check_text_mode_also_exits_2_not_0_for_an_exiting_module(ws: Path) -> None:
    _swap_module(ws, EXIT_AT_IMPORT)
    result = invoke(ws, "check")
    assert result.exit_code == 2 and "SystemExit" in result.output


def test_a_function_that_exits_mid_evaluation_makes_check_inconclusive(
    ws: Path,
) -> None:
    _swap_module(
        ws,
        MODULE_SOURCE.replace(
            'return "ok" if q.startswith("pass") else "wrong"', "sys.exit(0)"
        ),
    )
    result = invoke(ws, "check", "--json")
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert payload["verdict"] == "error" and payload["error"] == "run_failed"


def test_confirm_with_an_exiting_function_writes_nothing_and_exits_2(ws: Path) -> None:
    before = (ws / "reg" / "regression.jsonl").read_bytes()
    _swap_module(
        ws,
        MODULE_SOURCE.replace(
            'return "ok" if q.startswith("pass") else "wrong"', "sys.exit(0)"
        ),
    )
    result = invoke(ws, "confirm", "--repeats", "2")
    assert result.exit_code == 2, result.output
    assert (ws / "reg" / "regression.jsonl").read_bytes() == before
    _swap_module(ws, EXIT_AT_IMPORT)
    assert invoke(ws, "confirm", "--repeats", "2").exit_code == 2
    assert (ws / "reg" / "regression.jsonl").read_bytes() == before
    assert read_rows(ws)["P"]["metadata"]["regression"]["status"] == "armed"


def test_success_comes_only_from_the_verdict_logic(ws: Path) -> None:
    # The same file with a function that really runs and passes: exit 0 with a
    # verdict object. An exit(0) anywhere else never reaches this.
    result = invoke(ws, "check", "--json")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["verdict"] == "pass"


# A BaseException that is not SystemExit (or KeyboardInterrupt) is not a user
# "exit": it propagates instead of being folded into the closed inconclusive result.
_RAISE_AT_IMPORT = (
    "class _Boom(BaseException):\n    pass\n\nraise _Boom('boom-sentinel')\n"
)
_RAISE_IN_FUNCTION = "class _Boom(BaseException):\n    pass\n\ndef _raise():\n    raise _Boom('boom-sentinel')\n"


def _raising_module(kind: str) -> str:
    if kind == "import":
        return MODULE_SOURCE.replace("import sys", "import sys\n" + _RAISE_AT_IMPORT)
    return MODULE_SOURCE.replace(
        "import sys", "import sys\n" + _RAISE_IN_FUNCTION
    ).replace('return "ok" if q.startswith("pass") else "wrong"', "_raise()")


@pytest.mark.parametrize("command", ["confirm", "check"])
@pytest.mark.parametrize("kind", ["import", "function"])
def test_a_non_systemexit_baseexception_propagates(
    ws: Path, command: str, kind: str
) -> None:
    before = (ws / "reg" / "regression.jsonl").read_bytes()
    _swap_module(ws, _raising_module(kind))
    args = ["--repeats", "2"] if command == "confirm" else ["--json"]
    with pytest.raises(BaseException, match="boom-sentinel") as raised:
        invoke(ws, command, *args)
    assert type(raised.value).__name__ == "_Boom"
    assert (ws / "reg" / "regression.jsonl").read_bytes() == before


def test_systemexit_is_still_closed_while_a_baseexception_is_not(ws: Path) -> None:
    _swap_module(ws, EXIT_AT_IMPORT)
    assert invoke(ws, "check", "--json").exit_code == 2
    _swap_module(ws, _raising_module("import"))
    with pytest.raises(BaseException, match="boom-sentinel"):
        invoke(ws, "check", "--json")


_LEAK_MARKER = "customer-content-marker"


@pytest.mark.parametrize("command", ["confirm", "check"])
@pytest.mark.parametrize("kind", ["import", "function"])
def test_an_ordinary_exception_gives_a_sanitized_closed_exit_2(
    ws: Path, command: str, kind: str
) -> None:
    raising = f"raise ValueError({_LEAK_MARKER!r})"
    if kind == "import":
        source = MODULE_SOURCE.replace("import sys", "import sys\n" + raising)
    else:
        source = MODULE_SOURCE.replace(
            'return "ok" if q.startswith("pass") else "wrong"', raising
        )
    _swap_module(ws, source)
    before = (ws / "reg" / "regression.jsonl").read_bytes()
    args = ["--repeats", "2"] if command == "confirm" else []
    result = invoke(ws, command, *args)
    assert _LEAK_MARKER not in result.output
    assert "Traceback" not in result.output
    if kind == "import":
        assert (ws / "reg" / "regression.jsonl").read_bytes() == before
        # the loader is the closed boundary: exit 2 with the type name only
        assert result.exit_code == 2, result.output
        assert "ValueError" in result.output
    elif command == "check":
        assert result.exit_code == 2, result.output  # inconclusive, never a pass
