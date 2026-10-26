"""Pin the exit-code table in ``docs/features/onboarding.md``.

The exit codes asserted below are produced by the real click commands
(``CliRunner`` against ``traigent.cli.main.cli``). The final test compares the
documented table, row by row and word for word, with the expected rows kept
here, so rewording a row or changing a code means updating both on purpose.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from traigent.cli.main import cli

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC = REPO_ROOT / "docs" / "features" / "onboarding.md"


_GOOD_DATASET = (
    '{"input": {"q": "a"}, "output": "b"}\n{"input": {"q": "c"}, "output": "d"}\n'
)
_PLAIN_SOURCE = "def plain(x):\n    return x\n"
_BROKEN_SOURCE = "raise RuntimeError('boom at import time')\n"
_AGENT_SOURCE = (
    "import traigent\n"
    "\n"
    "@traigent.optimize(\n"
    "    eval_dataset={dataset!r},\n"
    "    objectives=['accuracy'],\n"
    "    configuration_space={{'temperature': [0.0, 0.5]}},\n"
    ")\n"
    "def answer(q: str, temperature: float = 0.0) -> str:\n"
    "    return 'b'\n"
)


@pytest.fixture()
def files(
    cli_scratch_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[dict[str, Path]]:
    monkeypatch.setenv("TRAIGENT_MOCK_LLM", "true")
    monkeypatch.setenv("TRAIGENT_OFFLINE_MODE", "true")
    paths = {
        "good_data": cli_scratch_dir / "good.jsonl",
        "bad_data": cli_scratch_dir / "bad.jsonl",
        "plain": cli_scratch_dir / "plain.py",
        "broken": cli_scratch_dir / "broken.py",
        "agent": cli_scratch_dir / "agent.py",
        "missing": cli_scratch_dir / "missing.py",
    }
    paths["good_data"].write_text(_GOOD_DATASET)
    paths["bad_data"].write_text("not json\n")
    paths["plain"].write_text(_PLAIN_SOURCE)
    paths["broken"].write_text(_BROKEN_SOURCE)
    paths["agent"].write_text(_AGENT_SOURCE.format(dataset=str(paths["good_data"])))
    yield paths


def _exit(args: list[str]) -> int:
    result = CliRunner().invoke(cli, args)
    return result.exit_code


# (command, scenario) -> exit code. Keep in step with the table in DOC.
def test_quickstart_exit_codes() -> None:
    assert _exit(["quickstart"]) == 0  # real keyless mock run

    no_winner = SimpleNamespace(best_score=None, best_config=None, trials=[])
    with patch(
        "traigent.examples.quickstart.__main__.answer.optimize",
        new=AsyncMock(return_value=no_winner),
    ):
        assert _exit(["quickstart"]) == 1


def test_validate_exit_codes(files: dict[str, Path]) -> None:
    assert _exit(["validate", str(files["good_data"])]) == 0
    assert _exit(["validate", str(files["bad_data"])]) == 1
    assert _exit(["validate", str(files["missing"])]) == 2


def test_check_exit_codes(files: dict[str, Path]) -> None:
    assert _exit(["check", str(files["agent"]), "--dry-run"]) == 0
    assert _exit(["check", str(files["plain"]), "--allow-empty"]) == 0
    assert _exit(["check", str(files["plain"])]) == 1  # nothing to check
    assert _exit(["check", str(files["missing"])]) == 1  # module not found
    assert _exit(["check", str(files["agent"])]) == 1  # no improvement over default
    assert _exit(["check"]) == 2  # missing argument


def test_optimize_exit_codes(files: dict[str, Path]) -> None:
    assert _exit(["optimize", str(files["agent"]), "-n", "2"]) == 0
    assert _exit(["optimize", str(files["plain"])]) == 1  # no decorated function
    assert _exit(["optimize", str(files["broken"])]) == 1  # file fails to load
    assert _exit(["optimize", str(files["missing"])]) == 2  # path does not exist


# (command, exit code) -> the full meaning sentence the doc must carry.
_EXPECTED_MEANING = {
    ("quickstart", 0): ("Mock demo finished; plumbing check passed"),
    ("quickstart", 1): ("No trial succeeded, so no best configuration was found"),
    ("validate", 0): ("Dataset passed validation"),
    ("validate", 1): ("Dataset path or content failed validation"),
    ("validate", 2): ("Usage error, such as a dataset path that does not exist"),
    ("check", 0): (
        "All checked functions improve on defaults, `--dry-run` completed, or nothing was found with `--allow-empty`"
    ),
    ("check", 1): (
        "No optimizable function found, the module path does not exist (reported as 1, not 2), or a function did not improve on its defaults"
    ),
    ("check", 2): ("Usage error, such as a missing `MODULE_PATH` argument"),
    ("optimize", 0): ("Every optimizable function in the file was optimized"),
    ("optimize", 1): (
        "File failed to load, no decorated function was found, or an optimization failed"
    ),
    ("optimize", 2): ("Usage error, such as a file path that does not exist"),
}


def test_documented_table_matches_pinned_codes() -> None:
    """The doc rows (codes and meanings) and the assertions above must agree."""
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## Exit codes", 1)[1].split("\n## ", 1)[0]
    prose = " ".join(section.split())
    rows = re.findall(
        r"^\| `traigent (\w+)` \| (\d) \| (.*?) \|$", section, re.MULTILINE
    )
    assert all(meaning.strip() for _, _, meaning in rows), rows
    assert "any other failure exits 1; usage errors exit 2" in prose
    assert "pre-1.0 and may change in a minor release" in prose
    documented = {(cmd, int(code)): meaning for cmd, code, meaning in rows}
    assert set(documented) == set(_EXPECTED_MEANING)
    for key, fragment in _EXPECTED_MEANING.items():
        assert documented[key] == fragment, (key, fragment, documented[key])
