"""config_generator must read/write source as UTF-8 under any locale (#3260)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_SCRIPT = textwrap.dedent("""
    import locale, sys
    from pathlib import Path

    from traigent.config_generator import generate_config
    from traigent.config_generator.apply import apply_config

    assert locale.getpreferredencoding(False).lower() in ("ascii", "ansi_x3.4-1968"), (
        locale.getpreferredencoding(False)
    )
    path = Path(sys.argv[1])
    result = generate_config(path, function_name="answer")
    apply_config(path, result, "answer", backup=False)
    text = path.read_text(encoding="utf-8")
    assert "na\\u00efve caf\\u00e9 \\u2713" in text, text
    assert "optimize" in text, text
    print("OK")
    """)

_SOURCE = (
    "def answer(question, model='gpt-4o-mini', temperature=0.7):\n"
    '    """A naïve café ✓ agent."""\n'
    "    return question\n"
)


def test_generate_and_apply_roundtrip_non_ascii_under_ascii_locale(
    tmp_path: Path,
) -> None:
    src = tmp_path / "agent.py"
    src.write_text(_SOURCE, encoding="utf-8")
    env = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.getcwd(),
        "TRAIGENT_MOCK_LLM": "true",
        "HOME": str(tmp_path),
    }
    env.pop("TRAIGENT_CONFIG_BASE_DIR", None)
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, str(src)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-1500:]


def test_declared_non_utf8_source_encoding_is_honoured(tmp_path: Path) -> None:
    from traigent.config_generator.apply import read_source_text

    src = tmp_path / "latin.py"
    src.write_bytes(b"# -*- coding: latin-1 -*-\nx = '\xe9'\n")
    text, encoding = read_source_text(src)
    assert encoding == "iso-8859-1"
    assert "x = 'é'" in text
