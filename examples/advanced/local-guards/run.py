"""Block a destructive tool call unless it was approved first, with local guards.

Steps: compile the TVL module with ``tvl-compile-runtime``, load the local guard
set, wrap the tool with ``@guarded``, run three scenarios.  Needs
``pip install "traigent[runtime-tvl]"`` for the compiler.  No API keys, no network.

Run from any directory:  python examples/advanced/local-guards/run.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from traigent.runtime import LocalGuardError, LocalGuardSet, guarded

HERE = Path(__file__).resolve().parent
DELETED: list[str] = []


def compile_guards(out_dir: Path) -> LocalGuardSet:
    """Compile the TVL module and load the local guard set from the result."""
    compiler = shutil.which(
        "tvl-compile-runtime", path=str(Path(sys.executable).parent)
    ) or shutil.which("tvl-compile-runtime")
    if compiler is None:
        raise SystemExit(
            'tvl-compile-runtime not found: pip install "traigent[runtime-tvl]"'
        )
    subprocess.run(
        [
            compiler,
            "--token-map",
            str(HERE / "token-map.json"),
            "--out-dir",
            str(out_dir),
            str(HERE / "delete_needs_approval.runtime.yml"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return LocalGuardSet.from_local(
        json.loads((out_dir / "runtime-local.json").read_text())
    )


@guarded("infra.delete_volume")  # the guard is checked before the body runs
def delete_volume(volume: str) -> str:
    DELETED.append(volume)  # the destructive side effect
    return f"deleted {volume}"


def run_scenario(
    guards: LocalGuardSet, name: str, steps: list[str], cid: str
) -> list[str]:
    """Run steps in one unit; "approve" records an approval, anything else is a volume."""
    DELETED.clear()
    log = [f"[{name}]"]
    sequence = 0
    with guards.unit(correlation_id=cid, ordered_attested=True) as unit:
        for step in steps:
            if step == "approve":
                unit.observe(
                    {
                        "type": "authorization_granted",
                        "ref": "destructive_change",
                        "correlation_id": cid,
                        "producer": "approver",
                        "sequence": sequence,
                    }
                )
                sequence += 1
                log.append("  approval recorded")
                continue
            try:
                log.append(f"  {delete_volume(step)}")
            except LocalGuardError as exc:
                log.append(
                    f"  REFUSED ({exc.args[0] if exc.args else exc}); "
                    f"side effect ran: {step in DELETED}"
                )
    log.append(f"  volumes deleted: {DELETED}")
    return log


def main() -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        guards = compile_guards(Path(tmp))
    lines: list[str] = []
    lines += run_scenario(guards, "1 no approval", ["vol-1"], "c1")
    lines += run_scenario(guards, "2 approve, then delete", ["approve", "vol-1"], "c2")
    lines += run_scenario(
        guards, "3 one approval, two deletes", ["approve", "vol-1", "vol-2"], "c3"
    )
    print("\n".join(lines))
    return lines


if __name__ == "__main__":
    main()
