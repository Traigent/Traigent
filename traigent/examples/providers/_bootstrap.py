"""Runtime dependency check for the per-provider examples.

Most providers run on the SDK's core install with no extra packages
(LiteLLM is a core dependency, so Groq / Bedrock / Vertex / Azure /
Gemini / the generic LiteLLM example all work out of the box). The
LangChain providers (OpenAI / Anthropic / OpenRouter) need their
LangChain package, which ``pip install "traigent[recommended]"``
already provides — so this helper only ever does anything for a bare
``pip install traigent`` user who picks a LangChain provider.

Safety posture (matches the web onboarding-funnel feature: no silent
installs — PEP 668, externally-managed envs, running as root in
containers, locked venvs): by default we **detect** the missing package
and **print the exact install command**. We install only when the user
explicitly says yes at an interactive prompt, and never in a
non-interactive shell (CI, notebooks, piped stdin) or when
``TRAIGENT_EXAMPLES_NO_INSTALL`` is set.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


def _stdin_is_interactive() -> bool:
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def ensure_packages(requirements: list[tuple[str, str | None]]) -> None:
    """Ensure each required package is importable, or guide the user.

    Args:
        requirements: ``(import_name, pip_package)`` pairs. A ``None``
            ``pip_package`` means the dependency is part of the SDK core
            install and is never auto-installed (skipped).

    Behaviour when something is missing:
        * interactive TTY and not opted out → prompt ``[y/N]``; install
          only on an explicit yes, otherwise print the command and exit;
        * non-interactive or ``TRAIGENT_EXAMPLES_NO_INSTALL`` set → print
          the exact ``pip install`` command and exit cleanly (no prompt,
          no install).
    """
    missing = [
        (mod, pkg)
        for mod, pkg in requirements
        if pkg is not None and importlib.util.find_spec(mod) is None
    ]
    if not missing:
        return

    pip_names = [pkg for _, pkg in missing if pkg]
    quoted = " ".join(f'"{name}"' for name in pip_names)
    install_cmd = f"{sys.executable} -m pip install {quoted}"
    human = ", ".join(pip_names)

    print(
        f"\n[traigent] This example needs the following package(s): {human}.",
        file=sys.stderr,
    )

    if (
        _truthy(os.environ.get("TRAIGENT_EXAMPLES_NO_INSTALL"))
        or not _stdin_is_interactive()
    ):
        print(f"[traigent] Install it with:\n    {install_cmd}\n", file=sys.stderr)
        raise SystemExit(
            "Missing example dependency — install the package above and re-run."
        )

    try:
        reply = (
            input(f"[traigent] Install {human} now with pip? [y/N] ").strip().lower()
        )
    except EOFError:
        reply = ""

    if reply not in {"y", "yes"}:
        print(
            f"[traigent] Skipped. Install it yourself with:\n    {install_cmd}\n",
            file=sys.stderr,
        )
        raise SystemExit(
            "Missing example dependency — install the package above and re-run."
        )

    print(f"[traigent] Installing {human} ...", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pip_names])


__all__ = ["ensure_packages"]
