"""Counting tokens must never download a Hugging Face tokenizer.

Owner ruling 2026-09-25: pricing a Llama-family (or Cohere command-r, or
older non-claude-3 Anthropic) model through ``litellm.token_counter`` used to
lazily fetch an "exact" tokenizer from huggingface.co on first use per model
(``traigent/utils/cost_calculator.py`` -> ``litellm.token_counter`` ->
``litellm._select_tokenizer`` -> ``huggingface_hub``/``tokenizers``). An
egress audit (Traigent PR #2433,
``docs/security/network-and-behaviour-manifest.md``) measured 8 blocked
network attempts and ~24s added latency pricing a Llama workload. The SDK
must fall back to litellm's own tiktoken-based approximate count for those
models instead -- never set ``HF_HUB_OFFLINE`` globally (that would also
block a customer's own, legitimate Hugging Face model downloads elsewhere in
the SDK), and never bypass an explicit opt-in.

The token-counting tests run in a fresh subprocess interpreter with a
PEP 578 audit hook that records AND blocks any DNS/connect attempt to a
non-loopback host, so a real (or absent) network path cannot itself hang or
flake the test, and so litellm's own broad ``except Exception`` around
tokenizer selection cannot hide a network attempt from the assertion --
the attempt is recorded before it is blocked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

_LLAMA_MODEL = "huggingface/meta-llama/Llama-2-7b-hf"
_PROMPT = "hello world this is a test prompt"

_HOOK_PREAMBLE = textwrap.dedent(
    """
    import sys

    _LOOPBACK = {"127.0.0.1", "::1", "localhost"}
    _hosts = []


    def _hook(event, args):
        if event in ("socket.getaddrinfo", "socket.connect"):
            host = args[0]
            if isinstance(host, bytes):
                host = host.decode("utf-8", "replace")
            if host is None or str(host) in _LOOPBACK:
                return
            _hosts.append(str(host))
            # Block it too: a real reachable/unreachable huggingface.co must
            # not decide whether this test is fast and deterministic.
            raise RuntimeError(f"blocked non-loopback network attempt: {event} {host!r}")


    sys.addaudithook(_hook)
    """
)

# Calls litellm.token_counter for a Llama model and reports what the audit
# hook saw. `_FORCE_HF_DOWNLOAD_FOR_TEST=1` simulates the pre-fix state by
# flipping the flag our fix sets back off, immediately after import.
_COUNT_SCRIPT = _HOOK_PREAMBLE + textwrap.dedent(
    """
    import json
    import os

    import traigent.utils.cost_calculator as cc  # noqa: F401,E402
    import litellm  # noqa: E402

    if os.environ.get("_FORCE_HF_DOWNLOAD_FOR_TEST") == "1":
        litellm.disable_hf_tokenizer_download = False

    tokens = litellm.token_counter(model=%(model)r, text=%(prompt)r)
    print(
        json.dumps(
            {
                "tokens": tokens,
                "hosts": sorted(set(_hosts)),
                "disable_hf_tokenizer_download": litellm.disable_hf_tokenizer_download,
            }
        )
    )
    """
) % {"model": _LLAMA_MODEL, "prompt": _PROMPT}

# Only imports and reports the flag -- no token_counter call, so this proves
# the opt-in env var routes through litellm's original download-capable
# tokenizer-selection path without ever attempting a real download.
_FLAG_ONLY_SCRIPT = _HOOK_PREAMBLE + textwrap.dedent(
    """
    import json

    import traigent.utils.cost_calculator as cc  # noqa: F401,E402
    import litellm  # noqa: E402

    print(
        json.dumps(
            {
                "hosts": sorted(set(_hosts)),
                "disable_hf_tokenizer_download": litellm.disable_hf_tokenizer_download,
            }
        )
    )
    """
)


def _run(script: str, extra_env: dict[str, str] | None = None) -> dict:
    env = os.environ.copy()
    env.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"guard subprocess failed (rc={result.returncode})\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_default_llama_token_count_reaches_no_external_host() -> None:
    """The fix: default settings never contact huggingface.co, and counting
    still produces a positive (approximate) token count."""
    payload = _run(_COUNT_SCRIPT)

    assert payload["disable_hf_tokenizer_download"] is True
    assert payload["hosts"] == [], (
        "token counting for a Llama model reached a non-loopback host "
        f"unexpectedly: {payload['hosts']}"
    )
    assert isinstance(payload["tokens"], int)
    assert payload["tokens"] > 0


def test_opt_in_env_preserves_the_download_capable_path() -> None:
    """TRAIGENT_HF_TOKENIZER_DOWNLOAD=1 keeps today's (pre-fix) behaviour.

    Asserted by checking which tokenizer-selection path litellm would take
    (the flag our fix sets, left untouched by the opt-in) -- no
    ``token_counter`` call here, so this needs no network either.
    """
    payload = _run(_FLAG_ONLY_SCRIPT, extra_env={"TRAIGENT_HF_TOKENIZER_DOWNLOAD": "1"})

    assert payload["disable_hf_tokenizer_download"] is not True, (
        "the opt-in env var no longer preserves litellm's original "
        "download-capable tokenizer selection"
    )
    assert payload["hosts"] == []


def test_sensitivity_control_pre_fix_state_would_have_reached_huggingface() -> None:
    """Without the fix, this SAME assertion set would have failed.

    A regression test that cannot fail is not evidence. This forces the
    pre-fix tokenizer-selection path (litellm.disable_hf_tokenizer_download
    left falsy, as it was before this change) in the same subprocess and
    confirms the audit hook actually observes an attempt to reach
    huggingface.co -- the exact network-visibility defect the fix closes --
    and that counting still recovers a positive count via litellm's own
    internal fallback either way.
    """
    payload = _run(_COUNT_SCRIPT, extra_env={"_FORCE_HF_DOWNLOAD_FOR_TEST": "1"})

    assert "huggingface.co" in payload["hosts"], (
        "expected the pre-fix tokenizer-selection path to reach "
        f"huggingface.co; observed hosts: {payload['hosts']}"
    )
    assert isinstance(payload["tokens"], int)
    assert payload["tokens"] > 0
