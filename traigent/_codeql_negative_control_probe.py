"""Deliberate CodeQL negative-control probe — DO NOT MERGE.

Exercises the default Python query pack's command-injection query
(py/command-line-injection, CWE-078) with a Flask-request-controlled value
flowing unsanitized into ``subprocess.run(..., shell=True)``, to prove the
codeql.yml workflow added in this PR can actually go red on a real pull
request. Nothing in this repo imports this module; it is reverted in the
very next commit.
"""

import subprocess

from flask import request


def run_diagnostic_ping() -> None:
    """Never called in production; exists only so CodeQL's data-flow
    analysis sees a request-controlled value reach a shell command."""
    target_host = request.args.get("host")
    subprocess.run("ping -c 1 " + target_host, shell=True)
