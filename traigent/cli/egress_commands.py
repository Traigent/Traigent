"""``traigent egress``: a local relay that logs what the SDK sends to the backend.

    traigent egress relay --upstream https://BACKEND [--port N] [--log PATH] [--exec CMD]
    traigent egress list [--log PATH]
    traigent egress show SEQ [--log PATH]

The relay is the only thing that touches the network; ``list`` and ``show`` read the
log file. The relay module is imported inside the ``relay`` command, so importing the
CLI never loads it. See :mod:`traigent.egress.log` for what the log holds (method,
route shape, body size, body checksum, response status; never values or headers).
"""

# Traceability: CONC-Layer-Core CONC-Quality-Security FUNC-SECURITY

from __future__ import annotations

import json
import os
import shlex
import subprocess  # nosec B404 - runs the user's own --exec command, no shell
import sys
from pathlib import Path
from typing import Any

import click

BACKEND_ENV_VARS = ("TRAIGENT_BACKEND_URL", "TRAIGENT_API_URL")
# Tells the SDK's URL gate that the pinned loopback http origin is the relay; see
# ``traigent.cloud.url_security`` (an explicit port and a loopback IP literal are
# required there, which the relay's own origin always satisfies).
RELAY_ENV_VAR = "TRAIGENT_EGRESS_RELAY"
_DEFAULT_LOG = Path(".traigent") / "egress" / "relay.jsonl"

_log_option = click.option(
    "--log",
    "log_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_LOG,
    show_default=True,
    help="The relay log file.",
)


def _run_child(command: str, origin: str) -> int:
    argv = shlex.split(command)
    if not argv:
        raise click.UsageError("--exec needs a command")
    env = dict(os.environ)
    for name in BACKEND_ENV_VARS:
        env[name] = origin
    env[RELAY_ENV_VAR] = "1"
    try:
        completed = subprocess.run(argv, env=env, check=False)  # nosec B603
    except KeyboardInterrupt:
        return 130
    except OSError:
        click.echo("error: the --exec command could not be started", err=True)
        return 127
    code = completed.returncode
    return 128 - code if code < 0 else code


@click.group()
def egress() -> None:
    """Log what the SDK sends to the Traigent backend (local relay)."""


@egress.command("relay")
@click.option("--upstream", required=True, help="Backend origin, e.g. https://host.")
@click.option("--port", type=click.IntRange(0, 65535), default=0, help="Default: free.")
@_log_option
@click.option(
    "--exec", "command", help="Run this command with the SDK pointed at the relay."
)
def relay_command(
    upstream: str, port: int, log_path: Path, command: str | None
) -> None:
    """Run the relay on 127.0.0.1 and log each request it forwards."""
    from traigent.egress.log import RelayLog, RelayLogError
    from traigent.egress.relay import EgressRelay, RelayConfigError

    try:
        log = RelayLog(log_path)
    except RelayLogError as exc:
        raise click.ClickException(str(exc)) from None
    try:
        relay = EgressRelay(upstream, log, port=port)
    except (RelayConfigError, OSError) as exc:
        log.close()
        raise click.ClickException(str(exc)) from None
    origin = relay.origin
    if command is None:
        click.echo(
            f"egress relay listening on {origin}; log: {log_path}\n"
            f"export TRAIGENT_BACKEND_URL={origin}\n"
            f"export {RELAY_ENV_VAR}=1",
        )
        try:
            relay.serve_forever()
        except KeyboardInterrupt:
            click.echo("egress relay stopped", err=True)
        finally:
            relay.stop()
        return
    relay.start()
    try:
        code = _run_child(command, origin)
    finally:
        relay.stop()
    sys.exit(code)


def _pair(path: Path) -> dict[int, dict[str, dict[str, Any]]]:
    from traigent.egress.log import iter_lines

    entries: dict[int, dict[str, dict[str, Any]]] = {}
    for entry in iter_lines(path):
        entries.setdefault(entry["seq"], {})[entry["kind"]] = entry
    return entries


@egress.command("list")
@_log_option
def list_command(log_path: Path) -> None:
    """List logged requests with their response status, oldest first."""
    entries = _pair(log_path)
    if not entries:
        click.echo("No logged requests.")
        return
    click.echo(
        f"{'SEQ':>5}  {'TIME (UTC)':<20}  {'METHOD':<7}  {'STATUS':>6}  {'BYTES':>9}  ROUTE"
    )
    for seq in sorted(entries):
        request = entries[seq].get("request")
        if request is None:
            continue
        response = entries[seq].get("response")
        status = str(response["status"]) if response else "-"
        click.echo(
            f"{seq:>5}  {request['ts']:<20}  {request['method']:<7}  {status:>6}  "
            f"{request['bytes']:>9}  {request['route']}"
        )


@egress.command("show")
@click.argument("seq", type=int)
@_log_option
def show_command(seq: int, log_path: Path) -> None:
    """Show the request and response lines for one SEQ."""
    found = _pair(log_path).get(seq)
    if not found:
        click.echo(f"error: no logged request with seq {seq}", err=True)
        raise click.exceptions.Exit(1)
    click.echo(
        json.dumps([found[k] for k in ("request", "response") if k in found], indent=2)
    )


def register_egress_commands(cli: click.Group) -> None:
    """Mount ``traigent egress`` on the main CLI group."""
    cli.add_command(egress)
