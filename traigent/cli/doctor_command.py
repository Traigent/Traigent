"""CLI command: ``traigent doctor`` (alias ``diagnose``).

Wires the SDK's existing ``diagnose()`` aggregator (``traigent/utils/diagnostics.py``,
previously not surfaced by any CLI command) together with the presence/format
checks currently scattered across ``models``, ``validate``, ``validate-config``,
and ``auth whoami`` into a single, free preflight pass: one "is my setup green?"
command with a PASS/WARN/FAIL/SKIP table and a non-zero exit on any FAIL.

Zero paid LLM calls by construction: every check here reads local state,
static tables (LiteLLM's ``model_cost``), or performs at most a lightweight,
non-LLM network call, and even that is skippable with ``--offline``.

Scope note (issue #1778): this command hosts the per-check improvements
tracked separately in #1779 (async-scorer guard), #1780 (upfront
scorer-signature validation), and #1781 (cross-vendor funds/quota
readiness) — it does not reimplement them. Where those checks are not yet
available, the corresponding row is reported as SKIP.
"""

from __future__ import annotations

import inspect
import json as json_module
import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal

import click

from traigent.utils.diagnostics import DiagnosticReport, diagnose
from traigent.utils.env_config import is_strict_cost_accounting

Status = Literal["PASS", "WARN", "FAIL", "SKIP"]

_STATUS_STYLE = {
    "PASS": ("green", "PASS"),
    "WARN": ("yellow", "WARN"),
    "FAIL": ("red", "FAIL"),
    "SKIP": ("dim", "SKIP"),
}

# Same prefixes `traigent auth whoami` accepts (traigent/cli/auth_commands.py).
_TRAIGENT_KEY_PREFIXES = ("tg_", "uk_", "sk_", "ak_", "tk_")

# Vendor API-key environment variables doctor checks for presence (mirrors
# the "KEY" filter TraigentDiagnostics._add_recommendations already uses).
_VENDOR_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "MISTRAL_API_KEY",
    "COHERE_API_KEY",
)


@dataclass
class DoctorCheck:
    """One row of the doctor report."""

    category: str
    status: Status
    message: str


@dataclass
class DoctorReport:
    """Aggregated PASS/WARN/FAIL/SKIP checks plus free-text recommendations."""

    checks: list[DoctorCheck] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def add(self, category: str, status: Status, message: str) -> None:
        self.checks.append(DoctorCheck(category, status, message))

    @property
    def has_fail(self) -> bool:
        return any(c.status == "FAIL" for c in self.checks)

    @property
    def has_warn(self) -> bool:
        return any(c.status == "WARN" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [
                {"category": c.category, "status": c.status, "message": c.message}
                for c in self.checks
            ],
            "recommendations": self.recommendations,
            "summary": {
                "total": len(self.checks),
                "pass": sum(1 for c in self.checks if c.status == "PASS"),
                "warn": sum(1 for c in self.checks if c.status == "WARN"),
                "fail": sum(1 for c in self.checks if c.status == "FAIL"),
                "skip": sum(1 for c in self.checks if c.status == "SKIP"),
            },
        }


def _fold_diagnostic_report(report: DoctorReport, diag: DiagnosticReport) -> None:
    """Fold ``diagnose()``'s report into the doctor table (issue #1778)."""
    for success in diag.successes:
        report.add(success["category"], "PASS", success["message"])
    for warning in diag.warnings:
        report.add(warning["category"], "WARN", warning["message"])
    for issue in diag.issues:
        message = issue["message"]
        if issue.get("fix"):
            message = f"{message} (fix: {issue['fix']})"
        report.add(issue["category"], "FAIL", message)
    report.recommendations.extend(diag.recommendations)


def _run_key_checks(report: DoctorReport) -> None:
    """Traigent key presence + tg_/uk_ format classification, and vendor
    key presence. Local-only: no network call (issue #1778's "vendor-key
    presence + uk_/tg_ classification")."""
    traigent_key = os.environ.get("TRAIGENT_API_KEY")
    if not traigent_key:
        report.add(
            "Auth",
            "WARN",
            "TRAIGENT_API_KEY not set (cloud features unavailable; local "
            "optimization still works)",
        )
    elif any(traigent_key.startswith(prefix) for prefix in _TRAIGENT_KEY_PREFIXES):
        matched = next(p for p in _TRAIGENT_KEY_PREFIXES if traigent_key.startswith(p))
        report.add(
            "Auth",
            "PASS",
            f"TRAIGENT_API_KEY present and recognized ('{matched}' prefix)",
        )
    else:
        report.add(
            "Auth",
            "FAIL",
            "TRAIGENT_API_KEY is set but does not match a known prefix "
            f"({', '.join(_TRAIGENT_KEY_PREFIXES)})",
        )

    vendor_keys_present = [v for v in _VENDOR_KEY_ENV_VARS if os.environ.get(v)]
    if vendor_keys_present:
        report.add(
            "Auth",
            "PASS",
            f"Vendor API key(s) present: {', '.join(vendor_keys_present)}",
        )
    else:
        report.add(
            "Auth",
            "WARN",
            "No vendor API key found ("
            + ", ".join(_VENDOR_KEY_ENV_VARS)
            + "); needed for any provider that isn't mocked",
        )


def _run_model_checks(report: DoctorReport, model_id: str | None) -> None:
    """Model liveness classification + LiteLLM pricing coverage.

    Both checks are static/offline: provider classification is a local
    pattern match and pricing coverage reads LiteLLM's bundled cost table.
    Neither makes a network call or an LLM completion.
    """
    if not model_id:
        report.add("Model", "SKIP", "no --model given; skipping model checks")
        return

    from traigent.providers.validation import get_provider_for_model

    provider = get_provider_for_model(model_id)
    if provider:
        report.add("Model", "PASS", f"'{model_id}' recognized as a {provider} model")
    else:
        report.add(
            "Model",
            "WARN",
            f"'{model_id}' is not in the known provider surface; may 404 at call time",
        )

    try:
        import litellm

        priced = model_id in litellm.model_cost or any(
            model_id in key or key in model_id for key in litellm.model_cost
        )
    except ImportError:
        report.add("Model", "SKIP", "litellm not importable; skipping pricing coverage")
        return

    if priced:
        report.add("Model", "PASS", f"'{model_id}' has LiteLLM pricing coverage")
    else:
        report.add(
            "Model",
            "WARN",
            f"'{model_id}' has no LiteLLM pricing entry; cost objectives will "
            "treat it as unpriced",
        )


def _run_cost_checks(report: DoctorReport) -> None:
    """Cost-cap/approval sanity (issue #1778): report the effective strict
    cost-accounting setting and its origin so a silent default is visible."""
    from traigent.utils.env_config import strict_cost_accounting_origin

    strict = is_strict_cost_accounting(include_run_default=False)
    origin = strict_cost_accounting_origin()
    if origin == "env":
        report.add(
            "Cost",
            "PASS",
            f"TRAIGENT_STRICT_COST_ACCOUNTING explicitly set: strict={strict}",
        )
    else:
        report.add(
            "Cost",
            "WARN" if not strict else "PASS",
            "TRAIGENT_STRICT_COST_ACCOUNTING not set; a run with a cost "
            "objective enables it automatically, otherwise unpriced models "
            "are treated as free",
        )


def _run_dataset_checks(report: DoctorReport, dataset_path: str | None) -> None:
    """Dataset shape check (issue #1778), reusing the same validator
    `traigent validate` already runs."""
    if not dataset_path:
        report.add("Dataset", "SKIP", "no --dataset given; skipping dataset checks")
        return

    from traigent.utils.validation import Validators

    result = Validators.validate_dataset(dataset_path)
    if result.is_valid:
        report.add("Dataset", "PASS", f"'{dataset_path}' passed shape validation")
    else:
        report.add(
            "Dataset",
            "FAIL",
            f"'{dataset_path}' failed shape validation: {result.get_feedback()}",
        )


def _run_scorer_checks(report: DoctorReport, scorer_spec: str | None) -> None:
    """Scorer sanity (issue #1778): import + callable + arity check.

    This is a minimal sanity check, not the richer upfront scorer-signature
    validation tracked in #1780 or the async-scorer guard tracked in #1779 —
    doctor is their host, not their implementation.
    """
    if not scorer_spec:
        report.add(
            "Scorer",
            "SKIP",
            "no --scorer given; skipping scorer checks (see #1779, #1780 for "
            "richer scorer validation)",
        )
        return

    module_name, _, func_name = scorer_spec.partition(":")
    if not func_name:
        report.add(
            "Scorer",
            "FAIL",
            f"'{scorer_spec}' is not in 'module:function' form",
        )
        return

    try:
        module = import_module(module_name)
        scorer = getattr(module, func_name)
    except (ImportError, AttributeError) as exc:
        report.add("Scorer", "FAIL", f"could not import '{scorer_spec}': {exc}")
        return

    if not callable(scorer):
        report.add("Scorer", "FAIL", f"'{scorer_spec}' is not callable")
        return

    from traigent.utils.function_identity import is_coroutine_callable

    param_count = len(inspect.signature(scorer).parameters)
    async_note = " (async)" if is_coroutine_callable(scorer) else ""
    report.add(
        "Scorer",
        "PASS",
        f"'{scorer_spec}' is importable and callable with {param_count} "
        f"parameter(s){async_note}",
    )


def _render_table(report: DoctorReport) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Category")
    table.add_column("Status")
    table.add_column("Detail")

    for check in report.checks:
        color, label = _STATUS_STYLE[check.status]
        table.add_row(check.category, f"[{color}]{label}[/{color}]", check.message)

    console.print("\n[bold blue]Traigent Doctor[/bold blue]\n")
    console.print(table)

    if report.recommendations:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in report.recommendations:
            console.print(f"  • {rec}")

    summary = report.to_dict()["summary"]
    console.print(
        f"\n{summary['pass']} passed, {summary['warn']} warned, "
        f"{summary['fail']} failed, {summary['skip']} skipped\n"
    )


@click.command("doctor")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output the report as machine-readable JSON.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat WARN as FAIL for the exit code.",
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help=(
        "Reserved for future network-touching checks; every current check "
        "is already local-only, so this has no effect yet. traigent doctor "
        "never issues an LLM completion regardless of this flag."
    ),
)
@click.option(
    "--dataset",
    "dataset_path",
    default=None,
    type=click.Path(exists=True),
    help="Dataset file to validate shape for (same check as `traigent validate`).",
)
@click.option(
    "--model",
    "model_id",
    default=None,
    help="Model ID to check provider recognition + LiteLLM pricing coverage for.",
)
@click.option(
    "--scorer",
    "scorer_spec",
    default=None,
    help="'module:function' reference to a scoring/evaluator function to sanity-check.",
)
def doctor(
    output_json: bool,
    strict: bool,
    offline: bool,  # noqa: ARG001 - see --offline help text
    dataset_path: str | None,
    model_id: str | None,
    scorer_spec: str | None,
) -> None:
    """Run a free, zero-LLM-cost preflight over environment, keys, model,
    cost, dataset, and scorer checks.

    Wraps the existing `diagnose()` aggregator plus the presence/format
    checks scattered across `models`, `validate`, `validate-config`, and
    `auth whoami` into one PASS/WARN/FAIL/SKIP table. Exits 1 on any FAIL
    (and on WARN too when --strict is given).
    """
    report = DoctorReport()

    _fold_diagnostic_report(report, diagnose())
    _run_key_checks(report)
    _run_model_checks(report, model_id)
    _run_cost_checks(report)
    _run_dataset_checks(report, dataset_path)
    _run_scorer_checks(report, scorer_spec)

    if output_json:
        click.echo(json_module.dumps(report.to_dict(), indent=2))
    else:
        _render_table(report)

    exit_code = 1 if (report.has_fail or (strict and report.has_warn)) else 0
    raise click.exceptions.Exit(exit_code)


__all__ = ["doctor", "DoctorReport", "DoctorCheck"]
