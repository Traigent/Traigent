"""Coding-agent enrichment for Traigent Optimizer proposals.

Adapters call user-installed coding-agent CLIs as subprocesses. Traigent does
not manage Anthropic/OpenAI/GitHub credentials and never calls provider SDKs
directly for this feature.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

from jsonschema import Draft7Validator

AGENT_MODES: Final[tuple[str, ...]] = (
    "static",
    "auto",
    "claude-code",
    "codex",
    "github-models",
    "command",
)
PROMPT_ID: Final[str] = "optimizer.decorate.enrichment.v1"
EMPTY_SHA256: Final[str] = hashlib.sha256(b"").hexdigest()
MAX_AGENT_RESPONSE_BYTES: Final[int] = 500_000
EVIDENCE_CATEGORIES: Final[frozenset[str]] = frozenset(
    {
        "literal_assignment",
        "framework_call_kwarg",
        "framework_constructor_arg",
        "comparison_threshold",
        "loop_bound",
        "config_dict_value",
        "string_template",
        "structural_pattern",
        "import",
        "other",
    }
)


class AgentRunner(Protocol):
    def __call__(
        self,
        args: list[str],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        stdin: int | None,
        text: bool,
        capture_output: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """subprocess.run-compatible callable."""


@dataclass(frozen=True, slots=True)
class AgentRunConfig:
    """Execution configuration for one optional enrichment attempt."""

    mode: str = "static"
    project_root: Path | None = None
    timeout_seconds: int = 120
    total_timeout_seconds: int | None = None
    budget_tokens: int = 8_000
    enrich_top_n: int = 3
    command: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class AgentRawResult:
    """Raw adapter result before schema/policy validation."""

    provider: str
    agent_version: str
    response_text: str
    warnings: tuple[str, ...] = ()


class AgentAdapter:
    """Base class for one-shot read-only coding-agent subprocess adapters."""

    provider: str = ""

    def __init__(self, runner: AgentRunner | None = None) -> None:
        self._runner = runner or subprocess.run

    def available(self, config: AgentRunConfig) -> bool:
        raise NotImplementedError

    def run(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        config: AgentRunConfig,
    ) -> AgentRawResult:
        raise NotImplementedError

    def _run_command(
        self,
        args: list[str],
        *,
        config: AgentRunConfig,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        try:
            return self._runner(
                args,
                cwd=str(config.project_root) if config.project_root else None,
                env=env,
                stdin=subprocess.DEVNULL,
                text=True,
                capture_output=True,
                timeout=config.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            return subprocess.CompletedProcess(args, 127, stdout="", stderr=str(exc))
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                args,
                124,
                stdout=_coerce_text(exc.stdout),
                stderr=_coerce_text(exc.stderr)
                or f"agent command timed out after {config.timeout_seconds}s",
            )


class ClaudeCodeAdapter(AgentAdapter):
    provider = "claude-code"

    def available(self, config: AgentRunConfig) -> bool:
        return shutil.which("claude") is not None

    def run(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        config: AgentRunConfig,
    ) -> AgentRawResult:
        del schema
        args = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--permission-mode",
            "dontAsk",
            "--effort",
            "medium",
            "--allowedTools=Read,Grep,Glob",
        ]
        if config.project_root is not None:
            args.append(f"--add-dir={config.project_root}")
        if config.model:
            args.extend(["--model", config.model])
        args.append(prompt)
        completed = self._run_command(args, config=config)
        if completed.returncode != 0:
            return AgentRawResult(
                provider=self.provider,
                agent_version=_safe_version(["claude", "--version"], self._runner),
                response_text="",
                warnings=(completed.stderr.strip() or "claude exited non-zero",),
            )
        return AgentRawResult(
            provider=self.provider,
            agent_version=_safe_version(["claude", "--version"], self._runner),
            response_text=completed.stdout,
        )


class CodexAdapter(AgentAdapter):
    provider = "codex"

    def available(self, config: AgentRunConfig) -> bool:
        return shutil.which("codex") is not None

    def run(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        config: AgentRunConfig,
    ) -> AgentRawResult:
        with tempfile.TemporaryDirectory(prefix="traigent-codex-agent-") as temp_dir:
            temp_path = Path(temp_dir)
            schema_path = temp_path / "agent_recommendation_schema.json"
            output_path = temp_path / "last_message.json"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            args = [
                "codex",
                "exec",
                "-C",
                str(config.project_root or Path.cwd()),
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
            ]
            if config.model:
                args.extend(["--model", config.model])
            args.append(prompt)
            completed = self._run_command(args, config=config)
            response_text = (
                output_path.read_text(encoding="utf-8")
                if output_path.exists()
                else completed.stdout
            )
        if completed.returncode != 0:
            return AgentRawResult(
                provider=self.provider,
                agent_version=_safe_version(["codex", "--version"], self._runner),
                response_text=response_text,
                warnings=(completed.stderr.strip() or "codex exited non-zero",),
            )
        return AgentRawResult(
            provider=self.provider,
            agent_version=_safe_version(["codex", "--version"], self._runner),
            response_text=response_text,
        )


class GitHubModelsAdapter(AgentAdapter):
    provider = "github-models"

    def available(self, config: AgentRunConfig) -> bool:
        return shutil.which("gh") is not None

    def run(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        config: AgentRunConfig,
    ) -> AgentRawResult:
        # `gh copilot` is only a command-suggestion/explanation tool; GitHub
        # Models is the non-interactive GitHub CLI inference path.
        model = config.model or "openai/gpt-4o-mini"
        system_prompt = (
            "Return only JSON matching the schema implied by the user prompt. "
            "Do not include Markdown."
        )
        args = [
            "gh",
            "models",
            "run",
            model,
            prompt,
            "--system-prompt",
            system_prompt,
            "--temperature",
            "0",
            "--max-tokens",
            "4096",
        ]
        completed = self._run_command(args, config=config)
        version = _github_models_version(self._runner)
        if completed.returncode != 0:
            return AgentRawResult(
                provider=self.provider,
                agent_version=version,
                response_text="",
                warnings=(
                    _trim_cli_error(completed.stderr) or "gh models exited non-zero",
                ),
            )
        return AgentRawResult(
            provider=self.provider,
            agent_version=version,
            response_text=completed.stdout,
            warnings=(
                "github-models adapter uses gh models; gh copilot has no "
                "structured JSON output mode",
            ),
        )


class CommandAdapter(AgentAdapter):
    provider = "command"

    def available(self, config: AgentRunConfig) -> bool:
        return bool(config.command)

    def run(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        config: AgentRunConfig,
    ) -> AgentRawResult:
        if not config.command:
            return AgentRawResult(
                provider=self.provider,
                agent_version="",
                response_text="",
                warnings=("missing command for command adapter",),
            )
        args = shlex.split(config.command)
        completed = self._run_command(
            args,
            config=config,
            extra_env={
                "TRAIGENT_OPTIMIZER_AGENT_PROMPT": prompt,
                "TRAIGENT_OPTIMIZER_AGENT_SCHEMA_JSON": json.dumps(schema),
                "TRAIGENT_OPTIMIZER_PROJECT_ROOT": str(config.project_root or ""),
            },
        )
        if completed.returncode != 0:
            return AgentRawResult(
                provider=self.provider,
                agent_version="",
                response_text=completed.stdout,
                warnings=(
                    completed.stderr.strip() or "command adapter exited non-zero",
                ),
            )
        return AgentRawResult(
            provider=self.provider,
            agent_version="custom",
            response_text=completed.stdout,
        )


def enrich_decorate_plan(
    plan: dict[str, Any],
    *,
    source_path: str | Path,
    function_source: str,
    config: AgentRunConfig,
    runner: AgentRunner | None = None,
) -> dict[str, Any]:
    """Return a decorate plan enriched by the requested coding agent."""

    if normalize_agent_mode(config.mode) == "static":
        plan["agent_enrichment"] = None
        return plan

    enriched = copy.deepcopy(plan)
    schema = load_agent_recommendation_schema()
    prompt, context_summary = build_agent_prompt(
        static_payload=enriched,
        function_source=function_source,
        source_path=Path(source_path),
        config=config,
    )
    adapter = select_adapter(config, runner=runner)
    if adapter is None:
        warnings = ["No requested optimizer coding-agent adapter is available."]
        enriched["agent_enrichment"] = _provenance(
            requested_mode=config.mode,
            status="skipped",
            provider="",
            agent_version="",
            prompt=prompt,
            response_text="",
            validation_status="not_validated",
            context_summary=context_summary,
            warnings=warnings,
        )
        return enriched

    schema_for_agent = schema if adapter.provider == "command" else _agent_cli_schema()
    raw_uncapped = adapter.run(prompt=prompt, schema=schema_for_agent, config=config)
    response_sha = _sha256(raw_uncapped.response_text)
    raw = _cap_response(raw_uncapped)
    payload, validation_errors = parse_and_validate_agent_response(
        raw.response_text,
        schema,
    )
    warnings = [*raw.warnings, *validation_errors]
    if payload is None:
        enriched["agent_enrichment"] = _provenance(
            requested_mode=config.mode,
            status="rejected",
            provider=raw.provider,
            agent_version=raw.agent_version,
            prompt=prompt,
            response_text=raw.response_text,
            validation_status="invalid",
            context_summary=context_summary,
            warnings=warnings,
        )
        return enriched

    payload_warnings = list(payload.get("warnings", []))
    warnings.extend(payload_warnings)
    applied_before = _agent_contribution_count(enriched)
    merge_warnings = merge_agent_recommendations(
        enriched,
        payload,
        context_text=prompt,
        include_payload_warnings=False,
    )
    applied_count = max(_agent_contribution_count(enriched) - applied_before, 0)
    warnings.extend(merge_warnings)
    enriched["agent_enrichment"] = _provenance(
        requested_mode=config.mode,
        status="completed",
        provider=raw.provider,
        agent_version=raw.agent_version,
        prompt=prompt,
        response_text=raw.response_text,
        validation_status=_merge_validation_status(applied_count, merge_warnings),
        context_summary=context_summary,
        warnings=warnings,
        response_sha=response_sha,
    )
    return enriched


def enrich_scan_report(
    report: dict[str, Any],
    *,
    config: AgentRunConfig,
    runner: AgentRunner | None = None,
) -> dict[str, Any]:
    """Optionally enrich the top scan candidates with coding-agent advice."""

    if normalize_agent_mode(config.mode) == "static":
        report["agent_enrichment"] = None
        return report

    enriched = copy.deepcopy(report)
    scan_root = Path(enriched["scan_root"])
    deadline = (
        time.monotonic() + config.total_timeout_seconds
        if config.total_timeout_seconds is not None
        else None
    )
    provenances: list[dict[str, Any]] = []
    for candidate in enriched.get("candidates", [])[: max(config.enrich_top_n, 0)]:
        if deadline is not None and time.monotonic() >= deadline:
            provenances.append(
                _provenance(
                    requested_mode=config.mode,
                    status="skipped",
                    provider="",
                    agent_version="",
                    prompt="",
                    response_text="",
                    validation_status="not_validated",
                    context_summary=(
                        f"scan enrichment total timeout reached after "
                        f"{config.total_timeout_seconds}s"
                    ),
                    warnings=["Scan agent enrichment stopped by total timeout."],
                )
            )
            break
        remaining_config = _config_with_remaining_timeout(config, deadline)
        source_path = scan_root / candidate["function"]["file"]
        function_source = _read_function_source(source_path, candidate["function"])
        pseudo_plan = _plan_from_candidate(candidate)
        enriched_plan = enrich_decorate_plan(
            pseudo_plan,
            source_path=source_path,
            function_source=function_source,
            config=remaining_config,
            runner=runner,
        )
        if enriched_plan.get("agent_enrichment") is not None:
            provenances.append(enriched_plan["agent_enrichment"])
        candidate["tvar_signals"] = [
            {
                "tvar": binding["tvar"],
                "confidence": binding["confidence"],
                "domain_source": binding["domain_source"],
                "evidence": binding["evidence"],
            }
            for binding in enriched_plan["proposed_tvar_bindings"]
        ]
        candidate["objective_candidates"] = enriched_plan["objective_candidates"]
    enriched["agent_enrichment"] = provenances or None
    return enriched


def merge_agent_recommendations(
    plan: dict[str, Any],
    payload: dict[str, Any],
    *,
    context_text: str = "",
    include_payload_warnings: bool = True,
) -> list[str]:
    """Merge valid recommendation payload into *plan* and return warnings."""

    warnings = list(payload.get("warnings", [])) if include_payload_warnings else []
    if payload.get("context_confidence") == "low":
        return [*warnings, "Agent enrichment skipped: context_confidence is low."]

    binding_by_name = {
        binding["tvar"]["name"]: binding
        for binding in plan.get("proposed_tvar_bindings", [])
    }
    for recommendation in payload.get("tvar_recommendations", []):
        warning = _merge_tvar_recommendation(
            plan,
            binding_by_name,
            recommendation,
            context_text=context_text,
        )
        if warning:
            warnings.append(warning)

    objective_by_name = {
        objective["name"]: index
        for index, objective in enumerate(plan.get("objective_candidates", []))
    }
    for recommendation in payload.get("objective_recommendations", []):
        objective = {
            "name": recommendation["name"],
            "direction": recommendation["direction"],
            "confidence": recommendation["confidence"],
            "source": "agent",
            "rationale": recommendation["rationale"],
            "required_dataset_fields": recommendation["required_dataset_fields"],
            "auto_measurable": recommendation["auto_measurable"],
            "requires_confirmation": recommendation["requires_confirmation"],
        }
        index = objective_by_name.get(objective["name"])
        if index is None:
            plan["objective_candidates"].append(objective)
            objective_by_name[objective["name"]] = len(plan["objective_candidates"]) - 1
        else:
            plan["objective_candidates"][index] = objective
    return warnings


def _merge_tvar_recommendation(
    plan: dict[str, Any],
    binding_by_name: dict[str, dict[str, Any]],
    recommendation: dict[str, Any],
    *,
    context_text: str,
) -> str | None:
    tvar = copy.deepcopy(recommendation["tvar"])
    name = tvar["name"]
    current_value = recommendation["current_value"]
    domain = tvar.get("domain", {})
    if recommendation["domain_intent"] != "search_space":
        return f"Skipped agent TVAR {name}: domain_intent is current_only."
    if not _has_search_space_domain(tvar, domain):
        return f"Skipped agent TVAR {name}: empty or degenerate search domain."
    if not _value_in_domain(current_value, domain):
        return f"Skipped agent TVAR {name}: current value is outside proposed domain."
    values = domain.get("values")
    if (
        isinstance(values, list)
        and len({_stable_value_key(value) for value in values}) <= 1
    ):
        return f"Skipped agent TVAR {name}: singleton enum is not a search space."
    if _is_model_tvar(name) and not _model_values_are_allowed(
        values,
        binding_by_name,
        current_value,
        context_text,
    ):
        return (
            f"Skipped agent TVAR {name}: proposed model values were not seen "
            "in the static plan or project context."
        )

    tvar.setdefault("default", current_value)
    existing = binding_by_name.get(name)
    if existing is not None:
        existing.update(
            {
                "tvar": tvar,
                "confidence": recommendation["confidence"],
                "domain_source": "agent",
                "evidence": recommendation["evidence"],
                "current_value": current_value,
            }
        )
        if "locator" in recommendation:
            existing["locator"] = recommendation["locator"]
        return None

    locator = recommendation.get("locator")
    if locator is None:
        return f"Skipped new agent TVAR {name}: missing locator."
    binding = {
        "tvar": tvar,
        "confidence": recommendation["confidence"],
        "domain_source": "agent",
        "evidence": recommendation["evidence"],
        "injection_mode": plan["injection_mode"],
        "current_value": current_value,
        "locator": locator,
    }
    plan["proposed_tvar_bindings"].append(binding)
    binding_by_name[name] = binding
    return None


def _is_model_tvar(name: str) -> bool:
    return name == "model" or name.endswith("_model") or name.endswith("model_name")


def _has_search_space_domain(tvar: dict[str, Any], domain: dict[str, Any]) -> bool:
    values = domain.get("values")
    if isinstance(values, list):
        return len(values) > 0

    range_spec = domain.get("range")
    if isinstance(range_spec, list) and len(range_spec) == 2:
        low, high = range_spec
        return (
            isinstance(low, (int, float))
            and not isinstance(low, bool)
            and isinstance(high, (int, float))
            and not isinstance(high, bool)
            and low < high
        )

    return tvar.get("type") == "bool"


def _stable_value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _model_values_are_allowed(
    values: Any,
    binding_by_name: dict[str, dict[str, Any]],
    current_value: Any,
    context_text: str,
) -> bool:
    if not isinstance(values, list) or not values:
        return False
    allowed: set[Any] = {current_value}
    for binding_name, binding in binding_by_name.items():
        if not _is_model_tvar(binding_name):
            continue
        tvar = binding.get("tvar", {})
        allowed.add(tvar.get("default"))
        allowed.add(binding.get("current_value"))
        domain_values = tvar.get("domain", {}).get("values")
        if isinstance(domain_values, list):
            allowed.update(domain_values)
    allowed.discard(None)
    for value in values:
        if value in allowed:
            continue
        if isinstance(value, str) and value in context_text:
            continue
        return False
    return True


def _cap_response(raw: AgentRawResult) -> AgentRawResult:
    encoded = raw.response_text.encode("utf-8")
    if len(encoded) <= MAX_AGENT_RESPONSE_BYTES:
        return raw
    capped = encoded[:MAX_AGENT_RESPONSE_BYTES].decode("utf-8", errors="ignore")
    return AgentRawResult(
        provider=raw.provider,
        agent_version=raw.agent_version,
        response_text=capped,
        warnings=(
            *raw.warnings,
            f"Agent response exceeded {MAX_AGENT_RESPONSE_BYTES} bytes and was truncated.",
        ),
    )


def parse_and_validate_agent_response(
    response_text: str,
    schema: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse lenient agent JSON, normalize common drift, then strict-validate it."""

    schema = schema or load_agent_recommendation_schema()
    payload = _extract_json_payload(response_text)
    if payload is None:
        return None, ["Agent response did not contain a JSON object."]
    cleaned_payload = _drop_nulls(payload)
    if not isinstance(cleaned_payload, dict):
        return None, ["Agent response did not contain a JSON object."]
    payload = _normalize_agent_payload(cleaned_payload)
    validator = Draft7Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        return None, [_format_validation_error(error) for error in errors]
    return payload, []


def build_agent_prompt(
    *,
    static_payload: dict[str, Any],
    function_source: str,
    source_path: Path,
    config: AgentRunConfig,
) -> tuple[str, str]:
    project_context = _collect_project_context(
        config.project_root or source_path.parent
    )
    rendered = _load_prompt_template().format(
        static_payload=json.dumps(static_payload, indent=2),
        function_source=function_source,
        project_context=project_context,
    )
    budget_chars = max(config.budget_tokens, 1) * 4
    if len(rendered) > budget_chars:
        rendered = rendered[:budget_chars] + "\n[truncated by Traigent budget]\n"
    summary = (
        f"prompt_chars={len(rendered)}; budget_tokens={config.budget_tokens}; "
        f"project_root={config.project_root or source_path.parent}"
    )
    return rendered, summary


def load_agent_recommendation_schema() -> dict[str, Any]:
    schema_path = (
        Path(__file__).with_name("schemas") / "agent_recommendation_schema.json"
    )
    return cast(dict[str, Any], json.loads(schema_path.read_text(encoding="utf-8")))


def _agent_cli_schema() -> dict[str, Any]:
    scalar: dict[str, Any] = {
        "anyOf": [
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "null"},
        ]
    }
    evidence = {
        "type": "object",
        "required": ["file", "line", "end_line", "snippet", "category", "detail"],
        "properties": {
            "file": {"type": "string"},
            "line": {"type": "integer"},
            "end_line": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "snippet": {"type": "string"},
            "category": {
                "type": "string",
                "enum": [
                    "literal_assignment",
                    "framework_call_kwarg",
                    "framework_constructor_arg",
                    "comparison_threshold",
                    "loop_bound",
                    "config_dict_value",
                    "string_template",
                    "structural_pattern",
                    "import",
                    "other",
                ],
            },
            "detail": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "additionalProperties": False,
    }
    locator = {
        "type": "object",
        "required": ["kind", "details"],
        "properties": {
            "kind": {"type": "string", "enum": ["ast_path", "regex", "line_col"]},
            "details": {
                "type": "object",
                "required": ["function", "line", "column", "tvar", "path", "pattern"],
                "properties": {
                    "function": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "line": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "column": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "tvar": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "path": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "pattern": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }
    domain = {
        "type": "object",
        "required": ["values", "range", "resolution"],
        "properties": {
            "values": {
                "anyOf": [
                    {"type": "array", "items": scalar},
                    {"type": "null"},
                ]
            },
            "range": {
                "anyOf": [
                    {"type": "array", "items": {"type": "number"}},
                    {"type": "null"},
                ]
            },
            "resolution": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        },
        "additionalProperties": False,
    }
    tvar = {
        "type": "object",
        "required": [
            "name",
            "type",
            "domain",
            "default",
            "scale",
            "agent",
            "is_tool",
            "constraints",
        ],
        "properties": {
            "name": {"type": "string"},
            "type": {"type": "string", "enum": ["bool", "int", "float", "str", "enum"]},
            "domain": {"anyOf": [domain, {"type": "null"}]},
            "default": scalar,
            "scale": {
                "anyOf": [
                    {"type": "string", "enum": ["linear", "log"]},
                    {"type": "null"},
                ]
            },
            "agent": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "is_tool": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
            "constraints": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ]
            },
        },
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "required": [
            "response_version",
            "context_confidence",
            "tvar_recommendations",
            "objective_recommendations",
            "warnings",
        ],
        "properties": {
            "response_version": {"type": "string"},
            "context_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "tvar_recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "tvar",
                        "confidence",
                        "domain_intent",
                        "current_value",
                        "evidence",
                        "locator",
                        "rationale",
                    ],
                    "properties": {
                        "tvar": tvar,
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "domain_intent": {
                            "type": "string",
                            "enum": ["search_space", "current_only"],
                        },
                        "current_value": scalar,
                        "evidence": evidence,
                        "locator": {"anyOf": [locator, {"type": "null"}]},
                        "rationale": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "objective_recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": [
                        "name",
                        "direction",
                        "confidence",
                        "rationale",
                        "required_dataset_fields",
                        "auto_measurable",
                        "requires_confirmation",
                        "evidence",
                    ],
                    "properties": {
                        "name": {"type": "string"},
                        "direction": {
                            "type": "string",
                            "enum": ["maximize", "minimize", "band"],
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                        },
                        "rationale": {"type": "string"},
                        "required_dataset_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "auto_measurable": {"type": "boolean"},
                        "requires_confirmation": {"type": "boolean"},
                        "evidence": {"anyOf": [evidence, {"type": "null"}]},
                    },
                    "additionalProperties": False,
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def select_adapter(
    config: AgentRunConfig,
    *,
    runner: AgentRunner | None = None,
) -> AgentAdapter | None:
    mode = normalize_agent_mode(config.mode)
    adapters: dict[str, AgentAdapter] = {
        "claude-code": ClaudeCodeAdapter(runner),
        "codex": CodexAdapter(runner),
        "github-models": GitHubModelsAdapter(runner),
        "command": CommandAdapter(runner),
    }
    if mode == "auto":
        for candidate in ("claude-code", "codex", "github-models"):
            adapter = adapters[candidate]
            if adapter.available(config):
                return adapter
        return None
    if mode == "static":
        return None
    adapter = adapters[mode]
    return adapter if adapter.available(config) else None


def normalize_agent_mode(mode: str) -> str:
    if mode not in AGENT_MODES:
        raise ValueError(
            f"Unknown optimizer agent mode {mode!r}. "
            f"Expected one of: {', '.join(AGENT_MODES)}."
        )
    return mode


def require_static_agent_mode(agent_mode: str) -> None:
    """Compatibility shim retained for callers that only support static mode."""

    if normalize_agent_mode(agent_mode) != "static":
        raise ValueError(
            "optimizer agent enrichment is not implemented for this call path; "
            "use --agent static or call an enrichment-enabled optimizer API."
        )


def _value_in_domain(value: Any, domain: dict[str, Any]) -> bool:
    values = domain.get("values")
    if isinstance(values, list):
        return value in values
    range_spec = domain.get("range")
    if isinstance(range_spec, list) and len(range_spec) == 2:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        low, high = range_spec
        if not isinstance(low, (int, float)) or isinstance(low, bool):
            return False
        if not isinstance(high, (int, float)) or isinstance(high, bool):
            return False
        return low <= value <= high
    return True


def _config_with_remaining_timeout(
    config: AgentRunConfig,
    deadline: float | None,
) -> AgentRunConfig:
    if deadline is None:
        return config
    remaining = max(int(deadline - time.monotonic()), 1)
    return AgentRunConfig(
        mode=config.mode,
        project_root=config.project_root,
        timeout_seconds=min(config.timeout_seconds, remaining),
        total_timeout_seconds=config.total_timeout_seconds,
        budget_tokens=config.budget_tokens,
        enrich_top_n=config.enrich_top_n,
        command=config.command,
        model=config.model,
    )


def _agent_contribution_count(plan: dict[str, Any]) -> int:
    return sum(
        1
        for binding in plan.get("proposed_tvar_bindings", [])
        if binding.get("domain_source") == "agent"
    ) + sum(
        1
        for objective in plan.get("objective_candidates", [])
        if objective.get("source") == "agent"
    )


def _merge_validation_status(applied_count: int, merge_warnings: list[str]) -> str:
    if not merge_warnings:
        return "valid"
    if applied_count > 0:
        return "partial"
    return "rejected_by_policy"


def _provenance(
    *,
    requested_mode: str,
    status: str,
    provider: str,
    agent_version: str,
    prompt: str,
    response_text: str,
    validation_status: str,
    context_summary: str,
    warnings: list[str],
    response_sha: str | None = None,
) -> dict[str, Any]:
    return {
        "requested_mode": requested_mode,
        "status": status,
        "provider": provider,
        "agent_version": agent_version,
        "prompt_id": PROMPT_ID,
        "prompt_sha256": _sha256(prompt),
        "response_sha256": response_sha or _sha256(response_text),
        "validation_status": validation_status,
        "context_summary": context_summary,
        "warnings": warnings,
    }


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict) and isinstance(parsed.get("result"), str):
        return _extract_json_payload(parsed["result"])
    if isinstance(parsed, dict) and isinstance(parsed.get("structured_output"), dict):
        return cast(dict[str, Any], parsed["structured_output"])
    return cast(dict[str, Any], parsed) if isinstance(parsed, dict) else None


def _drop_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_nulls(item) for key, item in value.items() if item is not None
        }
    if isinstance(value, list):
        return [_drop_nulls(item) for item in value]
    return value


def _normalize_agent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply safe compatibility fixes before strict schema validation."""

    normalized = copy.deepcopy(payload)
    for key in ("tvar_recommendations", "objective_recommendations"):
        recommendations = normalized.get(key)
        if not isinstance(recommendations, list):
            continue
        for recommendation in recommendations:
            if not isinstance(recommendation, dict):
                continue
            evidence = recommendation.get("evidence")
            if isinstance(evidence, dict):
                _normalize_evidence_category(evidence)
            tvar = recommendation.get("tvar")
            if isinstance(tvar, dict):
                _normalize_tvar_shape(tvar)
                domain = tvar.get("domain")
                if isinstance(domain, dict):
                    _normalize_tvar_domain(domain)
    return normalized


def _normalize_evidence_category(evidence: dict[str, Any]) -> None:
    category = evidence.get("category")
    if not isinstance(category, str) or category in EVIDENCE_CATEGORIES:
        return
    evidence.setdefault("detail", category)
    evidence["category"] = "other"


def _normalize_tvar_shape(tvar: dict[str, Any]) -> None:
    if "domain" not in tvar and ("resolution" in tvar or "step" in tvar):
        tvar["domain"] = {}
    domain = tvar.get("domain")
    if not isinstance(domain, dict):
        return
    for key in ("resolution", "step"):
        if key in tvar and key not in domain:
            domain[key] = tvar.pop(key)


def _normalize_tvar_domain(domain: dict[str, Any]) -> None:
    if "resolution" not in domain and "step" in domain:
        domain["resolution"] = domain.pop("step")


def _github_models_version(runner: AgentRunner) -> str:
    output = _safe_version(["gh", "extension", "list"], runner)
    for line in output.splitlines():
        if "github/gh-models" in line or line.startswith("gh models"):
            return line.strip()
    return output.strip()


def _trim_cli_error(stderr: str) -> str:
    return stderr.split("\n\nUsage:", 1)[0].strip()


def _format_validation_error(error: Any) -> str:
    path = ".".join(str(part) for part in error.path)
    return f"{path or '<root>'}: {error.message}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _safe_version(args: list[str], runner: AgentRunner) -> str:
    try:
        completed = runner(
            args,
            cwd=None,
            env=None,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return ""
    return (completed.stdout or completed.stderr).strip()


def _load_prompt_template() -> str:
    return (
        Path(__file__)
        .with_name("agent_prompts")
        .joinpath("decorate.md")
        .read_text(encoding="utf-8")
    )


def _collect_project_context(project_root: Path) -> str:
    snippets: list[str] = []
    for name in (
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        "pyproject.toml",
        "package.json",
    ):
        path = project_root / name
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        snippets.append(f"--- {name} ---\n{content[:6000]}")
    return "\n\n".join(snippets) if snippets else "[no project context files found]"


def _read_function_source(source_path: Path, function_info: dict[str, Any]) -> str:
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    start = max(int(function_info.get("line", 1)) - 1, 0)
    end = int(function_info.get("end_line", start + 1))
    return "\n".join(lines[start:end])


def _plan_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    function = candidate["function"]
    return {
        "plan_version": "0.1.0",
        "runtime": "python",
        "tool_version": "",
        "generated_at": "",
        "target": {
            "file": function["file"],
            "function": function.get("qualified_name", function["name"]),
            "line": function["line"],
            "candidate_id": candidate["fingerprint"]["candidate_id"],
            "source_hash": candidate["fingerprint"]["source_hash"],
            "source_span_hash": candidate["fingerprint"]["source_span_hash"],
        },
        "requested_emit_mode": "auto",
        "resolved_emit_mode": "tvl",
        "injection_mode": "context",
        "proposed_tvar_bindings": [
            {
                "tvar": signal["tvar"],
                "confidence": signal["confidence"],
                "domain_source": signal["domain_source"],
                "evidence": signal["evidence"],
                "injection_mode": "context",
                "current_value": signal["tvar"].get("default"),
                "locator": {
                    "kind": "line_col",
                    "details": {
                        "function": function["name"],
                        "line": signal["evidence"]["line"],
                        "tvar": signal["tvar"]["name"],
                    },
                },
            }
            for signal in candidate.get("tvar_signals", [])
        ],
        "selected_objectives": [],
        "objective_candidates": candidate.get("objective_candidates", []),
        "dataset_plan": {
            "status": "stub_required",
            "format": "jsonl",
            "expected_fields": [],
        },
        "agent_enrichment": None,
        "emitted_files": [],
        "confirmation_state": {
            "objectives_confirmed": False,
            "dataset_confirmed": False,
            "write_authorized": False,
        },
        "warnings": [],
    }
