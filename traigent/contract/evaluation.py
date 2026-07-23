"""No-execution evaluation-compatibility contract for the Traigent SDK.

:func:`validate_evaluation_contract` answers, WITHOUT running anything, "would
this (function, dataset, evaluator, injection) combination actually work under a
real optimization run?" It reuses the exact production decision points --
dataset normalization, injection-mode resolution, per-example call shaping,
and metric-callback binding -- but stops precisely at each execution boundary:

* the target function / provider wrapper is **never called**;
* metric functions and custom evaluators are **never called**;
* a :class:`~traigent.config.context.ConfigurationContext` is **never entered**;
* no backend/network client is imported and no cost is ever incurred.

The result is an :class:`~traigent.contract.report.EvaluationContractReport` --
frozen dataclasses that serialize with ``to_dict()`` and report configuration
**keys only**, never values, so a serialized report cannot leak a secret.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from traigent.contract.codes import EVALUATION_CONTRACT_VERSION, ContractCode
from traigent.contract.report import (
    ContractFinding,
    EvaluationContractReport,
    EvaluatorBinding,
    ExampleCallShape,
    InjectionSummary,
    NormalizedExampleSummary,
)

if TYPE_CHECKING:  # avoid import cost / cycles at runtime
    from traigent.api.decorators import InjectionOptions
    from traigent.config.types import InjectionMode
    from traigent.core.optimized_function import OptimizedFunction
    from traigent.evaluators.base import Dataset, EvaluationExample

# Placeholder used only to satisfy ``Signature.bind`` for the injected config
# parameter in parameter mode. It is never serialized, so no value leaks.
_CONFIG_BIND_SENTINEL = object()

# Placeholder metric ``output`` used only for binding. Never serialized, never
# passed to a metric that is actually invoked (metrics are never invoked here).
_METRIC_OUTPUT_SENTINEL = object()

# VALUE-FREE-BY-CONSTRUCTION is the primary guarantee: every finding message is
# composed ONLY of value-free tokens -- the exception TYPE name, the stable
# ContractCode, and identifiers already known to be safe (config KEY names, param
# names, metric_name, the dataset source kind). No finding message ever embeds
# ``str(exc)`` / ``repr(...)`` / a ``{!r}``-derived value, so an exotic repr trick
# (a ``str`` subclass whose ``__repr__`` lies, a crafted ``__signature__``) has
# nothing to leak *into*. ``_redact`` below is retained ONLY as a secondary
# defense-in-depth net over the one residual free-text surface (the
# ``CONTRACT_INTERNAL_ERROR`` catch-all, itself already value-free) -- correctness
# does NOT depend on it, and it is deliberately NOT applied to the value-free
# messages (running it there could corrupt an intentionally-reported KEY that
# happened to equal a value).
_REDACTED = "<redacted>"

# Leaf values shorter than this are never scrubbed. We deliberately draw the
# line at length 2: a realistic short secret with an escape (e.g. a 2-char
# ``"x\n"`` that renders as the repr ``'x\n'`` in a ``{value!r}`` error) MUST be
# removed, so anything of length >= 2 is scrubbed. Length-1 values are the one
# accepted gap -- a lone character (e.g. ``"x"``) carries negligible secret
# entropy and blindly replacing it would corrupt unrelated diagnostic text (the
# "x" inside the word "example"), which is not a realistic secret. Longer
# values -- real API keys, tokens, model/algorithm names -- are always removed.
# The report itself stays keys-only regardless.
_REDACT_MIN_LEN = 2


def _iter_leaf_values(value: Any) -> Iterator[str]:
    """Yield the scalar leaf VALUES (as strings) of a config/dataset structure.

    Mappings are walked by their VALUES only -- keys are intentionally
    reportable -- and sequences element-wise, so a nested secret is still
    surfaced. A ``Dataset``-shaped object is duck-typed and walked via its
    ``.examples`` (so a secret sitting in an inline ``Dataset`` instance handed
    straight to the contract is still scrubbed), and each ``EvaluationExample``-
    shaped object via its ``input_data`` / ``expected_output`` / ``metadata``
    payloads. ``None`` and booleans are skipped (never secrets).
    """
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_leaf_values(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _iter_leaf_values(item)
    elif hasattr(value, "examples") and not hasattr(value, "input_data"):
        # Dataset-shaped: walk its examples (each an EvaluationExample-shaped
        # object handled by the branch below). Guarded against an Evaluation
        # example that also exposes ``examples`` so we do not double-dispatch.
        for item in value.examples:
            yield from _iter_leaf_values(item)
    elif hasattr(value, "input_data") or hasattr(value, "expected_output"):
        for attr in ("input_data", "expected_output", "metadata"):
            yield from _iter_leaf_values(getattr(value, attr, None))
    elif value is None or isinstance(value, bool):
        return
    else:
        text = str(value)
        if text:
            yield text


def _redact(message: str, source: Any) -> str:
    """Scrub every occurrence of a leaf VALUE found in *source* out of *message*.

    Each leaf value is scrubbed in every form an error message might render it:

    * ``str(value)`` -- the plain string form;
    * ``repr(value)`` -- the form produced by ``{value!r}`` formatting (which
      SDK exceptions use, e.g. ``got {algorithm!r}``). ``repr`` escapes control
      characters and adds surrounding quotes, so a value like ``"sk-KEY\\n"``
      appears in the message as ``'sk-KEY\\n'`` and is *not* found by scrubbing
      ``str(value)`` alone;
    * the ``repr`` body without its surrounding quotes -- for a value embedded
      inside a larger repr (e.g. a container repr) where the outer quotes belong
      to a different token.

    Longest forms first so a value that contains a shorter one is scrubbed
    whole. All non-value text (parameter names, the offending KEY, the
    structural reason, the exception type) is preserved, so the diagnostic stays
    actionable without leaking the value that triggered it.
    """
    candidates: set[str] = set()
    for text in _iter_leaf_values(source):
        if len(text) < _REDACT_MIN_LEN or not text.strip():
            continue
        candidates.add(text)
        rendered = repr(text)
        candidates.add(rendered)
        if len(rendered) >= 2 and rendered[0] in "'\"" and rendered[-1] == rendered[0]:
            candidates.add(rendered[1:-1])
    for value in sorted(candidates, key=len, reverse=True):
        if value in message:
            message = message.replace(value, _REDACTED)
    return message


def validate_evaluation_contract(
    *,
    func: Callable[..., Any] | OptimizedFunction,
    dataset: Dataset
    | str
    | os.PathLike[str]
    | Sequence[dict[str, Any] | EvaluationExample],
    scoring_function: Callable[..., Any] | None = None,
    metric_functions: dict[str, Callable[..., Any]] | None = None,
    objectives: Sequence[str] = ("accuracy",),
    config: Mapping[str, Any] | None = None,
    injection_options: InjectionOptions | None = None,
    injection_mode: InjectionMode | str | None = None,
    config_param: str | None = None,
    max_examples: int | None = None,
) -> EvaluationContractReport:
    """Statically check evaluation compatibility WITHOUT executing anything.

    Args:
        func: A raw callable or a decorated ``OptimizedFunction``. For an
            ``OptimizedFunction`` the injection/evaluator settings come from the
            decorator; for a raw callable they come from the injection args below.
        dataset: A ``Dataset``, a JSONL/JSON path, or an inline sequence of
            ``{"input": ...}`` mappings / ``EvaluationExample`` objects.
        scoring_function: Optional single-objective scorer (mapped to an
            objective via ``build_metric_functions``).
        metric_functions: Optional named metric callbacks.
        objectives: Objective names (drives scorer-to-objective mapping).
        config: Example configuration; only its KEYS are reported.
        injection_options: Grouped ``InjectionOptions`` bundle (raw callables).
        injection_mode: Flat injection mode (raw callables).
        config_param: Flat config parameter name for parameter mode.
        max_examples: Cap on per-example work (example summaries + call shapes).

    Returns:
        An :class:`EvaluationContractReport`. ``report.ok`` is ``True`` iff no
        finding has ``severity == "error"``.

    Raises:
        TypeError: Only for a caller-contract violation (``func`` is neither a
            callable nor an ``OptimizedFunction``). Everything discovered about
            the inputs themselves is reported as a finding, not raised.
    """
    from traigent import __version__ as sdk_version
    from traigent.core.optimized_function import OptimizedFunction
    from traigent.evaluators.base import EvaluationExample

    is_optimized = isinstance(func, OptimizedFunction)
    if not is_optimized and not callable(func):
        raise TypeError("func must be a callable or an OptimizedFunction")

    underlying_func = func.func if isinstance(func, OptimizedFunction) else func
    config_dict: dict[str, Any] = dict(config) if config else {}

    findings: list[ContractFinding] = []

    try:
        (
            normalized,
            dataset_name,
            dataset_source,
            example_summaries,
            dataset_findings,
        ) = _normalize_dataset(dataset, max_examples)
        findings.extend(dataset_findings)

        examples = list(normalized.examples) if normalized is not None else []

        (
            injection_summary,
            effective_mode,
            effective_config_param,
            setup_ok,
            injection_findings,
        ) = _resolve_injection(
            func,
            is_optimized,
            underlying_func,
            injection_options,
            injection_mode,
            config_param,
            config_dict,
        )
        findings.extend(injection_findings)

        effective_param_name = effective_config_param or "config"
        inject_config_param = effective_mode == "parameter" and setup_ok

        call_shapes, shape_findings = _build_call_shapes(
            underlying_func,
            config_dict,
            effective_param_name,
            inject_config_param,
            effective_mode,
            examples,
            max_examples,
        )
        findings.extend(shape_findings)

        binding_example = (
            examples[0]
            if examples
            else EvaluationExample(input_data={}, expected_output=None, metadata={})
        )
        (
            evaluator_bindings,
            unsupported,
            evaluator_findings,
        ) = _build_evaluator_bindings(
            is_optimized,
            func,
            scoring_function,
            metric_functions,
            objectives,
            config_dict,
            binding_example,
        )
        findings.extend(evaluator_findings)

    except Exception as exc:  # unexpected internal error -> loud, never silent
        # VALUE-FREE by construction: an unanticipated exception can carry a
        # config/dataset VALUE in its ``str(exc)`` (a nested library error echoing
        # an argument, or an object whose ``__repr__`` lies), so the message names
        # ONLY the exception TYPE and a generic phrase -- never ``str(exc)``.
        # ``_redact`` is applied purely as a belt-and-suspenders net over an
        # already value-free string (which carries no intended config KEY to
        # corrupt); correctness does NOT depend on it.
        findings.append(
            ContractFinding(
                code=ContractCode.CONTRACT_INTERNAL_ERROR,
                severity="error",
                message=_redact(
                    f"Unexpected internal error ({type(exc).__name__}) while "
                    "validating the evaluation contract.",
                    (config_dict, dataset),
                ),
                location="contract",
            )
        )
        return EvaluationContractReport(
            ok=False,
            sdk_version=sdk_version,
            contract_version=EVALUATION_CONTRACT_VERSION,
            dataset_name="dataset",
            dataset_source=None,
            examples=(),
            injection=InjectionSummary(
                effective_mode="unknown",
                source="unknown",
                config_param=None,
                provider_class=None,
                config_keys=(),
                setup_ok=False,
                seamless_injected_names=(),
            ),
            call_shapes=(),
            evaluator_bindings=(),
            unsupported=(),
            findings=tuple(findings),
        )

    ok = not any(finding.severity == "error" for finding in findings)
    return EvaluationContractReport(
        ok=ok,
        sdk_version=sdk_version,
        contract_version=EVALUATION_CONTRACT_VERSION,
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        examples=example_summaries,
        injection=injection_summary,
        call_shapes=call_shapes,
        evaluator_bindings=evaluator_bindings,
        unsupported=unsupported,
        findings=tuple(findings),
    )


# --------------------------------------------------------------------------- #
# Stage: dataset normalization (reuses the production Dataset loaders)
# --------------------------------------------------------------------------- #
def _normalize_dataset(
    dataset: Any,
    max_examples: int | None,
) -> tuple[
    Dataset | None,
    str,
    str | None,
    tuple[NormalizedExampleSummary, ...],
    list[ContractFinding],
]:
    from traigent.evaluators.base import (
        Dataset,
        _example_correlation_key,
        load_inline_dataset,
    )

    findings: list[ContractFinding] = []
    dataset_source: str | None = None
    normalized: Dataset | None = None

    try:
        if isinstance(dataset, Dataset):
            normalized = dataset
        elif isinstance(dataset, (str, os.PathLike)):
            dataset_source = os.fspath(dataset)
            normalized = Dataset.from_jsonl(dataset_source)
        elif isinstance(dataset, Sequence):
            normalized = load_inline_dataset(dataset)
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset).__name__}")
    except Exception as exc:
        # VALUE-FREE by construction for EVERY input kind. A normalization error
        # can echo dataset input/expected VALUES: a validation error quoting the
        # offending payload, or ``Dataset.__post_init__`` rendering a duplicate/
        # unhashable id via ``{example_id!r}`` -- which an example_id whose
        # ``__repr__`` lies can weaponise so no post-hoc scrub can see the leaked
        # form. Post-hoc redaction is also blind to a file's contents (a path
        # source) and to a pre-built ``Dataset`` instance's example values. So we
        # NEVER embed ``str(exc)`` for ANY input kind: the message names only the
        # exception TYPE and the (value-free) source KIND and directs the caller
        # at the dataset source. The actual path stays reported separately as
        # ``dataset_source`` (a filename is not a secret).
        if isinstance(dataset, Dataset):
            source_label = "<Dataset>"
        elif isinstance(dataset, (str, os.PathLike)):
            source_label = "<dataset file>"
        else:
            source_label = "<inline sequence>"
        message = (
            f"Dataset normalization failed ({type(exc).__name__}) for "
            f"{source_label}; inspect the dataset source."
        )
        findings.append(
            ContractFinding(
                code=ContractCode.DATASET_NORMALIZATION_FAILED,
                severity="error",
                message=message,
                action=(
                    "Provide a JSONL/JSON path, a Dataset, or a sequence of "
                    "{'input': ...} mappings / EvaluationExample objects."
                ),
                location="dataset",
            )
        )
        return None, "dataset", dataset_source, (), findings

    dataset_name = str(getattr(normalized, "name", "dataset"))
    examples = list(normalized.examples)

    if not examples:
        findings.append(
            ContractFinding(
                code=ContractCode.DATASET_EMPTY,
                severity="error",
                message="Dataset normalized to zero examples.",
                action="Provide at least one example.",
                location="dataset",
            )
        )

    limit = len(examples) if max_examples is None else min(len(examples), max_examples)
    summaries: list[NormalizedExampleSummary] = []
    for index in range(limit):
        example = examples[index]
        input_data = example.input_data
        is_mapping = isinstance(input_data, Mapping)
        input_keys = (
            tuple(sorted(str(key) for key in input_data.keys())) if is_mapping else None
        )
        metadata = example.metadata or {}
        summaries.append(
            NormalizedExampleSummary(
                index=index,
                correlation_key=str(_example_correlation_key(example, index)),
                input_is_mapping=is_mapping,
                input_type=type(input_data).__name__,
                input_keys=input_keys,
                has_expected_output=example.expected_output is not None,
                metadata_keys=tuple(sorted(str(key) for key in metadata.keys())),
            )
        )
        if input_data is None or (is_mapping and len(input_data) == 0):
            findings.append(
                ContractFinding(
                    code=ContractCode.DATASET_MISSING_INPUT,
                    severity="warning",
                    message=f"Example {index} has an empty input payload.",
                    action="Ensure every example carries a non-empty input.",
                    location="dataset",
                    example_index=index,
                )
            )

    return normalized, dataset_name, dataset_source, tuple(summaries), findings


# --------------------------------------------------------------------------- #
# Stage: injection resolution (reuses the production provider machinery)
# --------------------------------------------------------------------------- #
def _resolve_injection(
    func: Any,
    is_optimized: bool,
    underlying_func: Callable[..., Any],
    injection_options: Any,
    injection_mode: Any,
    config_param: str | None,
    config_dict: dict[str, Any],
) -> tuple[InjectionSummary, str, str | None, bool, list[ContractFinding]]:
    from traigent.config.providers import (
        SeamlessParameterProvider,
        get_provider,
    )
    from traigent.config.types import InjectionMode
    from traigent.utils.exceptions import (
        ConfigurationError,
        FeatureNotAvailableError,
    )

    findings: list[ContractFinding] = []
    config_keys = tuple(sorted(str(key) for key in config_dict.keys()))
    # Value-free, actionable descriptor of the config KEYS in play (never their
    # values). Reused verbatim in the injection findings so a config-driven
    # failure still names the keys the caller can inspect, without echoing a
    # single value.
    config_keys_desc = ", ".join(config_keys) if config_keys else "none"

    effective_config_param = config_param
    if is_optimized:
        raw_mode: Any = func.injection_mode
        effective_config_param = func.config_param
        source = "optimized_function"
    else:
        from traigent.api.decorators import _resolve_injection_bundle_options

        flat_mode = InjectionMode.CONTEXT if injection_mode is None else injection_mode
        defaults = {
            "injection_mode": InjectionMode.CONTEXT,
            "config_param": None,
            "auto_override_frameworks": False,
            "framework_targets": None,
            "effectuation": False,
        }
        try:
            resolved = _resolve_injection_bundle_options(
                injection_options,
                flat_mode,
                config_param,
                False,
                None,
                False,
                defaults,
            )
        except TypeError as exc:
            findings.append(
                ContractFinding(
                    code=ContractCode.INJECTION_OPTIONS_CONFLICT,
                    severity="error",
                    message=(
                        f"Injection options were supplied inconsistently "
                        f"({type(exc).__name__}): the same option was given via both "
                        "a flat kwarg and the injection_options bundle."
                    ),
                    action=(
                        "Supply each injection option via exactly one of the flat "
                        "kwargs or the injection_options bundle, not both."
                    ),
                    location="injection",
                )
            )
            raw_mode = flat_mode
            effective_config_param = config_param
        else:
            raw_mode = resolved[0]
            effective_config_param = resolved[1]

        if injection_options is not None:
            source = "injection_options"
        elif injection_mode is not None or config_param is not None:
            source = "flat_kwargs"
        else:
            source = "default"

    effective_mode = str(getattr(raw_mode, "value", raw_mode))

    provider: Any = None
    provider_class: str | None = None
    setup_ok = False
    seamless_names: tuple[str, ...] = ()

    try:
        provider = get_provider(effective_mode, config_param=effective_config_param)
        provider_class = type(provider).__name__
    except FeatureNotAvailableError as exc:
        findings.append(
            ContractFinding(
                code=ContractCode.SEAMLESS_INJECTION_UNAVAILABLE,
                severity="warning",
                message=(
                    f"Seamless injection is unavailable ({type(exc).__name__}) for "
                    f"injection mode '{effective_mode}'."
                ),
                action="Install the seamless plugin or use a bundled mode.",
                location="injection",
            )
        )
    except ConfigurationError as exc:
        findings.append(
            ContractFinding(
                code=ContractCode.INJECTION_MODE_UNSUPPORTED,
                severity="error",
                message=(
                    f"Injection mode '{effective_mode}' is unsupported "
                    f"({type(exc).__name__})."
                ),
                action='Use injection_mode "context", "parameter", or "seamless".',
                location="injection",
            )
        )

    if provider is not None:
        try:
            # Construction only -- builds the wrapper exactly as the real
            # _setup_function_wrapper does; the wrapper is NEVER called.
            provider.inject_config(underlying_func, config_dict, effective_config_param)
            setup_ok = True
        except ConfigurationError as exc:
            if effective_mode == "parameter":
                findings.append(
                    ContractFinding(
                        code=ContractCode.INJECTION_CONFIG_PARAM_MISSING,
                        severity="error",
                        message=(
                            "Parameter-mode injection requires config parameter "
                            f"'{effective_config_param or 'config'}', which the target "
                            f"function does not accept ({type(exc).__name__})."
                        ),
                        action=(
                            f"Add a '{effective_config_param or 'config'}' parameter "
                            "to the function, or use injection_mode='context'."
                        ),
                        location="injection",
                    )
                )
            else:
                findings.append(
                    ContractFinding(
                        code=ContractCode.INJECTION_PROVIDER_SETUP_FAILED,
                        severity="error",
                        message=(
                            f"Injection provider setup failed ({type(exc).__name__}) "
                            f"for injection mode '{effective_mode}' "
                            f"(config keys: {config_keys_desc})."
                        ),
                        location="injection",
                    )
                )
        except (TypeError, ValueError) as exc:
            findings.append(
                ContractFinding(
                    code=ContractCode.INJECTION_PROVIDER_SETUP_FAILED,
                    severity="error",
                    message=(
                        f"Injection provider setup failed ({type(exc).__name__}) "
                        f"for injection mode '{effective_mode}' "
                        f"(config keys: {config_keys_desc})."
                    ),
                    location="injection",
                )
            )

        if (
            effective_mode == "seamless"
            and setup_ok
            and isinstance(provider, SeamlessParameterProvider)
        ):
            try:
                # Structural probe only: compiles + builds a FunctionType but
                # NEVER runs the function body (that happens at call time).
                _new_func, modified_vars = provider._transform_function(
                    underlying_func, config_dict
                )
                seamless_names = tuple(sorted(str(name) for name in modified_vars))
            except ConfigurationError as exc:
                findings.append(
                    ContractFinding(
                        code=ContractCode.SEAMLESS_INJECTION_UNAVAILABLE,
                        severity="warning",
                        message=(
                            "Seamless AST injection could not transform the target "
                            f"function ({type(exc).__name__}) for injection mode "
                            f"'{effective_mode}'."
                        ),
                        action=(
                            "Seamless AST injection needs importable source; use "
                            "injection_mode='context' for functions without source."
                        ),
                        location="injection",
                    )
                )

    summary = InjectionSummary(
        effective_mode=effective_mode,
        source=source,
        config_param=effective_config_param,
        provider_class=provider_class,
        config_keys=config_keys,
        setup_ok=setup_ok,
        seamless_injected_names=seamless_names,
    )
    return summary, effective_mode, effective_config_param, setup_ok, findings


# --------------------------------------------------------------------------- #
# Stage: per-example call shapes (reuses prepare_call_arguments + Signature.bind)
# --------------------------------------------------------------------------- #
def _seamless_satisfied_param_names(
    signature: inspect.Signature | None, config_dict: dict[str, Any]
) -> tuple[str, ...]:
    """Parameter names a seamless runtime shim fills from config at call time.

    Mirrors ``runtime_injector.create_runtime_shim`` / ``_prepare_arguments``:
    any parameter whose NAME is a configuration key (and that is bindable by
    keyword) is satisfied from the frozen per-trial config before the raw
    function runs. Modeling those as satisfiable -- exactly like parameter-mode
    injection by name -- is what stops a seamless ``def agent(question, model)``
    called with dataset ``{question}`` + config ``{model}`` from a false
    ``AGENT_BIND_FAILED``.
    """
    if signature is None:
        return ()
    return tuple(
        name
        for name, param in signature.parameters.items()
        if name in config_dict
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    )


def _build_call_shapes(
    underlying_func: Callable[..., Any],
    config_dict: dict[str, Any],
    effective_param_name: str,
    inject_config_param: bool,
    effective_mode: str,
    examples: list[Any],
    max_examples: int | None,
) -> tuple[tuple[ExampleCallShape, ...], list[ContractFinding]]:
    from traigent.evaluators.base import (
        _example_correlation_key,
        prepare_call_arguments,
    )

    findings: list[ContractFinding] = []
    shapes: list[ExampleCallShape] = []

    try:
        raw_signature: inspect.Signature | None = inspect.signature(underlying_func)
    except (TypeError, ValueError):
        raw_signature = None

    # Seamless injects config into the function at call time via a runtime shim,
    # so parameters whose name is a config key are satisfiable there. Seamless is
    # a best-effort surface: a residual binding gap is advisory, never fatal.
    seamless = effective_mode == "seamless"
    seamless_config_params = (
        _seamless_satisfied_param_names(raw_signature, config_dict) if seamless else ()
    )

    limit = len(examples) if max_examples is None else min(len(examples), max_examples)
    for index in range(limit):
        example = examples[index]
        input_data = example.input_data
        example_id = str(_example_correlation_key(example, index))

        # Runtime shape: production passes the provider-wrapped func (whose
        # @wraps signature mirrors the raw func) to _prepare_call_arguments,
        # which resolves the injection mode to "context" because
        # _traigent_injection_mode is never set in production.
        runtime_args, runtime_kwargs = prepare_call_arguments(
            underlying_func, config_dict, input_data, injection_mode="context"
        )
        expanded = isinstance(input_data, Mapping) and len(runtime_args) == 0
        runtime_keyword_names = tuple(sorted(str(key) for key in runtime_kwargs))

        effective_args = runtime_args
        effective_kwargs = dict(runtime_kwargs)
        if inject_config_param and effective_param_name not in effective_kwargs:
            # Parameter mode: the wrapper injects the config parameter before
            # calling the raw function. Only the NAME is modeled; the sentinel
            # value is never serialized.
            effective_kwargs[effective_param_name] = _CONFIG_BIND_SENTINEL
        for param_name in seamless_config_params:
            # Seamless mode: the runtime shim fills each config-named parameter
            # before the raw function runs. Model those as satisfiable by name;
            # the sentinel value is never serialized.
            effective_kwargs.setdefault(param_name, _CONFIG_BIND_SENTINEL)
        effective_keyword_names = tuple(sorted(str(key) for key in effective_kwargs))

        if raw_signature is None:
            bind_ok = False
            bind_error: str | None = None
            findings.append(
                ContractFinding(
                    code=ContractCode.AGENT_SIGNATURE_UNAVAILABLE,
                    severity="warning",
                    message=(
                        "Cannot introspect the target function's signature for "
                        f"example {index}."
                    ),
                    action="Wrap builtins/partials in a plain Python function.",
                    location="agent",
                    example_index=index,
                )
            )
        else:
            try:
                raw_signature.bind(*effective_args, **effective_kwargs)
                bind_ok = True
                bind_error = None
            except TypeError as exc:
                bind_ok = False
                # ``ExampleCallShape.bind_error`` keeps the ``Signature.bind`` text:
                # that text is value-free by CPython construction (it names
                # parameter names / counts only and NEVER reprs an argument VALUE),
                # so it is safe and diagnostically useful. The finding MESSAGE below
                # is nonetheless composed value-free (no ``{exc}``) to keep the
                # "every finding message is value-free" invariant uniform.
                bind_error = str(exc)
                if seamless:
                    # Best-effort surface: even after modeling the shim's
                    # config-named injection, a residual gap is advisory and must
                    # never flip report.ok (production seamless resolves the rest
                    # at call time / falls back rather than hard-failing here).
                    findings.append(
                        ContractFinding(
                            code=ContractCode.AGENT_BIND_FAILED,
                            severity="warning",
                            message=(
                                f"Example {index}: seamless best-effort binding gap "
                                "-- after modeling the runtime shim's config-named "
                                "injection, the call shape still does not bind to the "
                                "function signature. Seamless resolves parameters at "
                                "call time, so this is advisory."
                            ),
                            action=(
                                "Make each required parameter a dataset input key, "
                                "a configuration key, or give it a default."
                            ),
                            location="agent",
                            example_index=index,
                        )
                    )
                else:
                    findings.append(
                        ContractFinding(
                            code=ContractCode.AGENT_BIND_FAILED,
                            severity="error",
                            message=(
                                f"Example {index}: the resolved call shape does not "
                                "bind to the target function signature."
                            ),
                            action=(
                                "Align the function parameters with the dataset "
                                "input keys, or accept **kwargs."
                            ),
                            location="agent",
                            example_index=index,
                        )
                    )

        shapes.append(
            ExampleCallShape(
                example_id=example_id,
                expanded=expanded,
                runtime_args_count=len(runtime_args),
                runtime_keyword_names=runtime_keyword_names,
                effective_positional_count=len(effective_args),
                effective_keyword_names=effective_keyword_names,
                bind_ok=bind_ok,
                bind_error=bind_error,
            )
        )

    return tuple(shapes), findings


# --------------------------------------------------------------------------- #
# Stage: evaluator/metric binding (reuses build_metric_functions + the shared
# resolve_metric_call_binding extracted from LocalEvaluator)
# --------------------------------------------------------------------------- #
def _build_evaluator_bindings(
    is_optimized: bool,
    func: Any,
    scoring_function: Callable[..., Any] | None,
    metric_functions: dict[str, Callable[..., Any]] | None,
    objectives: Sequence[str],
    config_dict: dict[str, Any],
    example_obj: Any,
) -> tuple[tuple[EvaluatorBinding, ...], tuple[str, ...], list[ContractFinding]]:
    from traigent.core.optimization_pipeline import build_metric_functions
    from traigent.evaluators.local import resolve_metric_call_binding

    findings: list[ContractFinding] = []
    bindings: list[EvaluatorBinding] = []
    unsupported: list[str] = []

    if is_optimized:
        effective_scoring = (
            scoring_function if scoring_function is not None else func.scoring_function
        )
        effective_metrics = (
            metric_functions if metric_functions is not None else func.metric_functions
        )
        effective_objectives = tuple(func.objectives) or ("accuracy",)
        custom_evaluator = func.custom_evaluator
    else:
        effective_scoring = scoring_function
        effective_metrics = metric_functions
        effective_objectives = tuple(objectives) or ("accuracy",)
        custom_evaluator = None

    if custom_evaluator is not None:
        unsupported.append("custom_evaluator")
        findings.append(
            ContractFinding(
                code=ContractCode.CUSTOM_EVALUATOR_UNSUPPORTED,
                severity="warning",
                message=(
                    "A custom evaluator callback is configured; the contract cannot "
                    "inspect it without executing it, so it is reported as unsupported."
                ),
                action="Validate the custom evaluator with a dedicated runtime test.",
                location="evaluator",
            )
        )
        # A custom evaluator overrides the scoring/metric lane: production wraps
        # it in CustomEvaluatorWrapper and never binds the metric functions. So
        # the metric lane must NOT drive report.ok here -- skip it entirely
        # rather than let an un-bindable (and unused) metric flip the verdict.
        return tuple(bindings), tuple(unsupported), findings

    effective_metric_functions = build_metric_functions(
        effective_metrics, effective_scoring, effective_objectives
    )

    objective_names = set(effective_objectives)

    for metric_name in sorted(effective_metric_functions):
        metric_func = effective_metric_functions[metric_name]
        # A metric that is an optimization OBJECTIVE is hard-failed by
        # production's LocalEvaluator when it cannot be bound/introspected (it
        # defines the search signal); an auxiliary/informational metric only
        # degrades. Mirror that: objective failures are errors, others warnings.
        is_objective = metric_name in objective_names
        failure_severity = "error" if is_objective else "warning"
        try:
            binding = resolve_metric_call_binding(
                metric_func,
                _METRIC_OUTPUT_SENTINEL,
                example_obj,
                config_dict,
                {},
                0,
            )
        except (TypeError, ValueError) as exc:
            # VALUE-FREE: ``inspect.signature`` can raise with a VALUE in its text
            # -- a crafted ``__signature__`` (``unexpected object <repr> in
            # __signature__ attribute``) or the callable's own repr -- so the
            # message names ONLY the metric_name (a safe identifier) and the
            # exception TYPE, never ``str(exc)``. The signature could not be
            # introspected, so no parameter names are available to add.
            findings.append(
                ContractFinding(
                    code=ContractCode.EVALUATOR_SIGNATURE_UNAVAILABLE,
                    severity=failure_severity,
                    message=(
                        f"Metric '{metric_name}': signature could not be introspected "
                        f"({type(exc).__name__})."
                    ),
                    location="evaluator",
                    metric_name=metric_name,
                )
            )
            bindings.append(
                EvaluatorBinding(
                    metric_name=metric_name,
                    binding_mode="signature_unavailable",
                    matched_parameters=(),
                    unmatched_parameters=(),
                    bind_ok=False,
                    bind_error=(
                        f"signature introspection failed ({type(exc).__name__})"
                    ),
                )
            )
            continue

        # VALUE-FREE ``bind_error``: the underlying ``resolve_metric_call_binding``
        # bind_error is a ``Signature.bind`` message (names/counts only, never a
        # value), but to keep the guarantee uniform and independent of that fact we
        # store a composed value-free string rather than the raw bind text.
        bindings.append(
            EvaluatorBinding(
                metric_name=metric_name,
                binding_mode=binding.binding_mode,
                matched_parameters=binding.matched_parameters,
                unmatched_parameters=binding.unmatched_parameters,
                bind_ok=binding.bind_ok,
                bind_error=(
                    None
                    if binding.bind_ok
                    else "no argument candidate binds to the metric signature"
                ),
            )
        )

        if not binding.bind_ok:
            # VALUE-FREE: compose from the metric_name and the metric's own bindable
            # parameter NAMES (safe identifiers already surfaced on the binding) --
            # never ``binding.bind_error``'s raw text.
            unmatched_desc = (
                ", ".join(binding.unmatched_parameters)
                if binding.unmatched_parameters
                else "none"
            )
            findings.append(
                ContractFinding(
                    code=ContractCode.EVALUATOR_BIND_FAILED,
                    severity=failure_severity,
                    message=(
                        f"Metric '{metric_name}': no argument candidate binds to its "
                        f"signature (binding mode '{binding.binding_mode}', unmatched "
                        f"parameters: {unmatched_desc})."
                    ),
                    action=(
                        "Accept (output, expected) positionally or use recognized "
                        "parameter names (output/expected/llm_metrics/...)."
                    ),
                    location="evaluator",
                    metric_name=metric_name,
                )
            )
        elif binding.binding_mode.startswith("positional") and not (
            binding.matched_parameters
        ):
            findings.append(
                ContractFinding(
                    code=ContractCode.EVALUATOR_NO_RECOGNIZED_PARAMS,
                    severity="warning",
                    message=(
                        f"Metric '{metric_name}' binds only by bare positional "
                        f"fallback ({binding.binding_mode}); no parameter name was "
                        "recognized, so arguments are matched by position only."
                    ),
                    action=(
                        "Rename parameters to recognized names (output, expected, "
                        "llm_metrics) to avoid silent mis-scoring."
                    ),
                    location="evaluator",
                    metric_name=metric_name,
                )
            )

    return tuple(bindings), tuple(unsupported), findings
