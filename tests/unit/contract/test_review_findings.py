"""Regression tests for the 5 adversarial-review findings on the #1979 contract.

Each test pins one fix in :func:`traigent.contract.validate_evaluation_contract`
(and, for #5, the shared ``prepare_call_arguments`` helper it reuses):

* #1 -- config/dataset VALUES never leak through a free-text finding message.
* #2 -- seamless call shapes model the runtime shim's config-name injection, so
  a config-injected required parameter no longer produces a false hard failure;
  any residual seamless gap is best-effort (warning), never fatal.
* #3 -- an OBJECTIVE metric that cannot be bound/introspected is an error
  (production hard-fails it); an auxiliary metric only warns.
* #4 -- when a custom evaluator overrides the metric lane, an un-bindable
  (and unused) metric must not flip ``report.ok``.
* #5 -- ``BaseEvaluator._should_expand_input_mapping`` subclass overrides are
  honoured through the instance ``_prepare_call_arguments`` dispatch.
"""

from __future__ import annotations

import json
import warnings

import traigent
from traigent.contract import ContractCode, validate_evaluation_contract
from traigent.evaluators.base import BaseEvaluator, prepare_call_arguments
from traigent.evaluators.local import LocalEvaluator

from ._support import error_codes, find_finding, finding_codes

DATASET = [{"input": {"question": "q1"}, "output": "gold"}]


# --------------------------------------------------------------------------- #
# #1 SECURITY -- config / dataset values never leak via error messages
# --------------------------------------------------------------------------- #
def test_config_value_not_leaked_via_injection_error_message():
    # An invalid ``algorithm`` value makes TraigentConfig.from_dict raise a
    # ValueError that echoes the offending VALUE ("... got 'sk-SECRET-VALUE'").
    # The finding is now VALUE-FREE (exception TYPE + injection mode + config
    # KEYS), so no value can enter the message regardless of how the exception
    # rendered it.
    secret = "sk-SECRET-VALUE"

    def agent(question, config):
        return "x"

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}}],
        injection_mode="parameter",
        config={"algorithm": secret},
    )

    serialized = json.dumps(report.to_dict())
    # The VALUE must never appear anywhere in the serialized report...
    assert secret not in serialized
    # ...while the offending KEY and the finding stay (actionable, no leak).
    assert "algorithm" in serialized  # reported in injection.config_keys
    finding = find_finding(report, ContractCode.INJECTION_PROVIDER_SETUP_FAILED)
    assert finding is not None
    assert finding.severity == "error"
    # Value-free: names the exception TYPE, the injection mode, and the offending
    # config KEY -- never the value, and never a "<redacted>" scrub (the value
    # never entered the message in the first place).
    assert secret not in finding.message
    assert "algorithm" in finding.message
    assert "parameter" in finding.message


def test_config_value_repr_form_not_leaked_via_injection_error_message():
    # Capstone re-review #1: many SDK errors format the offending value with
    # ``!r`` (validate_algorithm_name: "... got {algorithm!r}"), so a value with
    # control/special chars appears in the message as its REPR (escaped + quoted)
    # -- NOT as ``str(value)``. Scrubbing only ``str(value)`` therefore leaves the
    # repr form intact. Both forms must be scrubbed.
    secret = "sk-SECRET\nVALUE\t#9999"

    def agent(question, config):
        return "x"

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}}],
        injection_mode="parameter",
        config={"algorithm": secret},
    )

    finding = find_finding(report, ContractCode.INJECTION_PROVIDER_SETUP_FAILED)
    assert finding is not None
    assert finding.severity == "error"

    # Value-free message: the {value!r} repr form the pre-fix scrubber missed can
    # no longer appear, because str(exc) is never embedded. No form of the secret
    # may survive in the raw message (json.dumps re-escapes backslashes and would
    # mask the repr form, so assert against the raw message too).
    for form in (secret, repr(secret), repr(secret)[1:-1]):
        assert form not in finding.message
    assert secret not in json.dumps(report.to_dict())
    # ...while the offending KEY stays (reported in injection.config_keys).
    assert "algorithm" in json.dumps(report.to_dict())


def test_dataset_value_not_leaked_via_normalization_error_message():
    # A duplicate example_id makes Dataset validation raise an error that echoes
    # the id VALUE ("Duplicate example_id 'sk-...'"); it must be scrubbed.
    secret = "sk-DATASET-SECRET-XYZ"
    report = validate_evaluation_contract(
        func=lambda question: "x",
        dataset=[
            {"input": {"q": "v"}, "example_id": secret},
            {"input": {"q": "w"}, "example_id": secret},
        ],
    )

    serialized = json.dumps(report.to_dict())
    assert secret not in serialized
    finding = find_finding(report, ContractCode.DATASET_NORMALIZATION_FAILED)
    assert finding is not None
    # Value-free: names the exception TYPE + source kind and points at the
    # source, never the exc text (so the {example_id!r} secret cannot ride along).
    assert "inspect the dataset source" in finding.message
    assert secret not in finding.message
    assert "Duplicate example_id" not in finding.message


def test_contract_internal_error_is_value_free(monkeypatch):
    # The outer CONTRACT_INTERNAL_ERROR catch-all used to embed str(exc) for an
    # UNANTICIPATED exception, which can carry a config/dataset VALUE (a nested
    # library error echoing an argument). It is now VALUE-FREE: it names ONLY the
    # exception TYPE, so such a value can never enter. Force an unexpected internal
    # failure whose message echoes a config value and assert it never appears.
    import traigent.contract.evaluation as evalmod

    secret = "sk-INTERNAL\nSECRET-9999"

    def exploding_build_call_shapes(underlying_func, config_dict, *args, **kwargs):
        # Simulate an internal bug whose message echoes a config VALUE (here via
        # the whole config dict's repr) -- exactly the leak the catch-all guards.
        raise RuntimeError(f"boom while shaping calls: config={config_dict!r}")

    monkeypatch.setattr(evalmod, "_build_call_shapes", exploding_build_call_shapes)

    def agent(question):
        return "x"

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}}],
        config={"model": secret},
    )

    assert report.ok is False
    finding = find_finding(report, ContractCode.CONTRACT_INTERNAL_ERROR)
    assert finding is not None
    assert finding.severity == "error"
    # Value-free: names ONLY the exception TYPE, never str(exc) -- so the echoed
    # ``config={...}`` repr cannot ride along (no "<redacted>" scrub needed).
    assert "RuntimeError" in finding.message
    assert "config=" not in finding.message
    # No form of the secret (str, repr, or repr-body) may survive in the message.
    for form in (secret, repr(secret), repr(secret)[1:-1]):
        assert form not in finding.message
    assert secret not in json.dumps(report.to_dict())


# --------------------------------------------------------------------------- #
# #1 (fix3) -- close the two remaining error-message value-leak points in the
# same CLASS: a short (2-char) escaped config value, and a path/Dataset-instance
# dataset whose offending values the redactor cannot see.
# --------------------------------------------------------------------------- #
def test_two_char_escaped_config_value_not_leaked():
    # fix3 #1: _redact used to skip leaf values shorter than 3 chars, so a
    # realistic 2-char secret with an escape ("x\n") survived via the repr form
    # ('x\n') of the {value!r} INJECTION_PROVIDER_SETUP_FAILED message. The
    # redaction floor is now length 2, so both str and repr forms are scrubbed.
    secret = "x\n"

    def agent(question, config):
        return "x"

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}}],
        injection_mode="parameter",
        config={"algorithm": secret},
    )

    finding = find_finding(report, ContractCode.INJECTION_PROVIDER_SETUP_FAILED)
    assert finding is not None
    assert finding.severity == "error"
    # Value-free: no form of the 2-char secret can survive because str(exc) is
    # never embedded (no "<redacted>" scrub needed either)...
    for form in (secret, repr(secret), repr(secret)[1:-1]):
        assert form not in finding.message
    # ...nor its JSON escape form ("x\\n") anywhere in the serialized report,
    # while the offending KEY stays (reported in injection.config_keys).
    serialized = json.dumps(report.to_dict())
    assert repr(secret)[1:-1] not in serialized
    assert "algorithm" in serialized


def test_file_backed_dataset_value_not_leaked_via_value_free_message(
    tmp_path, monkeypatch
):
    # fix3 #2: for a PATH-backed dataset the offending values live in the file,
    # invisible to _redact (its redaction_source was None). A duplicate
    # example_id carrying a secret leaked via Dataset.__post_init__'s
    # {example_id!r}. The handler now emits a VALUE-FREE message (exception TYPE
    # + a pointer to the dataset source) instead of embedding str(exc), while
    # keeping the stable DATASET_NORMALIZATION_FAILED code and report.ok == False.
    secret = "sk-DATASET\nSECRET"
    monkeypatch.setenv("TRAIGENT_DATASET_ROOT", str(tmp_path))
    dataset_path = tmp_path / "dupe.jsonl"
    dataset_path.write_text(
        json.dumps({"input": {"q": "v"}, "example_id": secret})
        + "\n"
        + json.dumps({"input": {"q": "w"}, "example_id": secret})
        + "\n"
    )

    report = validate_evaluation_contract(
        func=lambda question: "x",
        dataset=str(dataset_path),
    )

    assert report.ok is False
    finding = find_finding(report, ContractCode.DATASET_NORMALIZATION_FAILED)
    assert finding is not None
    assert finding.severity == "error"
    # Value-free: names the exception TYPE and points at the source, never the
    # exception text (so the {example_id!r} secret cannot ride along).
    assert "inspect the dataset source" in finding.message
    assert "Duplicate example_id" not in finding.message
    # No form of the secret survives in the raw message or the serialized report.
    for form in (secret, repr(secret), repr(secret)[1:-1]):
        assert form not in finding.message
    serialized = json.dumps(report.to_dict())
    assert secret not in serialized
    assert repr(secret) not in serialized
    assert repr(secret)[1:-1] not in serialized


def test_dataset_instance_example_value_not_leaked_via_internal_error(monkeypatch):
    # fix3 #3 (defense in depth): a Dataset INSTANCE handed straight to the
    # contract is the redactor's blind spot -- its example values are not in
    # config and were not walked by _iter_leaf_values. If any internal stage
    # raises with such a value in its message, the outer CONTRACT_INTERNAL_ERROR
    # catch-all (which redacts against (config_dict, dataset)) must scrub it.
    # _iter_leaf_values now walks Dataset.examples, so the secret is a candidate.
    import traigent.contract.evaluation as evalmod
    from traigent.evaluators.base import Dataset, EvaluationExample

    secret = "sk-INLINE-DATASET-SECRET-4242"
    dataset = Dataset(
        examples=[
            EvaluationExample(
                input_data={"question": secret}, expected_output=None, metadata={}
            )
        ],
        name="inline",
    )

    def exploding_normalize(ds, max_examples):
        # A forced normalization failure whose message echoes the example VALUE
        # -- the value only reachable by walking the Dataset instance's examples.
        raise RuntimeError(f"normalize boom leaking {secret}")

    monkeypatch.setattr(evalmod, "_normalize_dataset", exploding_normalize)

    report = validate_evaluation_contract(func=lambda question: "x", dataset=dataset)

    assert report.ok is False
    finding = find_finding(report, ContractCode.CONTRACT_INTERNAL_ERROR)
    assert finding is not None
    assert finding.severity == "error"
    # Value-free: the catch-all names only the exception TYPE, so the example
    # value echoed by the forced error cannot ride along (no scrub required).
    assert "RuntimeError" in finding.message
    # The secret sitting in the Dataset instance's example value must appear in
    # no form in the message, and nowhere in the serialized report.
    for form in (secret, repr(secret), repr(secret)[1:-1]):
        assert form not in finding.message
    assert secret not in json.dumps(report.to_dict())


# --------------------------------------------------------------------------- #
# #1 (fix4) -- VALUE-FREE messages defeat exotic repr tricks that post-hoc
# redaction cannot. Two confirmed classes: a ``str`` subclass whose ``__repr__``
# LIES (str() != repr()), and a crafted ``__signature__``. The finding messages
# carry ONLY the exception TYPE + safe identifiers, so no value can enter.
# --------------------------------------------------------------------------- #
def test_str_subclass_lying_repr_config_value_is_value_free():
    # A ``str`` subclass whose ``__repr__`` lies defeats scrubbing:
    # ``validate_algorithm_name`` renders ``got {algorithm!r}`` so repr() emits the
    # secret, while ``_iter_leaf_values`` only ever saw str() -- leaving the secret
    # unscrubbable. Value-free messages never embed str(exc), so the lie has
    # nowhere to land.
    class ReprLyingStr(str):
        def __repr__(self):
            return "'sk-CONFIG-REPR-LEAK'"

    def agent(question, config):
        return "x"

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}}],
        injection_mode="parameter",
        config={"algorithm": ReprLyingStr("innocuous")},
    )

    serialized = json.dumps(report.to_dict())
    assert "sk-CONFIG-REPR-LEAK" not in serialized
    finding = find_finding(report, ContractCode.INJECTION_PROVIDER_SETUP_FAILED)
    assert finding is not None
    assert finding.severity == "error"
    assert "sk-CONFIG-REPR-LEAK" not in finding.message
    # Value-free but actionable: the exception TYPE, the mode, and the config KEY.
    assert "algorithm" in finding.message
    assert "parameter" in finding.message


def test_str_subclass_lying_repr_dataset_id_is_value_free():
    # The same repr-lie against the dataset lane: a duplicate ``example_id`` str
    # subclass whose ``__repr__`` lies leaks through ``Dataset.__post_init__``'s
    # ``{example_id!r}``. The INLINE dataset path used to redact str(exc) and so
    # missed the lie; it is value-free now too.
    class ReprLyingStr(str):
        def __repr__(self):
            return "'sk-DATASET-REPR-LEAK'"

    dup = ReprLyingStr("dupe")
    report = validate_evaluation_contract(
        func=lambda question: "x",
        dataset=[
            {"input": {"q": "v"}, "example_id": dup},
            {"input": {"q": "w"}, "example_id": dup},
        ],
    )

    serialized = json.dumps(report.to_dict())
    assert "sk-DATASET-REPR-LEAK" not in serialized
    finding = find_finding(report, ContractCode.DATASET_NORMALIZATION_FAILED)
    assert finding is not None
    assert finding.severity == "error"
    assert "sk-DATASET-REPR-LEAK" not in finding.message
    assert "Duplicate example_id" not in finding.message
    assert "inspect the dataset source" in finding.message


def test_crafted_signature_metric_is_value_free():
    # A crafted ``__signature__`` makes ``inspect.signature`` raise with a VALUE in
    # its text -- the callable's own repr ("<function ... at 0x...> builtin has
    # invalid signature") or ``unexpected object <repr> in __signature__``.
    # EVALUATOR_SIGNATURE_UNAVAILABLE used to embed str(exc) in BOTH the finding
    # message and the binding's bind_error; both are value-free now (exception
    # TYPE + metric_name only).
    def bad_metric(output, expected):
        return 1.0

    bad_metric.__signature__ = "sk-EVAL-SIG-LEAK"  # invalid -> signature() raises

    report = validate_evaluation_contract(
        func=lambda question: "x",
        dataset=[{"input": {"question": "q1"}, "output": "gold"}],
        metric_functions={"accuracy": bad_metric},
        objectives=("accuracy",),
        config={"api_key": "sk-EVAL-SIG-LEAK"},
    )

    serialized = json.dumps(report.to_dict())
    assert "sk-EVAL-SIG-LEAK" not in serialized
    finding = find_finding(report, ContractCode.EVALUATOR_SIGNATURE_UNAVAILABLE)
    assert finding is not None
    assert finding.severity == "error"  # accuracy is an objective -> hard fail
    assert finding.metric_name == "accuracy"
    # Value-free: no object repr / memory address and no raw exc text.
    assert "0x" not in finding.message
    assert "<function" not in finding.message
    assert "builtin has invalid signature" not in finding.message
    # The binding's bind_error is value-free too (never str(exc)).
    binding = next(b for b in report.evaluator_bindings if b.metric_name == "accuracy")
    assert binding.bind_ok is False
    assert binding.bind_error is not None
    assert "0x" not in binding.bind_error
    assert "sk-EVAL-SIG-LEAK" not in binding.bind_error


# --------------------------------------------------------------------------- #
# #2 SEAMLESS -- config-injected required param binds OK (no false hard fail)
# --------------------------------------------------------------------------- #
def test_seamless_config_named_param_binds_without_false_failure():
    # def agent(question, model) called with dataset {question} + config {model}
    # works in production because the seamless runtime shim fills `model` from
    # config. The contract must therefore NOT report a hard AGENT_BIND_FAILED.
    def agent(question, model):
        return f"{question}-{model}"

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}}],
        injection_mode="seamless",
        config={"model": "gpt-4"},
    )

    assert report.ok is True
    shape = report.call_shapes[0]
    assert shape.bind_ok is True
    # The config-named parameter is modeled in the EFFECTIVE call shape.
    assert "model" in shape.effective_keyword_names
    assert "model" not in shape.runtime_keyword_names
    assert ContractCode.AGENT_BIND_FAILED not in finding_codes(report)


def test_seamless_residual_binding_gap_is_best_effort_not_fatal():
    # `extra` is neither a dataset key, a config key, nor defaulted -> a residual
    # gap. Seamless is best-effort, so this is advisory (warning), never an error
    # that flips report.ok.
    def agent(question, model, extra):
        return "x"

    report = validate_evaluation_contract(
        func=agent,
        dataset=[{"input": {"question": "q1"}}],
        injection_mode="seamless",
        config={"model": "gpt-4"},
    )

    assert report.ok is True
    assert error_codes(report) == []
    finding = find_finding(report, ContractCode.AGENT_BIND_FAILED)
    assert finding is not None
    assert finding.severity == "warning"
    assert report.call_shapes[0].bind_ok is False


# --------------------------------------------------------------------------- #
# #3 OBJECTIVE vs auxiliary metric severity
# --------------------------------------------------------------------------- #
def test_objective_metric_signature_unavailable_is_error():
    # metric_functions={"accuracy": max}: inspect.signature(max) raises, and
    # accuracy IS an objective -> production hard-fails -> error, report.ok False.
    report = validate_evaluation_contract(
        func=lambda question: "x",
        dataset=DATASET,
        metric_functions={"accuracy": max},
        objectives=("accuracy",),
    )
    assert report.ok is False
    finding = find_finding(report, ContractCode.EVALUATOR_SIGNATURE_UNAVAILABLE)
    assert finding is not None
    assert finding.severity == "error"
    assert finding.metric_name == "accuracy"


def test_auxiliary_metric_signature_unavailable_is_warning():
    # Same un-introspectable metric, but named "aux" (NOT an objective) -> only a
    # warning; the report stays ok (production only degrades auxiliary metrics).
    report = validate_evaluation_contract(
        func=lambda question: "x",
        dataset=DATASET,
        metric_functions={"aux": max},
        objectives=("accuracy",),
    )
    assert report.ok is True
    finding = find_finding(report, ContractCode.EVALUATOR_SIGNATURE_UNAVAILABLE)
    assert finding is not None
    assert finding.severity == "warning"
    assert finding.metric_name == "aux"


def test_objective_metric_bind_failure_is_error():
    # An over-arity scoring_function maps to the "accuracy" objective and cannot
    # bind -> error, report.ok False.
    def scorer(a, b, c, d):
        return 1.0

    report = validate_evaluation_contract(
        func=lambda question: "x",
        dataset=DATASET,
        scoring_function=scorer,
        objectives=("accuracy",),
    )
    assert report.ok is False
    finding = find_finding(report, ContractCode.EVALUATOR_BIND_FAILED)
    assert finding is not None
    assert finding.severity == "error"
    assert finding.metric_name == "accuracy"


def test_auxiliary_metric_bind_failure_is_warning():
    # An over-arity auxiliary metric (NOT an objective) that cannot bind -> only
    # a warning; report stays ok.
    def aux(a, b, c, d):
        return 1.0

    report = validate_evaluation_contract(
        func=lambda question: "x",
        dataset=DATASET,
        metric_functions={"aux": aux},
        objectives=("accuracy",),
    )
    assert report.ok is True
    finding = find_finding(report, ContractCode.EVALUATOR_BIND_FAILED)
    assert finding is not None
    assert finding.severity == "warning"
    assert finding.metric_name == "aux"


# --------------------------------------------------------------------------- #
# #4 CUSTOM_EVALUATOR precedence -- metric lane must not drive report.ok
# --------------------------------------------------------------------------- #
def test_custom_evaluator_skips_metric_lane_and_does_not_force_failure():
    def custom_eval(func, config, example):
        return 1.0

    def bad_scorer(a, b, c, d):  # over-arity: would EVALUATOR_BIND_FAILED
        return 1.0

    def raw_agent(question):
        return "x"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt_agent = traigent.optimize(
            configuration_space={"model": ["a", "b"]},
            custom_evaluator=custom_eval,
            scoring_function=bad_scorer,
        )(raw_agent)

    report = validate_evaluation_contract(func=opt_agent, dataset=DATASET)

    # A custom evaluator overrides the metric lane: the un-bindable scoring
    # function must NOT flip report.ok, and the lane is skipped entirely.
    assert report.ok is True
    assert "custom_evaluator" in report.unsupported
    assert ContractCode.EVALUATOR_BIND_FAILED not in finding_codes(report)
    assert report.evaluator_bindings == ()
    assert find_finding(report, ContractCode.CUSTOM_EVALUATOR_UNSUPPORTED) is not None


# --------------------------------------------------------------------------- #
# #5 EXTRACTION subclass dispatch through the instance method
# --------------------------------------------------------------------------- #
def test_should_expand_subclass_override_honored_through_instance_method():
    calls: list[tuple[str, tuple[str, ...]]] = []

    class SubEvaluator(LocalEvaluator):
        @staticmethod
        def _should_expand_input_mapping(func, payload):
            calls.append(("sub", tuple(sorted(payload))))
            return False  # force whole-mapping (opposite of the base default)

    def agent(a, b):
        return "x"

    payload = {"a": 1, "b": 2}

    # Base default: a multi-param mapping expands into keyword args.
    base_args, base_kwargs = LocalEvaluator()._prepare_call_arguments(
        agent, {}, payload
    )
    assert base_args == ()
    assert base_kwargs == {"a": 1, "b": 2}

    # The subclass override is invoked (dispatch honoured) and its decision wins:
    # the whole mapping is passed positionally instead.
    sub_args, sub_kwargs = SubEvaluator()._prepare_call_arguments(agent, {}, payload)
    assert sub_args == (payload,)
    assert sub_kwargs == {}
    assert calls == [("sub", ("a", "b"))]


def test_should_expand_regular_method_override_honored_through_instance_method():
    # Capstone re-review #3: the instance dispatch must bind ``self`` so a
    # subclass overriding _should_expand_input_mapping as a REGULAR (non-static)
    # method is honoured. With the pre-fix UNBOUND
    # ``type(self)._should_expand_input_mapping`` this override would receive
    # ``func`` as ``self`` (and drop ``payload``) -> wrong result / TypeError.
    calls: list[tuple[str, tuple[str, ...]]] = []

    class RegularMethodEvaluator(LocalEvaluator):
        def _should_expand_input_mapping(self, func, payload):  # regular method
            calls.append(("regular", tuple(sorted(payload))))
            return False  # force whole-mapping (opposite of the base default)

    def agent(a, b):
        return "x"

    payload = {"a": 1, "b": 2}

    sub_args, sub_kwargs = RegularMethodEvaluator()._prepare_call_arguments(
        agent, {}, payload
    )
    # The regular-method override ran (dispatch honoured, self bound) and its
    # decision won: the whole mapping is passed positionally, not expanded.
    assert calls == [("regular", ("a", "b"))]
    assert sub_args == (payload,)
    assert sub_kwargs == {}

    # The base default (@staticmethod) still expands a multi-param mapping into
    # kwargs -- proving the bound-dispatch fix is bit-identical for staticmethods.
    base_args, base_kwargs = LocalEvaluator()._prepare_call_arguments(
        agent, {}, payload
    )
    assert base_args == ()
    assert base_kwargs == {"a": 1, "b": 2}


def test_prepare_call_arguments_default_predicate_is_base_evaluator():
    # Contract callers keep the default predicate == BaseEvaluator's behaviour.
    def agent(a, b):
        return "x"

    payload = {"a": 1, "b": 2}
    default_args, default_kwargs = prepare_call_arguments(agent, {}, payload)
    base_args, base_kwargs = prepare_call_arguments(
        agent,
        {},
        payload,
        should_expand=BaseEvaluator._should_expand_input_mapping,
    )
    assert (default_args, default_kwargs) == (base_args, base_kwargs) == ((), payload)
