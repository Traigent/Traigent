"""Data-flow detection strategy for tuned variable identification.

Uses intraprocedural backward program slicing to find variables that flow
into statistical call sites (LLM invocations, retrieval queries, embedding
calls).  This is the formal-methods approach: a variable is a tuned-variable
candidate if and only if it lies on a data-flow path to a probabilistic
operation.

Thread-safe: the strategy object is stateless.  All mutable state lives in
local variables inside ``detect()``.

Example::

    strategy = DataFlowDetectionStrategy()
    candidates = strategy.detect(source_code, "my_function")
    for c in candidates:
        print(f"{c.name} flows to {c.reasoning}")
"""

from __future__ import annotations

import ast
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from traigent.tuned_variables.detection_strategies import (
    _REVERSE_MAPPING,
    _extract_literal_value,
    _infer_candidate_type,
    _make_location,
    _suggest_range,
)
from traigent.tuned_variables.detection_types import (
    CandidateType,
    DetectionConfidence,
    SourceLocation,
    TunedVariableCandidate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sink catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SinkPattern:
    """A method-name pattern that identifies a statistical call site."""

    method_name: str
    category: str  # "llm" | "retrieval" | "embedding"


# Default sink patterns derived from traigent/integrations/mappings.py
DEFAULT_SINK_PATTERNS: tuple[SinkPattern, ...] = (
    # LLM family
    SinkPattern("invoke", "llm"),
    SinkPattern("ainvoke", "llm"),
    SinkPattern("create", "llm"),
    SinkPattern("complete", "llm"),
    SinkPattern("chat", "llm"),
    SinkPattern("generate", "llm"),
    SinkPattern("generate_content", "llm"),
    SinkPattern("completion", "llm"),
    SinkPattern("acompletion", "llm"),
    SinkPattern("stream", "llm"),
    SinkPattern("astream", "llm"),
    SinkPattern("batch", "llm"),
    SinkPattern("abatch", "llm"),
    SinkPattern("run", "llm"),
    SinkPattern("arun", "llm"),
    # Retrieval family
    SinkPattern("search", "retrieval"),
    SinkPattern("similarity_search", "retrieval"),
    SinkPattern("get_relevant_documents", "retrieval"),
    SinkPattern("aget_relevant_documents", "retrieval"),
    SinkPattern("retrieve", "retrieval"),
    SinkPattern("query", "retrieval"),
    # Embedding family
    SinkPattern("embed", "embedding"),
    SinkPattern("embed_query", "embedding"),
    SinkPattern("embed_documents", "embedding"),
    SinkPattern("get_text_embedding", "embedding"),
)

# Method names that are too generic to match as bare function calls
# (e.g. `from x import run; run(...)` should NOT be a sink).
# These are only matched when called as an attribute: `obj.run(...)`.
_GENERIC_SINK_NAMES: frozenset[str] = frozenset(
    {"run", "arun", "create", "query", "search", "chat", "batch", "abatch"}
)

# Class names whose constructors configure a statistical component.
# When an object is constructed with one of these and later used as a
# sink-call receiver, the constructor kwargs are also sink arguments.
CONSTRUCTOR_CLASS_HINTS: frozenset[str] = frozenset(
    {
        # LLM providers
        "OpenAI",
        "AsyncOpenAI",
        "ChatOpenAI",
        "AzureChatOpenAI",
        "Anthropic",
        "AsyncAnthropic",
        "ChatAnthropic",
        "ChatMistralAI",
        "Mistral",
        "ChatGoogleGenerativeAI",
        "GenerativeModel",
        "Cohere",
        "HuggingFacePipeline",
        "InferenceClient",
        "Bedrock",
        "ChatBedrock",
        "BedrockChatClient",
        # Vector stores / retrievers
        "FAISS",
        "Chroma",
        "Pinecone",
        "Weaviate",
        "Qdrant",
        # Embedding providers
        "OpenAIEmbeddings",
        "HuggingFaceEmbeddings",
        "CohereEmbeddings",
        "OpenAIEmbedding",
        "HuggingFaceEmbedding",
        "CohereEmbedding",
    }
)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _names_in_expr(node: ast.expr | None) -> set[str]:
    """Extract all ``ast.Name`` references from an AST expression tree.

    Used by the backward walk to discover which variables feed into a
    right-hand-side expression.
    """
    if node is None:
        return set()

    names: set[str] = set()
    _names_in_expr_acc(node, names)
    return names


def _iter_expr_children(node: ast.AST) -> list[ast.AST]:
    """Return child nodes that may contain referenced variable names."""
    if isinstance(node, ast.Attribute):
        return [node.value]
    if isinstance(node, ast.Subscript):
        return [node.value, node.slice]
    if isinstance(node, ast.BinOp):
        return [node.left, node.right]
    if isinstance(node, ast.UnaryOp):
        return [node.operand]
    if isinstance(node, ast.BoolOp):
        return list(node.values)
    if isinstance(node, ast.Compare):
        return [node.left, *node.comparators]
    if isinstance(node, ast.IfExp):
        return [node.test, node.body, node.orelse]
    if isinstance(node, ast.Call):
        return [
            node.func,
            *node.args,
            *(keyword.value for keyword in node.keywords),
        ]
    if isinstance(node, ast.Starred):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [
            value.value
            for value in node.values
            if isinstance(value, ast.FormattedValue)
        ]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return list(node.elts)
    if isinstance(node, ast.Dict):
        return [*(key for key in node.keys if key is not None), *node.values]
    return []


def _names_in_expr_acc(node: ast.AST, acc: set[str]) -> None:
    """Accumulate Name.id references from *node* into *acc*."""
    if isinstance(node, ast.Name):
        acc.add(node.id)
        return
    for child in _iter_expr_children(node):
        _names_in_expr_acc(child, acc)


def _receiver_name(node: ast.expr) -> str | None:
    """Extract the base variable name from a method-call receiver.

    ``llm.invoke(...)``                        →  ``"llm"``
    ``client.chat.completions.create(...)``    →  ``"client"``
    ``litellm.completion(...)``                →  ``"litellm"``
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _receiver_name(node.value)
    return None


def _call_class_name(node: ast.Call) -> str | None:
    """Extract the class name from a constructor call.

    ``ChatOpenAI(temperature=0.7)``  →  ``"ChatOpenAI"``
    ``langchain_openai.ChatOpenAI()``  →  ``"ChatOpenAI"``
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


# ---------------------------------------------------------------------------
# Def-Use builder
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Definition:
    """A single variable definition (assignment)."""

    var_name: str
    line: int
    value_node: ast.expr | None  # RHS of assignment; None for unpacking etc.
    stmt_node: ast.AST  # the full assignment statement


@dataclass
class _DefUseMap:
    """Lightweight intraprocedural definition map for a function body."""

    definitions: dict[str, list[_Definition]] = field(default_factory=dict)
    func_params: set[str] = field(default_factory=set)

    def add(self, defn: _Definition) -> None:
        self.definitions.setdefault(defn.var_name, []).append(defn)


class _DefUseBuilder(ast.NodeVisitor):
    """Build a _DefUseMap from a function body via a single AST pass.

    Scope-aware: skips nested function bodies to avoid false definitions.
    """

    def __init__(self) -> None:
        self.result = _DefUseMap()
        self._scope_depth = 0

    # -- Scope tracking ----------------------------------------------------

    def _enter_nested(self, node: ast.AST) -> None:
        if self._scope_depth > 0:
            return  # Already inside the target function; skip nested def
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_nested(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_nested(node)

    # -- Definitions -------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        if self._scope_depth == 0:
            return
        for target in node.targets:
            self._record_targets(target, node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._visit_single_target_assign(node.target, node.value, node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._visit_single_target_assign(node.target, node.value, node)

    def visit_For(self, node: ast.For) -> None:
        if self._scope_depth == 0:
            return
        self._record_targets(node.target, None, node)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        if self._scope_depth == 0:
            return
        for item in node.items:
            if item.optional_vars is not None:
                self._record_targets(item.optional_vars, item.context_expr, node)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._visit_single_target_assign(node.target, node.value, node)

    # -- Helpers -----------------------------------------------------------

    def _visit_single_target_assign(
        self,
        target: ast.expr,
        value: ast.expr | None,
        stmt: ast.AST,
    ) -> None:
        """Shared handler for AnnAssign, AugAssign, and NamedExpr."""
        if self._scope_depth == 0:
            return
        if isinstance(target, ast.Name) and value is not None:
            self.result.add(_Definition(target.id, target.lineno, value, stmt))
        self.generic_visit(stmt)

    def _record_targets(
        self,
        target: ast.expr,
        value_node: ast.expr | None,
        stmt: ast.AST,
    ) -> None:
        """Record definitions for potentially-nested assignment targets."""
        if isinstance(target, ast.Name):
            self.result.add(_Definition(target.id, target.lineno, value_node, stmt))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                # Tuple unpacking — value_node for each element is None
                # since we can't statically resolve positional mapping.
                self._record_targets(elt, None, stmt)
        elif isinstance(target, ast.Starred):
            self._record_targets(target.value, None, stmt)


def _build_def_use_map(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> _DefUseMap:
    """Build a _DefUseMap for *func_node* including its parameter names."""
    builder = _DefUseBuilder()
    builder.visit(func_node)
    result = builder.result

    # Record function parameters.
    for arg in func_node.args.args:
        result.func_params.add(arg.arg)
    for arg in func_node.args.posonlyargs:
        result.func_params.add(arg.arg)
    for arg in func_node.args.kwonlyargs:
        result.func_params.add(arg.arg)
    if func_node.args.vararg:
        result.func_params.add(func_node.args.vararg.arg)
    if func_node.args.kwarg:
        result.func_params.add(func_node.args.kwarg.arg)

    return result


# ---------------------------------------------------------------------------
# Sink finder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SinkArgument:
    """A variable flowing into a statistical call site."""

    var_name: str
    param_name: str | None  # kwarg name, or None if positional/starred
    sink_method: str
    sink_category: str
    line: int


class _SinkFinder(ast.NodeVisitor):
    """Walk a function body to find statistical call sites and their arguments."""

    def __init__(
        self,
        sink_methods: dict[str, SinkPattern],
        constructor_hints: frozenset[str],
        def_use: _DefUseMap,
    ) -> None:
        self.sink_args: list[_SinkArgument] = []
        self._sink_methods = sink_methods
        self._constructor_hints = constructor_hints
        self._def_use = def_use
        self._scope_depth = 0

    # -- Scope tracking (skip nested functions) ----------------------------

    def _enter_nested(self, node: ast.AST) -> None:
        if self._scope_depth > 0:
            return
        self._scope_depth += 1
        self.generic_visit(node)
        self._scope_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._enter_nested(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._enter_nested(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._scope_depth == 0:
            self.generic_visit(node)
            return

        sink = self._match_sink(node)
        if sink is not None:
            self._extract_arguments(node, sink)
            self._trace_receiver(node, sink)
        self.generic_visit(node)

    # -- Matching ----------------------------------------------------------

    def _match_sink(self, node: ast.Call) -> SinkPattern | None:
        """Check if *node* is a call to a known statistical method.

        Generic method names (run, query, create, etc.) are only matched
        when called as an attribute (``obj.run(...)``), not as bare function
        calls, to avoid false positives like ``subprocess.run()``.
        """
        func = node.func
        # Method call: obj.method(...)
        if isinstance(func, ast.Attribute):
            method_name = func.attr
            return self._sink_methods.get(method_name)
        # Bare function call: completion(...), invoke(...)
        # Skip generic names to avoid false positives (e.g. `run(cmd)`)
        if isinstance(func, ast.Name):
            if func.id in _GENERIC_SINK_NAMES:
                return None
            return self._sink_methods.get(func.id)
        return None

    # -- Argument extraction -----------------------------------------------

    def _extract_arguments(self, node: ast.Call, sink: SinkPattern) -> None:
        """Extract Name-based arguments from a sink call."""
        line = getattr(node, "lineno", 0)

        # Keyword arguments
        for kw in node.keywords:
            names = _names_in_expr(kw.value)
            for name in names:
                self.sink_args.append(
                    _SinkArgument(
                        var_name=name,
                        param_name=kw.arg,  # None for **kwargs
                        sink_method=sink.method_name,
                        sink_category=sink.category,
                        line=line,
                    )
                )
            # Also check if kwarg value is a literal directly
            if isinstance(kw.value, ast.Constant) and kw.arg:
                self.sink_args.append(
                    _SinkArgument(
                        var_name=f"__literal__{kw.arg}",
                        param_name=kw.arg,
                        sink_method=sink.method_name,
                        sink_category=sink.category,
                        line=line,
                    )
                )

        # Positional arguments
        for arg in node.args:
            names = _names_in_expr(arg)
            for name in names:
                self.sink_args.append(
                    _SinkArgument(
                        var_name=name,
                        param_name=None,
                        sink_method=sink.method_name,
                        sink_category=sink.category,
                        line=line,
                    )
                )

    # -- Constructor tracing -----------------------------------------------

    def _trace_receiver(self, node: ast.Call, sink: SinkPattern) -> None:
        """If the receiver was constructed from a known class, treat its kwargs as sink args."""
        func = node.func
        if not isinstance(func, ast.Attribute):
            return

        receiver = _receiver_name(func.value)
        if receiver is None:
            return

        # Look up the definition of the receiver variable
        defs = self._def_use.definitions.get(receiver, [])
        for defn in defs:
            if defn.value_node is None:
                continue
            if not isinstance(defn.value_node, ast.Call):
                continue
            cls_name = _call_class_name(defn.value_node)
            if cls_name not in self._constructor_hints:
                continue

            # Found: receiver = KnownClass(kwargs...)
            # Extract constructor kwargs as sink arguments.
            ctor_call = defn.value_node
            line = getattr(ctor_call, "lineno", 0)
            for kw in ctor_call.keywords:
                names = _names_in_expr(kw.value)
                for name in names:
                    self.sink_args.append(
                        _SinkArgument(
                            var_name=name,
                            param_name=kw.arg,
                            sink_method=f"{cls_name}.__init__",
                            sink_category=sink.category,
                            line=line,
                        )
                    )
                # Literal constructor kwarg
                if isinstance(kw.value, ast.Constant) and kw.arg:
                    self.sink_args.append(
                        _SinkArgument(
                            var_name=f"__literal__{kw.arg}",
                            param_name=kw.arg,
                            sink_method=f"{cls_name}.__init__",
                            sink_category=sink.category,
                            line=line,
                        )
                    )


# ---------------------------------------------------------------------------
# DataFlowDetectionStrategy
# ---------------------------------------------------------------------------


def _hop_to_confidence(hop: int) -> DetectionConfidence:
    """Map hop distance from sink to confidence level."""
    if hop <= 1:
        return DetectionConfidence.HIGH
    if hop <= 3:
        return DetectionConfidence.MEDIUM
    return DetectionConfidence.LOW


class DataFlowDetectionStrategy:
    """Detect tuned variable candidates via intraprocedural backward slicing.

    Identifies variables that lie on a data-flow path to a statistical call
    site (LLM invocation, retrieval query, embedding call).  Implements the
    ``DetectionStrategy`` protocol.

    Args:
        sink_patterns: Method-name patterns identifying statistical calls.
        constructor_hints: Class names whose constructor kwargs count as
            sink arguments.
        max_hops: Maximum backward-walk depth (prevents unbounded traversal).
    """

    def __init__(
        self,
        *,
        sink_patterns: tuple[SinkPattern, ...] = DEFAULT_SINK_PATTERNS,
        constructor_hints: frozenset[str] = CONSTRUCTOR_CLASS_HINTS,
        max_hops: int = 5,
    ) -> None:
        self._sink_methods: dict[str, SinkPattern] = {
            sp.method_name: sp for sp in sink_patterns
        }
        self._constructor_hints = constructor_hints
        self._max_hops = max_hops

    def detect(
        self,
        source: str,
        function_name: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> list[TunedVariableCandidate]:
        """Detect tuned variable candidates by backward slicing from sinks.

        Args:
            source: Python source code.
            function_name: Target function to analyse.
            context: Optional dict.  Recognised keys:
                - ``existing_tvars``: set of variable names to skip.
                - ``extra_sinks``: list of ``{"method": str, "category": str}``
                  dicts for additional sink patterns.

        Returns:
            List of detected candidates, deduplicated by name.
        """
        existing_tvars: frozenset[str] = frozenset()
        extra_sink_methods: dict[str, SinkPattern] = {}
        if context:
            if "existing_tvars" in context:
                existing_tvars = frozenset(context["existing_tvars"])
            for extra in context.get("extra_sinks", []):
                sp = SinkPattern(extra["method"], extra.get("category", "custom"))
                extra_sink_methods[sp.method_name] = sp

        # Merge default + extra sinks
        sink_methods = {**self._sink_methods, **extra_sink_methods}

        # 1. Parse and find target function
        try:
            tree = ast.parse(source)
        except SyntaxError:
            logger.warning("Failed to parse source for data-flow detection")
            return []

        func_node = self._find_function(tree, function_name)
        if func_node is None:
            return []

        # 2. Build def-use map
        def_use = _build_def_use_map(func_node)

        # 3. Identify sinks and their arguments
        finder = _SinkFinder(sink_methods, self._constructor_hints, def_use)
        finder.visit(func_node)
        sink_args = finder.sink_args

        if not sink_args:
            return []

        # 4. Backward walk
        candidates = self._backward_walk(sink_args, def_use, func_node, existing_tvars)

        return candidates

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _find_function(
        tree: ast.Module, name: str
    ) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == name:
                    return node
        return None

    def _backward_walk(
        self,
        sink_args: list[_SinkArgument],
        def_use: _DefUseMap,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        existing_tvars: frozenset[str],
    ) -> list[TunedVariableCandidate]:
        """Walk backward through def-use chains from sink arguments.

        Uses BFS (FIFO) to guarantee shortest-hop-first traversal, so each
        variable gets its best (lowest hop) confidence assignment.
        """
        best: dict[str, TunedVariableCandidate] = {}
        visited: set[str] = set()
        worklist = self._init_worklist(sink_args, func_node, best, existing_tvars)

        while worklist:
            var_name, hop, sink_info = worklist.popleft()

            if var_name in visited or hop > self._max_hops:
                continue
            visited.add(var_name)

            if var_name in existing_tvars:
                continue

            self._process_definitions(
                var_name, hop, sink_info, def_use, visited, worklist, best
            )
            self._process_param(var_name, sink_info, def_use, func_node, best)

        return list(best.values())

    def _init_worklist(
        self,
        sink_args: list[_SinkArgument],
        func_node: ast.AST,
        best: dict[str, TunedVariableCandidate],
        existing_tvars: frozenset[str],
    ) -> deque[tuple[str, int, str]]:
        """Build the initial worklist from sink arguments."""
        worklist: deque[tuple[str, int, str]] = deque()
        for sa in sink_args:
            if sa.var_name.startswith("__literal__"):
                # Filter existing_tvars for literal kwargs too
                if sa.param_name and sa.param_name in existing_tvars:
                    continue
                self._emit_literal_kwarg(sa, func_node, best)
            else:
                worklist.append(
                    (sa.var_name, 0, f"{sa.sink_method}({sa.param_name or '?'})")
                )
        return worklist

    def _process_definitions(
        self,
        var_name: str,
        hop: int,
        sink_info: str,
        def_use: _DefUseMap,
        visited: set[str],
        worklist: deque[tuple[str, int, str]],
        best: dict[str, TunedVariableCandidate],
    ) -> None:
        """Process all definitions of a variable in the def-use map."""
        for defn in def_use.definitions.get(var_name, []):
            if defn.value_node is None:
                continue
            lit = _extract_literal_value(defn.value_node)
            if lit is not None:
                self._emit_candidate(var_name, lit, defn, hop, sink_info, best)
            else:
                for dep in _names_in_expr(defn.value_node):
                    if dep not in visited:
                        worklist.append((dep, hop + 1, sink_info))

    @staticmethod
    def _process_param(
        var_name: str,
        sink_info: str,
        def_use: _DefUseMap,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        best: dict[str, TunedVariableCandidate],
    ) -> None:
        """Emit a candidate for a function parameter reaching a sink."""
        if var_name not in def_use.func_params or var_name in best:
            return
        cand = TunedVariableCandidate(
            name=var_name,
            candidate_type=CandidateType.CATEGORICAL,
            confidence=DetectionConfidence.MEDIUM,
            location=SourceLocation(
                line=func_node.lineno,
                col_offset=func_node.col_offset,
            ),
            detection_source="dataflow",
            reasoning=f"Function parameter '{var_name}' flows to {sink_info}",
            canonical_name=_REVERSE_MAPPING.get(var_name),
        )
        DataFlowDetectionStrategy._update_best(best, cand)

    def _emit_candidate(
        self,
        var_name: str,
        value: Any,
        defn: _Definition,
        hop: int,
        sink_info: str,
        best: dict[str, TunedVariableCandidate],
    ) -> None:
        """Create and register a candidate from a literal definition."""
        confidence = _hop_to_confidence(hop)
        canonical = _REVERSE_MAPPING.get(var_name)
        ctype = _infer_candidate_type(value)
        suggested = _suggest_range(canonical, ctype, value)

        cand = TunedVariableCandidate(
            name=var_name,
            candidate_type=ctype,
            confidence=confidence,
            location=_make_location(defn.stmt_node),
            current_value=value,
            suggested_range=suggested,
            detection_source="dataflow",
            reasoning=f"'{var_name}' flows to {sink_info} (hop={hop})",
            canonical_name=canonical,
        )
        self._update_best(best, cand)

    def _emit_literal_kwarg(
        self,
        sa: _SinkArgument,
        func_node: ast.AST,
        best: dict[str, TunedVariableCandidate],
    ) -> None:
        """Handle a literal value passed directly as a kwarg to a sink call."""
        param = sa.param_name
        if param is None:
            return

        # Walk the function body to find the actual Call node and extract the literal
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node, "lineno", 0) != sa.line:
                continue
            for kw in node.keywords:
                if kw.arg == param and isinstance(kw.value, ast.Constant):
                    value = kw.value.value
                    canonical = _REVERSE_MAPPING.get(param)
                    ctype = _infer_candidate_type(value)
                    suggested = _suggest_range(canonical, ctype, value)

                    cand = TunedVariableCandidate(
                        name=param,
                        candidate_type=ctype,
                        confidence=DetectionConfidence.HIGH,
                        location=_make_location(kw),
                        current_value=value,
                        suggested_range=suggested,
                        detection_source="dataflow",
                        reasoning=(
                            f"Literal kwarg '{param}={value!r}' "
                            f"passed directly to {sa.sink_method}"
                        ),
                        canonical_name=canonical,
                    )
                    self._update_best(best, cand)
                    return

    @staticmethod
    def _update_best(
        best: dict[str, TunedVariableCandidate],
        cand: TunedVariableCandidate,
    ) -> None:
        """Keep the highest-confidence candidate per variable name."""
        _order = {
            DetectionConfidence.LOW: 0,
            DetectionConfidence.MEDIUM: 1,
            DetectionConfidence.HIGH: 2,
        }
        existing = best.get(cand.name)
        if existing is None or _order[cand.confidence] > _order[existing.confidence]:
            best[cand.name] = cand
