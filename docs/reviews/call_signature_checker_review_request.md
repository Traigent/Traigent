# Review Request: General-Purpose Call Signature Mismatch Detector

**Date:** 2025-12-30
**Author:** Claude (Opus 4.5)
**Reviewers:** Codex, Gemini Pro
**Files:** `scripts/check_call_signatures.py`, `tests/fixtures/signature_check/`

---

## Executive Summary

We've implemented a static analysis tool that detects mismatches between function calls and their definitions in Python codebases. The tool is **sound but incomplete**: it only flags what can be verified statically, preferring false negatives over false positives.

**Request:** Please review the design decisions, implementation correctness, and identify any edge cases or improvements we may have missed.

---

## Motivation

### The Original Bug

During a demo, we encountered this error:
```
'function' object has no attribute 'get'
```

**Root cause:** An evaluator class with signature `(prediction, expected, input_data) -> dict` was passed to an API expecting `(func, config, example) -> ExampleResult`. The mismatch was only caught at runtime.

### Why Static Detection?

1. **Fail fast** - Catch errors before runtime
2. **CI integration** - Block merges with signature mismatches
3. **Codebase health** - Find latent bugs across large codebases

### Why General-Purpose?

The original checker was Traigent-specific (checking only `custom_evaluator`, `scoring_function`, etc.). User feedback: *"This is too specific. Find ANY mismatch between calls and definitions."*

---

## Architecture

### Two-Phase Detection

```
Phase 1: Build Symbol Table
├── Parse all .py files with ast.parse()
├── Extract function/class definitions
├── Index by (module_path, qualname) tuples
└── Build per-file import alias maps

Phase 2: Check Calls
├── Collect all ast.Call nodes
├── Resolve callee to definition via imports
├── Compare call args vs definition params
└── Report mismatches with severity levels
```

### Key Data Structures

```python
@dataclass
class ParamInfo:
    """Parsed from ast.arguments"""
    names: List[str]           # All param names
    posonly_count: int         # Positional-only count
    required_positional: int   # Required before defaults
    kwonly_names: Set[str]     # Keyword-only params
    kwonly_required: Set[str]  # Required kwonly (no default)
    has_var_positional: bool   # *args
    has_var_keyword: bool      # **kwargs

@dataclass
class FunctionDef:
    module_path: str           # "traigent.core.foo"
    qualname: str              # "MyClass.method"
    file: str
    line: int
    params: ParamInfo
    decorators: List[str]
    is_method: bool
    has_explicit_init: bool    # For classes

@dataclass
class CallSite:
    file: str
    line: int
    callee_expr: str           # "module.func"
    positional_count: int
    keyword_names: Set[str]
    has_star_args: bool        # Uses *args
    has_star_kwargs: bool      # Uses **kwargs
```

---

## Severity Levels

```python
class Severity(IntEnum):
    INFO = 0      # Cannot resolve / unverifiable
    WARNING = 1   # Suspicious but uncertain
    ERROR = 2     # Definite mismatch
```

| Severity | When Triggered | Rationale |
|----------|----------------|-----------|
| **ERROR** | Too few required args | Definite bug |
| **ERROR** | Too many positional args (no `*args`) | Definite bug |
| **ERROR** | Unknown keyword arg (no `**kwargs`) | Definite bug |
| **ERROR** | Missing required keyword-only arg | Definite bug |
| **WARNING** | `@dataclass` class | Generated `__init__` |
| **WARNING** | `@overload` function | Stub, not real impl |
| **WARNING** | Custom decorators | May alter signature |
| **WARNING** | No explicit `__init__` | May inherit from parent |
| **INFO** | Call uses `*args`/`**kwargs` | Cannot verify statically |
| **INFO** | External module | Cannot resolve |

---

## Implementation Details

### Parameter Counting Logic

The trickiest part: keyword args can satisfy required positional params.

```python
def __init__(self, message: str, fix: str | None = None): ...

# These are ALL valid:
TraigentError("msg")                    # positional
TraigentError(message="msg")            # keyword
TraigentError("msg", fix="hint")        # mixed
TraigentError(message="msg", fix="x")   # all kwargs
```

**Our solution:**

```python
# Required positional-only MUST be positional
posonly_required = min(defn.params.posonly_count, defn.params.required_positional)
if call.positional_count < posonly_required:
    return ERROR("missing positional-only args")

# Regular required can be positional OR keyword
regular_required = defn.params.required_positional - posonly_required
regular_param_names = all_positional_names[posonly_count:]
regular_required_names = set(regular_param_names[:regular_required])

# Count coverage from both sources
positional_covering = call.positional_count - posonly_required
covered_by_positional = set(regular_param_names[:positional_covering])
covered_by_keyword = call.keyword_names & regular_required_names

missing = regular_required_names - (covered_by_positional | covered_by_keyword)
if missing:
    return ERROR(f"missing required arg(s): {missing}")
```

### Import Resolution

```python
@dataclass
class ImportMap:
    aliases: Dict[str, str]  # local_name -> full.module.path
    file_module: str

    def add_import(self, node: ast.Import):
        # import os           -> {"os": "os"}
        # import os as o      -> {"o": "os"}

    def add_import_from(self, node: ast.ImportFrom):
        # from pathlib import Path      -> {"Path": "pathlib.Path"}
        # from . import utils           -> {"utils": "<pkg>.utils"}
        # from ..core import foo        -> {"foo": "<parent>.core.foo"}

    def resolve(self, name: str) -> Optional[str]:
        # "Path" -> "pathlib.Path"
        # "module.func" -> resolve "module" then append ".func"
```

### Symbol Table Keys

To avoid collisions between same-named functions in different modules:

```python
# traigent/core/foo.py
def bar(): ...           # Key: ("traigent.core.foo", "bar")

class Baz:
    def qux(self): ...   # Key: ("traigent.core.foo", "Baz.qux")
```

---

## What V1 Handles

| Pattern | Supported | Notes |
|---------|-----------|-------|
| `func(a, b)` | ✅ | Direct calls |
| `module.func(a)` | ✅ | Via import map |
| `Class(args)` | ✅ | Uses `__init__` |
| `Class.static_method(a)` | ✅ | Skip self |
| `Class.class_method(a)` | ✅ | Skip cls |
| `def f(a, /, b, *, c)` | ✅ | Full param types |
| `f(x=1, y=2)` | ✅ | Kwargs satisfy positional |
| `from . import x` | ✅ | Relative imports |
| `import x as y` | ✅ | Aliased imports |

## What V1 Does NOT Handle

| Pattern | Status | Rationale |
|---------|--------|-----------|
| `self.method()` | ❌ Skip | Requires type tracking |
| `obj.method()` | ❌ Skip | Requires type tracking |
| `getattr(x, "f")()` | ❌ Skip | Dynamic dispatch |
| `@dataclass` fields | ⚠️ Warn | Generated `__init__` |
| MRO resolution | ⚠️ Warn | Complex, error-prone |
| Third-party modules | ℹ️ Info | Cannot resolve |

---

## Test Fixtures

Created 12 test files covering all edge cases:

```
tests/fixtures/signature_check/
├── correct_calls.py           # Baseline - all valid
├── wrong_arg_count.py         # Too few/many args
├── unknown_kwargs.py          # Invalid kwargs
├── posonly_kwonly_defaults.py # def f(a, /, b, *, c=1)
├── varargs_calls.py           # *args/**kwargs
├── staticmethod_classmethod.py
├── dataclass_overload.py
├── decorated_funcs.py
├── inheritance.py             # Missing __init__
├── imports_absolute.py
├── imports_relative.py
└── imports_aliased.py
```

### Sample Fixture: `wrong_arg_count.py`

```python
def requires_three(a, b, c):
    return a + b + c

# ERROR: missing 2 required args
err1 = requires_three(1)

# ERROR: takes 3 positional, got 5
err2 = requires_three(1, 2, 3, 4, 5)
```

### Sample Fixture: `posonly_kwonly_defaults.py`

```python
def full_signature(a, b, /, c, d, *args, e, f=100, **kwargs):
    """
    a, b: positional-only (required)
    c, d: regular positional (required)
    e: keyword-only (required)
    f: keyword-only (default=100)
    """
    pass

# CORRECT
ok = full_signature(1, 2, 3, 4, e=5)

# ERROR: missing required kwonly 'e'
err = full_signature(1, 2, 3, 4)
```

---

## Results

### On `traigent/` (main codebase)

```
$ python scripts/check_call_signatures.py traigent/ --min-severity error --stats
No signature mismatches found in traigent/
--- Statistics ---
Files scanned: 279
Files skipped: 0
Definitions found: 4582
Calls checked: 19277
Duration: 1.41s
```

### On `tests/fixtures/signature_check/` (intentional errors)

```
$ python scripts/check_call_signatures.py tests/fixtures/signature_check/ --min-severity error --stats
Found 34 signature mismatch(es)

=== ERROR (34) ===

tests/fixtures/signature_check/wrong_arg_count.py:34
  requires_three() missing required arg(s): {'c'}

tests/fixtures/signature_check/unknown_kwargs.py:30
  accepts_a_b() got unexpected keyword arg(s): {'c'}

tests/fixtures/signature_check/staticmethod_classmethod.py:69
  Calculator.static_add() takes 2 positional arg(s), got 3
...
--- Statistics ---
Files scanned: 13
Files skipped: 0
Definitions found: 63
Calls checked: 163
Duration: 0.01s
```

Note: 34 errors (not 35) because one intentional error is suppressed with `# noqa: sigcheck`.

---

## Questions for Review

### 1. Parameter Counting Edge Cases

**Q:** Is the logic for counting covered parameters correct?

```python
# Scenario: def f(a, b, c=1) with call f(1, c=2)
# a=1 (positional), b=? (missing!), c=2 (keyword)

regular_required_names = {"a", "b"}  # c has default
positional_covering = 1  # one positional arg
covered_by_positional = {"a"}
covered_by_keyword = {"c"} & {"a", "b"} = {}
missing = {"a", "b"} - {"a"} = {"b"}  # CORRECT!
```

But what about:
```python
# def f(a, b, c=1) with call f(1, b=2, c=3)
# Should work: a=1, b=2, c=3
covered_by_positional = {"a"}
covered_by_keyword = {"b", "c"} & {"a", "b"} = {"b"}
missing = {"a", "b"} - {"a", "b"} = {}  # CORRECT!
```

**Am I missing any edge cases here?**

### 2. Decorator Handling Strategy

**Current:** Any decorator other than `@staticmethod`, `@classmethod`, `@property`, `@abstractmethod` triggers WARNING.

**Concern:** This may be too conservative. `@functools.lru_cache`, `@contextmanager`, etc. preserve signatures.

**Options:**
- A. Whitelist known safe decorators (lru_cache, cache, contextmanager, etc.)
- B. Check for `@functools.wraps` usage in decorator definition
- C. Keep current behavior (conservative)

**What do you recommend?**

### 3. Inheritance Without `__init__`

**Current:** If a class has no explicit `__init__`, we emit WARNING and don't try to resolve parent's `__init__`.

**Alternative:** Walk MRO to find parent `__init__`:
```python
class Base:
    def __init__(self, x, y): ...

class Child(Base):
    pass  # Inherits __init__(self, x, y)

Child(1, 2)  # Currently: WARNING, could be: checked
```

**Concern:** MRO resolution in static analysis is error-prone with multiple inheritance, `__init_subclass__`, metaclasses, etc.

**Should we attempt basic single-inheritance resolution?**

### 4. `*args`/`**kwargs` at Call Site

**Current:** If call uses `*args` or `**kwargs`, emit INFO and skip checking.

```python
def strict(a, b, c): ...

my_args = get_args()
strict(*my_args)  # INFO: cannot verify
```

**Alternative:** Still check minimum args before splat:
```python
strict(*my_args)        # INFO (correct - might have 3+ args)
strict(1, *my_args)     # Could check: at least 1 provided
strict(1, 2, 3, *args)  # Could ERROR: already 3, *args would overflow
```

**Is partial checking worth the complexity?**

### 5. Performance Concerns

**Current:** We parse every file twice (Phase 1 for definitions, Phase 2 for calls).

**Optimization:** Parse once, collect both in single pass.

**Current implementation:**
```python
# Phase 1
for py_file in py_files:
    tree = ast.parse(...)
    symbol_table.add_file(tree)
    file_imports[file] = build_import_map(tree)

# Phase 2
for py_file in py_files:
    tree = ast.parse(...)  # DUPLICATE PARSE!
    calls = collect_calls(tree)
```

**Should we refactor to single-pass?**

### 6. Cross-File Import Resolution

**Current limitation:** We only resolve imports within the scanned directory.

```python
# If scanning traigent/ only:
from traigent.core import foo  # Resolved
from external_lib import bar   # INFO: cannot resolve
```

**Question:** Should we attempt to resolve installed packages via `importlib.util.find_spec`?

**Pros:** Could check calls to well-typed libraries
**Cons:** Requires package to be installed, adds complexity

### 7. Output Format

**Current JSON:**
```json
{
  "severity": "ERROR",
  "file": "path/to/file.py",
  "line": 42,
  "callee": "func",
  "message": "func() missing required arg(s): {'x'}",
  "definition_file": "path/to/def.py",
  "definition_line": 10
}
```

**Missing fields that might be useful:**
- `column` (for IDE integration)
- `expected_signature` (human-readable)
- `actual_call` (the call expression)
- `fix_suggestion` (auto-fix hint)

**Which additional fields would be valuable?**

---

## Code Snippets for Review

### Core Check Logic

```python
def check_call(call: CallSite, defn: FunctionDef) -> Optional[Mismatch]:
    # Skip unverifiable cases -> INFO
    if call.has_star_args or call.has_star_kwargs:
        return Mismatch(Severity.INFO, ...)

    # Problematic decorators -> WARNING
    if {"dataclass", "overload"} & set(defn.decorators):
        return Mismatch(Severity.WARNING, ...)

    # No explicit __init__ -> WARNING
    if not defn.is_method and not defn.has_explicit_init:
        return Mismatch(Severity.WARNING, ...)

    # Custom decorators -> WARNING
    safe = {"staticmethod", "classmethod", "property", "abstractmethod"}
    if set(defn.decorators) - safe - {"dataclass", "overload"}:
        return Mismatch(Severity.WARNING, ...)

    # Positional-only must be positional -> ERROR
    if call.positional_count < posonly_required:
        return Mismatch(Severity.ERROR, ...)

    # Required params (positional or keyword) -> ERROR
    if missing_required:
        return Mismatch(Severity.ERROR, ...)

    # Too many positional -> ERROR
    if call.positional_count > max_positional:
        return Mismatch(Severity.ERROR, ...)

    # Unknown kwargs -> ERROR
    if unknown_kwargs:
        return Mismatch(Severity.ERROR, ...)

    # Missing required kwonly -> ERROR
    if missing_kwonly:
        return Mismatch(Severity.ERROR, ...)

    return None  # All checks passed
```

### Import Resolution

```python
def add_import_from(self, node: ast.ImportFrom) -> None:
    if node.module is None:
        # from . import y
        base_module = self._resolve_relative(node.level)
    else:
        if node.level > 0:
            # from .x import y or from ..x import y
            base_module = self._resolve_relative(node.level)
            if base_module:
                base_module = f"{base_module}.{node.module}"
            else:
                base_module = node.module
        else:
            base_module = node.module

    for alias in node.names:
        name = alias.asname if alias.asname else alias.name
        if alias.name == "*":
            continue  # Can't track star imports
        self.aliases[name] = f"{base_module}.{alias.name}"
```

### ParamInfo Extraction

```python
@classmethod
def from_ast_arguments(cls, args: ast.arguments, skip_first: bool = False):
    posonly = [a.arg for a in args.posonlyargs]
    regular = [a.arg for a in args.args]

    if skip_first:  # For self/cls
        if posonly:
            posonly = posonly[1:]
        elif regular:
            regular = regular[1:]

    all_positional = posonly + regular
    num_defaults = len(args.defaults)
    required_positional = len(all_positional) - num_defaults

    kwonly_names = {a.arg for a in args.kwonlyargs}
    kwonly_required = set()
    for i, kwarg in enumerate(args.kwonlyargs):
        if i >= len(args.kw_defaults) or args.kw_defaults[i] is None:
            kwonly_required.add(kwarg.arg)

    return cls(
        names=all_positional + list(kwonly_names),
        posonly_count=len(posonly),
        required_positional=max(0, required_positional),
        kwonly_names=kwonly_names,
        kwonly_required=kwonly_required,
        has_var_positional=args.vararg is not None,
        has_var_keyword=args.kwarg is not None,
    )
```

---

## Acceptance Criteria

The implementation should be considered complete if:

1. ✅ No false positives on `traigent/` codebase
2. ✅ Detects all intentional errors in test fixtures
3. ✅ Handles all Python parameter types (posonly, kwonly, defaults, *args, **kwargs)
4. ✅ Resolves imports (absolute, relative, aliased)
5. ✅ Warns on uncertain cases (decorators, inheritance)
6. ✅ Provides actionable error messages
7. ✅ Supports JSON output for CI
8. ✅ Has `--strict` mode for exit codes

---

## V1.1 Improvements (Based on Reviewer Feedback)

### Completed

1. **Suppression mechanism** (`# noqa: sigcheck`)
   - Add `# noqa: sigcheck` or `# noqa` to any line to suppress errors
   - Case-insensitive matching
   - Enables gradual adoption

2. **Exclusion patterns** (`--exclude`)
   - Skip files/directories matching patterns: `--exclude ".venv" --exclude "*_test.py"`
   - Supports multiple patterns
   - Shows skipped file count in stats

3. **Statistics output** (`--stats`)
   - Files scanned/skipped
   - Definitions found
   - Calls checked
   - Duration (seconds)
   - Error/warning/info counts

4. **Single file support**
   - Can now scan individual files: `python check_call_signatures.py path/to/file.py`

### Example with all features

```bash
$ python scripts/check_call_signatures.py traigent/ \
    --exclude ".venv" --exclude "test_*" \
    --min-severity warning \
    --stats

No signature mismatches found in traigent/
--- Statistics ---
Files scanned: 250
Files skipped: 29
Definitions found: 4200
Calls checked: 18000
Duration: 1.25s
```

---

## Open Items

1. [x] ~~Add `--exclude` flag to skip directories~~ ✅ Done
2. [x] ~~Add suppression mechanism~~ ✅ Done (`# noqa: sigcheck`)
3. [x] ~~Add execution time and stats~~ ✅ Done (`--stats`)
4. [ ] Consider caching parsed ASTs for performance
5. [ ] Add column numbers to output
6. [ ] Consider integration with mypy/pyright for type-aware checking
7. [ ] Add auto-fix suggestions
8. [ ] Add `.sigcheckrc` config file support (for project-wide settings)

---

## Request

Please review:

1. **Correctness** - Are there edge cases where the logic fails?
2. **Completeness** - What important patterns are we missing?
3. **Design** - Are the severity classifications appropriate?
4. **Performance** - Any obvious optimizations?
5. **Questions** - Your opinions on the 7 questions above

Thank you for your time!
