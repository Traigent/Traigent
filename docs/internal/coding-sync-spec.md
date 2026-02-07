1. Motivation (why this exists)

Modern LLM coding assistants are good at spitting out code, but bad at:

* Knowing *why* the code exists.
* Adding features **incrementally** without breaking old ones.
* Showing a clear path from a **requirement** to the **code** and the **runtime behavior**.

Meng & Jackson’s “What You See Is What It Does” paper calls this problem **illegibility**, and argues that robust LLM coding needs:

* **Incrementality** – small features via small, local changes.
* **Integrity** – new changes don’t silently break old ones.
* **Transparency** – clear mapping from requirements → design → code → runtime flows.

This guide describes a system that:

1. Represents **requirements**, **functionalities**, **concepts**, **synchronizations**, and **code** as explicit, linked artifacts.
2. Automatically analyzes and links those layers (static and runtime).
3. Integrates with LLMs so that generated code is:

   * justified by requirements/functionality,
   * consistent with design,
   * and debuggable from runtime traces.

---

# 1. Success criteria

You know this thing “works” when all are true:

### 1.1 Traceability & coverage

* For **every requirement** (`REQ-*`), you can query:

  * linked functionalities (`FUNC-*`),
  * concepts (`CONC-*`) and synchronizations (`SYNC-*`) that realize it,
  * code units (functions/classes/files) implementing those,
  * and tests / runtime flows exercising them.

* For **every code unit**, you can answer:

  * “Which functionality/requirements justify this code?”
  * “Is it concept logic, synchronization glue, infra, or util?”

* Coverage metrics are tracked:

  * % of `REQ-*` linked to at least one `FUNC-*`.
  * % of `FUNC-*` linked to at least one `CONC-*` or `SYNC-*`.
  * % of code units linked to some concept/sync/func, or explicitly tagged `infra/util`.

### 1.2 Design integrity

* Concept code **never** directly calls other concepts’ internals or DB tables.
* Cross‑concept behavior is **factored into synchronizations**, not scattered in random controllers/services.
* Static checks can list:

  * concept boundary violations,
  * syncs that mutate concept state directly (instead of via actions),
  * “spider” modules with too many responsibilities.

### 1.3 Incremental changes

For a new feature:

* You can see a short list of **touched or added**:

  * requirements, functionalities, concepts/syncs, and code units.
* The diff surface is small:

  * mostly new syncs / small concept updates,
  * not hundreds of unrelated files.
* CI can fail if:

  * coverage regresses,
  * boundary rules are violated,
  * or new “orphan” code/requirements appear.

### 1.4 Runtime transparency

* Every request / job gets a **flow ID**.
* You can reconstruct a **human‑readable narrative** for a flow:

  * “Web/request(register) → Password.validate → User.register → Profile.register → JWT.generate → Web.respond …”
* Given a bug, you can:

  * find the flow,
  * see which synchronizations fired,
  * jump directly to the specs + code that produced it.

### 1.5 LLM integration

* LLMs mostly work with **structured specs** (concepts + syncs + requirements), not just raw code.
* Generated code is annotated with IDs and passes static traceability checks.
* For a failing test/flow, you can hand an LLM:

  * a slice of the action graph + relevant syncs/specs,
  * and get a plausible fix proposal that’s localized.

---

# 2. Core model (entities and relationships)

The system revolves around these node types:

* `Requirement` – “what we want” (`REQ-*`).
* `Functionality` – features / user stories (`FUNC-*`).
* `Concept` – user‑facing unit of behavior with its own state & actions (`CONC-*`).
* `Action` – operation of a concept (`CONC-Password.check`).
* `Synchronization` – rule “when these actions happen, under these conditions, then call these actions” (`SYNC-*`).
* `CodeUnit` – concrete implementation (file/class/function).
* `TestCase` – test identifiers (unit / integration / e2e).
* `ActionRecord` – runtime occurrence of an action/sync.
* `Flow` – group of ActionRecords sharing the same flow ID (usually 1 external request).

Key edges:

* Requirement → Functionality (`satisfied_by`)
* Functionality → Concept (`realized_by`)
* Functionality → Synchronization (`implemented_by`)
* Concept → Action (`has_action`)
* Synchronization → Action (`reads`/`invokes`)
* CodeUnit → Action or Synchronization (`implements`)
* TestCase → CodeUnit / Flow (`exercises`)
* ActionRecord → Synchronization (`triggered_by`)
* ActionRecord → Flow (`belongs_to`)

The actual storage format doesn’t have to be a graph DB; we can represent these links in YAML/JSON and derive views as needed.

---

# 3. File layout & schemas

Assume repo layout like:

```text
docs/
  requirements.yml
  functionalities.yml
  concepts/
    *.yml
  syncs/
    *.yml
  ai_assistant.md        # “how to work with this architecture” for LLMs/humans
reports/
  code_summaries.json
  trace_links.json
  gaps.md
  design_view.md
runtime/
  traces/                # DB or JSONL for action graphs
src/
  ...                    # Python/JS source
tests/
  ...
tools/
  trace_sync/            # Python CLI package implementing this spec
```

### 3.1 `docs/requirements.yml`

Schema (YAML list):

```yaml
- id: REQ-REG-001
  title: "User can register with email, username, password"
  description: >
    User provides email, username, and password and receives an auth token
    on successful registration.
  acceptance_criteria:
    - "On success, response includes token and profile data."
    - "On invalid password, user sees a clear error and no account is created."
  tags: [registration, auth]
  priority: high
  type: functional   # functional | non-functional
  status: active     # active | planned | deprecated
```

### 3.2 `docs/functionalities.yml`

```yaml
- id: FUNC-REG-REGISTER
  name: "Registration (happy path)"
  description: "Register a new user and log them in immediately."
  requirements: [REQ-REG-001]
  concepts: [CONC-User, CONC-Password, CONC-Profile, CONC-JWT]
  syncs: [SYNC-Registration, SYNC-NewUserToken, SYNC-RegistrationResponse]
  tags: [registration, backend]
  layer: feature      # epic | feature | subfeature | infra
  status: active
```

### 3.3 `docs/concepts/*.yml`

Concept spec schema:

```yaml
id: CONC-Password
name: Password
purpose: "Securely store and validate user passwords."
type_params: [UserId]     # optional
state:
  - name: password
    type: "mapping[UserId, string]"
    description: "Hashed password per user"
  - name: salt
    type: "mapping[UserId, string]"
    description: "Salt per user"
actions:
  - name: set
    inputs:
      - name: user
        type: "UserId"
      - name: password
        type: "string"
    outputs:
      - case: success
        fields:
          - name: user
            type: "UserId"
      - case: error
        fields:
          - name: error
            type: "string"
    description: "Set password for user, enforcing password policy."
  - name: check
    inputs:
      - name: user
        type: "UserId"
      - name: password
        type: "string"
    outputs:
      - case: result
        fields:
          - name: valid
            type: "boolean"
      - case: error
        fields:
          - name: error
            type: "string"
operational_principle:
  - "After a successful set(user,x), check(user,x) must return valid=true."
  - "After a successful set(user,x), check(user,y!=x) must return valid=false."
version: 1
status: active
```

This follows the structure in the paper (purpose / state / actions / operational principle) but serialized as YAML.

### 3.4 `docs/syncs/*.yml`

Synchronization spec schema:

```yaml
id: SYNC-Registration
name: "Registration core"
functionalities: [FUNC-REG-REGISTER]
when:
  - concept: CONC-Web
    action: request
    match:
      method: "register"
      bind:
        request: "?request"
        username: "?username"
        email: "?email"
        password: "?password"
where:
  - bind:
      user: "uuid()"   # arbitrary expression, executed in sync engine
then:
  - concept: CONC-User
    action: register
    args:
      user: "?user"
      name: "?username"
      email: "?email"
version: 1
status: active
```

The DSL is “when / where / then”, similar to the paper’s examples, but encoded as YAML.

### 3.5 `reports/code_summaries.json`

Array of objects:

```json
[
  {
    "file": "src/backend/auth/password_service.py",
    "symbol": "set_password",
    "kind": "function",
    "lang": "python",
    "role": "concept_core",  // concept_core | sync_impl | infra | util
    "concept_id": "CONC-Password",
    "action_name": "set",
    "sync_id": null,
    "signature": "set_password(user_id: str, password: str) -> PasswordResult",
    "summary": "Implements Password.set: hashes password, stores salt, and returns result.",
    "inputs": ["user_id", "password"],
    "outputs": ["PasswordResult(success | error)"],
    "side_effects": ["writes to password table", "logs events"],
    "dependencies": ["hashlib", "sqlalchemy", "app.db.password_table"],
    "error_handling": "Raises ValueError for invalid password; catches DB errors and wraps.",
    "performance_notes": "Single DB write per call.",
    "suggested_functionalities": [
      {"id": "FUNC-REG-REGISTER", "confidence": 0.7}
    ]
  }
]
```

### 3.6 `reports/trace_links.json`

Links between layers:

```json
[
  {
    "code_unit": "src/backend/auth/password_service.py::set_password",
    "concept_id": "CONC-Password",
    "action_name": "set",
    "sync_id": null,
    "functionalities": ["FUNC-REG-REGISTER"],
    "requirements": ["REQ-REG-001"],
    "source": "static+llm",
    "confidence": 0.92,
    "status": "confirmed",   // confirmed | proposed | rejected | stale
    "last_checked": "2025-11-21"
  }
]
```

### 3.7 Runtime traces

Represent runtime events as JSONL (or DB rows):

```json
{
  "id": "act-1234",
  "kind": "concept_action",     // concept_action | sync
  "concept_id": "CONC-Password",
  "action_name": "set",
  "sync_id": "SYNC-Registration",
  "flow_id": "flow-9876",
  "input": {"user": "user-123", "password": "***"},
  "output": {"case": "error", "error": "too_short"},
  "timestamp": "2025-11-21T12:34:56Z",
  "code_unit": "src/backend/auth/password_service.py::set_password",
  "version": "app-1.4.2"
}
```

---

# 4. Static tooling: the `trace_sync` package

Implement as a Python package (even if JS is in the repo) to drive the analysis/CLI.

## 4.1 Responsibilities

`trace_sync` must:

1. Load **requirements**, **functionalities**, **concepts**, **syncs**.
2. Walk `src/` and extract **CodeUnits** (Python + JS).
3. Classify each CodeUnit (`concept_core`, `sync_impl`, `infra`, `util`).
4. Summarize each CodeUnit (optionally via LLM).
5. Link CodeUnits ↔ {Concepts, Actions, Syncs, Functionalities, Requirements}.
6. Run design checks:

   * boundary violations,
   * orphans,
   * redundancy.
7. Emit reports + JSON data files.
8. Provide a CLI + “tools” endpoints for LLMs.

## 4.2 Package structure

Suggested:

```text
tools/trace_sync/
  __init__.py
  cli.py
  config.py
  models.py             # dataclasses for Requirement, Concept, Sync, CodeUnit, etc.
  specs_loader.py       # read YAML specs
  code_scan/
    __init__.py
    python_scanner.py
    js_scanner.py
  summarizer/
    __init__.py
    llm_client.py       # abstract; use environment-specific caller
    summarization.py
  linker/
    __init__.py
    static_hints.py
    similarity.py
    linking_pipeline.py
  checks/
    orphans.py
    boundaries.py
    redundancy.py
  reports/
    markdown.py
    json_export.py
```

## 4.3 CLI commands

Expose at least:

```bash
trace-sync scan       # full scan + summaries + linking + reports
trace-sync scan --changed-files <paths...>   # incremental
trace-sync report     # regenerate markdown reports from existing JSON
trace-sync check-ci   # run all checks, exit non-zero on violations
trace-sync explain-code <code_unit_id>
trace-sync explain-req <REQ-ID>
trace-sync explain-flow <flow_id>   # uses runtime traces if available
```

Each command returns machine-readable output (JSON) plus human-friendly text.

---

# 5. Static analysis pipeline (step-by-step)

### Step 1 – Load specs

* Parse `docs/requirements.yml`, `docs/functionalities.yml`.
* Load all `docs/concepts/*.yml`, `docs/syncs/*.yml`.
* Validate:

  * ID uniqueness.
  * References (e.g. functionality references existing concepts/syncs).
* Build in-memory indexes:

  * `requirements_by_id`
  * `functionalities_by_id`
  * `concepts_by_id`
  * `syncs_by_id`
  * inverse mappings (e.g. concept → funcs, sync → funcs).

### Step 2 – Scan code (`src/`)

Python:

* Use `ast` module to parse files.
* Extract:

  * top-level functions,
  * classes and methods,
  * docstrings,
  * imports,
  * simple call graph edges.

JS/TS:

* Use a JS parser (e.g., `esprima`, `acorn`, or TypeScript compiler) via subprocess or existing library.
* Extract similarly:

  * functions, classes, methods,
  * JSDoc comments,
  * imports/exports,
  * call graph hints.

For each code symbol, construct a `CodeUnit`:

```python
CodeUnit(
    id="src/backend/auth/password_service.py::set_password",
    file="src/backend/auth/password_service.py",
    symbol="set_password",
    kind="function",
    lang="python",
    ast=...,            # internal only
    docstring="...",
    imports=[...],
    calls=[...]
)
```

### Step 3 – Classify CodeUnits (`role`)

Heuristics (can be improved over time):

* If file path matches `concepts/<concept_name>.*` OR docstring has `Concept: CONC-*` → `role="concept_core"`.
* If file path matches `syncs/*` OR docstring has `Sync: SYNC-*` → `role="sync_impl"`.
* If imports HTTP/CLI frameworks or entrypoints → likely `infra`.
* If no business imports and appears reusable → `util`.

Optional: call LLM with:

> Here is a function signature, docstring, and some call/import info. Classify its role: `concept_core`, `sync_impl`, `infra`, or `util`. Briefly explain.

Store the classification and explanation.

### Step 4 – Summarize CodeUnits

For each CodeUnit:

* Build a compact prompt context:

  * file path + symbol name,
  * language,
  * signature,
  * docstring,
  * role,
  * relevant snippet (e.g. first ~50 lines),
  * list of imports,
  * list of calls,
  * the subset of concepts/syncs from specs that *might* be relevant (by name similarity / folder).

Ask LLM to return JSON with:

* `summary`
* `inputs` / `outputs`
* `side_effects`
* `dependencies` (semantic)
* `error_handling`
* `performance_notes`
* `candidate_concept_ids`, `candidate_action_names`, `candidate_sync_ids`
* `candidate_functionality_ids`

Write the results to `code_summaries.json`, keyed by `CodeUnit.id`.

### Step 5 – Linking (code → concepts/syncs/funcs/reqs)

Use three sources of evidence:

1. **Static hints**

   * Comments like:

     ```python
     # Concept: CONC-Password Action: set
     # Implements: FUNC-REG-REGISTER, REQ-REG-001
     ```

   * File paths (`password_service.py` ↔ `CONC-Password`).

   * Imports (`from concepts.password import PasswordService`).

2. **LLM suggestions** from Step 4.

3. **Textual similarity** between:

   * `summary` and functionality descriptions / concept purposes / sync names.

Algorithm outline per CodeUnit:

* Build candidate links:

  * Each candidate is (CodeUnit, concept/action/sync/func).
* Score them:

  * static hint → high base score (e.g. 0.9+),
  * name match → medium,
  * embedding similarity → medium,
  * LLM vote → adjust up/down.
* Keep top few per category.
* Mark high confidence ≥ threshold as `confirmed`; others as `proposed`.

Then:

* Derive requirement links from functionalities:

  * `CodeUnit → Funcs → Requirements`.

Write all to `trace_links.json`.

### Step 6 – Design checks

Implement separate modules under `checks/`.

1. **Boundary / independence check**

   * For each `concept_core` CodeUnit:

     * If it imports or calls code from another concept’s namespace, flag.
   * For each `sync_impl`:

     * If it writes directly to DB tables that belong to a concept, flag.

2. **Orphans & coverage**

   * Requirements with no `FUNC-*`.
   * Functionalities with no code (via trace links).
   * Concepts with no code or syncs.
   * Syncs with no code.
   * CodeUnits with no links and not marked `infra/util`.

3. **Redundancy**

   * Cluster `summary` text embeddings.
   * For clusters with multiple CodeUnits:

     * Show them and their linked concepts/funcs for manual review.

4. **Legibility metric**

   * For each `FUNC-*`:

     * Count number of associated concepts and syncs.
     * If the set is huge and scattered, mark as “sprawling feature”.

### Step 7 – Reports

Generate:

* `reports/module_summaries.md`

  * Per file: list of CodeUnits + summaries + linked IDs.

* `reports/gaps.md`

  * Orphans, coverage gaps, boundary violations.

* `reports/design_view.md`

  * Grouped by concept:

    * Concept spec summary.
    * Actions → CodeUnits → Syncs → Functionalities.

---

# 6. Runtime instrumentation & flows

You can implement minimal runtime tracing **without** building a full synchronization engine, but you’ll get more value if you do.

## 6.1 Flow IDs

* For HTTP: middleware that:

  * generates `flow_id = uuid4()` per incoming request,
  * stores it in a context:

    * Python: `contextvars.ContextVar("flow_id")`,
    * Node: async‑local‑storage.

* For jobs / CLI:

  * same idea; wrap entrypoint to set `flow_id`.

## 6.2 Concept action decorators / wrappers

Python example:

```python
from trace_sync.runtime import log_action

def concept_action(concept_id, action_name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            return log_action(
                kind="concept_action",
                concept_id=concept_id,
                action_name=action_name,
                fn=fn,
                args=args,
                kwargs=kwargs,
            )
        return wrapper
    return decorator
```

`log_action` should:

1. Read current `flow_id` from context.
2. Serialize input args minimally (types + key identifiers).
3. Log an “invocation started” record (optional).
4. Call the function.
5. Log “completed” record with:

   * `kind=concept_action`,
   * concept/action,
   * flow_id,
   * code_unit id,
   * simple output description (case + key fields),
   * error info if exception raised.
6. Return the result or re‑raise error.

JS: same idea using a higher‑order function.

## 6.3 Sync instrumentation

For synchronization implementations:

```python
@sync_rule("SYNC-RegistrationResponse")
def build_registration_response(...):
    ...
```

`@sync_rule` logs an ActionRecord with `kind="sync"` and `sync_id`.

You don’t have to implement a full rule engine immediately; at first these can just be the places where cross‑concept behavior lives.

## 6.4 Trace store & queries

Store ActionRecords:

* In a DB table (`actions`), or
* In newline‑delimited JSON under `runtime/traces/`.

Add a small `trace_sync.runtime` module with functions:

* `record_action(record: ActionRecord)`
* `get_flow(flow_id) -> List[ActionRecord]`
* `find_flows_by_request_id(request_id) -> List[flow_id]`

Add CLI commands:

```bash
trace-sync explain-flow <flow-id>
trace-sync explain-action <action-id>
```

These format flows as a human/LLM‑friendly narrative, e.g.:

```text
Flow flow-9876 (HTTP POST /register):

1. Web.request(method=register, username=alice, email=alice@example.com)
2. Password.validate(password=***, result=valid:false)
3. User.register(user=alice, result=error:user_exists)
4. Web.respond(error="User already exists", code=422)
```

Exactly the kind of trace used in the WYSIWID bug‑fix story.

---

# 7. Integration with coding assistants (tools + prompts)

This is key: the whole system should feel like a **native extension** of your coding assistant, not a separate world.

## 7.1 Expose `trace_sync` as tools

Assume your assistant can call tools via HTTP or a local CLI.

Wrap these CLI commands as tools:

1. `scan_project`

   * Runs: `trace-sync scan`
   * Returns:

     * coverage stats,
     * list of changed links,
     * any new gaps/violations.

2. `explain_requirement`

   * Params: `req_id`
   * Uses:

     * `trace-sync explain-req REQ-...`
   * Returns:

     * requirements text,
     * linked functionalities,
     * concepts/syncs,
     * top code units,
     * top tests,
     * sample flows (if available).

3. `explain_code_unit`

   * Params: `code_unit_id`
   * Returns the chain up to requirements, plus summary.

4. `suggest_links_for_code_unit`

   * Params: `file_path`, `symbol_name`
   * Runs partial pipeline on that file and returns suggested links, for incremental work.

5. `explain_flow`

   * Params: `flow_id` or `request_id`
   * Uses runtime traces to give the action graph.

6. `update_specs_for_new_feature`

   * Params: structured description of a new feature.
   * Orchestrates:

     * propose `FUNC-*`,
     * propose `CONC-*` changes,
     * propose `SYNC-*` additions,
     * return a patch to `docs/*`.

These tools let the assistant reason about *intent* and architecture, not just the source files.

## 7.2 `ai_assistant.md` / `claude.md` / `agent.md` content

Create a doc like `docs/ai_assistant.md` that becomes part of your system prompt for any coding agent working in this repo. Include rules like:

1. **Always work through the specs**

   * Before adding or changing code for a feature:

     * read the relevant `REQ-*` and `FUNC-*`,
     * read/edit the corresponding concepts in `docs/concepts/`,
     * read/edit relevant syncs in `docs/syncs/`.

2. **Keep IDs in sync**

   * When you implement or modify code for a concept action:

     * annotate the function/class with:

       ```python
       # Concept: CONC-Password
       # Action: set
       # Implements: FUNC-REG-REGISTER, REQ-REG-001
       ```

   * For synchronization code, annotate with:

     ```python
     # Sync: SYNC-Registration
     ```

3. **Do not cross concept boundaries directly**

   * Concept code must not call other concepts’ internal functions or DB tables. Use syncs or well‑defined APIs instead.

4. **After making changes**

   * Run `trace-sync scan --changed-files ...`.
   * Fix any reported:

     * orphan requirements/functionalities/concepts/syncs,
     * boundary violations,
     * coverage regressions.

5. **When asked to implement a new feature**

   * Propose or update:

     * `REQ-*` (if needed),
     * `FUNC-*`,
     * `CONC-*` specs,
     * `SYNC-*` specs.
   * Only then generate/update code in small, localized patches.

6. **For debugging**

   * Use `trace-sync explain-flow` on failing tests / bug reports.
   * Inspect the relevant syncs and concepts; propose changes at the spec layer first, then regenerate code.

You can tailor the wording, but those are the **behaviors** the LLM should adopt.

## 7.3 Using it inside a “coding agent” loop

When the assistant receives a task like:

> “Add email verification to registration.”

Your orchestrator can:

1. Call `trace_sync` tool:

   * `explain_requirement REQ-REG-001` (or create a new requirement).
2. Ask LLM to propose:

   * a new `CONC-EmailVerification` spec,
   * new syncs (`SYNC-SendVerificationEmail`, `SYNC-VerifyEmail`, etc.),
   * updated functionalities linking this requirement.
3. Write these specs to `docs/concepts/` and `docs/syncs/`.
4. Ask LLM to generate:

   * concept code (Python/JS),
   * sync implementations (or modifications).
5. Run `trace-sync scan --changed-files ...`.
6. Run tests and gather traces.
7. If a test fails, call `explain-flow` and feed trace + specs back into the LLM to debug.

This keeps the assistant *honest* about architecture and requirements, and gives you the legibility, incrementality, and transparency the paper argues for.

---

# 8. Implementation phases (so the agent doesn’t try to do everything at once)

You probably don’t want to drop this whole thing on an existing repo in one shot. A reasonable phased plan:

### Phase 1 – Static skeleton

* Implement:

  * specs loader (`docs/*.yml`),
  * Python/JS scanners,
  * basic `code_summaries.json` (without LLM),
  * simple linking based on comments/names.
* Run `trace-sync scan` in CI, but only **warn**.

### Phase 2 – LLM summaries + stronger linking

* Add summarization + linking via LLM.
* Implement orphan + boundary checks.
* Start **failing CI** on severe violations (e.g., new orphan requirements, concept boundary breaks).

### Phase 3 – Runtime traces

* Add flow IDs + `@concept_action` / `@sync_rule` instrumentation.
* Implement `explain-flow`.
* Use it in debugging manually.

### Phase 4 – LLM workflow integration

* Wire up tools for:

  * `explain_requirement`,
  * `update_specs_for_new_feature`,
  * `explain_flow`.
* Update `ai_assistant.md` / `claude.md` to enforce the behavior.

### Phase 5 – Architecture tightening

* Gradually refactor existing code into concept/sync structure.
* Tighten CI rules:

  * require certain coverage thresholds,
  * forbid new code units without links,
  * forbid cross‑concept calls.
