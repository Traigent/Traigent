Looking at where you are now versus the issues we discussed, here are concrete instructions for the CodeSync agent to tighten up the tagging system.

---

# Agent Instructions: Tagging System Improvements

## Context

The current tagging work (per `tagging_tracking.md`) has made good progress applying `CONC-Layer-*` and `CONC-Quality-*` tags across Traigent modules. However, the taxonomy itself remains under-specified, and several issues from our review are unaddressed. This instruction set guides the next phase of work.

---

## 1. Formalize the Taxonomy in a Config File

**Create `docs/traceability/taxonomy.yaml`** as the single source of truth. CodeSync and any linting tools must validate against this file.

```yaml
# docs/traceability/taxonomy.yaml
version: "1.0"
dimensions:
  Layer:
    required: true
    cardinality: exactly_one
    allowed:
      - CONC-Layer-API        # Public entrypoints: decorators, CLI, SDK surfaces
      - CONC-Layer-Core       # Domain logic, orchestration, optimization algorithms
      - CONC-Layer-Integration # External adapters, provider clients, framework hooks
      - CONC-Layer-Infra      # Storage backends, cloud clients, networking, security plumbing
      - CONC-Layer-Data       # Schemas, DTOs, TVL specs, dataset definitions
      - CONC-Layer-Tooling    # Dev tools, analyzers, scripts not shipped to users
    decision_tree: |
      1. Does this code define public API surface (decorators, CLI commands, SDK functions)?
         → Layer-API
      2. Does this code implement domain/business logic (optimization, orchestration, evaluation)?
         → Layer-Core
      3. Does this code adapt to external systems (LLM providers, frameworks, third-party APIs)?
         → Layer-Integration
      4. Does this code provide low-level infrastructure (storage, auth, networking, retries)?
         → Layer-Infra
      5. Does this code define data structures, schemas, or specs without behavior?
         → Layer-Data
      6. Is this development tooling not shipped to end users?
         → Layer-Tooling

  Quality:
    required: false
    cardinality: 0-2  # Reduced from 3 to force prioritization
    allowed:
      - CONC-Quality-Performance    # Optimizes time/resource behavior
      - CONC-Quality-Reliability    # Fault tolerance, error handling, recovery
      - CONC-Quality-Security       # Authentication, authorization, secrets, audit
      - CONC-Quality-Observability  # Logging, metrics, tracing instrumentation
      - CONC-Quality-Maintainability # Modularity, testability, documentation
      - CONC-Quality-Compatibility  # Cross-platform, version compatibility, interop
    decision_tree: |
      Apply a quality tag ONLY if this code's PRIMARY PURPOSE includes mechanisms for that quality.
      Ask: "If I removed all code related to <quality>, would the file lose significant functionality?"
      If yes → tag it. If no → don't.
      
      Maximum 2 tags. If you think 3+ apply, pick the 2 most essential.

  Compliance:
    required: false
    cardinality: 0-2
    allowed:
      - CONC-Compliance-SOC2-Audit
      - CONC-Compliance-GDPR-Retention
      - CONC-Compliance-NIST-AI-RMF
    notes: |
      Use sparingly. Only for code that directly implements compliance controls.
      Do NOT use CONC-Quality-Compliance (removed from taxonomy).

# Explicitly excluded from v1.0:
# - CONC-View-* (defer to future)
# - CONC-Lifecycle-* (defer to future)
# - CONC-Domain-* (defer to future)
# - CONC-ML-* (defer to future - revisit when ML lineage is priority)
# - CONC-Layer-Util (replaced by more specific layers)
# - CONC-Layer-CrossCutting (see migration notes)
# - CONC-Layer-Test (tests not tagged in v1.0)
# - CONC-Layer-Experimental (see migration notes)
```

---

## 2. Resolve Layer Ambiguities

### 2.1 Eliminate `CrossCutting` and `Util`

The tagging tracker shows `CONC-Layer-CrossCutting` was used in `traigent/utils/*`. This is a dumping ground. **Reclassify every file currently tagged `CrossCutting` or `Util`:**

| Current location | Likely correct layer | Rationale |
|------------------|---------------------|-----------|
| `utils/retry_consolidated.py` | **Layer-Infra** | Retry/backoff is infrastructure for remote calls |
| `utils/logging_utils.py` | **Layer-Infra** | Logging infrastructure |
| `utils/env_utils.py` | **Layer-Infra** | Environment configuration plumbing |
| `utils/persistence.py` | **Layer-Infra** | Storage abstraction |
| `utils/optimization_logger.py` | **Layer-Core** + Quality-Observability | Domain-specific logging for optimization |
| `utils/langchain_interceptor.py` | **Layer-Integration** | Adapts to LangChain |
| `utils/config_helpers.py` | **Layer-Data** | Configuration schemas |

**Action:** Audit every file in `traigent/utils/` and reassign to a proper layer. If a file genuinely spans multiple layers, it should be **split**, not tagged with a catch-all.

### 2.2 Handle `Experimental`

The tracker shows `CONC-Layer-Experimental` was applied to `traigent/experimental/*`. This isn't in our approved taxonomy.

**Decision options:**

1. **Tag as normal layers** (recommended): Experimental code is still code. Tag `experimental/simple_cloud/executor.py` as `Layer-Infra` if it's infrastructure, `Layer-Core` if it's domain logic.

2. **Exclude from tagging**: Add `experimental/` to a CodeSync ignore list. State explicitly: "Experimental code is not traced until promoted."

**Do not** create a `Layer-Experimental` tag. It conflates architectural role with development stage.

---

## 3. Remove `CONC-Quality-Compliance`

Search the codebase for any remaining `CONC-Quality-Compliance` tags and replace:

```bash
grep -r "CONC-Quality-Compliance" traigent/
```

For each hit:
- If it's about audit logging → `CONC-Quality-Observability` + `CONC-Compliance-SOC2-Audit`
- If it's about data retention → `CONC-Compliance-GDPR-Retention`
- If it's general security → `CONC-Quality-Security`

This removes the conflation between product quality and governance compliance.

---

## 4. Implement Taxonomy Validation in CodeSync

### 4.1 Add a taxonomy loader

```python
# tools/trace_sync/taxonomy.py
from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class DimensionSpec:
    required: bool
    cardinality: str  # "exactly_one", "0-2", "0-3", etc.
    allowed: list[str]
    decision_tree: Optional[str] = None

@dataclass
class Taxonomy:
    version: str
    dimensions: dict[str, DimensionSpec]
    
    @classmethod
    def load(cls, path: Path) -> "Taxonomy":
        with path.open() as f:
            data = yaml.safe_load(f)
        dimensions = {}
        for name, spec in data.get("dimensions", {}).items():
            dimensions[name] = DimensionSpec(
                required=spec.get("required", False),
                cardinality=spec.get("cardinality", "0-n"),
                allowed=spec.get("allowed", []),
                decision_tree=spec.get("decision_tree"),
            )
        return cls(version=data.get("version", "unknown"), dimensions=dimensions)
```

### 4.2 Add validation to the linter

```python
# tools/trace_sync/checks/taxonomy_check.py
from ..taxonomy import Taxonomy

def validate_tags(tags: list[str], taxonomy: Taxonomy) -> list[str]:
    """Returns list of error messages, empty if valid."""
    errors = []
    
    # Group tags by dimension
    by_dimension: dict[str, list[str]] = {}
    unknown_tags = []
    
    for tag in tags:
        found = False
        for dim_name, dim_spec in taxonomy.dimensions.items():
            if tag in dim_spec.allowed:
                by_dimension.setdefault(dim_name, []).append(tag)
                found = True
                break
        if not found:
            unknown_tags.append(tag)
    
    # Check unknown tags
    for tag in unknown_tags:
        if tag.startswith("CONC-"):
            errors.append(f"Unknown concept tag: {tag}")
    
    # Check required dimensions
    for dim_name, dim_spec in taxonomy.dimensions.items():
        count = len(by_dimension.get(dim_name, []))
        
        if dim_spec.required and count == 0:
            errors.append(f"Missing required {dim_name} tag")
        
        if dim_spec.cardinality == "exactly_one" and count != 1:
            errors.append(f"{dim_name}: expected exactly 1, got {count}")
        
        if dim_spec.cardinality.startswith("0-"):
            max_count = int(dim_spec.cardinality.split("-")[1])
            if count > max_count:
                errors.append(f"{dim_name}: max {max_count}, got {count}")
    
    return errors
```

### 4.3 Integrate into CI

Add to `trace-sync check-ci`:

```python
def check_taxonomy_compliance(repo_path: Path, taxonomy: Taxonomy) -> list[Violation]:
    violations = []
    for code_unit in scan_code_units(repo_path):
        tags = extract_tags(code_unit)
        errors = validate_tags(tags, taxonomy)
        for error in errors:
            violations.append(Violation(
                file=code_unit.file,
                message=error,
                severity="error"
            ))
    return violations
```

---

## 5. Update the Coverage Summary Script

The current `check_coverage.py` uses `TARGET_CONCEPTS` with layer tags. This conflates layers with the broader traceability model. Update to validate against:

1. **Layer coverage**: Every non-test file has exactly one `CONC-Layer-*`
2. **Requirement coverage**: Every `REQ-*` in `requirements.yml` has at least one linked `FUNC-*`
3. **Functionality coverage**: Every `FUNC-*` has at least one linked code unit

```python
# Revised target structure
COVERAGE_TARGETS = {
    "layers": {
        "CONC-Layer-API",
        "CONC-Layer-Core", 
        "CONC-Layer-Integration",
        "CONC-Layer-Infra",
        "CONC-Layer-Data",
        "CONC-Layer-Tooling",
    },
    "requirements": set(),  # Loaded from requirements.yml
    "functionalities": set(),  # Loaded from functionalities.yml
    "syncs": set(),  # Loaded from docs/syncs/*.yml
}
```

---

## 6. Document the Decision Procedure

Create `docs/traceability/TAGGING_GUIDE.md` with:

1. **When to tag**: Every Python file in `traigent/` except `__init__.py` and test files
2. **Layer decision tree**: Copy from taxonomy.yaml, with examples
3. **Quality decision rules**: "Apply only if PRIMARY purpose"
4. **Compliance rules**: "Only for direct control implementation"
5. **Common mistakes**:
   - Using `CrossCutting` as a catch-all (don't)
   - Tagging 3+ qualities (max 2)
   - Confusing `Integration` vs `Infra` (Integration = adapts to external APIs; Infra = internal plumbing)

---

## 7. Migration Checklist

For the agent to execute:

- [ ] Create `docs/traceability/taxonomy.yaml` per spec above
- [ ] Audit all `CONC-Layer-CrossCutting` tags → reassign to proper layers
- [ ] Audit all `CONC-Layer-Experimental` tags → decide: tag properly or exclude
- [ ] Remove all `CONC-Quality-Compliance` → replace with specific tags
- [ ] Reduce any files with 3+ quality tags to max 2
- [ ] Implement `tools/trace_sync/taxonomy.py` loader
- [ ] Implement `tools/trace_sync/checks/taxonomy_check.py`
- [ ] Add taxonomy validation to `check-ci` command
- [ ] Update `check_coverage.py` to load targets from specs
- [ ] Create `docs/traceability/TAGGING_GUIDE.md`
- [ ] Update `tagging_tracking.md` to reflect completed migrations

---

## 8. Validation Criteria

The taxonomy work is "done" when:

1. `trace-sync check-ci` passes with zero taxonomy violations
2. No files have `CrossCutting`, `Util`, `Experimental`, or `Quality-Compliance` tags
3. Every tagged file has exactly 1 layer and 0-2 quality tags
4. `taxonomy.yaml` is the single source of truth (no hardcoded tag lists elsewhere)
5. Coverage report shows Layer coverage ≥95% of eligible files

---

This gives the agent a concrete, auditable task list rather than abstract guidance. The key principle: **fewer, stricter tags with clear decision rules beat comprehensive but fuzzy taxonomies**.