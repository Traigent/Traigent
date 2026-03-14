# TVL 0.9 Implementation - Review Decisions

**Date:** 2025-12-15
**Status:** Revised - Focus on Remaining Gaps Only

---

## Review Summary

| Reviewer | Verdict |
|----------|---------|
| Gemini Pro | **Approved** - "The plan is solid and ready for implementation." |
| Codex | **Approve with Changes** - Plan drift detected; SDK already has most TVL 0.9 features |

---

## Critical Finding from Codex

**The original plan significantly underestimated what's already implemented.** The SDK already has:

| Feature | File | Status |
|---------|------|--------|
| TVL Header parsing | `spec_loader.py:436-454` | ✅ Done |
| Environment snapshot | `spec_loader.py:457-475` | ✅ Done |
| Evaluation set | `spec_loader.py:478-496` | ✅ Done |
| TVars parsing | `spec_loader.py:344-412` | ✅ Done |
| Banded objectives | `spec_loader.py:677-763`, `objectives.py:40-118` | ✅ Done |
| Structural constraints | `spec_loader.py:614-661` | ✅ Done |
| Derived constraints | `spec_loader.py:580-611` | ✅ Done |
| Promotion policy parsing | `spec_loader.py:415-433` | ✅ Done |
| Promotion gate logic | `promotion_gate.py:111-523` | ✅ Done |
| Convergence criteria | `spec_loader.py:499-521` | ✅ Done |
| Exploration budgets | `spec_loader.py:524-546` | ✅ Done |
| TOST equivalence test | `tvl/objectives.py` | ✅ Done |
| BH adjustment | `tvl/statistics.py` | ✅ Done |

**The implementation plan must be revised to focus on actual remaining gaps.**

---

## Actual Remaining Gaps (Post-Review)

Based on Codex's analysis, these are the **real gaps** to address:

| Gap | Priority | Description |
|-----|----------|-------------|
| **Constraint expression semantics** | HIGH | TVL uses `=` and dotted identifiers (`retriever.k`); SDK uses Python/CEL-like (`params`/`metrics`). Need translation rules or documentation. |
| **Registry domain fail-fast** | HIGH | Empty domain on unresolved registry is silent. Should fail-fast or require explicit resolver. |
| **evaluation_set wiring** | MEDIUM | Parsed but not wired to decorator evaluation settings. Define URI scheme support. |
| **Promotion gate integration point** | MEDIUM | Gate logic exists but needs CI/CLI/orchestrator hookup. |
| **tvars/configuration_space conflict** | LOW | Define precedence + warn/error if both present with conflicts. |
| **Deprecation timeline** | LOW | Set explicit deprecation dates for legacy formats. |

---

## Resolved Open Questions

Based on both Gemini Pro and Codex feedback:

| # | Question | Gemini | Codex | **Final Decision** |
|---|----------|--------|-------|-------------------|
| 1 | `tvars` vs `configuration_space` | Keep both | Keep both + precedence + warn on conflict | **Keep both, define precedence, warn/error on conflict, set deprecation timeline** |
| 2 | Banded objectives structure | Separate list | Extend `ObjectiveDefinition` (already done!) | **Use existing `ObjectiveDefinition.orientation="band"` (already implemented)** |
| 3 | RFC3339 validation | Loose | Soft-validate (warn) + optional strict mode | **Soft-validate by default, optional strict mode** |
| 4 | Promotion policy integration | Store it | Integrate into CI/CLI flow first | **Gate logic exists; focus on CI/CLI/orchestrator integration** |
| 5 | Convergence criteria | Store it | Parse/store now, implement callback later | **Already parsed; implement opt-in callback when needed** |
| 6 | Registry resolution | Default resolver | Default interface + fail-fast | **Provide default resolver interface + fail-fast if registry domains appear without resolver** |
| 7 | Derived constraints | Metadata only | Store as metadata, optionally evaluate | **Store as metadata; optionally evaluate when symbols available** |
| 8 | Constraint formats | Support both + deprecate | Support both + deprecation warnings | **Support both + add deprecation warnings + migration guidance** |

---

## Key Architectural Endorsements

Both reviewers validated:

1. **Repository Split** - "Excellent. Prevents SDK from becoming bloated with static analysis tools."

2. **Artifact Pattern** - "`TVLSpecArtifact` serves as a clean abstraction layer between raw YAML and optimization engine."

3. **Phased Parsing** - "Logical and allows for fail-fast validation during loading."

---

## Identified Risks & Mitigations

| Risk | Mitigation | Owner |
|------|------------|-------|
| **Complex Types** (`tuple[int, float]`) | Ensure `normalize_tvar_type` handles nested generics | Verify existing impl |
| **Formula Parsing Security** | Use safe parser (CEL-like subset), never `eval()` | ✅ Already done via AST validation |
| **Environment Inheritance** | Test that overlays don't clobber `snapshot_id` | Add test case |
| **Silent Registry Failure** | Add fail-fast when registry domain has no resolver | **NEW: Must implement** |

---

## Revised Implementation Priorities

Given that **most TVL 0.9 parsing is already done**, focus on:

```
Phase 1: Gap Fixes (HIGH)
├── Add registry domain fail-fast behavior
├── Define constraint expression translation rules
├── Add tvars/configuration_space conflict detection
└── Add deprecation warnings for legacy formats

Phase 2: Integration (MEDIUM)
├── Wire evaluation_set to decorator
├── Wire promotion gate to CLI/orchestrator
└── Document supported URI schemes

Phase 3: Documentation (HIGH)
├── Update CLAUDE.md with TVL 0.9 status
├── Create troubleshooting guide
├── Document constraint expression semantics
└── Set deprecation timeline

Phase 4: Testing (HIGH)
├── Add tests for edge cases (both formats present)
├── Add tests for registry domain fail-fast
├── Verify environment overlay + snapshot_id interaction
└── Integration tests for full TVL 0.9 specs
```

---

## Action Items

### Immediate (This PR)

- [ ] Update original plan document to reflect actual state
- [ ] Remove "Phase 1-5" from plan (already done)
- [ ] Add registry domain fail-fast to `DomainSpec`
- [ ] Add conflict detection when both `tvars` and `configuration_space` present
- [ ] Add deprecation warning for `configuration_space`

### Short Term

- [ ] Wire `evaluation_set` to decorator evaluation settings
- [ ] Add CLI command for promotion gate evaluation
- [ ] Document constraint expression semantics

### Medium Term

- [ ] Implement default registry resolver interface
- [ ] Add convergence callback for Optuna/NSGA-II
- [ ] Set formal deprecation dates

---

## Next Steps

1. ~~Wait for Codex feedback~~ **Done**
2. **Update implementation plan to reflect actual state**
3. **Focus on remaining gaps only**
4. **Create PR with targeted fixes**

---

*Document updated after Codex review revealed plan drift.*
