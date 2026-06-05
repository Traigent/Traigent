/*
 * Cost Enforcement Alloy Specification
 *
 * Formal model for verifying cost enforcement invariants in Traigent SDK.
 * Based on Daniel Jackson's concept-based design methodology.
 *
 * Reference: docs/traceability/concepts/cost_enforcement.yml
 *
 * Usage: Run with Alloy Analyzer to check assertions
 *   java -jar alloy4.2.jar cost_enforcement.als
 */

open util/ordering[Time]

-- Time for temporal modeling
sig Time {}

-- Float approximation (Alloy has no native floats)
sig Float {
    value: one Int
}

-- Predicate for float >= 0
pred nonNegative[f: Float] {
    f.value >= 0
}

-- Predicate for float comparison with epsilon tolerance
pred leq[a, b: Float] {
    a.value <= b.value
}

pred lessThan[a, b: Float] {
    a.value < b.value
}

-- Non-negative integer addition is valid only when it does not wrap.
pred nonWrappingAdd[a, b: Int] {
    a >= 0
    b >= 0
    a.plus[b] >= a
}

-- Add floats
fun addFloat[a, b: Float]: Float {
    { f: Float | f.value = a.value.plus[b.value] }
}

-- Permit dataclass
sig Permit {
    id: one Int,
    amount: one Float,
    active: Time -> one Bool
}

-- Boolean type
abstract sig Bool {}
one sig True, False extends Bool {}

-- Cost Enforcer state
sig CostEnforcer {
    -- State variables
    accumulated_cost: Time -> one Float,
    reserved_cost: Time -> one Float,
    in_flight_count: Time -> one Int,
    active_permits: Time -> set Permit,
    permit_counter: Time -> one Int,
    limit: one Float,
    mock_mode: one Bool
}

-- Initial state
pred init[e: CostEnforcer, t: Time] {
    nonNegative[e.limit]
    e.accumulated_cost[t].value = 0
    e.reserved_cost[t].value = 0
    e.in_flight_count[t] = 0
    no e.active_permits[t]
    e.permit_counter[t] = 0
    all p: Permit | p.active[t] = False
}

-- Invariant I1: in_flight_count >= 0
pred I1_InFlightNonNegative[e: CostEnforcer, t: Time] {
    e.in_flight_count[t] >= 0
}

-- Invariant I2: reserved_cost >= 0
pred I2_ReservedNonNegative[e: CostEnforcer, t: Time] {
    nonNegative[e.reserved_cost[t]]
}

-- Invariant I3: len(active_permits) == in_flight_count
pred I3_ActiveEqualsInFlight[e: CostEnforcer, t: Time] {
    #(e.active_permits[t]) = e.in_flight_count[t]
}

-- I4 admission guard: admission-time budget bound.
-- A permit may be granted only when the projected committed cost remains within
-- the limit. This is not a global state invariant because trackCost records the
-- actual post-execution cost, which may exceed the estimate reserved earlier.
--
-- The 3-term admission predicate (accumulated + reserved + estimated <= limit)
-- implies the immediate post-admission 2-term consequence
-- (accumulated + reserved <= limit), because after grant the new reserved
-- includes the just-admitted estimate. After any trackCost transition, actual
-- spend may make the 2-term form false, including while other admitted permits
-- remain in flight. The Python runtime therefore enforces I4 at admission,
-- not as a general state invariant.
pred I4_AdmissionBudgetBound[e: CostEnforcer, t: Time, estimated: Float] {
    nonWrappingAdd[e.accumulated_cost[t].value, e.reserved_cost[t].value]
    nonWrappingAdd[
        e.accumulated_cost[t].value.plus[e.reserved_cost[t].value],
        estimated.value
    ]
    e.accumulated_cost[t].value.plus[e.reserved_cost[t].value].plus[estimated.value]
        <= e.limit.value
}

pred committedWithinLimit[e: CostEnforcer, t: Time] {
    nonWrappingAdd[e.accumulated_cost[t].value, e.reserved_cost[t].value]
    e.accumulated_cost[t].value.plus[e.reserved_cost[t].value] <= e.limit.value
}

pred committedExceedsLimit[e: CostEnforcer, t: Time] {
    nonWrappingAdd[e.accumulated_cost[t].value, e.reserved_cost[t].value]
    e.accumulated_cost[t].value.plus[e.reserved_cost[t].value] > e.limit.value
}

-- Invariant I5: Released permits have active=False
pred I5_ReleasedInactive[e: CostEnforcer, p: Permit, t: Time] {
    p not in e.active_permits[t] implies p.active[t] = False
}

-- Invariant I6: Permit IDs monotonically increasing
pred I6_MonotonicCounter[e: CostEnforcer, t1: Time, t2: Time] {
    t1 in t2.prevs implies e.permit_counter[t1] <= e.permit_counter[t2]
}

-- Invariant I7: Denied permits have specific properties
pred I7_DeniedPermitProperties[e: CostEnforcer, p: Permit, t: Time] {
    p.id = -1 implies (p not in e.active_permits[t] and p.active[t] = False)
}

-- Invariant I8: Sum of permit amounts equals reserved_cost
-- (Simplified: count of permits * estimated amount approximates reserved)
pred I8_PermitSumEqualsReserved[e: CostEnforcer, t: Time] {
    let permits = e.active_permits[t] |
        (sum p: permits | p.amount.value) = e.reserved_cost[t].value
}

-- All invariants hold
pred allInvariantsHold[e: CostEnforcer, t: Time] {
    I1_InFlightNonNegative[e, t]
    I2_ReservedNonNegative[e, t]
    I3_ActiveEqualsInFlight[e, t]
    all p: Permit | p in e.active_permits[t] iff p.active[t] = True
    all p: Permit | I5_ReleasedInactive[e, p, t]
    all p: Permit | I7_DeniedPermitProperties[e, p, t]
    all disj p1, p2: e.active_permits[t] | p1.id != p2.id
    I8_PermitSumEqualsReserved[e, t]
}

-- Acquire permit operation
pred acquirePermit[e: CostEnforcer, t: Time, tNext: Time, p: Permit, estimated: Float] {
    -- Precondition: positive estimate and budget available at admission.
    nonNegative[estimated]
    estimated.value > 0
    I4_AdmissionBudgetBound[e, t, estimated]
    nonWrappingAdd[e.reserved_cost[t].value, estimated.value]
    e.in_flight_count[t].plus[1] > e.in_flight_count[t]
    e.permit_counter[t].plus[1] > e.permit_counter[t]
    p not in e.active_permits[t]
    p.active[t] = False
    all existing: e.active_permits[t] |
        existing.id != e.permit_counter[t].plus[1]

    -- Postcondition: state updated atomically
    e.permit_counter[tNext] = e.permit_counter[t].plus[1]
    p.id = e.permit_counter[tNext]
    p.amount = estimated
    p.active[tNext] = True
    e.reserved_cost[tNext].value = e.reserved_cost[t].value.plus[estimated.value]
    e.in_flight_count[tNext] = e.in_flight_count[t].plus[1]
    e.active_permits[tNext] = e.active_permits[t] + p
    e.accumulated_cost[tNext] = e.accumulated_cost[t]
    all q: Permit - p | q.active[tNext] = q.active[t]
}

-- Deny permit (budget exceeded)
pred denyPermit[e: CostEnforcer, t: Time, tNext: Time, p: Permit, estimated: Float] {
    -- Precondition: budget would be exceeded
    nonNegative[estimated]
    estimated.value > 0
    not I4_AdmissionBudgetBound[e, t, estimated]
    p not in e.active_permits[t]
    p.active[t] = False

    -- State unchanged except for denied permit creation
    p.id = -1
    p.amount.value = 0
    p.active[tNext] = False

    -- State unchanged
    e.permit_counter[tNext] = e.permit_counter[t]
    e.reserved_cost[tNext] = e.reserved_cost[t]
    e.in_flight_count[tNext] = e.in_flight_count[t]
    e.active_permits[tNext] = e.active_permits[t]
    e.accumulated_cost[tNext] = e.accumulated_cost[t]
    all q: Permit - p | q.active[tNext] = q.active[t]
}

-- Release permit operation
pred releasePermit[e: CostEnforcer, t: Time, tNext: Time, p: Permit] {
    -- Precondition: permit is active
    p in e.active_permits[t]
    p.active[t] = True
    p.amount.value >= 0
    e.in_flight_count[t] > 0
    e.reserved_cost[t].value >= p.amount.value
    e.reserved_cost[t].value.minus[p.amount.value] >= 0

    -- Postcondition: permit released
    p.active[tNext] = False
    e.active_permits[tNext] = e.active_permits[t] - p
    e.in_flight_count[tNext] = e.in_flight_count[t].minus[1]
    e.reserved_cost[tNext].value = e.reserved_cost[t].value.minus[p.amount.value]
    e.accumulated_cost[tNext] = e.accumulated_cost[t]
    e.permit_counter[tNext] = e.permit_counter[t]
    all q: Permit - p | q.active[tNext] = q.active[t]
}

-- Track cost operation
pred trackCost[e: CostEnforcer, t: Time, tNext: Time, p: Permit, actual: Float] {
    -- Implementation note: Python rejects negative costs at method entry
    -- (see CostEnforcer.track_cost); the model assumes inputs are non-negative.
    nonNegative[actual]

    -- Precondition: permit is active
    p in e.active_permits[t]
    p.active[t] = True
    p.amount.value >= 0
    e.in_flight_count[t] > 0
    e.reserved_cost[t].value >= p.amount.value
    e.reserved_cost[t].value.minus[p.amount.value] >= 0
    nonWrappingAdd[e.accumulated_cost[t].value, actual.value]

    -- Postcondition: permit released, cost tracked
    p.active[tNext] = False
    e.active_permits[tNext] = e.active_permits[t] - p
    e.in_flight_count[tNext] = e.in_flight_count[t].minus[1]
    e.reserved_cost[tNext].value = e.reserved_cost[t].value.minus[p.amount.value]
    e.accumulated_cost[tNext].value = e.accumulated_cost[t].value.plus[actual.value]
    e.permit_counter[tNext] = e.permit_counter[t]
    all q: Permit - p | q.active[tNext] = q.active[t]
}

-- Double release attempt (should fail)
pred doubleRelease[e: CostEnforcer, t: Time, tNext: Time, p: Permit] {
    -- Precondition: permit already released
    p not in e.active_permits[t] or p.active[t] = False

    -- Postcondition: state unchanged (idempotent)
    e.active_permits[tNext] = e.active_permits[t]
    e.in_flight_count[tNext] = e.in_flight_count[t]
    e.reserved_cost[tNext] = e.reserved_cost[t]
    e.accumulated_cost[tNext] = e.accumulated_cost[t]
    e.permit_counter[tNext] = e.permit_counter[t]
    all q: Permit | q.active[tNext] = q.active[t]
}

-- Mock-mode bypass was removed from the SDK; see manifest known_gaps for the
-- SDK-side tests that pin that behavior outside this Alloy model.

pred stutter[e: CostEnforcer, t: Time, tNext: Time] {
    e.active_permits[tNext] = e.active_permits[t]
    e.in_flight_count[tNext] = e.in_flight_count[t]
    e.reserved_cost[tNext] = e.reserved_cost[t]
    e.accumulated_cost[tNext] = e.accumulated_cost[t]
    e.permit_counter[tNext] = e.permit_counter[t]
    all p: Permit | p.active[tNext] = p.active[t]
}

-- System transition: frame conditions
pred frame[e: CostEnforcer, t: Time, tNext: Time] {
    tNext = t.next
    -- A valid step is an explicit stutter or one SDK operation shape.
    stutter[e, t, tNext] or
    (some p: Permit, est: Float | acquirePermit[e, t, tNext, p, est]) or
    (some p: Permit, est: Float | denyPermit[e, t, tNext, p, est]) or
    (some p: Permit | releasePermit[e, t, tNext, p]) or
    (some p: Permit, act: Float | trackCost[e, t, tNext, p, act]) or
    (some p: Permit | doubleRelease[e, t, tNext, p])
}

-- Trace: sequence of valid states
pred validTrace[e: CostEnforcer] {
    init[e, first]
    all t: Time - last | frame[e, t, t.next]
}

-- ============================================================================
-- ASSERTIONS (run with Alloy Analyzer)
-- ============================================================================

-- Assert: granted permits leave committed cost within the configured limit.
assert AdmissionCommitsWithinLimit {
    all e: CostEnforcer | validTrace[e] implies {
        all t: Time - last, p: Permit, est: Float |
            acquirePermit[e, t, t.next, p, est] implies {
                committedWithinLimit[e, t.next]
            }
    }
}

-- Assert: In-flight count never goes negative
assert InFlightNeverNegative {
    all e: CostEnforcer | validTrace[e] implies {
        all t: Time | I1_InFlightNonNegative[e, t]
    }
}

-- Assert: Reserved cost never goes negative
assert ReservedNeverNegative {
    all e: CostEnforcer | validTrace[e] implies {
        all t: Time | I2_ReservedNonNegative[e, t]
    }
}

-- Assert: Active permits always equals in-flight count
assert ActiveEqualsInFlight {
    all e: CostEnforcer | validTrace[e] implies {
        all t: Time | I3_ActiveEqualsInFlight[e, t]
    }
}

-- Assert: active_permits and Permit.active never diverge.
assert ReleasedPermitsInactive {
    all e: CostEnforcer | validTrace[e] implies {
        all t: Time, p: Permit |
            p in e.active_permits[t] iff p.active[t] = True
        all t: Time - last, p: Permit |
            releasePermit[e, t, t.next, p] implies {
                p not in e.active_permits[t.next]
                p.active[t.next] = False
            }
        all t: Time - last, p: Permit, actual: Float |
            trackCost[e, t, t.next, p, actual] implies {
                p not in e.active_permits[t.next]
                p.active[t.next] = False
            }
    }
}

-- Assert: Double release is idempotent (doesn't corrupt state)
assert DoubleReleaseIdempotent {
    all e: CostEnforcer, p: Permit, t: Time - last |
        (validTrace[e] and doubleRelease[e, t, t.next, p]) implies {
            I1_InFlightNonNegative[e, t.next]
            I2_ReservedNonNegative[e, t.next]
            I3_ActiveEqualsInFlight[e, t.next]
        }
}

-- Assert: Permit IDs are unique
assert UniquePermitIds {
    all e: CostEnforcer | validTrace[e] implies {
        all t: Time | all disj p1, p2: e.active_permits[t] | p1.id != p2.id
    }
}

-- Assert: Permit counter monotonically increases
assert MonotonicPermitCounter {
    all e: CostEnforcer | validTrace[e] implies {
        all t1, t2: Time | I6_MonotonicCounter[e, t1, t2]
    }
}

-- Assert: denied permits use the sentinel id and never enter active state.
assert DeniedPermitProperties {
    all e: CostEnforcer | validTrace[e] implies {
        all t: Time, p: Permit |
            I7_DeniedPermitProperties[e, p, t]
        all t: Time - last, p: Permit, est: Float |
            denyPermit[e, t, t.next, p, est] implies {
                p.id = -1
                p.amount.value = 0
                p.active[t.next] = False
            }
    }
}

-- Assert: Sum of active permit amounts equals reserved cost
assert PermitSumConsistency {
    all e: CostEnforcer | validTrace[e] implies {
        all t: Time | I8_PermitSumEqualsReserved[e, t]
    }
}

-- ============================================================================
-- CHECK COMMANDS (run these in Alloy Analyzer)
-- ============================================================================

check AdmissionCommitsWithinLimit for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int
check InFlightNeverNegative for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int
check ReservedNeverNegative for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int
check ActiveEqualsInFlight for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int
check ReleasedPermitsInactive for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int
check DoubleReleaseIdempotent for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int
check UniquePermitIds for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int
check MonotonicPermitCounter for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int
check DeniedPermitProperties for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int
check PermitSumConsistency for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int

-- ============================================================================
-- RUN COMMANDS (generate example traces)
-- ============================================================================

-- Find a valid trace with at least one granted permit.
run showAcquireReachable {
    some e: CostEnforcer, p: Permit, t: Time - last, estimated: Float |
        validTrace[e] and acquirePermit[e, t, t.next, p, estimated]
} for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int

-- Find a valid trace where denial is triggered after a prior admission.
run showDenyReachable {
    some e: CostEnforcer, pGranted, pDenied: Permit, tGrant, tDeny: Time - last,
         estimatedGranted, estimatedDenied: Float |
        validTrace[e] and
        tGrant in tDeny.prevs and
        acquirePermit[e, tGrant, tGrant.next, pGranted, estimatedGranted] and
        denyPermit[e, tDeny, tDeny.next, pDenied, estimatedDenied]
} for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int

-- Find a valid trace that releases a granted permit.
run showReleaseReachable {
    some e: CostEnforcer, p: Permit, tGrant, tRelease: Time - last, estimated: Float |
        validTrace[e] and
        tGrant in tRelease.prevs and
        acquirePermit[e, tGrant, tGrant.next, p, estimated] and
        releasePermit[e, tRelease, tRelease.next, p]
} for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int

-- Find a trace where actual post-execution cost exceeds the configured limit.
-- This is valid: the guard is pre-trial admission, not exact final spend.
run showTrackCostOverEstimate {
    some e: CostEnforcer, p: Permit, tGrant, tTrack: Time - last,
         estimated, actual: Float |
        validTrace[e] and
        tGrant in tTrack.prevs and
        acquirePermit[e, tGrant, tGrant.next, p, estimated] and
        trackCost[e, tTrack, tTrack.next, p, actual] and
        actual.value > estimated.value and
        committedExceedsLimit[e, tTrack.next]
} for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int

-- Find a double-release scenario
run showDoubleRelease {
    some e: CostEnforcer, p: Permit, tGrant, tRelease, tDouble: Time - last,
         estimated: Float |
        validTrace[e] and
        tGrant in tRelease.prevs and
        tRelease in tDouble.prevs and
        acquirePermit[e, tGrant, tGrant.next, p, estimated] and
        releasePermit[e, tRelease, tRelease.next, p] and
        doubleRelease[e, tDouble, tDouble.next, p]
} for 3 but 4 Time, 3 Permit, 4 Float, 1 CostEnforcer, 5 Int
