import { ValidationError } from '../core/errors.js';
import type { Metrics } from '../dtos/trial.js';
import type {
  NormalizedOptimizationSpec,
  OptimizationConstraint,
  OptimizationTrialRecord,
  SafetyConstraint,
} from './types.js';
import type { CandidateConfig } from './native-space.js';
import { buildPromotionDecision, getPromotionRejectionReason } from './native-promotion.js';

export function constraintName(
  constraint: OptimizationConstraint | SafetyConstraint,
  fallback: string
): string {
  return constraint.name || fallback;
}

export function constraintRequiresMetrics(
  constraint: OptimizationConstraint | SafetyConstraint
): boolean {
  // Prefer an explicit marker when present. Arity is only a convenience
  // fallback for simple callbacks and is fragile with default/rest params.
  return (
    constraint.length >= 2 || (constraint as { requiresMetrics?: boolean }).requiresMetrics === true
  );
}

export function evaluatePreTrialConstraints(
  spec: NormalizedOptimizationSpec,
  config: CandidateConfig,
  toErrorMessage: (error: unknown) => string
): boolean {
  for (const constraint of spec.constraints) {
    if (constraintRequiresMetrics(constraint)) {
      continue;
    }

    let passed = false;
    try {
      passed = constraint(config);
    } catch (error) {
      throw new ValidationError(
        `Constraint "${constraintName(constraint, 'constraint')}" failed during pre-trial validation: ${toErrorMessage(error)}`
      );
    }

    if (!passed) {
      return false;
    }
  }

  return true;
}

export function validatePostTrialConstraints(
  spec: NormalizedOptimizationSpec,
  config: CandidateConfig,
  metrics: Metrics,
  toErrorMessage: (error: unknown) => string
): string | undefined {
  for (const constraint of spec.constraints) {
    if (!constraintRequiresMetrics(constraint)) {
      continue;
    }

    let passed = false;
    try {
      passed = constraint(config, metrics);
    } catch (error) {
      return `Constraint "${constraintName(constraint, 'constraint')}" failed during post-trial validation: ${toErrorMessage(error)}`;
    }

    if (!passed) {
      return `Constraint "${constraintName(constraint, 'constraint')}" rejected the trial configuration.`;
    }
  }

  for (const constraint of spec.safetyConstraints) {
    let passed = false;
    try {
      passed = constraint(config, metrics);
    } catch (error) {
      return `Safety constraint "${constraintName(constraint, 'safetyConstraint')}" failed during post-trial validation: ${toErrorMessage(error)}`;
    }

    if (!passed) {
      return `Safety constraint "${constraintName(constraint, 'safetyConstraint')}" rejected the trial configuration.`;
    }
  }

  return undefined;
}

export function getCompletedTrials(
  trials: readonly OptimizationTrialRecord[]
): OptimizationTrialRecord[] {
  return trials.filter((trial) => trial.status !== 'rejected');
}

export function applyPostTrialGuards<
  T extends {
    status: 'completed' | 'rejected' | 'timeout' | 'error' | 'cancelled';
    actualCostUsd: number;
    evaluatedExamples: number;
    record?: OptimizationTrialRecord;
  },
>(spec: NormalizedOptimizationSpec, outcome: T, toErrorMessage: (error: unknown) => string): T {
  if (outcome.status !== 'completed' || !outcome.record) {
    return outcome;
  }

  const rejectionReason = validatePostTrialConstraints(
    spec,
    outcome.record.config,
    outcome.record.metrics,
    toErrorMessage
  );

  if (!rejectionReason) {
    const promotionRejectionReason = getPromotionRejectionReason(
      outcome.record,
      spec.promotionPolicy
    );
    if (!promotionRejectionReason) {
      return outcome;
    }

    const promotionDecision = buildPromotionDecision(
      outcome.record,
      undefined,
      spec.objectives,
      spec.promotionPolicy
    );

    return {
      ...outcome,
      status: 'rejected',
      record: {
        ...outcome.record,
        status: 'rejected',
        errorMessage: promotionRejectionReason,
        ...(promotionDecision ? { promotionDecision } : {}),
        metadata: {
          ...outcome.record.metadata,
          rejectionReason: promotionRejectionReason,
        },
      },
    };
  }

  return {
    ...outcome,
    status: 'rejected',
    record: {
      ...outcome.record,
      status: 'rejected',
      errorMessage: rejectionReason,
      metadata: {
        ...outcome.record.metadata,
        rejectionReason,
      },
    },
  };
}
