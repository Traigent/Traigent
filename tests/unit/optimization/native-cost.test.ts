import { describe, expect, it } from 'vitest';

import { ValidationError } from '../../../src/core/errors.js';
import {
  assertTrialCostMetricAvailable,
  extractTrialCost,
  normalizeCostMetrics,
} from '../../../src/optimization/native-cost.js';
import type { NormalizedOptimizationSpec } from '../../../src/optimization/types.js';

function createNormalizedSpec(
  overrides: Partial<NormalizedOptimizationSpec> = {},
): NormalizedOptimizationSpec {
  return {
    configurationSpace: {
      model: {
        type: 'enum',
        values: ['a', 'b'],
      },
    },
    objectives: [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
    defaultConfig: {},
    constraints: [],
    safetyConstraints: [],
    injection: {
      mode: 'context',
    },
    execution: {
      mode: 'native',
      contract: 'trial',
      repsPerTrial: 1,
      repsAggregation: 'mean',
    },
    evaluation: {
      data: [{ id: 1 }],
    },
    ...overrides,
  };
}

describe('native cost helpers', () => {
  it('normalizes total cost from detailed or legacy cost metrics', () => {
    expect(
      normalizeCostMetrics({
        input_cost: 0.2,
        output_cost: 0.3,
      }),
    ).toEqual({
      input_cost: 0.2,
      output_cost: 0.3,
      total_cost: 0.5,
      cost: 0.5,
    });

    expect(
      normalizeCostMetrics({
        cost: 0.7,
      }),
    ).toEqual({
      cost: 0.7,
      total_cost: 0.7,
    });
  });

  it('extracts total cost with fallback to cost and zero', () => {
    expect(extractTrialCost({ total_cost: 1.2, cost: 3 })).toBe(1.2);
    expect(extractTrialCost({ cost: 0.8 })).toBe(0.8);
    expect(extractTrialCost({ accuracy: 1 })).toBe(0);
  });

  it('requires a numeric cost metric when maxCostUsd is configured', () => {
    const spec = createNormalizedSpec({
      budget: {
        maxCostUsd: 1,
      },
    });

    expect(() =>
      assertTrialCostMetricAvailable(spec, {
        trialId: 'trial-1',
        trialNumber: 1,
        config: { model: 'a' },
        metrics: { accuracy: 0.9 },
        status: 'completed',
        metadata: {},
      }),
    ).toThrowError(ValidationError);
  });
});
