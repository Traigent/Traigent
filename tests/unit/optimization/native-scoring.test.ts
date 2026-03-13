import { describe, expect, it } from 'vitest';

import { ValidationError } from '../../../src/core/errors.js';
import {
  computeSearchScore,
  getObjectiveMetric,
  selectBestTrial,
} from '../../../src/optimization/native-scoring.js';
import type {
  NormalizedObjectiveDefinition,
  OptimizationTrialRecord,
} from '../../../src/optimization/types.js';

function createTrial(
  trialId: string,
  metrics: OptimizationTrialRecord['metrics'],
  metadata?: OptimizationTrialRecord['metadata']
): OptimizationTrialRecord {
  return {
    trialId,
    trialNumber: Number(trialId.replace('trial-', '')),
    config: { variant: trialId },
    metrics,
    duration: 1,
    status: 'completed',
    metadata,
  };
}

describe('native scoring helpers', () => {
  it('reads numeric objective metrics and rejects missing values', () => {
    const objective: NormalizedObjectiveDefinition = {
      metric: 'accuracy',
      direction: 'maximize',
      weight: 1,
    };

    expect(getObjectiveMetric(createTrial('trial-1', { accuracy: 0.9 }), objective)).toBe(0.9);

    expect(() =>
      getObjectiveMetric(createTrial('trial-2', { accuracy: Number.NaN }), objective)
    ).toThrow(/missing numeric metric "accuracy"/i);
  });

  it('scores maximize, minimize, and band objectives', () => {
    const maximizeObjective: NormalizedObjectiveDefinition = {
      metric: 'accuracy',
      direction: 'maximize',
      weight: 1,
    };
    const minimizeObjective: NormalizedObjectiveDefinition = {
      metric: 'latency',
      direction: 'minimize',
      weight: 1,
    };
    const bandObjective: NormalizedObjectiveDefinition = {
      metric: 'response_length',
      direction: 'band',
      weight: 1,
      band: {
        low: 120,
        high: 180,
        test: 'TOST',
        alpha: 0.05,
      },
    };

    expect(computeSearchScore({ accuracy: 0.9 }, [maximizeObjective])).toBe(0.9);
    expect(computeSearchScore({ latency: 120 }, [minimizeObjective])).toBe(-120);
    expect(computeSearchScore({ response_length: 150 }, [bandObjective])).toBe(0);
    expect(computeSearchScore({ response_length: 90 }, [bandObjective])).toBe(-30);
  });

  it('rejects malformed band objectives and missing objective metrics', () => {
    const missingBand: NormalizedObjectiveDefinition = {
      metric: 'response_length',
      direction: 'band',
      weight: 1,
    };

    expect(() => computeSearchScore({ response_length: 150 }, [missingBand])).toThrow(
      ValidationError
    );

    expect(() =>
      computeSearchScore({ accuracy: 0.9 }, [
        {
          metric: 'latency',
          direction: 'minimize',
          weight: 1,
        },
      ])
    ).toThrow(/missing numeric objective "latency"/i);
  });

  it('selects the best trial with default normalized scoring when promotion policy is absent', () => {
    const best = selectBestTrial(
      [
        createTrial('trial-1', { accuracy: 0.8, latency: 130 }),
        createTrial('trial-2', { accuracy: 0.95, latency: 150 }),
        createTrial('trial-3', { accuracy: 0.9, latency: 100 }),
      ],
      [
        { metric: 'accuracy', direction: 'maximize', weight: 1 },
        { metric: 'latency', direction: 'minimize', weight: 1 },
      ]
    );

    expect(best?.trialId).toBe('trial-3');
  });

  it('uses promotion minEffect and tieBreakers when comparing trials', () => {
    const best = selectBestTrial(
      [
        createTrial('trial-1', {
          accuracy: 0.92,
          latency: 100,
          cost: 0.2,
        }),
        createTrial('trial-2', {
          accuracy: 0.95,
          latency: 104,
          cost: 0.1,
        }),
      ],
      [
        { metric: 'accuracy', direction: 'maximize', weight: 1 },
        { metric: 'latency', direction: 'minimize', weight: 1 },
      ],
      {
        dominance: 'epsilon_pareto',
        minEffect: {
          accuracy: 0.05,
          latency: 10,
        },
        tieBreakers: {
          cost: 'minimize',
        },
      }
    );

    expect(best?.trialId).toBe('trial-2');
  });

  it('returns null when there are no completed trials and ignores rejected ones', () => {
    expect(
      selectBestTrial(
        [
          {
            ...createTrial('trial-1', { accuracy: 1 }),
            status: 'rejected',
          },
        ],
        [{ metric: 'accuracy', direction: 'maximize', weight: 1 }]
      )
    ).toBeNull();
  });

  it('treats incomplete tie-breaker metrics as neutral', () => {
    const best = selectBestTrial(
      [
        createTrial('trial-1', { accuracy: 0.9 }),
        createTrial('trial-2', { accuracy: 0.9, cost: 0.2 }),
      ],
      [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
      {
        tieBreakers: {
          cost: 'minimize',
        },
      }
    );

    expect(best?.trialId).toBe('trial-1');
  });

  it('uses tie-breakers after conflicting objective comparisons and preserves incumbent order on full ties', () => {
    const tied = selectBestTrial(
      [
        createTrial('trial-1', { accuracy: 0.9, latency: 120, cost: 0.2 }),
        createTrial('trial-2', { accuracy: 0.95, latency: 140, cost: 0.1 }),
      ],
      [
        { metric: 'accuracy', direction: 'maximize', weight: 1 },
        { metric: 'latency', direction: 'minimize', weight: 1 },
      ],
      {
        minEffect: {
          accuracy: 0.01,
          latency: 5,
        },
        tieBreakers: {
          cost: 'minimize',
        },
      }
    );

    expect(tied?.trialId).toBe('trial-2');

    const stable = selectBestTrial(
      [
        createTrial('trial-1', { accuracy: 0.9, cost: 0.1 }),
        createTrial('trial-2', { accuracy: 0.9, cost: 0.1 }),
      ],
      [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
      {
        tieBreakers: {
          cost: 'minimize',
        },
      }
    );

    expect(stable?.trialId).toBe('trial-1');
  });

  it('uses statistical promotion for standard objectives when metric samples are available', () => {
    const best = selectBestTrial(
      [
        createTrial(
          'trial-1',
          { accuracy: 0.762 },
          {
            metricSamples: {
              accuracy: [0.7, 0.77, 0.82, 0.73, 0.79],
            },
          }
        ),
        createTrial(
          'trial-2',
          { accuracy: 0.898 },
          {
            metricSamples: {
              accuracy: [0.84, 0.91, 0.93, 0.89, 0.92],
            },
          }
        ),
      ],
      [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
      {
        alpha: 0.05,
        minEffect: {
          accuracy: 0.01,
        },
      }
    );

    expect(best?.trialId).toBe('trial-2');
  });

  it('uses TOST-style statistical promotion for banded objectives when metric samples are available', () => {
    const best = selectBestTrial(
      [
        createTrial(
          'trial-1',
          { consistency: 0.71 },
          {
            metricSamples: {
              consistency: [0.7, 0.72, 0.68, 0.71, 0.69],
            },
          }
        ),
        createTrial(
          'trial-2',
          { consistency: 0.89 },
          {
            metricSamples: {
              consistency: [0.89, 0.9, 0.88, 0.91, 0.89],
            },
          }
        ),
      ],
      [
        {
          metric: 'consistency',
          direction: 'band',
          weight: 1,
          band: {
            low: 0.85,
            high: 0.95,
            test: 'TOST',
            alpha: 0.05,
          },
        },
      ],
      {
        alpha: 0.05,
      }
    );

    expect(best?.trialId).toBe('trial-2');
  });
});
