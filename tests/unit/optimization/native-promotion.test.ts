import { describe, expect, it } from 'vitest';

import {
  benjaminiHochbergAdjust,
  buildPromotionDecision,
  clopperPearsonLowerBound,
  compareTrialsWithStatisticalPromotion,
  evaluatePromotionChanceConstraints,
  getPromotionRejectionReason,
  getTrialMetricSamples,
} from '../../../src/optimization/native-promotion.js';
import type { OptimizationTrialRecord } from '../../../src/optimization/types.js';

function createTrial(
  trialId: string,
  metrics: OptimizationTrialRecord['metrics'],
  metadata?: Record<string, unknown>,
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

describe('native promotion helpers', () => {
  it('computes a clopper-pearson lower bound for explicit counts', () => {
    expect(clopperPearsonLowerBound(95, 100, 0.95)).toBeGreaterThan(0.8);
    expect(clopperPearsonLowerBound(70, 100, 0.95)).toBeLessThan(0.8);
  });

  it('uses the small-sample beta path for exact-ish lower bounds', () => {
    expect(clopperPearsonLowerBound(0, 10, 0.95)).toBe(0);
    expect(clopperPearsonLowerBound(8, 10, 0.95)).toBeCloseTo(0.4439, 3);
  });

  it('validates clopper-pearson inputs and BH-adjusts multi-objective p-values', () => {
    expect(() => clopperPearsonLowerBound(1, 0, 0.95)).toThrow(/trials must be positive/i);
    expect(() => clopperPearsonLowerBound(11, 10, 0.95)).toThrow(/successes must be in/i);
    expect(() => clopperPearsonLowerBound(1, 10, 1)).toThrow(/confidence must be in/i);

    expect(benjaminiHochbergAdjust([])).toEqual([]);
    expect(benjaminiHochbergAdjust([0.2])).toEqual([0.2]);
    expect(benjaminiHochbergAdjust([0.01, 0.04, 0.03])).toEqual([0.03, 0.04, 0.04]);
  });

  it('evaluates chance constraints from explicit counts metadata', () => {
    const results = evaluatePromotionChanceConstraints(
      createTrial(
        'trial-1',
        { accuracy: 0.9 },
        {
          chanceConstraintCounts: {
            safety: { successes: 95, trials: 100 },
          },
        },
      ),
      {
        chanceConstraints: [
          {
            name: 'safety',
            threshold: 0.8,
            confidence: 0.95,
          },
        ],
      },
    );

    expect(results).toEqual([
      expect.objectContaining({
        name: 'safety',
        satisfied: true,
      }),
    ]);
  });

  it('derives chance constraint counts from binary metric samples', () => {
    const results = evaluatePromotionChanceConstraints(
      createTrial(
        'trial-2',
        { safety: 0.75 },
        {
          metricSamples: {
            safety: [1, 1, 1, 1, 0],
          },
        },
      ),
      {
        chanceConstraints: [
          {
            name: 'safety',
            threshold: 0.3,
            confidence: 0.8,
          },
        ],
      },
    );

    expect(results[0]).toMatchObject({
      name: 'safety',
      observedRate: 0.8,
      satisfied: true,
    });
  });

  it('returns undefined for malformed metric samples and rejects invalid explicit counts', () => {
    expect(
      getTrialMetricSamples(
        createTrial('trial-2b', { safety: 0.75 }, {
          metricSamples: { safety: [1, Number.NaN, 0] },
        }),
        'safety',
      ),
    ).toBeUndefined();

    expect(() =>
      evaluatePromotionChanceConstraints(
        createTrial('trial-2c', { safety: 0.75 }, {
          chanceConstraintCounts: {
            safety: { successes: 3, trials: 2 },
          },
        }),
        {
          chanceConstraints: [
            {
              name: 'safety',
              threshold: 0.5,
              confidence: 0.95,
            },
          ],
        },
      ),
    ).toThrow(/invalid chanceConstraintCounts/i);
  });

  it('covers non-array sample metadata and additional statistical edge paths', () => {
    expect(
      getTrialMetricSamples(
        createTrial('trial-2d', { safety: 0.75 }, {
          metricSamples: { safety: 1 },
        }),
        'safety',
      ),
    ).toBeUndefined();

    expect(
      getPromotionRejectionReason(
        createTrial('trial-2e', { safety: 0.75 }, {
          chanceConstraintCounts: { safety: 1 },
        }),
        {
          chanceConstraints: [
            {
              name: 'safety',
              threshold: 0.5,
              confidence: 0.95,
            },
          ],
        },
      ),
    ).toMatch(/chance constraints rejected the trial/i);

    const largeCandidateSamples = Array.from({ length: 101 }, (_, index) => 0.86 + (index % 3) * 0.01);
    const largeIncumbentSamples = Array.from({ length: 101 }, (_, index) => 0.72 + (index % 4) * 0.01);
    expect(
      compareTrialsWithStatisticalPromotion(
        createTrial('trial-2f', { accuracy: 0.88 }, { metricSamples: { accuracy: largeCandidateSamples } }),
        createTrial('trial-2g', { accuracy: 0.75 }, { metricSamples: { accuracy: largeIncumbentSamples } }),
        [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
        { alpha: 0.05 },
      ),
    ).toBe(1);

    expect(
      compareTrialsWithStatisticalPromotion(
        createTrial('trial-2h', { accuracy: 0.85 }, { metricSamples: { accuracy: [0.7, 0.9, 1.1, 0.7] } }),
        createTrial('trial-2i', { accuracy: 0.85 }, { metricSamples: { accuracy: [0.9, 0.7, 0.9, 0.9] } }),
        [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
        { alpha: 0.05 },
      ),
    ).toBe(0);

    expect(
      compareTrialsWithStatisticalPromotion(
        createTrial('trial-2j', { stability: 1.2 }, { metricSamples: { stability: [1.2] } }),
        createTrial('trial-2k', { stability: 0.85 }, { metricSamples: { stability: [0.85] } }),
        [
          {
            metric: 'stability',
            direction: 'band',
            band: { low: 0.8, high: 0.9, alpha: 0.05 },
            weight: 1,
          },
        ],
        { alpha: 0.05 },
      ),
    ).toBe(-1);
  });

  it('rejects trials when chance constraint data is missing or unusable', () => {
    const missingReason = getPromotionRejectionReason(
      createTrial('trial-3', { accuracy: 0.9 }),
      {
        chanceConstraints: [
          {
            name: 'safety',
            threshold: 0.8,
            confidence: 0.95,
          },
        ],
      },
    );

    expect(missingReason).toMatch(/chance constraints rejected the trial/i);

    const nonBinaryReason = getPromotionRejectionReason(
      createTrial(
        'trial-4',
        { safety: 0.9 },
        {
          metricSamples: {
            safety: [0.8, 0.9, 1],
          },
        },
      ),
      {
        chanceConstraints: [
          {
            name: 'safety',
            threshold: 0.8,
            confidence: 0.95,
          },
        ],
      },
    );

    expect(nonBinaryReason).toMatch(/chance constraints rejected the trial/i);
  });

  it('builds a structured rejection decision for failed chance constraints', () => {
    const decision = buildPromotionDecision(
      createTrial('trial-cc', { accuracy: 0.9 }, {
        chanceConstraintCounts: {
          safety: { successes: 50, trials: 100 },
        },
      }),
      undefined,
      [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
      {
        chanceConstraints: [
          {
            name: 'safety',
            threshold: 0.8,
            confidence: 0.95,
          },
        ],
      },
    );

    expect(decision).toMatchObject({
      decision: 'reject',
      method: 'chance-constraints',
      chanceResults: [
        expect.objectContaining({
          name: 'safety',
          satisfied: false,
        }),
      ],
    });
  });

  it('falls back to deterministic promotion reporting when statistical comparison has no decision', () => {
    const decision = buildPromotionDecision(
      createTrial('trial-det-a', { accuracy: 0.95, cost: 0.1 }, {
        metricSamples: {
          accuracy: [0.84, 0.91, 0.93, 0.89, 0.92],
        },
      }),
      createTrial('trial-det-b', { accuracy: 0.92, cost: 0.2 }, {
        metricSamples: {
          accuracy: [0.83, 0.9, 0.92, 0.88, 0.91],
        },
      }),
      [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
      {
        alpha: 0.05,
        minEffect: { accuracy: 0.05 },
      },
    );

    expect(decision).toMatchObject({
      decision: 'no_decision',
      candidateTrialId: 'trial-det-a',
      incumbentTrialId: 'trial-det-b',
    });
  });

  it('returns undefined for missing statistical samples and rejects malformed band objectives', () => {
    const maximizeObjective = {
      metric: 'accuracy',
      direction: 'maximize',
      weight: 1,
    } as const;

    const missing = compareTrialsWithStatisticalPromotion(
      createTrial('trial-5', { accuracy: 0.8 }, { metricSamples: { accuracy: [0.8, 0.81] } }),
      createTrial('trial-6', { accuracy: 0.82 }, { metricSamples: { accuracy: [0.82] } }),
      [maximizeObjective],
      { alpha: 0.05 },
    );
    expect(missing).toBeUndefined();

    expect(() =>
      compareTrialsWithStatisticalPromotion(
        createTrial('trial-7', { consistency: 0.9 }, { metricSamples: { consistency: [0.9, 0.91] } }),
        createTrial('trial-8', { consistency: 0.88 }, { metricSamples: { consistency: [0.88, 0.89] } }),
        [
          {
            metric: 'consistency',
            direction: 'band',
            weight: 1,
          },
        ],
        { alpha: 0.05 },
      ),
    ).toThrow(/missing band metadata/i);
  });

  it('covers no-policy fast paths and satisfied promotion results', () => {
    const trial = createTrial(
      'trial-9',
      { accuracy: 0.9 },
      {
        chanceConstraintCounts: {
          safety: { successes: 29, trials: 30 },
        },
      },
    );

    expect(evaluatePromotionChanceConstraints(trial, undefined)).toEqual([]);
    expect(getPromotionRejectionReason(trial, undefined)).toBeUndefined();
    expect(
      getPromotionRejectionReason(trial, {
        chanceConstraints: [
          {
            name: 'safety',
            threshold: 0.5,
            confidence: 0.95,
          },
        ],
      }),
    ).toBeUndefined();
  });

  it('supports BH-adjusted multi-objective promotion, minimize objectives, and single-sample band comparisons', () => {
    const candidate = createTrial(
      'trial-10',
      { accuracy: 0.9, latency: 100, consistency: 0.9 },
      {
        metricSamples: {
          accuracy: [0.88, 0.9, 0.92, 0.89, 0.91],
          latency: [100, 102, 98, 101, 99],
          consistency: [0.9],
        },
      },
    );
    const incumbent = createTrial(
      'trial-11',
      { accuracy: 0.75, latency: 150, consistency: 0.7 },
      {
        metricSamples: {
          accuracy: [0.72, 0.75, 0.78, 0.74, 0.76],
          latency: [150, 152, 149, 148, 151],
          consistency: [0.7],
        },
      },
    );

    expect(
      compareTrialsWithStatisticalPromotion(
        candidate,
        incumbent,
        [
          { metric: 'accuracy', direction: 'maximize', weight: 1 },
          { metric: 'latency', direction: 'minimize', weight: 1 },
        ],
        { alpha: 0.05, adjust: 'BH' },
      ),
    ).toBe(1);

    expect(
      compareTrialsWithStatisticalPromotion(
        candidate,
        incumbent,
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
        { alpha: 0.05 },
      ),
    ).toBe(1);
  });

  it('covers zero-variance paired promotion, incumbent-worse rejection, and band-distance fallback ties', () => {
    expect(
      compareTrialsWithStatisticalPromotion(
        createTrial('trial-12', { accuracy: 0.9 }, { metricSamples: { accuracy: [0.9, 0.9, 0.9] } }),
        createTrial('trial-13', { accuracy: 0.8 }, { metricSamples: { accuracy: [0.8, 0.8, 0.8] } }),
        [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
        { alpha: 0.05, minEffect: { accuracy: 0.01 } },
      ),
    ).toBe(1);

    expect(
      compareTrialsWithStatisticalPromotion(
        createTrial('trial-14', { accuracy: 0.7 }, { metricSamples: { accuracy: [0.7, 0.7, 0.7] } }),
        createTrial('trial-15', { accuracy: 0.8 }, { metricSamples: { accuracy: [0.8, 0.8, 0.8] } }),
        [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
        { alpha: 0.05, minEffect: { accuracy: 0.01 } },
      ),
    ).toBe(-1);

    expect(
      compareTrialsWithStatisticalPromotion(
        createTrial('trial-16', { consistency: 0.9 }, { metricSamples: { consistency: [0.89, 0.9, 0.91] } }),
        createTrial('trial-17', { consistency: 0.87 }, { metricSamples: { consistency: [0.86, 0.87, 0.88] } }),
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
        { alpha: 0.05 },
      ),
    ).toBe(1);
  });

  it('returns neutral or incumbent-favoring results for tied and band-worse statistical comparisons', () => {
    expect(
      compareTrialsWithStatisticalPromotion(
        createTrial('trial-18', { accuracy: 0.8 }, { metricSamples: { accuracy: [0.8, 0.8, 0.8] } }),
        createTrial('trial-19', { accuracy: 0.8 }, { metricSamples: { accuracy: [0.8, 0.8, 0.8] } }),
        [{ metric: 'accuracy', direction: 'maximize', weight: 1 }],
        { alpha: 0.05, minEffect: { accuracy: 0.01 } },
      ),
    ).toBe(0);

    expect(
      compareTrialsWithStatisticalPromotion(
        createTrial('trial-20', { consistency: 0.7 }, { metricSamples: { consistency: [0.7] } }),
        createTrial('trial-21', { consistency: 0.9 }, { metricSamples: { consistency: [0.9] } }),
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
        { alpha: 0.05 },
      ),
    ).toBe(-1);
  });

  it('handles zero-variance multi-sample TOST comparisons', () => {
    expect(
      compareTrialsWithStatisticalPromotion(
        createTrial('trial-22', { consistency: 0.9 }, { metricSamples: { consistency: [0.9, 0.9, 0.9] } }),
        createTrial('trial-23', { consistency: 0.8 }, { metricSamples: { consistency: [0.8, 0.8, 0.8] } }),
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
        { alpha: 0.05 },
      ),
    ).toBe(1);
  });
});
