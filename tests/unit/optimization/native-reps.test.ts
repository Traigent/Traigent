import { describe, expect, it } from 'vitest';

import { aggregateRepetitionMetrics } from '../../../src/optimization/native-reps.js';

describe('native repetition aggregation', () => {
  const metricsList = [
    {
      accuracy: 0.2,
      latency: 120,
      input_cost: 0.1,
      output_cost: 0.2,
    },
    {
      accuracy: 0.5,
      latency: 90,
      input_cost: 0.2,
      output_cost: 0.3,
    },
    {
      accuracy: 0.9,
      latency: 60,
      input_cost: 0.3,
      output_cost: 0.4,
    },
  ] as const;

  it('aggregates by mean and normalizes cost aliases', () => {
    const result = aggregateRepetitionMetrics(metricsList, 'mean');

    expect(result.accuracy).toBeCloseTo((0.2 + 0.5 + 0.9) / 3, 10);
    expect(result.latency).toBe(90);
    expect(result.input_cost).toBeCloseTo(0.2, 10);
    expect(result.output_cost).toBeCloseTo(0.3, 10);
    expect(result.total_cost).toBeCloseTo(0.5, 10);
    expect(result.cost).toBeCloseTo(0.5, 10);
  });

  it('aggregates by median, min, and max', () => {
    expect(aggregateRepetitionMetrics(metricsList, 'median').accuracy).toBe(0.5);
    expect(aggregateRepetitionMetrics(metricsList, 'min').latency).toBe(60);
    expect(aggregateRepetitionMetrics(metricsList, 'max').latency).toBe(120);
  });

  it('returns null for metrics with no finite numeric values', () => {
    expect(
      aggregateRepetitionMetrics(
        [
          { accuracy: Number.NaN, notes: null },
          { accuracy: Number.POSITIVE_INFINITY, notes: 'ignored' },
        ],
        'mean'
      )
    ).toEqual({
      accuracy: null,
      notes: null,
    });
  });
});
