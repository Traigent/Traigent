import type { Metrics } from '../dtos/trial.js';
import type { RepetitionAggregationStrategy } from './types.js';
import { normalizeCostMetrics } from './native-cost.js';

function aggregateNumericValues(
  values: number[],
  strategy: RepetitionAggregationStrategy,
): number {
  switch (strategy) {
    case 'median': {
      const sorted = [...values].sort((left, right) => left - right);
      const midpoint = Math.floor(sorted.length / 2);
      return sorted.length % 2 === 0
        ? (sorted[midpoint - 1]! + sorted[midpoint]!) / 2
        : sorted[midpoint]!;
    }
    case 'min':
      return values.reduce((smallest, value) => Math.min(smallest, value), values[0]!);
    case 'max':
      return values.reduce((largest, value) => Math.max(largest, value), values[0]!);
    case 'mean':
    default:
      return values.reduce((sum, value) => sum + value, 0) / values.length;
  }
}

export function aggregateRepetitionMetrics(
  metricsList: readonly Metrics[],
  strategy: RepetitionAggregationStrategy,
): Metrics {
  const aggregated: Metrics = {};
  const metricNames = new Set<string>();

  for (const metrics of metricsList) {
    for (const name of Object.keys(metrics)) {
      metricNames.add(name);
    }
  }

  for (const metricName of metricNames) {
    const values = metricsList
      .map((metrics) => metrics[metricName])
      .filter(
        (value): value is number => typeof value === 'number' && Number.isFinite(value),
      );

    aggregated[metricName] =
      values.length === 0 ? null : aggregateNumericValues(values, strategy);
  }

  return normalizeCostMetrics(aggregated);
}

export function collectMetricSamples(
  metricsList: readonly Metrics[],
): Record<string, number[]> {
  const collected: Record<string, number[]> = {};

  for (const metrics of metricsList) {
    for (const [metricName, value] of Object.entries(metrics)) {
      if (typeof value !== 'number' || !Number.isFinite(value)) {
        continue;
      }
      const samples = collected[metricName] ?? [];
      samples.push(value);
      collected[metricName] = samples;
    }
  }

  return collected;
}

export function mergeMetricSamples(
  sampleMaps: ReadonlyArray<Record<string, readonly number[]> | undefined>,
): Record<string, number[]> {
  const merged: Record<string, number[]> = {};

  for (const sampleMap of sampleMaps) {
    if (!sampleMap) {
      continue;
    }

    for (const [metricName, values] of Object.entries(sampleMap)) {
      const numericValues = values.filter((value) => Number.isFinite(value));
      if (numericValues.length === 0) {
        continue;
      }

      const existing = merged[metricName] ?? [];
      existing.push(...numericValues);
      merged[metricName] = existing;
    }
  }

  return merged;
}
