import type { Metrics } from '../dtos/trial.js';
import type {
  NormalizedOptimizationSpec,
  OptimizationTrialRecord,
} from './types.js';
import { ensureFiniteNumber } from './number-utils.js';

export function normalizeCostMetrics(metrics: Metrics): Metrics {
  const normalized: Metrics = { ...metrics };
  const inputCost = normalized['input_cost'];
  const outputCost = normalized['output_cost'];
  const totalCost = normalized['total_cost'];
  const cost = normalized['cost'];

  const normalizedInputCost =
    typeof inputCost === 'number' && Number.isFinite(inputCost)
      ? inputCost
      : undefined;
  const normalizedOutputCost =
    typeof outputCost === 'number' && Number.isFinite(outputCost)
      ? outputCost
      : undefined;

  let normalizedTotalCost =
    typeof totalCost === 'number' && Number.isFinite(totalCost)
      ? totalCost
      : undefined;

  if (normalizedTotalCost === undefined) {
    if (
      normalizedInputCost !== undefined &&
      normalizedOutputCost !== undefined
    ) {
      normalizedTotalCost = normalizedInputCost + normalizedOutputCost;
    } else if (typeof cost === 'number' && Number.isFinite(cost)) {
      normalizedTotalCost = cost;
    }
  }

  if (normalizedInputCost !== undefined) {
    normalized['input_cost'] = normalizedInputCost;
  }
  if (normalizedOutputCost !== undefined) {
    normalized['output_cost'] = normalizedOutputCost;
  }
  if (normalizedTotalCost !== undefined) {
    normalized['total_cost'] = normalizedTotalCost;
    normalized['cost'] = normalizedTotalCost;
  }

  return normalized;
}

export function extractTrialCost(metrics: Metrics): number {
  const totalCost = metrics['total_cost'];
  if (typeof totalCost === 'number' && Number.isFinite(totalCost)) {
    return totalCost;
  }

  const cost = metrics['cost'];
  if (typeof cost === 'number' && Number.isFinite(cost)) {
    return cost;
  }

  return 0;
}

export function assertTrialCostMetricAvailable(
  spec: NormalizedOptimizationSpec,
  trialRecord: OptimizationTrialRecord,
): void {
  if (spec.budget?.maxCostUsd !== undefined) {
    const costMetric =
      typeof trialRecord.metrics['total_cost'] === 'number'
        ? trialRecord.metrics['total_cost']
        : trialRecord.metrics['cost'];
    ensureFiniteNumber(
      costMetric,
      'budget.maxCostUsd requires every trial to return numeric metrics.total_cost or metrics.cost.',
    );
  }
}
