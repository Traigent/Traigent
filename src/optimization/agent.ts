import { ValidationError } from '../core/errors.js';
import { cloneAndFreezePlainValue } from '../core/immutable.js';
import { isPlainObject } from '../core/is-plain-object.js';
import { withRuntimeMetricsCollector } from '../core/runtime-metrics.js';
import type { Metrics, TrialConfig } from '../dtos/trial.js';
import type {
  AggregationStrategy,
  AgentCustomEvaluator,
  EvaluationAggregationMap,
  EvaluationContext,
  InjectionMode,
  NativeTrialFunctionResult,
  NormalizedOptimizationSpec,
} from './types.js';
import { collectMetricSamples } from './native-reps.js';

type AnyFunction = (...args: unknown[]) => unknown;
type ContextCustomEvaluator = (context: EvaluationContext) => Metrics | Promise<Metrics>;
type LegacyCustomEvaluator = (
  agentFn: (input: unknown) => unknown | Promise<unknown>,
  config: TrialConfig['config'],
  row: unknown
) => Metrics | Promise<Metrics>;

function isLegacyCustomEvaluator(
  customEvaluator: AgentCustomEvaluator | undefined
): customEvaluator is LegacyCustomEvaluator {
  if (!customEvaluator) {
    return false;
  }
  return customEvaluator.length >= 3;
}

function isContextCustomEvaluator(
  customEvaluator: AgentCustomEvaluator | undefined
): customEvaluator is ContextCustomEvaluator {
  return Boolean(customEvaluator) && !isLegacyCustomEvaluator(customEvaluator);
}

export async function resolveEvaluationRows(
  spec: Pick<NormalizedOptimizationSpec, 'evaluation'>
): Promise<readonly unknown[]> {
  if (spec.evaluation?.data) {
    return spec.evaluation.data;
  }
  if (spec.evaluation?.loadData) {
    return spec.evaluation.loadData();
  }

  throw new ValidationError(
    'optimize() requires evaluation.data or evaluation.loadData for agent optimization.'
  );
}

function getInputFromRow(row: unknown, inputField: string | undefined): unknown {
  if (inputField === undefined) {
    if (isPlainObject(row) && 'input' in row) {
      return row['input'];
    }
    return row;
  }

  if (!isPlainObject(row) || !(inputField in row)) {
    throw new ValidationError(`Evaluation row is missing input field "${inputField}".`);
  }

  return row[inputField];
}

function getExpectedOutputFromRow(row: unknown, expectedField: string | undefined): unknown {
  if (expectedField !== undefined) {
    if (!isPlainObject(row) || !(expectedField in row)) {
      throw new ValidationError(`Evaluation row is missing expected field "${expectedField}".`);
    }
    return row[expectedField];
  }

  if (isPlainObject(row) && 'expected_output' in row) {
    return row['expected_output'];
  }
  if (isPlainObject(row) && 'output' in row) {
    return row['output'];
  }

  return undefined;
}

function getAggregationStrategy(
  aggregation: AggregationStrategy | EvaluationAggregationMap | undefined,
  metric: string
): AggregationStrategy {
  if (!aggregation) {
    if (
      metric === 'cost' ||
      metric === 'input_cost' ||
      metric === 'output_cost' ||
      metric === 'total_cost' ||
      metric === 'input_tokens' ||
      metric === 'output_tokens' ||
      metric === 'total_tokens'
    ) {
      return 'sum';
    }
    return 'mean';
  }

  if (typeof aggregation === 'string') {
    return aggregation;
  }

  return aggregation[metric] ?? aggregation.default ?? 'mean';
}

function aggregateValues(values: number[], strategy: AggregationStrategy): number {
  if (values.length === 0) {
    return 0;
  }

  switch (strategy) {
    case 'median': {
      const sorted = [...values].sort((left, right) => left - right);
      const midpoint = Math.floor(sorted.length / 2);
      return sorted.length % 2 === 0
        ? (sorted[midpoint - 1]! + sorted[midpoint]!) / 2
        : sorted[midpoint]!;
    }
    case 'sum':
      return values.reduce((total, value) => total + value, 0);
    case 'min':
      return values.reduce((smallest, value) => Math.min(smallest, value), values[0]!);
    case 'max':
      return values.reduce((largest, value) => Math.max(largest, value), values[0]!);
    case 'mean':
    default:
      return values.reduce((total, value) => total + value, 0) / values.length;
  }
}

function mergeParameterConfig(args: readonly unknown[], config: TrialConfig['config']): unknown[] {
  if (args.length === 0) {
    return [cloneAndFreezePlainValue({ ...config })];
  }

  if (args.length === 1) {
    return [args[0], cloneAndFreezePlainValue({ ...config })];
  }

  const updatedArgs = [...args];
  const currentConfig = updatedArgs[1];

  if (currentConfig === undefined) {
    updatedArgs[1] = cloneAndFreezePlainValue({ ...config });
    return updatedArgs;
  }

  if (!isPlainObject(currentConfig)) {
    throw new ValidationError(
      'Parameter injection expects the second argument to be an object config.'
    );
  }

  updatedArgs[1] = cloneAndFreezePlainValue({
    ...config,
    ...currentConfig,
  });
  return updatedArgs;
}

export function invokeFunctionWithConfig<T extends AnyFunction>(
  fn: T,
  thisArg: unknown,
  args: readonly unknown[],
  config: TrialConfig['config'],
  injectionMode: InjectionMode
): ReturnType<T> {
  if (injectionMode === 'parameter') {
    return fn.apply(thisArg, mergeParameterConfig(args, config)) as ReturnType<T>;
  }

  return fn.apply(thisArg, [...args]) as ReturnType<T>;
}

async function mapWithConcurrency<T, U>(
  values: readonly T[],
  concurrency: number,
  worker: (value: T, index: number) => Promise<U>
): Promise<U[]> {
  if (values.length === 0) {
    return [];
  }

  if (concurrency <= 1 || values.length === 1) {
    const results: U[] = [];
    for (const [index, value] of values.entries()) {
      results.push(await worker(value, index));
    }
    return results;
  }

  const results = new Array<U>(values.length);
  let nextIndex = 0;

  const runners = Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (nextIndex < values.length) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      results[currentIndex] = await worker(values[currentIndex]!, currentIndex);
    }
  });

  await Promise.all(runners);
  return results;
}

function selectRows(rows: readonly unknown[], trialConfig: TrialConfig): readonly unknown[] {
  if (trialConfig.dataset_subset.inline_rows) {
    return trialConfig.dataset_subset.inline_rows;
  }

  const indices =
    trialConfig.dataset_subset.indices.length > 0
      ? trialConfig.dataset_subset.indices
      : Array.from({ length: rows.length }, (_, index) => index);

  return indices.map((index) => rows[index]);
}

function getPrimaryObjectiveMetric(spec: NormalizedOptimizationSpec): string {
  return spec.objectives[0]?.metric ?? 'accuracy';
}

function ensureExpectedOutputIfNeeded(row: unknown, spec: NormalizedOptimizationSpec): unknown {
  const expectedOutput = getExpectedOutputFromRow(row, spec.evaluation?.expectedField);

  if (expectedOutput === undefined && spec.evaluation?.scoringFunction) {
    throw new ValidationError(
      'Evaluation rows must include an expected output field for the configured evaluator.'
    );
  }

  return expectedOutput;
}

async function evaluateExample(
  spec: NormalizedOptimizationSpec,
  row: unknown,
  output: unknown,
  runtimeMetrics: Metrics,
  config: TrialConfig['config']
): Promise<Metrics> {
  const expectedOutput = ensureExpectedOutputIfNeeded(row, spec);

  if (isContextCustomEvaluator(spec.evaluation?.customEvaluator)) {
    const metrics = await spec.evaluation.customEvaluator({
      output,
      expectedOutput,
      runtimeMetrics,
      row,
      config,
    } satisfies EvaluationContext);
    return {
      ...runtimeMetrics,
      ...metrics,
    };
  }

  const metrics: Metrics = { ...runtimeMetrics };
  const primaryMetric = getPrimaryObjectiveMetric(spec);

  if (spec.evaluation?.scoringFunction) {
    metrics[primaryMetric] = await spec.evaluation.scoringFunction(
      output,
      expectedOutput,
      runtimeMetrics,
      row
    );
  }

  if (spec.evaluation?.metricFunctions) {
    for (const [metric, fn] of Object.entries(spec.evaluation.metricFunctions)) {
      if (metric === primaryMetric && metrics[metric] !== undefined) {
        continue;
      }
      metrics[metric] = await fn(output, expectedOutput, runtimeMetrics, row);
    }
  }

  return metrics;
}

function aggregateMetrics(
  spec: NormalizedOptimizationSpec,
  exampleMetrics: readonly Metrics[]
): Metrics {
  const aggregated: Metrics = {};
  const metricNames = new Set<string>();

  for (const metrics of exampleMetrics) {
    for (const key of Object.keys(metrics)) {
      metricNames.add(key);
    }
  }

  for (const metricName of metricNames) {
    const values = exampleMetrics
      .map((metrics) => metrics[metricName])
      .filter((value): value is number => typeof value === 'number');

    if (values.length === 0) {
      aggregated[metricName] = null;
      continue;
    }

    aggregated[metricName] = aggregateValues(
      values,
      getAggregationStrategy(spec.evaluation?.aggregation, metricName)
    );
  }

  return aggregated;
}

function isObjectiveMetricSatisfied(metrics: Metrics, metricName: string): boolean {
  return typeof metrics[metricName] === 'number' && Number.isFinite(metrics[metricName]);
}

function validateObjectiveMetrics(spec: NormalizedOptimizationSpec, metrics: Metrics): void {
  for (const objective of spec.objectives) {
    if (!isObjectiveMetricSatisfied(metrics, objective.metric)) {
      throw new ValidationError(
        `Agent evaluation did not produce numeric metric "${objective.metric}".`
      );
    }
  }
}

function shouldCollectMetricSamples(spec: NormalizedOptimizationSpec): boolean {
  return spec.promotionPolicy !== undefined;
}

export function createAgentTrialFunction<T extends AnyFunction>(
  agentFn: T,
  spec: NormalizedOptimizationSpec,
  rows: readonly unknown[]
): (trialConfig: TrialConfig) => Promise<NativeTrialFunctionResult> {
  return async function runAgentTrial(
    trialConfig: TrialConfig
  ): Promise<NativeTrialFunctionResult> {
    const selectedRows = selectRows(rows, trialConfig);
    const exampleMetrics = await mapWithConcurrency(
      selectedRows,
      spec.execution.exampleConcurrency,
      async (row) => {
        const agentInput = getInputFromRow(row, spec.evaluation?.inputField);
        const startedAt = Date.now();
        const customEvaluator = spec.evaluation?.customEvaluator;
        const runAgentWithConfig = (input: unknown): unknown | Promise<unknown> =>
          invokeFunctionWithConfig(
            agentFn,
            undefined,
            [input],
            trialConfig.config,
            spec.injection.mode
          );

        if (isLegacyCustomEvaluator(customEvaluator)) {
          const { result: metricsResult, metrics: collectedMetrics } =
            await withRuntimeMetricsCollector(async () =>
              customEvaluator(runAgentWithConfig, { ...trialConfig.config }, row)
            );

          const metrics: Metrics = {
            ...collectedMetrics,
            ...((metricsResult ?? {}) as Metrics),
          };

          if (typeof metrics['latency'] !== 'number') {
            metrics['latency'] = (Date.now() - startedAt) / 1000;
          }

          return metrics;
        }

        const { result: output, metrics: collectedMetrics } = await withRuntimeMetricsCollector(
          async () => runAgentWithConfig(agentInput)
        );

        const runtimeMetrics: Metrics = {
          ...collectedMetrics,
          latency:
            typeof collectedMetrics['latency'] === 'number'
              ? (collectedMetrics['latency'] as number)
              : (Date.now() - startedAt) / 1000,
        };

        return evaluateExample(spec, row, output, runtimeMetrics, trialConfig.config);
      }
    );

    const aggregatedMetrics = aggregateMetrics(spec, exampleMetrics);
    validateObjectiveMetrics(spec, aggregatedMetrics);

    return {
      metrics: aggregatedMetrics,
      metadata: {
        evaluatedRows: selectedRows.length,
        ...(shouldCollectMetricSamples(spec)
          ? { metricSamples: collectMetricSamples(exampleMetrics) }
          : {}),
      },
    };
  };
}
