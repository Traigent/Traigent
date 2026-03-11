import { ValidationError } from "../core/errors.js";
import { withRuntimeMetricsCollector } from "../core/runtime-metrics.js";
import type { Metrics, TrialConfig } from "../dtos/trial.js";
import type {
  AggregationStrategy,
  AgentCustomEvaluator,
  EvaluationAggregationMap,
  EvaluationContext,
  InjectionMode,
  NativeTrialFunctionResult,
  NormalizedOptimizationSpec,
} from "./types.js";

type AnyFunction = (...args: any[]) => any;
type ContextCustomEvaluator = (
  context: EvaluationContext,
) => Metrics | Promise<Metrics>;
type LegacyCustomEvaluator = (
  agentFn: (input: unknown) => unknown | Promise<unknown>,
  config: TrialConfig["config"],
  row: unknown,
) => Metrics | Promise<Metrics>;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isLegacyCustomEvaluator(
  customEvaluator: AgentCustomEvaluator | undefined,
): customEvaluator is LegacyCustomEvaluator {
  if (!customEvaluator) {
    return false;
  }
  return customEvaluator.length >= 3;
}

function isContextCustomEvaluator(
  customEvaluator: AgentCustomEvaluator | undefined,
): customEvaluator is ContextCustomEvaluator {
  return Boolean(customEvaluator) && !isLegacyCustomEvaluator(customEvaluator);
}

export async function resolveEvaluationRows(
  spec: Pick<NormalizedOptimizationSpec, "evaluation">,
): Promise<readonly unknown[]> {
  if (spec.evaluation?.data) {
    return spec.evaluation.data;
  }
  if (spec.evaluation?.loadData) {
    return spec.evaluation.loadData();
  }

  throw new ValidationError(
    "optimize() requires evaluation.data or evaluation.loadData for agent optimization.",
  );
}

function getInputFromRow(row: unknown, inputField: string | undefined): unknown {
  if (inputField === undefined) {
    if (isPlainObject(row) && "input" in row) {
      return row["input"];
    }
    return row;
  }

  if (!isPlainObject(row) || !(inputField in row)) {
    throw new ValidationError(
      `Evaluation row is missing input field "${inputField}".`,
    );
  }

  return row[inputField];
}

function getExpectedOutputFromRow(
  row: unknown,
  expectedField: string | undefined,
): unknown {
  if (expectedField !== undefined) {
    if (!isPlainObject(row) || !(expectedField in row)) {
      throw new ValidationError(
        `Evaluation row is missing expected field "${expectedField}".`,
      );
    }
    return row[expectedField];
  }

  if (isPlainObject(row) && "expected_output" in row) {
    return row["expected_output"];
  }
  if (isPlainObject(row) && "output" in row) {
    return row["output"];
  }

  return undefined;
}

function getAggregationStrategy(
  aggregation: AggregationStrategy | EvaluationAggregationMap | undefined,
  metric: string,
): AggregationStrategy {
  if (!aggregation) {
    if (
      metric === "cost" ||
      metric === "input_cost" ||
      metric === "output_cost" ||
      metric === "total_cost" ||
      metric === "input_tokens" ||
      metric === "output_tokens" ||
      metric === "total_tokens"
    ) {
      return "sum";
    }
    return "mean";
  }

  if (typeof aggregation === "string") {
    return aggregation;
  }

  return aggregation[metric] ?? aggregation.default ?? "mean";
}

function aggregateValues(values: number[], strategy: AggregationStrategy): number {
  if (values.length === 0) {
    return 0;
  }

  switch (strategy) {
    case "median": {
      const sorted = [...values].sort((left, right) => left - right);
      const midpoint = Math.floor(sorted.length / 2);
      return sorted.length % 2 === 0
        ? (sorted[midpoint - 1]! + sorted[midpoint]!) / 2
        : sorted[midpoint]!;
    }
    case "sum":
      return values.reduce((total, value) => total + value, 0);
    case "min":
      return values.reduce((smallest, value) => Math.min(smallest, value), values[0]!);
    case "max":
      return values.reduce((largest, value) => Math.max(largest, value), values[0]!);
    case "mean":
    default:
      return values.reduce((total, value) => total + value, 0) / values.length;
  }
}

function mergeParameterConfig(
  args: readonly unknown[],
  config: TrialConfig["config"],
): unknown[] {
  if (args.length === 0) {
    return [{ ...config }];
  }

  if (args.length === 1) {
    return [args[0], { ...config }];
  }

  const updatedArgs = [...args];
  const currentConfig = updatedArgs[1];

  if (currentConfig === undefined) {
    updatedArgs[1] = { ...config };
    return updatedArgs;
  }

  if (!isPlainObject(currentConfig)) {
    throw new ValidationError(
      "Parameter injection expects the second argument to be an object config.",
    );
  }

  updatedArgs[1] = {
    ...config,
    ...currentConfig,
  };
  return updatedArgs;
}

export function invokeFunctionWithConfig<T extends AnyFunction>(
  fn: T,
  thisArg: unknown,
  args: readonly unknown[],
  config: TrialConfig["config"],
  injectionMode: InjectionMode,
): ReturnType<T> {
  if (injectionMode === "parameter") {
    return fn.apply(thisArg, mergeParameterConfig(args, config)) as ReturnType<T>;
  }

  return fn.apply(thisArg, [...args]) as ReturnType<T>;
}

function selectRows(
  rows: readonly unknown[],
  trialConfig: TrialConfig,
): readonly unknown[] {
  const inlineRows = (trialConfig.dataset_subset as TrialConfig["dataset_subset"] & {
    inline_rows?: readonly unknown[];
  }).inline_rows;
  if (inlineRows) {
    return inlineRows;
  }

  const indices =
    trialConfig.dataset_subset.indices.length > 0
      ? trialConfig.dataset_subset.indices
      : Array.from({ length: rows.length }, (_, index) => index);

  return indices.map((index) => {
    if (!Number.isInteger(index) || index < 0 || index >= rows.length) {
      throw new ValidationError(
        `Evaluation row index ${String(index)} is out of bounds for ${rows.length} row(s).`,
      );
    }
    return rows[index];
  });
}

function getPrimaryObjectiveMetric(spec: NormalizedOptimizationSpec): string {
  const primaryObjective = spec.objectives[0];
  if (!primaryObjective) {
    return "accuracy";
  }
  return primaryObjective.metric;
}

function ensureExpectedOutputIfNeeded(
  row: unknown,
  spec: NormalizedOptimizationSpec,
): unknown {
  const expectedOutput = getExpectedOutputFromRow(
    row,
    spec.evaluation?.expectedField,
  );

  if (expectedOutput === undefined && spec.evaluation?.scoringFunction) {
    throw new ValidationError(
      "Evaluation rows must include an expected output field for the configured evaluator.",
    );
  }

  return expectedOutput;
}

async function evaluateExample(
  spec: NormalizedOptimizationSpec,
  row: unknown,
  output: unknown,
  runtimeMetrics: Metrics,
  config: TrialConfig["config"],
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
    const score = await spec.evaluation.scoringFunction(
      output,
      expectedOutput,
      runtimeMetrics,
      row,
    );
    metrics[primaryMetric] = score ?? null;
  }

  if (spec.evaluation?.metricFunctions) {
    for (const [metricName, metricFn] of Object.entries(
      spec.evaluation.metricFunctions,
    )) {
      if (metricName === primaryMetric && metrics[metricName] !== undefined) {
        continue;
      }
      const value = await metricFn(output, expectedOutput, runtimeMetrics, row);
      metrics[metricName] = value ?? null;
    }
  }

  return metrics;
}

function ensureObjectiveMetrics(
  trialMetrics: Metrics,
  spec: NormalizedOptimizationSpec,
): void {
  for (const objective of spec.objectives) {
    const value = trialMetrics[objective.metric];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new ValidationError(
        `Objective metric "${objective.metric}" must resolve to a finite numeric value for every trial.`,
      );
    }
  }
}

function aggregateExampleMetrics(
  perExampleMetrics: readonly Metrics[],
  spec: NormalizedOptimizationSpec,
): Metrics {
  const metricNames = new Set<string>();
  for (const metrics of perExampleMetrics) {
    for (const key of Object.keys(metrics)) {
      metricNames.add(key);
    }
  }

  const aggregated: Metrics = {};
  for (const metricName of metricNames) {
    const values = perExampleMetrics
      .map((metrics) => metrics[metricName])
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));

    if (values.length === 0) {
      aggregated[metricName] = null;
      continue;
    }

    aggregated[metricName] = aggregateValues(
      values,
      getAggregationStrategy(spec.evaluation?.aggregation, metricName),
    );
  }

  return aggregated;
}

export function createAgentTrialFunction<T extends AnyFunction>(
  fn: T,
  spec: NormalizedOptimizationSpec,
  rows: readonly unknown[],
): (trialConfig: TrialConfig) => Promise<NativeTrialFunctionResult> {
  return async (trialConfig) => {
    const selectedRows = selectRows(rows, trialConfig);
    const perExampleMetrics: Metrics[] = [];

    for (const row of selectedRows) {
      const input = getInputFromRow(row, spec.evaluation?.inputField);
      const startedAt = Date.now();
      const customEvaluator = spec.evaluation?.customEvaluator;
      const runAgentWithConfig = (agentInput: unknown): unknown | Promise<unknown> =>
        invokeFunctionWithConfig(
          fn,
          undefined,
          [agentInput],
          trialConfig.config,
          spec.injection?.mode ?? "context",
        );

      if (isLegacyCustomEvaluator(customEvaluator)) {
        const { result: metricsResult, metrics: collectedMetrics } =
          await withRuntimeMetricsCollector(async () =>
            customEvaluator(runAgentWithConfig, { ...trialConfig.config }, row),
          );

        const metrics: Metrics = {
          ...collectedMetrics,
          ...((metricsResult ?? {}) as Metrics),
        };

        if (typeof metrics["latency"] !== "number") {
          metrics["latency"] = (Date.now() - startedAt) / 1000;
        }

        perExampleMetrics.push(metrics);
        continue;
      }

      const { result: output, metrics: collectedMetrics } =
        await withRuntimeMetricsCollector(async () => runAgentWithConfig(input));

      const runtimeMetrics: Metrics = {
        ...collectedMetrics,
        latency:
          typeof collectedMetrics["latency"] === "number"
            ? (collectedMetrics["latency"] as number)
            : (Date.now() - startedAt) / 1000,
      };

      const exampleMetrics = await evaluateExample(
        spec,
        row,
        output,
        runtimeMetrics,
        trialConfig.config,
      );
      perExampleMetrics.push(exampleMetrics);
    }

    const metrics = aggregateExampleMetrics(perExampleMetrics, spec);
    ensureObjectiveMetrics(metrics, spec);

    return {
      metrics,
      metadata: {
        examplesEvaluated: perExampleMetrics.length,
      },
    };
  };
}
