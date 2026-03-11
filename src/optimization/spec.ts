import { existsSync, readFileSync } from "node:fs";

import { TrialContext } from "../core/context.js";
import { ValidationError } from "../core/errors.js";
import { describeFrameworkAutoOverride } from "../integrations/registry.js";
import type { TrialConfig } from "../dtos/trial.js";
import {
  createAgentTrialFunction,
  invokeFunctionWithConfig,
  resolveEvaluationRows,
} from "./agent.js";
import { runHybridOptimization } from "./hybrid.js";
import { runNativeOptimization } from "./native.js";
import { stableValueKey } from "./stable-value.js";
import type {
  BuiltInObjectiveName,
  DerivedConstraintDefinition,
  EnumParamDefinition,
  FloatParamDefinition,
  FrameworkAutoOverrideStatus,
  FrameworkTarget,
  HybridOptimizeOptions,
  HybridConfigSpace,
  InjectionSpec,
  IntParamDefinition,
  NativeOptimizedFunction,
  NativeTrialFunctionResult,
  NativeOptimizeOptions,
  NormalizedObjectiveDefinition,
  NormalizedOptimizationSpec,
  ObjectiveInput,
  OptimizeOptions,
  PromotionPolicy,
  SeamlessResolution,
  OptimizationConstraints,
  OptimizationExecutionSpec,
  OptimizationResult,
  OptimizationSpec,
  ParameterConditions,
  ParameterConditionValue,
  ParameterDefinition,
  StructuralConstraintDefinition,
} from "./types.js";

const OPTIMIZATION_SPEC = Symbol.for("traigent.optimizationSpec");

const BUILT_IN_OBJECTIVES: Record<
  BuiltInObjectiveName,
  NormalizedObjectiveDefinition
> = {
  accuracy: { kind: "standard", metric: "accuracy", direction: "maximize", weight: 1 },
  cost: { kind: "standard", metric: "cost", direction: "minimize", weight: 1 },
  latency: { kind: "standard", metric: "latency", direction: "minimize", weight: 1 },
};

type AnyFunction = (...args: any[]) => any;
type NativeTrialFunction = (
  trialConfig: TrialConfig,
) => Promise<NativeTrialFunctionResult>;
const LOW_LEVEL_CONTRACT_WARNING =
  'execution.contract="trial" is deprecated and will be removed in a future release. Use the high-level agent contract instead.';

let hasWarnedAboutTrialContract = false;

function isHybridOptimizeOptions(
  options: OptimizeOptions,
): options is HybridOptimizeOptions {
  return options.mode !== "native";
}

function validateNativeOptimizationCompatibility(
  spec: NormalizedOptimizationSpec,
): void {
  for (const objective of spec.objectives) {
    if (objective.weight !== 1) {
      throw new ValidationError(
        `Native optimize() does not support weighted objective "${objective.metric}". Use mode: "hybrid".`,
      );
    }
  }

  for (const [name, definition] of Object.entries(spec.configurationSpace)) {
    if (definition.conditions !== undefined) {
      throw new ValidationError(
        `Native optimize() does not support conditional parameter "${name}" in optimization mode. Use mode: "hybrid".`,
      );
    }
  }

  if (spec.constraints) {
    const hasStructural = (spec.constraints.structural?.length ?? 0) > 0;
    const hasDerived = (spec.constraints.derived?.length ?? 0) > 0;
    if (hasStructural || hasDerived) {
      throw new ValidationError(
        "Native optimize() does not support structural or derived constraints. Use mode: \"hybrid\".",
      );
    }
  }

  if (spec.budget?.maxTrials !== undefined) {
    throw new ValidationError(
      "Native optimize() does not support budget.maxTrials on the spec. Use optimize({ maxTrials }) for native mode or mode: \"hybrid\".",
    );
  }

  if (spec.budget?.maxWallclockMs !== undefined) {
    throw new ValidationError(
      "Native optimize() does not support budget.maxWallclockMs on the spec. Use optimize({ timeoutMs }) for native mode or mode: \"hybrid\".",
    );
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isConditionValue(value: unknown): value is ParameterConditionValue {
  return (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  );
}

function normalizeWeight(weight: unknown): number {
  if (weight === undefined) return 1;
  if (typeof weight !== "number" || !Number.isFinite(weight) || weight <= 0) {
    throw new ValidationError(
      "Objective weights must be positive finite numbers.",
    );
  }
  return weight;
}

function valuesContain(values: readonly unknown[], expected: unknown): boolean {
  const expectedKey = stableValueKey(expected);
  return values.some((value) => stableValueKey(value) === expectedKey);
}

function normalizeBandTarget(
  band: unknown,
  field: string,
): { low: number; high: number } {
  if (!isPlainObject(band)) {
    throw new ValidationError(`${field} must be an object.`);
  }

  const hasBounds = band["low"] !== undefined || band["high"] !== undefined;
  const hasCenterTol = band["center"] !== undefined || band["tol"] !== undefined;

  if (hasBounds && hasCenterTol) {
    throw new ValidationError(
      `${field} must provide either low/high or center/tol, not both.`,
    );
  }

  if (hasBounds) {
    if (
      typeof band["low"] !== "number" ||
      !Number.isFinite(band["low"]) ||
      typeof band["high"] !== "number" ||
      !Number.isFinite(band["high"])
    ) {
      throw new ValidationError(`${field}.low and ${field}.high must be finite numbers.`);
    }
    if (band["low"] >= band["high"]) {
      throw new ValidationError(`${field}.low must be less than ${field}.high.`);
    }
    return {
      low: band["low"],
      high: band["high"],
    };
  }

  if (
    typeof band["center"] !== "number" ||
    !Number.isFinite(band["center"]) ||
    typeof band["tol"] !== "number" ||
    !Number.isFinite(band["tol"]) ||
    band["tol"] <= 0
  ) {
    throw new ValidationError(
      `${field}.center and ${field}.tol must be finite numbers and tol must be positive.`,
    );
  }

  return {
    low: band["center"] - band["tol"],
    high: band["center"] + band["tol"],
  };
}

function normalizePromotionPolicy(
  policy: unknown,
): PromotionPolicy | undefined {
  if (policy === undefined) {
    return undefined;
  }
  if (!isPlainObject(policy)) {
    throw new ValidationError("promotionPolicy must be an object when provided.");
  }

  const normalized: PromotionPolicy = {};

  if (policy["dominance"] !== undefined) {
    if (policy["dominance"] !== "epsilon_pareto") {
      throw new ValidationError(
        'promotionPolicy.dominance must be "epsilon_pareto" when provided.',
      );
    }
    normalized.dominance = "epsilon_pareto";
  }

  if (policy["alpha"] !== undefined) {
    if (
      typeof policy["alpha"] !== "number" ||
      !Number.isFinite(policy["alpha"]) ||
      policy["alpha"] <= 0 ||
      policy["alpha"] >= 1
    ) {
      throw new ValidationError("promotionPolicy.alpha must be in (0, 1).");
    }
    normalized.alpha = policy["alpha"];
  }

  if (policy["adjust"] !== undefined) {
    if (policy["adjust"] !== "none" && policy["adjust"] !== "BH") {
      throw new ValidationError('promotionPolicy.adjust must be "none" or "BH".');
    }
    normalized.adjust = policy["adjust"];
  }

  if (policy["minEffect"] !== undefined) {
    if (!isPlainObject(policy["minEffect"])) {
      throw new ValidationError("promotionPolicy.minEffect must be an object.");
    }
    normalized.minEffect = Object.fromEntries(
      Object.entries(policy["minEffect"]).map(([metric, value]) => {
        if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
          throw new ValidationError(
            `promotionPolicy.minEffect.${metric} must be a non-negative finite number.`,
          );
        }
        return [metric, value];
      }),
    );
  }

  if (policy["chanceConstraints"] !== undefined) {
    if (!Array.isArray(policy["chanceConstraints"])) {
      throw new ValidationError("promotionPolicy.chanceConstraints must be an array.");
    }
    normalized.chanceConstraints = policy["chanceConstraints"].map((entry, index) => {
      if (!isPlainObject(entry)) {
        throw new ValidationError(
          `promotionPolicy.chanceConstraints[${index}] must be an object.`,
        );
      }
      if (typeof entry["name"] !== "string" || entry["name"].trim().length === 0) {
        throw new ValidationError(
          `promotionPolicy.chanceConstraints[${index}].name must be a non-empty string.`,
        );
      }
      if (
        typeof entry["threshold"] !== "number" ||
        !Number.isFinite(entry["threshold"]) ||
        entry["threshold"] < 0 ||
        entry["threshold"] > 1
      ) {
        throw new ValidationError(
          `promotionPolicy.chanceConstraints[${index}].threshold must be in [0, 1].`,
        );
      }
      if (
        typeof entry["confidence"] !== "number" ||
        !Number.isFinite(entry["confidence"]) ||
        entry["confidence"] <= 0 ||
        entry["confidence"] >= 1
      ) {
        throw new ValidationError(
          `promotionPolicy.chanceConstraints[${index}].confidence must be in (0, 1).`,
        );
      }
      return {
        name: entry["name"].trim(),
        threshold: entry["threshold"],
        confidence: entry["confidence"],
      };
    });
  }

  if (policy["tieBreakers"] !== undefined) {
    if (!isPlainObject(policy["tieBreakers"])) {
      throw new ValidationError("promotionPolicy.tieBreakers must be an object.");
    }
    normalized.tieBreakers = Object.fromEntries(
      Object.entries(policy["tieBreakers"]).map(([metric, direction]) => {
        if (direction !== "maximize" && direction !== "minimize") {
          throw new ValidationError(
            `promotionPolicy.tieBreakers.${metric} must be "maximize" or "minimize".`,
          );
        }
        return [metric, direction];
      }),
    );
  }

  return Object.keys(normalized).length === 0 ? undefined : normalized;
}

function normalizeConstraintMessage(
  value: unknown,
  field: string,
): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ValidationError(`${field} must be a non-empty string when provided.`);
  }
  return value.trim();
}

function normalizeConstraintId(value: unknown, field: string): string | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ValidationError(`${field} must be a non-empty string when provided.`);
  }
  return value.trim();
}

function normalizeExpression(
  value: unknown,
  field: string,
): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ValidationError(`${field} must be a non-empty string.`);
  }
  return value.trim();
}

function normalizeStructuralConstraint(
  entry: unknown,
  index: number,
): StructuralConstraintDefinition {
  if (!isPlainObject(entry)) {
    throw new ValidationError(
      `constraints.structural[${index}] must be an object.`,
    );
  }

  const hasRequire = entry["require"] !== undefined;
  const hasWhen = entry["when"] !== undefined;
  const hasThen = entry["then"] !== undefined;

  if (hasRequire) {
    if (hasWhen || hasThen) {
      throw new ValidationError(
        `constraints.structural[${index}] cannot mix require with when/then.`,
      );
    }

    return {
      id: normalizeConstraintId(entry["id"], `constraints.structural[${index}].id`),
      require: normalizeExpression(
        entry["require"],
        `constraints.structural[${index}].require`,
      ),
      errorMessage: normalizeConstraintMessage(
        entry["errorMessage"],
        `constraints.structural[${index}].errorMessage`,
      ),
    };
  }

  if (!hasWhen || !hasThen) {
    throw new ValidationError(
      `constraints.structural[${index}] must provide either require or both when and then.`,
    );
  }

  return {
    id: normalizeConstraintId(entry["id"], `constraints.structural[${index}].id`),
    when: normalizeExpression(
      entry["when"],
      `constraints.structural[${index}].when`,
    ),
    then: normalizeExpression(
      entry["then"],
      `constraints.structural[${index}].then`,
    ),
    errorMessage: normalizeConstraintMessage(
      entry["errorMessage"],
      `constraints.structural[${index}].errorMessage`,
    ),
  };
}

function normalizeDerivedConstraint(
  entry: unknown,
  index: number,
): DerivedConstraintDefinition {
  if (!isPlainObject(entry)) {
    throw new ValidationError(`constraints.derived[${index}] must be an object.`);
  }

  return {
    id: normalizeConstraintId(entry["id"], `constraints.derived[${index}].id`),
    require: normalizeExpression(
      entry["require"],
      `constraints.derived[${index}].require`,
    ),
    errorMessage: normalizeConstraintMessage(
      entry["errorMessage"],
      `constraints.derived[${index}].errorMessage`,
    ),
  };
}

function normalizeConstraints(
  constraints: unknown,
): OptimizationConstraints | undefined {
  if (constraints === undefined) {
    return undefined;
  }

  if (!isPlainObject(constraints)) {
    throw new ValidationError("constraints must be an object when provided.");
  }

  const structuralRaw = constraints["structural"];
  const derivedRaw = constraints["derived"];

  const structural =
    structuralRaw === undefined
      ? undefined
      : Array.isArray(structuralRaw)
        ? structuralRaw.map(normalizeStructuralConstraint)
        : (() => {
            throw new ValidationError("constraints.structural must be an array when provided.");
          })();

  const derived =
    derivedRaw === undefined
      ? undefined
      : Array.isArray(derivedRaw)
        ? derivedRaw.map(normalizeDerivedConstraint)
        : (() => {
            throw new ValidationError("constraints.derived must be an array when provided.");
          })();

  if ((structural?.length ?? 0) === 0 && (derived?.length ?? 0) === 0) {
    throw new ValidationError(
      "constraints must include at least one structural or derived constraint.",
    );
  }

  return {
    ...(structural ? { structural } : {}),
    ...(derived ? { derived } : {}),
  };
}

function normalizeObjective(
  objective: ObjectiveInput,
): NormalizedObjectiveDefinition {
  if (typeof objective === "string") {
    const builtIn = BUILT_IN_OBJECTIVES[objective as BuiltInObjectiveName];
    if (!builtIn) {
      throw new ValidationError(
        `Unknown objective "${objective}". Use accuracy, cost, latency, or an explicit { metric, direction } object.`,
      );
    }
    return builtIn;
  }

  if (!objective || typeof objective !== "object") {
    throw new ValidationError("Objectives must be strings or objects.");
  }

  if (
    typeof objective.metric !== "string" ||
    objective.metric.trim().length === 0
  ) {
    throw new ValidationError("Objective objects require a non-empty metric.");
  }

  if ("band" in objective) {
    const band = normalizeBandTarget(
      objective.band,
      `Objective "${objective.metric}" band`,
    );
    if (objective.test !== undefined && objective.test !== "TOST") {
      throw new ValidationError(
        `Objective "${objective.metric}" band.test must be "TOST" when provided.`,
      );
    }
    if (
      objective.alpha !== undefined &&
      (typeof objective.alpha !== "number" ||
        !Number.isFinite(objective.alpha) ||
        objective.alpha <= 0 ||
        objective.alpha >= 1)
    ) {
      throw new ValidationError(
        `Objective "${objective.metric}" band.alpha must be in (0, 1).`,
      );
    }
    return {
      kind: "banded",
      metric: objective.metric,
      band,
      bandTest: "TOST",
      bandAlpha: objective.alpha ?? 0.05,
      weight: normalizeWeight(objective.weight),
    };
  }

  if (
    objective.direction !== "maximize" &&
    objective.direction !== "minimize"
  ) {
    throw new ValidationError(
      `Objective "${objective.metric}" must declare direction "maximize" or "minimize".`,
    );
  }

  return {
    kind: "standard",
    metric: objective.metric,
    direction: objective.direction,
    weight: normalizeWeight(objective.weight),
  };
}

function validateParameterName(name: string): void {
  if (!/^[A-Za-z_]\w*$/.test(name)) {
    throw new ValidationError(
      `Parameter "${name}" must be a valid identifier-like key.`,
    );
  }
}

function normalizeConditions(
  name: string,
  conditions: unknown,
): ParameterConditions | undefined {
  if (conditions === undefined) {
    return undefined;
  }

  if (!isPlainObject(conditions) || Object.keys(conditions).length === 0) {
    throw new ValidationError(
      `Parameter "${name}" conditions must be a non-empty object when provided.`,
    );
  }

  const normalizedEntries = Object.entries(conditions).map(([key, value]) => {
    validateParameterName(key);
    if (!isConditionValue(value)) {
      throw new ValidationError(
        `Parameter "${name}" conditions only support string, number, or boolean equality values.`,
      );
    }
    return [key, value] as const;
  });

  return Object.fromEntries(normalizedEntries);
}

function normalizeRangeDefinition<
  T extends FloatParamDefinition | IntParamDefinition,
>(name: string, kind: T["type"], definition: T): T {
  if (!Number.isFinite(definition.min) || !Number.isFinite(definition.max)) {
    throw new ValidationError(
      `${kind} parameters require finite min/max values.`,
    );
  }
  if (definition.max < definition.min) {
    throw new ValidationError(`${kind} parameters require max >= min.`);
  }
  if (
    definition.scale !== undefined &&
    definition.scale !== "linear" &&
    definition.scale !== "log"
  ) {
    throw new ValidationError(
      `${kind} parameters only support scale "linear" or "log".`,
    );
  }
  if (
    definition.scale === "log" &&
    (definition.min <= 0 || definition.max <= 0)
  ) {
    throw new ValidationError(
      `${kind} parameters with scale "log" require min/max > 0.`,
    );
  }
  if (definition.step !== undefined) {
    if (!Number.isFinite(definition.step) || definition.step <= 0) {
      throw new ValidationError(
        `${kind} parameters require step to be a positive finite number.`,
      );
    }
    if (kind === "int" && !Number.isInteger(definition.step)) {
      throw new ValidationError(
        "int parameters require step to be an integer.",
      );
    }
    if (definition.scale === "log" && definition.step <= 1) {
      throw new ValidationError(
        `${kind} parameters with scale "log" require step to be greater than 1 when provided.`,
      );
    }
  }

  return {
    ...definition,
    conditions: normalizeConditions(name, definition.conditions),
    scale: definition.scale ?? "linear",
  };
}

function normalizeParameterDefinition(
  name: string,
  definition: ParameterDefinition,
): ParameterDefinition {
  if (!definition || typeof definition !== "object") {
    throw new ValidationError("Parameter definitions must be objects.");
  }

  switch (definition.type) {
    case "enum":
      if (!Array.isArray(definition.values) || definition.values.length === 0) {
        throw new ValidationError(
          "enum parameters require a non-empty values array.",
        );
      }
      return {
        type: "enum",
        values: [...definition.values],
        conditions: normalizeConditions(name, definition.conditions),
        default: definition.default,
      } satisfies EnumParamDefinition;
    case "float":
      return normalizeRangeDefinition(name, "float", definition);
    case "int":
      return normalizeRangeDefinition(name, "int", definition);
    default:
      throw new ValidationError("Unsupported parameter definition type.");
  }
}

function validateConditionalDefault(
  name: string,
  definition: ParameterDefinition,
): void {
  if (definition.conditions === undefined) {
    if (definition.default !== undefined) {
      throw new ValidationError(
        `Parameter "${name}" default requires conditions to be defined.`,
      );
    }
    return;
  }

  if (definition.default === undefined) {
    throw new ValidationError(
      `Conditional parameter "${name}" requires a default fallback value.`,
    );
  }

  switch (definition.type) {
    case "enum":
      if (!valuesContain(definition.values, definition.default)) {
        throw new ValidationError(
          `Conditional enum parameter "${name}" default must be one of its values.`,
        );
      }
      return;
    case "int":
      if (
        typeof definition.default !== "number" ||
        !Number.isInteger(definition.default)
      ) {
        throw new ValidationError(
          `Conditional int parameter "${name}" default must be an integer.`,
        );
      }
      if (
        definition.default < definition.min ||
        definition.default > definition.max
      ) {
        throw new ValidationError(
          `Conditional int parameter "${name}" default must fall within min/max.`,
        );
      }
      return;
    case "float":
      if (
        typeof definition.default !== "number" ||
        !Number.isFinite(definition.default)
      ) {
        throw new ValidationError(
          `Conditional float parameter "${name}" default must be a finite number.`,
        );
      }
      if (
        definition.default < definition.min ||
        definition.default > definition.max
      ) {
        throw new ValidationError(
          `Conditional float parameter "${name}" default must fall within min/max.`,
        );
      }
      return;
    default:
      return;
  }
}

function validateConditionalDependencies(
  configurationSpace: Record<string, ParameterDefinition>,
): void {
  const names = Object.keys(configurationSpace);
  const visiting = new Set<string>();
  const visited = new Set<string>();

  const visit = (name: string): void => {
    if (visited.has(name)) {
      return;
    }
    if (visiting.has(name)) {
      throw new ValidationError(
        `Conditional parameters cannot form dependency cycles. Cycle detected at "${name}".`,
      );
    }

    visiting.add(name);
    const definition = configurationSpace[name]!;
    for (const dependency of Object.keys(definition.conditions ?? {})) {
      if (!(dependency in configurationSpace)) {
        throw new ValidationError(
          `Conditional parameter "${name}" references unknown dependency "${dependency}".`,
        );
      }
      if (dependency === name) {
        throw new ValidationError(
          `Conditional parameter "${name}" cannot depend on itself.`,
        );
      }
      visit(dependency);
    }
    visiting.delete(name);
    visited.add(name);
  };

  for (const name of names) {
    validateConditionalDefault(name, configurationSpace[name]!);
    visit(name);
  }
}

function normalizeBudget(
  budget: OptimizationSpec["budget"],
): OptimizationSpec["budget"] {
  if (budget === undefined) {
    return undefined;
  }

  const normalized = { ...budget };

  if (
    normalized.maxCostUsd !== undefined &&
    (typeof normalized.maxCostUsd !== "number" ||
      !Number.isFinite(normalized.maxCostUsd) ||
      normalized.maxCostUsd <= 0)
  ) {
    throw new ValidationError("budget.maxCostUsd must be a positive number.");
  }

  if (
    normalized.maxTrials !== undefined &&
    (!Number.isInteger(normalized.maxTrials) || normalized.maxTrials <= 0)
  ) {
    throw new ValidationError("budget.maxTrials must be a positive integer.");
  }

  if (
    normalized.maxWallclockMs !== undefined &&
    (!Number.isInteger(normalized.maxWallclockMs) ||
      normalized.maxWallclockMs <= 0)
  ) {
    throw new ValidationError(
      "budget.maxWallclockMs must be a positive integer.",
    );
  }

  return normalized;
}

function normalizeEvaluationSpec(
  evaluation: OptimizationSpec["evaluation"],
): OptimizationSpec["evaluation"] {
  if (!evaluation) {
    return undefined;
  }

  if (evaluation.data !== undefined && evaluation.loadData !== undefined) {
    throw new ValidationError(
      "Use either evaluation.data or evaluation.loadData, not both.",
    );
  }

  if (
    evaluation.customEvaluator &&
    (evaluation.scoringFunction || evaluation.metricFunctions)
  ) {
    throw new ValidationError(
      "evaluation.customEvaluator cannot be combined with scoringFunction or metricFunctions.",
    );
  }

  if (
    evaluation.inputField !== undefined &&
    (typeof evaluation.inputField !== "string" ||
      evaluation.inputField.trim().length === 0)
  ) {
    throw new ValidationError("evaluation.inputField must be a non-empty string.");
  }

  if (
    evaluation.expectedField !== undefined &&
    (typeof evaluation.expectedField !== "string" ||
      evaluation.expectedField.trim().length === 0)
  ) {
    throw new ValidationError(
      "evaluation.expectedField must be a non-empty string.",
    );
  }

  if (evaluation.aggregation !== undefined) {
    const validateAggregationStrategy = (
      value: unknown,
      field: string,
    ): void => {
      if (
        value !== "mean" &&
        value !== "median" &&
        value !== "sum" &&
        value !== "min" &&
        value !== "max"
      ) {
        throw new ValidationError(
          `${field} must be one of "mean", "median", "sum", "min", or "max".`,
        );
      }
    };

    if (typeof evaluation.aggregation === "string") {
      validateAggregationStrategy(
        evaluation.aggregation,
        "evaluation.aggregation",
      );
    } else if (isPlainObject(evaluation.aggregation)) {
      for (const [metric, strategy] of Object.entries(evaluation.aggregation)) {
        validateAggregationStrategy(
          strategy,
          `evaluation.aggregation.${metric}`,
        );
      }
    } else {
      throw new ValidationError(
        "evaluation.aggregation must be a strategy string or a per-metric object.",
      );
    }
  }

  return evaluation;
}

function normalizeInjectionSpec(
  injection: OptimizationSpec["injection"],
): (Required<Pick<InjectionSpec, "mode">> & InjectionSpec) | undefined {
  if (!injection) {
    return undefined;
  }
  const mode = injection?.mode ?? "context";
  if (mode !== "context" && mode !== "parameter" && mode !== "seamless") {
    throw new ValidationError(
      'injection.mode must be "context", "parameter", or "seamless".',
    );
  }

  if (
    injection.autoOverrideFrameworks !== undefined &&
    typeof injection.autoOverrideFrameworks !== "boolean"
  ) {
    throw new ValidationError(
      "injection.autoOverrideFrameworks must be a boolean when provided.",
    );
  }

  if (injection.frameworkTargets !== undefined) {
    if (!Array.isArray(injection.frameworkTargets)) {
      throw new ValidationError(
        "injection.frameworkTargets must be an array when provided.",
      );
    }

    for (const [index, target] of injection.frameworkTargets.entries()) {
      if (
        target !== "openai" &&
        target !== "langchain" &&
        target !== "vercel-ai"
      ) {
        throw new ValidationError(
          `injection.frameworkTargets[${index}] must be "openai", "langchain", or "vercel-ai".`,
        );
      }
    }
  }

  if (mode === "seamless") {
    return {
      ...injection,
      mode,
      autoOverrideFrameworks: injection.autoOverrideFrameworks ?? true,
    };
  }

  return {
    ...injection,
    mode,
    autoOverrideFrameworks: injection.autoOverrideFrameworks ?? false,
  };
}

function normalizeExecutionSpec(
  execution: OptimizationSpec["execution"],
): OptimizationExecutionSpec | undefined {
  if (!execution) {
    return undefined;
  }

  const mode = execution.mode ?? "hybrid";
  const contract = execution.contract;
  if (mode !== "native" && mode !== "hybrid") {
    throw new ValidationError('execution.mode must be "native" or "hybrid".');
  }
  if (
    contract !== undefined &&
    contract !== "agent" &&
    contract !== "trial"
  ) {
    throw new ValidationError(
      'execution.contract must be "agent" or "trial".',
    );
  }

  return {
    ...execution,
    mode,
    ...(contract ? { contract } : {}),
  };
}

function inferExecutionContract(
  spec: Pick<NormalizedOptimizationSpec, "evaluation" | "injection" | "execution">,
): "agent" | "trial" {
  if (spec.execution?.contract) {
    return spec.execution.contract;
  }

  if (
    spec.evaluation?.scoringFunction ||
    spec.evaluation?.metricFunctions ||
    spec.evaluation?.customEvaluator ||
    spec.evaluation?.inputField ||
    spec.evaluation?.expectedField ||
    spec.injection?.mode === "parameter" ||
    spec.injection?.mode === "seamless"
  ) {
    return "agent";
  }

  return "trial";
}

function normalizeDefaultConfig(
  value: unknown,
): Record<string, unknown> | undefined {
  if (value === undefined) {
    return undefined;
  }
  if (!isPlainObject(value)) {
    throw new ValidationError("defaultConfig must be an object when provided.");
  }
  return { ...value };
}

function resolveAutoLoadedConfig(
  spec: OptimizationSpec,
): Record<string, unknown> | undefined {
  if (!spec.autoLoadBest) {
    return undefined;
  }
  if (typeof spec.loadFrom !== "string" || spec.loadFrom.trim().length === 0) {
    throw new ValidationError(
      "autoLoadBest requires loadFrom to be a non-empty path.",
    );
  }
  if (!existsSync(spec.loadFrom)) {
    return undefined;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(spec.loadFrom, "utf8"));
  } catch (error) {
    throw new ValidationError(
      `Failed to load best config from "${spec.loadFrom}": ${String(error)}`,
    );
  }
  if (!isPlainObject(parsed)) {
    throw new ValidationError(
      `Loaded config from "${spec.loadFrom}" must be a JSON object.`,
    );
  }
  return { ...parsed };
}

function mergeConfig(
  baseConfig: Record<string, unknown> | undefined,
  overrideConfig: TrialConfig["config"] | null | undefined,
): TrialConfig["config"] | undefined {
  if (!baseConfig && !overrideConfig) {
    return undefined;
  }
  return {
    ...(baseConfig ?? {}),
    ...(overrideConfig ?? {}),
  };
}

function mergeOptimizeOptions(
  spec: NormalizedOptimizationSpec,
  options: OptimizeOptions,
): OptimizeOptions {
  if (!spec.execution) {
    return options;
  }

  const merged = { ...spec.execution, ...options } as Record<string, unknown>;
  if (merged["mode"] === "native") {
    return merged as unknown as NativeOptimizeOptions;
  }
  return merged as unknown as HybridOptimizeOptions;
}

export function normalizeOptimizationSpec(
  spec: OptimizationSpec,
): NormalizedOptimizationSpec {
  if (!spec || typeof spec !== "object") {
    throw new ValidationError("Optimization spec must be an object.");
  }

  if (!isPlainObject(spec.configurationSpace)) {
    throw new ValidationError("Optimization spec requires configurationSpace.");
  }

  const configurationSpace = Object.fromEntries(
    Object.entries(spec.configurationSpace).map(([name, definition]) => {
      validateParameterName(name);
      return [name, normalizeParameterDefinition(name, definition)];
    }),
  );

  if (Object.keys(configurationSpace).length === 0) {
    throw new ValidationError(
      "Optimization spec requires at least one configuration parameter.",
    );
  }

  validateConditionalDependencies(configurationSpace);

  if (!Array.isArray(spec.objectives) || spec.objectives.length === 0) {
    throw new ValidationError(
      "Optimization spec requires at least one objective.",
    );
  }

  const objectives = spec.objectives.map(normalizeObjective);
  const budget = normalizeBudget(spec.budget);
  const constraints = normalizeConstraints(spec.constraints);
  const defaultConfig = normalizeDefaultConfig(spec.defaultConfig);
  const promotionPolicy = normalizePromotionPolicy(spec.promotionPolicy);
  const evaluation = normalizeEvaluationSpec(spec.evaluation);
  const injection = normalizeInjectionSpec(spec.injection);
  const execution = normalizeExecutionSpec(spec.execution);

  return {
    configurationSpace,
    objectives,
    budget,
    constraints,
    defaultConfig,
    promotionPolicy,
    execution,
    autoLoadBest: spec.autoLoadBest,
    loadFrom: spec.loadFrom,
    evaluation,
    injection,
  };
}

function defineHiddenProperty(
  target: object,
  key: PropertyKey,
  value: unknown,
): void {
  Object.defineProperty(target, key, {
    value,
    enumerable: false,
    configurable: false,
    writable: false,
  });
}

function emitTrialContractWarning(): void {
  if (hasWarnedAboutTrialContract) {
    return;
  }
  hasWarnedAboutTrialContract = true;
  process.emitWarning(LOW_LEVEL_CONTRACT_WARNING, "DeprecationWarning");
}

function createAppliedTrialConfig(config: TrialConfig["config"]): TrialConfig {
  return {
    trial_id: "applied_config",
    trial_number: 0,
    experiment_run_id: "applied_config",
    config: { ...config },
    dataset_subset: {
      indices: [],
      total: 1,
    },
    metadata: {
      source: "applyBestConfig",
    },
  };
}

export const param = {
  enum(
    values: readonly (string | number | boolean)[],
    options?: Omit<EnumParamDefinition, "type" | "values">,
  ): EnumParamDefinition {
    return normalizeParameterDefinition("parameter", {
      type: "enum",
      values,
      ...options,
    }) as EnumParamDefinition;
  },
  float(definition: Omit<FloatParamDefinition, "type">): FloatParamDefinition {
    return normalizeParameterDefinition("parameter", {
      type: "float",
      ...definition,
    }) as FloatParamDefinition;
  },
  int(definition: Omit<IntParamDefinition, "type">): IntParamDefinition {
    return normalizeParameterDefinition("parameter", {
      type: "int",
      ...definition,
    }) as IntParamDefinition;
  },
  bool(
    options?: Omit<EnumParamDefinition<boolean>, "type" | "values">,
  ): EnumParamDefinition<boolean> {
    return normalizeParameterDefinition("parameter", {
      type: "enum",
      values: [false, true],
      ...options,
    }) as EnumParamDefinition<boolean>;
  },
};

export function getOptimizationSpec(
  target: unknown,
): NormalizedOptimizationSpec | undefined {
  if (typeof target === "function") {
    return (
      target as unknown as Record<PropertyKey, NormalizedOptimizationSpec>
    )[OPTIMIZATION_SPEC];
  }

  if (isPlainObject(target) && "configurationSpace" in target) {
    return normalizeOptimizationSpec(target as unknown as OptimizationSpec);
  }

  return undefined;
}

export function toHybridConfigSpace(target: unknown): HybridConfigSpace {
  const spec = getOptimizationSpec(target);
  if (!spec) {
    throw new ValidationError(
      "toHybridConfigSpace() requires a wrapped function or optimization spec.",
    );
  }

  return {
    tunables: Object.entries(spec.configurationSpace).map(
      ([name, definition]) => {
        if (definition.conditions) {
          throw new ValidationError(
            `toHybridConfigSpace() does not support conditional parameter "${name}" yet.`,
          );
        }
        switch (definition.type) {
          case "enum":
            return {
              name,
              type: "enum",
              domain: { values: [...definition.values] },
            };
          case "float":
            return {
              name,
              type: "float",
              domain: {
                range: [definition.min, definition.max] as [number, number],
              },
              scale: definition.scale,
            };
          case "int":
            return {
              name,
              type: "int",
              domain: {
                range: [definition.min, definition.max] as [number, number],
              },
              scale: definition.scale,
            };
          default:
            throw new ValidationError(
              `Unsupported parameter type for "${name}" in hybrid config space.`,
            );
        }
      },
    ),
    constraints: spec.constraints ?? {},
  };
}

export function optimize(specInput: OptimizationSpec) {
  const spec = normalizeOptimizationSpec(specInput);

  return function <T extends AnyFunction>(fn: T): NativeOptimizedFunction<T> {
    let appliedConfig: TrialConfig["config"] | undefined = mergeConfig(
      spec.defaultConfig,
      resolveAutoLoadedConfig(specInput),
    );

    function getFrameworkAutoOverrideStatus(): FrameworkAutoOverrideStatus {
      const resolvedFrameworkStatus = describeFrameworkAutoOverride(
        spec.injection?.frameworkTargets as
          | readonly FrameworkTarget[]
          | undefined,
        spec.injection?.autoOverrideFrameworks ??
          (spec.injection?.mode === "seamless"),
      );

      return {
        ...resolvedFrameworkStatus,
        requestedTargets: resolvedFrameworkStatus.requestedTargets
          ? [...resolvedFrameworkStatus.requestedTargets]
          : undefined,
        activeTargets: [...resolvedFrameworkStatus.activeTargets],
        selectedTargets: [...resolvedFrameworkStatus.selectedTargets],
      };
    }

    function getSeamlessResolution(): SeamlessResolution | undefined {
      if (spec.injection?.mode !== "seamless") {
        return undefined;
      }

      const status = getFrameworkAutoOverrideStatus();
      if (!status.enabled) {
        return undefined;
      }

      return {
        path: "framework",
        reason: status.reason,
        experimental: false,
        targets: [...status.selectedTargets],
      };
    }

    const wrapped = function wrappedOptimizedFunction(
      this: unknown,
      ...args: Parameters<T>
    ): ReturnType<T> {
      const activeTrialConfig = TrialContext.getConfigOrUndefined();
      const activeConfig =
        activeTrialConfig?.config ??
        appliedConfig ??
        (spec.defaultConfig ? { ...spec.defaultConfig } : undefined);

      if (!activeConfig) {
        return fn.apply(this, args) as ReturnType<T>;
      }

      const invoke = () =>
        invokeFunctionWithConfig(
          fn,
          this,
          args,
          activeConfig,
          spec.injection?.mode ?? "context",
        );

      if (activeTrialConfig) {
        return invoke() as ReturnType<T>;
      }

      return TrialContext.run(createAppliedTrialConfig(activeConfig), invoke) as ReturnType<T>;
    } as NativeOptimizedFunction<T>;

    defineHiddenProperty(wrapped, OPTIMIZATION_SPEC, spec);

    defineHiddenProperty(
      wrapped,
      "optimize",
      async (options: OptimizeOptions) => {
        const resolvedOptions = mergeOptimizeOptions(spec, options);
        const executionContract = inferExecutionContract(spec);

        if (executionContract === "trial") {
          emitTrialContractWarning();
          if (isHybridOptimizeOptions(resolvedOptions)) {
            return runHybridOptimization(
              fn as unknown as NativeTrialFunction,
              spec,
              specInput,
              resolvedOptions as HybridOptimizeOptions,
              fn.name,
            );
          }

          validateNativeOptimizationCompatibility(spec);
          return runNativeOptimization(
            fn as unknown as NativeTrialFunction,
            spec,
            resolvedOptions as NativeOptimizeOptions,
          );
        }

        const evaluationRows = await resolveEvaluationRows(spec);
        if (!Array.isArray(evaluationRows) || evaluationRows.length === 0) {
          throw new ValidationError(
            "optimize() requires evaluation data to be a non-empty array.",
          );
        }

        const hydratedSpec: NormalizedOptimizationSpec = {
          ...spec,
          evaluation: {
            ...spec.evaluation,
            data: evaluationRows,
            loadData: undefined,
          },
        };

        const trialFn = createAgentTrialFunction(fn, hydratedSpec, evaluationRows);
        if (isHybridOptimizeOptions(resolvedOptions)) {
          return runHybridOptimization(
            trialFn,
            hydratedSpec,
            specInput,
            resolvedOptions as HybridOptimizeOptions,
            fn.name,
          );
        }

        validateNativeOptimizationCompatibility(hydratedSpec);
        return runNativeOptimization(
          trialFn,
          hydratedSpec,
          resolvedOptions as NativeOptimizeOptions,
        );
      },
    );

    defineHiddenProperty(
      wrapped,
      "applyBestConfig",
      (result: OptimizationResult) => {
        appliedConfig = mergeConfig(spec.defaultConfig, result.bestConfig);
        return appliedConfig ? { ...appliedConfig } : undefined;
      },
    );

    defineHiddenProperty(
      wrapped,
      "currentConfig",
      () => (appliedConfig ? { ...appliedConfig } : undefined),
    );

    defineHiddenProperty(
      wrapped,
      "frameworkAutoOverrideStatus",
      () => getFrameworkAutoOverrideStatus(),
    );

    defineHiddenProperty(
      wrapped,
      "seamlessResolution",
      () => getSeamlessResolution(),
    );

    return wrapped;
  };
}
