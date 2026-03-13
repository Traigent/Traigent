import { TrialContext } from '../core/context.js';
import { ValidationError } from '../core/errors.js';
import { isPlainObject } from '../core/is-plain-object.js';
import type { TrialConfig } from '../dtos/trial.js';
import {
  createAgentTrialFunction,
  invokeFunctionWithConfig,
  resolveEvaluationRows,
} from './agent.js';
import { runHybridOptimization } from './hybrid.js';
import { runNativeOptimization } from './native.js';
import { resolveSeamlessFunction } from '../seamless/runtime.js';
import type {
  BuiltInObjectiveName,
  EnumParamDefinition,
  FloatParamDefinition,
  HybridOptimizeOptions,
  HybridConfigSpace,
  IntParamDefinition,
  OptimizationConstraint,
  NativeOptimizedFunction,
  NativeTrialFunctionResult,
  NormalizedObjectiveDefinition,
  NormalizedOptimizationSpec,
  ObjectiveInput,
  OptimizeOptions,
  OptimizationResult,
  OptimizationSpec,
  ParameterDefinition,
  TvlPromotionPolicy,
  SafetyConstraint,
  SeamlessResolution,
} from './types.js';

const OPTIMIZATION_SPEC = Symbol.for('traigent.optimizationSpec');
const LOW_LEVEL_CONTRACT_WARNING =
  'execution.contract="trial" is deprecated and will be removed in a future release. Use the high-level agent contract instead.';

const BUILT_IN_OBJECTIVES: Record<BuiltInObjectiveName, NormalizedObjectiveDefinition> = {
  accuracy: { metric: 'accuracy', direction: 'maximize', weight: 1 },
  cost: { metric: 'cost', direction: 'minimize', weight: 1 },
  latency: { metric: 'latency', direction: 'minimize', weight: 1 },
};

type AnyFunction = (...args: any[]) => any;
type NativeTrialFunction = (trialConfig: TrialConfig) => Promise<NativeTrialFunctionResult>;

let hasWarnedAboutTrialContract = false;

function isHybridOptimizeOptions(options: OptimizeOptions): options is HybridOptimizeOptions {
  return options.mode === 'hybrid';
}

function normalizeWeight(weight: unknown): number {
  if (weight === undefined) {
    return 1;
  }
  if (typeof weight !== 'number' || !Number.isFinite(weight) || weight <= 0) {
    throw new ValidationError('Objective weights must be positive finite numbers.');
  }
  return weight;
}

function normalizeObjective(objective: ObjectiveInput): NormalizedObjectiveDefinition {
  if (typeof objective === 'string') {
    const builtIn = BUILT_IN_OBJECTIVES[objective as BuiltInObjectiveName];
    if (!builtIn) {
      throw new ValidationError(
        `Unknown objective "${objective}". Use accuracy, cost, latency, or an explicit { metric, direction } object.`
      );
    }
    return builtIn;
  }

  if (!objective || typeof objective !== 'object') {
    throw new ValidationError('Objectives must be strings or objects.');
  }

  if (typeof objective.metric !== 'string' || objective.metric.trim().length === 0) {
    throw new ValidationError('Objective objects require a non-empty metric.');
  }

  if (objective.band !== undefined) {
    const { low, high, test, alpha } = objective.band;
    if (
      typeof low !== 'number' ||
      !Number.isFinite(low) ||
      typeof high !== 'number' ||
      !Number.isFinite(high)
    ) {
      throw new ValidationError(
        `Objective "${objective.metric}" band requires finite low/high values.`
      );
    }
    if (low >= high) {
      throw new ValidationError(`Objective "${objective.metric}" band requires low < high.`);
    }
    if (
      alpha !== undefined &&
      (typeof alpha !== 'number' || !Number.isFinite(alpha) || alpha <= 0 || alpha >= 1)
    ) {
      throw new ValidationError(
        `Objective "${objective.metric}" band alpha must be a finite number in (0, 1).`
      );
    }
    if (test !== undefined && test !== 'TOST') {
      throw new ValidationError(
        `Objective "${objective.metric}" band test must be "TOST" when provided.`
      );
    }

    return {
      metric: objective.metric,
      direction: 'band',
      weight: normalizeWeight(objective.weight),
      band: {
        low,
        high,
        test: 'TOST',
        alpha: alpha ?? 0.05,
      },
    };
  }

  if (objective.direction !== 'maximize' && objective.direction !== 'minimize') {
    throw new ValidationError(
      `Objective "${objective.metric}" must declare direction "maximize" or "minimize", or provide a band target.`
    );
  }

  return {
    metric: objective.metric,
    direction: objective.direction,
    weight: normalizeWeight(objective.weight),
  };
}

function validateParameterName(name: string): void {
  if (!/^[A-Za-z_]\w*$/.test(name)) {
    throw new ValidationError(`Parameter "${name}" must be a valid identifier-like key.`);
  }
}

function normalizeRangeDefinition<T extends FloatParamDefinition | IntParamDefinition>(
  kind: T['type'],
  definition: T
): T {
  if (!Number.isFinite(definition.min) || !Number.isFinite(definition.max)) {
    throw new ValidationError(`${kind} parameters require finite min/max values.`);
  }
  if (definition.max < definition.min) {
    throw new ValidationError(`${kind} parameters require max >= min.`);
  }
  if (
    definition.scale !== undefined &&
    definition.scale !== 'linear' &&
    definition.scale !== 'log'
  ) {
    throw new ValidationError(`${kind} parameters only support scale "linear" or "log".`);
  }
  if (definition.scale === 'log' && (definition.min <= 0 || definition.max <= 0)) {
    throw new ValidationError(`${kind} parameters with scale "log" require min/max > 0.`);
  }
  if (definition.step !== undefined) {
    if (!Number.isFinite(definition.step) || definition.step <= 0) {
      throw new ValidationError(`${kind} parameters require step to be a positive finite number.`);
    }
    if (kind === 'int' && !Number.isInteger(definition.step)) {
      throw new ValidationError('int parameters require step to be an integer.');
    }
    if (definition.scale === 'log' && definition.step <= 1) {
      throw new ValidationError(
        `${kind} parameters with scale "log" require step to be greater than 1 when provided.`
      );
    }
  }

  return {
    ...definition,
    scale: definition.scale ?? 'linear',
  };
}

function normalizeParameterDefinition(definition: ParameterDefinition): ParameterDefinition {
  if (!definition || typeof definition !== 'object') {
    throw new ValidationError('Parameter definitions must be objects.');
  }

  switch (definition.type) {
    case 'enum':
      if (!Array.isArray(definition.values) || definition.values.length === 0) {
        throw new ValidationError('enum parameters require a non-empty values array.');
      }
      return {
        type: 'enum',
        values: [...definition.values],
      } satisfies EnumParamDefinition;
    case 'float':
      return normalizeRangeDefinition('float', definition);
    case 'int':
      return normalizeRangeDefinition('int', definition);
    default:
      throw new ValidationError('Unsupported parameter definition type.');
  }
}

function normalizeEvaluationSpec(
  evaluation: OptimizationSpec['evaluation']
): NormalizedOptimizationSpec['evaluation'] {
  if (!evaluation) {
    return undefined;
  }

  if (evaluation.data !== undefined && evaluation.loadData !== undefined) {
    throw new ValidationError('Use either evaluation.data or evaluation.loadData, not both.');
  }

  if (evaluation.customEvaluator && (evaluation.scoringFunction || evaluation.metricFunctions)) {
    throw new ValidationError(
      'evaluation.customEvaluator cannot be combined with scoringFunction or metricFunctions.'
    );
  }

  if (
    evaluation.inputField !== undefined &&
    (typeof evaluation.inputField !== 'string' || evaluation.inputField.trim().length === 0)
  ) {
    throw new ValidationError('evaluation.inputField must be a non-empty string.');
  }

  if (
    evaluation.expectedField !== undefined &&
    (typeof evaluation.expectedField !== 'string' || evaluation.expectedField.trim().length === 0)
  ) {
    throw new ValidationError('evaluation.expectedField must be a non-empty string.');
  }

  return evaluation;
}

function normalizeDefaultConfig(
  defaultConfig: OptimizationSpec['defaultConfig']
): NormalizedOptimizationSpec['defaultConfig'] {
  if (defaultConfig === undefined) {
    return {};
  }
  if (!isPlainObject(defaultConfig)) {
    throw new ValidationError('defaultConfig must be an object when provided.');
  }
  return { ...defaultConfig };
}

function normalizeConstraintList(
  constraints: OptimizationSpec['constraints']
): readonly OptimizationConstraint[] {
  if (constraints === undefined) {
    return [];
  }
  if (!Array.isArray(constraints)) {
    throw new ValidationError('constraints must be an array of functions.');
  }
  for (const constraint of constraints) {
    if (typeof constraint !== 'function') {
      throw new ValidationError('constraints must contain only functions.');
    }
  }
  return [...constraints];
}

function normalizeSafetyConstraintList(
  safetyConstraints: OptimizationSpec['safetyConstraints']
): readonly SafetyConstraint[] {
  if (safetyConstraints === undefined) {
    return [];
  }
  if (!Array.isArray(safetyConstraints)) {
    throw new ValidationError('safetyConstraints must be an array of functions.');
  }
  for (const constraint of safetyConstraints) {
    if (typeof constraint !== 'function') {
      throw new ValidationError('safetyConstraints must contain only functions.');
    }
  }
  return [...safetyConstraints];
}

function normalizePromotionPolicy(
  policy: OptimizationSpec['promotionPolicy']
): TvlPromotionPolicy | undefined {
  if (policy === undefined) {
    return undefined;
  }
  if (!isPlainObject(policy)) {
    throw new ValidationError('promotionPolicy must be an object when provided.');
  }

  const policyRecord = policy as Record<string, unknown>;
  const normalized: TvlPromotionPolicy = {};

  if (policyRecord['dominance'] !== undefined) {
    if (policyRecord['dominance'] !== 'epsilon_pareto') {
      throw new ValidationError(
        'promotionPolicy.dominance must be "epsilon_pareto" when provided.'
      );
    }
    normalized.dominance = 'epsilon_pareto';
  }

  if (policyRecord['alpha'] !== undefined) {
    if (
      typeof policyRecord['alpha'] !== 'number' ||
      !Number.isFinite(policyRecord['alpha']) ||
      policyRecord['alpha'] <= 0 ||
      policyRecord['alpha'] >= 1
    ) {
      throw new ValidationError('promotionPolicy.alpha must be in (0, 1).');
    }
    normalized.alpha = policyRecord['alpha'];
  }

  if (policyRecord['adjust'] !== undefined) {
    if (policyRecord['adjust'] !== 'none' && policyRecord['adjust'] !== 'BH') {
      throw new ValidationError('promotionPolicy.adjust must be "none" or "BH".');
    }
    normalized.adjust = policyRecord['adjust'];
  }

  if (policyRecord['minEffect'] !== undefined) {
    if (!isPlainObject(policyRecord['minEffect'])) {
      throw new ValidationError('promotionPolicy.minEffect must be an object.');
    }
    normalized.minEffect = Object.fromEntries(
      Object.entries(policyRecord['minEffect']).map(([metric, value]) => {
        if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
          throw new ValidationError(
            `promotionPolicy.minEffect.${metric} must be a non-negative finite number.`
          );
        }
        return [metric, value];
      })
    );
  }

  if (policyRecord['tieBreakers'] !== undefined) {
    if (!isPlainObject(policyRecord['tieBreakers'])) {
      throw new ValidationError('promotionPolicy.tieBreakers must be an object.');
    }
    normalized.tieBreakers = Object.fromEntries(
      Object.entries(policyRecord['tieBreakers']).map(([metric, direction]) => {
        if (direction !== 'maximize' && direction !== 'minimize') {
          throw new ValidationError(
            `promotionPolicy.tieBreakers.${metric} must be "maximize" or "minimize".`
          );
        }
        return [metric, direction];
      })
    );
  }

  if (policyRecord['chanceConstraints'] !== undefined) {
    if (!Array.isArray(policyRecord['chanceConstraints'])) {
      throw new ValidationError('promotionPolicy.chanceConstraints must be an array.');
    }
    normalized.chanceConstraints = policyRecord['chanceConstraints'].map((entry, index) => {
      if (!entry || typeof entry !== 'object') {
        throw new ValidationError(`promotionPolicy.chanceConstraints[${index}] must be an object.`);
      }

      const entryRecord = entry as Record<string, unknown>;
      if (typeof entryRecord['name'] !== 'string' || entryRecord['name'].trim().length === 0) {
        throw new ValidationError(
          `promotionPolicy.chanceConstraints[${index}].name must be a non-empty string.`
        );
      }
      if (
        typeof entryRecord['threshold'] !== 'number' ||
        !Number.isFinite(entryRecord['threshold'])
      ) {
        throw new ValidationError(
          `promotionPolicy.chanceConstraints[${index}].threshold must be a finite number.`
        );
      }
      if (
        typeof entryRecord['confidence'] !== 'number' ||
        !Number.isFinite(entryRecord['confidence']) ||
        entryRecord['confidence'] <= 0 ||
        entryRecord['confidence'] > 1
      ) {
        throw new ValidationError(
          `promotionPolicy.chanceConstraints[${index}].confidence must be in (0, 1].`
        );
      }
      return {
        name: entryRecord['name'].trim(),
        threshold: entryRecord['threshold'],
        confidence: entryRecord['confidence'],
      };
    });
  }

  return Object.keys(normalized).length === 0 ? undefined : normalized;
}

function normalizeInjectionSpec(
  injection: OptimizationSpec['injection']
): NormalizedOptimizationSpec['injection'] {
  const mode = injection?.mode ?? 'context';
  if (mode !== 'context' && mode !== 'parameter' && mode !== 'seamless') {
    throw new ValidationError('injection.mode must be "context", "parameter", or "seamless".');
  }

  if (
    injection?.autoOverrideFrameworks !== undefined &&
    typeof injection.autoOverrideFrameworks !== 'boolean'
  ) {
    throw new ValidationError('injection.autoOverrideFrameworks must be a boolean when provided.');
  }

  if (mode === 'seamless') {
    return {
      ...injection,
      mode,
      autoOverrideFrameworks: injection?.autoOverrideFrameworks ?? true,
    };
  }

  return {
    ...injection,
    mode,
    autoOverrideFrameworks: injection?.autoOverrideFrameworks ?? false,
  };
}

function normalizeExecutionSpec(
  execution: OptimizationSpec['execution']
): NormalizedOptimizationSpec['execution'] {
  const mode = execution?.mode ?? 'native';
  const contract = execution?.contract ?? 'agent';
  const exampleConcurrency = execution?.exampleConcurrency ?? 1;
  const repsPerTrial = execution?.repsPerTrial ?? 1;
  const repsAggregation = execution?.repsAggregation ?? 'mean';

  if (mode !== 'native' && mode !== 'hybrid') {
    throw new ValidationError('execution.mode must be "native" or "hybrid".');
  }

  if (contract !== 'agent' && contract !== 'trial') {
    throw new ValidationError('execution.contract must be "agent" or "trial".');
  }

  if (
    execution?.maxTotalExamples !== undefined &&
    (!Number.isInteger(execution.maxTotalExamples) || execution.maxTotalExamples <= 0)
  ) {
    throw new ValidationError('execution.maxTotalExamples must be a positive integer.');
  }

  if (
    execution?.maxWallclockMs !== undefined &&
    (!Number.isInteger(execution.maxWallclockMs) || execution.maxWallclockMs <= 0)
  ) {
    throw new ValidationError('execution.maxWallclockMs must be a positive integer.');
  }

  if (!Number.isInteger(exampleConcurrency) || exampleConcurrency <= 0) {
    throw new ValidationError('execution.exampleConcurrency must be a positive integer.');
  }

  if (!Number.isInteger(repsPerTrial) || repsPerTrial <= 0) {
    throw new ValidationError('execution.repsPerTrial must be a positive integer.');
  }

  if (
    repsAggregation !== 'mean' &&
    repsAggregation !== 'median' &&
    repsAggregation !== 'min' &&
    repsAggregation !== 'max'
  ) {
    throw new ValidationError(
      'execution.repsAggregation must be "mean", "median", "min", or "max".'
    );
  }

  return {
    ...execution,
    mode,
    contract,
    exampleConcurrency,
    repsPerTrial,
    repsAggregation,
  } satisfies NormalizedOptimizationSpec['execution'];
}

export function normalizeOptimizationSpec(spec: OptimizationSpec): NormalizedOptimizationSpec {
  if (!spec || typeof spec !== 'object') {
    throw new ValidationError('Optimization spec must be an object.');
  }

  if (!isPlainObject(spec.configurationSpace)) {
    throw new ValidationError('Optimization spec requires configurationSpace.');
  }

  const configurationSpace = Object.fromEntries(
    Object.entries(spec.configurationSpace).map(([name, definition]) => {
      validateParameterName(name);
      return [name, normalizeParameterDefinition(definition)];
    })
  );

  if (Object.keys(configurationSpace).length === 0) {
    throw new ValidationError('Optimization spec requires at least one configuration parameter.');
  }

  if (!Array.isArray(spec.objectives) || spec.objectives.length === 0) {
    throw new ValidationError('Optimization spec requires at least one objective.');
  }

  const objectives = spec.objectives.map(normalizeObjective);

  if (
    spec.budget?.maxCostUsd !== undefined &&
    (typeof spec.budget.maxCostUsd !== 'number' ||
      !Number.isFinite(spec.budget.maxCostUsd) ||
      spec.budget.maxCostUsd <= 0)
  ) {
    throw new ValidationError('budget.maxCostUsd must be a positive number.');
  }

  return {
    configurationSpace,
    objectives,
    budget: spec.budget,
    defaultConfig: normalizeDefaultConfig(spec.defaultConfig),
    promotionPolicy: normalizePromotionPolicy(spec.promotionPolicy),
    constraints: normalizeConstraintList(spec.constraints),
    safetyConstraints: normalizeSafetyConstraintList(spec.safetyConstraints),
    evaluation: normalizeEvaluationSpec(spec.evaluation),
    injection: normalizeInjectionSpec(spec.injection),
    execution: normalizeExecutionSpec(spec.execution),
  };
}

function defineHiddenProperty(target: object, key: PropertyKey, value: unknown): void {
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
  process.emitWarning(LOW_LEVEL_CONTRACT_WARNING, 'DeprecationWarning');
}

function createAppliedTrialConfig(config: TrialConfig['config']): TrialConfig {
  return {
    trial_id: 'applied_config',
    trial_number: 0,
    experiment_run_id: 'applied_config',
    config: { ...config },
    dataset_subset: {
      indices: [],
      total: 1,
    },
    metadata: {
      source: 'applyBestConfig',
    },
  };
}

export const param = {
  enum(values: readonly (string | number | boolean)[]): EnumParamDefinition {
    return normalizeParameterDefinition({
      type: 'enum',
      values,
    }) as EnumParamDefinition;
  },
  float(definition: Omit<FloatParamDefinition, 'type'>): FloatParamDefinition {
    return normalizeParameterDefinition({
      type: 'float',
      ...definition,
    }) as FloatParamDefinition;
  },
  int(definition: Omit<IntParamDefinition, 'type'>): IntParamDefinition {
    return normalizeParameterDefinition({
      type: 'int',
      ...definition,
    }) as IntParamDefinition;
  },
};

export function getOptimizationSpec(target: unknown): NormalizedOptimizationSpec | undefined {
  if (typeof target === 'function') {
    return (target as unknown as Record<PropertyKey, NormalizedOptimizationSpec>)[
      OPTIMIZATION_SPEC
    ];
  }

  if (isPlainObject(target) && 'configurationSpace' in target) {
    return normalizeOptimizationSpec(target as unknown as OptimizationSpec);
  }

  return undefined;
}

export function toHybridConfigSpace(target: unknown): HybridConfigSpace {
  const spec = getOptimizationSpec(target);
  if (!spec) {
    throw new ValidationError(
      'toHybridConfigSpace() requires a wrapped function or optimization spec.'
    );
  }

  return {
    tunables: Object.entries(spec.configurationSpace).map(([name, definition]) => {
      switch (definition.type) {
        case 'enum':
          return {
            name,
            type: 'enum',
            domain: { values: [...definition.values] },
          };
        case 'float':
          return {
            name,
            type: 'float',
            domain: { range: [definition.min, definition.max] as [number, number] },
            scale: definition.scale,
          };
        case 'int':
          return {
            name,
            type: 'int',
            domain: { range: [definition.min, definition.max] as [number, number] },
            scale: definition.scale,
          };
        default:
          throw new ValidationError(
            `Unsupported parameter type for "${name}" in hybrid config space.`
          );
      }
    }),
    constraints: {},
  };
}

export function optimize(specInput: OptimizationSpec) {
  const spec = normalizeOptimizationSpec(specInput);

  return function <T extends AnyFunction>(fn: T): NativeOptimizedFunction<T> {
    let appliedConfig: TrialConfig['config'] | undefined;
    let resolvedSeamlessFn: T | undefined;
    let resolvedSeamlessInfo: SeamlessResolution | undefined;

    function getInvocationFunction(): T {
      if (spec.injection.mode !== 'seamless') {
        return fn;
      }

      if (!resolvedSeamlessFn) {
        const seamlessResolution = resolveSeamlessFunction(
          fn,
          Object.keys(spec.configurationSpace),
          spec.injection.frameworkTargets,
          spec.injection.autoOverrideFrameworks ?? true
        );
        resolvedSeamlessFn = seamlessResolution.fn;
        resolvedSeamlessInfo = seamlessResolution.resolution;
      }

      return resolvedSeamlessFn;
    }

    const wrapped = function wrappedOptimizedFunction(
      this: unknown,
      ...args: Parameters<T>
    ): ReturnType<T> {
      const activeTrialConfig = TrialContext.getConfigOrUndefined();
      const activeConfig =
        activeTrialConfig?.config ??
        appliedConfig ??
        (Object.keys(spec.defaultConfig).length > 0 ? spec.defaultConfig : undefined);

      if (!activeConfig) {
        return fn.apply(this, args) as ReturnType<T>;
      }
      const invocationFn = getInvocationFunction();

      const invoke = () =>
        invokeFunctionWithConfig(invocationFn, this, args, activeConfig, spec.injection.mode);

      if (activeTrialConfig) {
        return invoke() as ReturnType<T>;
      }

      return TrialContext.run(createAppliedTrialConfig(activeConfig), invoke) as ReturnType<T>;
    } as NativeOptimizedFunction<T>;

    defineHiddenProperty(wrapped, OPTIMIZATION_SPEC, spec);

    defineHiddenProperty(wrapped, 'optimize', async (options: OptimizeOptions) => {
      if (!options || typeof options !== 'object') {
        throw new ValidationError('optimize() options are required.');
      }

      if (isHybridOptimizeOptions(options)) {
        return runHybridOptimization(
          fn as unknown as NativeTrialFunction,
          spec,
          specInput,
          options,
          fn.name
        );
      }

      if (spec.execution.mode === 'hybrid') {
        throw new ValidationError('execution.mode="hybrid" is not supported in this checkout.');
      }

      if (spec.execution.contract === 'trial') {
        emitTrialContractWarning();
        return runNativeOptimization(fn as unknown as NativeTrialFunction, spec, options);
      }

      const evaluationRows = await resolveEvaluationRows(spec);
      if (!Array.isArray(evaluationRows) || evaluationRows.length === 0) {
        throw new ValidationError('optimize() requires evaluation data to be a non-empty array.');
      }

      const hydratedSpec: NormalizedOptimizationSpec = {
        ...spec,
        evaluation: {
          ...spec.evaluation,
          data: evaluationRows,
          loadData: undefined,
        },
      };
      return runNativeOptimization(
        createAgentTrialFunction(getInvocationFunction(), hydratedSpec, evaluationRows),
        hydratedSpec,
        options
      );
    });

    defineHiddenProperty(wrapped, 'applyBestConfig', (result: OptimizationResult) => {
      if (result.bestConfig) {
        getInvocationFunction();
      }
      appliedConfig = result.bestConfig ? { ...result.bestConfig } : undefined;
      return appliedConfig ? { ...appliedConfig } : undefined;
    });

    defineHiddenProperty(wrapped, 'currentConfig', () => {
      const current =
        appliedConfig ??
        (Object.keys(spec.defaultConfig).length > 0 ? spec.defaultConfig : undefined);
      return current ? { ...current } : undefined;
    });

    defineHiddenProperty(wrapped, 'seamlessResolution', () =>
      resolvedSeamlessInfo
        ? {
            ...resolvedSeamlessInfo,
            targets: resolvedSeamlessInfo.targets ? [...resolvedSeamlessInfo.targets] : undefined,
          }
        : undefined
    );

    return wrapped;
  };
}
