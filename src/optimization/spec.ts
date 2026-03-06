import { ValidationError } from '../core/errors.js';
import type { TrialConfig } from '../dtos/trial.js';
import { runNativeOptimization } from './native.js';
import type {
  BuiltInObjectiveName,
  EnumParamDefinition,
  FloatParamDefinition,
  HybridConfigSpace,
  IntParamDefinition,
  NativeOptimizedFunction,
  NativeOptimizeOptions,
  NativeTrialFunctionResult,
  NormalizedObjectiveDefinition,
  NormalizedOptimizationSpec,
  ObjectiveDefinition,
  ObjectiveInput,
  OptimizationSpec,
  ParameterDefinition,
} from './types.js';

const OPTIMIZATION_SPEC = Symbol.for('traigent.optimizationSpec');

const BUILT_IN_OBJECTIVES: Record<
  BuiltInObjectiveName,
  NormalizedObjectiveDefinition
> = {
  accuracy: { metric: 'accuracy', direction: 'maximize', weight: 1 },
  cost: { metric: 'cost', direction: 'minimize', weight: 1 },
  latency: { metric: 'latency', direction: 'minimize', weight: 1 },
};

type AnyFunction = (...args: any[]) => any;
type NativeTrialFunction = (
  trialConfig: TrialConfig,
) => Promise<NativeTrialFunctionResult>;

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function normalizeWeight(weight: unknown): number {
  if (weight === undefined) return 1;
  if (typeof weight !== 'number' || !Number.isFinite(weight) || weight <= 0) {
    throw new ValidationError('Objective weights must be positive finite numbers.');
  }
  return weight;
}

function normalizeObjective(
  objective: ObjectiveInput,
): NormalizedObjectiveDefinition {
  if (typeof objective === 'string') {
    const builtIn = BUILT_IN_OBJECTIVES[objective as BuiltInObjectiveName];
    if (!builtIn) {
      throw new ValidationError(
        `Unknown objective "${objective}". Use accuracy, cost, latency, or an explicit { metric, direction } object.`,
      );
    }
    return builtIn;
  }

  if (!objective || typeof objective !== 'object') {
    throw new ValidationError('Objectives must be strings or objects.');
  }

  if (
    typeof objective.metric !== 'string' ||
    objective.metric.trim().length === 0
  ) {
    throw new ValidationError('Objective objects require a non-empty metric.');
  }

  if (
    objective.direction !== 'maximize' &&
    objective.direction !== 'minimize'
  ) {
    throw new ValidationError(
      `Objective "${objective.metric}" must declare direction "maximize" or "minimize".`,
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
    throw new ValidationError(
      `Parameter "${name}" must be a valid identifier-like key.`,
    );
  }
}

function normalizeRangeDefinition<T extends FloatParamDefinition | IntParamDefinition>(
  kind: T['type'],
  definition: T,
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
    throw new ValidationError(
      `${kind} parameters only support scale "linear" or "log".`,
    );
  }
  if (definition.step !== undefined) {
    if (!Number.isFinite(definition.step) || definition.step <= 0) {
      throw new ValidationError(
        `${kind} parameters require step to be a positive finite number.`,
      );
    }
    if (kind === 'int' && !Number.isInteger(definition.step)) {
      throw new ValidationError('int parameters require step to be an integer.');
    }
  }

  return {
    ...definition,
    scale: definition.scale ?? 'linear',
  };
}

function normalizeParameterDefinition(
  definition: ParameterDefinition,
): ParameterDefinition {
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

export function normalizeOptimizationSpec(
  spec: OptimizationSpec,
): NormalizedOptimizationSpec {
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
    }),
  );

  if (Object.keys(configurationSpace).length === 0) {
    throw new ValidationError(
      'Optimization spec requires at least one configuration parameter.',
    );
  }

  if (!Array.isArray(spec.objectives) || spec.objectives.length === 0) {
    throw new ValidationError(
      'Optimization spec requires at least one objective.',
    );
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

  if (
    spec.evaluation?.data !== undefined &&
    spec.evaluation?.loadData !== undefined
  ) {
    throw new ValidationError(
      'Use either evaluation.data or evaluation.loadData, not both.',
    );
  }

  return {
    configurationSpace,
    objectives,
    budget: spec.budget,
    evaluation: spec.evaluation,
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

export function getOptimizationSpec(
  target: unknown,
): NormalizedOptimizationSpec | undefined {
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
      'toHybridConfigSpace() requires a wrapped function or optimization spec.',
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
            `Unsupported parameter type for "${name}" in hybrid config space.`,
          );
      }
    }),
    constraints: {},
  };
}

export function optimize(specInput: OptimizationSpec) {
  const spec = normalizeOptimizationSpec(specInput);

  return function <T extends AnyFunction>(fn: T): NativeOptimizedFunction<T> {
    const target = fn as NativeOptimizedFunction<T>;
    defineHiddenProperty(target, OPTIMIZATION_SPEC, spec);

    defineHiddenProperty(
      target,
      'optimize',
      async (options: NativeOptimizeOptions) => {
        return runNativeOptimization(
          target as unknown as NativeTrialFunction,
          spec,
          options,
        );
      },
    );

    return target;
  };
}
