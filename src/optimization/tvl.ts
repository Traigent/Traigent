import { readFile } from 'node:fs/promises';

import { parse as parseYaml } from 'yaml';

import { ValidationError } from '../core/errors.js';
import { isPlainObject } from '../core/is-plain-object.js';
import { normalizeOptimizationSpec } from './spec.js';
import { compileTvlConstraint } from './tvl-expression.js';
import type { OptimizationConstraint } from './types.js';
import type {
  NativeOptimizeOptions,
  NativeTvlCompatibilityReport,
  ObjectiveInput,
  OptimizationSpec,
  ParameterDefinition,
  TvlLoadOptions,
  TvlPromotionPolicy,
  TvlSpecArtifact,
} from './types.js';

function toFiniteNumber(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(`${field} must be a finite number.`);
  }
  return value;
}

function toPositiveInteger(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
    throw new ValidationError(`${field} must be a positive integer.`);
  }
  return value;
}

function normalizeTvarType(type: string, field: string): string {
  const normalized = type.trim().toLowerCase();
  if (normalized.length === 0) {
    throw new ValidationError(`${field} must be a non-empty string.`);
  }
  return normalized;
}

function parseTvarDomain(name: string, type: string, domain: unknown): ParameterDefinition {
  const domainObject = isPlainObject(domain) ? domain : {};
  const domainObjectAny = domainObject as any;
  const domainValues = Array.isArray(domain) ? domain : domainObjectAny.values;
  const scale = domainObjectAny.log === true || domainObjectAny.scale === 'log' ? 'log' : 'linear';

  if (type.startsWith('tuple')) {
    if (!Array.isArray(domainValues) || domainValues.length === 0) {
      throw new ValidationError(
        `tvars.${name}.domain must provide a non-empty values array for tuple variables.`
      );
    }
    for (const [index, value] of domainValues.entries()) {
      if (!Array.isArray(value) || value.length === 0) {
        throw new ValidationError(
          `tvars.${name}.domain.values[${index}] must be a non-empty array.`
        );
      }
    }
    return {
      type: 'enum',
      values: domainValues.map((value) => [...(value as unknown[])]),
    };
  }

  if (type.startsWith('callable')) {
    if (!Array.isArray(domainValues) || domainValues.length === 0) {
      throw new ValidationError(
        `tvars.${name}.domain must provide a non-empty values array for callable variables.`
      );
    }
    for (const [index, value] of domainValues.entries()) {
      if (typeof value !== 'string' || value.trim().length === 0) {
        throw new ValidationError(
          `tvars.${name}.domain.values[${index}] must be a non-empty string.`
        );
      }
    }
    return {
      type: 'enum',
      values: domainValues.map((value) => String(value)),
    };
  }

  switch (type) {
    case 'bool':
      return {
        type: 'enum',
        values: [false, true],
      };
    case 'enum':
    case 'enum[str]':
    case 'enum[int]':
    case 'enum[float]':
    case 'enum[bool]': {
      if (!Array.isArray(domainValues) || domainValues.length === 0) {
        throw new ValidationError(`tvars.${name}.domain must provide a non-empty values array.`);
      }
      return {
        type: 'enum',
        values: [...domainValues],
      };
    }
    case 'int':
    case 'float': {
      const range = Array.isArray(domain)
        ? domain
        : isPlainObject(domain) && Array.isArray((domain as any).range)
          ? (domain as any).range
          : undefined;
      if (!range || range.length !== 2) {
        throw new ValidationError(
          `tvars.${name}.domain.range must contain exactly two numeric values.`
        );
      }

      const min = toFiniteNumber(range[0], `tvars.${name}.domain.range[0]`);
      const max = toFiniteNumber(range[1], `tvars.${name}.domain.range[1]`);
      const step =
        domainObjectAny.step === undefined
          ? undefined
          : toFiniteNumber(domainObjectAny.step, `tvars.${name}.domain.step`);

      if (type === 'int') {
        if (!Number.isInteger(min) || !Number.isInteger(max)) {
          throw new ValidationError(
            `tvars.${name}.domain.range must use integers for int variables.`
          );
        }
        if (step !== undefined && !Number.isInteger(step)) {
          throw new ValidationError(
            `tvars.${name}.domain.step must be an integer for int variables.`
          );
        }
        return {
          type: 'int',
          min,
          max,
          scale,
          ...(step !== undefined ? { step } : {}),
        };
      }

      return {
        type: 'float',
        min,
        max,
        scale,
        ...(step !== undefined ? { step } : {}),
      };
    }
    default:
      throw new ValidationError(
        `TVL variable "${name}" type "${type}" is not supported by the native JS SDK.`
      );
  }
}

function parseTvars(raw: unknown): {
  configurationSpace: Record<string, ParameterDefinition>;
  defaultConfig: Record<string, unknown>;
} {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new ValidationError('TVL tvars must be a non-empty array.');
  }

  const configurationSpace: Record<string, ParameterDefinition> = {};
  const defaultConfig: Record<string, unknown> = {};

  for (const [index, entry] of raw.entries()) {
    if (!isPlainObject(entry)) {
      throw new ValidationError(`tvars[${index}] must be an object.`);
    }
    const entryAny = entry as any;

    const name = entryAny.name;
    const type = entryAny.type;
    if (typeof name !== 'string' || name.trim().length === 0) {
      throw new ValidationError(`tvars[${index}].name must be a non-empty string.`);
    }
    if (typeof type !== 'string') {
      throw new ValidationError(`tvars[${index}].type must be a non-empty string.`);
    }

    configurationSpace[name] = parseTvarDomain(
      name,
      normalizeTvarType(type, `tvars[${index}].type`),
      entryAny.domain
    );

    if (entryAny.default !== undefined) {
      defaultConfig[name] = entryAny.default;
    }
  }

  return { configurationSpace, defaultConfig };
}

function normalizeBandTarget(
  band: Record<string, unknown>,
  field: string
): { low: number; high: number } {
  const bandAny = band as any;
  if (Array.isArray(bandAny.target)) {
    if (bandAny.target.length !== 2) {
      throw new ValidationError(`${field}.target must contain exactly two values.`);
    }
    const low = toFiniteNumber(bandAny.target[0], `${field}.target[0]`);
    const high = toFiniteNumber(bandAny.target[1], `${field}.target[1]`);
    if (low >= high) {
      throw new ValidationError(`${field}.target requires low < high.`);
    }
    return { low, high };
  }

  if (isPlainObject(bandAny.target)) {
    const center = toFiniteNumber((bandAny.target as any).center, `${field}.target.center`);
    const tol = toFiniteNumber((bandAny.target as any).tol, `${field}.target.tol`);
    if (tol <= 0) {
      throw new ValidationError(`${field}.target.tol must be positive.`);
    }
    return {
      low: center - tol,
      high: center + tol,
    };
  }

  if (bandAny.low !== undefined || bandAny.high !== undefined) {
    const low = toFiniteNumber(bandAny.low, `${field}.low`);
    const high = toFiniteNumber(bandAny.high, `${field}.high`);
    if (low >= high) {
      throw new ValidationError(`${field} requires low < high.`);
    }
    return { low, high };
  }

  if (bandAny.center !== undefined || bandAny.tol !== undefined) {
    const center = toFiniteNumber(bandAny.center, `${field}.center`);
    const tol = toFiniteNumber(bandAny.tol, `${field}.tol`);
    if (tol <= 0) {
      throw new ValidationError(`${field}.tol must be positive.`);
    }
    return {
      low: center - tol,
      high: center + tol,
    };
  }

  throw new ValidationError(`${field} must provide target, low/high, or center/tol.`);
}

function parseObjectives(raw: unknown): ObjectiveInput[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    throw new ValidationError('TVL objectives must be a non-empty array.');
  }

  return raw.map((entry, index) => {
    if (!isPlainObject(entry)) {
      throw new ValidationError(`objectives[${index}] must be an object.`);
    }
    const entryAny = entry as any;

    const metric = entryAny.name;
    if (typeof metric !== 'string' || metric.trim().length === 0) {
      throw new ValidationError(`objectives[${index}].name must be a non-empty string.`);
    }

    if (entryAny.band !== undefined) {
      if (!isPlainObject(entryAny.band)) {
        throw new ValidationError(`objectives[${index}].band must be an object.`);
      }

      const band = normalizeBandTarget(entryAny.band, `objectives[${index}].band`);
      const test = entryAny.band.test === undefined ? 'TOST' : String(entryAny.band.test);
      if (test !== 'TOST') {
        throw new ValidationError(`objectives[${index}].band.test must be "TOST".`);
      }

      const alpha =
        entryAny.band.alpha === undefined
          ? 0.05
          : toFiniteNumber(entryAny.band.alpha, `objectives[${index}].band.alpha`);
      if (alpha <= 0 || alpha >= 1) {
        throw new ValidationError(`objectives[${index}].band.alpha must be in (0, 1).`);
      }

      return {
        metric: metric.trim(),
        direction: 'band',
        band,
        ...(entryAny.weight !== undefined
          ? {
              weight: toFiniteNumber(entryAny.weight, `objectives[${index}].weight`),
            }
          : {}),
      };
    }

    if (entryAny.direction !== 'maximize' && entryAny.direction !== 'minimize') {
      throw new ValidationError(`objectives[${index}].direction must be "maximize" or "minimize".`);
    }

    return {
      metric: metric.trim(),
      direction: entryAny.direction,
      ...(entryAny.weight !== undefined
        ? {
            weight: toFiniteNumber(entryAny.weight, `objectives[${index}].weight`),
          }
        : {}),
    };
  });
}

function parseStructuralConstraints(raw: unknown): OptimizationConstraint[] {
  if (raw === undefined) {
    return [];
  }
  if (!Array.isArray(raw)) {
    throw new ValidationError('constraints.structural must be an array.');
  }

  return raw.map((entry, index) => {
    if (!isPlainObject(entry)) {
      throw new ValidationError(`constraints.structural[${index}] must be an object.`);
    }
    const entryAny = entry as any;
    const id =
      typeof entryAny.id === 'string' && entryAny.id.trim().length > 0
        ? entryAny.id.trim()
        : `structural_${index + 1}`;
    const errorMessage =
      typeof entryAny.error_message === 'string'
        ? entryAny.error_message
        : typeof entryAny.errorMessage === 'string'
          ? entryAny.errorMessage
          : undefined;

    if (typeof entryAny.expr === 'string') {
      return compileTvlConstraint(id, entryAny.expr, errorMessage, 'expr');
    }
    if (typeof entryAny.require === 'string') {
      return compileTvlConstraint(id, entryAny.require, errorMessage, 'expr');
    }
    if (typeof entryAny.when === 'string' && typeof entryAny.then === 'string') {
      return compileTvlConstraint(id, entryAny.then, errorMessage, 'implication', entryAny.when);
    }

    throw new ValidationError(
      `constraints.structural[${index}] must provide expr, require, or when/then.`
    );
  });
}

function parseDerivedConstraints(raw: unknown): OptimizationConstraint[] {
  if (raw === undefined) {
    return [];
  }
  if (!Array.isArray(raw)) {
    throw new ValidationError('constraints.derived must be an array.');
  }

  return raw.map((entry, index) => {
    if (!isPlainObject(entry)) {
      throw new ValidationError(`constraints.derived[${index}] must be an object.`);
    }
    const entryAny = entry as any;
    const id =
      typeof entryAny.id === 'string' && entryAny.id.trim().length > 0
        ? entryAny.id.trim()
        : `derived_${index + 1}`;
    const errorMessage =
      typeof entryAny.error_message === 'string'
        ? entryAny.error_message
        : typeof entryAny.errorMessage === 'string'
          ? entryAny.errorMessage
          : undefined;
    const expression =
      typeof entryAny.expr === 'string'
        ? entryAny.expr
        : typeof entryAny.require === 'string'
          ? entryAny.require
          : undefined;
    if (!expression) {
      throw new ValidationError(`constraints.derived[${index}] must provide expr or require.`);
    }
    const constraint = compileTvlConstraint(id, expression, errorMessage, 'expr');
    constraint.requiresMetrics = true;
    return constraint;
  });
}

function parseConstraints(raw: unknown): OptimizationConstraint[] {
  if (raw === undefined) {
    return [];
  }
  if (!isPlainObject(raw)) {
    throw new ValidationError('constraints must be an object when provided.');
  }
  const rawAny = raw as any;

  return [
    ...parseStructuralConstraints(rawAny['structural']),
    ...parseDerivedConstraints(rawAny['derived']),
  ];
}

function parseExplorationStrategy(raw: unknown): Partial<Pick<NativeOptimizeOptions, 'algorithm'>> {
  if (!isPlainObject(raw)) {
    return {};
  }
  const rawAny = raw as any;
  const strategyType =
    typeof rawAny.strategy === 'string'
      ? rawAny.strategy
      : isPlainObject(rawAny.strategy) && typeof (rawAny.strategy as any).type === 'string'
        ? (rawAny.strategy as any).type
        : undefined;
  if (!strategyType) {
    return {};
  }

  const normalized = strategyType.trim().toLowerCase();
  if (normalized === 'grid') {
    return { algorithm: 'grid' };
  }
  if (normalized === 'random') {
    return { algorithm: 'random' };
  }
  if (normalized === 'bayesian' || normalized === 'pareto-optimal') {
    return { algorithm: 'bayesian' };
  }

  if (normalized === 'pareto_optimal' || normalized === 'nsga2') {
    throw new ValidationError(
      `exploration.strategy type "${strategyType}" is only supported in hybrid/server optimization, not the native JS runtime.`
    );
  }

  throw new ValidationError(
    `exploration.strategy type "${strategyType}" is not supported by the native JS SDK.`
  );
}

function parseExplorationBudgets(raw: unknown): {
  optimizeOptions: Partial<Pick<NativeOptimizeOptions, 'algorithm' | 'maxTrials'>>;
  specPatch: Pick<OptimizationSpec, 'budget' | 'execution'>;
} {
  const optimizeOptions: Partial<Pick<NativeOptimizeOptions, 'algorithm' | 'maxTrials'>> = {};
  const specFields = {} as Pick<OptimizationSpec, 'budget' | 'execution'>;

  if (!isPlainObject(raw) || !isPlainObject((raw as any).budgets)) {
    return {
      optimizeOptions,
      specPatch: specFields,
    };
  }

  const budgets = (raw as any).budgets;
  if (budgets['max_trials'] !== undefined) {
    optimizeOptions.maxTrials = toPositiveInteger(
      budgets['max_trials'],
      'exploration.budgets.max_trials'
    );
  }
  if (budgets['max_spend_usd'] !== undefined) {
    const maxCostUsd = toFiniteNumber(
      budgets['max_spend_usd'],
      'exploration.budgets.max_spend_usd'
    );
    if (maxCostUsd <= 0) {
      throw new ValidationError('exploration.budgets.max_spend_usd must be a positive number.');
    }
    specFields.budget = { maxCostUsd };
  }
  if (budgets['max_wallclock_s'] !== undefined) {
    const seconds = toFiniteNumber(
      budgets['max_wallclock_s'],
      'exploration.budgets.max_wallclock_s'
    );
    if (seconds <= 0) {
      throw new ValidationError('exploration.budgets.max_wallclock_s must be a positive number.');
    }
    specFields.execution = {
      maxWallclockMs: Math.round(seconds * 1000),
    };
  }

  return {
    optimizeOptions,
    specPatch: specFields,
  };
}

function parsePromotionPolicy(raw: unknown): TvlPromotionPolicy | undefined {
  if (raw === undefined) {
    return undefined;
  }
  if (!isPlainObject(raw)) {
    throw new ValidationError('promotion_policy must be an object when provided.');
  }
  const rawAny = raw as any;

  const normalized: TvlPromotionPolicy = {};

  if (rawAny.dominance !== undefined) {
    if (rawAny.dominance !== 'epsilon_pareto') {
      throw new ValidationError('promotion_policy.dominance must be "epsilon_pareto".');
    }
    normalized.dominance = 'epsilon_pareto';
  }

  if (rawAny.alpha !== undefined) {
    const alpha = toFiniteNumber(rawAny.alpha, 'promotion_policy.alpha');
    if (alpha <= 0 || alpha >= 1) {
      throw new ValidationError('promotion_policy.alpha must be in (0, 1).');
    }
    normalized.alpha = alpha;
  }

  if (rawAny.adjust !== undefined) {
    if (rawAny.adjust !== 'none' && rawAny.adjust !== 'BH') {
      throw new ValidationError('promotion_policy.adjust must be "none" or "BH".');
    }
    normalized.adjust = rawAny.adjust;
  }

  if (rawAny.min_effect !== undefined) {
    if (!isPlainObject(rawAny.min_effect)) {
      throw new ValidationError('promotion_policy.min_effect must be an object.');
    }
    normalized.minEffect = Object.fromEntries(
      Object.entries(rawAny.min_effect).map(([metric, value]) => {
        const numeric = toFiniteNumber(value, `promotion_policy.min_effect.${metric}`);
        if (numeric < 0) {
          throw new ValidationError(`promotion_policy.min_effect.${metric} must be non-negative.`);
        }
        return [metric, numeric];
      })
    );
  }

  if (rawAny.tie_breakers !== undefined) {
    if (!isPlainObject(rawAny.tie_breakers)) {
      throw new ValidationError('promotion_policy.tie_breakers must be an object.');
    }
    normalized.tieBreakers = Object.fromEntries(
      Object.entries(rawAny.tie_breakers).map(([metric, value]) => {
        if (value !== 'maximize' && value !== 'minimize') {
          throw new ValidationError(
            `promotion_policy.tie_breakers.${metric} must be "maximize" or "minimize".`
          );
        }
        return [metric, value];
      })
    );
  }

  if (rawAny.chance_constraints !== undefined) {
    if (!Array.isArray(rawAny.chance_constraints)) {
      throw new ValidationError('promotion_policy.chance_constraints must be an array.');
    }
    normalized.chanceConstraints = rawAny.chance_constraints.map(
      (entry: unknown, index: number) => {
        if (!isPlainObject(entry)) {
          throw new ValidationError(
            `promotion_policy.chance_constraints[${index}] must be an object.`
          );
        }
        const entryAny = entry as any;
        if (typeof entryAny.name !== 'string' || entryAny.name.trim().length === 0) {
          throw new ValidationError(
            `promotion_policy.chance_constraints[${index}].name must be a non-empty string.`
          );
        }
        const threshold = toFiniteNumber(
          entryAny.threshold,
          `promotion_policy.chance_constraints[${index}].threshold`
        );
        const confidence = toFiniteNumber(
          entryAny.confidence,
          `promotion_policy.chance_constraints[${index}].confidence`
        );
        if (confidence <= 0 || confidence > 1) {
          throw new ValidationError(
            `promotion_policy.chance_constraints[${index}].confidence must be in (0, 1].`
          );
        }
        return {
          name: entryAny.name.trim(),
          threshold,
          confidence,
        };
      }
    );
    // Native JS preserves chance constraints in TVL metadata, but statistical
    // enforcement is still deferred pending a Python-style promotion layer.
  }

  return Object.keys(normalized).length === 0 ? undefined : normalized;
}

function extractSpecHeader(raw: Record<string, unknown>): {
  moduleId?: string;
  tvlVersion?: string;
} {
  const rawAny = raw as any;
  const spec = isPlainObject(rawAny.spec) ? (rawAny.spec as any) : {};
  const moduleId =
    typeof spec.id === 'string' && spec.id.trim().length > 0 ? spec.id.trim() : undefined;
  const tvlVersion =
    typeof rawAny.tvl_version === 'string' || typeof rawAny.tvl_version === 'number'
      ? String(rawAny.tvl_version)
      : typeof spec.version === 'string' || typeof spec.version === 'number'
        ? String(spec.version)
        : undefined;

  return { moduleId, tvlVersion };
}

export function getNativeTvlCompatibilityReport(
  usedFeatures: Partial<Record<string, boolean>> = {}
): NativeTvlCompatibilityReport {
  const items: NativeTvlCompatibilityReport['items'] = [
    {
      feature: 'tvars',
      status: 'supported-with-reduced-semantics',
      reason:
        'Native JS supports bool, enum, int, float, tuple, and callable TVL variable forms, but not the full canonical TVL tvar surface.',
      used: usedFeatures['tvars'] === true,
    },
    {
      feature: 'banded-objectives',
      status: 'supported-with-reduced-semantics',
      reason:
        'Native JS supports banded objectives and TOST-style band comparison, but not the full Python promotion reporting model.',
      used: usedFeatures['banded-objectives'] === true,
    },
    {
      feature: 'promotion-policy',
      status: 'supported-with-reduced-semantics',
      reason:
        'Native JS applies minEffect, tieBreakers, chanceConstraints, and sample-based promotion, but not the full Python promotion-gate lifecycle/reporting semantics.',
      used: usedFeatures['promotion-policy'] === true,
    },
    {
      feature: 'constraints',
      status: 'supported-with-reduced-semantics',
      reason:
        'TVL structural and derived constraints compile into a safe native callback subset instead of preserving the canonical schema objects end-to-end.',
      used: usedFeatures['constraints'] === true,
    },
    {
      feature: 'exploration-strategy',
      status: 'supported-with-reduced-semantics',
      reason:
        'Native JS supports grid, random, and bayesian strategies. Multi-objective/server-oriented strategies remain hybrid-only.',
      used: usedFeatures['exploration-strategy'] === true,
    },
    {
      feature: 'hybrid-session-features',
      status: 'hybrid-only',
      reason:
        'Session persistence, remote orchestration, and full control-plane semantics belong to the hybrid/server path, not this native checkout.',
      used: usedFeatures['hybrid-session-features'] === true,
    },
  ];

  return {
    scope: 'native',
    items,
    usedFeatures: items.filter((item) => item.used).map((item) => item.feature),
    warnings: items
      .filter((item) => item.used && item.status !== 'supported')
      .map((item) => `${item.feature}: ${item.reason}`),
  };
}

export function parseTvlSpec(source: string): TvlSpecArtifact {
  const parsed = parseYaml(source) as unknown;
  if (!isPlainObject(parsed)) {
    throw new ValidationError('TVL source must parse to an object.');
  }
  const parsedAny = parsed as any;

  const { configurationSpace, defaultConfig } = parseTvars(parsedAny.tvars);
  const objectives = parseObjectives(parsedAny.objectives);
  const constraints = parseConstraints(parsedAny.constraints);
  const promotionPolicy = parsePromotionPolicy(parsedAny.promotion_policy ?? undefined);
  const strategy = parseExplorationStrategy(parsedAny.exploration);
  const budgets = parseExplorationBudgets(parsedAny.exploration);
  const { moduleId, tvlVersion } = extractSpecHeader(parsed);
  // This flag is intentionally about explicit exploration strategy selection,
  // not the mere presence of exploration budgets/settings. The warning for this
  // feature is meant to explain reduced-semantics strategy mapping, especially
  // when TVL specifies a strategy type that has hybrid/server-oriented meaning.
  const usedFeatures: Partial<Record<string, boolean>> = {
    tvars: Object.keys(configurationSpace).length > 0,
    'banded-objectives': objectives.some(
      (objective) =>
        typeof objective === 'object' &&
        objective !== null &&
        'band' in objective &&
        objective.band !== undefined
    ),
    'promotion-policy': promotionPolicy !== undefined,
    constraints: constraints.length > 0,
    'exploration-strategy':
      isPlainObject(parsedAny['exploration']) &&
      (typeof parsedAny['exploration']['strategy'] === 'string' ||
        (isPlainObject(parsedAny['exploration']['strategy']) &&
          typeof parsedAny['exploration']['strategy']['type'] === 'string')),
  };

  const spec: OptimizationSpec = {
    configurationSpace,
    objectives,
    ...(Object.keys(defaultConfig).length > 0 ? { defaultConfig } : {}),
    ...(promotionPolicy ? { promotionPolicy } : {}),
    ...(constraints.length > 0 ? { constraints } : {}),
    ...(budgets.specPatch.budget ? { budget: budgets.specPatch.budget } : {}),
    ...(budgets.specPatch.execution ? { execution: budgets.specPatch.execution } : {}),
  };
  normalizeOptimizationSpec(spec);

  const optimizeOptions =
    Object.keys({ ...strategy, ...budgets.optimizeOptions }).length > 0
      ? { ...strategy, ...budgets.optimizeOptions }
      : undefined;

  return {
    spec,
    ...(optimizeOptions ? { optimizeOptions } : {}),
    ...(moduleId ? { moduleId } : {}),
    ...(tvlVersion ? { tvlVersion } : {}),
    nativeCompatibility: getNativeTvlCompatibilityReport(usedFeatures),
    metadata: {
      ...(isPlainObject(parsedAny['metadata']) ? parsedAny['metadata'] : {}),
      ...(parsedAny['promotion'] !== undefined ? { legacyPromotion: parsedAny['promotion'] } : {}),
      ...(isPlainObject(parsedAny['exploration'])
        ? {
            strategyType:
              typeof parsedAny['exploration']['strategy'] === 'string'
                ? parsedAny['exploration']['strategy']
                : isPlainObject(parsedAny['exploration']['strategy']) &&
                    typeof parsedAny['exploration']['strategy']['type'] === 'string'
                  ? parsedAny['exploration']['strategy']['type']
                  : undefined,
          }
        : {}),
    },
    ...(promotionPolicy ? { promotionPolicy } : {}),
  };
}

export async function loadTvlSpec(input: string | TvlLoadOptions): Promise<TvlSpecArtifact> {
  const options = typeof input === 'string' ? ({ path: input } satisfies TvlLoadOptions) : input;

  if (options.path) {
    const source = await readFile(options.path, 'utf8');
    const loaded = parseTvlSpec(source);
    return {
      ...loaded,
      metadata: {
        ...loaded.metadata,
        path: options.path,
      },
    };
  }

  if (options.source) {
    return parseTvlSpec(options.source);
  }

  throw new ValidationError('loadTvlSpec() requires either path or source.');
}
