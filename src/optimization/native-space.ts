import { createHash } from 'node:crypto';
import { ValidationError } from '../core/errors.js';
import { PythonRandom } from './python-random.js';
import type {
  FloatParamDefinition,
  IntParamDefinition,
  NormalizedOptimizationSpec,
  ParameterDefinition,
} from './types.js';

export type CandidateConfig = Record<string, unknown>;

export function roundToPrecision(value: number): number {
  return Number.parseFloat(value.toPrecision(12));
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function canonicalize(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, nested]) => [key, canonicalize(nested)]),
    );
  }
  return value;
}

export function stableJson(value: unknown): string {
  return JSON.stringify(canonicalize(value));
}

export function hashJson(value: unknown): string {
  return createHash('sha256').update(stableJson(value)).digest('hex');
}

export function configKey(config: CandidateConfig): string {
  return stableJson(config);
}

function hasActiveDefaultConfig(spec: NormalizedOptimizationSpec): boolean {
  return Object.keys(spec.defaultConfig).length > 0;
}

export function applyDefaultConfig(
  spec: NormalizedOptimizationSpec,
  config: CandidateConfig,
): CandidateConfig {
  if (!hasActiveDefaultConfig(spec)) {
    return { ...config };
  }
  return {
    ...spec.defaultConfig,
    ...config,
  };
}

export function getOrderedParameterEntries(
  configurationSpace: NormalizedOptimizationSpec['configurationSpace'],
): [string, ParameterDefinition][] {
  const names = Object.keys(configurationSpace).sort((left, right) => {
    if (left === 'model' && right !== 'model') return 1;
    if (right === 'model' && left !== 'model') return -1;
    return left.localeCompare(right);
  });

  return names.map((name) => [name, configurationSpace[name]!]);
}

export function ensureLogBounds(
  name: string,
  definition: FloatParamDefinition | IntParamDefinition,
): void {
  if (definition.min <= 0 || definition.max <= 0) {
    throw new ValidationError(
      `Log-scaled parameter "${name}" requires min and max to be greater than 0.`,
    );
  }
}

function buildLinearIntValues(definition: IntParamDefinition): number[] {
  const step = definition.step ?? 1;
  if (!Number.isInteger(step) || step <= 0) {
    throw new ValidationError(
      'Grid search requires int parameters to use a positive integer step.',
    );
  }

  const values: number[] = [];
  for (let value = definition.min; value <= definition.max; value += step) {
    values.push(value);
  }

  if (values.at(-1) !== definition.max) {
    values.push(definition.max);
  }

  return [...new Set(values)];
}

export function buildLogIntValues(
  name: string,
  definition: IntParamDefinition,
): number[] {
  ensureLogBounds(name, definition);
  if (definition.step === undefined) {
    throw new ValidationError(
      'Grid search requires log-scaled int parameters to define a multiplicative step.',
    );
  }
  if (!Number.isFinite(definition.step) || definition.step <= 1) {
    throw new ValidationError(
      'Grid search requires log-scaled int parameters to use a multiplicative step greater than 1.',
    );
  }

  const values: number[] = [];
  let current = definition.min;
  const limit = definition.max;
  while (current <= limit) {
    values.push(current);
    const next = Math.round(current * definition.step);
    if (next <= current) {
      throw new ValidationError(
        `Log-scaled int parameter "${name}" requires step to advance the range.`,
      );
    }
    current = next;
  }

  if (values.at(-1) !== definition.max) {
    values.push(definition.max);
  }

  return [...new Set(values)];
}

export function buildIntValues(
  name: string,
  definition: IntParamDefinition,
): number[] {
  if (definition.scale === 'log') {
    return buildLogIntValues(name, definition);
  }
  return buildLinearIntValues(definition);
}

function buildLinearFloatValues(definition: FloatParamDefinition): number[] {
  if (definition.step === undefined) {
    throw new ValidationError(
      'Grid search requires float parameters to define step.',
    );
  }
  if (definition.step <= 0 || !Number.isFinite(definition.step)) {
    throw new ValidationError(
      'Grid search requires float parameters to use a positive finite step.',
    );
  }

  const values: number[] = [];
  const epsilon = definition.step / 1000;
  for (
    let value = definition.min;
    value <= definition.max + epsilon;
    value += definition.step
  ) {
    values.push(
      roundToPrecision(
        clamp(
          definition.min +
            Math.round((value - definition.min) / definition.step) *
              definition.step,
          definition.min,
          definition.max,
        ),
      ),
    );
  }

  if (values.at(-1) !== definition.max) {
    values.push(roundToPrecision(definition.max));
  }

  return [...new Set(values)];
}

export function buildLogFloatValues(
  name: string,
  definition: FloatParamDefinition,
): number[] {
  ensureLogBounds(name, definition);
  if (definition.step === undefined) {
    throw new ValidationError(
      'Grid search requires log-scaled float parameters to define a multiplicative step.',
    );
  }
  if (!Number.isFinite(definition.step) || definition.step <= 1) {
    throw new ValidationError(
      'Grid search requires log-scaled float parameters to use a multiplicative step greater than 1.',
    );
  }

  const values: number[] = [];
  let current = definition.min;
  while (current <= definition.max) {
    values.push(roundToPrecision(current));
    const next = current * definition.step;
    if (next <= current) {
      throw new ValidationError(
        `Log-scaled float parameter "${name}" requires step to advance the range.`,
      );
    }
    current = next;
  }

  if (values.at(-1) !== definition.max) {
    values.push(roundToPrecision(definition.max));
  }

  return [...new Set(values)];
}

export function buildFloatValues(
  name: string,
  definition: FloatParamDefinition,
): number[] {
  if (definition.scale === 'log') {
    return buildLogFloatValues(name, definition);
  }
  return buildLinearFloatValues(definition);
}

export function buildDiscreteValues(
  name: string,
  definition: ParameterDefinition,
): unknown[] {
  switch (definition.type) {
    case 'enum':
      return [...definition.values];
    case 'int':
      return buildIntValues(name, definition);
    case 'float':
      return buildFloatValues(name, definition);
    default:
      throw new ValidationError(`Unsupported parameter type for "${name}".`);
  }
}

function isDiscreteDefinition(definition: ParameterDefinition): boolean {
  switch (definition.type) {
    case 'enum':
      return true;
    case 'int':
      return true;
    case 'float':
      return definition.step !== undefined;
    default:
      return false;
  }
}

function isDiscreteSpace(entries: [string, ParameterDefinition][]): boolean {
  return entries.every(([, definition]) => isDiscreteDefinition(definition));
}

export function discreteCardinality(
  entries: [string, ParameterDefinition][],
): number | null {
  if (!isDiscreteSpace(entries)) {
    return null;
  }

  return entries.reduce((product, [name, definition]) => {
    const size = buildDiscreteValues(name, definition).length;
    return product * size;
  }, 1);
}

function sampleLogValue(
  name: string,
  definition: FloatParamDefinition | IntParamDefinition,
  random: PythonRandom,
): number {
  ensureLogBounds(name, definition);
  const minLog = Math.log10(definition.min);
  const maxLog = Math.log10(definition.max);
  const exponent = random.uniform(minLog, maxLog);
  return 10 ** exponent;
}

export function sampleParameter(
  name: string,
  definition: ParameterDefinition,
  random: PythonRandom,
): unknown {
  switch (definition.type) {
    case 'enum':
      return random.choice(definition.values);
    case 'int': {
      if (definition.scale === 'log') {
        if (definition.step !== undefined) {
          return random.choice(buildIntValues(name, definition));
        }
        return clamp(
          Math.round(sampleLogValue(name, definition, random)),
          definition.min,
          definition.max,
        );
      }

      if (definition.step !== undefined && definition.step !== 1) {
        return random.choice(buildIntValues(name, definition));
      }
      return random.randint(definition.min, definition.max);
    }
    case 'float': {
      if (definition.scale === 'log') {
        if (definition.step !== undefined) {
          return random.choice(buildFloatValues(name, definition));
        }
        return roundToPrecision(sampleLogValue(name, definition, random));
      }

      const sampled = random.uniform(definition.min, definition.max);
      if (definition.step === undefined) {
        return roundToPrecision(sampled);
      }

      const snapped =
        Math.round((sampled - definition.min) / definition.step) *
          definition.step +
        definition.min;
      return roundToPrecision(clamp(snapped, definition.min, definition.max));
    }
    default:
      return undefined;
  }
}

export function cartesianProduct(
  entries: [string, unknown[]][],
): CandidateConfig[] {
  let product: CandidateConfig[] = [{}];

  for (const [name, values] of entries) {
    const next: CandidateConfig[] = [];
    for (const candidate of product) {
      for (const value of values) {
        next.push({ ...candidate, [name]: value });
      }
    }
    product = next;
  }

  return product;
}

export function sampleCandidateConfig(
  entries: [string, ParameterDefinition][],
  random: PythonRandom,
): CandidateConfig {
  return Object.fromEntries(
    entries.map(([name, definition]) => [
      name,
      sampleParameter(name, definition, random),
    ]),
  );
}
