/**
 * MeasuresDict validation - mirrors Python SDK's traigent/cloud/dtos.py MeasuresDict.
 *
 * Enforces:
 * - Max 50 keys (prevent unbounded memory)
 * - Python identifier keys (^[a-zA-Z_][a-zA-Z0-9_]*$)
 * - Numeric values only (number or null)
 */
import { z } from 'zod';

/** Maximum number of measure keys allowed */
export const MAX_MEASURES_KEYS = 50;

/** Pattern for valid measure keys (Python identifiers) */
export const MEASURE_KEY_PATTERN = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

/**
 * Zod schema for a single measure key.
 */
export const MeasureKeySchema = z.string().regex(MEASURE_KEY_PATTERN, {
  message: 'Measure key must be a valid Python identifier (e.g., "accuracy_score", not "accuracy-score")',
});

/**
 * Zod schema for a single measure value.
 */
export const MeasureValueSchema = z.number().nullable();

/**
 * Zod schema for MeasuresDict with all validations.
 */
export const MeasuresDictSchema = z
  .record(MeasureKeySchema, MeasureValueSchema)
  .refine((obj) => Object.keys(obj).length <= MAX_MEASURES_KEYS, {
    message: `MeasuresDict cannot have more than ${MAX_MEASURES_KEYS} keys`,
  });

export type MeasuresDict = z.infer<typeof MeasuresDictSchema>;

/**
 * Validate and sanitize a measures dictionary.
 * Filters out invalid keys and non-numeric values with warnings.
 */
export function sanitizeMeasures(
  input: Record<string, unknown>,
  options: { strict?: boolean; warn?: (msg: string) => void } = {}
): MeasuresDict {
  const { strict = false, warn = console.warn } = options;
  const result: MeasuresDict = {};
  const keys = Object.keys(input);

  if (keys.length > MAX_MEASURES_KEYS) {
    const msg = `MeasuresDict has ${keys.length} keys, truncating to ${MAX_MEASURES_KEYS}`;
    if (strict) {
      throw new Error(msg);
    }
    warn(msg);
  }

  for (const key of keys.slice(0, MAX_MEASURES_KEYS)) {
    // Validate key pattern
    if (!MEASURE_KEY_PATTERN.test(key)) {
      const msg = `Invalid measure key "${key}" - must be a valid Python identifier`;
      if (strict) {
        throw new Error(msg);
      }
      warn(msg);
      continue;
    }

    const value = input[key];

    // Validate value is numeric
    if (value === null) {
      result[key] = null;
    } else if (typeof value === 'number' && !Number.isNaN(value)) {
      result[key] = value;
    } else {
      const msg = `Non-numeric measure value for "${key}": ${typeof value} - use metadata for non-numeric values`;
      if (strict) {
        throw new Error(msg);
      }
      warn(msg);
      // Skip non-numeric values
    }
  }

  return result;
}

/**
 * Create an empty MeasuresDict.
 */
export function createEmptyMeasures(): MeasuresDict {
  return {};
}

/**
 * Merge multiple MeasuresDict objects, later values override earlier.
 */
export function mergeMeasures(...dicts: MeasuresDict[]): MeasuresDict {
  const merged: MeasuresDict = {};

  for (const dict of dicts) {
    for (const [key, value] of Object.entries(dict)) {
      merged[key] = value;
    }
  }

  // Validate final result
  return MeasuresDictSchema.parse(merged);
}

/**
 * Add a prefix to all measure keys.
 * Useful for namespacing metrics from different sources.
 */
export function prefixMeasures(
  dict: MeasuresDict,
  prefix: string
): MeasuresDict {
  const result: MeasuresDict = {};

  for (const [key, value] of Object.entries(dict)) {
    const prefixedKey = `${prefix}${key}`;
    if (MEASURE_KEY_PATTERN.test(prefixedKey)) {
      result[prefixedKey] = value;
    }
  }

  return result;
}
