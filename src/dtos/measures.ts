/**
 * MeasuresDict validation for metric-value maps.
 *
 * This aligns with the canonical TraigentSchema value contracts used in
 * execution/metric_submission_schema.json and
 * evaluation/configuration_run_schema.json:
 * - Max 50 keys
 * - Python identifier keys (^[a-zA-Z_][a-zA-Z0-9_]*$)
 * - Finite numeric values only (number or null)
 *
 * It does not model measure catalog definitions from measures/measure_schema.json.
 */
import { z } from 'zod';

/** Maximum number of measure keys allowed */
export const MAX_MEASURES_KEYS = 50;

/** Pattern for valid measure keys (Python identifiers) */
export const MEASURE_KEY_PATTERN = /^[a-zA-Z_]\w*$/;

/**
 * Zod schema for a single measure key.
 */
export const MeasureKeySchema = z.string().regex(MEASURE_KEY_PATTERN, {
  message: 'Measure key must be a valid Python identifier (e.g., "accuracy_score", not "accuracy-score")',
});

/**
 * Zod schema for a single measure value.
 */
export const MeasureValueSchema = z.number().finite().nullable();

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
 * Handle validation error based on strict mode.
 */
function handleValidationError(
  msg: string,
  strict: boolean,
  warn: (msg: string) => void
): void {
  if (strict) {
    throw new Error(msg);
  }
  warn(msg);
}

/**
 * Check if a value is a valid measure value (number or null).
 */
function isValidMeasureValue(value: unknown): value is number | null {
  return value === null || (typeof value === 'number' && Number.isFinite(value));
}

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
    handleValidationError(
      `MeasuresDict has ${keys.length} keys, truncating to ${MAX_MEASURES_KEYS}`,
      strict,
      warn
    );
  }

  for (const key of keys.slice(0, MAX_MEASURES_KEYS)) {
    if (!MEASURE_KEY_PATTERN.test(key)) {
      handleValidationError(
        `Invalid measure key "${key}" - must be a valid Python identifier`,
        strict,
        warn
      );
      continue;
    }

    const value = input[key];

    if (isValidMeasureValue(value)) {
      result[key] = value;
    } else {
      handleValidationError(
        `Non-numeric measure value for "${key}": ${typeof value} - use metadata for non-numeric values`,
        strict,
        warn
      );
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
