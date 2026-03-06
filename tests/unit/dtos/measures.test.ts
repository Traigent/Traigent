/**
 * Unit tests for MeasuresDict validation.
 */
import { describe, it, expect, vi } from 'vitest';
import {
  MeasuresDictSchema,
  sanitizeMeasures,
  mergeMeasures,
  prefixMeasures,
  createEmptyMeasures,
  MAX_MEASURES_KEYS,
  MEASURE_KEY_PATTERN,
} from '../../../src/dtos/measures.js';

describe('MEASURE_KEY_PATTERN', () => {
  it('should match valid Python identifiers', () => {
    expect(MEASURE_KEY_PATTERN.test('accuracy')).toBe(true);
    expect(MEASURE_KEY_PATTERN.test('accuracy_score')).toBe(true);
    expect(MEASURE_KEY_PATTERN.test('_private')).toBe(true);
    expect(MEASURE_KEY_PATTERN.test('__dunder__')).toBe(true);
    expect(MEASURE_KEY_PATTERN.test('CamelCase')).toBe(true);
    expect(MEASURE_KEY_PATTERN.test('with123numbers')).toBe(true);
  });

  it('should reject invalid identifiers', () => {
    expect(MEASURE_KEY_PATTERN.test('accuracy-score')).toBe(false);
    expect(MEASURE_KEY_PATTERN.test('123starts_with_number')).toBe(false);
    expect(MEASURE_KEY_PATTERN.test('has spaces')).toBe(false);
    expect(MEASURE_KEY_PATTERN.test('has.dots')).toBe(false);
    expect(MEASURE_KEY_PATTERN.test('')).toBe(false);
  });
});

describe('MeasuresDictSchema', () => {
  it('should validate valid measures dict', () => {
    const measures = {
      accuracy: 0.95,
      latency_ms: 120.5,
      cost: null,
    };

    const result = MeasuresDictSchema.safeParse(measures);
    expect(result.success).toBe(true);
  });

  it('should reject dict with too many keys', () => {
    const measures: Record<string, number> = {};
    for (let i = 0; i < MAX_MEASURES_KEYS + 1; i++) {
      measures[`metric_${i}`] = i;
    }

    const result = MeasuresDictSchema.safeParse(measures);
    expect(result.success).toBe(false);
  });

  it('should reject invalid key patterns', () => {
    const measures = { 'invalid-key': 0.5 };
    const result = MeasuresDictSchema.safeParse(measures);
    expect(result.success).toBe(false);
  });
});

describe('sanitizeMeasures()', () => {
  it('should pass through valid measures', () => {
    const input = { accuracy: 0.95, latency: 100 };
    const result = sanitizeMeasures(input);
    expect(result).toEqual(input);
  });

  it('should filter out invalid keys with warning', () => {
    const warn = vi.fn();
    const input = { valid_key: 0.5, 'invalid-key': 0.3 };

    const result = sanitizeMeasures(input, { warn });

    expect(result).toEqual({ valid_key: 0.5 });
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('Invalid measure key')
    );
  });

  it('should filter out non-numeric values with warning', () => {
    const warn = vi.fn();
    const input = { accuracy: 0.95, model_name: 'gpt-4' };

    const result = sanitizeMeasures(input as Record<string, unknown>, { warn });

    expect(result).toEqual({ accuracy: 0.95 });
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('Non-numeric measure value')
    );
  });

  it('should throw in strict mode for invalid keys', () => {
    const input = { 'invalid-key': 0.5 };

    expect(() => sanitizeMeasures(input, { strict: true })).toThrow(
      'Invalid measure key'
    );
  });

  it('should truncate to MAX_MEASURES_KEYS', () => {
    const warn = vi.fn();
    const input: Record<string, number> = {};
    for (let i = 0; i < 60; i++) {
      input[`metric_${i}`] = i;
    }

    const result = sanitizeMeasures(input, { warn });

    expect(Object.keys(result).length).toBe(MAX_MEASURES_KEYS);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining('truncating')
    );
  });

  it('should throw in strict mode for too many keys', () => {
    const input: Record<string, number> = {};
    for (let i = 0; i < 60; i++) {
      input[`metric_${i}`] = i;
    }

    expect(() => sanitizeMeasures(input, { strict: true })).toThrow(
      'truncating'
    );
  });

  it('should throw in strict mode for non-numeric values', () => {
    const input = { accuracy: 0.95, model_name: 'gpt-4' };

    expect(() => sanitizeMeasures(input as Record<string, unknown>, { strict: true })).toThrow(
      'Non-numeric measure value'
    );
  });

  it('should preserve null values', () => {
    const input = { accuracy: 0.95, missing: null };
    const result = sanitizeMeasures(input);
    expect(result).toEqual({ accuracy: 0.95, missing: null });
  });

  it('should filter out NaN values', () => {
    const warn = vi.fn();
    const input = { valid: 0.5, invalid: NaN };
    const result = sanitizeMeasures(input, { warn });
    expect(result).toEqual({ valid: 0.5 });
  });
});

describe('mergeMeasures()', () => {
  it('should merge multiple dicts', () => {
    const a = { accuracy: 0.9 };
    const b = { latency: 100 };
    const c = { cost: 0.01 };

    const result = mergeMeasures(a, b, c);

    expect(result).toEqual({
      accuracy: 0.9,
      latency: 100,
      cost: 0.01,
    });
  });

  it('should override earlier values with later ones', () => {
    const a = { accuracy: 0.9 };
    const b = { accuracy: 0.95 };

    const result = mergeMeasures(a, b);

    expect(result).toEqual({ accuracy: 0.95 });
  });

  it('should return empty dict for no arguments', () => {
    const result = mergeMeasures();
    expect(result).toEqual({});
  });
});

describe('createEmptyMeasures()', () => {
  it('should return an empty object', () => {
    const result = createEmptyMeasures();
    expect(result).toEqual({});
    expect(Object.keys(result).length).toBe(0);
  });
});

describe('prefixMeasures()', () => {
  it('should add prefix to all keys', () => {
    const measures = { accuracy: 0.95, latency: 100 };
    const result = prefixMeasures(measures, 'langchain_');

    expect(result).toEqual({
      langchain_accuracy: 0.95,
      langchain_latency: 100,
    });
  });

  it('should skip keys that become invalid after prefixing', () => {
    const measures = { accuracy: 0.95 };
    // Prefix starting with number would be invalid
    const result = prefixMeasures(measures, '123_');

    expect(result).toEqual({});
  });

  it('should handle empty dict', () => {
    const result = prefixMeasures({}, 'prefix_');
    expect(result).toEqual({});
  });
});
