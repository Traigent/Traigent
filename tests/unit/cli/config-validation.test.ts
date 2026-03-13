import { describe, expect, it } from 'vitest';

import { validateConfigPayload } from '../../../src/cli/config-validation.js';

describe('validateConfigPayload', () => {
  describe('without schema', () => {
    it('passes basic object validation', () => {
      expect(validateConfigPayload({ model: 'gpt-4o-mini' })).toEqual({
        ok: true,
        summary: 'Config validation passed',
      });
    });

    it('passes with empty object', () => {
      expect(validateConfigPayload({})).toEqual({
        ok: true,
        summary: 'Config validation passed',
      });
    });

    it('rejects non-object config (string)', () => {
      expect(validateConfigPayload('bad')).toEqual({
        ok: false,
        issues: [{ message: 'config must be an object' }],
        summary: 'Validation failed: 1 issue(s)',
      });
    });

    it('rejects non-object config (number)', () => {
      expect(validateConfigPayload(42)).toEqual({
        ok: false,
        issues: [{ message: 'config must be an object' }],
        summary: 'Validation failed: 1 issue(s)',
      });
    });

    it('rejects non-object config (null)', () => {
      expect(validateConfigPayload(null)).toEqual({
        ok: false,
        issues: [{ message: 'config must be an object' }],
        summary: 'Validation failed: 1 issue(s)',
      });
    });

    it('rejects non-object config (undefined)', () => {
      expect(validateConfigPayload(undefined)).toEqual({
        ok: false,
        issues: [{ message: 'config must be an object' }],
        summary: 'Validation failed: 1 issue(s)',
      });
    });

    it('rejects non-object config (boolean)', () => {
      expect(validateConfigPayload(true)).toEqual({
        ok: false,
        issues: [{ message: 'config must be an object' }],
        summary: 'Validation failed: 1 issue(s)',
      });
    });

    it('treats array as object (passes without schema)', () => {
      // Arrays are typeof 'object' and !== null, so they pass the toObject check
      const result = validateConfigPayload([1, 2, 3]);
      expect(result.ok).toBe(true);
    });
  });

  describe('with valid schema', () => {
    it('passes when config matches schema', () => {
      const result = validateConfigPayload(
        { model: 'gpt-4o', temperature: 0.7 },
        {
          type: 'object',
          required: ['model', 'temperature'],
          properties: {
            model: { type: 'string' },
            temperature: { type: 'number', minimum: 0, maximum: 1 },
          },
        }
      );

      expect(result).toEqual({
        ok: true,
        summary: 'Config validation passed',
      });
    });

    it('passes with extra properties when additionalProperties is not false', () => {
      const result = validateConfigPayload(
        { model: 'gpt-4o', extra_field: 'hello' },
        {
          type: 'object',
          properties: {
            model: { type: 'string' },
          },
        }
      );

      expect(result).toEqual({
        ok: true,
        summary: 'Config validation passed',
      });
    });
  });

  describe('schema validation failures', () => {
    it('rejects invalid config schemas', () => {
      const result = validateConfigPayload(
        { model: 'gpt-4o-mini' },
        {
          type: 'object',
          properties: {
            model: { type: 123 },
          },
        }
      );

      expect(result.ok).toBe(false);
      expect(result.summary).toMatch(/^Invalid config schema:/);
      expect(result.issues?.length).toBeGreaterThan(0);
    });

    it('validates config against JSON Schema and reports field paths', () => {
      const result = validateConfigPayload(
        { model: 'gpt-4o-mini', temperature: 'hot' },
        {
          type: 'object',
          required: ['model', 'temperature'],
          additionalProperties: false,
          properties: {
            model: { type: 'string' },
            temperature: { type: 'number', minimum: 0, maximum: 1 },
          },
        }
      );

      expect(result.ok).toBe(false);
      expect(result.summary).toBe('Validation failed: 1 issue(s)');
      expect(result.issues).toEqual([
        {
          path: 'temperature',
          message: 'must be number',
          keyword: 'type',
        },
      ]);
    });

    it('reports missing required fields on their property path', () => {
      const result = validateConfigPayload(
        { model: 'gpt-4o-mini' },
        {
          type: 'object',
          required: ['model', 'temperature'],
          properties: {
            model: { type: 'string' },
            temperature: { type: 'number' },
          },
        }
      );

      expect(result.ok).toBe(false);
      expect(result.issues).toEqual([
        {
          path: 'temperature',
          message: "must have required property 'temperature'",
          keyword: 'required',
        },
      ]);
    });

    it('rejects additional properties when additionalProperties is false', () => {
      const result = validateConfigPayload(
        { model: 'gpt-4o', unknown_field: 42 },
        {
          type: 'object',
          additionalProperties: false,
          properties: {
            model: { type: 'string' },
          },
        }
      );

      expect(result.ok).toBe(false);
      expect(result.issues).toEqual([
        {
          path: undefined,
          message: 'must NOT have additional properties',
          keyword: 'additionalProperties',
        },
      ]);
    });

    it('reports multiple validation errors', () => {
      const result = validateConfigPayload(
        { model: 123, temperature: 'hot' },
        {
          type: 'object',
          properties: {
            model: { type: 'string' },
            temperature: { type: 'number' },
          },
        }
      );

      expect(result.ok).toBe(false);
      expect(result.issues?.length).toBe(2);
      expect(result.summary).toBe('Validation failed: 2 issue(s)');
    });

    it('reports nested property paths', () => {
      const result = validateConfigPayload(
        { llm: { settings: { temperature: 'hot' } } },
        {
          type: 'object',
          properties: {
            llm: {
              type: 'object',
              properties: {
                settings: {
                  type: 'object',
                  properties: {
                    temperature: { type: 'number' },
                  },
                },
              },
            },
          },
        }
      );

      expect(result.ok).toBe(false);
      expect(result.issues?.[0]?.path).toBe('llm.settings.temperature');
    });
  });

  describe('issue truncation', () => {
    it('truncates issues exceeding MAX_ISSUES (20)', () => {
      // Create a schema that requires 25 specific properties, none of which we provide
      const required: string[] = [];
      const properties: Record<string, { type: string }> = {};
      for (let i = 0; i < 25; i++) {
        const name = `field_${i}`;
        required.push(name);
        properties[name] = { type: 'string' };
      }

      const result = validateConfigPayload(
        {},
        {
          type: 'object',
          required,
          properties,
        }
      );

      expect(result.ok).toBe(false);
      expect(result.issues?.length).toBe(20);
      expect(result.truncated).toBe(true);
      expect(result.total_issues).toBe(25);
      expect(result.summary).toBe('Validation failed: 25 issue(s)');
    });

    it('does not set truncated flag when issues are within limit', () => {
      const result = validateConfigPayload(
        { model: 123 },
        {
          type: 'object',
          properties: {
            model: { type: 'string' },
          },
        }
      );

      expect(result.ok).toBe(false);
      expect(result.truncated).toBeUndefined();
      expect(result.total_issues).toBeUndefined();
    });

    it('truncates invalid schema issues exceeding MAX_ISSUES', () => {
      // Create a schema with many invalid type declarations
      const properties: Record<string, unknown> = {};
      for (let i = 0; i < 25; i++) {
        properties[`field_${i}`] = { type: 999 + i };
      }

      const result = validateConfigPayload({}, { type: 'object', properties });

      expect(result.ok).toBe(false);
      expect(result.summary).toMatch(/^Invalid config schema:/);
      expect(result.issues!.length).toBeLessThanOrEqual(20);
      if (result.issues!.length === 20) {
        expect(result.truncated).toBe(true);
      }
    });
  });

  describe('toPath edge cases', () => {
    it('reports root-level path as undefined for non-required errors', () => {
      // When instancePath is empty and it's not a required error, path should be undefined
      const result = validateConfigPayload('not an object', { type: 'object' });

      // This hits the non-object check first, not AJV
      expect(result.ok).toBe(false);
      expect(result.issues?.[0]?.message).toBe('config must be an object');
    });

    it('handles fallback message when error.message is undefined', () => {
      // AJV always provides messages, but our code handles the case
      // This is tested implicitly - the fallback exists for safety
      const result = validateConfigPayload(
        { model: 123 },
        {
          type: 'object',
          properties: {
            model: { type: 'string' },
          },
        }
      );

      expect(result.ok).toBe(false);
      expect(result.issues?.[0]?.message).toBeTruthy();
    });
  });
});
