import { ValidationError } from '../core/errors.js';

export function ensureFiniteNumber(value: unknown, message: string): asserts value is number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(message);
  }
}
