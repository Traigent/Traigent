import { isPlainObject } from './is-plain-object.js';

function clonePlainValue<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((entry) => clonePlainValue(entry)) as T;
  }

  if (isPlainObject(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, clonePlainValue(entry)]),
    ) as T;
  }

  return value;
}

function freezePlainValue<T>(value: T): T {
  if (Array.isArray(value)) {
    for (const entry of value) {
      freezePlainValue(entry);
    }
    return Object.freeze(value);
  }

  if (isPlainObject(value)) {
    for (const entry of Object.values(value)) {
      freezePlainValue(entry);
    }
    return Object.freeze(value);
  }

  return value;
}

export function cloneAndFreezePlainValue<T>(value: T): T {
  return freezePlainValue(clonePlainValue(value));
}
