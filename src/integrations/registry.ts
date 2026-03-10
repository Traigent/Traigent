import type { FrameworkTarget } from '../optimization/types.js';

const activeFrameworkTargets = new Set<FrameworkTarget>();

export function registerFrameworkTarget(target: FrameworkTarget): void {
  activeFrameworkTargets.add(target);
}

export function hasRegisteredFrameworkTarget(
  targets: readonly FrameworkTarget[] | undefined,
): boolean {
  return resolveRegisteredFrameworkTargets(targets).length > 0;
}

export function getRegisteredFrameworkTargets(): FrameworkTarget[] {
  return [...activeFrameworkTargets].sort();
}

export function resolveRegisteredFrameworkTargets(
  targets: readonly FrameworkTarget[] | undefined,
): FrameworkTarget[] {
  if (!targets || targets.length === 0) {
    return getRegisteredFrameworkTargets();
  }

  return targets.filter((target) => activeFrameworkTargets.has(target));
}

export function clearRegisteredFrameworkTargets(): void {
  activeFrameworkTargets.clear();
}
