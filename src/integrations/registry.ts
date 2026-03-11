import type {
  FrameworkAutoOverrideStatus,
  FrameworkTarget,
} from '../optimization/types.js';

const activeFrameworkTargets = new Set<FrameworkTarget>();

export function registerFrameworkTarget(target: FrameworkTarget): void {
  activeFrameworkTargets.add(target);
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

export function hasRegisteredFrameworkTarget(
  targets: readonly FrameworkTarget[] | undefined,
): boolean {
  return resolveRegisteredFrameworkTargets(targets).length > 0;
}

export function describeFrameworkAutoOverride(
  targets: readonly FrameworkTarget[] | undefined,
  autoOverrideFrameworks = true,
): FrameworkAutoOverrideStatus {
  const activeTargets = getRegisteredFrameworkTargets();
  const requestedTargets = targets ? [...targets] : undefined;
  const selectedTargets = autoOverrideFrameworks
    ? resolveRegisteredFrameworkTargets(targets)
    : [];

  if (!autoOverrideFrameworks) {
    return {
      autoOverrideFrameworks,
      requestedTargets,
      activeTargets,
      selectedTargets,
      enabled: false,
      reason:
        "Framework auto-override is disabled for this seamless configuration.",
    };
  }

  if (activeTargets.length === 0) {
    return {
      autoOverrideFrameworks,
      requestedTargets,
      activeTargets,
      selectedTargets,
      enabled: false,
      reason:
        "No wrapped framework targets are currently registered for seamless interception.",
    };
  }

  if (requestedTargets && selectedTargets.length === 0) {
    return {
      autoOverrideFrameworks,
      requestedTargets,
      activeTargets,
      selectedTargets,
      enabled: false,
      reason:
        "None of the requested framework targets are currently registered.",
    };
  }

  return {
    autoOverrideFrameworks,
    requestedTargets,
    activeTargets,
    selectedTargets,
    enabled: selectedTargets.length > 0,
    reason:
      requestedTargets && requestedTargets.length > 0
        ? "Using the requested registered framework targets for seamless interception."
        : "Using all active registered framework targets for seamless interception.",
  };
}

export function clearRegisteredFrameworkTargets(): void {
  activeFrameworkTargets.clear();
}
