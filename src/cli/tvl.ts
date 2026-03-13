import { loadTvlSpec } from '../optimization/tvl.js';

export interface TvlInspectionResult {
  file: string;
  configurationKeys: string[];
  objectiveMetrics: string[];
  usedFeatures: string[];
  warnings: string[];
  nativeCompatibility: Awaited<ReturnType<typeof loadTvlSpec>>['nativeCompatibility'];
}

function getObjectiveMetric(
  objective: Awaited<ReturnType<typeof loadTvlSpec>>['spec']['objectives'][number]
): string {
  return typeof objective === 'string' ? objective : objective.metric;
}

export async function inspectTvlFiles(paths: string[]): Promise<TvlInspectionResult[]> {
  return Promise.all(
    paths.map(async (file) => {
      const loaded = await loadTvlSpec({ path: file });
      return {
        file,
        configurationKeys: Object.keys(loaded.spec.configurationSpace),
        objectiveMetrics: loaded.spec.objectives.map(getObjectiveMetric),
        usedFeatures: [...loaded.nativeCompatibility.usedFeatures],
        warnings: [...loaded.nativeCompatibility.warnings],
        nativeCompatibility: loaded.nativeCompatibility,
      };
    })
  );
}
