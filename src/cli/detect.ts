import {
  discoverTunedVariablesFromFile,
  type TunedVariableDiscoveryResult,
} from '../tuned-variables/index.js';

export interface DetectTunedVariablesOptions {
  functionName?: string;
  includeLowConfidence?: boolean;
}

export interface DetectTunedVariablesFileResult {
  file: string;
  results: TunedVariableDiscoveryResult[];
}

export function detectTunedVariablesInFiles(
  files: readonly string[],
  options: DetectTunedVariablesOptions = {}
): DetectTunedVariablesFileResult[] {
  return files.map((file) => ({
    file,
    results: discoverTunedVariablesFromFile(file, {
      functionName: options.functionName,
      includeLowConfidence: options.includeLowConfidence ?? false,
    }),
  }));
}
