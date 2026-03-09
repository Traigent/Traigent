export function normalizeText(value) {
  return String(value)
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ");
}

export function exactMatchScore(actual, expected) {
  return normalizeText(actual).includes(normalizeText(expected)) ? 1 : 0;
}

export function semanticSimilarityScore(actual, expected) {
  const actualTokens = new Set(normalizeText(actual).split(" ").filter(Boolean));
  const expectedTokens = normalizeText(expected).split(" ").filter(Boolean);
  if (expectedTokens.length === 0) return 0;
  const matched = expectedTokens.filter((token) => actualTokens.has(token)).length;
  return matched / expectedTokens.length;
}

export function classificationScore(actual, expected) {
  return normalizeText(actual) === normalizeText(expected) ? 1 : 0;
}

export function codeGenerationScore(actual, task) {
  const text = normalizeText(actual);
  let score = 0;
  if (text.includes("function") || text.includes("def ")) score += 0.4;
  if (text.includes("return")) score += 0.3;
  if (normalizeText(task).includes("even") && text.includes("% 2")) score += 0.3;
  if (normalizeText(task).includes("adds") && text.includes("+")) score += 0.3;
  return Math.min(score, 1);
}
