/**
 * Sample dataset for sentiment classification demo.
 * This simulates what would normally come from a JSONL file.
 */

export interface DatasetExample {
  input: { text: string };
  output: 'positive' | 'negative' | 'neutral';
}

export const SENTIMENT_DATASET: DatasetExample[] = [
  { input: { text: "This product is amazing! Exceeded all my expectations." }, output: "positive" },
  { input: { text: "Terrible experience, very disappointed with the quality." }, output: "negative" },
  { input: { text: "The item arrived on time and works fine." }, output: "neutral" },
  { input: { text: "Best purchase I've ever made! Highly recommend!" }, output: "positive" },
  { input: { text: "Complete waste of money. Broke after one day." }, output: "negative" },
  { input: { text: "It does what it's supposed to do. Nothing special." }, output: "neutral" },
  { input: { text: "Absolutely love it! Changed my life!" }, output: "positive" },
  { input: { text: "Would not recommend. Poor customer service too." }, output: "negative" },
  { input: { text: "Average product. Met basic expectations." }, output: "neutral" },
  { input: { text: "Five stars! Perfect in every way!" }, output: "positive" },
  { input: { text: "The product works but the packaging was damaged." }, output: "neutral" },
  { input: { text: "Not bad, not great. It's okay for the price." }, output: "neutral" },
  { input: { text: "I'm somewhat satisfied but expected better battery life." }, output: "neutral" },
  { input: { text: "Great value! Works exactly as described and arrived early." }, output: "positive" },
  { input: { text: "Frustrating setup process but eventually worked fine." }, output: "neutral" },
  { input: { text: "This is the worst product I have ever purchased in my entire life." }, output: "negative" },
  { input: { text: "Meh. It's a product. It exists. I have mixed feelings." }, output: "neutral" },
  { input: { text: "Incredible quality! The craftsmanship is outstanding!" }, output: "positive" },
  { input: { text: "Save your money. This product is a scam." }, output: "negative" },
  { input: { text: "Decent for beginners, but professionals might want something better." }, output: "neutral" },
];

/**
 * Get a subset of the dataset by indices.
 */
export function getDatasetSubset(indices: number[]): DatasetExample[] {
  return indices.map(i => SENTIMENT_DATASET[i]).filter(Boolean);
}
