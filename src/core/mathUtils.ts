export const calculateEntropy = (values: number[]): number => {
  const total = values.reduce((sum, val) => sum + val, 0);
  return values.reduce((entropy, val) => {
    const p = val / total;
    return entropy - (p > 0 ? p * Math.log2(p) : 0);
  }, 0);
};

export const calculateEpochs = (datasetSize: number, complexity: number): number => {
  return Math.min(500, Math.max(50, Math.round(100 * complexity / Math.log(datasetSize)));
};

export const calculateLearningRate = (entropy: number): number => {
  return Math.min(0.01, Math.max(0.0001, 0.1 * Math.exp(-0.5 * entropy)));
};
