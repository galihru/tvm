export const calculateEntropy = (values: number[]): number => {
    const total = values.reduce((sum, val) => sum + val, 0);
    return values.reduce((entropy, val) => {
        const p = val / total;
        return entropy - (p > 0 ? p * Math.log2(p) : 0);
    }, 0);
};

export const calculateEpochs = (datasetSize: number, complexity: number): number => {
    return Math.min(500, Math.max(20, Math.round(50 + 150 * Math.log(complexity) / Math.log(datasetSize))));
};

export const calculateLearningRate = (entropy: number): number => {
    return Math.min(0.1, Math.max(1e-5, 0.1 * Math.exp(-1.5 * entropy)));
};

export const calculateBatchSize = (datasetSize: number): number => {
    return Math.pow(2, Math.floor(Math.log2(Math.sqrt(datasetSize))));
};
