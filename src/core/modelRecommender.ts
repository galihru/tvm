import { calculateEpochs, calculateLearningRate } from './mathUtils';

const MODEL_RECOMMENDATIONS = {
  'image_low': {
    model: 'MobileNetV2',
    layers: [
      { type: 'convolution', filters: 32, kernel: 3 },
      { type: 'separable-conv', filters: 64 },
      { type: 'global-pooling' }
    ],
    paper: "MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler, 2018)"
  },
  'image_high': {
    model: 'EfficientNetB7',
    layers: [
      { type: 'convolution', filters: 64, kernel: 7 },
      { type: 'mb-conv', filters: 128, expansion: 6 },
      { type: 'mb-conv', filters: 256, expansion: 6 },
      { type: 'attention-pooling' }
    ],
    paper: "EfficientNet: Rethinking Model Scaling for CNN (Tan, 2019)"
  },
  'tabular_low': {
    model: 'MLP (Multi-Layer Perceptron)',
    layers: [
      { type: 'dense', units: 64, activation: 'relu' },
      { type: 'dropout', rate: 0.2 },
      { type: 'dense', units: 32, activation: 'relu' }
    ],
    paper: "Deep Learning for Tabular Data (Borisov, 2021)"
  },
  'text_medium': {
    model: 'DistilBERT',
    layers: [
      { type: 'embedding', dim: 768 },
      { type: 'transformer', heads: 12, layers: 6 },
      { type: 'pooling' }
    ],
    paper: "DistilBERT, a distilled version of BERT (Sanh, 2019)"
  }
};

export const recommendModel = (analysis: any) => {
  const key = analysis.recommendationKey;
  const complexityLevel = analysis.complexity > 1e6 ? 'high' : analysis.complexity > 1e4 ? 'medium' : 'low';
  const modelKey = `${analysis.dataType}_${complexityLevel}`;
  
  const recommendation = MODEL_RECOMMENDATIONS[modelKey] || {
    model: 'Default Neural Network',
    layers: [
      { type: 'dense', units: 128, activation: 'relu' },
      { type: 'dense', units: 64, activation: 'relu' }
    ],
    paper: "Deep Learning (Goodfellow, 2016)"
  };

  const epochs = calculateEpochs(analysis.size, analysis.complexity);
  const learningRate = calculateLearningRate(analysis.entropy);

  return {
    ...recommendation,
    hyperparameters: {
      epochs,
      learningRate,
      batchSize: Math.min(256, Math.pow(2, Math.floor(Math.log2(analysis.size/100)))),
      validationSplit: 0.2,
      earlyStopping: true
    },
    explanation: `Based on ${analysis.size} samples with ${analysis.complexity.toExponential(2)} complexity, 
    we recommend ${recommendation.model} from research: "${recommendation.paper}". 
    Use initial learning rate of ${learningRate.toFixed(5)} with ${epochs} epochs.`
  };
};
