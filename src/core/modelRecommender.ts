import { calculateEpochs, calculateLearningRate, calculateBatchSize } from './mathUtils';

export type QualityLevel = 'low' | 'medium' | 'high';
export type DataType = 'image' | 'tabular' | 'text';
export type ModelKey = `${DataType}_${QualityLevel}`;

export interface ModelRecommendationEntry {
  model: string;
  layers: Array<Record<string, unknown>>;
  paper: string;
}

export const MODEL_RECOMMENDATIONS: Record<ModelKey, ModelRecommendationEntry> = {
  image_low: {
    model: 'MobileNetV2',
    layers: [
      { type: 'convolution', filters: 32, kernel: 3, activation: 'relu' },
      { type: 'depthwise-conv', kernel: 3 },
      { type: 'pointwise-conv', filters: 64 },
      { type: 'global-average-pooling' }
    ],
    paper: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler, 2018)'
  },
  image_medium: {
    model: 'ResNet50',
    layers: [
      { type: 'convolution', filters: 64, kernel: 7, stride: 2 },
      { type: 'max-pooling', pool: 3, stride: 2 },
      { type: 'residual', filters: [64, 64, 256], blocks: 3 },
      { type: 'residual', filters: [128, 128, 512], blocks: 4 },
      { type: 'global-average-pooling' }
    ],
    paper: 'Deep Residual Learning for Image Recognition (He, 2016)'
  },
  image_high: {
    model: 'EfficientNetB0',
    layers: [
      { type: 'measurement', detail: 'example' }
    ],
    paper: 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan, 2019)'
  },

  tabular_low: {
    model: 'MLP (3 Layers)',
    layers: [
      { type: 'dense', units: 64, activation: 'relu' },
      { type: 'dropout', rate: 0.2 },
      { type: 'dense', units: 32, activation: 'relu' }
    ],
    paper: 'Tabular Data: Deep Learning is Not All You Need (Borisov, 2021)'
  },
  tabular_medium: {
    model: 'XGBoost',
    layers: [],
    paper: 'XGBoost: A Scalable Tree Boosting System (Chen & Guestrin, 2016)'
  },
  tabular_high: {
    model: 'TabTransformer',
    layers: [
      { type: 'embedding', input_dim: 100, output_dim: 32 },
      { type: 'transformer', heads: 4, key_dim: 16 },
      { type: 'flatten' },
      { type: 'dense', units: 128, activation: 'gelu' }
    ],
    paper: 'TabTransformer: Tabular Data Modeling Using Contextual Embeddings (Huang, 2020)'
  },

  text_low: {
    model: 'Bag-of-Words + MLP',
    layers: [
      { type: 'vectorize', method: 'tf-idf' },
      { type: 'dense', units: 128, activation: 'relu' }
    ],
    paper: 'Classical Approaches to Text Classification (Sebastiani, 2002)'
  },
  text_medium: {
    model: 'DistilBERT',
    layers: [
      { type: 'embedding', vocab_size: 30522, embed_dim: 768 },
      { type: 'transformer', num_heads: 12, ff_dim: 3072, num_layers: 6 },
      { type: 'pooling', mode: 'mean' }
    ],
    paper: 'DistilBERT, a distilled version of BERT (Sanh, 2019)'
  },
  text_high: {
    model: 'RoBERTa',
    layers: [
      { type: 'embedding', vocab_size: 50265, embed_dim: 1024 },
      { type: 'transformer', num_heads: 16, ff_dim: 4096, num_layers: 24 },
      { type: 'pooling', mode: 'cls' }
    ],
    paper: 'RoBERTa: A Robustly Optimized BERT Approach (Liu, 2019)'
  }
};

export interface AnalysisInput {
  dataType: DataType;
  complexity: number;
  size: number;
  entropy?: number;
}

export interface ModelWithHyperparameters extends ModelRecommendationEntry {
  hyperparameters: {
    epochs: number;
    learningRate: number;
    batchSize: number;
    validationSplit: number;
    earlyStopping: boolean;
  };
  explanation: string;
}

export const recommendModel = (
  analysis: AnalysisInput
): ModelWithHyperparameters => {
  const complexityLevel: QualityLevel =
    analysis.complexity > 1e6
      ? 'high'
      : analysis.complexity > 1e4
      ? 'medium'
      : 'low';

  const key = `${analysis.dataType}_${complexityLevel}` as ModelKey;
  const fallback = `${analysis.dataType}_medium` as ModelKey;

  const recommendation =
    MODEL_RECOMMENDATIONS[key] ?? MODEL_RECOMMENDATIONS[fallback];

  // Hitung hyperparameters
  const epochs = calculateEpochs(analysis.size, analysis.complexity);
  const learningRate = calculateLearningRate(analysis.entropy ?? 1);
  const batchSize = calculateBatchSize(analysis.size);

  return {
    ...recommendation,
    hyperparameters: {
      epochs,
      learningRate,
      batchSize,
      validationSplit: 0.2,
      earlyStopping: true
    },
    explanation: `Berdasarkan dataset ${analysis.dataType} ` +
      `dengan ${analysis.size} sampel dan kompleksitas ${analysis.complexity.toExponential(2)}, ` +
      `kami rekomendasikan model ${recommendation.model} ` +
      `(paper: "${recommendation.paper}"). ` +
      `Gunakan ${epochs} epoch dengan learning rate ${learningRate.toFixed(5)}.`
  };
};
