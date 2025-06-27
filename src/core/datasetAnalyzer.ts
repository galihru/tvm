import { calculateEntropy } from './mathUtils';

export const analyzeDataset = (metadata: any) => {
  let analysis: any = {};

  if (metadata.type === 'image') {
    analysis.complexity = metadata.classes * metadata.avgResolution * metadata.channels;
    analysis.dataType = 'image';
  }
  
  else if (metadata.type === 'tabular') {
    const entropy = calculateEntropy(Object.values(metadata.classDistribution));
    analysis.complexity = entropy * metadata.features;
    analysis.dataType = 'tabular';
  }
  
  else if (metadata.type === 'text') {
    analysis.complexity = metadata.avgLength * metadata.vocabSize;
    analysis.dataType = 'text';
  }

  return {
    ...metadata,
    ...analysis,
    recommendationKey: `${analysis.dataType}_${Math.round(analysis.complexity)}`
  };
};
