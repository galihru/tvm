export interface TabularDatasetSummary {
  type: 'tabular';
  size: number;
  features: number;
  classes: number;
  classDistribution: Record<string, number>;
}

export interface ImageDatasetSummary {
  type: 'image';
  size: number;
  avgResolution: number;
  channels: number;
}

export interface TextDatasetSummary {
  type: 'text';
  size: number;
  avgTextLength: number;
  vocabSize: number;
}

export interface PdfDatasetSummary {
  type: 'pdf';
  size: number;
  avgPages: number;
}

// Union of all dataset summaries
type DatasetSummary =
  | TabularDatasetSummary
  | ImageDatasetSummary
  | TextDatasetSummary
  | PdfDatasetSummary;

/**
 * Load a dataset from a File or Blob and return its summary.
 * Use the `type` field to discriminate between dataset kinds.
 */
export declare function loadDataset(
  file: File | Blob
): Promise<DatasetSummary>;

/**
 * Analyze a dataset summary and return metadata for model recommendation.
 */
export declare function analyzeDataset(
  metadata: DatasetSummary
): Record<string, unknown>;

/**
 * Recommend a model based on analysis metadata.
 */
export declare function recommendModel(
  analysis: Record<string, unknown>
): {
  model: string;
  layers: Array<{
    type: string;
    filters?: number;
    kernel?: number;
    activation?: string;
  }>;
  paper: string;
  hyperparameters: {
    epochs: number;
    learningRate: number;
    batchSize: number;
    validationSplit: number;
    earlyStopping: boolean;
  };
  explanation: string;
};
