export declare function loadDataset(file: File): Promise<{
    type: string;
    size: number;
    classes?: number;
    features?: number;
    classDistribution?: Record<string, number>;
    avgResolution?: number;
    channels?: number;
    avgTextLength?: number;
    avgPages?: number;
    vocabSize?: number;
}>;

export declare function analyzeDataset(metadata: any): any;

export declare function recommendModel(analysis: any): {
    model: string;
    layers: any[];
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
