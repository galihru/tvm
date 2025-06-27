import { loadImageDataset } from './loaders/imageLoader';
import { loadPDFDataset } from './loaders/pdfLoader';
import { loadCSVDataset } from './loaders/csvLoader';
import { analyzeDataset } from './core/datasetAnalyzer';
import { recommendModel } from './core/modelRecommender';

export const loadDataset = async (file: File): Promise<any> => {
    if (file.name.endsWith('.zip')) {
        if (file.name.toLowerCase().includes('image')) {
            return loadImageDataset(file);
        } else if (file.name.toLowerCase().includes('pdf')) {
            return loadPDFDataset(file);
        } else {
            try {
                return await loadImageDataset(file);
            } catch (e) {
                return await loadPDFDataset(file);
            }
        }
    } else if (file.name.endsWith('.csv')) {
        return loadCSVDataset(file);
    } else {
        throw new Error('Unsupported file format');
    }
};

export { analyzeDataset, recommendModel };
