import Papa, { ParseResult, ParseError, ParseConfig } from 'papaparse';

export interface CSVDatasetSummary {
  type: 'tabular';
  size: number;
  features: number;
  classes: number;
  classDistribution: Record<string, number>;
}

export const loadCSVDataset = (file: Blob): Promise<CSVDatasetSummary> => {
  return new Promise<CSVDatasetSummary>((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      const csvData = e.target?.result as string;
      const config: ParseConfig<Record<string, unknown>> = {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results: ParseResult<Record<string, unknown>>) => {
          const data = results.data;
          if (!data.length) {
            reject(new Error('CSV file is empty'));
            return;
          }

          const columns = results.meta.fields ?? [];
          const target = columns[columns.length - 1];
          const dist: Record<string, number> = {};

          data.forEach((row) => {
            const label = row[target] as string | undefined;
            if (label !== undefined) {
              dist[label] = (dist[label] || 0) + 1;
            }
          });

          resolve({
            type: 'tabular',
            size: data.length,
            features: columns.length - 1,
            classes: Object.keys(dist).length,
            classDistribution: dist,
          });
        },
        error: (err: ParseError) => {
          reject(err);
        },
      };

      Papa.parse<Record<string, unknown>>(csvData, config);
    };

    reader.onerror = () => {
      reject(new Error('Failed to read CSV file'));
    };

    reader.readAsText(file);
  });
};
