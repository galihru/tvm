import Papa, {
  ParseResult,
  ParseError,
  ParseConfig,
} from 'papaparse';

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
          if (data.length === 0) {
            reject(new Error('CSV file is empty'));
            return;
          }

          // Ambil nama kolom terakhir sebagai label
          const columns = results.meta.fields ?? [];
          const targetColumn = columns[columns.length - 1];

          // Hitung distribusi kelas
          const classDistribution: Record<string, number> = {};
          data.forEach((row) => {
            const label = row[targetColumn] as string | undefined;
            if (label !== undefined) {
              classDistribution[label] = (classDistribution[label] || 0) + 1;
            }
          });

          resolve({
            type: 'tabular',
            size: data.length,
            features: columns.length - 1,
            classes: Object.keys(classDistribution).length,
            classDistribution,
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
