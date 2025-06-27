import Papa from 'papaparse';

export const loadCSVDataset = async (file: Blob): Promise<any> => {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const csvData = e.target?.result as string;
            Papa.parse(csvData, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: (results) => {
                    const data = results.data;
                    if (data.length === 0) {
                        reject(new Error("CSV file is empty"));
                        return;
                    }

                    const columns = results.meta.fields || [];
                    const targetColumn = columns[columns.length - 1];
                    const classDistribution: Record<string, number> = {};

                    data.forEach((row: any) => {
                        const label = row[targetColumn];
                        if (label !== undefined) {
                            classDistribution[label] = (classDistribution[label] || 0) + 1;
                        }
                    });

                    resolve({
                        type: 'tabular',
                        size: data.length,
                        features: columns.length - 1,
                        classes: Object.keys(classDistribution).length,
                        classDistribution
                    });
                },
                error: (error) => {
                    reject(error);
                }
            });
        };
        reader.onerror = () => {
            reject(new Error("Failed to read CSV file"));
        };
        reader.readAsText(file);
    });
};
