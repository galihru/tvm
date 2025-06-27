import JSZip from 'jszip';
import pdf from 'pdf-parse';

export const loadPDFDataset = async (zipFile: Blob): Promise<any> => {
    const zip = new JSZip();
    await zip.loadAsync(zipFile);

    const classes: string[] = [];
    const pdfCounts: Record<string, number> = {};
    let totalPDFs = 0;
    let totalPages = 0;
    let totalTextLength = 0;
    let sampleCount = 0;

    const folderRegex = /(.+)\//;

    const filePromises: Promise<void>[] = [];
    zip.forEach((relativePath, file) => {
        if (!file.dir && relativePath.endsWith('.pdf')) {
            const match = relativePath.match(folderRegex);
            if (match) {
                const className = match[1];
                if (!classes.includes(className)) {
                    classes.push(className);
                    pdfCounts[className] = 0;
                }

                pdfCounts[className]++;
                totalPDFs++;

                if (totalPDFs % 20 === 0) {
                    filePromises.push((async () => {
                        try {
                            const pdfBlob = await file.async('arraybuffer');
                            const data = await pdf(Buffer.from(pdfBlob));
                            totalTextLength += data.text.length;
                            totalPages += data.numpages;
                            sampleCount++;
                        } catch (e) {
                            console.error(`Error processing PDF ${relativePath}:`, e);
                        }
                    })());
                }
            }
        }
    });

    await Promise.all(filePromises);

    const avgTextLength = sampleCount > 0 ? totalTextLength / sampleCount : 0;
    const avgPages = sampleCount > 0 ? totalPages / sampleCount : 0;

    return {
        type: 'text',
        size: totalPDFs,
        classes: classes.length,
        classDistribution: pdfCounts,
        avgTextLength,
        avgPages,
        vocabSize: Math.round(avgTextLength * 0.1)
    };
};
