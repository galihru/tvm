// src/loaders/pdfLoader.ts

import JSZip from 'jszip';
import pdfParse, { PdfParseData } from 'pdf-parse';

export interface PdfDatasetSummary {
  type: 'pdf';
  size: number;
  classes: number;
  classDistribution: Record<string, number>;
  avgTextLength: number;
  avgPages: number;
  vocabSize: number;
}

export const loadPDFDataset = async (zipFile: Blob): Promise<PdfDatasetSummary> => {
  const zip = new JSZip();
  await zip.loadAsync(zipFile);

  const classes: string[] = [];
  const pdfCounts: Record<string, number> = {};
  let totalPDFs = 0;
  let totalPages = 0;
  let totalTextLength = 0;
  let sampleCount = 0;

  // Ambil semua entry .pdf di dalam zip
  zip.forEach((relativePath, file) => {
    if (!file.dir && relativePath.endsWith('.pdf')) {
      const folderMatch = relativePath.match(/^([^\/]+)\//);
      if (folderMatch) {
        const className = folderMatch[1];
        if (!classes.includes(className)) {
          classes.push(className);
          pdfCounts[className] = 0;
        }
        pdfCounts[className]++;
        totalPDFs++;
      }
    }
  });

  // Proses parsing hanya tiap 20 file pertama (atau sesuai logika lama)
  const fileEntries = Object.entries(pdfCounts);
  const parsePromises: Promise<void>[] = [];

  let processed = 0;
  zip.forEach((relativePath, file) => {
    if (!file.dir && relativePath.endsWith('.pdf') && processed % 20 === 0) {
      const match = relativePath.match(/^([^\/]+)\//);
      if (match) {
        parsePromises.push(
          (async () => {
            const arrayBuf = await file.async('arraybuffer');
            const buffer = Buffer.from(arrayBuf);
            const data: PdfParseData = await pdfParse(buffer);

            totalTextLength += data.text.length;
            totalPages += data.numpages;
          })()
        );
      }
    }
    processed++;
  });

  await Promise.all(parsePromises);

  const avgTextLength = parsePromises.length
    ? totalTextLength / parsePromises.length
    : 0;
  const avgPages = parsePromises.length
    ? totalPages / parsePromises.length
    : 0;

  return {
    type: 'pdf',
    size: totalPDFs,
    classes: classes.length,
    classDistribution: pdfCounts,
    avgTextLength,
    avgPages,
    vocabSize: Math.round(avgTextLength * 0.1),
  };
};
