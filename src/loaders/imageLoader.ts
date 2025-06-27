import JSZip from 'jszip';

export const loadImageDataset = async (zipFile: Blob): Promise<any> => {
    const zip = new JSZip();
    await zip.loadAsync(zipFile);

    const classes: string[] = [];
    const imageCounts: Record<string, number> = {};
    let totalImages = 0;
    let totalResolution = 0;
    let sampleCount = 0;

    const folderRegex = /(.+)\//;

    const filePromises: Promise<void>[] = [];
    zip.forEach((relativePath, file) => {
        if (!file.dir && /\.(jpe?g|png|gif|webp)$/i.test(relativePath)) {
            const match = relativePath.match(folderRegex);
            if (match) {
                const className = match[1];
                if (!classes.includes(className)) {
                    classes.push(className);
                    imageCounts[className] = 0;
                }

                imageCounts[className]++;
                totalImages++;

                if (totalImages % 100 === 0) {
                    filePromises.push((async () => {
                        try {
                            const imgBlob = await file.async('blob');
                            const img = await createImageBitmap(imgBlob);
                            totalResolution += img.width * img.height;
                            sampleCount++;
                        } catch (e) {
                            console.error(`Error processing image ${relativePath}:`, e);
                        }
                    })());
                }
            }
        }
    });

    await Promise.all(filePromises);

    return {
        type: 'image',
        size: totalImages,
        classes: classes.length,
        classDistribution: imageCounts,
        avgResolution: sampleCount > 0 ? totalResolution / sampleCount : 0,
        channels: 3
    };
};
