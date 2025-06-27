export const loadImageDataset = async (zipFile: Blob) => {
  const JSZip = (await import('jszip')).default;
  const zip = new JSZip();
  await zip.loadAsync(zipFile);

  const classes: string[] = [];
  const imageCounts: Record<string, number> = {};
  let totalImages = 0;
  let totalResolution = 0;

  for (const [path, file] of Object.entries(zip.files)) {
    if (!file.dir && /\.(jpe?g|png|gif)$/i.test(path)) {
      const pathParts = path.split('/');
      if (pathParts.length > 1) {
        const className = pathParts[pathParts.length - 2];
        
        if (!classes.includes(className)) {
          classes.push(className);
          imageCounts[className] = 0;
        }

        imageCounts[className]++;
        totalImages++;
        
        if (totalImages % 100 === 0) {
          const img = await createImageBitmap(await file.async('blob'));
          totalResolution += img.width * img.height;
        }
      }
    }
  }

  return {
    type: 'image',
    size: totalImages,
    classes: classes.length,
    classDistribution: imageCounts,
    avgResolution: totalResolution / Math.floor(totalImages / 100),
    channels: 3
  };
};
