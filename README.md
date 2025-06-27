# WebNN Model Advisor - Intelligent Neural Network Architecture Recommendation

The **TVM WebNN Model Advisor** is an advanced AI-powered module that provides intelligent neural network architecture recommendations based on dataset characteristics. Leveraging mathematical formulations from cutting-edge research, this module analyzes your dataset (CSV, images, or PDFs) and recommends optimal model architectures, hyperparameters, and training configurations.

## Features

- **Intelligent Model Recommendation**: Automatically suggests suitable neural network architectures
- **Dataset Analysis**: Supports CSV, image datasets (ZIP), and PDF documents (ZIP)
- **Hyperparameter Optimization**: Calculates optimal epochs, learning rate, and batch size
- **Research-Backed**: Based on formulations from top-tier publications (Q1 journals)
- **WebNN Compatible**: Generates architectures compatible with WebNN API
- **Easy Integration**: Simple API for browser and Node.js environments

## Installation

```bash
npm install tvm
```

Or for GitHub Package:

```bash
npm install @4211421036/tvm
```

## Mathematical Foundations

### 1. Dataset Complexity Metric

The complexity metric (C) is calculated differently for each data type:

**Image Data**:

$$C_{image}$$ = $$N_{classes}$$ × $$R_{avg}$$ × D

Where:
- $$N_{classes}$$ = Number of classes
- $$R_{avg}$$ = Average resolution (width × height)
- D = Number of channels

**Tabular Data**:

$$C_{tabular}$$ = H × F

Where:
- H = Shannon entropy of class distribution
- F = Number of features

**Text Data**:

$$C_{text}$$ = $$L_{avg}$$ × V

Where:
- $$L_{avg}$$ = Average text length
- V = Vocabulary size

### 2. Epoch Calculation

Based on Prechelt's Early Stopping Principle:

$$epochs = \min(500, \max(20, 50 + 150 × \ln{(C)} / \ln{(N)}))$$

Where:
- C = Dataset complexity
- N = Number of samples

### 3. Learning Rate Optimization

Adaptive learning rate using entropy-based decay (Smith, 2017):

$$\alpha = 0.1 \times \exp{(-1.5 \times H)}$$

Where H is the Shannon entropy of class distribution.

## API Reference

### `loadDataset(file: File): Promise<DatasetMetadata>`

Loads and analyzes dataset metadata.

**Parameters**:
- `file`: Input file (CSV, ZIP of images, or ZIP of PDFs)

**Returns**:
```typescript
{
  type: 'image' | 'tabular' | 'text';
  size: number;
  classes?: number;
  features?: number;
  classDistribution?: Record<string, number>;
  avgResolution?: number;
  channels?: number;
  avgTextLength?: number;
  avgPages?: number;
  vocabSize?: number;
}
```

### `analyzeDataset(metadata: DatasetMetadata): AnalysisResult`

Computes dataset complexity metrics.

**Returns**:
```typescript
{
  complexity: number;
  entropy?: number;
  dataType: string;
  recommendationKey: string;
}
```

### `recommendModel(analysis: AnalysisResult): ModelRecommendation`

Generates model recommendation with hyperparameters.

**Returns**:
```typescript
{
  model: string;
  layers: Layer[];
  paper: string;
  hyperparameters: {
    epochs: number;
    learningRate: number;
    batchSize: number;
    validationSplit: number;
    earlyStopping: boolean;
  };
  explanation: string;
}
```

## Usage Example

```javascript
import { loadDataset, analyzeDataset, recommendModel } from 'webnn-advisor';

async function processDataset(file) {
  try {
    // Load and analyze dataset
    const metadata = await loadDataset(file);
    const analysis = analyzeDataset(metadata);
    
    // Get model recommendation
    const recommendation = recommendModel(analysis);
    
    console.log('Recommended Model:', recommendation.model);
    console.log('Hyperparameters:', recommendation.hyperparameters);
    console.log('Architecture:');
    recommendation.layers.forEach(layer => {
      console.log(`- ${layer.type}: ${JSON.stringify(layer)}`);
    });
    
    return recommendation;
  } catch (error) {
    console.error('Dataset processing error:', error);
  }
}

// Browser file input handling
document.getElementById('datasetInput').addEventListener('change', async (e) => {
  const recommendation = await processDataset(e.target.files[0]);
  // Visualize recommendation in UI
});
```

## Real-World Application

```html
<!-- index.html -->
<!DOCTYPE html>
<html>
<head>
  <title>WebNN Model Advisor</title>
  <script type="module">
    import { loadDataset, analyzeDataset, recommendModel } from './node_modules/webnn-advisor/dist/index.js';
    
    window.processDataset = async (file) => {
      const metadata = await loadDataset(file);
      const analysis = analyzeDataset(metadata);
      return recommendModel(analysis);
    };
  </script>
</head>
<body>
  <input type="file" onchange="processDataset(this.files[0]).then(console.log)">
</body>
</html>
```

## Research References

1. **MobileNetV2: Inverted Residuals and Linear Bottlenecks**  
   Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). CVPR.  
   [DOI: 10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474)

2. **Early Stopping - But When?**  
   Prechelt, L. (1998). Neural Networks: Tricks of the Trade.  
   [DOI: 10.1007/3-540-49430-8_3](https://doi.org/10.1007/3-540-49430-8_3)

3. **A Bayesian Perspective on Generalization and Stochastic Gradient Descent**  
   Smith, L. N., & Topin, N. (2017). ICLR.  
   [arXiv:1710.06451](https://arxiv.org/abs/1710.06451)

## Development Workflow

1. Install dependencies:
```bash
npm ci
```

2. Build project:
```bash
npm run build
```

3. Run tests:
```bash
npm test
```

4. Start development server:
```bash
npm run dev
```

## Contribution Guidelines

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a pull request

## License

[MIT License](LICENSE)
