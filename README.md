# Confident Naturalness Explanation (CNE) Framework

This repository implements the Confident Naturalness Explanation (CNE) framework, a novel approach for assessing and explaining patterns that form naturalness in protected areas through the analysis of satellite imagery. The framework combines explainable AI and uncertainty quantification to provide a quantitative metric that describes the confident contribution of specific land cover patterns to the concept of naturalness.

## Overview

Protected natural areas characterized by minimal modern human footprint are often challenging to assess. The CNE framework addresses this by:

1. Extracting land cover patterns from satellite imagery using semantic segmentation
2. Quantifying the importance of each pattern through surrogate modeling
3. Estimating uncertainty using Monte Carlo dropout
4. Combining importance and certainty into a single CNE metric

![CNE Framework Overview](https://i.imgur.com/placeholder.png)

## Key Features

- **Gray-Box Approach**: Combines a black-box semantic segmentation model with a transparent white-box surrogate model
- **Uncertainty Quantification**: Uses Monte Carlo dropout to estimate epistemic uncertainty
- **CNE Metric**: Provides a quantifiable value (0-1) for each pattern's confident contribution to naturalness
- **Uncertainty-Aware Segmentation**: Generates maps showing uncertainty levels at pixel level

## Framework Components

### 1. Explainability Component

The explainability part consists of:

- **Black-Box Model**: DeepLabV3 semantic segmentation model trained on CORINE land cover classes
- **Vectorization**: Converting segmentation masks to pattern distribution vectors
- **White-Box Model**: Logistic regression to identify feature importance for naturalness classification

### 2. Uncertainty Quantification Component

- **Monte Carlo Dropout**: Enabling dropout at inference time to sample from multiple model versions
- **Standard Deviation**: Calculating variability across Monte Carlo samples to measure uncertainty
- **Spatial Aggregation**: Summing uncertainty values across spatial dimensions

### 3. CNE Metric Calculation

The CNE metric combines importance coefficients (α) from the explainability component with uncertainty values (u) from the uncertainty quantification:

```
CNE_c = α_c+ / u_c
```

Where:
- `α_c+` = max(α_c, 0) - only positive coefficients are considered (patterns that positively contribute to naturalness)
- `u_c` = sum of standard deviation across spatial dimensions for each pattern

## Dataset

The framework is demonstrated using:

- **AnthroProtect Dataset**: 24,000 multispectral Sentinel-2 images of the Fennoscandia region, labeled as either protected or anthropogenic areas
- **CORINE Land Cover**: Well-understood land cover classification with 44 classes

## Results

The CNE metric reveals patterns that confidently contribute to naturalness:

| Pattern | CNE Metric | Distribution% |
|---------|------------|--------------|
| Moors and heathland | 1.00 | 13.2 |
| Peat bogs | 0.81 | 6.1 |
| Bare rock | 0.65 | 6.8 |
| Broad-leaved forest | 0.61 | 13.4 |
| Sparsely vegetated areas | 0.49 | 24.1 |
| Coniferous forest | 0.44 | 18.2 |
| Watercourses | 0.23 | 0.2 |
| Glaciers and perpetual snow | 0.21 | 1.1 |
| Natural grassland | 0.19 | 0.05 |
| Water bodies | 0.18 | 5.2 |

## Requirements

```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
pandas>=1.1.0
tifffile>=2020.9.3
scikit-learn>=0.23.0
tqdm>=4.50.0
plotly>=4.14.0
matplotlib>=3.3.0
```

## Usage

The main script `naturalness_assessment.py` provides a complete implementation of the CNE framework:

```python
# Example configuration
config = {
    # Data settings
    'csv_file': 'infos.csv',
    'image_folder': 'data/anthroprotect/tiles/s2',
    'target_folder': 'data/anthroprotect/new_masks',
    'train_ratio': 0.8,
    'batch_size': 4,
    
    # Model settings
    'model_type': 'deeplabv3',
    'input_channels': 3,
    'output_classes': 43,  # CORINE land cover classes
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'checkpoint_path': 'checkpoint/model_final.pth',
    
    # MC Dropout settings
    'num_samples': 25,  # Number of Monte Carlo samples
    
    # Results settings
    'results_path': 'results/cne_assessment',
}

# Run the pipeline
from naturalness_assessment import end_to_end_pipeline
results = end_to_end_pipeline(config)
```

## Pipeline Steps

1. **Segmentation Model Training/Loading**:
   - Load a pre-trained DeepLabV3 model or train a new one on CORINE land cover data

2. **Uncertainty Quantification**:
   - Generate multiple predictions using MC dropout
   - Calculate standard deviation across predictions
   - Create uncertainty-aware segmentation masks

3. **Explainability through Surrogate Modeling**:
   - Extract pattern distribution vectors from segmentation outputs
   - Train a logistic regression model to classify naturalness
   - Extract importance coefficients for each pattern

4. **CNE Metric Calculation**:
   - Combine importance coefficients and uncertainty values
   - Normalize to 0-1 scale
   - Rank patterns by their confident contribution to naturalness

## Visualization

The framework generates:
- Predicted segmentation masks
- Uncertainty-aware segmentation maps highlighting areas of low confidence
- CNE metric values for each pattern
- Feature importance visualizations

## Advantages over Previous Approaches

The CNE framework addresses limitations of previous naturalness assessment methods:

1. **Objectivity**: Uses data-driven coefficients instead of hand-crafted weights
2. **Validity**: Encompasses all distinctive patterns in the assessment
3. **Uncertainty Awareness**: Accounts for model confidence in the evaluation
4. **Quantitative Assessment**: Provides a clear metric for pattern importance

## Citation

If you use this code in your research, please cite:

```
@article{emam2024cne,
  title={Confident Naturalness Explanation (CNE): A Framework to Explain and Assess Patterns Forming Naturalness},
  author={Emam, Ahmed and Farag, Mohamed and Roscher, Ribana},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={21},
  year={2024},
  pages={8500505},
  doi={10.1109/LGRS.2024.3365196}
}
```

## License

[MIT License](LICENSE)

## Acknowledgments

This work was supported in part by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under various grants including RO 4839/5-1, SCHM 3322/4-1, and the DFG's Excellence Strategy.
