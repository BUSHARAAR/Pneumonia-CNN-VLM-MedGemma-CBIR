# Task 1 – Pneumonia Detection: Comparative Study (5 Models)

This report compares five architectures on PneumoniaMNIST using the same preprocessing, augmentation, and evaluation protocol.

## Models Compared

- Basic CNN (SimpleCNN)

- ResNet18

- EfficientNet-B0

- ViT-Tiny (patch16_224)

- MambaNet (SSM-inspired token mixer)

## Common Data Pipeline

- Dataset: PneumoniaMNIST official splits (train/val/test)

- Normalization: mean=0.5719, std=0.1684 computed from the train split

- Augmentation (train only): mild brightness/contrast, rotation (±7°), translation (≤2px), mild Gaussian noise

- No flips (preserves CXR laterality)

## Common Training Methodology

- Loss: CrossEntropyLoss

- Optimizer: AdamW (lr=0.0003, weight_decay=0.0001)

- Scheduler: CosineAnnealingLR (epochs=15)

- Batch size: 256

- Selection: best checkpoint by validation AUC

## Quantitative Results (Test)

### Summary Table

![](outputs_compare/comparison_table.png)

- Full CSV: `outputs_compare/comparison_metrics.csv`

### ROC Overlay

![](outputs_compare/overlay_roc.png)

### Metric Comparison Charts

![](outputs_compare/bar_auc.png)

![](outputs_compare/bar_f1.png)

![](outputs_compare/bar_accuracy.png)

## Per-model Detailed Outputs

### vit_tiny

- Metrics JSON: `outputs_compare/vit_tiny/metrics.json`

- Confusion matrix: ![](outputs_compare/vit_tiny/confusion_matrix.png)

- ROC curve: ![](outputs_compare/vit_tiny/roc_curve.png)

- Failure grid: ![](outputs_compare/vit_tiny/failure_grid.png)

- Failure images folder: `outputs_compare/vit_tiny/failures/`

### resnet18

- Metrics JSON: `outputs_compare/resnet18/metrics.json`

- Confusion matrix: ![](outputs_compare/resnet18/confusion_matrix.png)

- ROC curve: ![](outputs_compare/resnet18/roc_curve.png)

- Failure grid: ![](outputs_compare/resnet18/failure_grid.png)

- Failure images folder: `outputs_compare/resnet18/failures/`

### simplecnn

- Metrics JSON: `outputs_compare/simplecnn/metrics.json`

- Confusion matrix: ![](outputs_compare/simplecnn/confusion_matrix.png)

- ROC curve: ![](outputs_compare/simplecnn/roc_curve.png)

- Failure grid: ![](outputs_compare/simplecnn/failure_grid.png)

- Failure images folder: `outputs_compare/simplecnn/failures/`

### efficientnet_b0

- Metrics JSON: `outputs_compare/efficientnet_b0/metrics.json`

- Confusion matrix: ![](outputs_compare/efficientnet_b0/confusion_matrix.png)

- ROC curve: ![](outputs_compare/efficientnet_b0/roc_curve.png)

- Failure grid: ![](outputs_compare/efficientnet_b0/failure_grid.png)

- Failure images folder: `outputs_compare/efficientnet_b0/failures/`

### mambanet

- Metrics JSON: `outputs_compare/mambanet/metrics.json`

- Confusion matrix: ![](outputs_compare/mambanet/confusion_matrix.png)

- ROC curve: ![](outputs_compare/mambanet/roc_curve.png)

- Failure grid: ![](outputs_compare/mambanet/failure_grid.png)

- Failure images folder: `outputs_compare/mambanet/failures/`

## Failure Case Discussion (General)

- Many errors are due to the extremely low resolution (28×28), where subtle opacities are hard to distinguish.

- Borderline / mild pneumonia patterns can resemble normal radiographs after downsampling.

- A fixed threshold=0.5 may not be optimal; a sensitivity-focused threshold could reduce false negatives at the cost of more false positives.

- Some failures may reflect dataset ambiguity or weak signal-to-noise after normalization/augmentation.

## Strengths and Limitations

**Strengths**

- Fully reproducible pipeline with consistent preprocessing across all models.

- Complete metric suite + ROC/CM + failure cases for error analysis.

- Covers CNNs, transformer (ViT), and SSM-inspired Mamba-style architecture.


**Limitations**

- PneumoniaMNIST is downsampled; performance may not translate directly to clinical high-res CXR datasets.

- No probability calibration analysis (e.g., reliability curve).

- Threshold optimization not included by default (can be added).
