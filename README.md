# pneumonia-task1-CNN_comparative-study
Comparative CNN–Transformer–Mamba study for pneumonia detection (Task-1)

# Pneumonia Detection – Task 1 (Comparative Study)

This repository contains a complete and reproducible solution for **Task 1: CNN Classification with Comprehensive Analysis**.

The work performs a **fair comparative evaluation** of five architectures on the PneumoniaMNIST dataset:

- Basic CNN (SimpleCNN)
- ResNet-18
- EfficientNet-B0
- Vision Transformer (ViT-Tiny)
- MambaNet (state-space inspired model)

## Dataset
- **PneumoniaMNIST** (MedMNIST v2)
- Official train/validation/test splits
- Dataset is automatically downloaded via `medmnist`

## Data Pipeline
- Train-split mean/std normalization
- Medical-aware augmentation:
  - Brightness/contrast adjustment
  - Mild rotation (±7°)
  - Small translation (≤2 px)
  - Mild Gaussian noise
- No flips (preserves CXR laterality)

## Training
- Loss: CrossEntropyLoss
- Optimizer: AdamW
- LR Scheduler: CosineAnnealingLR
- Model selection: Best validation AUC
- Fully reproducible (fixed seed)

## Evaluation
Metrics reported on held-out test set:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
  

Additional analysis:
- Confusion matrix
- ROC curve (per model + overlay)
- Failure case visualization

## Per-Model Evaluation Plots

### EfficientNet-B0
- Confusion Matrix  
  ![](outputs_compare/efficientnet_b0/confusion_matrix.png)
- ROC Curve  
  ![](outputs_compare/efficientnet_b0/roc_curve.png)
- Failure Cases  
  ![](outputs_compare/efficientnet_b0/failure_grid.png)

### ResNet-18
- ![](outputs_compare/resnet18/confusion_matrix.png)
- ![](outputs_compare/resnet18/roc_curve.png)
- ![](outputs_compare/resnet18/failure_grid.png)

### Vision Transformer (ViT-Tiny)
- ![](outputs_compare/vit_tiny/confusion_matrix.png)
- ![](outputs_compare/vit_tiny/roc_curve.png)
- ![](outputs_compare/vit_tiny/failure_grid.png)

### MambaNet
- ![](outputs_compare/mambanet/confusion_matrix.png)
- ![](outputs_compare/mambanet/roc_curve.png)
- ![](outputs_compare/mambanet/failure_grid.png)


## How to Run

### Install dependencies
```bash

pip install -r requirements.txt

#Train and evaluate all models
python pipeline.py \
  --models simplecnn,resnet18,efficientnet_b0,vit_tiny,mambanet \
  --epochs 15 \
  --batch 256

#Outputs
#All results are saved to:
outputs_compare/

#Report: A full technical report is available in:
task1_comparative_report.md

