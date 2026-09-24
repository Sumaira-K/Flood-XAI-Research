# Flood-XAI: Explainable AI for Satellite-Based Flood Analysis

An exploratory Explainable AI (XAI) project for analyzing satellite imagery of flooded regions using **Sentinel-1 SAR data, Convolutional Neural Networks (CNNs), Grad-CAM, and SHAP**.

The project investigates not only whether a deep learning model can classify satellite images as **Flood** or **Non-Flood**, but also **why the model makes its predictions**.

--- 

## Overview

Flood detection from satellite imagery is an important application of remote sensing and artificial intelligence. However, a model prediction alone does not explain which regions of an image influenced that prediction.

This project explores an end-to-end workflow for:

**Satellite Imagery → CNN → Flood Prediction → Grad-CAM → SHAP → XAI Analysis**

The **Sen1Floods11** dataset is used as the primary data source. Sentinel-1 SAR imagery provides VV and VH polarization channels, while the associated hand-labeled masks provide pixel-level water annotations.

For the initial proof-of-concept, the pixel-level annotations are converted into image-level **Flood / Non-Flood** labels using an experimental water-coverage threshold.

---

## Research Objective

The primary objective is to explore how Explainable AI techniques can be applied to satellite-based flood classification.

The project focuses on three questions:

1. Can a CNN learn an initial flood classification task from Sentinel-1 SAR imagery?
2. Which spatial regions does the CNN use when making a prediction?
3. How do different XAI techniques, particularly **Grad-CAM and SHAP**, explain the same prediction?

---

## Dataset

### Sen1Floods11

The project uses the **Sen1Floods11** dataset, which contains Sentinel-1 satellite imagery and hand-labeled flood masks.

Each Sentinel-1 image contains two SAR channels:

* **VV** — Vertical transmit / Vertical receive
* **VH** — Vertical transmit / Horizontal receive

The hand-labeled masks contain:

| Value | Meaning          |
| ----- | ---------------- |
| `-1`  | NoData / invalid |
| `0`   | Not Water        |
| `1`   | Water            |

The dataset is fundamentally designed for **pixel-level flood segmentation**.

For this exploratory classification experiment, the masks were used to derive image-level labels.

---

## Project Workflow

```text
Sen1Floods11 Dataset
        │
        ▼
Sentinel-1 VV + VH
        │
        ▼
Data Preprocessing
        │
        ├── Resize to 128 × 128
        └── Per-channel standardization
        │
        ▼
CNN Flood Classifier
        │
        ▼
Flood / Non-Flood Prediction
        │
        ├──────────────────┐
        ▼                  ▼
    Grad-CAM              SHAP
        │                  │
        ▼                  ▼
Spatial Attention    Spatial Attribution
        │                  │
        └────────┬─────────┘
                 ▼
       Ground-Truth Comparison
                 │
                 ▼
           XAI Analysis
```

---

## Technical Implementation

### 1. Data Exploration

The notebook explores:

* Sentinel-1 SAR image structure
* VV and VH channels
* Ground-truth flood masks
* NoData regions
* Water coverage
* Dataset sample distribution

### 2. Data Preprocessing

Selected images are:

* Loaded from GeoTIFF files using `rasterio`
* Resized from `512 × 512` to `128 × 128`
* Converted to `float32`
* Standardized independently for VV and VH channels

The resulting input format is:

```text
(2, 128, 128)
```

where `2` represents the VV and VH channels.

### 3. Image-Level Classification

Because Sen1Floods11 provides pixel-level labels, an experimental image-level classification task was created.

The rule used was:

```text
Water coverage ≥ 5% → Flood
Water coverage < 5% → Non-Flood
```

This threshold is a project-specific preprocessing decision and is **not part of the original Sen1Floods11 annotation scheme**.

### 4. CNN Model

A lightweight CNN was implemented in PyTorch.

Architecture:

```text
Input: 2 × 128 × 128

Conv2D (2 → 16)
        ↓
ReLU
        ↓
Max Pooling

Conv2D (16 → 32)
        ↓
ReLU
        ↓
Max Pooling

Conv2D (32 → 64)
        ↓
ReLU
        ↓
Global Average Pooling
        ↓
Fully Connected Layer
        ↓
Flood / Non-Flood
```

The model was intentionally kept small because the experiment uses a limited exploratory subset.

---

## Model Training

The experimental dataset contains:

* **24 images**
* **19 training samples**
* **5 testing samples**

Class distribution:

| Class     | Samples |
| --------- | ------: |
| Non-Flood |      14 |
| Flood     |      10 |

Training configuration:

* Framework: **PyTorch**
* Optimizer: **Adam**
* Learning rate: `0.001`
* Loss: **Weighted Cross-Entropy Loss**
* Epochs: `30`
* Device: CPU

---

## Classification Results

On the five-image test set, the model produced:

| Metric    | Result |
| --------- | -----: |
| Accuracy  |   100% |
| Precision |   100% |
| Recall    |   100% |
| F1-score  |   100% |

### Important Note

These results should **not** be interpreted as representative flood-detection performance.

The test set contains only five images, so the results are included as a proof-of-concept demonstration of the modeling and XAI pipeline rather than as a benchmark.

---

# Explainable AI Analysis

## Grad-CAM

**Grad-CAM (Gradient-weighted Class Activation Mapping)** was used to investigate which spatial regions were associated with the CNN's prediction.

It uses:

* Activations from a convolutional layer
* Gradients of the predicted class
* Weighted feature maps

The resulting heatmap provides a spatial explanation of the model's prediction.

### Question addressed

> Where is the CNN focusing when making its prediction?

The Grad-CAM explanation was compared with the Sen1Floods11 ground-truth flood mask.

---

## SHAP

**SHAP (SHapley Additive exPlanations)** was used to estimate the contribution of spatial input regions to the model's prediction.

An image masker was used to progressively mask spatial regions and estimate their effect on the model output.

### Question addressed

> Which spatial regions contribute to the model's prediction?

The refined SHAP analysis was compared with the ground-truth flood annotation.

---

## Grad-CAM vs SHAP

For the selected test image:

```text
Sample: Ghana_319168
Actual Class: Flood
Predicted Class: Flood
```

Using the exploratory top-20% attribution criterion:

| XAI Method | Water Overlap |
| ---------- | ------------: |
| Grad-CAM   |         0.00% |
| SHAP       |        61.65% |

These values represent an **exploratory spatial overlap analysis for one test image**.

They should not be interpreted as standardized XAI performance scores or as evidence that one method is generally superior to the other.

The difference demonstrates that Grad-CAM and SHAP can provide different perspectives on the same model prediction.

---

## Visual Analysis

The notebook includes visual comparisons of:

* Sentinel-1 VV imagery
* Ground-truth flood masks
* Grad-CAM heatmaps
* SHAP attribution maps
* XAI overlays with ground-truth boundaries

These visualizations help investigate whether model explanations correspond to annotated flood regions.

---

## Project Structure

```text
Flood-XAI/
│
├── data/
│   └── sen1floods11/
│       ├── S1Hand/
│       └── LabelHand/
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── src/
│
├── results/
│
├── README.md
└── venv/
```

---

## Technologies Used

### Programming & Frameworks

* Python
* PyTorch
* NumPy
* scikit-learn
* Matplotlib

### Remote Sensing & Data Processing

* Rasterio
* GeoTIFF
* Sentinel-1 SAR
* Sen1Floods11

### Explainable AI

* Grad-CAM
* SHAP

### Development

* Jupyter Notebook
* VS Code
* Git / GitHub

---

## Key Technical Concepts

This project provided practical exposure to:

* Satellite image analysis
* Synthetic Aperture Radar (SAR)
* VV/VH polarization
* GeoTIFF processing
* Pixel-level segmentation masks
* Image classification
* CNN architecture
* PyTorch model training
* Model evaluation
* Gradient-based Explainable AI
* SHAP-based feature attribution
* Spatial attribution analysis
* Ground-truth comparison
* XAI limitations and interpretation

---

## Limitations

The current implementation is intentionally exploratory.

### Dataset Size

Only 24 selected samples were used for the initial experiment.

### Classification Formulation

The original Sen1Floods11 task is pixel-level segmentation. This project temporarily converts the annotations into image-level labels to demonstrate the CNN + XAI workflow.

### Model Size

The CNN is a lightweight proof-of-concept architecture rather than a state-of-the-art flood segmentation model.

### XAI Evaluation

Grad-CAM and SHAP overlap were evaluated on a selected test image using an experimental top-20% attribution threshold.

### SHAP Channel Interpretation

The current SHAP image masker treats spatial regions jointly across the input channels. Therefore, the current SHAP result should be interpreted as a **combined spatial attribution**, rather than independent VV and VH feature importance.

---

## Future Work

The current project establishes a foundation for a more rigorous satellite flood XAI study.

Potential extensions include:

### 1. Larger Dataset

Expand the experiment to a substantially larger portion of Sen1Floods11.

### 2. Pixel-Level Segmentation

Move from image classification to the original segmentation problem using architectures such as:

* U-Net
* U-Net variants
* Other segmentation networks

### 3. Multi-Image XAI Evaluation

Evaluate Grad-CAM and SHAP explanations across multiple flood events rather than a single test image.

### 4. Channel-Level Analysis

Investigate the individual contribution of:

* VV
* VH

using an appropriate channel-aware attribution methodology.

### 5. More Rigorous XAI Evaluation

Use multiple quantitative measures to evaluate the relationship between model explanations and ground-truth flood regions.

### 6. Geographic Generalization

Evaluate the model across different flood events and geographical regions.

---

## Research Takeaway

This project demonstrates a complete proof-of-concept pipeline for applying Explainable AI to satellite-based flood classification.

The key focus is not simply obtaining a prediction, but investigating:

> **What regions of satellite imagery contributed to the model's decision?**

By combining **Sentinel-1 SAR imagery, CNN-based classification, Grad-CAM, and SHAP**, the project provides a foundation for further research into interpretable deep learning for disaster and flood analysis.

---

## References

* Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1 Satellite Imagery.*
* Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
* Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.*

---

## Author

**Sumaira K.**

B.Tech — Computer Science Engineering / Artificial Intelligence & Data Science

This project was developed as an exploratory research implementation focused on **Explainable AI, satellite imagery, and disaster/flood analysis**.
