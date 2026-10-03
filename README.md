# RetinAI — Diabetic Retinopathy Identification Using AI

> A reproducible deep-learning research project for **5-class diabetic retinopathy severity classification** from retinal fundus photographs.

RetinAI investigates automated diabetic retinopathy (DR) grading using deep learning, with an emphasis on **reproducible experimentation, class-imbalance handling, ordinal evaluation, and reliable model evaluation**.

The current research pipeline uses the **APTOS 2019 Blindness Detection dataset** and an **EfficientNet-B0** image-classification model implemented with PyTorch.

> [!WARNING]
> **Research prototype only.**
>
> RetinAI is not a medical diagnostic device and must not be used as a substitute for examination or diagnosis by a qualified healthcare professional.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [DR Severity Classes](#dr-severity-classes)
- [Dataset](#dataset)
- [Research Objectives](#research-objectives)
- [System Pipeline](#system-pipeline)
- [Model](#model)
- [Data Preprocessing](#data-preprocessing)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Training Strategy](#training-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Verified Experimental Results](#verified-experimental-results)
- [Research Findings](#research-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Dataset and Manifest Workflow](#dataset-and-manifest-workflow)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference and Web Integration](#inference-and-web-integration)
- [Reproducibility](#reproducibility)
- [Engineering and Code Quality](#engineering-and-code-quality)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Medical Disclaimer](#medical-disclaimer)
- [License](#license)

---

# Project Overview

Diabetic retinopathy is a diabetes-related retinal disease that can progress through multiple stages of severity. Automated analysis of retinal fundus photographs can potentially assist screening workflows by providing a consistent computer-vision-based assessment of disease severity.

RetinAI treats diabetic retinopathy as an **ordinal 5-class classification problem** rather than a simple binary classification task.

The model predicts one of five severity grades:

```text
0 → No DR
1 → Mild DR
2 → Moderate DR
3 → Severe DR
4 → Proliferative DR
```

The project focuses not only on model accuracy, but also on metrics that are particularly informative for an **imbalanced and ordinal medical classification problem**, including:

- Macro F1
- Balanced Accuracy
- Quadratic Weighted Kappa (QWK)
- Per-class Precision, Recall and F1
- Confusion Matrix
- Validation-based model selection

The research workflow was rebuilt to remove previously unsupported performance claims and ensure that reported results are tied to reproducible experiments and saved artifacts.

---

# Problem Statement

A major challenge in diabetic retinopathy classification is that the five disease grades are **highly imbalanced**.

The majority class is typically the no-DR category, while severe and proliferative cases are much less represented.

This creates several problems:

1. A model can achieve high overall accuracy while performing poorly on minority classes.
2. Severe cases are more important to distinguish than simple majority-class performance suggests.
3. DR grades are ordinal — confusing grade 3 with grade 4 is different from confusing grade 0 with grade 4.
4. Standard accuracy alone does not adequately describe model performance.

Therefore, RetinAI evaluates the model using multiple complementary metrics, with particular attention to **Macro F1, Balanced Accuracy, and QWK**.

---

# DR Severity Classes

The APTOS 2019 labels are mapped to five diabetic retinopathy severity grades:

| Label | Severity         | Description                                   |
| ----: | ---------------- | --------------------------------------------- |
|     0 | No DR            | No visible signs of diabetic retinopathy      |
|     1 | Mild             | Mild diabetic retinopathy                     |
|     2 | Moderate         | Moderate diabetic retinopathy                 |
|     3 | Severe           | Severe non-proliferative diabetic retinopathy |
|     4 | Proliferative DR | Proliferative diabetic retinopathy            |

The classes are **ordinal**:

```text
0 < 1 < 2 < 3 < 4
```

This ordering is important when interpreting errors and motivates the use of **Quadratic Weighted Kappa** and ordinal-aware loss experiments.

---

# Dataset

## APTOS 2019 Blindness Detection

The current reproducible research pipeline uses **APTOS 2019 only**.

The dataset contains retinal fundus photographs annotated with five diabetic retinopathy severity grades.

The cleaned project manifest contains:

| Split      |    Images |
| ---------- | --------: |
| Train      |     2,346 |
| Validation |       496 |
| Test       |       487 |
| **Total**  | **3,329** |

The dataset is represented through a frozen manifest so that training and evaluation operate on explicit image paths, labels and split assignments.

The cleaned manifest also underwent duplicate checking to ensure that the same image content does not appear across different dataset splits.

### Why a Manifest?

Instead of allowing every training script to independently discover files, RetinAI uses a manifest-driven workflow.

Conceptually:

```text
APTOS 2019
    │
    ▼
Dataset Verification
    │
    ▼
Manifest Construction
    │
    ▼
Quality / Consistency Audits
    │
    ▼
Fixed Train / Validation / Test Splits
    │
    ▼
Reproducible Experiments
```

This makes the dataset used by an experiment explicit and auditable.

---

# Research Objectives

The project investigates the following questions:

### 1. Baseline Performance

How well can EfficientNet-B0 classify the five DR severity levels using a standard weighted cross-entropy objective?

### 2. Class Imbalance

Can class-aware techniques improve performance on underrepresented severity grades?

### 3. Ordinal Learning

Can incorporating the ordered relationship between DR grades improve ordinal agreement measured by QWK?

### 4. Robust Evaluation

How does model performance change across:

- Accuracy
- Balanced Accuracy
- Macro F1
- QWK
- Per-class performance

rather than relying on accuracy alone?

### 5. Reproducibility

Can every reported experiment be traced to:

- a dataset manifest,
- a configuration,
- a training run,
- a checkpoint,
- and evaluation artifacts?

---

# System Pipeline

The current research workflow can be summarized as:

```text
              APTOS 2019 Dataset
                      │
                      ▼
              Dataset Verification
                      │
                      ▼
             Manifest Construction
                      │
                      ▼
             Quality / Data Audits
                      │
                      ▼
          Train / Validation / Test
                      │
                      ▼
              Image Preprocessing
                      │
                      ▼
             Data Augmentation
                      │
                      ▼
              EfficientNet-B0
                      │
                      ▼
             5-Class Predictions
                      │
                      ▼
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
   Accuracy       Macro F1          QWK
       │              │              │
       └──────────────┼──────────────┘
                      ▼
              Experiment Analysis
```

---

# Model

## EfficientNet-B0

The primary research architecture is **EfficientNet-B0**.

The model is used as a 5-class image classifier:

```text
Fundus Image
     │
     ▼
EfficientNet-B0
     │
     ▼
Feature Representation
     │
     ▼
Classification Head
     │
     ▼
5 Output Classes
     │
     ├── Class 0: No DR
     ├── Class 1: Mild
     ├── Class 2: Moderate
     ├── Class 3: Severe
     └── Class 4: Proliferative DR
```

The current baseline configuration uses:

- **Architecture:** EfficientNet-B0
- **Input resolution:** 384 × 384
- **Number of classes:** 5
- **Optimizer:** AdamW
- **Learning-rate scheduler:** CosineAnnealingLR
- **Batch size:** Experiment-dependent
- **Early stopping:** Validation QWK
- **Mixed precision:** Enabled when CUDA is available
- **Gradient clipping:** 1.0

---

# Data Preprocessing

The preprocessing pipeline includes retinal-image preparation before model training.

The project supports:

- Image resizing
- Retina-focused preprocessing
- Image normalization
- Training-time augmentation
- Optional quality filtering
- Class-aware augmentation experiments

Training and validation transformations are kept separate so that augmentation intended for training does not leak into evaluation.

---

# Handling Class Imbalance

Class imbalance is one of the central challenges of this project.

RetinAI experiments with several approaches.

## Weighted Cross-Entropy

Class weights are calculated from the training distribution and passed to the loss function.

This gives greater importance to underrepresented classes.

```text
Majority class
      ↓
Lower relative weight

Minority class
      ↓
Higher relative weight
```

## Focal Loss

Focal loss is available as an experimental alternative that places greater emphasis on difficult examples.

## Class-Aware Sampling

The project includes a class-aware sampler that modifies how training samples are selected.

## Balanced Batch Sampling

Balanced batch sampling can construct batches using controlled per-class quotas.

## Class-Aware Augmentation

The training pipeline also supports stronger augmentation for selected minority classes.

These methods are treated as **experimental strategies**, not as assumptions that automatically improve performance.

---

# Ordinal Loss

Because DR severity is ordered, the project also investigates an ordinal-aware objective.

The implemented `OrdinalDistanceLoss` combines classification learning with a penalty based on the distance between the expected predicted grade and the target grade.

Conceptually:

```text
Classification component
        +
Ordinal distance component
        ↓
Ordinal-aware training objective
```

This is intended to penalize predictions that are farther away from the correct severity level more strongly than nearby ordinal errors.

> **Important:** The implemented loss is not a differentiable QWK loss. QWK is used as an evaluation metric and model-selection metric, while the ordinal loss uses classification and expected-grade distance components.

---

# Training Strategy

The training pipeline supports configurable experiments through command-line arguments.

Important training controls include:

- Model architecture
- Image resolution
- Batch size
- Learning rate
- Weight decay
- Number of epochs
- Loss function
- Label smoothing
- Focal-loss gamma
- Ordinal distance weight
- Sampling strategy
- Augmentation strategy
- Random seed
- Early-stopping patience
- Gradient clipping

Example configuration:

```text
Model:             EfficientNet-B0
Image size:        384 × 384
Epochs:            25
Learning rate:     0.0002
Weight decay:      0.0001
Batch size:        32
Seed:              42
```

Actual experiment configurations are stored with their corresponding experiment artifacts.

---

# Evaluation Metrics

RetinAI uses multiple metrics because no single metric adequately represents performance on an imbalanced ordinal medical classification task.

| Metric            | Purpose                                                  |
| ----------------- | -------------------------------------------------------- |
| Accuracy          | Overall percentage of correct predictions                |
| Balanced Accuracy | Average recall across classes                            |
| Macro F1          | Treats every class equally when calculating F1           |
| Precision         | Measures how many predicted positives are correct        |
| Recall            | Measures how many actual positives are detected          |
| Per-class F1      | Examines performance for individual DR grades            |
| QWK               | Measures agreement while accounting for ordinal distance |
| Confusion Matrix  | Shows class-to-class prediction errors                   |

## Why QWK?

Quadratic Weighted Kappa is particularly useful for DR grading because the labels have an inherent order.

For example:

```text
Actual:    2
Predicted: 3
```

is a smaller ordinal error than:

```text
Actual:    2
Predicted: 0
```

QWK accounts for this ordering when measuring agreement.

---

# Verified Experimental Results

The repository previously contained performance claims that were not backed by reproducible checkpoints.

Those claims have been **removed/retracted**.

The current README reports only results associated with the reconstructed experiment workflow.

## EXP-000 — Weighted Cross-Entropy Baseline

| Metric            | Test Result |
| ----------------- | ----------: |
| Accuracy          |  **0.7803** |
| Balanced Accuracy |  **0.6180** |
| Macro F1          |  **0.6100** |
| QWK               |  **0.8769** |

**Configuration:**

- EfficientNet-B0
- 384 × 384 input
- Weighted Cross-Entropy
- Reproducible manifest
- Standard training pipeline

This experiment serves as the baseline against which subsequent experiments are compared.

---

## EXP-001R — Ordinal-Aware Loss

The ordinal-loss experiment improved the main evaluation metrics relative to the baseline.

| Metric            | Test Result |
| ----------------- | ----------: |
| Accuracy          |  **0.8049** |
| Balanced Accuracy |  **0.6474** |
| Macro F1          |  **0.6443** |
| QWK               |  **0.8959** |

Compared with EXP-000:

| Metric            | EXP-000 |   EXP-001R |
| ----------------- | ------: | ---------: |
| Accuracy          |  0.7803 | **0.8049** |
| Balanced Accuracy |  0.6180 | **0.6474** |
| Macro F1          |  0.6100 | **0.6443** |
| QWK               |  0.8769 | **0.8959** |

The improvement is particularly relevant because both Macro F1 and Balanced Accuracy increased, rather than the gain being limited to overall accuracy.

---

## Other Experiments

Additional experiments investigated targeted augmentation and class-aware sampling.

### Targeted Augmentation

A targeted augmentation experiment was evaluated during the research process but **was not retained as the preferred improvement** because it did not provide sufficient validation evidence for adoption.

### Class-Aware Sampling

A class-aware sampling experiment produced:

| Metric   | Validation Result |
| -------- | ----------------: |
| QWK      |            0.8843 |
| Macro F1 |            0.6420 |

This experiment was not used to make an unsupported claim about test-set improvement.

---

# Research Findings

The verified experiments suggest several useful observations.

### 1. Accuracy alone is insufficient

The baseline accuracy of 0.7803 does not fully describe class-balanced performance.

Balanced Accuracy and Macro F1 reveal substantially more difficulty across minority classes.

### 2. Ordinal learning helped the evaluated baseline

EXP-001R improved:

- Accuracy
- Balanced Accuracy
- Macro F1
- QWK

relative to EXP-000.

This suggests that incorporating ordinal distance into the training objective was beneficial for the tested configuration.

### 3. Minority classes remain challenging

The model performs substantially better on the majority/no-DR category than on severe minority categories.

This remains an important limitation and research direction.

### 4. QWK is useful for this task

Because DR grades are ordered, QWK provides information that ordinary accuracy cannot capture.

---

# Repository Structure

```text
RetinAI-DR/
│
├── api/
│   └──                    # FastAPI backend
│
├── app/
│   └──                    # Streamlit application
│
├── configs/
│   └──                    # Experiment/configuration files
│
├── data/
│   └──                    # Dataset manifests and local dataset assets
│
├── experiments/
│   └──                    # Experiment configurations and artifacts
│
├── scripts/
│   ├── audit_manifest_quality.py
│   ├── dedup_manifest.py
│   ├── evaluate_experiment.py
│   ├── evaluate_manifest_checkpoint.py
│   ├── evaluate_manifest_ensemble.py
│   ├── train_experiment.py
│   ├── train_manifest.py
│   └── verify_dataset.py
│
├── src/
│   └── dr_detection/
│       ├── dataset.py
│       ├── infer.py
│       ├── loss.py
│       ├── metrics.py
│       ├── models.py
│       ├── quality.py
│       ├── sampler.py
│       ├── train.py
│       └── transforms.py
│
├── tests/
│
├── notebooks/
│   └── colab_experiments_and_training.ipynb
│
├── pyproject.toml
├── pyrightconfig.json
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Yash00109/RetinAI-Diabetic-Retinopathy-Identification-using-AI.git
cd RetinAI-DR
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install the project

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The project dependencies include:

- PyTorch
- Torchvision
- NumPy
- Pandas
- scikit-learn
- Pillow
- OpenCV
- Matplotlib
- Seaborn
- tqdm

---

# Dataset and Manifest Workflow

The repository uses manifest-driven dataset handling.

A typical workflow is:

```text
APTOS 2019
    ↓
Dataset verification
    ↓
Manifest generation
    ↓
Manifest audit
    ↓
Duplicate checking
    ↓
Fixed train/val/test splits
    ↓
Experiment training
```

The manifest records information required to reproduce the experiment dataset.

The project also includes tools for:

- Dataset verification
- Manifest auditing
- Duplicate detection
- Quality analysis
- Split validation

---

# Training

The project supports both experiment-oriented and manifest-oriented training scripts.

Example:

```powershell
python scripts/train_manifest.py `
    --manifest data/manifests/aptos_fixed_manifest.csv `
    --model efficientnet_b0 `
    --image-size 384 `
    --epochs 25 `
    --batch-size 32 `
    --learning-rate 0.0002 `
    --weight-decay 0.0001 `
    --seed 42
```

For configurable experiments:

```powershell
python scripts/train_experiment.py `
    --manifest data/manifests/aptos_fixed_manifest.csv `
    --experiment-id EXP-000 `
    --model efficientnet_b0 `
    --image-size 384 `
    --epochs 25 `
    --batch-size 32 `
    --learning-rate 0.0002 `
    --weight-decay 0.0001 `
    --loss weighted_ce `
    --seed 42
```

Use `--help` on the respective script for the complete set of available options.

---

# Evaluation

The project provides dedicated evaluation scripts for trained checkpoints.

Typical evaluation outputs include:

- Accuracy
- Balanced Accuracy
- Macro F1
- Per-class metrics
- QWK
- Confusion matrix
- Prediction files
- Experiment metrics

Example:

```powershell
python scripts/evaluate_experiment.py `
    --experiment-dir experiments/EXP-000 `
    --manifest data/manifests/aptos_fixed_manifest.csv
```

Checkpoint-level evaluation is also supported through:

```powershell
python scripts/evaluate_manifest_checkpoint.py --help
```

Ensemble evaluation is available through:

```powershell
python scripts/evaluate_manifest_ensemble.py --help
```

Ensemble results should only be reported when the underlying checkpoints and evaluation protocol are reproducible and verified.

---

# Inference and Web Integration

The trained model can be integrated into an application through the project's inference and API components.

The repository includes:

- Model inference utilities
- FastAPI integration
- Streamlit application support
- Image-quality checks
- Prediction processing

The general inference flow is:

```text
Input Fundus Image
       ↓
Image Loading
       ↓
Quality / Preprocessing Checks
       ↓
Model Inference
       ↓
5-Class Probability Distribution
       ↓
Predicted DR Grade
```

The web layer is intended for **research demonstrations and experimentation**, not clinical deployment.

---

# Reproducibility

Reproducibility is a major design goal of the project.

Each experiment should preserve enough information to reproduce and audit the run, including:

- Experiment ID
- Dataset manifest
- Model architecture
- Input resolution
- Loss function
- Optimizer settings
- Learning rate
- Weight decay
- Random seed
- Sampling strategy
- Augmentation configuration
- Training history
- Best checkpoint
- Evaluation metrics
- Predictions

Conceptually:

```text
Experiment
    │
    ├── Configuration
    ├── Dataset Manifest
    ├── Training History
    ├── Best Checkpoint
    ├── Validation Metrics
    ├── Test Metrics
    └── Predictions
```

This allows experimental claims to be traced back to concrete artifacts instead of manually entered numbers.

---

# Engineering and Code Quality

The repository underwent a dedicated code-quality and static-analysis cleanup.

## Pyrefly

The codebase was checked using **Pyrefly**.

Final state:

```text
INFO 0 errors
```

The cleanup addressed genuine type-checking issues rather than suppressing them indiscriminately.

### Dataset typing

Dataset classes were explicitly integrated with PyTorch's dataset interface:

```python
class AptosDataset(torch.utils.data.Dataset):
```

and:

```python
class ManifestImageDataset(torch.utils.data.Dataset):
```

Pandas-derived values are explicitly converted where required by the type checker.

### Ensemble evaluation safety

The ensemble evaluation pipeline now explicitly verifies that:

- validation labels exist,
- test labels exist,
- validation probability accumulators exist,
- test probability accumulators exist.

This prevents potentially unsafe `None` values from reaching averaging and prediction operations.

### Redundant assignments removed

Redundant model configuration assignments such as:

```python
model.num_classes = 5
```

were removed where the class count was already supplied during model creation.

### Unnecessary suppression removed

A previously unnecessary Pyrefly suppression in the training pipeline was removed after verifying that the underlying code passed static analysis without it.

Legitimate suppressions elsewhere in the repository were retained where they correspond to genuine third-party or dynamic-typing limitations.

### Configuration improvements

`pyrightconfig.json` was updated so project scripts are included in import resolution and notebooks are excluded from static source checking.

`seaborn` was also added explicitly to the project dependencies because it is used by the evaluation tooling.

---

# Testing and Validation

Before committing research changes, the repository should be checked with:

```powershell
pyrefly check
```

Expected result:

```text
INFO 0 errors
```

Git whitespace validation:

```powershell
git diff --check
```

The notebook may produce an LF/CRLF line-ending warning on Windows; this is separate from an actual whitespace error.

---

# Limitations

RetinAI has several important limitations.

### Dataset limitation

The current research results are based on **APTOS 2019 only**.

Performance on other datasets and clinical populations cannot be inferred from these experiments.

### Class imbalance

Severe and proliferative DR classes contain substantially fewer examples than the majority class.

This contributes to lower minority-class performance.

### External validation

The verified results presented in this README are not evidence of clinical generalization.

Independent external validation is required before making claims about performance on other populations or clinical environments.

### Dataset shift

Differences in:

- camera hardware,
- image quality,
- patient population,
- acquisition protocols,
- disease prevalence,

may affect model performance.

### Clinical use

The model has not been clinically validated and must not be used to make medical decisions.

---

# Future Work

Potential future research directions include:

- Additional minority-class data
- Better class-balancing strategies
- More systematic augmentation studies
- Additional ordinal-learning approaches
- EfficientNet-B variants and alternative architectures
- External validation
- Calibration analysis
- Robustness testing
- Explainability analysis such as Grad-CAM
- Test-time augmentation
- Carefully validated checkpoint ensembles
- Prospective clinical evaluation

Any future performance improvement should be reported only when supported by reproducible experiments.

---

# Research Integrity

An important principle of this repository is:

> **No performance number should be reported unless it can be traced to a reproducible experiment.**

Earlier versions of the project contained unsupported performance claims. Those claims were removed during the repository audit.

The current README therefore distinguishes between:

- verified experimental results,
- ongoing experiments,
- research hypotheses,
- and future work.

This is intentional and is part of the project's reproducibility standard.

---

# Medical Disclaimer

> **RetinAI is a research and educational prototype.**
>
> It is not a medical device, diagnostic system, or substitute for professional medical evaluation.
>
> Model predictions are probabilistic outputs from a machine-learning model and may be incorrect.
>
> The system must not be used to diagnose, treat, triage, or make clinical decisions about a patient.
>
> Any real-world clinical application would require appropriate clinical validation, regulatory review, safety evaluation, and oversight by qualified healthcare professionals.

---

# License

This project is intended for research and educational purposes.

See the repository's `LICENSE` file for the applicable license terms.

---

# Acknowledgements

This project uses the **APTOS 2019 Blindness Detection** dataset for diabetic retinopathy research.

The project also builds upon open-source Python and deep-learning technologies including:

- PyTorch
- Torchvision
- NumPy
- Pandas
- scikit-learn
- OpenCV
- Pillow
- Matplotlib
- Seaborn

---

## RetinAI

**Diabetic Retinopathy Identification Using Artificial Intelligence**

A reproducible research pipeline for 5-class diabetic retinopathy severity classification from retinal fundus photographs.
