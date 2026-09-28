# RetinAI — Diabetic Retinopathy Detection using Artificial Intelligence

> **Research prototype** for automated diabetic retinopathy severity classification from retinal fundus photographs using deep learning.

EfficientNet-based classification pipeline with fundus image quality assurance, multi-dataset manifest training, Grad-CAM visualization, FastAPI backend, and interactive evaluation interfaces.

> [!WARNING]
> **This project is a research prototype.** It is NOT a clinical diagnostic tool and must NOT be used for independent medical diagnosis. All predictions require review by a qualified ophthalmologist.

## Problem Statement

Diabetic retinopathy (DR) is a leading cause of preventable blindness, affecting approximately one-third of people with diabetes. Early detection through regular retinal screening can prevent up to 95% of severe vision loss. This project aims to assist — not replace — clinical screening by automating the classification of fundus images into five standard DR severity grades.

## DR Severity Grades

| Label | Grade            | Description                                                                         |
| ----: | ---------------- | ----------------------------------------------------------------------------------- |
|     0 | No DR            | Healthy retina, no visible microvascular abnormalities                              |
|     1 | Mild             | Microaneurysms only                                                                 |
|     2 | Moderate         | More than microaneurysms but less than severe                                       |
|     3 | Severe           | >20 intraretinal hemorrhages in 4 quadrants, or venous beading in 2+, or IRMA in 1+ |
|     4 | Proliferative DR | Neovascularization, vitreous/preretinal hemorrhage                                  |

## Dataset

- **APTOS 2019 Blindness Detection** (via HuggingFace)
- **Total images**: 3,662
- **Train / Validation / Test split**: 2,562 / 550 / 550 (stratified)

### Class Distribution

| Class | Label            | Count | Percentage |
| :---: | :--------------- | ----: | ---------: |
|   0   | No DR            | 1,805 |     49.3%  |
|   1   | Mild             |   370 |     10.1%  |
|   2   | Moderate         |   999 |     27.3%  |
|   3   | Severe           |   193 |      5.3%  |
|   4   | Proliferative DR |   295 |      8.1%  |

> Severe class imbalance (10:1 ratio between classes 0 and 3) is a key challenge addressed through weighted loss, balanced sampling, and class-aware augmentation.

## Current Status

> [!IMPORTANT]
> **The reproducible research baseline is under active reconstruction.**
> Previously reported metrics (92.40% accuracy, 0.928 QWK) were **not backed by reproducible checkpoints** and have been retracted. The current pipeline is being rebuilt from scratch with full experiment tracking and reproducibility.

### Verified Experiments

| Experiment | Model | Resolution | Loss | Status | Notes |
|:-----------|:------|:-----------|:-----|:-------|:------|
| EXP-000 | EfficientNet-B0 | 384 | Weighted CE | 🔄 Pending | Baseline reproduction |

Results will be added here as experiments complete with genuine checkpoints and metrics.

## Highlights

- **EfficientNet Architecture**: Deep CNN classification pipeline tailored for retinal fundus images.
- **Fundus Quality Gate**: Automated pre-inference QA screening out low-contrast, blur, invalid crops, missing retinas, off-center frames, and flipped orientations.
- **FastAPI REST Service (`api/`)**: Backend for serving predictions with quality checks and Grad-CAM.
- **Comprehensive Evaluation Suite**: QWK, Macro F1, Balanced Accuracy, per-class metrics, referable DR sensitivity/specificity, calibration, and robustness testing.
- **Multi-Dataset Manifest Builder**: Harmonizes APTOS 2019, DeepDRiD, and DDR datasets.
- **Grad-CAM Explanations**: Visual saliency heatmaps highlighting retinal features influencing predictions.
- **Streamlit Demo**: Standalone web UI for rapid local testing and demonstrations.

---

## Methodology

### Preprocessing
1. **Retina circular crop**: Removes irrelevant black background
2. **Aspect-ratio preserving square padding**: Prevents distortion
3. **Ben Graham contrast enhancement**: Equalizes camera flash variations

### Training Strategy
- **Optimizer**: AdamW with weight decay
- **Scheduler**: CosineAnnealingLR
- **Early stopping**: Based on validation QWK
- **Mixed precision**: AMP when CUDA available
- **Gradient clipping**: max_norm=1.0

### Class Imbalance Strategy
- **Weighted CrossEntropy**: Inverse-frequency class weights from training set only
- **Focal Loss**: Configurable gamma for minority-class focus
- **Balanced Batch Sampler**: Equal per-class quota per mini-batch
- **Class-aware augmentation**: Stronger augmentation for minority classes

### Evaluation Metrics
| Metric | Purpose |
|:-------|:--------|
| Accuracy | Overall agreement |
| Balanced Accuracy | Mean per-class recall |
| Macro F1 | Class-balanced harmonic mean |
| Weighted F1 | Frequency-weighted harmonic mean |
| QWK | Ordinal inter-rater agreement |
| Per-class Precision/Recall/F1 | Class-level performance |
| Confusion Matrix | Error pattern analysis |
| Referable DR Sensitivity/Specificity | Clinical safety (Grade ≥ 2) |
| ROC-AUC, PR-AUC | Threshold-independent discrimination |
| ECE, Brier Score | Probability calibration |

---

## Repository Structure

```text
RetinAI-DR/
├── api/                  # FastAPI REST API (endpoints, schemas, inference service)
├── app/                  # Streamlit single-image demo application
├── artifacts/            # Model checkpoints and evaluation outputs
├── configs/              # Model configurations and dataset source specifications
├── data/                 # Fundus datasets (raw) and processed CSV manifests
├── experiments/          # Reproducible experiment artifacts (EXP-000, EXP-001, ...)
├── scripts/              # Dataset prep, manifest creation, QA audit, training & eval
├── src/dr_detection/     # Reusable ML library (models, datasets, QA, metrics, losses)
├── tests/                # Automated pytest test suite
├── legacy/               # Archived unsupported claims (for provenance tracking)
├── pyproject.toml        # Build configuration and project metadata
└── requirements.txt      # Python dependencies
```

---

## Setup & Installation

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Yash00109/RetinAI-Diabetic-Retinopathy-Identification-using-AI.git
cd RetinAI-DR

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 2. Run Tests

```bash
pytest
```

---

## Dataset Pipeline

### 1. Build and Audit Manifest

```bash
python scripts/build_dataset_manifest.py \
  --sources configs/dataset_sources.json \
  --output data/processed/manifest.csv \
  --summary data/processed/manifest_summary.json \
  --require-images
```

Run the automated fundus quality audit:

```bash
python scripts/audit_manifest_quality.py \
  --manifest data/processed/manifest.csv \
  --output artifacts/quality_audit/manifest_quality_audit.csv \
  --summary artifacts/quality_audit/manifest_quality_summary.json \
  --accepted-output data/processed/manifest_quality_accepted.csv \
  --rejected-output artifacts/quality_audit/manifest_quality_rejected.csv
```

---

## Training

Train EfficientNet-B0 baseline on the quality-screened manifest:

```bash
python scripts/train_manifest.py \
  --manifest data/processed/manifest_quality_accepted.csv \
  --output-dir experiments/EXP-000 \
  --model efficientnet_b0 \
  --epochs 25 \
  --batch-size 64 \
  --workers 4 \
  --image-size 384 \
  --learning-rate 0.0002 \
  --weight-decay 0.0001 \
  --drop-rate 0.25 \
  --patience 8 \
  --seed 42
```

---

## Inference & Demos

### API Server

```bash
$env:PYTHONPATH = "src"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Streamlit Application

```bash
& .\.venv\Scripts\streamlit.exe run app/app.py
```

### Grad-CAM Visual Heatmaps

```bash
python scripts/make_gradcam.py \
  --config configs/efficientnet_b0.json \
  --checkpoint artifacts/smoke_test/best_model.pt \
  --image path/to/fundus.png \
  --output artifacts/gradcam.png
```

---

## Reproduction Instructions

Every experiment is stored in `experiments/EXP-XXX/` with:
- `config.json` — full configuration
- `train_history.csv` — epoch-by-epoch training log
- `best_model.pt` — best checkpoint (selected by validation QWK)
- `metrics.json` — final evaluation metrics
- `test_predictions.csv` — per-sample predictions
- `confusion_matrix.png` — visual confusion matrix

To reproduce any experiment, use the saved `config.json` with `scripts/train_manifest.py`.

---

## Limitations

- **Single dataset**: Trained only on APTOS 2019; performance on other populations is unknown.
- **No external validation**: Model has not been validated on independent clinical datasets.
- **Class imbalance**: Severe and Proliferative classes have limited training samples.
- **Image quality dependence**: Performance may degrade on low-quality or non-standard fundus photographs.
- **No longitudinal assessment**: Single-image classification only; does not track disease progression.

## Future Work

- Train and compare multiple architectures (EfficientNet-B3/B4, ConvNeXt-Tiny)
- Implement ordinal/hybrid loss functions
- Test-time augmentation and ensemble methods
- Threshold optimization on validation set
- External validation on DDR/DeepDRiD datasets
- Grad-CAM analysis for clinical interpretability

---

## Disclaimer

> **This software is a research prototype and is NOT approved for clinical use.** It does not independently diagnose, treat, or prevent any disease. All outputs are probabilistic classifications intended to assist — never replace — clinical judgment by qualified healthcare professionals. Predictions should be interpreted as: *"Potentially referable diabetic retinopathy detected. Clinical assessment by a qualified ophthalmologist is recommended."*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
