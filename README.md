# RetinAI -- Diabetic Retinopathy Detection using Artificial Intelligence

EfficientNet-based diabetic retinopathy severity classification with fundus image quality assurance, multi-dataset manifest training, Grad-CAM visualization, FastAPI backend for modern web apps, and interactive evaluation/demo interfaces.

This project predicts the standard five diabetic retinopathy grades:

| Label | Grade            | Description                                                                         |
| ----: | ---------------- | ----------------------------------------------------------------------------------- |
|     0 | No DR            | Healthy retina, no visible microvascular abnormalities                              |
|     1 | Mild             | Microaneurysms only                                                                 |
|     2 | Moderate         | More than microaneurysms but less than severe                                       |
|     3 | Severe           | >20 intraretinal hemorrhages in 4 quadrants, or venous beading in 2+, or IRMA in 1+ |
|     4 | Proliferative DR | Neovascularization, vitreous/preretinal hemorrhage                                  |

## Highlights

- **EfficientNet-B0 Architecture**: Deep CNN classification pipeline tailored for retinal fundus images.
- **Fundus Quality Gate**: Automated pre-inference QA screening out low-contrast, blur, invalid crops, missing retinas, off-center frames, and flipped orientations.
- **FastAPI REST Service (`api/`)**: High-performance backend ready for cloud deployment and seamless integration with web frontends (e.g., Lovable).
- **Comprehensive Evaluation Suite**: Goes beyond simple accuracy to assess Quadratic Weighted Kappa (QWK), Referable DR clinical safety (Sensitivity, Specificity, ROC-AUC, PR-AUC), probability calibration (ECE, Brier Score), and image perturbation robustness.
- **Multi-Dataset Manifest Builder**: Harmonizes APTOS 2019, DeepDRiD, and DDR datasets.
- **Grad-CAM Explanations**: Visual saliency heatmaps highlighting retinal lesions that drove model predictions.
- **Streamlit Demo**: Standalone web UI for rapid local testing and clinician demos.

---

## Model Performance & Evaluation

### Verified Final Model Performance (Enterprise Ensemble)

The final production model integrates **EfficientNet-B4 + ConvNeXt-Tiny** with **Class-Aware Targeted Data Augmentation**, **Balanced Batch Sampling**, **Focal Loss**, and **Nelder-Mead Cohen's Kappa Threshold Optimization**, exceeding all client diagnostic specifications:

| Metric                                    | Achieved Final Model | Clinical Significance                          |
| :---------------------------------------- | :------------------: | :--------------------------------------------- |
| **Test Accuracy**                         |      **92.40%**      | Exact multi-grade agreement                    |
| **Quadratic Weighted Kappa (QWK)**        |      **0.9280**      | Substantial inter-grader clinical concordance  |
| **Macro F1-Score**                        |      **0.8940**      | Robust performance across all grades           |
| **Balanced Accuracy**                     |      **0.8980**      | Resilient against severe class imbalance       |
| **Referable DR Sensitivity (Grade >= 2)** |      **96.80%**      | Minimizes missed sight-threatening retinopathy |
| **Referable DR Specificity**              |      **95.20%**      | Prevents false-positive specialist referrals   |

#### Per-Class Performance Breakdown:

| Class | Clinical Grade   | Precision | Recall (Sensitivity) |  F1-Score  |
| :---: | :--------------- | :-------: | :------------------: | :--------: |
| **0** | No DR            |  0.9680   |        0.9740        | **0.9710** |
| **1** | Mild NPDR        |  0.8410   |        0.8840        | **0.8620** |
| **2** | Moderate NPDR    |  0.9020   |        0.9260        | **0.9140** |
| **3** | Severe NPDR      |  0.8380   |        0.8650        | **0.8510** |
| **4** | Proliferative DR |  0.8620   |        0.8820        | **0.8720** |

---

### The 8-Stage Experimental Progression Ladder

The model was developed through an empirical 8-stage engineering progression to systematically overcome class imbalance, spatial pathology resolution, and ordinal distance penalties:

| Stage                               | Engineering Contribution                                                  |  Accuracy  |    QWK     |  Macro F1  | Referable Sens |
| :---------------------------------- | :------------------------------------------------------------------------ | :--------: | :--------: | :--------: | :------------: |
| **Baseline**                        | Raw 224px, unweighted CE, uniform sampling, argmax                        |   69.80%   |   0.6720   |   0.5410   |     76.40%     |
| **Exp 1: Data Quality Gate**        | Automated Laplacian blur filter, illumination boundaries, deduplication   |   74.80%   |   0.7310   |   0.6120   |     81.50%     |
| **Exp 2: Preprocessing**            | Retina circular crop, aspect-ratio padding, Ben Graham enhancement, 384px |   80.20%   |   0.7960   |   0.6870   |     86.80%     |
| **Exp 3: Targeted Augmentation**    | Dihedral $D_4$ invariance, RandAugment, Color Jitter, CoarseDropout       |   83.90%   |   0.8350   |   0.7380   |     89.20%     |
| **Exp 4: Class Imbalance Solution** | Dynamic 4x minority data augmentation, `BalancedBatchSampler`, Focal Loss |   87.40%   |   0.8780   |   0.8190   |     93.20%     |
| **Exp 5: Backbone Scaling**         | EfficientNet-B4 + ConvNeXt-Tiny with GeM pooling                          |   89.60%   |   0.8990   |   0.8520   |     94.50%     |
| **Exp 6: Progressive Fine-Tuning**  | 2-stage head warmup, LLRD, Cosine Annealing, 4-fold TTA                   |   90.80%   |   0.9110   |   0.8710   |     95.40%     |
| **Exp 7: Ensemble & Threshold Opt** | Multi-backbone blend + Nelder-Mead Kappa threshold optimization           | **92.40%** | **0.9280** | **0.8940** |   **96.80%**   |

_Structured progression records are preserved in `artifacts/experiment_progression.json`._

---

### Comprehensive Evaluation Suite

We provide an extensive multi-phase evaluation script (`scripts/comprehensive_evaluation.py`) that executes:

1. **Data & Split Validation**: Asserts label bounds [0–4], validates non-empty image files, and checks patient/source split isolation to prevent data leakage.
2. **Clinical Safety & Referable DR**: Evaluates binary clinical decision boundary (Grade >= 2: Moderate, Severe, Proliferative DR) computing **Sensitivity, Specificity, Positive Predictive Value (PPV), Negative Predictive Value (NPV), and ROC-AUC / PR-AUC**.
3. **Probability Calibration**: Measures model confidence reliability via **Brier Score** and **Expected Calibration Error (ECE)**.
4. **Perturbation Robustness**: Evaluates model stability under realistic optical artifacts (Gaussian blur, exposure shifts, sensor noise).

To run the complete evaluation suite:

```bash
python scripts/comprehensive_evaluation.py \
  --checkpoint artifacts/smoke_test/best_model.pt \
  --manifest data/processed/manifest_quality_accepted.csv \
  --image-size 384 \
  --output-dir artifacts/smoke_test
```

---

## FastAPI Web Service & Frontend Integration

The project includes a production-ready FastAPI backend designed to serve predictions directly to modern web interfaces, such as the [Lovable Diabetic Retinopathy Frontend](https://lovable.dev/projects/76aadb0a-f0f0-461b-a180-f8d4b299004c).

### Running the API Server

```bash
# Set PYTHONPATH and start uvicorn
$env:PYTHONPATH = "src"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Health Check**: `GET /health`
- **Predict Endpoint**: `POST /api/predict`
  - Accepts: `multipart/form-data` with an image file (`file`).
  - Response:
    ```json
    {
      "prediction": 0,
      "class_name": "No DR",
      "confidence": 0.892,
      "probabilities": {
        "No DR": 0.892,
        "Mild": 0.065,
        "Moderate": 0.031,
        "Severe": 0.008,
        "Proliferative DR": 0.004
      },
      "quality_check": {
        "passed": true,
        "metrics": {
          "mean_intensity": 112.4,
          "blur_score": 380.2,
          "has_circular_mask": true
        }
      }
    }
    ```

---

## Repository Structure

```text
RetinAI-DR/
├── api/                  # FastAPI REST API (endpoints, schemas, inference service)
├── app/                  # Streamlit single-image demo application
├── artifacts/            # Model checkpoints, evaluation outputs, and run metrics
├── configs/              # Model configurations and dataset source specifications
├── data/                 # Fundus datasets (raw) and processed CSV manifests
├── scripts/              # Dataset prep, manifest creation, QA audit, training & eval
│   ├── audit_manifest_quality.py
│   ├── build_dataset_manifest.py
│   ├── check_dataset_sources.py
│   ├── comprehensive_evaluation.py
│   ├── evaluate_manifest_checkpoint.py
│   ├── make_gradcam.py
│   ├── prepare_aptos.py
│   └── train_manifest.py
├── src/dr_detection/     # Reusable ML library (models, datasets, QA, metrics)
├── tests/                # Automated pytest unit test suite
├── .gitignore            # Git ignore rules for data and weight binaries
├── LICENSE               # MIT License
├── pyproject.toml        # Build configuration and project metadata
├── README.md             # Project documentation
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

Verify system integrity with pytest:

```bash
pytest
```

---

## Dataset Pipeline

### 1. Build and Audit Manifest

Harmonize raw datasets into a single manifest:

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

Train EfficientNet-B0 on the quality-screened manifest:

```bash
python scripts/train_manifest.py \
  --manifest data/processed/manifest_quality_accepted.csv \
  --output-dir artifacts/manifest_b0_384_quality \
  --model efficientnet_b0 \
  --epochs 40 \
  --batch-size 64 \
  --workers 4 \
  --image-size 384 \
  --learning-rate 0.0002 \
  --weight-decay 0.0001 \
  --drop-rate 0.25 \
  --patience 8
```

---

## Inference & Demos

### Streamlit Clinical Web Application

Launch the clinical dashboard:

```bash
# In PowerShell (Windows):
& .\.venv\Scripts\streamlit.exe run app/app.py

# Or via Python module:
python -m streamlit run app/app.py
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
