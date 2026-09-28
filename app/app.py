from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from dr_detection.infer import predict_image

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RetinAI-DR | Research Prototype",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        border-left: 6px solid #3B82F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .badge-card {
        padding: 1rem 1.25rem;
        border-radius: 10px;
        background: #1E293B;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #334155;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .severity-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
    }
    
    .clinical-alert-referable {
        background-color: #FEF2F2;
        color: #991B1B;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .clinical-alert-routine {
        background-color: #ECFDF5;
        color: #065F46;
        border-left: 5px solid #10B981;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .research-disclaimer {
        background-color: #FFFBEB;
        color: #92400E;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-weight: 500;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONSTANTS & SAMPLE TEST CASES
# -----------------------------------------------------------------------------
BENCHMARK_SAMPLES = {
    "Grade 0: Normal / No DR": "data/raw/aptos/train_images/hf_aptos_02005.png",
    "Grade 1: Mild NPDR (Microaneurysms)": "data/raw/aptos/train_images/hf_aptos_00209.png",
    "Grade 2: Moderate NPDR (Hemorrhages / Exudates)": "data/raw/aptos/train_images/hf_aptos_03571.png",
    "Grade 3: Severe NPDR (4-2-1 Rule)": "data/raw/aptos/train_images/hf_aptos_03359.png",
    "Grade 4: Proliferative DR (Neovascularization)": "data/raw/aptos/train_images/hf_aptos_01764.png",
}

DEFAULT_CONFIG = "configs/efficientnet_b0.json"
DEFAULT_CHECKPOINT = "artifacts/smoke_test/best_model.pt"


def _load_verified_metrics(checkpoint_path: str) -> dict | None:
    """Load verified metrics from the checkpoint directory, if they exist."""
    checkpoint_dir = Path(checkpoint_path).parent
    metrics_path = checkpoint_dir / "test_metrics.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/ophthalmology.png", width=64)
    st.title("RetinAI-DR")
    st.caption("Research Prototype — Diabetic Retinopathy Classification")
    st.divider()

    st.subheader("Pipeline Configuration")
    config_path = st.text_input("Config Path", DEFAULT_CONFIG)
    checkpoint_path = st.text_input("Checkpoint Path", DEFAULT_CHECKPOINT)
    use_tta = st.checkbox("Test-Time Augmentation (TTA)", value=True, help="Averages horizontal flip predictions for stabilized confidence")
    use_opt_thresholds = st.checkbox("Optimal Cutoffs", value=True, help="Applies calibrated continuous cutoffs to maximize QWK")

    st.divider()

    # Display verified metrics if they exist
    st.subheader("Verified Model Performance")
    verified_metrics = _load_verified_metrics(checkpoint_path)
    if verified_metrics:
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            acc = verified_metrics.get("accuracy", 0)
            st.metric("Accuracy", f"{acc*100:.2f}%")
        with col_sb2:
            qwk = verified_metrics.get("quadratic_weighted_kappa", 0)
            st.metric("QWK", f"{qwk:.4f}")
        macro_f1 = verified_metrics.get("macro_f1", 0)
        st.metric("Macro F1", f"{macro_f1:.4f}")
    else:
        st.info("No verified metrics found for the selected checkpoint. Run evaluation to generate metrics.")

    st.caption("Active Device: CPU / CUDA Auto-detected")

# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700;">👁️ RetinAI-DR Research Prototype</h1>
    <p style="margin: 0.35rem 0 0 0; color: #94A3B8; font-size: 1.05rem;">
        Automated Diabetic Retinopathy Grading, Fundus Quality Gate, and Explainable AI (Grad-CAM)
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="research-disclaimer">
    ⚠️ <strong>Research Prototype Disclaimer:</strong> This system is NOT a clinical diagnostic tool. 
    All predictions are probabilistic classifications intended to assist — never replace — clinical judgment 
    by qualified healthcare professionals.
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🔬 Classification & Explainability", "📊 Model Information", "🚀 API Integration"])

# =============================================================================
# TAB 1: CLASSIFICATION & EXPLAINABLE AI
# =============================================================================
with tabs[0]:
    col_input, col_results = st.columns([1, 1.25], gap="large")

    with col_input:
        st.subheader("1. Fundus Image Input")
        input_mode = st.radio(
            "Select input source:",
            ["🔬 Sample Gallery", "📁 Upload Fundus Photograph"],
            horizontal=True,
        )

        image_to_process = None
        sample_label = None

        if input_mode == "🔬 Sample Gallery":
            selected_case = st.selectbox("Select Sample Case:", list(BENCHMARK_SAMPLES.keys()))
            sample_path = BENCHMARK_SAMPLES[selected_case]
            if os.path.exists(sample_path):
                image_to_process = sample_path
                sample_label = selected_case
            else:
                st.warning(f"Sample file not found at `{sample_path}`. Please upload an image.")
        else:
            uploaded_file = st.file_uploader(
                "Upload digital retinal fundus photograph (PNG, JPG, JPEG):",
                type=["png", "jpg", "jpeg", "tiff"],
            )
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tf:
                    tf.write(uploaded_file.read())
                    image_to_process = tf.name
                sample_label = f"Uploaded: {uploaded_file.name}"

        if image_to_process:
            st.image(image_to_process, caption=f"Input: {sample_label}", use_container_width=True)

    with col_results:
        st.subheader("2. Classification Results")

        if not image_to_process:
            st.info("👈 Please select a sample or upload a fundus photograph to begin.")
        elif not os.path.exists(checkpoint_path):
            st.error(f"Checkpoint not found at `{checkpoint_path}`. Please verify the checkpoint path in the sidebar.")
        else:
            with st.spinner("Running quality gate, neural inference, and classification..."):
                try:
                    result = predict_image(
                        config_path=config_path,
                        checkpoint_path=checkpoint_path,
                        image_path=image_to_process,
                        use_tta=use_tta,
                        use_threshold_optimization=use_opt_thresholds,
                        generate_gradcam=True,
                    )
                except Exception as e:
                    st.error(f"Inference Error: {str(e)}")
                    result = None

            if result:
                # ----------------- Quality Gate Section -----------------
                st.markdown("#### Pre-Flight Quality Gate")
                q = result["quality"]
                if result["accepted"]:
                    st.success("✅ Image Quality Accepted: Passed sharpness, contrast, and anatomical orientation gates.")
                else:
                    st.error(f"❌ Image Rejected: {', '.join(q.get('rejection_reasons', ['Quality threshold failed']))}")

                col_q1, col_q2, col_q3, col_q4 = st.columns(4)
                with col_q1:
                    st.metric("Sharpness", f"{q.get('blur_score', 0):.1f}", help="Laplacian variance (Threshold >= 100)")
                with col_q2:
                    st.metric("Illumination", f"{q.get('brightness', 0):.1f}", help="Mean intensity [15.0 - 240.0]")
                with col_q3:
                    st.metric("Contrast", f"{q.get('contrast', 0):.1f}")
                with col_q4:
                    st.metric("Circularity", f"{q.get('retina_circularity', 0):.2f}")

                st.divider()

                # ----------------- Classification Result -----------------
                if result["accepted"]:
                    pred = result["prediction"]
                    rec = result["clinical_recommendation"]

                    st.markdown("#### Classification Result")
                    grade_id = pred["class_id"]
                    grade_name = pred["class_name"]
                    badge_color = rec["color"]

                    st.markdown(
                        f"""
                        <div style="background: {badge_color}20; border: 2px solid {badge_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                            <span style="font-size: 1.3rem; font-weight: 700; color: {badge_color};">
                                Grade {grade_id}: {grade_name}
                            </span>
                            <span style="float: right; font-size: 1rem; font-weight: 600; color: #CBD5E1;">
                                Confidence: {pred['confidence']*100:.1f}% | Score: {pred['expected_continuous_score']:.2f}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Referable DR Callout
                    ref = pred["referable_dr"]
                    if ref["is_referable"]:
                        st.markdown(
                            f"""
                            <div class="clinical-alert-referable">
                                <strong>⚠️ POTENTIALLY REFERABLE DIABETIC RETINOPATHY DETECTED (Probability: {ref['probability']*100:.1f}%)</strong><br>
                                <em>Recommendation:</em> {rec['action']}<br>
                                <em>Note:</em> Clinical assessment by a qualified ophthalmologist is recommended.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"""
                            <div class="clinical-alert-routine">
                                <strong>✅ LOW RISK (Non-referable)</strong><br>
                                <em>Recommendation:</em> {rec['action']}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Class Probabilities Distribution Chart
                    st.markdown("#### Class Probability Distribution")
                    prob_df = pd.DataFrame({
                        "Grade": list(pred["probabilities"].keys()),
                        "Probability (%)": [v * 100 for v in pred["probabilities"].values()],
                    })
                    st.bar_chart(prob_df.set_index("Grade"), color="#3B82F6")

                    # Explainable AI (Grad-CAM)
                    if "gradcam" in result and os.path.exists(result["gradcam"]["output_path"]):
                        st.divider()
                        st.markdown("#### 🔬 Grad-CAM Visualization")
                        st.caption("Highlights retinal features influencing the classification. For research/educational purposes.")
                        cam_img_path = result["gradcam"]["output_path"]
                        st.image(cam_img_path, caption="Grad-CAM Activation Heatmap Overlay", use_container_width=True)

                    # Export Report
                    st.divider()
                    report_json = json.dumps(result, indent=2)
                    st.download_button(
                        label="📥 Download Report (JSON)",
                        data=report_json,
                        file_name="retinai_report.json",
                        mime="application/json",
                    )

# =============================================================================
# TAB 2: MODEL INFORMATION
# =============================================================================
with tabs[1]:
    st.subheader("Model Information")

    st.markdown(f"""
    | Property | Value |
    |:---------|:------|
    | **Config** | `{config_path}` |
    | **Checkpoint** | `{checkpoint_path}` |
    | **Architecture** | EfficientNet-B0 (via timm) |
    | **Status** | Research baseline under reconstruction |
    """)

    st.divider()

    st.subheader("Verified Metrics")
    if verified_metrics:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Accuracy", f"{verified_metrics.get('accuracy', 0)*100:.2f}%")
        with col_m2:
            st.metric("QWK", f"{verified_metrics.get('quadratic_weighted_kappa', 0):.4f}")
        with col_m3:
            st.metric("Macro F1", f"{verified_metrics.get('macro_f1', 0):.4f}")
        with col_m4:
            bal_acc = verified_metrics.get("balanced_accuracy", 0)
            st.metric("Balanced Acc", f"{bal_acc*100:.2f}%" if bal_acc else "N/A")

        if "classification_report" in verified_metrics:
            st.subheader("Per-Class Results")
            report = verified_metrics["classification_report"]
            rows = []
            class_names = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
            for cls_id in range(5):
                cls_key = str(cls_id)
                if cls_key in report:
                    r = report[cls_key]
                    rows.append({
                        "Class": f"{cls_id} - {class_names.get(cls_id, cls_key)}",
                        "Precision": f"{r.get('precision', 0):.4f}",
                        "Recall": f"{r.get('recall', 0):.4f}",
                        "F1-Score": f"{r.get('f1-score', 0):.4f}",
                        "Support": int(r.get("support", 0)),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if "confusion_matrix" in verified_metrics:
            st.subheader("Confusion Matrix")
            cm = verified_metrics["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=[f"True {i}" for i in range(len(cm))],
                columns=[f"Pred {i}" for i in range(len(cm[0]))],
            )
            st.dataframe(cm_df, use_container_width=True)
    else:
        st.warning("No verified metrics available for the selected checkpoint. Train and evaluate a model to generate verified metrics.")

    st.divider()

    # Experiment tracking placeholder
    st.subheader("Experiment Tracking")
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        exp_dirs = sorted([d for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith("EXP-")])
        if exp_dirs:
            for exp_dir in exp_dirs:
                metrics_file = exp_dir / "metrics.json"
                config_file = exp_dir / "config.json"
                if metrics_file.exists():
                    m = json.loads(metrics_file.read_text(encoding="utf-8"))
                    with st.expander(f"📌 {exp_dir.name} — Accuracy: {m.get('accuracy', 0)*100:.2f}% | QWK: {m.get('quadratic_weighted_kappa', 0):.4f}"):
                        st.json(m)
                        if config_file.exists():
                            st.json(json.loads(config_file.read_text(encoding="utf-8")))
                else:
                    st.text(f"{exp_dir.name}: No metrics.json found")
        else:
            st.info("No experiments found. Run training to create EXP-000.")
    else:
        st.info("No experiments directory found. Run training to begin.")

# =============================================================================
# TAB 3: API INTEGRATION
# =============================================================================
with tabs[2]:
    st.subheader("API Integration")
    st.markdown("""
    RetinAI-DR provides a REST API for programmatic access to the classification pipeline.
    """)

    col_api1, col_api2 = st.columns(2)
    with col_api1:
        st.markdown("#### Endpoints")
        st.code("""
# 1. Health Check
GET  /health
Response: {"status": "online", "device": "CPU", "version": "0.2.0-research"}

# 2. Model Information
GET  /model-info
Response: {"model_architecture": "...", "verified_metrics": {...}}

# 3. Classification
POST /predict?gradcam=true
Body: multipart/form-data with 'file' (PNG/JPG image)

# 4. Explainability Heatmap (Base64)
POST /explain
Body: multipart/form-data with 'file'
        """, language="bash")

    with col_api2:
        st.markdown("#### Python Client Example")
        st.code("""
import requests

API_URL = "http://localhost:8000/predict"
image_path = "patient_fundus.png"

with open(image_path, "rb") as f:
    response = requests.post(
        API_URL, files={"file": f}, params={"gradcam": True}
    )

data = response.json()
print("Grade:", data["prediction"]["class_name"])
print("Confidence:", data["prediction"]["confidence"])
print("Referable:", data["prediction"]["referable_dr"]["is_referable"])
        """, language="python")

    st.divider()
    st.markdown("""
    > **Disclaimer:** This API serves a research prototype. Predictions are probabilistic 
    > classifications and must not be used for independent clinical diagnosis.
    """)
