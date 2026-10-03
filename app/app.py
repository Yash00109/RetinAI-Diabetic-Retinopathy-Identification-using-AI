from __future__ import annotations



import json

import os

import tempfile

from pathlib import Path


import altair as alt
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

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght\@300;400;500;600;700&display=swap');



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
VERIFIED_RESULTS = {
    "experiment": "EXP-001R",
    "dataset": "APTOS 2019",
    "model": "EfficientNet-B0",
    "resolution": "224x224",
    "accuracy": 0.8049,
    "balanced_accuracy": 0.6474,
    "macro_f1": 0.6443,
    "qwk": 0.8959,
}

GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR (PDR)",
}

# Raw names emitted by dr_detection.infer -> grade id
RAW_NAME_TO_GRADE = {
    "No DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferative DR": 4,
}
RAW_NAME_TO_GRADE.update({label: grade for grade, label in GRADE_LABELS.items()})


def _html(markup: str) -> str:
    """Flatten HTML to one line so Markdown never renders indented lines as code."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def _ordered_probabilities(probabilities: dict) -> pd.DataFrame:
    """Return class probabilities ordered by grade 0 -> 4 with explicit grade ids."""
    rows = []
    for raw_name, prob in probabilities.items():
        grade = RAW_NAME_TO_GRADE.get(raw_name)
        if grade is None:
            raise ValueError(f"Unrecognised class name from model: {raw_name!r}")
        rows.append({
            "grade_id": grade,
            "Grade": f"{grade} · {GRADE_LABELS[grade]}",
            "Probability (%)": prob * 100,
        })
    return pd.DataFrame(rows).sort_values("grade_id").reset_index(drop=True)


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



    st.subheader("Verified Research Results")
    st.caption("Best verified test-set result from EXP-001R.")

    sb1, sb2 = st.columns(2)
    with sb1:
        st.metric("Accuracy", f'{VERIFIED_RESULTS["accuracy"] * 100:.2f}%')
        st.metric("Macro F1", f'{VERIFIED_RESULTS["macro_f1"]:.4f}')
    with sb2:
        st.metric("QWK", f'{VERIFIED_RESULTS["qwk"]:.4f}')
        st.metric("Balanced Acc.", f'{VERIFIED_RESULTS["balanced_accuracy"] * 100:.2f}%')

    st.caption(
        f'{VERIFIED_RESULTS["experiment"]} · {VERIFIED_RESULTS["dataset"]} · '
        f'{VERIFIED_RESULTS["model"]}'
    )

    st.divider()
    st.caption("Active device: CPU / CUDA auto-detected")



# -----------------------------------------------------------------------------

# HEADER

# -----------------------------------------------------------------------------

st.markdown(_html("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 1.85rem; font-weight: 700;">👁️ RetinAI-DR Research Prototype</h1>
    <p style="margin: 0.35rem 0 0 0; color: #94A3B8; font-size: 1.05rem;">
        APTOS 2019 · 5-Class Severity Classification · Quality Gate · Grad-CAM
    </p>
</div>
"""), unsafe_allow_html=True)



st.markdown(_html("""
<div class="research-disclaimer">
    ⚠️ <strong>Research Prototype Disclaimer:</strong> This system is NOT a clinical diagnostic tool.
    All predictions are probabilistic classifications intended to assist — never replace — clinical judgment
    by qualified healthcare professionals.
</div>
"""), unsafe_allow_html=True)



tabs = st.tabs(["🔬 Analyze Fundus Image", "📊 Research & Model", "🚀 API Integration"])



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
                    prob_df = _ordered_probabilities(pred["probabilities"])
                    grade_id = int(pred["class_id"])
                    grade_name = GRADE_LABELS[grade_id]
                    badge_color = rec["color"]

                    st.markdown(
                        _html(f"""
                        <div style="background: {badge_color}20; border: 2px solid {badge_color}; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                            <span style="font-size: 1.3rem; font-weight: 700; color: {badge_color};">Grade {grade_id}: {grade_name}</span>
                            <span style="float: right; font-size: 1rem; font-weight: 600; color: #CBD5E1;">P(Grade {grade_id}): {pred['confidence']*100:.1f}% | Score: {pred['expected_continuous_score']:.2f}</span>
                        </div>
                        """),
                        unsafe_allow_html=True,
                    )

                    top_name = max(pred["probabilities"], key=lambda name: pred["probabilities"][name])
                    top_grade = RAW_NAME_TO_GRADE[top_name]
                    if top_grade != grade_id:
                        st.info(
                            f"Grade {grade_id} ({grade_name}) is derived from the ordinal score "
                            f"({pred['expected_continuous_score']:.2f}) and its cutoffs. The highest raw class "
                            f"probability is Grade {top_grade} ({GRADE_LABELS[top_grade]}). "
                            "Untick 'Optimal Cutoffs' in the sidebar to compare."
                        )

                    # Referable DR Callout
                    ref = pred["referable_dr"]
                    if ref["is_referable"]:
                        st.markdown(
                            _html(f"""
                            <div class="clinical-alert-referable">
                                <strong>⚠️ POTENTIALLY REFERABLE DIABETIC RETINOPATHY DETECTED (Probability: {ref['probability']*100:.1f}%)</strong><br>
                                <em>Recommendation:</em> {rec['action']}<br>
                                <em>Note:</em> Clinical assessment by a qualified ophthalmologist is recommended.
                            </div>
                            """),
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            _html(f"""
                            <div class="clinical-alert-routine">
                                <strong>✓ No referable DR threshold crossed</strong><br>
                                <em>Recommendation:</em> {rec['action']}
                            </div>
                            """),
                            unsafe_allow_html=True,
                        )

                    # Class Probability Distribution (ordered: No DR -> PDR)
                    st.markdown("#### Class Probability Distribution")
                    prob_chart = (
                        alt.Chart(prob_df)
                        .mark_bar(color="#3B82F6")
                        .encode(
                            x=alt.X(
                                "Grade:N",
                                sort=prob_df["Grade"].tolist(),
                                title=None,
                                axis=alt.Axis(labelAngle=-30, labelLimit=220),
                            ),
                            y=alt.Y(
                                "Probability (%):Q",
                                scale=alt.Scale(domain=[0, 100]),
                            ),
                            tooltip=[
                                "Grade",
                                alt.Tooltip("Probability (%):Q", format=".1f"),
                            ],
                        )
                    )
                    st.altair_chart(prob_chart, use_container_width=True)



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

    | Config | `{config_path}` |

    | Checkpoint | `{checkpoint_path}` |

    | Architecture | EfficientNet-B0 (via timm) |

    | Status | Research baseline under reconstruction |

    """)



    st.divider()



    st.subheader("Verified Research Metrics")
    st.caption(
        "Verified test-set results from EXP-001R. These describe the research "
        "experiment, not the currently uploaded image."
    )

    cols = st.columns(4)
    cols[0].metric("Test Accuracy", f'{VERIFIED_RESULTS["accuracy"] * 100:.2f}%')
    cols[1].metric("QWK", f'{VERIFIED_RESULTS["qwk"]:.4f}')
    cols[2].metric("Macro F1", f'{VERIFIED_RESULTS["macro_f1"]:.4f}')
    cols[3].metric("Balanced Accuracy", f'{VERIFIED_RESULTS["balanced_accuracy"] * 100:.2f}%')

    st.markdown("#### Verified Experiment")
    st.dataframe(pd.DataFrame([{
        "Experiment": VERIFIED_RESULTS["experiment"],
        "Dataset": VERIFIED_RESULTS["dataset"],
        "Model": VERIFIED_RESULTS["model"],
        "Resolution": VERIFIED_RESULTS["resolution"],
    }]), use_container_width=True, hide_index=True)

    st.markdown("#### Baseline vs Ordinal Experiment")
    st.dataframe(pd.DataFrame([
        {"Metric": "Accuracy", "EXP-000": "78.03%", "EXP-001R": "80.49%"},
        {"Metric": "Balanced Accuracy", "EXP-000": "61.80%", "EXP-001R": "64.74%"},
        {"Metric": "Macro F1", "EXP-000": "0.6100", "EXP-001R": "0.6443"},
        {"Metric": "QWK", "EXP-000": "0.8769", "EXP-001R": "0.8959"},
    ]), use_container_width=True, hide_index=True)

    st.info(
        "Research integrity: unsupported performance claims have been removed. "
        "Only verified experiment results are displayed."
    )

    st.divider()
    st.subheader("Experiment Record")
    st.dataframe(pd.DataFrame([
        {"Experiment":"EXP-000","Model":"EfficientNet-B0","Loss":"Weighted Cross-Entropy","Accuracy":"78.03%","Balanced Accuracy":"61.80%","Macro F1":"0.6100","QWK":"0.8769","Evaluation":"Verified test result"},
        {"Experiment":"EXP-001R","Model":"EfficientNet-B0","Loss":"Ordinal-aware loss","Accuracy":"80.49%","Balanced Accuracy":"64.74%","Macro F1":"0.6443","QWK":"0.8959","Evaluation":"Verified test result"},
        {"Experiment":"EXP-002","Model":"EfficientNet-B0","Loss":"Targeted augmentation","Accuracy":"—","Balanced Accuracy":"—","Macro F1":"—","QWK":"—","Evaluation":"Not retained as a verified improvement"},
        {"Experiment":"EXP-003R","Model":"EfficientNet-B0","Loss":"Class-aware sampling","Accuracy":"—","Balanced Accuracy":"—","Macro F1":"0.6420","QWK":"0.8843","Evaluation":"Validation result only"},
    ]), use_container_width=True, hide_index=True)

    st.caption("A dash (—) means the metric is not presented as a verified test-set result.")

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



API_URL = "http\://localhost:8000/predict"

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

    > **Disclaimer:** This API serves a research prototype. Predictions are probabilistic&#x20;

    > classifications and must not be used for independent clinical diagnosis.

    """)
