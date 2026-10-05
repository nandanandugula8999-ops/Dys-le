"""
Hybrid AI-Based Dyslexia Detection System - Streamlit Web Application
Features:
- Multi-Image & Single-Image Screening with Individual Result Averaging
- Aggregated Consensus Risk Assessment across Multiple Handwriting Samples
- Deep Feature Representation & Model-Wise Consensus Breakdown
- Benchmark Performance Dashboard (Full-Dataset Results, Confusion Matrices, ROC, PCA)
- Educational Clinical Guide on Dyslexia Handwriting Patterns
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

# Ensure root project directory and app directory are on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

for p in [str(BASE_DIR), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.config import (
    CLASSES, CLASS_RISK_MAP, RESULTS_DIR, MODELS_DIR,
    CNN_MODEL_PATH, SVM_MODEL_PATH, RF_MODEL_PATH, SCALER_PATH, METADATA_PATH
)
import importlib
import src.hybrid_pipeline
importlib.reload(src.hybrid_pipeline)
from src.hybrid_pipeline import HybridDyslexiaDetector

try:
    from utils import get_demo_samples, generate_clinical_report, generate_batch_clinical_report
except (ImportError, ModuleNotFoundError):
    from app.utils import get_demo_samples, generate_clinical_report, generate_batch_clinical_report


# Streamlit Page Config
st.set_page_config(
    page_title="Hybrid AI Dyslexia Screening System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge-card {
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .sample-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector():
    """Cache the loaded hybrid detector to ensure instant inference."""
    try:
        if CNN_MODEL_PATH.exists() and SVM_MODEL_PATH.exists() and RF_MODEL_PATH.exists():
            return HybridDyslexiaDetector(
                cnn_path=CNN_MODEL_PATH,
                svm_path=SVM_MODEL_PATH,
                rf_path=RF_MODEL_PATH,
                scaler_path=SCALER_PATH
            )
        else:
            return None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


# Sidebar Navigation & System Status
st.sidebar.image("https://img.icons8.com/fluency/96/brain.png", width=70)
st.sidebar.title("Dyslexia AI Portal")
st.sidebar.markdown("**Hybrid CNN + SVM + RF Screening**")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🔍 Live Handwriting Screening",
        "📁 Batch Screening & Cohort Report",
        "📊 Benchmark & Analytics",
        "📖 Clinical & Dyslexia Guide"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Architecture")

detector = load_detector()
if detector is not None and (not hasattr(detector, "predict_batch") or not hasattr(detector, "preprocess_to_pil")):
    st.cache_resource.clear()
    detector = load_detector()

if detector is not None:

    st.sidebar.success("✅ CNN Backbone: Active (98.8% Val Acc)")
    st.sidebar.success("✅ SVM Classifier (RBF): Active (97.0% Acc)")
    st.sidebar.success("✅ Random Forest (150T): Active (98.8% Acc)")
    st.sidebar.success("✅ Hybrid Consensus: Active (88.7% Test Acc)")
else:
    st.sidebar.warning("⏳ Models are currently loading. Standby...")

st.sidebar.markdown("---")
st.sidebar.caption("Dyslexia Screening System v1.1.0 | Full-Scale 208k Dataset")


# ==========================================
# PAGE 1: LIVE HANDWRITING SCREENING
# ==========================================
if menu == "🔍 Live Handwriting Screening":
    st.markdown('<div class="main-header">🧠 AI-Assisted Dyslexia Screening</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Submit a single character or multiple handwriting samples at once. '
        'The system analyzes each image individually and calculates the aggregated result average across all samples.</div>',
        unsafe_allow_html=True
    )

    if detector is None:
        st.warning("⚠️ Trained models are currently not loaded. Please verify the models directory.")
    else:
        tab1, tab2 = st.tabs(["📤 Upload Sample(s)", "🎯 Benchmark Test Samples"])
        submitted_images = []
        submitted_filenames = []

        with tab1:
            uploaded_files = st.file_uploader(
                "Upload one or multiple handwriting images (PNG, JPG, JPEG)",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True
            )
            if uploaded_files:
                for f in uploaded_files:
                    submitted_images.append(Image.open(f))
                    submitted_filenames.append(f.name)

        with tab2:
            st.markdown("Select individual benchmark samples or run the multi-sample test suite:")
            demo_samples = get_demo_samples(num_per_class=2)

            col_btn, _ = st.columns([2, 1])
            with col_btn:
                if st.button("⚡ Run Multi-Sample Test Suite (6 Benchmark Samples at Once)", type="primary"):
                    for cls_name, s_list in demo_samples.items():
                        for s_path, s_name in s_list:
                            submitted_images.append(Image.open(s_path))
                            submitted_filenames.append(f"{cls_name}_{s_name}")

            st.markdown("---")
            if demo_samples:
                col_demos = st.columns(3)
                for idx, (cls_name, samples) in enumerate(demo_samples.items()):
                    with col_demos[idx]:
                        st.markdown(f"**Class: {cls_name}**")
                        for s_path, s_name in samples:
                            img = Image.open(s_path)
                            st.image(img, caption=s_name, width=90)
                            if st.button(f"Analyze {s_name}", key=f"btn_{cls_name}_{s_name}"):
                                submitted_images = [img]
                                submitted_filenames = [f"{cls_name} ({s_name})"]

        # ==========================================
        # SCENARIO A: MULTIPLE IMAGES (AVERAGED RESULTS)
        # ==========================================
        if len(submitted_images) > 1:
            st.markdown("---")
            st.markdown(f"### 📋 Multi-Image Assessment: Averaged Results ({len(submitted_images)} Samples)")

            with st.spinner(f"Evaluating {len(submitted_images)} handwriting images and calculating average results..."):
                batch_res = detector.predict_batch(submitted_images, submitted_filenames)

            overall_pred = batch_res["overall_prediction"]
            overall_conf = batch_res["overall_confidence_percentage"]
            risk_level = batch_res["overall_risk_level"]
            risk_color = batch_res["overall_risk_color"]
            risk_desc = batch_res["overall_risk_description"]
            total_count = batch_res["total_images"]

            # Executive Average Consensus Card
            st.markdown(f"""
            <div class="badge-card" style="background-color: {risk_color};">
                <div style="font-size: 1.05rem; opacity: 0.9;">AGGREGATED SESSION RESULT (AVERAGE ACROSS {total_count} IMAGES)</div>
                <div style="font-size: 2.3rem; font-weight: 800; letter-spacing: 1px;">{overall_pred.upper()}</div>
                <div style="font-size: 1.2rem; font-weight: 600; margin-top: 5px;">
                    Session Risk Level: {risk_level} | Average Confidence: {overall_conf}
                </div>
                <div style="font-size: 0.95rem; margin-top: 8px; opacity: 0.95;">{risk_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # Key Statistics Cards
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            c_s1.metric("Total Samples", total_count)
            c_s2.metric(
                "Reversal Rate (Dyslexic)",
                f"{batch_res['reversal_rate']:.1f}%",
                f"{batch_res['reversal_count']} of {total_count} images"
            )
            c_s3.metric(
                "Correction Rate (Overwriting)",
                f"{batch_res['corrected_rate']:.1f}%",
                f"{batch_res['corrected_count']} of {total_count} images"
            )
            c_s4.metric(
                "Normal Rate (Standard)",
                f"{batch_res['normal_rate']:.1f}%",
                f"{batch_res['normal_count']} of {total_count} images"
            )

            # Averaged Probability Distribution Chart
            st.markdown("### 📊 Averaged Probability Distribution Across All Images")
            avg_df = pd.DataFrame({
                "Category": CLASSES,
                "Consensus Hybrid Average (%)": [batch_res["average_probabilities"][c] * 100 for c in CLASSES],
                "CNN Average (%)": [batch_res["model_averages"]["CNN"][c] * 100 for c in CLASSES],
                "SVM Average (%)": [batch_res["model_averages"]["SVM"][c] * 100 for c in CLASSES],
                "Random Forest Average (%)": [batch_res["model_averages"]["Random_Forest"][c] * 100 for c in CLASSES]
            }).set_index("Category")

            st.bar_chart(avg_df)

            # Individual Image Result Gallery
            st.markdown("### 🖼️ Individual Sample Breakdown")
            num_cols = min(4, total_count)
            cols = st.columns(num_cols)

            for idx, r in enumerate(batch_res["individual_results"]):
                with cols[idx % num_cols]:
                    im = submitted_images[idx]
                    pred = r["prediction"]
                    conf = r["confidence_percentage"]
                    badge_color = CLASS_RISK_MAP[pred]["color"]

                    st.markdown(f"""
                    <div class="sample-card">
                        <div style="font-size:0.8rem; color:#64748b; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                            {r.get('filename', f'Image {idx+1}')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(im, use_container_width=True)
                    st.markdown(f"""
                    <div style="text-align:center; margin-top:-5px; margin-bottom:12px;">
                        <span style="background-color:{badge_color}; color:white; padding:3px 8px; border-radius:12px; font-size:0.8rem; font-weight:bold;">
                            {pred} ({conf})
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

            # Detailed Results Table
            st.markdown("### 📑 Detailed Tabular Summary")
            rows = []
            for idx, r in enumerate(batch_res["individual_results"]):
                rows.append({
                    "Sample": r.get("filename", f"Image_{idx+1}"),
                    "Consensus Decision": r["prediction"],
                    "Confidence": r["confidence_percentage"],
                    "CNN Prediction": r["models"]["CNN"]["prediction"],
                    "SVM Prediction": r["models"]["SVM"]["prediction"],
                    "RF Prediction": r["models"]["Random_Forest"]["prediction"],
                    "Agreement": "Unanimous" if r["unanimous_agreement"] else "Majority",
                    "Reversal Prob (%)": f"{r['ensemble_probabilities']['Reversal']*100:.1f}%",
                    "Corrected Prob (%)": f"{r['ensemble_probabilities']['Corrected']*100:.1f}%",
                    "Normal Prob (%)": f"{r['ensemble_probabilities']['Normal']*100:.1f}%"
                })

            df_summary = pd.DataFrame(rows)
            st.dataframe(df_summary, use_container_width=True)

            # Clinical Report Generation for Multi-Image Session
            st.markdown("---")
            st.subheader("Generate Multi-Image Clinical Screening Summary")
            col_id, col_notes = st.columns([1, 2])
            with col_id:
                session_id = st.text_input("Session / Student Identifier", value="STUDENT-SCREEN-2026")
            with col_notes:
                notes = st.text_area("Clinician / Educator Observations", value="Multi-character handwriting screening conducted.")

            report_txt = generate_batch_clinical_report(batch_res, session_id=session_id, notes=notes)
            st.download_button(
                label="📥 Download Aggregated Clinical Report (.txt)",
                data=report_txt,
                file_name=f"Dyslexia_MultiSample_{session_id}.txt",
                mime="text/plain"
            )

        # ==========================================
        # SCENARIO B: SINGLE IMAGE (DETAILED VIEW)
        # ==========================================
        elif len(submitted_images) == 1:
            selected_image = submitted_images[0]
            source_label = submitted_filenames[0]

            # Perform Hybrid Prediction
            with st.spinner("Applying adaptive normalization, extracting 128-d features & computing consensus..."):
                result = detector.predict(selected_image)

            st.markdown("---")
            st.subheader("Adaptive Image Preprocessing & Normalization Pipeline")

            prep_meta = result.get("preprocessing", {})
            auto_inv = prep_meta.get("auto_inverted", False)
            bg_kind = prep_meta.get("background_detected", "dark_canvas")
            centered = prep_meta.get("centered", False)

            if auto_inv:
                st.info("💡 **Camera/Paper Input Detected**: Automatically inverted dark ink on light paper, suppressed shadows, and centered glyph for the neural network.")

            c_prep1, c_prep2, c_prep3 = st.columns(3)
            with c_prep1:
                st.markdown("**1. Original Input**")
                st.image(selected_image, width=140, caption=f"Source: {source_label}")

            with c_prep2:
                st.markdown("**2. Adaptive Normalization**")
                if hasattr(detector, "preprocess_to_pil"):
                    prep_img = detector.preprocess_to_pil(selected_image)
                elif hasattr(detector, "preprocess_image"):
                    prep_res = detector.preprocess_image(selected_image, return_details=True)
                    prep_img = prep_res[2] if isinstance(prep_res, tuple) else selected_image.convert("L").resize((64, 64))
                else:
                    prep_img = selected_image.convert("L").resize((64, 64))
                cap = "Auto-Inverted & Centered (Paper)" if auto_inv else "Standard Dark Canvas"
                st.image(prep_img, width=140, caption=cap)

            with c_prep3:
                st.markdown("**3. CNN Input Tensor Map (64x64)**")
                arr_prep = np.array(prep_img) / 255.0
                fig_heat, ax_heat = plt.subplots(figsize=(3, 3))
                ax_heat.imshow(arr_prep, cmap="viridis")
                ax_heat.axis("off")
                st.pyplot(fig_heat)
                plt.close(fig_heat)

            st.markdown("---")
            st.subheader("Screening Consensus & Risk Assessment")

            pred_class = result["prediction"]
            confidence = result["confidence_percentage"]
            risk_level = result["risk_level"]
            risk_color = result["risk_color"]
            description = result["risk_description"]

            st.markdown(f"""
            <div class="badge-card" style="background-color: {risk_color};">
                <div style="font-size: 1.1rem; opacity: 0.9;">CONSENSUS CLASSIFICATION</div>
                <div style="font-size: 2.3rem; font-weight: 800; letter-spacing: 1px;">{pred_class.upper()}</div>
                <div style="font-size: 1.2rem; font-weight: 600; margin-top: 5px;">Risk Assessment: {risk_level} (Confidence: {confidence})</div>
                <div style="font-size: 0.95rem; margin-top: 8px; opacity: 0.95;">{description}</div>
            </div>
            """, unsafe_allow_html=True)

            # 3-Classifier Comparison Cards
            st.markdown("### Individual Classifier Breakdown")
            c_cnn, c_svm, c_rf = st.columns(3)

            with c_cnn:
                cnn_pred = result["models"]["CNN"]["prediction"]
                cnn_conf = result["models"]["CNN"]["confidence"] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color:#2563eb; margin:0;">🧠 CNN Backbone</h4>
                    <p style="font-size:0.85rem; color:#64748b; margin-bottom:8px;">Deep Spatial Feature Learning</p>
                    <h3 style="margin:5px 0;">{cnn_pred}</h3>
                    <p style="font-weight:600; color:#0284c7;">Confidence: {cnn_conf:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            with c_svm:
                svm_pred = result["models"]["SVM"]["prediction"]
                svm_conf = result["models"]["SVM"]["confidence"] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color:#16a34a; margin:0;">⚡ Support Vector Machine</h4>
                    <p style="font-size:0.85rem; color:#64748b; margin-bottom:8px;">RBF Kernel Hyperplane Separation</p>
                    <h3 style="margin:5px 0;">{svm_pred}</h3>
                    <p style="font-weight:600; color:#16a34a;">Confidence: {svm_conf:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            with c_rf:
                rf_pred = result["models"]["Random_Forest"]["prediction"]
                rf_conf = result["models"]["Random_Forest"]["confidence"] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="color:#d97706; margin:0;">🌲 Random Forest</h4>
                    <p style="font-size:0.85rem; color:#64748b; margin-bottom:8px;">150 Non-Linear Decision Trees</p>
                    <h3 style="margin:5px 0;">{rf_pred}</h3>
                    <p style="font-weight:600; color:#d97706;">Confidence: {rf_conf:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # Consensus Probability Distribution
            st.markdown("### Multimodal Probability Distribution")
            chart_data = pd.DataFrame({
                "Class": CLASSES,
                "CNN (%)": [result["models"]["CNN"]["probabilities"][c] * 100 for c in CLASSES],
                "SVM (%)": [result["models"]["SVM"]["probabilities"][c] * 100 for c in CLASSES],
                "Random Forest (%)": [result["models"]["Random_Forest"]["probabilities"][c] * 100 for c in CLASSES],
                "Consensus Hybrid (%)": [result["ensemble_probabilities"][c] * 100 for c in CLASSES]
            }).set_index("Class")

            st.bar_chart(chart_data)

            # Agreement Status & Latent Insights
            st.markdown("### Deep Latent Representation")
            c_agree, c_feat = st.columns([1, 2])

            with c_agree:
                agree_status = "Unanimous Consensus (3 / 3)" if result["unanimous_agreement"] else "Majority / Soft-Voting (2 / 3)"
                st.info(f"**Classifier Alignment:** {agree_status}")
                if pred_class == "Reversal":
                    st.error("⚠️ **Clinical Flag:** Letter reversal pattern detected. Recommended for follow-up motor-perceptual evaluation.")
                elif pred_class == "Corrected":
                    st.warning("⚠️ **Developmental Flag:** Letter retracing / correction pattern detected.")
                else:
                    st.success("✅ **Standard Pattern:** Handwriting displays standard orientation.")

            with c_feat:
                feats = np.array(result["feature_vector_sample"])
                st.caption(f"128-dimensional CNN dense layer representation (First 8 coordinates): {np.round(feats, 3)}")
                fig_f, ax_f = plt.subplots(figsize=(6, 1.8))
                ax_f.plot(range(len(feats)), feats, marker="o", color="#3b82f6", linewidth=2)
                ax_f.set_title("Latent Coordinate Activations", fontsize=10)
                ax_f.set_xlabel("Latent Index", fontsize=8)
                ax_f.set_ylabel("Activation", fontsize=8)
                ax_f.grid(alpha=0.3)
                st.pyplot(fig_f)
                plt.close(fig_f)

            # Grad-CAM Explainability
            st.markdown("### 🔥 Model Explainability (Grad-CAM)")
            st.caption("Heatmap shows which stroke regions drove the CNN decision. Red = high influence.")
            with st.spinner("Computing Grad-CAM attribution..."):
                try:
                    explanation = detector.explain(selected_image)
                    import base64 as _b64
                    import io as _io
                    _uri = explanation["gradcam_base64"]
                    _raw = _b64.b64decode(_uri.split(",", 1)[1])
                    st.image(Image.open(_io.BytesIO(_raw)), width=180,
                             caption=f"Grad-CAM: {explanation['target_class']} "
                                     f"(layer {explanation['conv_layer']}, "
                                     f"focus {explanation['focus_score']})")
                except Exception as e:
                    st.warning(f"Grad-CAM unavailable: {e}")

            # Clinical Report Generation
            st.markdown("---")
            st.subheader("Generate Clinical Screening Summary")
            patient_id = st.text_input("Patient / Subject Identifier", value="SUBJ-2026-0042")
            clinician_notes = st.text_area("Clinician / Educator Observations", value="Single handwriting sample screening.")

            report_txt = generate_clinical_report(result, patient_id=patient_id, notes=clinician_notes)
            st.download_button(
                label="📥 Download Clinical Screening Report (.txt)",
                data=report_txt,
                file_name=f"Dyslexia_Screening_{patient_id}.txt",
                mime="text/plain"
            )


# ==========================================
# PAGE 2: BATCH SCREENING & COHORT REPORT
# ==========================================
elif menu == "📁 Batch Screening & Cohort Report":
    st.markdown('<div class="main-header">📁 Cohort Batch Screening & Average Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload multiple handwriting samples to perform cohort-level triage and compute session averages.</div>', unsafe_allow_html=True)

    if detector is None:
        st.warning("⚠️ Trained models are currently not loaded.")
    else:
        uploaded_files = st.file_uploader(
            "Upload multiple handwriting images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if uploaded_files:
            st.write(f"Loaded **{len(uploaded_files)}** files for analysis.")
            if st.button("🚀 Process & Calculate Averaged Results", type="primary"):
                imgs = [Image.open(f) for f in uploaded_files]
                fnames = [f.name for f in uploaded_files]

                with st.spinner("Processing batch and calculating averages..."):
                    batch_res = detector.predict_batch(imgs, fnames)

                st.success("Batch screening and averaging complete!")

                # Cohort Summary Metrics
                c1, c2, c3, c4 = st.columns(4)
                total = batch_res["total_images"]
                c1.metric("Total Samples", total)
                c2.metric("Reversals (Dyslexic)", f"{batch_res['reversal_count']} ({batch_res['reversal_rate']:.1f}%)")
                c3.metric("Corrected (Overwriting)", f"{batch_res['corrected_count']} ({batch_res['corrected_rate']:.1f}%)")
                c4.metric("Normal (Standard)", f"{batch_res['normal_count']} ({batch_res['normal_rate']:.1f}%)")

                # Average Probabilities
                st.markdown("### 📊 Average Class Probabilities Across Cohort")
                df_avg_prob = pd.DataFrame({
                    "Class": CLASSES,
                    "Average Probability (%)": [batch_res["average_probabilities"][c] * 100 for c in CLASSES]
                }).set_index("Class")
                st.bar_chart(df_avg_prob)

                # Individual Table
                rows = []
                for idx, r in enumerate(batch_res["individual_results"]):
                    rows.append({
                        "Filename": r["filename"],
                        "Consensus Prediction": r["prediction"],
                        "Confidence": r["confidence_percentage"],
                        "Risk Level": r["risk_level"],
                        "CNN": r["models"]["CNN"]["prediction"],
                        "SVM": r["models"]["SVM"]["prediction"],
                        "RF": r["models"]["Random_Forest"]["prediction"],
                        "Unanimous": "Yes" if r["unanimous_agreement"] else "No"
                    })

                df_batch = pd.DataFrame(rows)
                st.dataframe(df_batch, use_container_width=True)

                csv = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Batch Results (CSV)",
                    data=csv,
                    file_name="batch_dyslexia_screening_results.csv",
                    mime="text/csv"
                )


# ==========================================
# PAGE 3: BENCHMARK & ANALYTICS
# ==========================================
elif menu == "📊 Benchmark & Analytics":
    st.markdown('<div class="main-header">📊 Model Benchmark & Analytics (Full 208,372 Images)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation across ALL 56,723 unseen test images comparing CNN, SVM, Random Forest, and Hybrid Ensemble.</div>', unsafe_allow_html=True)

    csv_path = RESULTS_DIR / "model_comparison_table.csv"
    if csv_path.exists():
        df_bench = pd.read_csv(csv_path)
        st.subheader("Official Full-Scale Benchmark Matrix")
        st.dataframe(df_bench, use_container_width=True)

        best_row = df_bench.loc[df_bench["Accuracy"].idxmax()]
        st.success(f"🏆 **Top Performing Model:** {best_row['Model']} with **{best_row['Accuracy']}% Accuracy** and **{best_row['Macro F1']}% Macro F1**.")
    else:
        st.info("Benchmark summary table loading...")

    st.markdown("---")
    st.subheader("Analytical Visualizations")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("**1. Confusion Matrices Across All 56,723 Test Images**")
        cm_img_path = RESULTS_DIR / "confusion_matrices.png"
        if cm_img_path.exists():
            st.image(str(cm_img_path), use_container_width=True)

    with col_v2:
        st.markdown("**2. Multiclass ROC Curves**")
        roc_img_path = RESULTS_DIR / "roc_curves.png"
        if roc_img_path.exists():
            st.image(str(roc_img_path), use_container_width=True)

    col_v3, col_v4 = st.columns(2)
    with col_v3:
        st.markdown("**3. Comparative Metrics Bar Chart**")
        comp_img_path = RESULTS_DIR / "model_comparison.png"
        if comp_img_path.exists():
            st.image(str(comp_img_path), use_container_width=True)

    with col_v4:
        st.markdown("**4. 2D PCA Latent Space Separation**")
        pca_img_path = RESULTS_DIR / "feature_tsne.png"
        if pca_img_path.exists():
            st.image(str(pca_img_path), use_container_width=True)


# ==========================================
# PAGE 4: DYSLEXIA HANDWRITING GUIDE
# ==========================================
elif menu == "📖 Clinical & Dyslexia Guide":
    st.markdown('<div class="main-header">📖 Clinical Guide: Handwriting & Dyslexia</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Understanding motor-perceptual handwriting patterns, reversals, and compensatory mechanisms.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Key Handwriting Patterns
        1. **Reversals (Mirror Writing)**:
           - Common in early childhood, but persistent reversals beyond age 7–8 strongly correlate with dyslexia.
           - Classic kinetic confusions: `b` ↔ `d`, `p` ↔ `q`, `m` ↔ `w`, `2` ↔ `5`, `s` ↔ `z`.
           - Inversion of vertical or horizontal ascenders/descenders.
        
        2. **Corrected / Overwritten Strokes**:
           - Individuals with dyslexia frequently notice hesitation or directional errors midway through character production.
           - This leads to repeated overwriting, multi-stroke retracing, and irregular ink density.
        
        3. **Standard / Normal Handwriting**:
           - Fluid stroke directionality, consistent slant, and standard orientation relative to the writing baseline.
        """)

    with col2:
        st.markdown("""
        ### Why a Hybrid AI Approach with Multi-Sample Averaging?
        - **Multi-Sample Averaging**:
          A single letter can occasionally be ambiguous due to individual handwriting style. Evaluating multiple characters written by the same subject and **averaging the individual results** drastically reduces false positives and provides an accurate, reliable dyslexia risk profile.
        
        - **Deep Feature Representation (CNN)**:
          Convolutional layers automatically capture spatial hierarchies, stroke curvature, and topological orientation without handcrafted feature engineering.
        
        - **Support Vector Machine (SVM)**:
          SVM maximizes the decision boundary margin in the 128-dimensional embedding space, providing robust generalization even on edge cases.
        
        - **Random Forest (RF)**:
          An ensemble of decorrelated decision trees reduces variance and captures complex non-linear feature interactions.
        """)

    st.markdown("---")
    st.info(
        "💡 **Clinical Context:** This automated screening system is intended to provide rapid, accessible, "
        "low-cost preliminary triage for educators and clinicians. Positive screening results should prompt "
        "comprehensive multi-tiered psycho-educational evaluation."
    )
