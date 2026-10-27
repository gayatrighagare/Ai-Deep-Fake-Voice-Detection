"""AI-Based Deepfake Voice Detection System
Interactive Streamlit Dashboard.

Features:
1. Audio upload with file size & format validation.
2. Built-in audio playback before and after processing.
3. Feature extraction (Mel-Spectrogram, MFCCs, Spectral Centroid/Rolloff).
4. Prediction from trained Deep 2D-CNN with probability score and authenticity disclaimer.
5. Time-domain waveform, Log-Mel spectrogram, and MFCC visualizations.
6. Comprehensive model evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix).
7. Detection history table with CSV export.
8. Interactive dataset guide for ASVspoof, In-the-Wild, and training pipeline.
"""

import os
import sys
import io
import time
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import (
    load_and_preprocess_audio,
    extract_mel_spectrogram,
    extract_mfcc,
    extract_spectral_features,
    AudioValidationError,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    TARGET_SAMPLE_RATE,
    DEFAULT_AUDIO_DURATION
)
from src.detector import VoiceDetector
from src.model import load_metrics_summary
from src.visualizer import (
    plot_waveform,
    plot_mel_spectrogram,
    plot_mfcc,
    plot_confusion_matrix,
    plot_training_history
)
from src.utils import create_history_record, export_history_to_csv, history_to_dataframe

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Deepfake Voice Detection System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished college-level UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 24px 28px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .main-header p {
        color: #e0e7ff;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }
    .badge-fake {
        display: inline-block;
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 8px 18px;
        font-size: 1.25rem;
        font-weight: 800;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(239, 68, 68, 0.3);
        margin: 8px 0;
    }
    .badge-real {
        display: inline-block;
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: white;
        padding: 8px 18px;
        font-size: 1.25rem;
        font-weight: 800;
        border-radius: 8px;
        box-shadow: 0 3px 10px rgba(16, 185, 129, 0.3);
        margin: 8px 0;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        color: #92400e;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-top: 14px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

MODEL_PATH = "models/deepfake_detector.keras"
METRICS_PATH = "models/metrics.json"

@st.cache_resource
def get_detector(model_file: str):
    """Loads and caches the inference detector."""
    return VoiceDetector(model_path=model_file)

detector = get_detector(MODEL_PATH)


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/sound-recording-copyright.png", width=64)
    st.title("System Control")
    st.markdown("---")

    # Model status indicator
    if detector.is_model_loaded:
        st.success("🟢 Model Status: **LOADED & ACTIVE**")
        st.caption(f"Weights: `{MODEL_PATH}`")
    else:
        st.error("🔴 Model Status: **NOT FOUND**")
        st.warning("Train a model using `python train.py` to enable predictions.")

    st.markdown("---")
    st.subheader("⚙️ Detection Parameters")
    decision_threshold = st.slider(
        "Classification Threshold (Fake)",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Audio with P(Fake) >= Threshold will be classified as AI-Generated."
    )

    st.markdown("---")
    st.subheader("💡 Quick Demo Samples")
    st.markdown("Test the detector with pre-packaged audio clips:")

    demo_dir = "data/demo_samples"
    demo_options = ["None (Upload Own Audio)"]
    if os.path.exists(demo_dir):
        available_demos = [f for f in os.listdir(demo_dir) if f.endswith((".wav", ".mp3"))]
        demo_options.extend(available_demos)

    selected_demo = st.selectbox("Select Benchmark Sample", options=demo_options)

    st.markdown("---")
    st.caption("College Capstone Project: **AI Deepfake Voice Detection**")
    st.caption("Architecture: **2D Convolutional Neural Network (CNN)**")
    st.caption("Features: **128-band Mel Spectrogram (Librosa)**")


# ---------------------------------------------------------
# Main Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🎙️ AI-Based Deepfake Voice Detection System</h1>
    <p>A deep-learning acoustic analysis platform designed to distinguish authentic human speech 
    from generative AI-synthesized voices, neural vocoder clones, and acoustic manipulations.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab_analysis, tab_evaluation, tab_history, tab_guide = st.tabs([
    "🔬 Audio Analysis & Prediction",
    "📊 Model Architecture & Evaluation",
    "📜 Detection History & Export",
    "📚 Dataset Setup & Training Guide"
])


# =========================================================
# TAB 1: Audio Analysis & Prediction
# =========================================================
with tab_analysis:
    col_upload, col_preview = st.columns([1.2, 1])

    with col_upload:
        st.subheader("1. Audio Source Selection")
        uploaded_file = st.file_uploader(
            "Upload speech recording for deepfake analysis",
            type=["wav", "mp3", "ogg", "flac", "m4a"],
            help=f"Supported: WAV, MP3, FLAC, OGG. Max file size: {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
        )

        audio_bytes = None
        source_filename = None

        if selected_demo != "None (Upload Own Audio)":
            demo_path = os.path.join(demo_dir, selected_demo)
            if os.path.exists(demo_path):
                with open(demo_path, "rb") as df:
                    audio_bytes = df.read()
                source_filename = selected_demo
                st.info(f"Loaded demo audio sample: `{selected_demo}`")
        elif uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            source_filename = uploaded_file.name

    with col_preview:
        st.subheader("2. Playback & Validation")
        if audio_bytes is not None and source_filename is not None:
            st.write(f"**File:** `{source_filename}` ({len(audio_bytes) / 1024:.1f} KB)")
            st.audio(audio_bytes, format="audio/wav")
        else:
            st.info("Upload an audio recording or pick a demo sample to preview and analyze.")

    analyze_clicked = st.button(
        "🚀 Run Deepfake Analysis",
        type="primary",
        disabled=(audio_bytes is None or not detector.is_model_loaded),
        use_container_width=True
    )

    if not detector.is_model_loaded:
        st.warning(
            "⚠️ No trained model weights found (`models/deepfake_detector.keras`). "
            "Please run `python train.py` in the terminal to train the model, then reload this app."
        )

    # Execution upon button click
    if analyze_clicked and audio_bytes is not None:
        try:
            with st.spinner("Decoding audio, extracting Log-Mel spectrogram, and running CNN inference..."):
                start_time = time.time()
                analysis = detector.analyze_audio(
                    file_input=io.BytesIO(audio_bytes),
                    filename=source_filename,
                    decision_threshold=decision_threshold
                )
                latency_ms = round((time.time() - start_time) * 1000, 1)

            # Log to session history
            history_record = create_history_record(source_filename, analysis)
            st.session_state["history"].insert(0, history_record)

            st.markdown("---")
            st.subheader("3. Detection Verdict & Acoustic Assessment")

            # Metrics and Verdict Row
            res_col1, res_col2, res_col3, res_col4 = st.columns([1.5, 1, 1, 1])

            with res_col1:
                st.markdown("**Classification Result:**")
                if analysis["is_fake"]:
                    st.markdown(f'<div class="badge-fake">🚨 {analysis["prediction"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-real">✅ {analysis["prediction"]}</div>', unsafe_allow_html=True)

            with res_col2:
                st.metric(
                    label="Prediction Confidence",
                    value=f"{analysis['confidence']}%",
                    delta=f"Threshold {int(decision_threshold * 100)}%"
                )

            with res_col3:
                st.metric(
                    label="Raw Model Output P(Fake)",
                    value=f"{analysis['raw_fake_prob']:.4f}",
                    help="Continuous sigmoid output between 0.0 (Genuine) and 1.0 (Synthesized)."
                )

            with res_col4:
                st.metric(
                    label="Analysis Latency",
                    value=f"{latency_ms} ms",
                    help="Preprocessing and CNN forward pass time."
                )

            # Observations & Explanations
            meta = analysis["metadata"]
            st.markdown(f"**Acoustic Risk Evaluation:** `{analysis['risk_level']}`")

            obs_col, stats_col = st.columns([1.4, 1])
            with obs_col:
                st.markdown("**Key Acoustic Observations:**")
                for obs in analysis["observations"]:
                    st.markdown(f"- {obs}")

            with stats_col:
                st.markdown("**Audio Physical Characteristics:**")
                st.write(f"- Duration: **{meta.get('original_duration_sec', 0)} seconds**")
                st.write(f"- Resampled Rate: **{meta.get('sample_rate', 16000)} Hz**")
                st.write(f"- Spectral Centroid: **{meta.get('mean_spectral_centroid_hz', 0):.1f} Hz**")
                st.write(f"- Spectral Rolloff: **{meta.get('mean_spectral_rolloff_hz', 0):.1f} Hz**")

            # Authenticity Disclaimer Callout (Required Feature)
            st.markdown(f"""
            <div class="disclaimer-box">
                <strong>⚖️ Authenticity & Confidence Disclaimer:</strong><br>
                {analysis["disclaimer"]}
            </div>
            """, unsafe_allow_html=True)

            # Visualizations (Waveform, Mel-Spectrogram, MFCC)
            st.markdown("---")
            st.subheader("4. Visual Acoustic Signal Analysis")

            v_tab1, v_tab2, v_tab3 = st.tabs([
                "📈 Time-Domain Waveform",
                "🌈 Log-Mel Spectrogram (CNN Input)",
                "📊 MFCC Heatmap (Vocal Tract Filter)"
            ])

            with v_tab1:
                fig_wave = plot_waveform(
                    analysis["audio_signal"],
                    analysis["sample_rate"],
                    title=f"Time-Domain Audio Waveform - {source_filename}",
                    is_fake=analysis["is_fake"]
                )
                st.pyplot(fig_wave)
                plt.close(fig_wave)
                st.caption("Visualizes amplitude variations over time. Natural speech displays organic envelope decay and breath transients.")

            with v_tab2:
                fig_spec = plot_mel_spectrogram(
                    analysis["mel_spectrogram"],
                    analysis["sample_rate"],
                    title=f"Log-Mel Spectrogram (128 Mel Bins x 128 Frames) - {source_filename}"
                )
                st.pyplot(fig_spec)
                plt.close(fig_spec)
                st.caption("Log-Mel Spectrogram representation fed to the 2D-CNN. Synthesized audio frequently displays horizontal aperiodic banding or unnatural high-frequency phase artifacts.")

            with v_tab3:
                fig_mfcc = plot_mfcc(
                    analysis["mfcc"],
                    title=f"40 Mel-Frequency Cepstral Coefficients (MFCC) - {source_filename}"
                )
                st.pyplot(fig_mfcc)
                plt.close(fig_mfcc)
                st.caption("MFCC coefficients capture vocal tract resonance shapes. Deepfakes often lack subtle formant resonance dynamics.")

        except AudioValidationError as e:
            st.error(f"❌ Audio Validation Error: {str(e)}")
        except Exception as e:
            st.error(f"❌ An error occurred during audio analysis: {str(e)}")


# =========================================================
# TAB 2: Model Architecture & Evaluation
# =========================================================
with tab_evaluation:
    st.subheader("Model Performance & Architecture Specifications")
    st.write(
        "This section presents the architectural specifications and empirical test set performance metrics "
        "of the trained Deepfake Voice Detection Convolutional Neural Network (CNN)."
    )

    metrics = load_metrics_summary(METRICS_PATH)

    if metrics is not None:
        test_m = metrics.get("test_metrics", {})
        ds_m = metrics.get("dataset_summary", {})

        # Top Metric Cards
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Test Accuracy", f"{test_m.get('accuracy', 0)}%")
        m_col2.metric("Precision", f"{test_m.get('precision', 0)}%")
        m_col3.metric("Recall", f"{test_m.get('recall', 0)}%")
        m_col4.metric("F1-Score", f"{test_m.get('f1_score', 0)}%")
        m_col5.metric("ROC-AUC", f"{test_m.get('roc_auc', 0)}")

        st.markdown("---")
        col_cm, col_arch = st.columns([1, 1.2])

        with col_cm:
            st.markdown("#### Test Set Confusion Matrix")
            cm_img_path = "models/confusion_matrix.png"
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, use_container_width=True)
            elif "confusion_matrix" in metrics:
                cm_arr = np.array(metrics["confusion_matrix"]["matrix"])
                fig_cm = plot_confusion_matrix(cm_arr, class_names=["REAL", "FAKE"])
                st.pyplot(fig_cm)
                plt.close(fig_cm)
            else:
                st.info("Confusion matrix plot not available.")

        with col_arch:
            st.markdown("#### Network Architecture Summary")
            st.markdown(f"- **Architecture:** `{metrics.get('model_architecture', '2D-CNN')}`")
            st.markdown(f"- **Total Learnable Parameters:** `{metrics.get('total_parameters', 0):,}`")
            st.markdown(f"- **Input Feature Tensor:** `(128 Mel Bins, 128 Time Frames, 1 Channel)`")
            st.markdown(f"- **Trained Checkpoint Path:** `{metrics.get('model_weights_path', 'models/deepfake_detector.keras')}`")
            st.markdown(f"- **Last Training Date:** `{metrics.get('timestamp', 'N/A')}`")

            st.markdown("#### Dataset Split Distribution")
            st.write(f"- Total Samples: **{ds_m.get('total_samples', 0)}**")
            st.write(f"- Training Samples: **{ds_m.get('train_samples', 0)}** (70%)")
            st.write(f"- Validation Samples: **{ds_m.get('val_samples', 0)}** (15%)")
            st.write(f"- Held-Out Test Samples: **{ds_m.get('test_samples', 0)}** (15%)")

        st.markdown("---")
        st.markdown("#### Training & Validation Learning Curves")
        hist_img_path = "models/training_history.png"
        if os.path.exists(hist_img_path):
            st.image(hist_img_path, use_container_width=True)
        else:
            st.info("Training history curve not found. Run `python train.py` to regenerate.")
    else:
        st.warning("⚠️ No model evaluation summary (`models/metrics.json`) found.")
        st.info(
            "To train the model and generate real evaluation metrics, execute:\n"
            "```bash\n"
            "python generate_dataset.py\n"
            "python train.py\n"
            "```"
        )


# =========================================================
# TAB 3: Detection History & Export
# =========================================================
with tab_history:
    st.subheader("Session Detection History & Export")
    st.write("Track and download verification results recorded during the current application session.")

    history_records = st.session_state["history"]

    if history_records:
        h_df = history_to_dataframe(history_records)

        # Overview Stats
        stat1, stat2, stat3, stat4 = st.columns(4)
        total_scanned = len(h_df)
        fake_count = int(np.sum(h_df["Verdict"] == "FAKE"))
        real_count = int(np.sum(h_df["Verdict"] == "REAL"))
        avg_conf = float(h_df["Confidence (%)"].mean())

        stat1.metric("Total Audio Analyzed", total_scanned)
        stat2.metric("Detected Deepfakes", fake_count)
        stat3.metric("Verified Genuine", real_count)
        stat4.metric("Avg Confidence", f"{avg_conf:.1f}%")

        st.markdown("---")
        st.dataframe(h_df, use_container_width=True)

        col_csv, col_clear = st.columns([1, 4])
        with col_csv:
            csv_data = export_history_to_csv(history_records)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name=f"deepfake_voice_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
        with col_clear:
            if st.button("🗑️ Clear History"):
                st.session_state["history"] = []
                st.rerun()
    else:
        st.info("No audio recordings analyzed in this session yet. Run an analysis in Tab 1 to populate the history.")


# =========================================================
# TAB 4: Dataset Setup & Training Guide
# =========================================================
with tab_guide:
    st.subheader("Academic Dataset Setup & Methodology Guide")
    st.markdown("""
    This system is built for college academic demonstration and practical deployment in synthetic speech defense.

    ### 1. Project Workflow & Architecture
    ```
    ┌─────────────────────────┐
    │  Raw Audio (.wav/.mp3)  │
    └───────────┬─────────────┘
                ▼
    ┌─────────────────────────┐
    │  Validation & Trimming  │  (Check silence, length, resample to 16 kHz)
    └───────────┬─────────────┘
                ▼
    ┌─────────────────────────┐
    │ Log-Mel Feature Extract │  (128 Mel Bins x 128 Time Frames)
    └───────────┬─────────────┘
                ▼
    ┌─────────────────────────┐
    │   2D-CNN Architecture   │  (4 Conv Blocks + BatchNorm + GlobalAvgPool)
    └───────────┬─────────────┘
                ▼
    ┌─────────────────────────┐
    │ Binary Decision Engine  │  (Sigmoid Probability P(Fake) & Confidence)
    └─────────────────────────┘
    ```

    ### 2. Public Datasets for College Projects & Benchmarks
    To train or evaluate on large-scale standard benchmarks, students and researchers can utilize:
    
    1. **ASVspoof 2019 / 2021 (Logical Access - LA):**
       - The academic benchmark for voice spoofing counter-measures.
       - Contains genuine human speech and spoofed speech generated by 17 different TTS and VC algorithms.
       - [Download from ASVspoof.org](https://www.asvspoof.org/)
    
    2. **In-the-Wild Audio Deepfake Dataset:**
       - Real-world deepfake speech collected from social media, celebrities, and public figures.
       - Captures diverse acoustic noise and real compression artifacts.
    
    3. **WaveFake Dataset:**
       - Speech samples synthesized by 6 different state-of-the-art vocoders: MelGAN, Parallel WaveGAN, Multi-band MelGAN, HiFi-GAN, WaveGlow, and Full-band MelGAN.
       - [WaveFake on GitHub / Zenodo](https://github.com/joonson/wavefake)
    
    4. **LibriSpeech / LJSpeech (Genuine Audio):**
       - Open-source high quality clean human speech recordings ideal for genuine human voice baselines.

    ### 3. Step-by-Step Training Instructions
    To train or retrain the model locally:
    
    ```bash
    # Step 1: Generate or prepare dataset
    python generate_dataset.py

    # Step 2: Train the 2D-CNN model and evaluate on the test split
    python train.py --epochs 25 --batch_size 16 --lr 0.0005

    # Step 3: Run standalone evaluation on custom folder
    python evaluate.py --model models/deepfake_detector.keras --data_dir data

    # Step 4: Launch the Streamlit dashboard
    streamlit run app.py
    ```
    """)
