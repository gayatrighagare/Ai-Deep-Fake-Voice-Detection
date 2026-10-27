# 🎙️ AI-Based Deepfake Voice Detection System

> **College Capstone & Academic Project**  
> An end-to-end Machine Learning and Audio Forensics system designed to detect AI-generated voice clones, neural vocoder artifacts, and synthetic speech manipulation using **Librosa**, **TensorFlow/Keras (2D-CNN)**, and **Streamlit**.

---

## 📌 1. Project Abstract & Objectives

With the rise of generative AI voice cloning algorithms (e.g., ElevenLabs, VITS, HiFi-GAN, WaveNet, Tacotron2), malicious actors can synthesize realistic human voices with only a few seconds of reference audio. This introduces severe threats to biometric voice authentication, financial authorization, and social engineering defenses.

The **AI-Based Deepfake Voice Detection System** solves this challenge by analyzing acoustic anomalies and spectral inconsistencies embedded in audio recordings. Rather than relying on superficial heuristic checks, the system extracts high-resolution **Log-Mel Spectrograms** and passes them through a **Deep 2D Convolutional Neural Network (CNN)** trained to detect vocoder artifacts (such as high-frequency phase aliasing, unnatural formant transitions, and robotic pitch quantization).

### Key Features
- **Modern Interactive Dashboard:** Built with Streamlit, providing an academic and forensic-grade interface.
- **Audio Validation & Preprocessing:** Automatic file format validation, max size checks (25 MB), empty/silent audio detection, silence trimming, and 16 kHz resampling via Librosa.
- **In-Browser Audio Playback:** Listen to genuine and synthetic speech samples directly in the web UI.
- **Genuine Deep Learning Model:** Deep 2D-CNN with progressive convolution, batch normalization, spatial pooling, and dropout regularization.
- **Probabilistic Confidence Score:** Clearly distinguished from an absolute authenticity guarantee, accompanied by an acoustic forensic explanation and disclaimer.
- **Multi-Faceted Acoustic Visualizations:** Time-domain waveform, Log-Mel Spectrogram, and 40 Mel-Frequency Cepstral Coefficients (MFCCs).
- **Academic Evaluation Suite:** Confusion Matrix heatmap, accuracy, precision, recall, F1-score, and ROC-AUC curves.
- **Audit Logging & CSV Export:** Session history tracking with single-click CSV export for reporting.

---

## 🏛️ 2. Project Architecture & Pipeline

```
+-------------------------------------------------------------------------+
|                              AUDIO INPUT                                |
|           (.wav / .mp3 / .ogg / .flac up to 25 MB)                      |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                  AUDIO PREPROCESSING (Librosa & NumPy)                  |
|  - Validation: Format, File Size, Empty/Silence Check                   |
|  - Normalization: Resample to 16,000 Hz (Standard Speech Sampling)     |
|  - Silence Trimming: librosa.effects.trim(top_db=25)                    |
|  - Fixed Duration: Standardize to 3.0 seconds (48,000 samples)          |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    ACOUSTIC FEATURE EXTRACTION                          |
|  - Log-Mel Spectrogram: (128 Mel frequency bins x 128 time steps)       |
|  - 40 Mel-Frequency Cepstral Coefficients (MFCCs)                       |
|  - Spectral Summary: Centroid, Rolloff, Zero-Crossing Rate              |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                DEEP 2D CONVOLUTIONAL NEURAL NETWORK                     |
|  - Conv2D (32 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout      |
|  - Conv2D (64 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout      |
|  - Conv2D (128 filters, 3x3) + BatchNorm + ReLU + MaxPool + Dropout     |
|  - Conv2D (128 filters, 3x3) + BatchNorm + ReLU + GlobalAvgPooling2D    |
|  - Dense(128, ReLU) + Dense(64, ReLU) + Dense(1, Sigmoid)               |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    INFERENCE & VERDICT INTERPRETATION                   |
|  - Output Probability: P(Fake) in [0.0, 1.0]                            |
|  - Verdict: REAL (Human Voice) vs. FAKE (AI Synthesized / Clone)        |
|  - Confidence Score (%) & Risk Level Assessment                         |
|  - Forensic Disclaimer (Probabilistic Assessment vs. Guarantee)         |
+------------------------------------+------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         STREAMLIT WEB DASHBOARD                         |
|  - Tab 1: Audio Analysis, Playback, Waveform & Spectrograms             |
|  - Tab 2: Model Performance, Confusion Matrix & Metrics                 |
|  - Tab 3: Detection History Table & Downloadable CSV Audit              |
|  - Tab 4: Dataset Setup, Public Benchmarks & Documentation              |
+-------------------------------------------------------------------------+
```

---

## 📁 3. Project Structure

```
Deepfake_Voice_Detection/
├── app.py                     # Streamlit web application dashboard
├── train.py                   # Standalone CNN model training & evaluation script
├── evaluate.py                # Standalone benchmark evaluation script
├── generate_dataset.py        # Dataset synthesis & demo sample generator
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive documentation
├── src/
│   ├── __init__.py            # Package initialization
│   ├── audio_processor.py     # Librosa loading, validation, Mel-spectrogram & MFCC
│   ├── model.py               # Keras 2D-CNN architecture, compilation & metrics
│   ├── detector.py            # End-to-end inference engine & confidence scorer
│   ├── visualizer.py          # Matplotlib waveforms, spectrograms & confusion matrix
│   └── utils.py               # History logger & CSV export utilities
├── data/
│   ├── real/                  # Genuine human voice audio recordings (Class 0)
│   ├── fake/                  # Synthetic/deepfake audio recordings (Class 1)
│   └── demo_samples/          # Pre-packaged audio clips for 1-click UI testing
└── models/
    ├── deepfake_detector.keras # Trained Convolutional Neural Network weights
    ├── metrics.json           # Empirical test evaluation metrics & parameters
    ├── confusion_matrix.png   # Test set confusion matrix visualization
    └── training_history.png   # Accuracy and loss convergence curves
```

---

## ⚙️ 4. Installation & Setup Instructions

### Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux.
- **Python Version:** Python 3.10 is strongly recommended for full TensorFlow and Librosa compatibility.

### Step 1: Open Terminal in VS Code
Open VS Code, press `Ctrl + \`` (or go to **Terminal -> New Terminal**), and ensure you are in the project folder:
```powershell
cd c:\Users\SHREE\OneDrive\Documents\Deepfake_Voice_Detection
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Create a Python 3.10 virtual environment
py -3.10 -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```
*(If you encounter execution policy restrictions in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🗄️ 5. Dataset Setup & Preparation

The model requires two folders in the `data/` directory:
- `data/real/` containing `.wav` or `.mp3` files of authentic human speakers.
- `data/fake/` containing `.wav` or `.mp3` files of AI-generated or cloned voices.

### Option A: Generate Benchmark Dataset Automatically
To prepare a balanced benchmark dataset and ready-to-test demo samples immediately, run:
```powershell
python generate_dataset.py
```
This script models acoustic vocal-tract formants for genuine human voices (with physiological jitter and microprosody) and generates synthetic deepfake samples with vocoder phase artifacts and Windows SAPI neural TTS.

### Option B: Use Public Academic Datasets
For advanced college capstone research, you can download public deepfake audio datasets:
1. **ASVspoof 2019 / 2021 (Logical Access):** The international standard benchmark containing genuine human speech and spoofed samples from 17 TTS/VC algorithms ([asvspoof.org](https://www.asvspoof.org/)).
2. **In-the-Wild Audio Deepfake Dataset:** Real-world deepfakes of public figures and politicians scraped from social media.
3. **WaveFake:** Deepfakes produced by 6 state-of-the-art vocoders (MelGAN, HiFi-GAN, WaveGlow, etc.).
4. **LibriSpeech / LJSpeech:** High-quality public domain human speech recordings for genuine voice baselines.

Simply place your `.wav` files into `data/real` and `data/fake` respectively.

---

## 🏋️ 6. Model Training & Evaluation

To train the 2D Convolutional Neural Network on the dataset:

```powershell
python train.py --epochs 25 --batch_size 16 --lr 0.0005
```

### What Happens During Training?
1. Audio files are validated, normalized, and converted to `(128, 128, 1)` Log-Mel Spectrograms.
2. Dataset is split into:
   - **70% Training:** Used for gradient descent optimization.
   - **15% Validation:** Monitored via Early Stopping and Learning Rate Decay.
   - **15% Held-Out Test:** Evaluated strictly once after training completes.
3. The best model checkpoint is saved to `models/deepfake_detector.keras`.
4. Empirical metrics (Accuracy, Precision, Recall, F1, ROC-AUC) are saved to `models/metrics.json`.
5. High-resolution diagnostic plots (`confusion_matrix.png`, `training_history.png`) are generated.

### Running Standalone Evaluation
To evaluate any saved model checkpoint on test data:
```powershell
python evaluate.py --model models/deepfake_detector.keras --data_dir data
```

---

## 🚀 7. Running the Web Application

Launch the interactive Streamlit dashboard:

```powershell
streamlit run app.py
```

Streamlit will automatically open your default web browser at `http://localhost:8501`.

### Step-by-Step Testing Guide in the UI:
1. **Sidebar Demo Selector:** In the left sidebar under *Quick Demo Samples*, choose `demo_real_human_1.wav` or `demo_fake_deepfake_1.wav`.
2. **Audio Preview:** Click the play button on the audio player to listen to the sample before analysis.
3. **Run Analysis:** Click **🚀 Run Deepfake Analysis**.
4. **Review Results:**
   - **Verdict Badge:** Displays `✅ GENUINE HUMAN VOICE / REAL` or `🚨 AI-GENERATED / FAKE`.
   - **Confidence Metric:** Displays statistical certainty (e.g. 98.4%).
   - **Acoustic Observations:** Highlights specific acoustic clues (formant naturalness, high-frequency vocoder ringing).
   - **Disclaimer:** Read the academic disclaimer distinguishing probability from legal authenticity.
   - **Visualizations:** Switch between Time-Domain Waveform, Log-Mel Spectrogram, and MFCC Heatmap.
5. **Explore Other Tabs:**
   - Check **Model Architecture & Evaluation** to see the test set confusion matrix and performance curves.
   - Check **Detection History & Export** to view audit logs and click **Download CSV Report**.

---

## 🔬 8. Theoretical Insights & Viva / Presentation Q&A

### Q1: Why use Log-Mel Spectrograms instead of raw 1D waveforms?
> **Answer:** Raw audio waveforms consist of 16,000 to 44,100 samples per second, making long-range temporal modeling computationally prohibitive. Log-Mel Spectrograms map audio into time-frequency representations aligned with the human auditory system (the Mel scale). Neural vocoders (such as HiFi-GAN or WaveNet) leave distinct spatial texture artifacts in high-frequency Mel bands, which 2D-CNNs can detect through convolutional pattern recognition.

### Q2: What are vocoder artifacts in AI-synthesized voices?
> **Answer:** Generative neural vocoders synthesize waveforms from acoustic intermediate representations. This synthesis frequently introduces:
> - **Aperiodic high-frequency phase inconsistencies** (metallic or buzzing timbre).
> - **Oversmoothed harmonic contours** (loss of biological micro-fluctuations).
> - **Pitch quantization** (unnaturally stable or robotic fundamental frequency $F_0$).

### Q3: Why is a confidence score not an absolute guarantee of authenticity?
> **Answer:** Deep learning classifiers compute a posterior probability $P(\text{Fake}|\text{Features})$ based on training distribution patterns. High-bitrate audio compression (e.g., telephone codecs or low-bitrate MP3), background noise, acoustic reverberation, or novel unencountered generative architectures can shift acoustic features, causing false positives or negatives. Therefore, confidence scores must be interpreted as probabilistic indicators rather than definitive forensic certainty.

---

## 📜 9. Technology Stack

- **Deep Learning Framework:** TensorFlow 2.16 & Keras
- **Acoustic Signal Processing:** Librosa & SoundFile
- **Numerical Computation:** NumPy & SciPy
- **Web UI & Visualization:** Streamlit, Matplotlib, Seaborn
- **Statistical Evaluation:** Scikit-learn, Pandas

---

## 👨‍💻 10. Author & Academic Attribution
- **Project:** AI-Based Deepfake Voice Detection System
- **Purpose:** College Academic Demonstration & Capstone Submission
- **License:** Open Source for Academic & Educational Use
