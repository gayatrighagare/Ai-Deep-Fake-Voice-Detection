"""Standalone Evaluation & Performance Verification Script.
Evaluates the trained Deepfake Detector CNN on an audio directory,
producing accuracy, precision, recall, F1, and confusion matrix.
"""

import os
import sys
import glob
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import load_and_preprocess_audio, extract_mel_spectrogram
from src.model import load_detector_model
from src.visualizer import plot_confusion_matrix


def evaluate_directory(model_path: str = "models/deepfake_detector.keras", data_dir: str = "data"):
    """Evaluates the trained model on real/ and fake/ subfolders of data_dir."""
    if not os.path.exists(model_path):
        print(f"[Error] Model weights file not found at: {model_path}")
        print("Please train the model first by running `python train.py`.")
        return

    print(f"[*] Loading model from '{model_path}'...")
    model = load_detector_model(model_path)
    if model is None:
        print("[Error] Failed to load model weights.")
        return

    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"[Error] Test folders '{real_dir}' or '{fake_dir}' not found.")
        return

    exts = ("*.wav", "*.mp3", "*.flac")
    real_files = []
    fake_files = []
    for ext in exts:
        real_files.extend(glob.glob(os.path.join(real_dir, ext)))
        fake_files.extend(glob.glob(os.path.join(fake_dir, ext)))

    print(f"[*] Found {len(real_files)} REAL audio files and {len(fake_files)} FAKE audio files.")

    all_files = [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
    if not all_files:
        print("[Error] No audio files found to evaluate.")
        return

    X_list = []
    y_true = []
    valid_files = []

    for fpath, label in all_files:
        try:
            y, sr, _ = load_and_preprocess_audio(fpath, filename=os.path.basename(fpath))
            spec = extract_mel_spectrogram(y, sr=sr)
            X_list.append(spec)
            y_true.append(label)
            valid_files.append((fpath, label))
        except Exception as e:
            print(f"  [Skipped] {os.path.basename(fpath)}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y_true = np.array(y_true, dtype=np.int32)

    print(f"[*] Running inference on {len(X)} samples...")
    raw_preds = model.predict(X, verbose=0).flatten()
    y_pred = (raw_preds >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("           MODEL EVALUATION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total Evaluated Samples: {len(y_true)}")
    print(f"Overall Accuracy:        {acc * 100:.2f}%")
    print(f"Precision (Fake class):  {prec * 100:.2f}%")
    print(f"Recall (Fake class):     {rec * 100:.2f}%")
    print(f"F1-Score:                {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["REAL", "FAKE"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Voice Detector")
    parser.add_argument("--model", type=str, default="models/deepfake_detector.keras", help="Path to trained model")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing real/ and fake/ subfolders")
    args = parser.parse_args()

    evaluate_directory(model_path=args.model, data_dir=args.data_dir)
