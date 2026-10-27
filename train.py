"""Model Training Script for Deepfake Voice Detection.
Extracts Log-Mel Spectrogram features from genuine and deepfake audio,
trains a Deep 2D Convolutional Neural Network (CNN), evaluates on test set,
and persists model weights, evaluation metrics, and diagnostic plots.
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime
from typing import Tuple, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio_processor import (
    load_and_preprocess_audio,
    extract_mel_spectrogram,
    AudioValidationError,
    TARGET_SAMPLE_RATE,
    DEFAULT_AUDIO_DURATION,
)
from src.model import build_deepfake_detector, save_metrics_summary
from src.visualizer import plot_confusion_matrix, plot_training_history


def load_dataset_from_directories(
    data_dir: str = "data",
    max_samples_per_class: int = 500
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads audio from data/real (class 0) and data/fake (class 1),

    extracts Log-Mel Spectrograms, and returns (X, y, filenames).
    """
    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise FileNotFoundError(
            f"Dataset directories not found. Expected '{real_dir}' and '{fake_dir}'. "
            "Please run `python generate_dataset.py` or populate folders with audio files."
        )

    extensions = ("*.wav", "*.mp3", "*.ogg", "*.flac")
    real_files = []
    fake_files = []
    for ext in extensions:
        real_files.extend(glob.glob(os.path.join(real_dir, ext)))
        fake_files.extend(glob.glob(os.path.join(fake_dir, ext)))

    real_files = sorted(real_files)[:max_samples_per_class]
    fake_files = sorted(fake_files)[:max_samples_per_class]

    print(f"[*] Found {len(real_files)} REAL audio files and {len(fake_files)} FAKE audio files.")

    if len(real_files) == 0 or len(fake_files) == 0:
        raise ValueError(
            f"Insufficient samples: {len(real_files)} real and {len(fake_files)} fake. "
            "Both classes must contain at least 5 audio files for training."
        )

    features = []
    labels = []
    processed_files = []

    print("[*] Extracting Log-Mel Spectrogram features for REAL audio...")
    for fpath in real_files:
        try:
            y, sr, _ = load_and_preprocess_audio(fpath, filename=os.path.basename(fpath))
            spec = extract_mel_spectrogram(y, sr=sr)
            features.append(spec)
            labels.append(0)  # 0 = REAL
            processed_files.append(fpath)
        except AudioValidationError as e:
            print(f"  [Skipping] {os.path.basename(fpath)}: {e}")
        except Exception as e:
            print(f"  [Error] {os.path.basename(fpath)}: {e}")

    print("[*] Extracting Log-Mel Spectrogram features for FAKE audio...")
    for fpath in fake_files:
        try:
            y, sr, _ = load_and_preprocess_audio(fpath, filename=os.path.basename(fpath))
            spec = extract_mel_spectrogram(y, sr=sr)
            features.append(spec)
            labels.append(1)  # 1 = FAKE
            processed_files.append(fpath)
        except AudioValidationError as e:
            print(f"  [Skipping] {os.path.basename(fpath)}: {e}")
        except Exception as e:
            print(f"  [Error] {os.path.basename(fpath)}: {e}")

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    print(f"[+] Successfully extracted features for {len(X)} audio samples. Feature tensor shape: {X.shape}")
    return X, y, processed_files


def train(
    data_dir: str = "data",
    models_dir: str = "models",
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    test_size: float = 0.20,
    val_size: float = 0.15
):
    """Executes the full training and evaluation pipeline."""
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "deepfake_detector.keras")
    metrics_save_path = os.path.join(models_dir, "metrics.json")
    history_plot_path = os.path.join(models_dir, "training_history.png")
    cm_plot_path = os.path.join(models_dir, "confusion_matrix.png")

    print("=" * 65)
    print("   AI-BASED DEEPFAKE VOICE DETECTION SYSTEM - TRAINING PIPELINE")
    print("=" * 65)

    # 1. Load and process dataset
    X, y, filenames = load_dataset_from_directories(data_dir)

    # 2. Stratified train/val/test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    val_ratio_adjusted = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adjusted, random_state=42, stratify=y_train_val
    )

    print(f"[*] Dataset partitions: Train={len(X_train)}, Validation={len(X_val)}, Test={len(X_test)}")
    print(f"    Train labels: Real={np.sum(y_train == 0)}, Fake={np.sum(y_train == 1)}")
    print(f"    Val labels:   Real={np.sum(y_val == 0)}, Fake={np.sum(y_val == 1)}")
    print(f"    Test labels:  Real={np.sum(y_test == 0)}, Fake={np.sum(y_test == 1)}")

    # 3. Build model architecture
    print("\n[*] Constructing 2D Convolutional Neural Network...")
    model = build_deepfake_detector(input_shape=X_train.shape[1:], learning_rate=learning_rate)
    model.summary()

    # 4. Callbacks: monitor val_loss for best generalized calibration
    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=6,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 5. Train Model
    print("\n[*] Commencing Model Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Load best checkpoint
    if os.path.exists(model_save_path):
        model = tf.keras.models.load_model(model_save_path)

    # 6. Evaluate on Unseen Test Partition
    print("\n" + "=" * 65)
    print("           EVALUATING MODEL ON HELD-OUT TEST DATA")
    print("=" * 65)

    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.50).astype(int)

    test_acc = float(accuracy_score(y_test, y_pred))
    test_prec = float(precision_score(y_test, y_pred, zero_division=0))
    test_rec = float(recall_score(y_test, y_pred, zero_division=0))
    test_f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred)
    try:
        test_auc = float(roc_auc_score(y_test, y_pred_probs))
    except Exception:
        test_auc = 0.0

    print(f"Test Accuracy:  {test_acc * 100:.2f}%")
    print(f"Test Precision: {test_prec * 100:.2f}%")
    print(f"Test Recall:    {test_rec * 100:.2f}%")
    print(f"Test F1-Score:  {test_f1 * 100:.2f}%")
    print(f"Test ROC-AUC:   {test_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["REAL (0)", "FAKE (1)"]))

    # 7. Generate and save diagnostic visualizations
    print("[*] Generating evaluation plots...")
    cm_fig = plot_confusion_matrix(cm, class_names=["REAL", "FAKE"], title="Deepfake Detector Test Confusion Matrix")
    cm_fig.savefig(cm_plot_path, dpi=200, bbox_inches="tight")
    plt.close(cm_fig)

    hist_fig = plot_training_history(history.history)
    hist_fig.savefig(history_plot_path, dpi=200, bbox_inches="tight")
    plt.close(hist_fig)

    # 8. Save Metrics Summary for Streamlit Dashboard
    metrics_summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_architecture": "Deep 2D-CNN (Convolutional Neural Network)",
        "input_shape": list(X_train.shape[1:]),
        "total_parameters": int(model.count_params()),
        "dataset_summary": {
            "total_samples": len(X),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "real_samples": int(np.sum(y == 0)),
            "fake_samples": int(np.sum(y == 1)),
        },
        "test_metrics": {
            "accuracy": round(test_acc * 100, 2),
            "precision": round(test_prec * 100, 2),
            "recall": round(test_rec * 100, 2),
            "f1_score": round(test_f1 * 100, 2),
            "roc_auc": round(test_auc, 4),
        },
        "confusion_matrix": {
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
            "matrix": cm.tolist()
        },
        "training_epochs_completed": len(history.history["loss"]),
        "model_weights_path": model_save_path
    }

    save_metrics_summary(metrics_summary, metrics_save_path)
    print(f"\n[+] Model saved to: {model_save_path}")
    print(f"[+] Metrics saved to: {metrics_save_path}")
    print(f"[+] Diagnostic plots saved to: {history_plot_path} and {cm_plot_path}")
    print("[+] Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Voice Detector CNN")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing real/ and fake/ subfolders")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save model checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
