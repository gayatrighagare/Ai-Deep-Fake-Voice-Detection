"""Audio and Evaluation Visualizations using Matplotlib and Seaborn.
Provides publication-quality plots for audio waveforms, Mel-spectrograms,
MFCC heatmaps, confusion matrices, and training history curves.
"""

from typing import Optional, List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/Streamlit usage
import matplotlib.pyplot as plt
import seaborn as sns


# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {
    "primary": "#4F46E5",
    "real": "#10B981",
    "fake": "#EF4444",
    "neutral": "#64748B",
    "bg": "#0F172A",
}


def plot_waveform(
    y: np.ndarray,
    sr: int,
    title: str = "Audio Time-Domain Waveform",
    is_fake: Optional[bool] = None
) -> plt.Figure:
    """Generates a clean waveform plot showing speech envelope and amplitude."""
    fig, ax = plt.subplots(figsize=(10, 2.8), dpi=150)
    time_axis = np.linspace(0, len(y) / sr, len(y))

    color = PALETTE["fake"] if is_fake is True else (PALETTE["real"] if is_fake is False else PALETTE["primary"])
    ax.plot(time_axis, y, color=color, linewidth=0.9, alpha=0.85)
    ax.fill_between(time_axis, y, -y, color=color, alpha=0.15)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time (seconds)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    ax.set_ylim([-1.05, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_mel_spectrogram(
    mel_spec: np.ndarray,
    sr: int,
    title: str = "Log-Mel Spectrogram (Acoustic Texture & Vocoder Analysis)"
) -> plt.Figure:
    """Plots a 2D Mel-Spectrogram showing frequency distribution over time.

    Neural TTS artifacts typically exhibit unnatural high-frequency energy banding
    or oversmoothed harmonics.
    """
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=150)

    # If normalized mel_spec is passed (128, 128)
    im = ax.imshow(
        mel_spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="bilinear"
    )

    cbar = fig.colorbar(im, ax=ax, format="%+2.1f")
    cbar.set_label("Normalized Energy / Intensity", fontsize=9)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time Frames", fontsize=10)
    ax.set_ylabel("Mel Frequency Bins (0 - 128)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_mfcc(
    mfcc: np.ndarray,
    title: str = "Mel-Frequency Cepstral Coefficients (MFCCs)"
) -> plt.Figure:
    """Plots MFCC heatmap showing vocal tract acoustic filter characteristics."""
    fig, ax = plt.subplots(figsize=(10, 3.0), dpi=150)

    im = ax.imshow(
        mfcc,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Coefficient Value", fontsize=9)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Time Frames", fontsize=10)
    ax.set_ylabel("MFCC Index (1 - 40)", fontsize=10)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ["REAL", "FAKE"],
    title: str = "Deepfake Detection Confusion Matrix"
) -> plt.Figure:
    """Plots a styled confusion matrix with counts and percentages."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=150)

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.divide(cm, cm_sum, where=cm_sum != 0) * 100

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_perc[i, j]:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={"fontsize": 11, "fontweight": "bold"}
    )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold")
    ax.set_ylabel("Ground Truth Class", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_training_history(history_dict: dict) -> plt.Figure:
    """Plots training and validation loss and accuracy across epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)

    epochs = range(1, len(history_dict.get("accuracy", [])) + 1)

    # Accuracy Plot
    ax1.plot(epochs, history_dict.get("accuracy", []), "b-o", label="Training Accuracy", markersize=4)
    if "val_accuracy" in history_dict:
        ax1.plot(epochs, history_dict.get("val_accuracy", []), "g-s", label="Validation Accuracy", markersize=4)
    ax1.set_title("Model Accuracy Curve", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Loss Plot
    ax2.plot(epochs, history_dict.get("loss", []), "r-o", label="Training Loss", markersize=4)
    if "val_loss" in history_dict:
        ax2.plot(epochs, history_dict.get("val_loss", []), "orange", linestyle="--", marker="s", label="Validation Loss", markersize=4)
    ax2.set_title("Model Loss (Binary Cross-Entropy)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    return fig
