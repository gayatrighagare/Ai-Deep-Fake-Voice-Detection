"""Deep Learning Model Architecture & Training Utilities.
Implements a 2D Convolutional Neural Network (CNN) specialized in
analyzing Log-Mel Spectrograms for deepfake voice detection.
"""

import os
import json
from typing import Tuple, Dict, Any, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers


def build_deepfake_detector(
    input_shape: Tuple[int, int, int] = (128, 128, 1),
    learning_rate: float = 0.0005
) -> tf.keras.Model:
    """Constructs a Deep CNN architecture optimized for acoustic artifact detection

    in Log-Mel Spectrograms.

    Binary classification:
        0 = REAL (Genuine Human Voice)
        1 = FAKE (AI-Synthesized / Deepfake Voice)
    """
    model = models.Sequential(name="DeepfakeVoiceDetectorCNN")
    model.add(layers.Input(shape=input_shape))

    # Block 1: Low-level acoustic texture
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(32, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.2))

    # Block 2: Formant transitions & harmonics
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.Conv2D(64, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 3: High-frequency spectral artifacts
    model.add(layers.Conv2D(128, (3, 3), padding="same", activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))

    # Global spatial aggregation
    model.add(layers.GlobalAveragePooling2D())

    # Fully connected decision head
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid", name="prediction_output"))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )

    return model


def load_detector_model(model_path: str) -> Optional[tf.keras.Model]:
    """Safely loads a trained Keras model from disk."""
    if not os.path.exists(model_path):
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def save_metrics_summary(metrics_dict: Dict[str, Any], filepath: str):
    """Saves evaluation and model metadata to JSON for display in Streamlit."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=4)


def load_metrics_summary(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads evaluation and model metadata from JSON."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
