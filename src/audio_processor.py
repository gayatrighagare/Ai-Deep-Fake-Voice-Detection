"""Audio Preprocessing and Feature Extraction Module using Librosa and NumPy.
Handles file validation, audio decoding, normalization, silence trimming,
and acoustic feature extraction (Mel Spectrogram, MFCCs, Spectral features).
"""

import io
import os
from typing import Tuple, Dict, Any, Union
import numpy as np
import librosa
import soundfile as sf


# Allowed audio MIME types and extensions
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB limit
TARGET_SAMPLE_RATE = 16000              # Standard 16 kHz for speech analysis
DEFAULT_AUDIO_DURATION = 3.0           # Standard 3.0 seconds window
MEL_BINS = 128
TIME_STEPS = 128


class AudioValidationError(Exception):
    """Raised when an audio file fails validation (format, size, silence, corruption)."""
    pass


def validate_audio_file(
    file_input: Union[str, io.BytesIO, bytes],
    filename: str = "audio_sample.wav"
) -> int:
    """Validates the input audio file by checking extension, size, and readability.

    Args:
        file_input: File path, BytesIO, or raw bytes.
        filename: Name of the file with extension.

    Returns:
        int: File size in bytes.

    Raises:
        AudioValidationError: If validation fails.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported audio format '{ext}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    # Determine file size
    if isinstance(file_input, str):
        if not os.path.exists(file_input):
            raise AudioValidationError(f"Audio file '{file_input}' not found on disk.")
        file_size = os.path.getsize(file_input)
    elif isinstance(file_input, io.BytesIO):
        file_input.seek(0, io.SEEK_END)
        file_size = file_input.tell()
        file_input.seek(0)
    elif isinstance(file_input, bytes):
        file_size = len(file_input)
    else:
        # File-like object (e.g. Streamlit UploadedFile)
        try:
            file_size = file_input.size
        except AttributeError:
            raise AudioValidationError("Invalid audio input object provided.")

    if file_size == 0:
        raise AudioValidationError("The uploaded audio file is empty (0 bytes). Please upload a valid recording.")

    if file_size > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        actual_mb = file_size / (1024 * 1024)
        raise AudioValidationError(
            f"Audio file exceeds maximum size limit ({actual_mb:.1f} MB > {max_mb:.0f} MB). Please upload a shorter clip."
        )

    return file_size


def load_and_preprocess_audio(
    file_input: Union[str, io.BytesIO, bytes],
    filename: str = "audio_sample.wav",
    target_sr: int = TARGET_SAMPLE_RATE,
    target_duration: float = DEFAULT_AUDIO_DURATION
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Loads an audio file, checks for silence/corruption, trims silences,

    and normalizes length to fixed duration.

    Args:
        file_input: File path or stream.
        filename: Original file name.
        target_sr: Sampling rate to resample to (Hz).
        target_duration: Fixed duration in seconds.

    Returns:
        Tuple: (audio_signal (1D numpy array), sample_rate (int), metadata (dict))

    Raises:
        AudioValidationError: If decoding fails or audio is silent.
    """
    validate_audio_file(file_input, filename)

    try:
        if isinstance(file_input, (io.BytesIO, bytes)):
            if isinstance(file_input, bytes):
                bio = io.BytesIO(file_input)
            else:
                bio = file_input
                bio.seek(0)
            y, sr = librosa.load(bio, sr=target_sr, mono=True)
        else:
            y, sr = librosa.load(file_input, sr=target_sr, mono=True)
    except Exception as e:
        raise AudioValidationError(
            f"Failed to decode audio file. The file may be corrupt or encoded in an unsupported codec. Details: {str(e)}"
        )

    # Empty audio check
    if len(y) == 0:
        raise AudioValidationError("Audio stream contains no audio samples.")

    # Silence check (compute RMS energy and standard deviation)
    std_dev = float(np.std(y))
    rms = float(np.sqrt(np.mean(y ** 2)))
    if std_dev < 1e-4 or rms < 1e-4:
        raise AudioValidationError(
            "The audio recording appears to be completely silent. Please provide a clip with audible speech."
        )

    # Original metadata
    raw_duration = float(len(y) / sr)
    if raw_duration < 0.3:
        raise AudioValidationError(
            f"Audio clip is too short ({raw_duration:.2f}s). Minimum required duration is 0.5 seconds."
        )

    # Trim leading/trailing silence (top_db=25)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) < int(0.3 * sr):
        # Fallback to original if trim clipped everything
        y_trimmed = y

    # Fixed-length standardization (pad or truncate to target_samples)
    target_samples = int(target_sr * target_duration)
    if len(y_trimmed) < target_samples:
        # Pad with zeros or edge reflection
        pad_width = target_samples - len(y_trimmed)
        y_fixed = np.pad(y_trimmed, (0, pad_width), mode='constant')
    else:
        # Take center or beginning segment
        y_fixed = y_trimmed[:target_samples]

    # Amplitude peak normalization to avoid clipping
    max_val = np.max(np.abs(y_fixed))
    if max_val > 0:
        y_fixed = y_fixed / max_val

    metadata = {
        "original_duration_sec": round(raw_duration, 2),
        "processed_duration_sec": target_duration,
        "sample_rate": sr,
        "total_samples": len(y_fixed),
        "rms_energy": round(rms, 4),
        "peak_amplitude": round(float(np.max(np.abs(y))), 4),
    }

    return y_fixed, sr, metadata


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    n_mels: int = MEL_BINS,
    time_steps: int = TIME_STEPS
) -> np.ndarray:
    """Extracts a normalized Log-Mel Spectrogram resized to (n_mels, time_steps, 1).

    This 2D representation is the input tensor for our Deep CNN Classifier.

    Args:
        y: 1D audio time series.
        sr: Sample rate.
        n_mels: Number of Mel frequency bands.
        time_steps: Fixed number of time frame steps.

    Returns:
        np.ndarray: Log-Mel spectrogram tensor of shape (n_mels, time_steps, 1).
    """
    # Calculate hop_length to approximate time_steps
    n_fft = 1024
    hop_length = int(len(y) / (time_steps - 1)) if time_steps > 1 else 512

    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=sr // 2,
        power=2.0
    )

    # Convert to decibels (log-mel)
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Adjust width to exactly time_steps if off by 1-2 frames
    if log_mel.shape[1] > time_steps:
        log_mel = log_mel[:, :time_steps]
    elif log_mel.shape[1] < time_steps:
        pad_amount = time_steps - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_amount)), mode='edge')

    # Normalize to [0, 1] range for neural network numerical stability
    spec_min = log_mel.min()
    spec_max = log_mel.max()
    if spec_max - spec_min > 1e-6:
        norm_mel = (log_mel - spec_min) / (spec_max - spec_min)
    else:
        norm_mel = np.zeros_like(log_mel)

    # Add channel dimension for 2D CNN -> (128, 128, 1)
    return np.expand_dims(norm_mel, axis=-1)


def extract_mfcc(
    y: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    n_mfcc: int = 40
) -> np.ndarray:
    """Extracts Mel-Frequency Cepstral Coefficients (MFCCs).

    Args:
        y: 1D audio time series.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients to compute.

    Returns:
        np.ndarray: MFCC matrix of shape (n_mfcc, time_frames).
    """
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=512)


def extract_spectral_features(
    y: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE
) -> Dict[str, float]:
    """Computes acoustic summary statistics (Centroid, Rolloff, ZCR)

    useful for explaining acoustic anomalies between real and synthetic voices.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]

    return {
        "mean_spectral_centroid_hz": float(np.mean(centroid)),
        "mean_spectral_rolloff_hz": float(np.mean(rolloff)),
        "mean_zero_crossing_rate": float(np.mean(zcr)),
    }
