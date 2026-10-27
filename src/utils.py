"""Utility functions for session tracking, audio metadata formatting,
and CSV exporting.
"""

from datetime import datetime
from typing import List, Dict, Any
import pandas as pd


def create_history_record(
    filename: str,
    prediction_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Formats an inference result into a structured record for the history log."""
    meta = prediction_result.get("metadata", {})
    return {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Filename": filename,
        "Prediction": prediction_result.get("prediction", "UNKNOWN"),
        "Verdict": "FAKE" if prediction_result.get("is_fake") else "REAL",
        "Confidence (%)": prediction_result.get("confidence", 0.0),
        "Raw Prob P(Fake)": prediction_result.get("raw_fake_prob", 0.0),
        "Risk Level": prediction_result.get("risk_level", "N/A"),
        "Duration (s)": meta.get("original_duration_sec", 0.0),
        "Sample Rate (Hz)": meta.get("sample_rate", 16000),
        "RMS Energy": meta.get("rms_energy", 0.0),
        "Spectral Centroid (Hz)": round(meta.get("mean_spectral_centroid_hz", 0.0), 1),
    }


def history_to_dataframe(history_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts the list of history entries into a styled pandas DataFrame."""
    if not history_records:
        return pd.DataFrame(columns=[
            "Timestamp", "Filename", "Verdict", "Confidence (%)",
            "Risk Level", "Duration (s)", "Sample Rate (Hz)", "Spectral Centroid (Hz)"
        ])
    return pd.DataFrame(history_records)


def export_history_to_csv(history_records: List[Dict[str, Any]]) -> str:
    """Exports history records to CSV string for Streamlit download button."""
    df = history_to_dataframe(history_records)
    return df.to_csv(index=False)
