"""
VoxGuard - Mel-Spectrogram Feature Extraction
==============================================
Extracts standardized log-magnitude Mel-spectrograms conforming to the
VoxGuard shared interface contract:
- Exactly (128, 128) float32 matrix
- sr = 16,000 Hz
- n_fft = 1024
- hop_length = 512
- n_mels = 128
- fmin = 0 Hz, fmax = 8000 Hz
- Converted to log-decibel (dB) scale
- Explicit time-dimension padding and center-cropping
- Saved as .npy file inside data/processed/{split}/{filename}.npy
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np

from .config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    TARGET_SHAPE,
    TARGET_TIME_STEPS,
)
from .audio_preprocessing import preprocess_audio_pipeline

logger = logging.getLogger(__name__)


def extract_mel_spectrogram(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """
    Compute log-magnitude Mel-spectrogram from preprocessed audio.

    Args:
        y: 1D audio waveform array (mono, 16 kHz).
        sr: Audio sample rate in Hz (default: 16000).
        n_fft: Number of FFT points (default: 1024).
        hop_length: Hop length between successive STFT frames (default: 512).
        n_mels: Number of Mel frequency filterbanks (default: 128).
        fmin: Minimum frequency in Hz (default: 0.0).
        fmax: Maximum frequency in Hz (default: 8000.0).

    Returns:
        2D Mel-spectrogram in decibels (n_mels, time_steps) as np.ndarray.
    """
    import librosa

    if y.size == 0:
        # Handle zero-length input edge case safely
        return np.zeros((n_mels, TARGET_TIME_STEPS), dtype=np.float32)

    # Compute raw power Mel-spectrogram
    mel_power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    # Convert power spectrogram to decibels (log scale)
    # Using top_db=80 to match standard forensic voice analysis dynamic range
    mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=80.0)

    return mel_db.astype(np.float32)


def pad_or_crop_spectrogram(
    spectrogram: np.ndarray,
    target_time_steps: int = TARGET_TIME_STEPS,
    crop_mode: str = "center",
) -> np.ndarray:
    """
    Explicitly crop or pad the spectrogram along the time axis (axis 1)
    to enforce an exact matrix shape of (128, target_time_steps).

    - If time_steps > target: crops along the time dimension.
    - If time_steps < target: pads symmetrically along the time dimension using
      the minimum dB floor so padded regions represent background silence.
    - If time_steps == target: returns unmodified.

    Args:
        spectrogram: 2D array of shape (n_mels, current_time_steps).
        target_time_steps: Required number of frames (default: 128).
        crop_mode: 'center' (default) or 'front'.

    Returns:
        Padded or cropped 2D array of shape (n_mels, target_time_steps).
    """
    n_mels, current_time_steps = spectrogram.shape

    if current_time_steps == target_time_steps:
        return spectrogram.astype(np.float32)

    if current_time_steps > target_time_steps:
        # Cropping along time dimension
        if crop_mode == "center":
            start_idx = (current_time_steps - target_time_steps) // 2
        else:  # 'front'
            start_idx = 0
        end_idx = start_idx + target_time_steps
        result = spectrogram[:, start_idx:end_idx]
    else:
        # Padding along time dimension
        pad_total = target_time_steps - current_time_steps
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        # Use the floor value (minimum dB) for silence padding
        pad_val = float(spectrogram.min()) if spectrogram.size > 0 else -80.0
        result = np.pad(
            spectrogram,
            ((0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_val,
        )

    # Verify output dimensions match contract
    assert result.shape == (n_mels, target_time_steps), (
        f"Spectrogram shape {result.shape} does not match expected ({n_mels}, {target_time_steps})"
    )

    return result.astype(np.float32)


def save_feature_file(
    feature_matrix: np.ndarray,
    output_path: Union[str, Path],
) -> Path:
    """
    Save the extracted (128, 128) feature matrix as a .npy file.

    Args:
        feature_matrix: 2D numpy array of shape (128, 128).
        output_path: Destination path ending with .npy.

    Returns:
        Resolved Path to the saved .npy file.

    Raises:
        ValueError: If the feature matrix does not match (128, 128).
    """
    out = Path(output_path).resolve()

    if feature_matrix.shape != TARGET_SHAPE:
        raise ValueError(
            f"Interface contract violation: feature matrix has shape {feature_matrix.shape}, "
            f"expected {TARGET_SHAPE}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, feature_matrix.astype(np.float32))
    return out


def process_and_extract(
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
    top_db: int = 20,
    crop_mode: str = "center",
) -> Tuple[bool, str]:
    """
    End-to-end processing and extraction for a single audio file:
    1. Preprocess audio (load, mono, resample, trim silence, peak normalize)
    2. Extract log Mel-spectrogram
    3. Crop or pad along time axis to exactly (128, 128)
    4. Save as .npy feature file

    Args:
        audio_path: Path to source audio file.
        output_path: Destination .npy file path.
        sr: Sample rate.
        n_fft: FFT size.
        hop_length: Hop length.
        n_mels: Number of Mel bands (128).
        fmin: Minimum frequency (0 Hz).
        fmax: Maximum frequency (8000 Hz).
        top_db: Silence trimming threshold.
        crop_mode: Cropping strategy ('center' or 'front').

    Returns:
        Tuple of (success: bool, message: str).
    """
    try:
        # Step 1: Ingest and preprocess audio waveform
        waveform = preprocess_audio_pipeline(audio_path, target_sr=sr, top_db=top_db)

        # Step 2: Compute Mel-spectrogram in decibels
        mel_db = extract_mel_spectrogram(
            waveform,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )

        # Step 3: Pad or crop to exact (128, 128) dimensions
        feature_matrix = pad_or_crop_spectrogram(
            mel_db,
            target_time_steps=TARGET_TIME_STEPS,
            crop_mode=crop_mode,
        )

        # Step 4: Save .npy feature file
        save_feature_file(feature_matrix, output_path)

        return True, "Success"

    except Exception as exc:
        err_msg = f"Failed processing {Path(audio_path).name}: {str(exc)}"
        logger.warning(err_msg)
        return False, err_msg

