"""
VoxGuard - Audio Preprocessing Pipeline
=======================================
Implements standardized audio loading, mono conversion, resampling,
silence trimming, and peak normalization using Librosa.

Pipeline Steps:
1. Load audio file (.flac, .wav, .mp3, etc.)
2. Convert stereo channels to single-channel mono
3. Resample waveform to exactly 16,000 Hz
4. Trim leading and trailing silence (20 dB threshold)
5. Peak-normalize amplitude into [-1.0, 1.0] interval
"""

import logging
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)


def load_audio(
    file_path: Union[str, Path],
    target_sr: Optional[int] = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file using librosa with error checking.

    Args:
        file_path: Path to the target audio file.
        target_sr: Target sample rate in Hz. If None, preserves original rate.
        mono: If True, automatically combines multi-channel audio to mono.

    Returns:
        Tuple of (audio_waveform: np.ndarray, sample_rate: int).

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be decoded or is empty.
    """
    import librosa

    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        waveform, sr = librosa.load(path, sr=target_sr, mono=mono)
    except Exception as exc:
        raise ValueError(f"Failed to load/decode audio file {path.name}: {exc}") from exc

    if waveform is None or waveform.size == 0:
        raise ValueError(f"Audio file is empty or corrupted: {path.name}")

    return waveform, sr


def convert_to_mono(y: np.ndarray) -> np.ndarray:
    """
    Ensure the audio signal is single-channel (mono).

    Args:
        y: Audio waveform array (1D for mono, 2D for multi-channel).

    Returns:
        1D mono audio waveform as np.ndarray.
    """
    import librosa

    if y.ndim > 1:
        # Multi-channel audio (channels, samples) or (samples, channels)
        if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
            y = librosa.to_mono(y)
        else:
            y = librosa.to_mono(y.T)
    return y.astype(np.float32)


def resample_audio(
    y: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Resample the audio waveform to the target sample rate.

    Args:
        y: Audio waveform array.
        orig_sr: Original sampling rate in Hz.
        target_sr: Target sampling rate in Hz (default: 16000).

    Returns:
        Resampled audio waveform as np.ndarray.
    """
    import librosa

    if orig_sr == target_sr:
        return y.astype(np.float32)

    resampled = librosa.resample(
        y,
        orig_sr=orig_sr,
        target_sr=target_sr,
        res_type="soxr_hq" if hasattr(librosa, "resample") else "kaiser_best",
    )
    return resampled.astype(np.float32)


def trim_silence(y: np.ndarray, top_db: int = 20) -> np.ndarray:
    """
    Trim non-speech silence intervals from the beginning and end of the audio.

    Args:
        y: Audio waveform array.
        top_db: The threshold (in decibels) below reference to consider as silence.

    Returns:
        Trimmed audio waveform as np.ndarray. If trimming results in an empty
        array (e.g. pure silence clip), returns the original signal.
    """
    import librosa

    if y.size == 0:
        return y

    try:
        trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        # Avoid discarding valid clips that fall below top_db threshold
        if trimmed.size > 0:
            return trimmed.astype(np.float32)
        logger.warning("Trimming resulted in 0-length audio; falling back to untrimmed signal.")
        return y.astype(np.float32)
    except Exception as exc:
        logger.warning(f"Silence trimming encountered an issue: {exc}. Retaining untrimmed signal.")
        return y.astype(np.float32)


def normalize_audio(y: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply peak amplitude normalization to scale audio into the [-1.0, 1.0] range.

    Args:
        y: Audio waveform array.
        epsilon: Small positive float to prevent division by zero.

    Returns:
        Peak-normalized audio waveform as np.ndarray.
    """
    if y.size == 0:
        return y

    peak = np.max(np.abs(y))
    if peak > epsilon:
        y = y / (peak + epsilon)
    else:
        # Signal is completely zero / silent
        y = np.zeros_like(y)

    return y.astype(np.float32)


def preprocess_audio_pipeline(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    top_db: int = 20,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """
    Execute the complete audio preprocessing sequence:
    1. Load audio
    2. Convert to mono
    3. Resample to target_sr (16,000 Hz)
    4. Apply silence trimming (top_db)
    5. Perform peak amplitude normalization

    Args:
        file_path: Path to the input raw audio file.
        target_sr: Required sample rate (default 16000).
        top_db: Threshold in dB below reference to treat as silence.
        epsilon: Epsilon for peak normalization.

    Returns:
        Preprocessed mono audio signal as 1D np.ndarray.
    """
    # 1. Load (librosa handles mono & initial target_sr directly)
    waveform, sr = load_audio(file_path, target_sr=target_sr, mono=True)

    # 2. Strict Mono Verification
    waveform = convert_to_mono(waveform)

    # 3. Resample if sr did not match target
    if sr != target_sr:
        waveform = resample_audio(waveform, orig_sr=sr, target_sr=target_sr)

    # 4. Silence Trimming
    waveform = trim_silence(waveform, top_db=top_db)

    # 5. Amplitude Normalization
    waveform = normalize_audio(waveform, epsilon=epsilon)

    return waveform

