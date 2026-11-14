"""
VoxGuard - Preprocessing Configuration Module
==============================================
Centralized configuration parameters for audio ingestion, Mel-spectrogram
feature extraction, dataset loading, and stratified train/val/test splitting.

Interface Contract Standards:
- Audio Resampling: 16,000 Hz Mono
- Mel Filterbanks: 128 bands (0 Hz - 8,000 Hz)
- FFT Window: 1024, Hop Length: 512
- Output Tensor: Exactly (128, 128)
- Feature File Format: .npy
- Label Mapping: 0 = Real (bonafide), 1 = Fake (spoof)
- Stratified Split: Train 70%, Val 15%, Test 15%
"""

import os
from pathlib import Path

# ==============================================================================
# BASE DIRECTORY PATHS
# ==============================================================================
# Resolves dynamically relative to the repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Base data directory
DATA_DIR = Path(os.getenv("VOXGUARD_DATA_DIR", PROJECT_ROOT / "data"))

# Raw dataset directories (configurable via environment variables or CLI flags)
RAW_DIR = DATA_DIR / "raw"
ASVSPOOF_RAW_DIR = Path(os.getenv("VOXGUARD_ASVSPOOF_DIR", RAW_DIR / "asvspoof"))
INTHEWILD_RAW_DIR = Path(os.getenv("VOXGUARD_INTHEWILD_DIR", RAW_DIR / "inthewild"))

# Processed feature output directories
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"

# Output metadata file and visualization directory
METADATA_FILE = PROCESSED_DIR / "metadata.csv"
VISUALIZATION_DIR = PROCESSED_DIR / "visualizations"

# ==============================================================================
# AUDIO PREPROCESSING & MEL-SPECTROGRAM SPECIFICATIONS
# ==============================================================================
# Target audio sampling rate (Hz)
SAMPLE_RATE: int = 16000

# Force mono audio conversion
MONO: bool = True

# Silence trimming threshold (dB below reference)
SILENCE_TOP_DB: int = 20

# Audio normalization epsilon to prevent division by zero
EPSILON: float = 1e-8

# Short-Time Fourier Transform (STFT) & Mel Filterbank parameters
N_FFT: int = 1024
HOP_LENGTH: int = 512
N_MELS: int = 128
FMIN: float = 0.0
FMAX: float = 8000.0

# Fixed output matrix dimensions (n_mels, time_steps)
TARGET_SHAPE: tuple[int, int] = (128, 128)
TARGET_TIME_STEPS: int = 128

# Mel power-to-dB conversion reference
DB_REF = "max"

# ==============================================================================
# DATASET & LABEL SPECIFICATIONS (INTERFACE CONTRACT)
# ==============================================================================
# Binary classification label mapping
LABEL_REAL: int = 0
LABEL_FAKE: int = 1

LABEL_NAME_MAP = {
    LABEL_REAL: "Real",
    LABEL_FAKE: "Fake",
}

# Dataset identifier tags
DATASET_ASVSPOOF: str = "asvspoof"
DATASET_INTHEWILD: str = "inthewild"

# Required splits and metadata column headers
SPLITS: list[str] = ["train", "val", "test"]
METADATA_COLUMNS: list[str] = ["filepath", "label", "dataset", "split"]

# Supported raw audio file extensions
SUPPORTED_AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".flac",
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
)

# ==============================================================================
# STRATIFIED SPLIT DISTRIBUTION
# ==============================================================================
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Global reproducible random seed
RANDOM_SEED: int = 42


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def ensure_directories() -> None:
    """Create all required raw and processed directories if they do not exist."""
    directories = [
        RAW_DIR,
        ASVSPOOF_RAW_DIR,
        INTHEWILD_RAW_DIR,
        PROCESSED_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        VISUALIZATION_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_split_dir(split_name: str) -> Path:
    """
    Return the corresponding processed subdirectory for a given split name.
    
    Args:
        split_name: 'train', 'val', or 'test'
        
    Returns:
        Path to the split directory.
    """
    if split_name == "train":
        return TRAIN_DIR
    elif split_name == "val":
        return VAL_DIR
    elif split_name == "test":
        return TEST_DIR
    else:
        raise ValueError(f"Unknown split name '{split_name}'. Must be one of {SPLITS}.")

