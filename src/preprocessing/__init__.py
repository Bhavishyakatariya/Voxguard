"""
VoxGuard Preprocessing Module
=============================
Provides audio ingestion, silence trimming, peak normalization,
Mel-spectrogram extraction, stratified splitting, and dataset loading
for ASVspoof 2021 LA and In-the-Wild corpora.
"""

from .config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    FMIN,
    FMAX,
    TARGET_SHAPE,
    LABEL_REAL,
    LABEL_FAKE,
    DATASET_ASVSPOOF,
    DATASET_INTHEWILD,
    SPLITS,
    METADATA_COLUMNS,
)
from .audio_preprocessing import (
    load_audio,
    convert_to_mono,
    resample_audio,
    trim_silence,
    normalize_audio,
    preprocess_audio_pipeline,
)
from .feature_extraction import (
    extract_mel_spectrogram,
    pad_or_crop_spectrogram,
    save_feature_file,
    process_and_extract,
)
from .dataset_loader import (
    DatasetSample,
    load_asvspoof_dataset,
    load_inthewild_dataset,
    load_all_datasets,
)
from .dataset_split import stratified_split

__all__ = [
    "SAMPLE_RATE",
    "N_FFT",
    "HOP_LENGTH",
    "N_MELS",
    "FMIN",
    "FMAX",
    "TARGET_SHAPE",
    "LABEL_REAL",
    "LABEL_FAKE",
    "DATASET_ASVSPOOF",
    "DATASET_INTHEWILD",
    "SPLITS",
    "METADATA_COLUMNS",
    "load_audio",
    "convert_to_mono",
    "resample_audio",
    "trim_silence",
    "normalize_audio",
    "preprocess_audio_pipeline",
    "extract_mel_spectrogram",
    "pad_or_crop_spectrogram",
    "save_feature_file",
    "process_and_extract",
    "DatasetSample",
    "load_asvspoof_dataset",
    "load_inthewild_dataset",
    "load_all_datasets",
    "stratified_split",
]

