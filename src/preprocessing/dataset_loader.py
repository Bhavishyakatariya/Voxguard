"""
VoxGuard - Dataset Ingestion & Protocol Loader
==============================================
Standardized data loaders for:
1. ASVspoof 2021 Logical Access (LA) Dataset
2. In-The-Wild Deepfake Speech Dataset

Shared Interface Label Contract:
- Label 0 = Real (bonafide)
- Label 1 = Fake (spoof)
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import (
    ASVSPOOF_RAW_DIR,
    INTHEWILD_RAW_DIR,
    LABEL_REAL,
    LABEL_FAKE,
    DATASET_ASVSPOOF,
    DATASET_INTHEWILD,
    SUPPORTED_AUDIO_EXTENSIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """Represents a standardized audio sample ready for processing."""
    audio_path: Path
    sample_id: str
    label: int  # 0 = Real, 1 = Fake
    dataset: str  # 'asvspoof' or 'inthewild'

    def __post_init__(self):
        if self.label not in (LABEL_REAL, LABEL_FAKE):
            raise ValueError(f"Invalid label {self.label}. Must be 0 (Real) or 1 (Fake).")
        if self.dataset not in (DATASET_ASVSPOOF, DATASET_INTHEWILD):
            raise ValueError(f"Invalid dataset '{self.dataset}'. Must be 'asvspoof' or 'inthewild'.")


def find_audio_files(directory: Path) -> Dict[str, Path]:
    """
    Recursively index all supported audio files in a directory by their file stem.
    Uses high-speed single-pass directory traversal.

    Args:
        directory: Root path to search.

    Returns:
        Dictionary mapping audio file stem (e.g. 'LA_E_1000147') to its absolute Path.
    """
    import os
    indexed = {}
    if not directory.is_dir():
        return indexed

    ext_set = {ext.lower() for ext in SUPPORTED_AUDIO_EXTENSIONS}
    for root, _, files in os.walk(directory):
        for fname in files:
            dot_idx = fname.rfind(".")
            if dot_idx != -1:
                ext = fname[dot_idx:].lower()
                if ext in ext_set:
                    stem = fname[:dot_idx]
                    indexed[stem] = Path(root) / fname
    return indexed


# ==============================================================================
# ASVSPOOF 2021 LA DATASET LOADER
# ==============================================================================
def load_asvspoof_dataset(
    raw_dir: Optional[Union[str, Path]] = None,
    protocol_file: Optional[Union[str, Path]] = None,
    max_samples: Optional[int] = None,
) -> List[DatasetSample]:
    """
    Load the ASVspoof 2021 Logical Access (LA) dataset.

    Parses standard trial keys / protocol text files:
    Format typically: <SPEAKER_ID> <AUDIO_FILE_NAME> <SYSTEM_ID> - <KEY>
    where KEY is 'bonafide' or 'spoof'.

    Label Mapping:
        'bonafide' -> 0 (Real)
        'spoof'    -> 1 (Fake)

    Args:
        raw_dir: Path to directory containing ASVspoof files (default: config.ASVSPOOF_RAW_DIR).
        protocol_file: Optional explicit path to protocol/trial key file.
        max_samples: Optional maximum number of samples to load (for debugging/benchmarking).

    Returns:
        List of DatasetSample objects.
    """
    dir_path = Path(raw_dir if raw_dir is not None else ASVSPOOF_RAW_DIR).resolve()
    if not dir_path.is_dir():
        logger.warning(f"ASVspoof directory does not exist: {dir_path}")
        return []

    # 1. Index available audio files in the folder (including flac/ subfolder)
    audio_index = find_audio_files(dir_path)
    logger.info(f"Indexed {len(audio_index)} audio files in ASVspoof directory: {dir_path}")

    # 2. Locate protocol/trial file
    proto_path = None
    if protocol_file:
        candidate = Path(protocol_file).resolve()
        if candidate.is_file():
            proto_path = candidate
    else:
        # Auto-detect protocol text files (prefer files with ground truth labels)
        candidate_patterns = [
            "*trial_metadata*.txt",
            "*metadata*.txt",
            "*keys*.txt",
            "*trl.txt",
            "*eval*.txt",
            "*.txt",
        ]
        all_candidates = []
        for pat in candidate_patterns:
            for p in dir_path.rglob(pat):
                if p.is_file() and p not in all_candidates:
                    all_candidates.append(p)

        if all_candidates:
            def proto_priority(p: Path) -> tuple:
                name = p.name.lower()
                # Prioritize trial_metadata or keys over blind eval trial lists
                priority = 10
                if "trial_metadata" in name:
                    priority = 100
                elif "metadata" in name:
                    priority = 90
                elif "key" in name:
                    priority = 80
                elif "trn" in name or "dev" in name:
                    priority = 70
                elif "trl" in name and "eval" not in name:
                    priority = 60
                return (priority, p.stat().st_size)

            sorted_matches = sorted(all_candidates, key=proto_priority, reverse=True)
            proto_path = sorted_matches[0]

    samples: List[DatasetSample] = []
    real_samples: List[DatasetSample] = []
    fake_samples: List[DatasetSample] = []
    target_per_class = (max_samples // 2) if max_samples else None

    if proto_path and proto_path.is_file():
        logger.info(f"Parsing ASVspoof protocol file: {proto_path}")
        with open(proto_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f, start=1):
                tokens = line.strip().split()
                if not tokens or len(tokens) < 2:
                    continue

                # Standard ASVspoof protocol line formats:
                # Col 1: Speaker ID, Col 2: Audio File Name, ..., Last Col: Key (bonafide/spoof)
                file_id = Path(tokens[1]).stem
                lower_tokens = [t.lower() for t in tokens]

                if "bonafide" in lower_tokens:
                    label = LABEL_REAL
                elif "spoof" in lower_tokens:
                    label = LABEL_FAKE
                elif lower_tokens[-1] == "0":
                    label = LABEL_REAL
                elif lower_tokens[-1] == "1":
                    label = LABEL_FAKE
                else:
                    # Not a recognized label row (e.g. unlabeled challenge submission trial list)
                    continue

                # Locate corresponding audio file
                audio_file = audio_index.get(file_id)
                if audio_file and audio_file.is_file():
                    sample = DatasetSample(
                        audio_path=audio_file,
                        sample_id=file_id,
                        label=label,
                        dataset=DATASET_ASVSPOOF,
                    )
                    if target_per_class:
                        if label == LABEL_REAL and len(real_samples) < target_per_class:
                            real_samples.append(sample)
                        elif label == LABEL_FAKE and len(fake_samples) < target_per_class:
                            fake_samples.append(sample)
                        if len(real_samples) >= target_per_class and len(fake_samples) >= target_per_class:
                            break
                    else:
                        samples.append(sample)

        if target_per_class:
            samples = real_samples + fake_samples
    else:
        # Fallback: Check if organized into bonafide/ and spoof/ subdirectories
        logger.info("No protocol file found; checking for directory-based structure (bonafide/ vs spoof/)...")
        for stem, p in audio_index.items():
            parent_name = p.parent.name.lower()
            if "bonafide" in parent_name or "real" in parent_name:
                samples.append(
                    DatasetSample(audio_path=p, sample_id=stem, label=LABEL_REAL, dataset=DATASET_ASVSPOOF)
                )
            elif "spoof" in parent_name or "fake" in parent_name:
                samples.append(
                    DatasetSample(audio_path=p, sample_id=stem, label=LABEL_FAKE, dataset=DATASET_ASVSPOOF)
                )
            if max_samples and len(samples) >= max_samples:
                break

    logger.info(f"Loaded {len(samples)} valid ASVspoof samples.")
    return samples


# ==============================================================================
# IN-THE-WILD DATASET LOADER
# ==============================================================================
def load_inthewild_dataset(
    raw_dir: Optional[Union[str, Path]] = None,
    meta_file: Optional[Union[str, Path]] = None,
    max_samples: Optional[int] = None,
) -> List[DatasetSample]:
    """
    Load the In-The-Wild deepfake speech dataset.

    Parses the dataset's metadata file (meta.csv) or directory structure.

    Label Mapping:
        'bona-fide', 'bonafide', 'real', '0' -> 0 (Real)
        'spoof', 'fake', 'deepfake', '1'     -> 1 (Fake)

    Args:
        raw_dir: Path to directory containing In-the-Wild files (default: config.INTHEWILD_RAW_DIR).
        meta_file: Optional explicit path to meta.csv file.
        max_samples: Optional maximum number of samples to load.

    Returns:
        List of DatasetSample objects.
    """
    dir_path = Path(raw_dir if raw_dir is not None else INTHEWILD_RAW_DIR).resolve()
    if not dir_path.is_dir():
        logger.warning(f"In-the-Wild directory does not exist: {dir_path}")
        return []

    # 1. Index available audio files
    audio_index = find_audio_files(dir_path)
    logger.info(f"Indexed {len(audio_index)} audio files in In-The-Wild directory: {dir_path}")

    # 2. Locate meta.csv
    csv_path = None
    if meta_file:
        candidate = Path(meta_file).resolve()
        if candidate.is_file():
            csv_path = candidate
    else:
        # Search for meta.csv
        candidates = list(dir_path.rglob("meta*.csv")) + list(dir_path.rglob("*.csv"))
        if candidates:
            csv_path = candidates[0]

    samples: List[DatasetSample] = []
    real_samples: List[DatasetSample] = []
    fake_samples: List[DatasetSample] = []
    target_per_class = (max_samples // 2) if max_samples else None

    if csv_path and csv_path.is_file():
        logger.info(f"Parsing In-The-Wild metadata CSV: {csv_path}")
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            fieldnames = [col.strip().lower() for col in (reader.fieldnames or [])]

            # Detect file column name
            file_col = next(
                (c for c in (reader.fieldnames or []) if c.lower() in ("file", "filename", "audio", "audio_file", "utt_id", "id", "name")),
                None,
            )
            # Detect label column name
            label_col = next(
                (c for c in (reader.fieldnames or []) if c.lower() in ("label", "class", "target", "annotation", "tag")),
                None,
            )

            if file_col and label_col:
                for row in reader:
                    raw_filename = str(row[file_col]).strip()
                    file_stem = Path(raw_filename).stem
                    raw_label = str(row[label_col]).strip().lower()

                    if raw_label in ("bona-fide", "bonafide", "real", "original", "0"):
                        label = LABEL_REAL
                    elif raw_label in ("spoof", "fake", "deepfake", "synthetic", "1"):
                        label = LABEL_FAKE
                    else:
                        continue

                    # Match audio file by stem or filename
                    audio_file = audio_index.get(file_stem)
                    if not audio_file:
                        candidate_direct = dir_path / raw_filename
                        if candidate_direct.is_file():
                            audio_file = candidate_direct

                    if audio_file and audio_file.is_file():
                        sample = DatasetSample(
                            audio_path=audio_file,
                            sample_id=file_stem,
                            label=label,
                            dataset=DATASET_INTHEWILD,
                        )
                        if target_per_class:
                            if label == LABEL_REAL and len(real_samples) < target_per_class:
                                real_samples.append(sample)
                            elif label == LABEL_FAKE and len(fake_samples) < target_per_class:
                                fake_samples.append(sample)
                            if len(real_samples) >= target_per_class and len(fake_samples) >= target_per_class:
                                break
                        else:
                            samples.append(sample)
            else:
                logger.warning(f"Could not identify file/label columns in {csv_path}. Headers: {fieldnames}")

        if target_per_class:
            samples = real_samples + fake_samples

    # Fallback: Directory structure (e.g. real/ vs fake/ subfolders)
    if not samples:
        logger.info("Checking for directory-based structure (real/ vs fake/)...")
        for stem, p in audio_index.items():
            parent_name = p.parent.name.lower()
            if any(term in parent_name for term in ("real", "bonafide", "original")):
                samples.append(
                    DatasetSample(audio_path=p, sample_id=stem, label=LABEL_REAL, dataset=DATASET_INTHEWILD)
                )
            elif any(term in parent_name for term in ("fake", "spoof", "deepfake", "synth")):
                samples.append(
                    DatasetSample(audio_path=p, sample_id=stem, label=LABEL_FAKE, dataset=DATASET_INTHEWILD)
                )
            if max_samples and len(samples) >= max_samples:
                break

    logger.info(f"Loaded {len(samples)} valid In-The-Wild samples.")
    return samples


# ==============================================================================
# UNIFIED DATASET INGESTION
# ==============================================================================
def load_all_datasets(
    asvspoof_dir: Optional[Union[str, Path]] = None,
    inthewild_dir: Optional[Union[str, Path]] = None,
    asvspoof_protocol: Optional[Union[str, Path]] = None,
    inthewild_meta: Optional[Union[str, Path]] = None,
    max_samples_per_dataset: Optional[int] = None,
) -> Dict[str, List[DatasetSample]]:
    """
    Load and index both ASVspoof 2021 LA and In-The-Wild datasets.

    Returns:
        Dictionary with keys 'asvspoof' and 'inthewild' containing their respective samples.
    """
    asv_samples = load_asvspoof_dataset(
        raw_dir=asvspoof_dir,
        protocol_file=asvspoof_protocol,
        max_samples=max_samples_per_dataset,
    )
    wild_samples = load_inthewild_dataset(
        raw_dir=inthewild_dir,
        meta_file=inthewild_meta,
        max_samples=max_samples_per_dataset,
    )

    return {
        DATASET_ASVSPOOF: asv_samples,
        DATASET_INTHEWILD: wild_samples,
    }

